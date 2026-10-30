"""Predict Hi-C contacts with optional S2D enhancement of pseudobulk ATAC."""
import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from numpy import savez_compressed
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__:
    from .dataloader import create_topic_data_dict
    from .model import Model_PbulkOnly_MultiRes, Model_PbulkTopic_MultiRes
    from .s2d_module import S2DModule
    from .testing import test_Dataset_pbulk_multilevel, test_Dataset_pbulk_topic_multilevel
    from .inference import (
        _AnchorAlignedInferenceDataset, maybe_data_parallel, load_model_state,
        unwrap_checkpoint,
    )
    from .utils import (
        get_chrom_starts, get_effective_genome_size, get_starts, load_atac_eval,
        load_ctcf_motif, normalize, normalize_bulk_dict,
    )
else:
    from dataloader import create_topic_data_dict
    from model import Model_PbulkOnly_MultiRes, Model_PbulkTopic_MultiRes
    from s2d_module import S2DModule
    from testing import test_Dataset_pbulk_multilevel, test_Dataset_pbulk_topic_multilevel
    from inference import (
        _AnchorAlignedInferenceDataset, maybe_data_parallel, load_model_state,
        unwrap_checkpoint,
    )
    from utils import (
        get_chrom_starts, get_effective_genome_size, get_starts, load_atac_eval,
        load_ctcf_motif, normalize, normalize_bulk_dict,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HERTA inference with optional S2D enhancement")
    parser.add_argument("--data-path", default="./data/processed", help="root containing atac/dna/topic subfolders")
    parser.add_argument("--ctcf-path", default=None, help="optional CTCF motif pickle; defaults to <data-path>/dna/<genome>_ctcf_motif_score.p")
    parser.add_argument("--model-path", required=True, help="path to model checkpoint")
    parser.add_argument("--save-path", default="./predictions", help="prediction output directory")
    parser.add_argument("-ct", required=True, help="cell type prefix in prepared inputs")
    parser.add_argument("-chrom", "--test-chrom", nargs="+", required=True, help="chromosome numbers to predict")
    parser.add_argument("--genome", default="hg38")
    parser.add_argument("-j", "--workers", default=8, type=int)
    parser.add_argument("-b", "--batch-size", default=32, type=int)
    parser.add_argument("-offset", default=0, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--model-type",
        choices=["pbulk_topic", "pbulk_only"],
        default="pbulk_topic",
    )
    parser.add_argument("--target-resolution", choices=["5kb", "10kb", "25kb", "all"], default="all")
    parser.add_argument("--use-s2d", action="store_true", help="enable S2D enhancement on pseudobulk ATAC")
    parser.add_argument("--s2d-checkpoint", default=None, help="path to S2D checkpoint; required with --use-s2d")
    parser.add_argument("--s2d-base-channels", type=int, default=48)
    parser.add_argument("--s2d-num-groups", type=int, default=12)
    parser.add_argument("--s2d-device", type=int, default=0)
    args = parser.parse_args(argv)
    if args.use_s2d and not args.s2d_checkpoint:
        parser.error("--s2d-checkpoint is required with --use-s2d")
    return args


def extract_s2d_state_dict(raw_state_dict, expected_keys):
    prefixes = (
        "module.s2d.module.",
        "module.s2d.",
        "s2d.module.",
        "s2d.",
        "module.",
        "",
    )
    expected_keys = set(expected_keys)

    for prefix in prefixes:
        if prefix:
            candidate = {
                key[len(prefix):]: value
                for key, value in raw_state_dict.items()
                if key.startswith(prefix) and key[len(prefix):] in expected_keys
            }
        else:
            candidate = {key: value for key, value in raw_state_dict.items() if key in expected_keys}

        if candidate:
            return candidate

    raise ValueError("No S2D weights found in checkpoint")


def load_s2d_module(checkpoint_path, base_channels, num_groups, device):
    checkpoint = unwrap_checkpoint(torch.load(checkpoint_path, map_location="cpu"))
    s2d = S2DModule(base_channels=base_channels, num_groups=num_groups)
    s2d_state_dict = extract_s2d_state_dict(checkpoint, s2d.state_dict().keys())
    s2d.load_state_dict(s2d_state_dict)
    s2d.to(device)
    s2d.eval()
    print(f"S2D loaded, residual_weight={s2d.residual_weight.item():.4f}")
    return s2d


def resolve_s2d_device(args):
    if args.disable_cuda or not torch.cuda.is_available():
        return torch.device("cpu")

    device_count = torch.cuda.device_count()
    if args.s2d_device < 0 or args.s2d_device >= device_count:
        raise ValueError(f"s2d-device {args.s2d_device} is invalid; available CUDA devices: 0..{device_count - 1}")

    return torch.device(f"cuda:{args.s2d_device}")


def enhance_pbulk(pbulk, s2d, s2d_device, target_device):
    enhanced = s2d(pbulk.to(s2d_device).unsqueeze(1)).squeeze(1)
    return enhanced.to(target_device)


def main(argv=None):
    args = parse_args(argv)
    import pandas as pd
    model_type = args.model_type
    args.data_path = os.path.expanduser(args.data_path)
    if args.ctcf_path is not None:
        args.ctcf_path = os.path.expanduser(args.ctcf_path)
    args.model_path = os.path.expanduser(args.model_path)
    args.save_path = os.path.expanduser(args.save_path)
    if args.s2d_checkpoint is not None:
        args.s2d_checkpoint = os.path.expanduser(args.s2d_checkpoint)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        if args.deterministic:
            cudnn.deterministic = True
    else:
        args.device = torch.device("cpu")

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f"data_path does not exist: {args.data_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"model_path does not exist: {args.model_path}")
    if args.use_s2d and not os.path.exists(args.s2d_checkpoint):
        raise FileNotFoundError(f"s2d_checkpoint does not exist: {args.s2d_checkpoint}")
    os.makedirs(args.save_path, exist_ok=True)

    test_chrom = [int(chrom) for chrom in args.test_chrom]
    input_size = 4010000
    start_dict, end_dict = get_chrom_starts(args.genome)
    effective_genome_size = get_effective_genome_size(args.genome)

    print("============================ Running HERTA inference ============================")
    print(f"Cell type:       {args.ct}")
    print(f"Model type:      {args.model_type}")
    print(f"Checkpoint:      {args.model_path}")
    print(f"Chromosomes:     {test_chrom}")
    print(f"S2D:             {'enabled' if args.use_s2d else 'disabled'}")

    ctcf_dict = load_ctcf_motif(args.data_path, args.genome, args.ctcf_path)
    pbulk_dict, _ = load_atac_eval(args.data_path, args.ct, pbulk_only=True)
    normalized_pbulk_dict = normalize_bulk_dict(pbulk_dict, effective_genome_size)
    del pbulk_dict

    topic_dict = None
    if model_type == "pbulk_topic":
        topic_data_path = os.path.join(args.data_path, "topic")
        topic_npz = os.path.join(topic_data_path, f"{args.ct}_topics_aggregated.npz")
        if not os.path.exists(topic_npz):
            raise FileNotFoundError(f"Topic NPZ file not found: {topic_npz}")
        topic_dict_raw = create_topic_data_dict({args.ct: ""}, n_topics=50, data_path=topic_data_path)
        topic_dict = topic_dict_raw[args.ct]
        model = maybe_data_parallel(Model_PbulkTopic_MultiRes(), args.device)
    else:
        model = maybe_data_parallel(Model_PbulkOnly_MultiRes(), args.device)

    model = load_model_state(model, args.model_path, args.device)
    model.eval()

    s2d_module = None
    s2d_device = None
    if args.use_s2d:
        s2d_device = resolve_s2d_device(args)
        print(f"Loading S2D from {args.s2d_checkpoint} on {s2d_device}")
        s2d_module = load_s2d_module(
            args.s2d_checkpoint,
            args.s2d_base_channels,
            args.s2d_num_groups,
            s2d_device,
        )

    resolution_configs = {
        "25kb": {"step": 2.5e04},
        "10kb": {"step": 1e04},
        "5kb": {"step": 5e03},
    }
    if args.target_resolution != "all":
        resolution_configs = {args.target_resolution: resolution_configs[args.target_resolution]}

    y_hat_dict = {}
    test_chrom_dfs = {}

    for resolution, config in resolution_configs.items():
        print(f"\n--- Processing {resolution} resolution ---")
        dataset_kwargs = {
            "input_size": input_size,
            "pbulk_dict": normalized_pbulk_dict,
            "ctcf_dict": ctcf_dict,
            "transform": normalize,
            **dict(zip(
                ["list_cts", "list_starts", "list_chroms"],
                get_starts(
                    chrom_list=test_chrom,
                    start_dict=start_dict,
                    end_dict=end_dict,
                    chrom_start_offset=args.offset,
                    chrom_end_offset=5000000,
                    multi_ct=False,
                    ct_list=[args.ct],
                    step=config["step"],
                ),
            )),
        }
        if model_type == "pbulk_topic":
            dataset_kwargs["topic_dict"] = topic_dict
            test_dataset = test_Dataset_pbulk_topic_multilevel(**dataset_kwargs)
        else:
            test_dataset = test_Dataset_pbulk_multilevel(**dataset_kwargs)


        resolution_bp = int(config["step"])
        if any(int(start) % resolution_bp for start in dataset_kwargs["list_starts"]):
            raise ValueError("Use starts on the output-resolution grid; default -offset 0 is supported")
        reverse_shift_bp = 4_000_000 + resolution_bp - input_size
        print(f"Reverse window shift: {reverse_shift_bp:+d} bp", flush=True)
        if reverse_shift_bp:
            reverse_kwargs = dict(dataset_kwargs)
            reverse_kwargs["list_starts"] = [
                int(x) + reverse_shift_bp for x in dataset_kwargs["list_starts"]
            ]
            reverse_dataset = type(test_dataset)(**reverse_kwargs)
            test_dataset = _AnchorAlignedInferenceDataset(test_dataset, reverse_dataset)

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        resolution_predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                raw_pbulk = batch["pbulk"].to(args.device)
                ctcf = batch["ctcf"].to(args.device)

                pbulk = raw_pbulk
                if args.use_s2d:
                    pbulk = enhance_pbulk(raw_pbulk, s2d_module, s2d_device, args.device)

                x = torch.cat([pbulk.unsqueeze(1), ctcf.unsqueeze(1)], dim=1)

                raw_pbulk_rev = torch.flip(batch["reverse_pbulk"].to(args.device) if "reverse_pbulk" in batch else raw_pbulk, [1])
                ctcf_rev = torch.flip(batch["reverse_ctcf"].to(args.device) if "reverse_ctcf" in batch else ctcf, [1])
                pbulk_rev = raw_pbulk_rev
                if args.use_s2d:
                    pbulk_rev = enhance_pbulk(raw_pbulk_rev, s2d_module, s2d_device, args.device)
                x_rev = torch.cat([pbulk_rev.unsqueeze(1), ctcf_rev.unsqueeze(1)], dim=1)

                if model_type == "pbulk_topic":
                    topic = batch["topic"].to(args.device)
                    predictions_left = model(x, topic)
                    topic_rev = torch.flip(batch["reverse_topic"].to(args.device) if "reverse_topic" in batch else topic, [1])
                    predictions_right = model(x_rev, topic_rev)
                else:
                    predictions_left = model(x)
                    predictions_right = model(x_rev)

                pred_left = predictions_left["stage1"][resolution]
                pred_right = predictions_right["stage1"][resolution]
                pred_combined = torch.cat((pred_left, torch.flip(pred_right, [1])), 1)
                resolution_predictions.append(pred_combined.cpu().numpy())

        y_hat_dict[resolution] = np.concatenate(resolution_predictions, axis=0)
        test_chrom_dfs[resolution] = get_starts(
            chrom_list=test_chrom,
            start_dict=start_dict,
            end_dict=end_dict,
            chrom_start_offset=args.offset,
            chrom_end_offset=5000000,
            multi_ct=False,
            ct_list=[args.ct],
            step=config["step"],
        )
        print(f"{resolution} predictions shape: {y_hat_dict[resolution].shape}")

    for resolution in resolution_configs:
        test_chrom_df = pd.DataFrame(test_chrom_dfs[resolution][2], test_chrom_dfs[resolution][2])
        for chrom in test_chrom:
            start_end = np.where(test_chrom_df[0] == f"chr{chrom}")[0]
            if len(start_end) == 0:
                print(f"Warning: no samples for {resolution} chr{chrom}")
                continue
            pred_data = y_hat_dict[resolution][start_end[0]:start_end[-1] + 1]
            s2d_suffix = "_s2d" if args.use_s2d else ""
            out_path = os.path.join(args.save_path, f"prediction_{args.ct}{s2d_suffix}_{resolution}_chr{chrom}.npz")
            print(f"Saving {out_path} shape={pred_data.shape}")
            savez_compressed(out_path, pred_data)

    print(f"Inference done. Predictions saved to {args.save_path}.")


if __name__ == "__main__":
    main()
