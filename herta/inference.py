
import argparse
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from numpy import savez_compressed
from torch.utils.data import DataLoader
from tqdm import tqdm

if __package__:
    from .dataloader import create_topic_data_dict
else:
    from dataloader import create_topic_data_dict
if __package__:
    from .model import Model_PbulkOnly_MultiRes, Model_PbulkTopic_MultiRes
else:
    from model import Model_PbulkOnly_MultiRes, Model_PbulkTopic_MultiRes
if __package__:
    from .testing import test_Dataset_pbulk_multilevel, test_Dataset_pbulk_topic_multilevel
else:
    from testing import test_Dataset_pbulk_multilevel, test_Dataset_pbulk_topic_multilevel
if __package__:
    from .utils import (
        get_chrom_starts,
        get_effective_genome_size,
        get_starts,
        load_atac_eval,
        load_ctcf_motif,
        normalize,
        normalize_bulk_dict,
    )
else:
    from utils import (
        get_chrom_starts,
        get_effective_genome_size,
        get_starts,
        load_atac_eval,
        load_ctcf_motif,
        normalize,
        normalize_bulk_dict,
    )


class _AnchorAlignedInferenceDataset(torch.utils.data.Dataset):
    """Pair forward input with an independently located reverse input."""
    def __init__(self, forward_dataset, reverse_dataset):
        self.forward_dataset = forward_dataset
        self.reverse_dataset = reverse_dataset

    def __len__(self):
        return len(self.forward_dataset)

    def __getitem__(self, index):
        sample = dict(self.forward_dataset[index])
        reverse = self.reverse_dataset[index]
        for name in ("pbulk", "ctcf", "topic"):
            if name in reverse:
                sample["reverse_" + name] = reverse[name]
        return sample


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HERTA inference")
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
    parser.add_argument("--model-type", choices=["pbulk_topic", "pbulk_only"], default="pbulk_topic")
    parser.add_argument("--target-resolution", choices=["25kb", "10kb", "5kb", "all"], default="all")
    return parser.parse_args(argv)


def maybe_data_parallel(model, device=None):
    if device is not None and device.type != "cuda":
        return model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        return nn.DataParallel(model)
    return model


def unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict"):
            if isinstance(checkpoint.get(key), dict):
                return checkpoint[key]
    return checkpoint


def load_model_state(model, model_path, device):
    state_dict = unwrap_checkpoint(torch.load(model_path, map_location="cpu"))
    keys = list(state_dict.keys())
    has_module_prefix = bool(keys) and all(key.startswith("module.") for key in keys)

    if isinstance(model, nn.DataParallel):
        if not has_module_prefix:
            state_dict = {f"module.{key}": value for key, value in state_dict.items()}
    else:
        if has_module_prefix:
            state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    return model.to(device)


def main(argv=None):
    args = parse_args(argv)
    import pandas as pd

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
    os.makedirs(args.save_path, exist_ok=True)

    test_chrom = [int(x) for x in args.test_chrom]
    input_size = 4010000
    start_dict, end_dict = get_chrom_starts(args.genome)
    effective_genome_size = get_effective_genome_size(args.genome)

    print("============================ Running HERTA inference ============================")
    print(f"Cell type:       {args.ct}")
    print(f"Model type:      {args.model_type}")
    print(f"Checkpoint:      {args.model_path}")
    print(f"Chromosomes:     {test_chrom}")

    ctcf_dict = load_ctcf_motif(args.data_path, args.genome, args.ctcf_path)
    pbulk_dict, _ = load_atac_eval(args.data_path, args.ct, pbulk_only=True)
    normalized_pbulk_dict = normalize_bulk_dict(pbulk_dict, effective_genome_size)
    del pbulk_dict

    topic_dict = None
    if args.model_type == "pbulk_topic":
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
        if args.model_type == "pbulk_topic":
            dataset_kwargs["topic_dict"] = topic_dict
            test_dataset = test_Dataset_pbulk_topic_multilevel(**dataset_kwargs)
        else:
            test_dataset = test_Dataset_pbulk_multilevel(**dataset_kwargs)

        # Keep input length; relocate only the reverse window.
        resolution_bp = int(config["step"])
        if any(int(s) % resolution_bp for s in dataset_kwargs["list_starts"]):
            raise ValueError("Use starts on the output-resolution grid; default -offset 0 is supported")
        reverse_shift_bp = 4_000_000 + resolution_bp - input_size
        print(f"Reverse input shift: {reverse_shift_bp:+d} bp")
        if reverse_shift_bp != 0:
            reverse_kwargs = dict(dataset_kwargs)
            reverse_kwargs["list_starts"] = [
                int(s) + reverse_shift_bp
                for s in dataset_kwargs["list_starts"]
            ]
            reverse_dataset = type(test_dataset)(**reverse_kwargs)
            test_dataset = _AnchorAlignedInferenceDataset(
                test_dataset, reverse_dataset
            )

        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        resolution_predictions = []

        with torch.no_grad():
            for batch in tqdm(test_loader):
                pbulk = batch["pbulk"].to(args.device)
                ctcf = batch["ctcf"].to(args.device)
                x = torch.cat([pbulk.unsqueeze(1), ctcf.unsqueeze(1)], dim=1)

                if args.model_type == "pbulk_topic":
                    topic = batch["topic"].to(args.device)
                    predictions_left = model(x, topic)
                    topic_rev = torch.flip(batch["reverse_topic"].to(args.device) if "reverse_topic" in batch else topic, [1])
                    pbulk_rev = torch.flip(batch["reverse_pbulk"].to(args.device) if "reverse_pbulk" in batch else pbulk, [1])
                    ctcf_rev = torch.flip(batch["reverse_ctcf"].to(args.device) if "reverse_ctcf" in batch else ctcf, [1])
                    x_rev = torch.cat([pbulk_rev.unsqueeze(1), ctcf_rev.unsqueeze(1)], dim=1)
                    predictions_right = model(x_rev, topic_rev)
                else:
                    predictions_left = model(x)
                    pbulk_rev = torch.flip(batch["reverse_pbulk"].to(args.device) if "reverse_pbulk" in batch else pbulk, [1])
                    ctcf_rev = torch.flip(batch["reverse_ctcf"].to(args.device) if "reverse_ctcf" in batch else ctcf, [1])
                    x_rev = torch.cat([pbulk_rev.unsqueeze(1), ctcf_rev.unsqueeze(1)], dim=1)
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
            out_path = os.path.join(args.save_path, f"prediction_{args.ct}_{resolution}_chr{chrom}.npz")
            print(f"Saving {out_path} shape={pred_data.shape}")
            savez_compressed(out_path, pred_data)

    print(f"Inference done. Predictions saved to {args.save_path}.")


if __name__ == "__main__":
    main()
