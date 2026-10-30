"""Train S2D with a frozen HERTA model and low-cell ATAC sampling."""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

import numpy as np
from pathlib import Path

import os

if __package__:
    from .utils import (
        get_chrom_starts, get_effective_genome_size, get_num_cells, get_libsize,
        load_hic_multi_resolution, load_atac, load_ctcf_motif, get_starts, normalize,
    )
    from .dataloader import create_topic_data_dict
    from .dataloader_s2d import Dataset_pbulk_topic_multilevel_S2D
    from .model import Model_PbulkTopic_MultiRes
    from .s2d_module import S2DModule
    from .training import (
        multiresolution_weighted_mse_loss, train_pbulk_topic_multilevel,
        validate_pbulk_topic_multilevel, training_loop_multilevel,
    )
else:
    from utils import (
        get_chrom_starts, get_effective_genome_size, get_num_cells, get_libsize,
        load_hic_multi_resolution, load_atac, load_ctcf_motif, get_starts, normalize,
    )
    from dataloader import create_topic_data_dict
    from dataloader_s2d import Dataset_pbulk_topic_multilevel_S2D
    from model import Model_PbulkTopic_MultiRes
    from s2d_module import S2DModule
    from training import (
        multiresolution_weighted_mse_loss, train_pbulk_topic_multilevel,
        validate_pbulk_topic_multilevel, training_loop_multilevel,
    )

parser = argparse.ArgumentParser(description="Train S2D Module - Multi-GPU Strategy")

parser.add_argument(
    "--data-path", metavar="DIR", default="./datasets", help="path to dataset"
)
parser.add_argument(
    "--ctcf-path",
    default=None,
    help="optional CTCF motif pickle; defaults to <data-path>/dna/<genome>_ctcf_motif_score.p",
)
parser.add_argument(
    "--save-path", default="./checkpoints/s2d_low_cell", help="path to save S2D checkpoints"
)
parser.add_argument(
    "--herta-checkpoint",
    required=True,
    help="path to the pretrained HERTA pbulk+topic checkpoint",
)

parser.add_argument("-ct", "--ct-list", nargs="+", required=True,
                    help="cell type prefixes present in the prepared input data")
parser.add_argument("--genome", default="hg38", help="genome of training cell type")
parser.add_argument(
    "--min-stripe-signal",
    default=-170,
    type=int,
    help="Stripe signal cutoff",
)
parser.add_argument(
    "-j",
    "--workers",
    default=8,
    type=int,
    help="number of data loading workers (default: 8)",
)
parser.add_argument(
    "--epochs", default=30, type=int, help="number of total epochs to run"
)
parser.add_argument(
    "-b", "--batch-size", default=32, type=int, help="batch size (default: 32)"
)
parser.add_argument(
    "--patience", default=8, type=int, help="training patience in epochs (default: 8)"
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-4,
    type=float,
    metavar="LR",
    help="initial learning rate for S2D",
    dest="lr",
)
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-3,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-3)",
    dest="weight_decay",
)
parser.add_argument(
    "--seed", default=42, type=int, help="seed for initializing training"
)
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument("--deterministic", action="store_true", help="cudnn deterministic")
parser.add_argument("--mod-name", default="", help="model name suffix")

parser.add_argument(
    "--min-cells",
    default=100,
    type=int,
    help="minimum number of sampled cells during training (default: 100)"
)
parser.add_argument(
    "--max-cells",
    default=1000,
    type=int,
    help="maximum number of sampled cells during training (default: 1000)"
)

parser.add_argument(
    "--s2d-gpus",
    nargs="+",
    type=int,
    default=[1, 2],
    help="GPU IDs used by the S2D DataParallel module (default: [1, 2])"
)
parser.add_argument(
    "--herta-gpu",
    default=3,
    type=int,
    help="GPU ID used by the frozen HERTA model (default: 3)"
)


class S2D_HERTA_Wrapper_MultiGPU(nn.Module):
    def __init__(self, s2d_module_parallel, herta_model, s2d_main_device, herta_device):
        super().__init__()
        self.s2d = s2d_module_parallel
        self.herta = herta_model.to(herta_device)

        self.s2d_main_device = s2d_main_device
        self.herta_device = herta_device

        for param in self.herta.parameters():
            param.requires_grad = False

        print(f"\nMulti-GPU allocation:")
        print(f"  - S2D model -> DataParallel on GPUs {self.s2d.device_ids} (trainable)")
        print(f"  - HERTA model -> GPU {herta_device} (frozen)")
        print(f"  - Main device: GPU {s2d_main_device}")

    def forward(self, X, topics):
        X = X.to(self.s2d_main_device)
        topics_s2d = topics.to(self.s2d_main_device)

        pbulk = X[:, 0:1, :]
        ctcf = X[:, 1:2, :]

        enhanced_pbulk = self.s2d(pbulk)

        enhanced_X = torch.cat([enhanced_pbulk, ctcf], dim=1)

        enhanced_X_herta = enhanced_X.to(self.herta_device)
        topics_herta = topics_s2d.to(self.herta_device)

        predictions = self.herta(enhanced_X_herta, topics_herta)

        predictions_return = self._move_predictions_to_device(predictions, self.s2d_main_device)

        return predictions_return

    def _move_predictions_to_device(self, predictions, device):
        if isinstance(predictions, dict):
            return {k: self._move_predictions_to_device(v, device) for k, v in predictions.items()}
        elif isinstance(predictions, torch.Tensor):
            return predictions.to(device)
        else:
            return predictions

    def train(self, mode=True):
        self.s2d.train(mode)
        self.herta.eval()
        return self

    def eval(self):
        self.s2d.eval()
        self.herta.eval()
        return self


def main(argv=None):
    args = parser.parse_args(argv)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device(f"cuda:{args.s2d_gpus[0]}")
        if args.deterministic:
            cudnn.deterministic = True
            print("Using deterministic training mode")
    else:
        raise RuntimeError("This script requires CUDA and enough GPUs for the requested allocation")

    required_gpus = len(set(args.s2d_gpus + [args.herta_gpu]))
    if torch.cuda.device_count() < required_gpus:
        raise RuntimeError(f"Need at least {required_gpus} GPUs, but detected {torch.cuda.device_count()}")

    torch.manual_seed(args.seed)

    data_path = args.data_path
    save_path = args.save_path
    BATCH_SIZE = args.batch_size
    N_EPOCHS = args.epochs
    initial_rate = args.lr
    wd = args.weight_decay
    ct_list = args.ct_list
    genome = args.genome

    Path(save_path).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("S2D low-cell training - multi-GPU strategy")
    print(f"{'='*70}")
    print(f"Cell sampling range: [{args.min_cells}, {args.max_cells}]")
    print(f"Training cell types: {ct_list}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {N_EPOCHS}")
    print(f"Learning rate: {initial_rate}")
    print(f"Weight decay: {wd}")
    print(f"Patience: {args.patience}")
    print(f"S2D GPUs: {args.s2d_gpus}")
    print(f"HERTA GPU: {args.herta_gpu}")
    print(f"Save path: {save_path}")
    print(f"{'='*70}\n")

    train_chrom = [1,2,4,6,7,8,9,10,11,12,13,14,16,17,19]
    val_chrom = [3, 18]
    input_size = 4010000
    train_step = 5e04
    val_step = 5e04

    start_dict, end_dict = get_chrom_starts(genome)
    effective_genome_size = get_effective_genome_size(genome)

    print("Loading training data...")
    print(f"Cell types: {ct_list}")

    hic_dict_5kb = {}
    hic_dict_10kb = {}
    hic_dict_25kb = {}
    pbulk_dict = {}
    n_cells = {}
    libsize_cell = {}

    for ct in ct_list:
        multi_hic_dict = load_hic_multi_resolution(data_path, ct)
        hic_dict_5kb[ct] = multi_hic_dict['5kb'][ct]
        hic_dict_10kb[ct] = multi_hic_dict['10kb'][ct]
        hic_dict_25kb[ct] = multi_hic_dict['25kb'][ct]

        pbulk_dict[ct], _ = load_atac(data_path, ct, pbulk_only=True)
        n_cells[ct] = get_num_cells(pbulk_dict[ct])
        libsize_cell[ct] = get_libsize(pbulk_dict[ct])
        print(f"  {ct}: loaded pbulk data")

    ctcf_dict = load_ctcf_motif(data_path, genome, args.ctcf_path)
    print("  Loaded CTCF motif data")

    print("\nLoading topic data...")
    for ct in ct_list:
        npz_path = os.path.join(data_path, f"{ct}_topics_aggregated.npz")
        if not os.path.exists(npz_path):
            npz_path = os.path.join(data_path, "topic", f"{ct}_topics_aggregated.npz")
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Topic data not found: {npz_path}")

    topk_paths = {ct: "" for ct in ct_list}
    topic_dict = create_topic_data_dict(
        topk_paths,
        n_topics=50,
        data_path=args.data_path
    )
    print(f"  Loaded topic data for {len(ct_list)} cell types\n")

    print(f"{'='*70}")
    print("Creating S2D dataset")
    print(f"{'='*70}")

    train_dataset = Dataset_pbulk_topic_multilevel_S2D(
        input_size,
        effective_genome_size,
        hic_dict_5kb,
        hic_dict_10kb,
        hic_dict_25kb,
        pbulk_dict,
        ctcf_dict,
        topic_dict,
        n_cells,
        libsize_cell,
        *get_starts(
            chrom_list=train_chrom,
            start_dict=start_dict,
            end_dict=end_dict,
            chrom_start_offset=4000000,
            chrom_end_offset=5000000,
            multi_ct=True,
            ct_list=ct_list,
            step=train_step,
        ),
        transform=normalize,
        is_training=True,
        min_cells=args.min_cells,
        max_cells=args.max_cells,
    )

    val_dataset = Dataset_pbulk_topic_multilevel_S2D(
        input_size,
        effective_genome_size,
        hic_dict_5kb,
        hic_dict_10kb,
        hic_dict_25kb,
        pbulk_dict,
        ctcf_dict,
        topic_dict,
        n_cells,
        libsize_cell,
        *get_starts(
            chrom_list=val_chrom,
            start_dict=start_dict,
            end_dict=end_dict,
            chrom_start_offset=4000000,
            chrom_end_offset=5000000,
            multi_ct=True,
            ct_list=ct_list,
            step=val_step,
        ),
        transform=normalize,
        is_training=False,
        sample_cells=True,
        min_cells=args.min_cells,
        max_cells=args.max_cells,
    )

    print(f"Created Dataset_pbulk_topic_multilevel_S2D")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Cell sampling range: [{args.min_cells}, {args.max_cells}]")

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )

    print(f"\n{'='*70}")
    print("Model initialization - multi-GPU strategy")
    print(f"{'='*70}")

    print(f"Loading HERTA checkpoint: {args.herta_checkpoint}")
    herta_checkpoint = torch.load(args.herta_checkpoint, map_location='cpu', weights_only=False)

    herta_model = Model_PbulkTopic_MultiRes()

    state_dict = herta_checkpoint
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    herta_model.load_state_dict(state_dict)
    print("  HERTA model loaded")

    s2d_base = S2DModule(base_channels=48, num_groups=12).to(f"cuda:{args.s2d_gpus[0]}")
    s2d_parallel = nn.DataParallel(s2d_base, device_ids=args.s2d_gpus)
    print(f"  S2D module created (DataParallel on GPUs {args.s2d_gpus})")

    s2d_main_device = torch.device(f"cuda:{args.s2d_gpus[0]}")
    herta_device = torch.device(f"cuda:{args.herta_gpu}")

    combined_model = S2D_HERTA_Wrapper_MultiGPU(
        s2d_parallel,
        herta_model,
        s2d_main_device,
        herta_device
    )

    s2d_params = sum(p.numel() for p in s2d_base.parameters())
    herta_params = sum(p.numel() for p in herta_model.parameters())
    trainable_params = sum(p.numel() for p in combined_model.parameters() if p.requires_grad)

    print(f"\nParameter statistics:")
    print(f"  S2D parameters: {s2d_params:,} (trainable, GPUs {args.s2d_gpus})")
    print(f"  HERTA parameters: {herta_params:,} (frozen, GPU {args.herta_gpu})")
    print(f"  Total trainable parameters: {trainable_params:,}")
    print(f"{'='*70}\n")

    optimizer = torch.optim.SGD(
        combined_model.s2d.parameters(),
        lr=initial_rate,
        momentum=0.9,
        weight_decay=wd
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[20, 30, 40],
        gamma=0.1
    )

    criterion = multiresolution_weighted_mse_loss

    print(f"{'='*70}")
    print("Starting S2D training (multi-GPU strategy)")
    print(f"{'='*70}\n")

    model, optimizer, _ = training_loop_multilevel(
        train_pbulk_topic_multilevel,
        validate_pbulk_topic_multilevel,
        combined_model,
        ct_list,
        args.seed,
        criterion,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        N_EPOCHS,
        args.patience,
        save_path,
        args.device,
        args.min_stripe_signal,
        resolution_suffix=f"s2d_multi_gpu_{args.min_cells}_{args.max_cells}"
    )

    print(f"\n{'='*70}")
    print("Training completed")
    print(f"{'='*70}")
    print(f"Model saved to: {save_path}")
    print(f"Cell sampling range: [{args.min_cells}, {args.max_cells}]")
    print(f"GPU allocation: S2D=GPUs{args.s2d_gpus}, HERTA=GPU{args.herta_gpu}")


if __name__ == "__main__":
    main()
