import argparse
import logging
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader

if __package__:
    from .dataloader import Dataset_pbulk_multilevel
else:
    from dataloader import Dataset_pbulk_multilevel
if __package__:
    from .model import Model_PbulkOnly_MultiRes
else:
    from model import Model_PbulkOnly_MultiRes
if __package__:
    from .training import (
        multiresolution_weighted_mse_loss,
        train_pbulk_multilevel,
        validate_pbulk_multilevel,
        training_loop_multilevel,
    )
else:
    from training import (
        multiresolution_weighted_mse_loss,
        train_pbulk_multilevel,
        validate_pbulk_multilevel,
        training_loop_multilevel,
    )
if __package__:
    from .utils import (
        get_chrom_starts,
        get_effective_genome_size,
        get_libsize,
        get_num_cells,
        get_starts,
        load_atac,
        load_ctcf_motif,
        load_hic_multi_resolution,
        normalize,
    )
else:
    from utils import (
        get_chrom_starts,
        get_effective_genome_size,
        get_libsize,
        get_num_cells,
        get_starts,
        load_atac,
        load_ctcf_motif,
        load_hic_multi_resolution,
        normalize,
    )

TRAIN_CHROM = [1, 2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19]
VAL_CHROM = [3, 18]
TEST_CHROM = [5, 15, 20, 21]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="HERTA stage-1 3-track Pbulk+CTCF pretraining")
    parser.add_argument("--data-path", default="./data/processed", help="root containing hic/atac/dna subfolders")
    parser.add_argument("--ctcf-path", default=None, help="optional CTCF motif pickle; defaults to <data-path>/dna/<genome>_ctcf_motif_score.p")
    parser.add_argument("--save-path", default="./checkpoints/pretrain", help="checkpoint output directory")
    parser.add_argument("-ct", "--ct-list", nargs="+", required=True, help="cell type prefixes in the prepared input data")
    parser.add_argument("--genome", default="hg38")
    parser.add_argument("--min-stripe-signal", default=-170, type=int)
    parser.add_argument("-j", "--workers", default=8, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("-b", "--batch-size", default=72, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--lr", "--learning-rate", default=5e-5, type=float, dest="lr")
    parser.add_argument("--wd", "--weight-decay", default=5e-2, type=float, dest="weight_decay")
    parser.add_argument("--seed", default=1229, type=int)
    parser.add_argument("--disable-cuda", action="store_true")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--mod-name", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        if args.deterministic:
            cudnn.deterministic = True
            logging.warning("CUDNN deterministic mode enabled; training may be slower.")
    else:
        args.device = torch.device("cpu")

    torch.manual_seed(args.seed)

    data_path = args.data_path
    save_path = args.save_path
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f"data_path does not exist: {data_path}")
    os.makedirs(save_path, exist_ok=True)
    if args.mod_name:
        save_path = os.path.join(save_path, args.mod_name)
        os.makedirs(save_path, exist_ok=True)

    print(f"Saving to {save_path}")
    print(f"Training cell types: {args.ct_list}")
    print(args)

    input_size = 4010000
    train_step = 5e04
    val_step = 5e04
    start_dict, end_dict = get_chrom_starts(args.genome)
    effective_genome_size = get_effective_genome_size(args.genome)

    hic_dict_5kb = {}
    hic_dict_10kb = {}
    hic_dict_25kb = {}
    pbulk_dict = {}
    n_cells = {}
    libsize_cell = {}

    for ct in args.ct_list:
        multi_hic_dict = load_hic_multi_resolution(data_path, ct)
        hic_dict_5kb[ct] = multi_hic_dict["5kb"][ct]
        hic_dict_10kb[ct] = multi_hic_dict["10kb"][ct]
        hic_dict_25kb[ct] = multi_hic_dict["25kb"][ct]

        pbulk_dict[ct], _ = load_atac(data_path, ct, pbulk_only=True)
        n_cells[ct] = get_num_cells(pbulk_dict[ct])
        libsize_cell[ct] = get_libsize(pbulk_dict[ct])
        print(f"  ✓ {ct}: loaded Hi-C + pbulk")

    ctcf_dict = load_ctcf_motif(data_path, args.genome, args.ctcf_path)

    train_dataset = Dataset_pbulk_multilevel(
        input_size,
        effective_genome_size,
        hic_dict_5kb,
        hic_dict_10kb,
        hic_dict_25kb,
        pbulk_dict,
        ctcf_dict,
        n_cells,
        libsize_cell,
        *get_starts(
            chrom_list=TRAIN_CHROM,
            start_dict=start_dict,
            end_dict=end_dict,
            chrom_start_offset=4000000,
            chrom_end_offset=5000000,
            multi_ct=True,
            ct_list=args.ct_list,
            step=train_step,
        ),
        transform=normalize,
        is_training=True,
    )

    val_dataset = Dataset_pbulk_multilevel(
        input_size,
        effective_genome_size,
        hic_dict_5kb,
        hic_dict_10kb,
        hic_dict_25kb,
        pbulk_dict,
        ctcf_dict,
        n_cells,
        libsize_cell,
        *get_starts(
            chrom_list=VAL_CHROM,
            start_dict=start_dict,
            end_dict=end_dict,
            chrom_start_offset=4000000,
            chrom_end_offset=5000000,
            multi_ct=True,
            ct_list=args.ct_list,
            step=val_step,
        ),
        transform=normalize,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=True,
    )

    model = Model_PbulkOnly_MultiRes()
    if args.device.type == "cuda" and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40], gamma=0.1)

    print("\n" + "=" * 70)
    print("HERTA STAGE-1 PRETRAIN CONFIGURATION")
    print("=" * 70)
    print(f"Model:        Model_PbulkOnly_MultiRes (3 tracks)")
    print(f"Cell types:   {', '.join(args.ct_list)}")
    print(f"Chrom split:  train={TRAIN_CHROM}, val={VAL_CHROM}, held-out={TEST_CHROM}")
    print(f"Batch size:   {args.batch_size}")
    print(f"Epochs:       {args.epochs}")
    print(f"Device:       {args.device}")
    print("=" * 70 + "\n")

    training_loop_multilevel(
        train_pbulk_multilevel,
        validate_pbulk_multilevel,
        model,
        args.ct_list,
        args.seed,
        multiresolution_weighted_mse_loss,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        args.epochs,
        args.patience,
        save_path,
        args.device,
        args.min_stripe_signal,
        resolution_suffix="multiresolution",
    )


if __name__ == "__main__":
    main()
