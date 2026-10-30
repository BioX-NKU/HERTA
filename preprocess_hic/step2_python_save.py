#!/usr/bin/env python
"""Convert HiCDCPlus intermediate text files into sparse HERTA Hi-C dictionaries."""

import argparse
import os
import pickle
import shutil
import filecmp

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

import math


def zscore(count, mu, sdev):
    count, mu, sdev = float(count), float(mu), float(sdev)
    if not all(math.isfinite(x) for x in (count, mu, sdev)) or sdev <= 0:
        raise ValueError("HiCDC+ inputs require finite counts/mu and positive finite sdev")
    result = (count - mu) / sdev
    if not math.isfinite(result):
        raise ValueError("Nonfinite HiCDC+ Z-score")
    return result

def get_resolution_label(resolution):
    if resolution >= 1000000:
        return f"{resolution // 1000000}Mb"
    if resolution >= 1000:
        return f"{resolution // 1000}kb"
    return f"{resolution}bp"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Save HiCDCPlus-normalized Hi-C z-score matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--assembly", type=str, default="hg38", help="Genome assembly.")
    parser.add_argument("--resolution", type=int, default=10000, help="Hi-C resolution in bp.")
    parser.add_argument("--workdir", type=str, default=".", help="HiCDCPlus working directory.")
    parser.add_argument("--cell_type_prefix", default="", help="Optional sample identifier for a canonical training input copy")
    parser.add_argument("--save_path", default="", help="Optional processed_input root; requires cell_type_prefix")
    args = parser.parse_args(argv)
    if args.resolution <= 0:
        parser.error("resolution must be positive")
    if bool(args.cell_type_prefix) != bool(args.save_path):
        parser.error("cell_type_prefix and save_path must be supplied together")
    if args.cell_type_prefix and (os.path.basename(args.cell_type_prefix) != args.cell_type_prefix or args.cell_type_prefix in (".", "..")):
        parser.error("cell_type_prefix must be a filename-safe identifier")
    return args


def get_chromosomes(assembly):
    if assembly in ["hg19", "hg38"]:
        return [f"chr{i}" for i in range(1, 23)] + ["chrX"]
    if assembly in ["mm9", "mm10"]:
        return [f"chr{i}" for i in range(1, 20)] + ["chrX"]
    raise ValueError(f"Unsupported assembly: {assembly}")


def main(argv=None):
    args = parse_args(argv)
    print(vars(args))

    resolution = args.resolution
    res_label = get_resolution_label(resolution)
    workdir = os.path.abspath(args.workdir)
    print(f"Processing Hi-C data at {resolution} bp resolution ({res_label})")
    print(f"Workdir: {workdir}")

    input_dir = os.path.join(workdir, "intermediate")
    simplified_dir = os.path.join(workdir, f"zvalue_intermediate_{res_label}")
    os.makedirs(simplified_dir, exist_ok=True)

    hic_dict = {}
    failed_chromosomes = []

    for chrom in get_chromosomes(args.assembly):
        print(f"Processing {chrom}")
        input_file = os.path.join(input_dir, f"normalized_{chrom}.txt")

        if not os.path.exists(input_file):
            print(f"Warning: {input_file} does not exist, skipping {chrom}")
            failed_chromosomes.append(chrom)
            continue

        try:
            df = pd.read_csv(input_file, sep="\t")
            required_cols = ["counts", "mu", "sdev", "start1", "start2"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: missing columns {missing_cols} in {chrom}, skipping")
                failed_chromosomes.append(chrom)
                continue

            for coordinate in ("start1", "start2"):
                if not np.isfinite(df[coordinate]).all() or (df[coordinate] < 0).any() or (df[coordinate] % resolution != 0).any():
                    raise ValueError(f"Invalid or unaligned {coordinate} coordinates in {chrom}")
            df.loc[:, "zvalue"] = [
                zscore(count, mu, sdev)
                for count, mu, sdev in zip(
                    df.loc[:, "counts"].to_list(),
                    df.loc[:, "mu"].to_list(),
                    df.loc[:, "sdev"].to_list(),
                )
            ]
            if not np.isfinite(df.loc[:, "zvalue"]).all():
                raise ValueError(f"non-finite z-values detected in {chrom}")

            simplified = df.loc[:, ["start1", "start2", "zvalue"]]
            output_file = os.path.join(simplified_dir, f"normalized_{chrom}_simplified_{res_label}.txt")
            simplified.to_csv(output_file, sep="\t", header=None, index=None)
            del simplified

            hic = pd.read_table(output_file, header=None, sep="\t")
            if len(hic) == 0:
                print(f"Warning: no data for {chrom}")
                failed_chromosomes.append(chrom)
                continue

            end = hic[[0, 1]].values.max() + resolution
            hic = hic.pivot(index=0, columns=1, values=2).fillna(0)
            hic = hic.reindex(
                columns=np.arange(0, end, resolution).astype(int),
                index=np.arange(0, end, resolution).astype(int),
                fill_value=0,
            )
            hic_dict[chrom] = csr_matrix(hic)
            print(f"{chrom} processed: {hic.shape} matrix, {(hic != 0).sum().sum()} non-zero entries")

        except Exception as exc:
            print(f"Error processing {chrom}: {exc}")
            failed_chromosomes.append(chrom)

    if failed_chromosomes:
        raise RuntimeError(
            "Failed to convert chromosomes: " + ", ".join(failed_chromosomes)
        )

    if not hic_dict:
        raise RuntimeError("No chromosomes were converted; refusing to write an empty Hi-C dictionary.")

    output_dir = os.path.join(workdir, f"zvalue_{res_label}")
    os.makedirs(output_dir, exist_ok=True)

    output_pickle = os.path.join(output_dir, f"normalized_hic_zscore_dict_{res_label}.p")
    with open(output_pickle, "wb") as handle:
        pickle.dump(hic_dict, handle, protocol=4)

    if args.cell_type_prefix:
        canonical_dir = os.path.join(os.path.abspath(args.save_path), "hic")
        os.makedirs(canonical_dir, exist_ok=True)
        canonical = os.path.join(canonical_dir, f"{args.cell_type_prefix}_hic_zscore_dict_{res_label}.p")
        if os.path.abspath(canonical) != os.path.abspath(output_pickle):
            if os.path.exists(canonical) and not filecmp.cmp(output_pickle, canonical, shallow=False):
                raise FileExistsError(f"Canonical training input differs: {canonical}")
            if not os.path.exists(canonical):
                shutil.copy2(output_pickle, canonical)
        print(f"Canonical HERTA training input: {canonical}")

    print("\nProcessing completed")
    print(f"Resolution: {resolution} bp ({res_label})")
    print(f"Chromosomes processed: {len(hic_dict)}")
    print(f"Output saved to: {output_pickle}")

    print("\nChromosome data summary:")
    for chrom, matrix in hic_dict.items():
        non_zero_count = matrix.nnz
        total_bins = matrix.shape[0] * matrix.shape[1]
        sparsity = (1 - non_zero_count / total_bins) * 100
        print(f"  {chrom}: {matrix.shape} matrix, {non_zero_count:,} interactions, {sparsity:.1f}% sparse")


if __name__ == "__main__":
    main()
