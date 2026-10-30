#!/usr/bin/env python
"""Train a regulatory topic dictionary from 500 bp ATAC tiles."""

import os
import sys
import argparse
import pickle
import shutil
import numpy as np
import pandas as pd
from scipy.sparse import vstack, hstack, csr_matrix
from datetime import datetime
from pathlib import Path


if __package__:
    from .util_chrom_start import get_chrom_starts
else:
    from util_chrom_start import get_chrom_starts


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Learn global topic dictionary from training cell types',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--train_cells', 
        nargs='+', 
        required=True,
        help='Cell type prefixes of the training tile files'
    )
    
    parser.add_argument(
        '--tile_dir',
        type=str,
        required=True,
        help='Directory containing {cell_type}_tile_500bp_dict.p files'
    )
    
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        help='Directory to save the topic model'
    )
    
    parser.add_argument(
        '--genome',
        type=str,
        default='hg38',
        choices=['hg38', 'hg19', 'mm10'],
        help='Genome assembly'
    )
    
    parser.add_argument(
        '--n_topics',
        type=int,
        default=50,
        help='Number of topics to learn'
    )
    
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=40,
        help='Number of CPU cores for parallel processing'
    )
    
    parser.add_argument(
        '--n_iter',
        type=int,
        default=500,
        help='Number of Gibbs sampling iterations (official tutorial uses 500)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=50.0,
        help='Dirichlet hyperparameter for document-topic distribution'
    )
    
    parser.add_argument(
        '--eta',
        type=float,
        default=0.1,
        help='Dirichlet hyperparameter for topic-word distribution'
    )
    
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )
    

    parser.add_argument(
        '--mallet_path',
        type=str,
        required=True,
        help='Path to Mallet binary (e.g., ~/mallet-2.0.8/bin/mallet)'
    )
    
    parser.add_argument(
        '--tmp_path',
        type=str,
        default='/tmp/mallet_tmp',
        help='Temporary directory for Mallet files'
    )
    

    parser.add_argument(
        '--mallet_memory',
        type=str,
        default='200G',
        help='Memory allocated to Mallet (e.g., 200G, 100G)'
    )
    
    return parser.parse_args(argv)


def get_valid_chromosomes(genome):
    """
    
    Args:
    
    Returns:
    """
    start_dict, end_dict = get_chrom_starts(genome)
    

    exclude_chroms = ['chrY', 'chrM']
    valid_chroms = [chrom for chrom in start_dict.keys() 
                    if chrom not in exclude_chroms]
    
    print(f"\n{'='*60}")
    print(f"Genome: {genome}")
    print(f"Valid chromosomes: {len(valid_chroms)}")
    print(f"  {', '.join(valid_chroms)}")
    print(f"Excluded: {', '.join(exclude_chroms)}")
    print(f"{'='*60}\n")
    
    return valid_chroms


def validate_and_convert_dtype(matrix, context=""):
    """
    
    Args:
    
    Returns:
    
    Raises:
    """
    if np.issubdtype(matrix.dtype, np.integer):
        return matrix
    
    print(f"  WARNING  {context}: dtype is {matrix.dtype}, expected integer")
    

    if matrix.nnz > 0:
        nonzero_data = matrix.data
        

        if not np.allclose(nonzero_data, np.round(nonzero_data)):
            raise ValueError(
                f"ERROR CRITICAL ERROR: {context} contains non-integer values!\n"
                f"   Example values: {nonzero_data[:10]}\n"
                f"   LDA requires integer count data."
            )
        

        if nonzero_data.min() < 0:
            raise ValueError(
                f"ERROR CRITICAL ERROR: {context} contains negative values!\n"
                f"   Min value: {nonzero_data.min()}\n"
                f"   Fragment counts must be non-negative integers."
            )
    
    print(f"     OK Values are integers (stored as float), converting to int32...")
    return matrix.astype(np.int32)


def load_tile_data(tile_dir, train_cells, valid_chroms):
    """
    
    Args:
    
    Returns:
        dict: {cell_type: {chrom: csr_matrix}}
    """
    print(f"\n{'='*60}")
    print("STEP 1: Loading tile data")
    print(f"{'='*60}")
    
    tile_data = {}
    
    for ct in train_cells:
        tile_file = os.path.join(tile_dir, f"{ct}_tile_500bp_dict.p")
        
        if not os.path.exists(tile_file):
            raise FileNotFoundError(f"Missing file: {tile_file}")
        
        print(f"\n-> Loading {ct}...")
        with open(tile_file, 'rb') as f:
            ct_data = pickle.load(f)
        

        ct_data_filtered = {chrom: ct_data[chrom] 
                           for chrom in valid_chroms 
                           if chrom in ct_data}
        

        sample_chrom = valid_chroms[0]
        ct_data_filtered[sample_chrom] = validate_and_convert_dtype(
            ct_data_filtered[sample_chrom],
            context=f"{ct} {sample_chrom}"
        )
        

        for chrom in ct_data_filtered:
            if chrom != sample_chrom:
                ct_data_filtered[chrom] = validate_and_convert_dtype(
                    ct_data_filtered[chrom],
                    context=f"{ct} {chrom}"
                )
        

        sample_matrix = ct_data_filtered[sample_chrom]
        n_cells = sample_matrix.shape[0]
        print(f"  Cells: {n_cells}")
        print(f"  Chromosomes: {len(ct_data_filtered)}")
        print(f"  Data type: {sample_matrix.dtype} OK")
        print(f"  Format: {sample_matrix.format}")
        print(f"  Value range: [{sample_matrix.min()}, {sample_matrix.max()}]")
        

        for chrom in valid_chroms[:3]:
            if chrom in ct_data_filtered:
                n_bins = ct_data_filtered[chrom].shape[1]
                print(f"    {chrom}: {n_bins:,} bins")
        print(f"    ...")
        
        tile_data[ct] = ct_data_filtered
    
    print(f"\nOK Loaded {len(train_cells)} cell types")
    print(f"OK All data types verified and converted to integer")
    return tile_data


def merge_chromosomes_vertically(tile_data, train_cells, valid_chroms):
    """
    
    Args:
        tile_data: {cell_type: {chrom: matrix}}
    
    Returns:
        dict: {chrom: merged_matrix [total_cells x n_bins_chrom]}
    """
    print(f"\n{'='*60}")
    print("STEP 2: Vertical stacking (merging cell types)")
    print(f"{'='*60}")
    
    merged_data_per_chrom = {}
    
    for chrom in valid_chroms:
        matrices_for_chrom = []
        
        for ct in train_cells:
            if chrom in tile_data[ct]:
                matrices_for_chrom.append(tile_data[ct][chrom])
        

        merged_chrom = vstack(matrices_for_chrom, format='csr')
        

        if not np.issubdtype(merged_chrom.dtype, np.integer):
            raise ValueError(
                f"ERROR After vstack, {chrom} dtype became {merged_chrom.dtype}!"
            )
        
        merged_data_per_chrom[chrom] = merged_chrom
        
        if chrom in valid_chroms[:3]:
            print(f"  {chrom}: {merged_chrom.shape[0]:,} cells x {merged_chrom.shape[1]:,} bins")
    
    print(f"  ...")
    total_cells = merged_data_per_chrom[valid_chroms[0]].shape[0]
    print(f"\nOK Merged {len(train_cells)} cell types -> {total_cells:,} total cells")
    
    return merged_data_per_chrom


def merge_chromosomes_horizontally(merged_data_per_chrom, valid_chroms):
    """
    
    Args:
        merged_data_per_chrom: {chrom: matrix}
    
    Returns:
        tuple: (merged_matrix, chrom_boundaries)
    """
    print(f"\n{'='*60}")
    print("STEP 3: Horizontal concatenation (merging chromosomes)")
    print(f"{'='*60}")
    
    all_matrices = []
    chrom_boundaries = {}
    current_offset = 0
    
    print(f"\nChromosome boundaries:")
    print(f"{'Chromosome':<12} {'Start':<12} {'End':<12} {'N_bins':<12}")
    print(f"{'-'*60}")
    
    for chrom in valid_chroms:
        matrix = merged_data_per_chrom[chrom]
        n_bins = matrix.shape[1]
        
        all_matrices.append(matrix)
        
        chrom_boundaries[chrom] = {
            'start': current_offset,
            'end': current_offset + n_bins,
            'n_bins': n_bins
        }
        
        print(f"{chrom:<12} {current_offset:<12,} {current_offset + n_bins:<12,} {n_bins:<12,}")
        
        current_offset += n_bins
    

    print(f"\nConcatenating {len(valid_chroms)} chromosomes...")
    merged_matrix = hstack(all_matrices, format='csr')
    

    if not np.issubdtype(merged_matrix.dtype, np.integer):
        raise ValueError(f"ERROR After hstack, dtype became {merged_matrix.dtype}!")
    
    print(f"\nOK Merged genome-wide matrix: {merged_matrix.shape[0]:,} cells x {merged_matrix.shape[1]:,} bins")
    print(f"OK Dtype: {merged_matrix.dtype}")
    
    return merged_matrix, chrom_boundaries


def generate_metadata(merged_matrix, valid_chroms, chrom_boundaries, train_cells, tile_data):
    """
    
    Args:
    
    Returns:
        tuple: (region_names, cell_names)
    """
    print(f"\n{'='*60}")
    print("STEP 4: Generating metadata")
    print(f"{'='*60}")
    

    print("\nGenerating region names...")
    region_names = []
    for chrom in valid_chroms:
        n_bins = chrom_boundaries[chrom]['n_bins']
        for bin_idx in range(n_bins):
            start = bin_idx * 500
            end = (bin_idx + 1) * 500
            region_names.append(f"{chrom}:{start}-{end}")
    
    print(f"  Total regions: {len(region_names):,}")
    print(f"  First 3: {region_names[:3]}")
    print(f"  Last 3: {region_names[-3:]}")
    

    print("\nGenerating cell names...")
    cell_names = []
    for ct in train_cells:
        n_cells_ct = tile_data[ct][valid_chroms[0]].shape[0]
        for i in range(n_cells_ct):
            cell_names.append(f"{ct}_cell_{i}")
    
    print(f"  Total cells: {len(cell_names):,}")
    print(f"  First 3: {cell_names[:3]}")
    print(f"  Last 3: {cell_names[-3:]}")
    

    assert len(region_names) == merged_matrix.shape[1]
    assert len(cell_names) == merged_matrix.shape[0]
    
    print(f"\nOK Metadata generated successfully")
    
    return region_names, cell_names


def create_cistopic_object_wrapper(merged_matrix, region_names, cell_names):
    """
    """
    print(f"\n{'='*60}")
    print("STEP 5: Creating CistopicObject")
    print(f"{'='*60}")
    
    print("\nTransposing matrix to [regions x cells]...")
    print(f"  Input: {merged_matrix.shape[0]:,} cells x {merged_matrix.shape[1]:,} bins")
    print(f"  Format: {merged_matrix.format}, Dtype: {merged_matrix.dtype}")
    

    if not np.issubdtype(merged_matrix.dtype, np.integer):
        raise ValueError(f"ERROR Input matrix has non-integer dtype: {merged_matrix.dtype}")
    

    merged_matrix_T = merged_matrix.T.tocsr()
    
    print(f"  Output: {merged_matrix_T.shape[0]:,} regions x {merged_matrix_T.shape[1]:,} cells")
    print(f"  Format: {merged_matrix_T.format}, Dtype: {merged_matrix_T.dtype}")
    

    if merged_matrix_T.dtype != merged_matrix.dtype:
        raise ValueError(f"ERROR Dtype changed during transpose!")
    

    if not merged_matrix_T.has_canonical_format:
        merged_matrix_T.sum_duplicates()
    if not merged_matrix_T.has_sorted_indices:
        merged_matrix_T.sort_indices()
    

    print("\nWARNING  Directly constructing CistopicObject to preserve ALL bins")
    print("   (bypassing create_cistopic_object which has filtering bug)")
    
    from pycisTopic.cistopic_class import CistopicObject
    import pandas as pd
    

    print("\nBinarizing matrix...")
    binary_matrix = merged_matrix_T.copy()
    binary_matrix.data = (binary_matrix.data >= 1).astype(np.int32)
    
    print(f"  Binary matrix: {binary_matrix.shape}")
    print(f"  Sparsity: {1 - binary_matrix.nnz / np.prod(binary_matrix.shape):.2%}")
    

    print("\nCreating metadata...")
    cell_data = pd.DataFrame(index=cell_names)
    cell_data['sample_id'] = [name.split('_')[0] for name in cell_names]
    
    region_data = pd.DataFrame(index=region_names)
    

    print("\nCalculating statistics...")
    region_acc_cells = np.array(binary_matrix.sum(axis=1)).flatten()
    region_data['cisTopic_nr_acc'] = region_acc_cells
    
    cell_acc_regions = np.array(binary_matrix.sum(axis=0)).flatten()
    cell_data['cisTopic_nr_frag'] = cell_acc_regions
    
    print(f"  Region accessibility: [{region_acc_cells.min()}, {region_acc_cells.max()}] cells")
    print(f"  Cell accessibility: [{cell_acc_regions.min()}, {cell_acc_regions.max()}] regions")
    

    print("\nConstructing CistopicObject...")
    cistopic_obj = CistopicObject(
        fragment_matrix=merged_matrix_T,
        binary_matrix=binary_matrix,
        cell_names=list(cell_names),
        region_names=list(region_names),
        cell_data=cell_data,
        region_data=region_data,
        path_to_fragments={},
        project='ATAC2HiC_GlobalTopics'
    )
    
    print(f"\nOK CistopicObject created successfully")
    print(f"  N cells: {len(cistopic_obj.cell_names):,}")
    print(f"  N regions: {len(cistopic_obj.region_names):,}")
    

    if len(cistopic_obj.region_names) != len(region_names):
        raise ValueError(
            f"ERROR Regions were filtered!\n"
            f"   Input: {len(region_names):,}\n"
            f"   Output: {len(cistopic_obj.region_names):,}"
        )
    
    if cistopic_obj.region_names[0] != 'chr1:0-500':
        raise ValueError(
            f"ERROR First region is {cistopic_obj.region_names[0]}, not chr1:0-500!"
        )
    
    print(f"  OK All {len(region_names):,} bins preserved")
    print(f"  OK First region: {cistopic_obj.region_names[0]} (correct!)")
    print(f"  OK Bins alignment: VERIFIED")
    
    return cistopic_obj


def find_latest_file(search_dirs, patterns):
    candidates = []
    seen = set()

    for search_dir in search_dirs:
        search_dir = Path(os.path.expanduser(str(search_dir)))
        if not search_dir.exists():
            continue
        for pattern in patterns:
            for path in search_dir.glob(pattern):
                if not path.is_file():
                    continue
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda p: p.stat().st_mtime)


def persist_mallet_artifacts(tmp_path, save_dir):
    save_dir = Path(os.path.expanduser(str(save_dir)))
    tmp_path = Path(os.path.expanduser(str(tmp_path)))
    save_dir.mkdir(parents=True, exist_ok=True)

    artifact_specs = [
        ('inferencer.mallet', ('inferencer.mallet', '*_inferencer.mallet')),
        ('topickeys.txt', ('topickeys.txt', '*_topickeys.txt')),
        ('doctopics.txt', ('doctopics.txt', '*_doctopics.txt')),
        ('state.mallet.gz', ('state.mallet.gz', '*_state.mallet.gz')),
    ]

    print("\nPersisting MALLET artifacts:")
    for dest_name, patterns in artifact_specs:
        src = find_latest_file([save_dir, tmp_path], patterns)
        dest = save_dir / dest_name

        if src is None:
            if dest_name == 'inferencer.mallet':
                raise FileNotFoundError(
                    f"MALLET inferencer was not found in {tmp_path} or {save_dir}."
                )
            print(f"  - {dest_name}: not found, skipping")
            continue

        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
            print(f"  - {dest_name}: copied from {src}")
        else:
            print(f"  - {dest_name}: already in {save_dir}")


def select_lda_model(models, n_topics):
    if isinstance(models, dict):
        return models[n_topics]

    if isinstance(models, (list, tuple)):
        if len(models) == 1:
            return models[0]
        raise ValueError(f"Model with {n_topics} topics not found in returned model list.")

    raise TypeError(f"Unexpected pycisTopic model container: {type(models).__name__}")


def run_lda(cistopic_obj, n_topics, n_cpu, n_iter, alpha, eta, random_state, 
            save_dir, mallet_path, tmp_path):
    """
    
    Args:
        cistopic_obj: CistopicObject
    
    Returns:
        CistopicLDAModel
    """
    from pycisTopic.lda_models import run_cgs_models_mallet
    print(f"\n{'='*60}")
    print("STEP 6: Running Mallet LDA topic modeling")
    print(f"{'='*60}")
    

    mallet_path = os.path.expanduser(mallet_path)
    tmp_path = os.path.expanduser(tmp_path)
    os.makedirs(tmp_path, exist_ok=True)
    

    if not os.path.exists(mallet_path):
        raise FileNotFoundError(
            f"ERROR Mallet not found at: {mallet_path}\n"
            f"   Please check --mallet_path argument"
        )
    
    print(f"\nParameters:")
    print(f"  n_topics: {n_topics}")
    print(f"  n_cpu: {n_cpu}")
    print(f"  n_iter: {n_iter}")
    print(f"  alpha: {alpha} (per-topic: {alpha/n_topics:.3f})")
    print(f"  eta: {eta}")
    print(f"  random_state: {random_state}")
    print(f"  mallet_path: {mallet_path}")
    print(f"  tmp_path: {tmp_path}")
    
    if n_iter < 200:
        print(f"WARNING  WARNING: n_iter={n_iter} may be insufficient!")
    
    print(f"\n Starting LDA training...")
    start_time = datetime.now()
    print(f"   Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        models = run_cgs_models_mallet(
            cistopic_obj,
            n_topics=[n_topics],
            n_cpu=n_cpu,
            n_iter=n_iter,
            random_state=random_state,
            alpha=alpha,
            alpha_by_topic=True,
            eta=eta,
            eta_by_topic=False,
            tmp_path=tmp_path,
            save_path=save_dir,
            mallet_path=mallet_path,
            reuse_corpus=False
        )
    except Exception as e:
        print(f"\nERROR LDA training failed: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check Java: java -version")
        print(f"  2. Check Mallet: ls {mallet_path}")
        print(f"  3. Check tmp space: df -h {tmp_path}")
        raise
    
    end_time = datetime.now()
    duration = end_time - start_time
    print(f"   End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"   Duration: {duration}")
    
    model = select_lda_model(models, n_topics)
    
    if model is None:
        raise ValueError(
            f"ERROR Model with {n_topics} topics not found!\n"
            f"   Available: {list(models.keys())}"
        )
    
    print(f"\nOK LDA completed successfully")
    print(f"  N topics: {model.n_topic}")
    print(f"  N cells: {model.n_cells}")
    print(f"  N regions: {model.n_regions}")

    persist_mallet_artifacts(tmp_path, save_dir)
    
    return model


def main(argv=None):
    args = parse_args(argv)
    

    os.environ['MALLET_MEMORY'] = args.mallet_memory
    
    print(f"\n{'='*80}")
    print("pycisTopic Global Topic Dictionary Learning")
    print(f"{'='*80}")
    print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("Configuration:")
    print(f"  Training cells: {', '.join(args.train_cells)}")
    print(f"  Genome: {args.genome}")
    print(f"  N topics: {args.n_topics}")
    print(f"  N CPU: {args.n_cpu}")
    print(f"  Mallet memory: {args.mallet_memory}")
    
    try:
        # Pipeline
        valid_chroms = get_valid_chromosomes(args.genome)
        tile_data = load_tile_data(args.tile_dir, args.train_cells, valid_chroms)
        merged_data_per_chrom = merge_chromosomes_vertically(tile_data, args.train_cells, valid_chroms)
        merged_matrix, chrom_boundaries = merge_chromosomes_horizontally(merged_data_per_chrom, valid_chroms)
        region_names, cell_names = generate_metadata(merged_matrix, valid_chroms, chrom_boundaries, args.train_cells, tile_data)
        cistopic_obj = create_cistopic_object_wrapper(merged_matrix, region_names, cell_names)
        model = run_lda(cistopic_obj, args.n_topics, args.n_cpu, args.n_iter, 
                       args.alpha, args.eta, args.random_state, args.save_dir,
                       args.mallet_path, args.tmp_path)
        
        print(f"\n{'='*80}")
        print("OK SUCCESS - All steps completed")
        print(f"{'='*80}")
        print(f"End: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nOutput: {os.path.join(args.save_dir, f'Topic{args.n_topics}.pkl')}")
        print(f"Inferencer: {os.path.join(args.save_dir, 'inferencer.mallet')}")
        
    except Exception as e:
        print(f"\n{'='*80}")
        print("ERROR ERROR")
        print(f"{'='*80}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
