#!/usr/bin/env python
"""Prepare ATAC tile matrices and metacell masks from fragments and LSI coordinates."""

import random
import sys
import numpy as np
import pandas as pd
import os, argparse
import itertools
import scipy
from scipy.sparse import csr_matrix
from scipy import sparse
import pickle
from scipy.sparse import coo_matrix, csr_matrix, find
from sklearn.neighbors import NearestNeighbors

def core_barcode(value):
    value = str(value)
    if value.count("#") > 1 or not value or value.endswith("#"):
        raise ValueError("Invalid cell barcode: " + repr(value))
    return value.split("#", 1)[-1]

def alignment_positions(lsi_names, tile_names):
    lsi = [core_barcode(x) for x in lsi_names]
    tile = [core_barcode(x) for x in tile_names]
    if len(set(lsi)) != len(lsi) or len(set(tile)) != len(tile):
        raise ValueError("Duplicate barcodes after removing sample prefixes")
    if set(lsi) != set(tile):
        raise ValueError("LSI and tile barcode sets differ")
    positions = {name: i for i, name in enumerate(lsi)}
    return [positions[name] for name in tile]

parser = argparse.ArgumentParser(description="Set-up data preparations",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--cell_type_prefix", help="Cell type and its prefix for the file names.",
                    type=str, default="")
parser.add_argument("--fragment_file", help="Path and name of the input fragment file.",
                    type=str, default="")
parser.add_argument("--barcode_file", help="Path and name of the valid barcode file.",
                    type=str, default="")
parser.add_argument("--genome_assembly", help="Genome assembly of the scATAC files.",
                    type=str, default="")
parser.add_argument("--chrom_sizes", help="Path to the chromosome sizes file for the genome assembly.",
                    type=str, default="")
parser.add_argument("--lsi_file", help="Path and name of the LSI file from ArchR running.",
                    type=str, default="")
parser.add_argument("--save_path", help="Path to save all the intermediate and final files.",
                    type=str, default="")

parser.add_argument("--n_neighbors", type=int, default=100, help="Cells per metacell neighborhood")
parser.add_argument("--max_overlap", type=int, default=33, help="Maximum neighborhood overlap")
parser.add_argument("--metacell_seed", type=int, default=10, help="Seed for greedy metacell selection")

chrom_size = None
VALID_CHROM = []
DEFAULT_CHROMS = ['chr1', 'chr2', 'chr3', 'chr4', 'chr5','chr6','chr7',
                  'chr8', 'chr9','chr10','chr11','chr12','chr13','chr14',
                  'chr15','chr16','chr17','chr18','chr19','chr20',
                  'chr21','chr22','chrX']


def load_chrom_sizes(chrom_sizes_path, genome=None):
    if chrom_sizes_path == "":
        raise ValueError("--chrom_sizes is required.")
    if not os.path.exists(chrom_sizes_path):
        raise FileNotFoundError(f"Chromosome sizes file does not exist: {chrom_sizes_path}")
    loaded = pd.read_csv(chrom_sizes_path, sep = '\t', header = None, index_col = 0)
    if not loaded.index.is_unique or (loaded.iloc[:, 0] <= 0).any():
        raise ValueError("Chromosome sizes must have unique names and positive lengths")
    expected = DEFAULT_CHROMS
    if genome in ("mm9", "mm10"):
        if any(chrom in loaded.index for chrom in ("chr20", "chr21", "chr22")):
            raise ValueError("Human autosomes in mouse chromosome sizes; check genome_assembly")
        expected = [f"chr{i}" for i in range(1, 20)] + ["chrX"]
    elif genome and genome not in ("hg19", "hg38"):
        raise ValueError("Supported genome_assembly values: hg19, hg38, mm9, mm10")
    selected = [chrom for chrom in expected if chrom in loaded.index.to_list()]
    if len(selected) == 0:
        raise ValueError(f"No supported chromosomes were found in: {chrom_sizes_path}")
    return loaded, selected


#########################
#     Load functions    #
#########################

def atac_processing(frag_path, nrows_skip, valid_barcode_path,
                    save_path, cell_type_prefix, valid_chrom = None):
    if valid_chrom is None:
        valid_chrom = VALID_CHROM
    if chrom_size is None:
        raise RuntimeError("Chromosome sizes were not loaded before atac_processing().")
    #1. read fragment file and filter for valid cell barcode
    frag = pd.read_csv(frag_path, sep = '\t', header=None, skiprows = nrows_skip,
                       compression='gzip' if str(frag_path).endswith(('.gz', '.bgz')) else 'infer',
                       dtype={0: str, 3: str})
    dist = frag[2]-frag[1]
    valid_barcode = pd.read_csv(valid_barcode_path, index_col = 0)
    # valid_barcode = valid_barcode.index
    valid_barcode = valid_barcode.iloc[:,0]
    valid_barcode = [core_barcode(i) for i in valid_barcode.to_list()]
    if len(set(valid_barcode)) != len(valid_barcode):
        raise ValueError("QC barcode CSV contains duplicate cell identities")
    frag = frag[frag[3].isin(valid_barcode)]
    if frag.empty or set(frag[3]) != set(valid_barcode):
        raise ValueError("QC barcode identities do not match retained fragment cells")
    if not frag[0].isin(valid_chrom).any():
        raise ValueError("No fragment chromosomes match the selected reference build")
    print("Finished loading fragment file.")
    
    ########### Create aggregated tile file for 500bp tiles ###########
    
    #2a. aggregate for each tile
    frag_stacked = pd.concat([frag[[0,1,3,4]],frag[[0,2,3,4]].rename(columns={2:1})], ignore_index=True)
    frag_stacked[4] = 1 #binarize count
    frag_stacked[1] = frag_stacked[1]//500 # aggregate counts per 500bp tile
    barcodes, barcode_id = np.unique(frag_stacked[3], return_inverse=True)
    frag_stacked[5] = barcode_id
    #2b. create tile files
    tile_dict = {}
    for chrom in list(set(frag_stacked[0])):
        if chrom in valid_chrom:
            frag_grouped = frag_stacked[frag_stacked[0]==chrom][[1,5,4]].groupby([5,1]).sum().reset_index()
            frag_grouped = csr_matrix((frag_grouped[4].astype(np.float64), 
                                       (frag_grouped[5].values, frag_grouped[1].values)), 
                                       shape=(barcodes.shape[0], chrom_size.loc[chrom].values[0]//500+1))
            tile_dict[chrom] = frag_grouped
            print(chrom)
    #2c. save to file
    with open(f"{save_path}/atac/{cell_type_prefix}_tile_500bp_dict.p", "wb") as handle:
        pickle.dump(tile_dict, handle)
    np.save(f"{save_path}/atac/{cell_type_prefix}_tile_500bp_barcode.npy", barcodes.astype(str))
    del frag_stacked, tile_dict
    print("500bp tile data processed.")
    
    ########### Create pseudo-bulk data for evaluation ###########
    
    #3a. aggregate for 50bp tiles
    frag_stacked = pd.concat([frag[[0,1,4]],frag[[0,2,4]].rename(columns={2:1})], ignore_index=True)
    frag_stacked[4] = 1 #binarize counte
    frag_stacked[1] = frag_stacked[1]//50 # aggregate counts per 50bp tile
    tile_dict = {}
    for chrom in list(set(frag_stacked[0])):
        if chrom in valid_chrom:
            frag_grouped = frag_stacked[frag_stacked[0]==chrom][[1,4]].groupby([1]).sum().reset_index()
            frag_grouped = pd.DataFrame(frag_grouped[4].values, 
                                        index = frag_grouped[1].values).reindex(np.arange(0,
                                                                                          chrom_size.loc[chrom].values[0]//50+1)).fillna(0).values
            tile_dict[chrom] = frag_grouped
    #3b. save
    with open(f"{save_path}/atac/{cell_type_prefix}_tile_pbulk_50bp_dict.p", "wb") as handle:
        pickle.dump(tile_dict, handle)
    del frag_stacked, tile_dict
    print("pseudo-bulk data processed.")
    
    ########### Create aggregated tile file for 50bp tiles ###########
    
    #4a. aggregate for each tile
    frag_stacked = pd.concat([frag[[0,1,3,4]],frag[[0,2,3,4]].rename(columns={2:1})], ignore_index=True)
    frag_stacked[4] = 1 #binarize counte
    frag_stacked[1] = frag_stacked[1]//50 # aggregate counts per 50bp tile
    barcodes, barcode_id = np.unique(frag_stacked[3], return_inverse=True)
    frag_stacked[5] = barcode_id
    #4b. create tile files
    tile_dict = {}
    for chrom in valid_chrom:
        frag_grouped = frag_stacked[frag_stacked[0]==chrom][[1,5,4]].groupby([5,1]).sum().reset_index()
        frag_grouped = csr_matrix((frag_grouped[4].astype(np.float64), 
                                   (frag_grouped[5], frag_grouped[1])), 
                                   shape=(barcodes.shape[0], chrom_size.loc[chrom].values[0]//50+1))
        tile_dict[chrom] = frag_grouped
    #4c. save
    with open(f"{save_path}/atac/{cell_type_prefix}_tile_50bp_dict.p", "wb") as handle:
        pickle.dump(tile_dict, handle)  # save it into a file named save.p
    np.save(f"{save_path}/atac/{cell_type_prefix}_tile_50bp_barcode.npy",barcodes.astype(str))
    del frag_stacked, tile_dict
    print("50bp tile file processed.")
    
    return None

def get_max_overlap(y,x):
    return np.max(x.dot(y))

def generate_cicero_metacell(nbrs, max_overlap, sampled_id = [0], seed=10):
    order = np.arange(nbrs.shape[0])
    np.random.seed(seed)
    np.random.shuffle(order)
    selected_idx = [False]*nbrs.shape[0]
    for i in sampled_id:
        selected_idx[i] = True
    for idx in order:
        selected_cells = nbrs[selected_idx]
        candidate = nbrs[idx]
        overlap = get_max_overlap(candidate,selected_cells)
        if overlap < max_overlap:
            selected_idx[idx] = True
    return selected_idx    

def metacell_computing(lsi_path, save_path, cell_type_prefix,
                       n_neighbors = 100, max_overlap = 33, seed=10):
    lsi = pd.read_csv(lsi_path, index_col = 0)
    tile_barcodes = np.load(f"{save_path}/atac/{cell_type_prefix}_tile_500bp_barcode.npy", allow_pickle=False)
    tile50 = np.load(f"{save_path}/atac/{cell_type_prefix}_tile_50bp_barcode.npy", allow_pickle=False)
    if not np.array_equal(tile_barcodes, tile50):
        raise ValueError("50bp and 500bp tile barcode orders differ")
    order = alignment_positions(lsi.index, tile_barcodes)
    lsi = lsi.iloc[order]
    if not np.isfinite(lsi.values).all():
        raise ValueError("LSI contains nonfinite dimensions")
    if not 1 <= n_neighbors <= len(lsi) or not 0 < max_overlap <= n_neighbors:
        raise ValueError("Set explicit valid n_neighbors/max_overlap for this cell count")
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean').fit(lsi.values)
    nbrs = nbrs.kneighbors_graph(lsi.values).toarray()
    selected_idx = generate_cicero_metacell(nbrs, max_overlap = max_overlap, seed=seed)
    metacell_assignment = nbrs[selected_idx,:]
    pd.DataFrame(metacell_assignment).to_csv(f"{save_path}/atac/{cell_type_prefix}_metacell_mask.csv")
    np.save(f"{save_path}/atac/{cell_type_prefix}_metacell_barcodes.npy", tile_barcodes)
    return None

#########################
#     Script running    #
#########################

def main(argv=None):
    global chrom_size, VALID_CHROM

    args = parser.parse_args(argv)
    config = vars(args)
    print(config)
    CELL_TYPE = config['cell_type_prefix']
    FRAG_FILE = config['fragment_file']
    BARCODE_FILE = config['barcode_file']
    GENOME_ASSEMBLY = config['genome_assembly']
    CHROM_SIZES = config['chrom_sizes']
    LSI_PATH = config['lsi_file']
    SAVE_PATH = config['save_path']
    chrom_size, VALID_CHROM = load_chrom_sizes(CHROM_SIZES, GENOME_ASSEMBLY)
    os.makedirs(os.path.join(SAVE_PATH, "atac"), exist_ok=True)

    
    _ = atac_processing(frag_path = FRAG_FILE, nrows_skip = 0, 
                        valid_barcode_path = BARCODE_FILE,
                        save_path = SAVE_PATH, cell_type_prefix = CELL_TYPE,
                        valid_chrom = VALID_CHROM)
    _ = metacell_computing(lsi_path = LSI_PATH, save_path = SAVE_PATH, cell_type_prefix = CELL_TYPE,
                          n_neighbors=args.n_neighbors, max_overlap=args.max_overlap,
                          seed=args.metacell_seed)
    
if __name__ == '__main__':
    main()
