
import pickle
import os

import numpy as np
from scipy import sparse
import torch


def get_effective_genome_size(genome):
    if "hg" in genome:
        effective_genome_size = 2913022398
    elif "mm" in genome:
        effective_genome_size = 2652783500
    else:
        raise ValueError(
            "Please compute effective_genome_size manuallly and modify this function accordingly."
        )
    return effective_genome_size

def get_metacell_profile(sc_atac_dict, nbrs):
    metacell_sc_atac_dict = {}
    metacell = nbrs
    for chrom in list(sc_atac_dict.keys()):
        metacell_sc_atac_dict[chrom] = (
            sparse.csr_matrix(metacell) * sc_atac_dict[chrom]
        )
        metacell_sc_atac_dict[chrom] = sparse.csr_matrix(
            metacell_sc_atac_dict[chrom].toarray().transpose()
        )

    return metacell_sc_atac_dict

def _normalize_resolution_label(resolution):
    """Return canonical resolution label used in clean HERTA Hi-C filenames."""
    if isinstance(resolution, int):
        return f"{resolution}kb"
    if isinstance(resolution, float) and resolution.is_integer():
        return f"{int(resolution)}kb"
    resolution = str(resolution)
    if resolution.endswith("kb"):
        return resolution
    if resolution.endswith("bp"):
        bp = int(resolution[:-2])
        if bp % 1000 != 0:
            raise ValueError(f"Unsupported Hi-C resolution label: {resolution}")
        return f"{bp // 1000}kb"
    if resolution.isdigit():
        return f"{resolution}kb"
    raise ValueError(f"Unsupported Hi-C resolution label: {resolution}")


def load_hic_multi_resolution(data_path, ct, resolutions=("5kb", "10kb", "25kb")):
    """Load ``<data_path>/hic/{ct}_hic_zscore_dict_{resolution}.p``.

    Resolution labels are 5kb,10kb and25kb. Dictionaries may be chromosome-keyed
    or already nested by cell type; the returned mapping is resolution→cell→chr.
    """
    hic_dir = os.path.join(data_path, "hic")
    multi_hic_dict = {}

    for resolution in resolutions:
        resolution = _normalize_resolution_label(resolution)
        filename = f"{ct}_hic_zscore_dict_{resolution}.p"
        filepath = os.path.join(hic_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Hi-C file not found: {filepath}")

        with open(filepath, "rb") as handle:
            hic_dict = pickle.load(handle)

        if ct not in hic_dict:
            hic_dict = {ct: hic_dict}
        multi_hic_dict[resolution] = hic_dict

    return multi_hic_dict


def load_hic(data_path, ct, resolution="10kb"):
    """Single-resolution compatibility wrapper around load_hic_multi_resolution."""
    resolution = _normalize_resolution_label(resolution)
    multi_dict = load_hic_multi_resolution(data_path, ct, [resolution])
    zscore_dict = multi_dict[resolution]
    qvalue_dict = multi_dict[resolution]
    return zscore_dict, qvalue_dict

def load_atac(data_path, ct, pbulk_only=False):
    bulk_atac_dict = pickle.load(
        open(os.path.join(data_path, "atac/{}_tile_50bp_dict.p".format(ct)), "rb")
    )
    if pbulk_only:
        sc_atac_dict = None
    else:
        sc_atac_dict = pickle.load(
            open(os.path.join(data_path, "atac/{}_tile_500bp_dict.p".format(ct)), "rb")
        )
        import pandas as pd
        metacell_mask = pd.read_csv(
            os.path.join(data_path, "atac/{}_metacell_mask.csv".format(ct)), index_col=0
        ).values
        sc_atac_dict = get_metacell_profile(sc_atac_dict, metacell_mask)

    return bulk_atac_dict, sc_atac_dict

def load_atac_eval(data_path, ct, pbulk_only=False):
    bulk_atac_dict = pickle.load(
        open(os.path.join(data_path, "atac/{}_tile_pbulk_50bp_dict.p".format(ct)), "rb")
    )
    
    for chrom in bulk_atac_dict.keys():
        bulk_atac_dict[chrom] = sparse.csr_matrix(bulk_atac_dict[chrom])
        
    if pbulk_only:
        sc_atac_dict = None
    else:
        sc_atac_dict = pickle.load(
            open(os.path.join(data_path, "atac/{}_tile_500bp_dict.p".format(ct)), "rb")
        )
        import pandas as pd
        metacell_mask = pd.read_csv(
            os.path.join(data_path, "atac/{}_metacell_mask.csv".format(ct)), index_col=0
        ).values
        sc_atac_dict = get_metacell_profile(sc_atac_dict, metacell_mask)

    return bulk_atac_dict, sc_atac_dict

def load_ctcf_motif(data_path, genome, ctcf_path=None):
    if ctcf_path is None:
        ctcf_path = os.path.join(
            data_path, "dna/{}_ctcf_motif_score.p".format(genome)
        )
    ctcf_path = os.path.expanduser(str(ctcf_path))
    if not os.path.isfile(ctcf_path):
        raise FileNotFoundError(f"CTCF motif score file not found: {ctcf_path}")

    with open(ctcf_path, "rb") as handle:
        ctcf_dict = pickle.load(handle)
    for chrom in ctcf_dict.keys():
        ctcf_dict[chrom] = ctcf_dict[chrom].T

    return ctcf_dict

def get_num_cells(atac_dict, dim=0):
    return atac_dict["chr19"].shape[dim]

def get_libsize(bulk_atac_dict):
    libsize_cell = np.zeros(get_num_cells(bulk_atac_dict))
    for chrom in bulk_atac_dict.keys():
        libsize_cell += np.array(bulk_atac_dict[chrom].sum(1).T)[0]

    return libsize_cell

def normalize(tensor, effective_genome_size=0, libsize=0, training=False):
    tensor = torch.tensor(tensor)
    if training:
        assert libsize != 0, "Libsize cannot be 0 during training."
        scale_factor = (libsize * 150) / effective_genome_size
        tensor.div_(torch.as_tensor(scale_factor))
    tensor = torch.log(tensor + 1)
    return tensor

def normalize_bulk_dict(pbulk_dict, effective_genome_size):
    normalized_pbulk_dict = {}
    libsize = 0
    for chrom in pbulk_dict.keys():
        libsize += pbulk_dict[chrom].toarray().sum()
    scale_factor = (libsize * 150) / (effective_genome_size)
    for chrom in pbulk_dict.keys():
        y = pbulk_dict[chrom].toarray()
        y_t = y / scale_factor
        normalized_pbulk_dict[chrom] = sparse.csr_matrix(y_t)
    return normalized_pbulk_dict

def get_y_vstripe(start, hic, input_size, resolution=1e04):
    hic_start_row = start
    prediction_bins = int(2000000 / resolution)
    hic_start_col = start + prediction_bins * np.int64(resolution)
    ind_row = np.int64(hic_start_row // resolution)
    ind_col = np.int64(hic_start_col // resolution)
    vert = hic[ind_row : ind_row + prediction_bins, ind_col].toarray()

    hic_start_row = start + prediction_bins * np.int64(resolution)
    hic_start_col = start + (1+prediction_bins) * np.int64(resolution)
    ind_row = np.int64(hic_start_row // resolution)
    ind_col = np.int64(hic_start_col // resolution)

    hori = hic[ind_row, ind_col : ind_col + prediction_bins].toarray()

    y = np.append(vert, hori).astype(np.float32)
    y = y.clip(-16, 16)
    return y

def get_starts(
    chrom_list,
    start_dict,
    end_dict,
    chrom_start_offset,
    chrom_end_offset,
    multi_ct=False,
    ct_list=[],
    step=5e4,
):
    """
    Function to get the start and chrom list for train/test/val
    """
    ctl = []
    startl = []
    chroml = []
    for chrom in chrom_list:
        chrom = "chr{}".format(chrom)
        cur_starts = list(
            np.arange(
                start_dict[chrom] + chrom_start_offset,
                end_dict[chrom] - chrom_end_offset,
                step,
            ).astype(int)
        )
        startl = startl + cur_starts
        chroml = chroml + list(np.repeat(chrom, len(cur_starts)))

    if multi_ct:
        for ct in ct_list:
            ctl += list(np.repeat(ct, len(chroml)))
        startl = startl * len(ct_list)
        chroml = chroml * len(ct_list)
        assert (len(startl) == len(chroml)) & (len(ctl) == len(chroml))

    return ctl, startl, chroml

def get_chrom_starts(genome):
    if genome == "hg38":
        start_dict = {
            "chr1": 0,
            "chr2": 0,
            "chr3": 0,
            "chr4": 0,
            "chr5": 0,
            "chr6": 0,
            "chr7": 0,
            "chr8": 0,
            "chr9": 0,
            "chr10": 0,
            "chr11": 0,
            "chr12": 0,
            "chr13": 0,
            "chr14": 0,
            "chr15": 0,
            "chr16": 0,
            "chr17": 0,
            "chr18": 0,
            "chr19": 0,
            "chr20": 0,
            "chr21": 0,
            "chr22": 0,
            "chrX": 0,
        }
        end_dict = {
            "chr1": 248950000,
            "chr2": 242190000,
            "chr3": 198020000,
            "chr4": 190210000,
            "chr5": 180910000,
            "chr6": 170800000,
            "chr7": 159130000,
            "chr8": 145130000,
            "chr9": 138390000,
            "chr10": 133790000,
            "chr11": 135000000,
            "chr12": 133270000,
            "chr13": 114360000,
            "chr14": 107040000,
            "chr15": 101990000,
            "chr16": 90330000,
            "chr17": 81190000,
            "chr18": 78070000,
            "chr19": 58610000,
            "chr20": 63020000,
            "chr21": 46700000,
            "chr22": 51300000,
            "chrX": 155270000,
        }

    elif genome == "hg19":
        start_dict = {
            "chr1": 0,
            "chr2": 0,
            "chr3": 0,
            "chr4": 0,
            "chr5": 0,
            "chr6": 0,
            "chr7": 0,
            "chr8": 0,
            "chr9": 0,
            "chr10": 0,
            "chr11": 0,
            "chr12": 0,
            "chr13": 0,
            "chr14": 0,
            "chr15": 0,
            "chr16": 0,
            "chr17": 0,
            "chr18": 0,
            "chr19": 0,
            "chr20": 0,
            "chr21": 0,
            "chr22": 0,
            "chrX": 0,
        }

        end_dict = {
            "chr1": 249250000,
            "chr2": 243190000,
            "chr3": 198020000,
            "chr4": 191150000,
            "chr5": 180910000,
            "chr6": 171110000,
            "chr7": 159130000,
            "chr8": 146360000,
            "chr9": 141210000,
            "chr10": 135530000,
            "chr11": 135000000,
            "chr12": 133850000,
            "chr13": 115160000,
            "chr14": 107340000,
            "chr15": 102530000,
            "chr16": 90350000,
            "chr17": 81190000,
            "chr18": 78070000,
            "chr19": 59120000,
            "chr20": 63020000,
            "chr21": 48120000,
            "chr22": 51300000,
            "chrX": 155270000,
        }

    elif genome == "mm10":
        start_dict = {
            "chr1": 0,
            "chr2": 0,
            "chr3": 0,
            "chr4": 0,
            "chr5": 0,
            "chr6": 0,
            "chr7": 0,
            "chr8": 0,
            "chr9": 0,
            "chr10": 0,
            "chr11": 0,
            "chr12": 0,
            "chr13": 0,
            "chr14": 0,
            "chr15": 0,
            "chr16": 0,
            "chr17": 0,
            "chr18": 0,
            "chr19": 0,
            "chrX": 0,
        }

        end_dict = {
            "chr1": 195470000,
            "chr2": 182110000,
            "chr3": 160030000,
            "chr4": 156500000,
            "chr5": 151830000,
            "chr6": 149730000,
            "chr7": 145440000,
            "chr8": 129400000,
            "chr9": 124590000,
            "chr10": 130690000,
            "chr11": 122080000,
            "chr12": 120120000,
            "chr13": 120420000,
            "chr14": 124900000,
            "chr15": 104040000,
            "chr16": 98200000,
            "chr17": 94980000,
            "chr18": 90700000,
            "chr19": 61430000,
            "chrX": 171030000,
        }
    else:
        raise Exception(
            "Chromosome info not provided for given genome version. Please compute manually and update here."
        )

    return start_dict, end_dict
