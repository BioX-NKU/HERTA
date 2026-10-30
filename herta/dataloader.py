
import numpy as np
import torch
from pathlib import Path
if __package__:
    from .utils import get_y_vstripe
else:
    from utils import get_y_vstripe

def create_topic_data_dict(topk_paths, n_topics=50, data_path=None, **kwargs):
    """Load pre-aggregated topic data."""
    if data_path is None:
        raise ValueError("data_path is required")
    
    ct_list = list(topk_paths.keys())
    
    chrom_lengths = {
        "chr1": 248950000, "chr2": 242190000, "chr3": 198020000,
        "chr4": 190210000, "chr5": 180910000, "chr6": 170800000,
        "chr7": 159130000, "chr8": 145130000, "chr9": 138390000,
        "chr10": 133790000, "chr11": 135000000, "chr12": 133270000,
        "chr13": 114360000, "chr14": 107040000, "chr15": 101990000,
        "chr16": 90330000, "chr17": 81190000, "chr18": 78070000,
        "chr19": 58610000, "chr20": 63020000, "chr21": 46700000,
    }
    
    ct_dict = {}
    data_dir = Path(data_path)
    if (data_dir / "topic").is_dir():
        data_dir = data_dir / "topic"
    
    for ct in ct_list:
        print(f"Loading aggregated topic data for {ct}...")
        
        npz_path = data_dir / f"{ct}_topics_aggregated.npz"
        
        if not npz_path.exists():
            raise FileNotFoundError(f"File not found: {npz_path}")
        
        data = np.load(npz_path)
        available_chroms = data['chromosomes'].tolist()
        
        ct_dict[ct] = {}
        
        for chrom in chrom_lengths.keys():
            if chrom not in available_chroms:
                continue
            
            matrix = data[f'{chrom}_matrix']
            n_bins = int(data[f'{chrom}_n_bins'])
            
            ct_dict[ct][chrom] = TopicChromLoader(matrix, n_bins, n_topics=n_topics)
            
            size_mb = matrix.nbytes / (1024**2)
            print(f"  {chrom}: {size_mb:.1f}MB")
        
        print(f"  ✓ Loaded {len(ct_dict[ct])} chromosomes")
    
    return TopicDataDict(ct_dict, n_topics=n_topics)

class TopicDataDict:
    def __init__(self, ct_dict, n_topics=50):
        self.n_topics = n_topics
        self.data = ct_dict
    
    def __getitem__(self, ct):
        return TopicCellTypeAccessor(self.data[ct])

class TopicCellTypeAccessor:
    def __init__(self, chrom_loaders):
        self.chrom_loaders = chrom_loaders
    
    def __getitem__(self, chrom):
        return TopicChromAccessor(self.chrom_loaders[chrom])

class TopicChromAccessor:
    def __init__(self, loader):
        self.loader = loader
    
    def __getitem__(self, key):
        if isinstance(key, slice):
            start = key.start if key.start is not None else 0
            stop = key.stop
            return self.loader.get_region(start, stop)
        else:
            raise NotImplementedError("Use [start:end] slicing")

class TopicChromLoader:
    def __init__(self, matrix, n_bins, n_topics=50, resolution=500):
        self.matrix = matrix
        self.n_bins = n_bins
        self.n_topics = n_topics
        self.resolution = resolution
    
    def get_region(self, start, stop):
        bin_start = start // self.resolution
        bin_end = stop // self.resolution
        region = self.matrix[bin_start:bin_end]
        # Return a writable contiguous copy.
        return np.array(region, dtype=np.float32, copy=True)

class Dataset_pbulk_multilevel(torch.utils.data.Dataset):
    """Multi-resolution V-stripe dataset for the 3-track Pbulk+CTCF pretrain model."""
    
    def __init__(
        self,
        input_size,
        effective_genome_size,
        hic_dict_5kb,
        hic_dict_10kb,
        hic_dict_25kb,
        pbulk_dict,
        ctcf_dict,
        n_cells,
        libsize_cell,
        list_cts,
        list_starts,
        list_chroms,
        transform,
        is_training=False,
    ):
        self.input_size = input_size
        self.effective_genome_size = effective_genome_size
        self.hic_dict_5kb = hic_dict_5kb
        self.hic_dict_10kb = hic_dict_10kb
        self.hic_dict_25kb = hic_dict_25kb
        self.pbulk_dict = pbulk_dict
        self.ctcf_dict = ctcf_dict
        self.n_cells = n_cells
        self.libsize_cell = libsize_cell
        self.list_cts = list_cts
        self.list_starts = list_starts
        self.list_chroms = list_chroms
        self.is_training = is_training
        self.transform = transform

    def __len__(self):
        return len(self.list_starts)

    def __getitem__(self, index):
        ct = self.list_cts[index]
        chrom = self.list_chroms[index]
        start = self.list_starts[index]

        shift_input_by = 0
        train_step = 5e04
        pbulk_res = 50

        if self.is_training:
            start = (
                np.int64(np.random.randint(0, np.int64(train_step / 1e4)) * 1e4) + start
            )
            shift_input_by = np.random.randint(-1, 2)

        input_window = np.int64(self.input_size / pbulk_res)
        atac_start_ind = np.int64(start / pbulk_res)

        X1 = self.pbulk_dict[ct][chrom]
        X1 = X1[
            :,
            atac_start_ind + shift_input_by : atac_start_ind + shift_input_by + input_window,
        ]

        X2 = self.ctcf_dict[chrom]
        X2 = X2[atac_start_ind + shift_input_by : atac_start_ind + shift_input_by + input_window]
        X2 = torch.tensor(X2.toarray().astype(np.float32).T)

        if self.is_training:
            size = np.random.randint(500, 5000)
            mask = np.random.choice(
                np.arange(self.n_cells[ct]), size=size, replace=True
            )
            X1 = X1[mask]
            cur_libsize = self.libsize_cell[ct][mask].sum()
        else:
            cur_libsize = self.libsize_cell[ct].sum()

        X1 = np.array(X1.sum(0)).astype(np.float32)
        X1 = self.transform(
            X1,
            effective_genome_size=self.effective_genome_size,
            libsize=cur_libsize,
            training=True,
        )

        if isinstance(X1, torch.Tensor):
            if len(X1.shape) > 1:
                X1 = X1.squeeze()
        else:
            X1 = torch.tensor(X1)
            if len(X1.shape) > 1:
                X1 = X1.squeeze()

        if len(X2.shape) > 1:
            X2 = X2.squeeze()

        targets = {}
        
        if chrom in self.hic_dict_5kb[ct]:
            hic_5kb = self.hic_dict_5kb[ct][chrom]
            y_5kb = get_y_vstripe(start, hic_5kb, self.input_size, resolution=5000)
            y_5kb = torch.tensor(y_5kb)
            y_5kb = y_5kb[0:400]
            targets['target_5kb'] = y_5kb
        
        if chrom in self.hic_dict_10kb[ct]:
            hic_10kb = self.hic_dict_10kb[ct][chrom]
            y_10kb = get_y_vstripe(start, hic_10kb, self.input_size, resolution=10000)
            y_10kb = torch.tensor(y_10kb)
            y_10kb = y_10kb[0:200]
            targets['target_10kb'] = y_10kb
        
        if chrom in self.hic_dict_25kb[ct]:
            hic_25kb = self.hic_dict_25kb[ct][chrom]
            y_25kb = get_y_vstripe(start, hic_25kb, self.input_size, resolution=25000)
            y_25kb = torch.tensor(y_25kb)
            y_25kb = y_25kb[0:80]
            targets['target_25kb'] = y_25kb

        result = {
            'pbulk': X1,
            'ctcf': X2,
        }
        result.update(targets)
        
        return result

class Dataset_pbulk_topic_multilevel(torch.utils.data.Dataset):
    """Multi-resolution V-stripe dataset for the final 6-track Pbulk+CTCF+Topic model."""
    
    def __init__(
        self,
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
        list_cts,
        list_starts,
        list_chroms,
        transform,
        is_training=False,
    ):
        self.input_size = input_size
        self.effective_genome_size = effective_genome_size
        self.hic_dict_5kb = hic_dict_5kb
        self.hic_dict_10kb = hic_dict_10kb
        self.hic_dict_25kb = hic_dict_25kb
        self.pbulk_dict = pbulk_dict
        self.ctcf_dict = ctcf_dict
        self.topic_dict = topic_dict
        self.n_cells = n_cells
        self.libsize_cell = libsize_cell
        self.list_cts = list_cts
        self.list_starts = list_starts
        self.list_chroms = list_chroms
        self.is_training = is_training
        self.transform = transform

    def __len__(self):
        return len(self.list_starts)

    def __getitem__(self, index):
        ct = self.list_cts[index]
        chrom = self.list_chroms[index]
        start = self.list_starts[index]

        shift_input_by = 0
        train_step = 5e04
        pbulk_res = 50
        scatac_res = 500

        if self.is_training:
            start = (
                np.int64(np.random.randint(0, np.int64(train_step / 1e4)) * 1e4) + start
            )
            shift_input_by = np.random.randint(-1, 2)

        input_window = np.int64(self.input_size / pbulk_res)
        atac_start_ind = np.int64(start / pbulk_res)
        sc_input_window = np.int64(self.input_size / scatac_res)
        sc_atac_start_ind = np.int64(start / scatac_res)

        X1 = self.pbulk_dict[ct][chrom]
        X1 = X1[
            :,
            atac_start_ind + shift_input_by : atac_start_ind + shift_input_by + input_window,
        ]
        
        # Topic loader returns a writable copy.
        Z_topic = self.topic_dict[ct][chrom][
            (sc_atac_start_ind + shift_input_by) * scatac_res : \
            (sc_atac_start_ind + shift_input_by + sc_input_window) * scatac_res
        ]
        
        X2 = self.ctcf_dict[chrom]
        X2 = X2[atac_start_ind + shift_input_by : atac_start_ind + shift_input_by + input_window]
        X2 = torch.tensor(X2.toarray().astype(np.float32).T)

        if self.is_training:
            size = np.random.randint(500, 5000)
            mask = np.random.choice(np.arange(self.n_cells[ct]), size=size, replace=True)
            X1 = X1[mask]
            cur_libsize = self.libsize_cell[ct][mask].sum()
        else:
            cur_libsize = self.libsize_cell[ct].sum()

        X1 = np.array(X1.sum(0)).astype(np.float32)
        X1 = self.transform(
            X1,
            effective_genome_size=self.effective_genome_size,
            libsize=cur_libsize,
            training=True,
        )

        # Convert topic region to tensor; the region is already float32.
        Z_topic = torch.from_numpy(Z_topic)

        if isinstance(X1, torch.Tensor):
            if len(X1.shape) > 1:
                X1 = X1.squeeze()
        else:
            X1 = torch.tensor(X1)
            if len(X1.shape) > 1:
                X1 = X1.squeeze()

        if len(X2.shape) > 1:
            X2 = X2.squeeze()

        targets = {}
        
        if chrom in self.hic_dict_5kb[ct]:
            hic_5kb = self.hic_dict_5kb[ct][chrom]
            y_5kb = get_y_vstripe(start, hic_5kb, self.input_size, resolution=5000)
            y_5kb = torch.tensor(y_5kb)
            y_5kb = y_5kb[0:400]
            targets['target_5kb'] = y_5kb
        
        if chrom in self.hic_dict_10kb[ct]:
            hic_10kb = self.hic_dict_10kb[ct][chrom]
            y_10kb = get_y_vstripe(start, hic_10kb, self.input_size, resolution=10000)
            y_10kb = torch.tensor(y_10kb)
            y_10kb = y_10kb[0:200]
            targets['target_10kb'] = y_10kb
        
        if chrom in self.hic_dict_25kb[ct]:
            hic_25kb = self.hic_dict_25kb[ct][chrom]
            y_25kb = get_y_vstripe(start, hic_25kb, self.input_size, resolution=25000)
            y_25kb = torch.tensor(y_25kb)
            y_25kb = y_25kb[0:80]
            targets['target_25kb'] = y_25kb

        result = {
            'topic': Z_topic,
            'pbulk': X1,
            'ctcf': X2,
        }
        result.update(targets)
        
        return result

