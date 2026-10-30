import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

if __package__:
    from .dataloader import (
        create_topic_data_dict, TopicDataDict, TopicCellTypeAccessor,
        TopicChromAccessor, TopicChromLoader,
    )
    from .utils import get_y_vstripe
else:
    from dataloader import (
        create_topic_data_dict, TopicDataDict, TopicCellTypeAccessor,
        TopicChromAccessor, TopicChromLoader,
    )
    from utils import get_y_vstripe



class Dataset_pbulk_topic_multilevel_S2D(torch.utils.data.Dataset):
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
        min_cells=100,
        max_cells=1000,
        sample_cells=None,
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
        self.sample_cells = is_training if sample_cells is None else sample_cells
        self.transform = transform

        self.min_cells = min_cells
        self.max_cells = max_cells

        print(f"S2D Dataset initialized: cell sampling range [{min_cells}, {max_cells}]")

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

        Z_topic = self.topic_dict[ct][chrom][
            (sc_atac_start_ind + shift_input_by) * scatac_res : \
            (sc_atac_start_ind + shift_input_by + sc_input_window) * scatac_res
        ]

        X2 = self.ctcf_dict[chrom]
        X2 = X2[atac_start_ind + shift_input_by : atac_start_ind + shift_input_by + input_window]
        X2 = torch.tensor(X2.toarray().astype(np.float32).T)

        if self.sample_cells:
            size = np.random.randint(self.min_cells, self.max_cells + 1)
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
