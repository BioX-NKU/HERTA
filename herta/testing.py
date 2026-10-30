"""Inference datasets for pseudobulk/CTCF and topic-conditioned HERTA models."""

import numpy as np
import torch


class test_Dataset_pbulk_multilevel(torch.utils.data.Dataset):
    """Test dataset for the 3-track Pbulk+CTCF pretrain model."""
    
    def __init__(
        self,
        input_size,
        pbulk_dict,
        ctcf_dict,
        list_cts,
        list_starts,
        list_chroms,
        transform,
        is_training=False,
    ):
        self.input_size = input_size
        self.pbulk_dict = pbulk_dict
        self.ctcf_dict = ctcf_dict
        self.list_cts = list_cts
        self.list_starts = list_starts
        self.list_chroms = list_chroms
        self.is_training = is_training
        self.transform = transform

    def __len__(self):
        return len(self.list_starts)

    def __getitem__(self, index):
        chrom = self.list_chroms[index]
        start = self.list_starts[index]

        shift_input_by = 0
        pbulk_res = 50

        input_window = int(self.input_size / pbulk_res)
        atac_start_ind = int(start / pbulk_res)

        # Process pbulk data
        X1 = self.pbulk_dict[chrom]
        X1 = X1[
            max(0, atac_start_ind + shift_input_by) : atac_start_ind
            + shift_input_by
            + input_window
        ]
        X1 = self.transform(X1.toarray().astype(np.float32).T)

        # Process CTCF data
        X2 = self.ctcf_dict[chrom]
        X2 = X2[max(0, atac_start_ind) : atac_start_ind + input_window]
        X2 = torch.tensor(X2.toarray().astype(np.float32).T)

        # Handle negative start positions
        if start < 0:
            desired_shape = (1, 80200)
            num_zeros = desired_shape[1] - X1.shape[1]
            X1 = torch.cat([torch.zeros((X1.shape[0], num_zeros)), X1], dim=1)
            X2 = torch.cat([torch.zeros((X2.shape[0], num_zeros)), X2], dim=1)

        # Ensure proper dimensions
        if isinstance(X1, torch.Tensor):
            if len(X1.shape) > 1:
                X1 = X1.squeeze()
        else:
            X1 = torch.tensor(X1)
            if len(X1.shape) > 1:
                X1 = X1.squeeze()

        if len(X2.shape) > 1:
            X2 = X2.squeeze()

        return {
            'pbulk': X1,     # [80200]
            'ctcf': X2,      # [80200]
        }

class test_Dataset_pbulk_topic_multilevel(torch.utils.data.Dataset):
    """Test dataset for the final 6-track Pbulk+CTCF+Topic model."""
    
    def __init__(
        self,
        input_size,
        pbulk_dict,
        ctcf_dict,
        topic_dict,
        list_cts,
        list_starts,
        list_chroms,
        transform,
        is_training=False,
    ):
        self.input_size = input_size
        self.pbulk_dict = pbulk_dict
        self.ctcf_dict = ctcf_dict
        self.topic_dict = topic_dict
        self.list_cts = list_cts
        self.list_starts = list_starts
        self.list_chroms = list_chroms
        self.is_training = is_training
        self.transform = transform

    def __len__(self):
        return len(self.list_starts)

    def __getitem__(self, index):
        chrom = self.list_chroms[index]
        start = self.list_starts[index]

        shift_input_by = 0
        pbulk_res = 50
        scatac_res = 500

        input_window = int(self.input_size / pbulk_res)
        atac_start_ind = int(start / pbulk_res)
        sc_input_window = int(self.input_size / scatac_res)
        sc_atac_start_ind = int(start / scatac_res)

        # Process pbulk data
        X1 = self.pbulk_dict[chrom]
        X1 = X1[
            max(0, atac_start_ind + shift_input_by) : atac_start_ind
            + shift_input_by
            + input_window
        ]
        X1 = self.transform(X1.toarray().astype(np.float32).T)

        # Process topic data
        Z_topic = self.topic_dict[chrom][
            max(0, sc_atac_start_ind + shift_input_by) * scatac_res : \
            (sc_atac_start_ind + shift_input_by + sc_input_window) * scatac_res
        ]
        Z_topic = torch.from_numpy(Z_topic)

        # Process CTCF data
        X2 = self.ctcf_dict[chrom]
        X2 = X2[max(0, atac_start_ind) : atac_start_ind + input_window]
        X2 = torch.tensor(X2.toarray().astype(np.float32).T)

        # Handle negative start positions
        if start < 0:
            desired_shape = (1, 80200)
            num_zeros = desired_shape[1] - X1.shape[1]
            X1 = torch.cat([torch.zeros((X1.shape[0], num_zeros)), X1], dim=1)
            X2 = torch.cat([torch.zeros((X2.shape[0], num_zeros)), X2], dim=1)

            desired_shape_topic = (8020, 50)
            num_zeros_topic = desired_shape_topic[0] - Z_topic.shape[0]
            Z_topic = torch.cat([torch.zeros((num_zeros_topic, Z_topic.shape[1])), Z_topic], dim=0)

        # Ensure proper dimensions
        if isinstance(X1, torch.Tensor):
            if len(X1.shape) > 1:
                X1 = X1.squeeze()
        else:
            X1 = torch.tensor(X1)
            if len(X1.shape) > 1:
                X1 = X1.squeeze()

        if len(X2.shape) > 1:
            X2 = X2.squeeze()

        return {
            'topic': Z_topic,
            'pbulk': X1,
            'ctcf': X2,
        }
