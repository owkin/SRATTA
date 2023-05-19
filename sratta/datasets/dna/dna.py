import os
from pathlib import Path

import numpy as np
import openml
import torch
from loguru import logger

from sratta.attack_pipeline.attack_steps.torch_dataset import TorchDatasetFromNumpy
from sratta.datasets.generic_dataset import GenericDataset


class DNADataset(GenericDataset):
    def __init__(
        self,
        num_centers,
        dataset_size,
        test_dataset_size,
        split_with_dirichlet,
        dirichlet_param,
        dataset_folder,
    ):
        self.input_dim = 180
        self.output_dim = 4
        self.num_labels = 4
        self.name = "dna"
        self.task = "classification"
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataset_folder = dataset_folder
        super(DNADataset, self).__init__(
            num_centers,
            dataset_size,
            test_dataset_size,
            split_with_dirichlet,
            dirichlet_param,
        )

    def get_pooled_dataset(self):
        dataset = openml.datasets.get_dataset("dna")
        X, _, _, _ = dataset.get_data(dataset_format="dataframe")
        X = self.remove_duplicate(X)

        y = X.values[:, -1].copy().astype(np.long)
        X = X.values[:, :-1].astype(np.float32)
        return TorchDatasetFromNumpy(X, y)

    def remove_duplicate(self, X):
        file_path = os.path.join(Path(__file__).parent, "list_idx.npy")
        list_idx = np.load(file_path)
        logger.info(f"Keeping only {len(list_idx)} index out of {len(X)}")
        return X.iloc[list_idx]
        # list_str = []
        # list_idx = []
        # logger.info("Removing duplicate (may takes 1 min)...")
        # for idx in range(len(X)):
        #     str_value = str(X.values[idx, :-1])
        #     if str_value not in list_str:
        #         list_str.append(str_value)
        #         list_idx.append(idx)
        # return X.iloc[list_idx]

    def oracle(self, candidate):
        atol = 1.0 / 1000.0
        candidate_values = list(set(candidate.ravel()))
        possible_values = np.array([0.0, 1.0])

        for cv in candidate_values:
            if np.min(np.abs(cv - possible_values)) > atol:
                return False

        return True

    def project_candidate(self, candidate):
        type_to_keep = candidate.dtype
        return np.around(candidate).astype(type_to_keep)

    def compare_candidate(self, candidate, list_bank):
        rtol = 1e-9
        atol = 1.0 / 100.0
        return [np.allclose(candidate, a, rtol=rtol, atol=atol) for a in list_bank]
