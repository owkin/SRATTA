import os

import numpy as np
import torch
import torchvision
from torchvision.transforms import transforms

from sratta.datasets.generic_dataset import GenericDataset


class CIFAR10Dataset(GenericDataset):
    def __init__(
        self,
        num_centers,
        dataset_size,
        test_dataset_size,
        split_with_dirichlet,
        dirichlet_param,
        dataset_folder,
    ):
        self.input_dim = 32 * 32 * 3
        self.output_dim = 10
        self.num_labels = 10
        self.name = "cifar10"
        self.task = "classification"
        self.dataset_size = dataset_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.dataset_folder = os.path.join(dataset_folder, "cifar10")
        super(CIFAR10Dataset, self).__init__(
            num_centers,
            dataset_size,
            test_dataset_size,
            split_with_dirichlet,
            dirichlet_param,
        )

    def get_pooled_dataset(self):
        transform = transforms.Compose([transforms.ToTensor()])
        cifar10 = torchvision.datasets.CIFAR10(
            root=self.dataset_folder, download=True, transform=transform
        )
        return cifar10

    def compare_candidate(self, candidate, list_bank):
        rtol = 1e-9
        atol = 1.0 / 256.0 / 20.0
        return [np.allclose(candidate, a, rtol=rtol, atol=atol) for a in list_bank]

    def project_candidate(self, candidate):
        return np.around(candidate * 255) / 255.0

    def oracle(self, candidate):
        precision = 1.0 / 256.0 / 20.0
        candidate_values = list(set(candidate.ravel()))
        possible_values = np.linspace(0.0, 1.0, 256)

        for cv in candidate_values:
            if np.min(np.abs(cv - possible_values)) > precision:
                return False
        return True
