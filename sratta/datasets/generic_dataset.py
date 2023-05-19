from abc import ABC, abstractmethod

import numpy as np
from torch.utils.data import DataLoader, Subset

from sratta.attack_pipeline.attack_steps.torch_dataset import TorchDataSetWithIndices


class GenericDataset(ABC):
    def __init__(
        self,
        num_centers,
        dataset_size,
        test_dataset_size,
        split_with_dirichlet,
        dirichlet_param=0.5,
    ):
        self.num_centers = num_centers
        self.dataset_size = dataset_size
        self.split_with_dirichlet = split_with_dirichlet
        self.dirichlet_param = dirichlet_param
        self.test_dataset_size = test_dataset_size
        (
            self.list_dataset,
            self.dataset_test,
            self.list_data_all_centers,
            self.list_data_per_center,
        ) = self.generate_data()

    @abstractmethod
    def get_pooled_dataset(self):
        pass

    def generate_data(self):
        pooled_dataset = self.get_pooled_dataset()
        list_dataset = []
        list_data_all_centers = []
        list_data_per_center = [[] for _ in range(self.num_centers)]
        shuffled_indices = np.arange(len(pooled_dataset))
        np.random.shuffle(shuffled_indices)
        shuffle_pooled_dataset = Subset(pooled_dataset, shuffled_indices)
        dataset_test = TorchDataSetWithIndices(
            Subset(shuffle_pooled_dataset, np.arange(self.test_dataset_size))
        )
        shuffle_pooled_dataset = Subset(
            shuffle_pooled_dataset,
            np.arange(self.test_dataset_size, len(shuffle_pooled_dataset)),
        )
        len_per_dataset = len(shuffle_pooled_dataset) // self.num_centers
        for center in range(self.num_centers):
            indices = np.arange(
                center * len_per_dataset, (center + 1) * len_per_dataset
            )
            dataset = Subset(shuffle_pooled_dataset, indices)

            if self.split_with_dirichlet:
                list_final_indices = []
                list_labels = np.array([y for (_, y) in dataset])
                proba_new_centers = np.random.dirichlet(
                    self.dirichlet_param * np.ones(self.num_labels),
                )
                list_final_labels = np.random.choice(
                    self.num_labels,
                    p=proba_new_centers,
                    size=self.dataset_size,
                )
                for label in range(self.num_labels):
                    num_sample = np.count_nonzero(list_final_labels == label)
                    list_final_indices += list(
                        np.argwhere(list_labels == label).ravel()[:num_sample]
                    )
                np.random.shuffle(list_final_indices)
                assert len(list_final_indices) == self.dataset_size
            else:
                list_final_indices = np.arange(self.dataset_size)
            list_dataset.append(
                TorchDataSetWithIndices(Subset(dataset, list_final_indices))
            )

        for center in range(self.num_centers):
            for sample in list_dataset[center]:
                sample_numpy = sample[1][0].detach().cpu().numpy().ravel()
                list_data_per_center[center].append(sample_numpy)
                list_data_all_centers.append(sample_numpy)

        return list_dataset, dataset_test, list_data_all_centers, list_data_per_center

    def get_data(
        self,
        batch_size,
    ):
        list_dataloader = []
        list_iterator = []

        for center in range(self.num_centers):
            list_dataloader.append(
                DataLoader(
                    self.list_dataset[center],
                    batch_size=batch_size,
                )
            )
            list_iterator.append(iter(list_dataloader[center]))

        loader_test = DataLoader(
            self.dataset_test,
            batch_size=batch_size,
        )

        return (
            list_dataloader,
            loader_test,
            list_iterator,
            self.list_dataset,
            self.dataset_test,
        )

    @abstractmethod
    def compare_candidate(self, candidate, list_bank):
        pass

    @abstractmethod
    def project_candidate(self, candidate):
        pass

    @abstractmethod
    def oracle(self, candidate):
        pass
