import torch
from torch.utils.data import Dataset


class TorchDataSetWithIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, item):
        return item, self.dataset[item]

    def __len__(self):
        return len(self.dataset)


class TorchDatasetFromNumpy(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return self.X.shape[0]
