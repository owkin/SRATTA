from torch.utils.data import Dataset


class DataSetWithIndices(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return item, self.dataset[item]
