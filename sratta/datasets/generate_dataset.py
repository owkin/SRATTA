from sratta.datasets.cifar10.cifar10 import CIFAR10Dataset
from sratta.datasets.dna.dna import DNADataset
from sratta.datasets.fashion_mnist.fashion_mnist_dataset import FashionMNISTData
from sratta.datasets.mm_tcga_brca.mm_tcga_brca_dataset import TCGAMMData


def generate_dataset(config):
    if config.parameters.dataset == "fashion_mnist":
        dataset = FashionMNISTData(
            config.parameters.num_centers,
            config.parameters.dataset_size,
            config.parameters.test_dataset_size,
            config.parameters.split_with_dirichlet,
            config.parameters.dirichlet_param,
            config.experiment.dataset_folder,
        )
    elif config.parameters.dataset == "cifar10":
        dataset = CIFAR10Dataset(
            config.parameters.num_centers,
            config.parameters.dataset_size,
            config.parameters.test_dataset_size,
            config.parameters.split_with_dirichlet,
            config.parameters.dirichlet_param,
            config.experiment.dataset_folder,
        )
    elif config.parameters.dataset == "dna":
        dataset = DNADataset(
            config.parameters.num_centers,
            config.parameters.dataset_size,
            config.parameters.test_dataset_size,
            config.parameters.split_with_dirichlet,
            config.parameters.dirichlet_param,
            config.experiment.dataset_folder,
        )
    elif config.parameters.dataset == "tcga_mm":
        dataset = TCGAMMData(
            config.parameters.num_centers,
            config.parameters.dataset_size,
            config.parameters.test_dataset_size,
            config.experiment.dataset_folder,
        )
    else:
        raise NotImplementedError
    return dataset
