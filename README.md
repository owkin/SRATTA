# SRATTA: Sample Re-ATTribution Attack of Secure Aggregation in Federated Learning.

## Table of Contents
- [SRATTA: Sample Re-ATTribution Attack of Secure Aggregation in Federated Learning.](#sratta-sample-re-attribution-attack-of-secure-aggregation-in-federated-learning)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Installation](#installation)
  - [Using this code](#using-this-code)
    - [Quick start](#quick-start)
    - [How to use the run\_experiment.py script](#how-to-use-the-run_experimentpy-script)
    - [Experiments presented in the paper](#experiments-presented-in-the-paper)
  - [Repository overview](#repository-overview)
  - [Citing this work](#citing-this-work)
  - [License](#license)

## Overview

This repository is a python implementation of the paper [SRATTA: Sample Re-ATTribution Attack of Secure Aggregation in Federated Learning.]().

## Installation

First of all you need to clone the repository

```bash
git clone https://github.com/owkin/sratta_code.git
cd sratta_code
```

We recommend using a fresh virtual environment to avoid conflict between dependencies. We used a conda environment with python 3.8.

```bash 
conda create --name sratta python=3.8
conda activate sratta

```

Now that your environment is activated, install our code, and the associated dependencies.

```bash
pip install -e .
```

## Using this code

### Quick start

To run a experiment, run the `run_experiment.py` script. You will need to select a configuration file. Several of them are provided in the [`configuration`](./configurations/) folder.

```bash
python run_experiment.py --config "configurations/fashionMNIST.yaml" --run_name "my_first_run" --num_exp_repeat 1 --save_dir "./mlflow/mlruns"
```

To track the metric of the experiment, we use [mlflow](https://mlflow.org/). To start a `mlflow` server, and watch your results, please open a new terminal, activate the virtual environment, cd
to the directory of the cloned repository and launch the mlflow UI:

```bash
# change path below to match your installation path
cd /path/towards/sratta/repo/sratta_code
conda activate sratta
mlflow ui --backend-store-uri ./mlflow/mlruns
```

And then navigate to http://localhost:5000 in your browser. Note that you need to specify the same path for ` --backend-store-uri` and `--save_dir` argument of run_experiment. 
### How to use the run_experiment.py script

The script `run_experiment.py` allows you to simulate an attack. It accepts several arguments, either directly in the configuration file, or via the command line (in which case the value provided in the configuration file will be ignored.)

- `--batch_size`: int, the number of samples in the batch used for the local updates.
- `--num_updates`: int, the number of local updates done by each center before the aggregation step. 
- `--num_rounds`: int, the number of optimization rounds.
- `--dataset_size`: int, the number of samples in each center's dataset
- `--test_dataset_siz`: int, the number of sample in the dataset used for testing.  
- `--max_sample`: int, max number of best reconstruction candidates added to reconstructed samples at each iteration. 
- `--num_trainings`: int, number of simulated training.
- `--num_centers`: int, number of centers. 
- `--abs_detection_treshold`: float, absolute threshold for selection of reconstruction candidate.
- `--rel_detection_treshold`: float, relative threshold for selection of reconstruction candidate.
- `--num_hidden_neurons`: int, number of neurons in the hidden layer. 
- `--dataset`: `cifar10`, `dna` or `fashion_mnist`, dataset used. 
- `--prun_risky_rel_lambda_threshold`: float, relative defense threshold. `$\beta$` in the paper.
- `--prun_risky_update_threshold`: int, number of minimal sample activating neuron to avoid censoring. `q` in the paper.
- `--use_kmeans_for_clustering`: bool, define if `SRATTA` or `kmean` algorithm is used to group reconstructed samples.
- `--split_with_dirichlet`: bool, if `True`, the dataset is split between sample using a dirichlet distribution. See the paper appendix for more details.
- `--dirichlet_param`: float, dirichlet parameter.
- `--seed`: int, random seed generator, for reproducibility.
- `--lr_type`: `log`, `lin` or `constant`. Learning rate strategy. With constant, learning rate is constant for each training. With `log` / `lin` the learning rates change in log scale/ linearly. 
- `--lr_value`: `float`, with `lr_type` = `constant`, set the value of the learning scale.
- `--lr_max`: int, set the max value of learning rate when using  `log`, `lin` for `lr_type`.
- `--lr_min`: int, set the min value of learning rate when using  `log`, `lin` for `lr_type`.

- `--num_exp_repeat`: int, number of repetition of the simulation, for confident interval.
- `--n_jobs`: int, number of job used. 
- `--dataset_folder`: str, where to store the dataset.
- `--temp_directory_location`: str, used for temporary files.
- `--log_data`: bool, whether to log the reconstructed samples. Come with overhead. 
- `--experiment_name`: str, experiment name. 
- `--run_name`: str, run name. 
- `--save_dir`: str, directory where `mlflow` saves the logs.
### Experiments presented in the paper

If you wish to reproduce the experiment we present in the main paper, run the following scripts. Each script corresponds to one figure/table. Please note that here nothing is parallelized, and running all those experiments will be very long. 

```bash
bash scripts/reproduce_attack_results.sh
bash scripts/reproduce_clustering_baseline_results.sh
bash scripts/reproduce_defense_results.sh
```

## Repository overview

```
.
├── LICENSE.txt
├── README.md
├── configurations
│   ├── defenses
│   │   ├── cifar10_search_lr.yaml
│   │   ├── dna_search_lr.yaml
│   │   └── fashion_mnist_search_lr.yaml
│   ├── cifar10.yaml
│   ├── dna.yaml
│   └── fashionMNIST.yaml
├── run_experiment.py
├── scripts
│   ├── reproduce_attack_results.sh
│   ├── reproduce_clustering_baseline_results.sh
│   └── reproduce_defense_results.sh
├── setup.py
└── sratta
    ├── __init__.py
    ├── attack_pipeline
    │   ├── __init__.py
    │   ├── attack_steps
    │   │   ├── FL_training.py
    │   │   ├── __init__.py
    │   │   ├── build_relationships.py
    │   │   ├── building_graph.py
    │   │   ├── construction_overbar_A.py
    │   │   ├── graph_connection.py
    │   │   ├── sample_matching.py
    │   │   ├── sample_recovery.py
    │   │   └── torch_dataset.py
    │   └── run_attacks.py
    ├── datasets
    │   ├── cifar10
    │   │   └── cifar10.py
    │   ├── dna
    │   │   ├── dna.py
    │   │   └── list_idx.npy
    │   ├── fashion_mnist
    │   │   └── fashion_mnist_dataset.py
    │   ├── generate_dataset.py
    │   └── generic_dataset.py
    └── utils
        ├── check_results.py
        ├── cox_loss.py
        ├── dataset_with_indices.py
        ├── hooks.py
        ├── make_cluster_using_kmeans.py
        ├── plot_and_log_result.py
        ├── post_process_mlflow_metrics.py
        └── set_determinism.py

```

## Citing this work
If our work helps you in your research, consider citing us.

```bibtex
@article{marchant2023sratta,
  title={SRATTA: Sample Re-ATTribution Attack of Secure Aggregation in Federated Learning.},
  author={Marchand, Tanguy and Loeb, Regis and Ogier du Terrail, Jean and Marteau-Ferey, Ulysse and Pignet,  Arthur},
  year={2023},
}
```
## License

This code is released under an [MIT license](./LICENSE.txt).
