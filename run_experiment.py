import argparse

from omegaconf import DictConfig, ListConfig, OmegaConf

import mlflow
from sratta.attack_pipeline.run_attacks import run_attacks

DEFAULT_CONFIG = "configurations/dna_def_lr.yaml"

CONFIG_NAME_DICT = {
    "experiment": [
        "dataset_folder",
        "log_data",
        "experiment_name",
        "run_name",
        "save_dir",
        "temp_directory_location",
    ],
    "parameters": [
        "batch_size",
        "num_updates",
        "num_rounds",
        "dataset_size",
        "test_dataset_size",
        "max_sample",
        "num_trainings",
        "num_centers",
        "lr_type",
        "lr_value",
        "lr_max",
        "lr_min",
        "abs_detection_treshold",
        "rel_detection_treshold",
        "num_hidden_neurons",
        "dataset",
        "num_exp_repeat",
        "seed",
        "n_jobs",
        "prun_risky_rel_lambda_threshold",
        "prun_risky_update_threshold",
        "use_kmeans_for_clustering",
        "split_with_dirichlet",
        "dirichlet_param",
    ],
}


def run_experiment(config):
    mlflow.set_tracking_uri(f"file:{config.experiment.save_dir}")
    mlflow.set_experiment(config.experiment.experiment_name)
    with mlflow.start_run(run_name=config.experiment.run_name) as run:
        log_params_from_omegaconf_dict(config)

        run_attacks(config, run.info.run_id)


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlflow.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f"{parent_name}.{i}", v)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # main arg, config file
    parser.add_argument(
        "--config",
        help="config file path",
        type=str,
        default=DEFAULT_CONFIG,
    )

    # parameters arg, will erase the ones in config if provided
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_updates", type=int)
    parser.add_argument("--num_rounds", type=int)
    parser.add_argument("--dataset_size", type=int)
    parser.add_argument("--test_dataset_size", type=int)
    parser.add_argument("--max_sample", type=int)
    parser.add_argument("--num_trainings", type=int)
    parser.add_argument("--num_centers", type=int)

    parser.add_argument("--abs_detection_treshold", type=float)
    parser.add_argument("--rel_detection_treshold", type=float)
    parser.add_argument("--num_hidden_neurons", type=int)  # 28*28 = 784
    parser.add_argument(
        "--dataset", type=str, choices=["cifar10", "dna", "fashion_mnist"]
    )
    parser.add_argument("--num_exp_repeat", type=int)
    parser.add_argument("--n_jobs", type=int)
    parser.add_argument("--prun_risky_rel_lambda_threshold", type=float)
    parser.add_argument("--prun_risky_update_threshold", type=int)
    parser.add_argument("--use_kmeans_for_clustering", type=bool)
    parser.add_argument("--split_with_dirichlet", type=bool)
    parser.add_argument("--dirichlet_param", type=float)
    parser.add_argument("--seed", type=int)

    parser.add_argument("--lr_type", type=str, choices=["log", "lin", "constant"])
    parser.add_argument("--lr_value", type=float)
    parser.add_argument("--lr_max", type=int)
    parser.add_argument("--lr_min", type=int)

    # experiment arg, will erase the ones in config if provided
    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--temp_directory_location", type=str)
    parser.add_argument("--log_data", type=bool)
    parser.add_argument("--experiment_name", type=str)
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--save_dir", type=str)

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    for key in args.__dict__.keys():
        if args.__dict__[key] is not None:
            if key in CONFIG_NAME_DICT["experiment"]:
                config.experiment[key] = args.__dict__[key]
            elif key in CONFIG_NAME_DICT["parameters"]:
                if key.startswith("lr"):
                    config.parameters.lr[key.split("_")[1]] = args.__dict__[key]
                else:
                    config.parameters[key] = args.__dict__[key]

    run_experiment(config)
