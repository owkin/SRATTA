import os
import tempfile

import numpy as np
from joblib import Parallel, delayed
from loguru import logger

import mlflow
from sratta.attack_pipeline.attack_steps.build_relationships import build_relationships
from sratta.attack_pipeline.attack_steps.building_graph import exploit_relationship
from sratta.attack_pipeline.attack_steps.construction_overbar_A import (
    construction_overbar_A,
)
from sratta.attack_pipeline.attack_steps.FL_training import perform_fl_trainings
from sratta.attack_pipeline.attack_steps.sample_matching import run_sample_matching
from sratta.attack_pipeline.attack_steps.sample_recovery import recover_samples
from sratta.datasets.generate_dataset import generate_dataset
from sratta.utils.check_results import check_result
from sratta.utils.make_cluster_using_kmeans import clusters_samples_with_kmeans
from sratta.utils.plot_and_log_result import print_and_log_result
from sratta.utils.post_process_mlflow_metrics import post_process_mlflow_metrics
from sratta.utils.set_determinism import set_determinism


def run_attacks(config, run_id):
    rng_keys = np.random.default_rng(seed=config.parameters.seed).integers(
        low=0, high=10000000, size=config.parameters.num_exp_repeat
    )
    if config.parameters.n_jobs > 1:
        Parallel(n_jobs=config.parameters.n_jobs)(
            delayed(run_and_log_attack)(config, idx_exp_repeat, rng_key, run_id)
            for idx_exp_repeat, rng_key in enumerate(rng_keys)
        )
    else:
        for idx_exp_repeat, rng_key in enumerate(rng_keys):
            run_and_log_attack(config, idx_exp_repeat, rng_key, run_id)
    post_process_mlflow_metrics(config)


def run_and_log_attack(config, idx_exp_repeat, rng_key, run_id):
    set_determinism(rng_key)
    mlflow.set_tracking_uri(f"file:{config.experiment.save_dir}")
    mlflow.set_experiment(config.experiment.experiment_name)
    with mlflow.start_run(run_id=run_id, nested=True):
        with tempfile.TemporaryDirectory() as temp_log_file:
            log_file_path = os.path.join(
                temp_log_file, f"log_file_{idx_exp_repeat}.txt"
            )
            logger.add(log_file_path)

            run_attack(config, idx_exp_repeat)

            mlflow.log_artifact(log_file_path)


def run_attack(config, idx_exp_repeat):
    if config.experiment.log_data:
        dict_intermediates_quantities = {}

    # STEP 0: Generate the Dataset
    dataset = generate_dataset(config)

    with tempfile.TemporaryDirectory(
        # dir=Path(config.experiment.temp_directory_location).resolve()
    ) as temp_dir:

        # STEP 1: Perform FL trainings

        model = perform_fl_trainings(dataset, config, idx_exp_repeat, temp_dir)

        # STEP 2: Sample Recovery
        (
            list_recovered_samples,
            list_truth_centers,
            list_idx_recovered_samples,
        ) = recover_samples(
            config,
            dataset,
            idx_exp_repeat,
            temp_dir,
        )

        if config.experiment.log_data:
            dict_intermediates_quantities["list_truth_centers"] = list_truth_centers
            dict_intermediates_quantities[
                "list_idx_recovered_samples"
            ] = list_idx_recovered_samples

        if config.parameters.use_kmeans_for_clustering:
            clustered_centers = clusters_samples_with_kmeans(
                list_recovered_samples, config
            )

        elif len(list_recovered_samples) > 0:
            # STEP 3: Sample Matching
            (
                list_samples_active_per_neurons_per_round_per_training,
                found_all_samples_per_round_per_training,
                list_sample_used_during_round_per_round_per_training,
            ) = run_sample_matching(
                config,
                list_recovered_samples,
                dataset,
                temp_dir,
            )
            if config.experiment.log_data:
                dict_intermediates_quantities[
                    "list_samples_active_per_neurons_per_round_per_training"
                ] = list_samples_active_per_neurons_per_round_per_training
                dict_intermediates_quantities[
                    "found_all_samples_per_round_per_training"
                ] = found_all_samples_per_round_per_training
                dict_intermediates_quantities[
                    "list_sample_used_during_round_per_round_per_training"
                ] = list_sample_used_during_round_per_round_per_training

            # Step 4: Construction of \overbar A
            sample_activations_before_the_round_per_round_per_training = (
                construction_overbar_A(
                    config,
                    model,
                    list_recovered_samples,
                    list_idx_recovered_samples,
                    list_sample_used_during_round_per_round_per_training,
                    temp_dir,
                )
            )
            if config.experiment.log_data:
                dict_intermediates_quantities[
                    "sample_activations_before_the_round_per_round_per_training"
                ] = sample_activations_before_the_round_per_round_per_training

            # Step 5: Construction of S_1, S_2, S_3
            list_relationship = build_relationships(
                config,
                found_all_samples_per_round_per_training,
                list_recovered_samples,
                sample_activations_before_the_round_per_round_per_training,
                list_sample_used_during_round_per_round_per_training,
                list_samples_active_per_neurons_per_round_per_training,
            )

            # Step 6: Apply theorems on S_1, S_2, S_3
            clustered_centers = exploit_relationship(
                list_relationship,
                list_truth_centers,
                list_idx_recovered_samples,
            )

            if config.experiment.log_data:
                dict_intermediates_quantities["list_relationship"] = list_relationship

            # Step 7: Check result
            check_result(clustered_centers, list_truth_centers)
            if config.experiment.log_data:
                dict_intermediates_quantities["clustered_centers"] = clustered_centers

        else:  # no sample recovered, probably because a defense was performed.
            logger.info("No sample has been recovered.")
            clustered_centers = []
            if config.experiment.log_data:
                dict_intermediates_quantities[
                    "list_samples_active_per_neurons_per_round_per_training"
                ] = [[] for _ in range(config.parameters.num_trainings)]
                dict_intermediates_quantities[
                    "found_all_samples_per_round_per_training"
                ] = [
                    {idx: False for idx in range(config.parameters.num_hidden_neurons)}
                    for _ in range(config.parameters.num_trainings)
                ]
                dict_intermediates_quantities[
                    "list_sample_used_during_round_per_round_per_training"
                ] = [[] for _ in range(config.parameters.num_trainings)]
                dict_intermediates_quantities[
                    "sample_activations_before_the_round_per_round_per_training"
                ] = [[] for _ in range(config.parameters.num_trainings)]
                dict_intermediates_quantities["clustered_centers"] = clustered_centers

        # Step 8: print result
        print_and_log_result(
            clustered_centers, list_truth_centers, idx_exp_repeat, config
        )

        if config.experiment.log_data:
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_path = os.path.join(
                    tmpdirname, f"dict_intermediates_quantities_{idx_exp_repeat}.npy"
                )
                np.save(temp_path, dict_intermediates_quantities, allow_pickle=True)
                mlflow.log_artifact(temp_path)
