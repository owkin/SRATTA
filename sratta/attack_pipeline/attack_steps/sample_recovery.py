import gc
import os
import pickle

import numpy as np
from loguru import logger
from tqdm import tqdm

import mlflow


def recover_samples(
    config,
    dataset,
    idx_exp_repeat,
    temp_dir,
):
    list_recovered_samples, list_truth_centers, list_idx_recovered_samples = [], [], []
    for idx_training in range(config.parameters.num_trainings):
        logger.info(
            f"Sample recovery from {idx_training}  / {config.parameters.num_trainings}"
        )
        # We analytically reconstruct samples from the FL updates
        (
            list_recovered_samples,
            list_truth_centers,
            list_idx_recovered_samples,
        ) = recover_samples_for_one_training(
            config,
            list_recovered_samples,
            list_idx_recovered_samples,
            list_truth_centers,
            dataset,
            idx_training,
            temp_dir,
        )

    mlflow.log_metric(
        f"num_sample_recover/{idx_exp_repeat}", len(list_recovered_samples)
    )
    fraction_sample_recover = len(list_recovered_samples) / (
        config.parameters.dataset_size * config.parameters.num_centers
    )
    mlflow.log_metric(f"frac_sample_recover/{idx_exp_repeat}", fraction_sample_recover)
    return (list_recovered_samples, list_truth_centers, list_idx_recovered_samples)


def recover_samples_for_one_training(
    config,
    list_recovered_samples,
    list_idx_recovered_samples,
    list_truth_centers,
    dataset,
    idx_training,
    temp_dir,
):
    with open(os.path.join(temp_dir, f"list_updates_{idx_training}.pkl"), "rb") as file:
        list_updates = pickle.load(file)
        with open(
            os.path.join(
                temp_dir,
                f"list_of_samples_activating_neuron_per_round_{idx_training}.pkl",
            ),
            "rb",
        ) as file:
            list_of_samples_activating_neuron_per_round = pickle.load(file)
    list_samples_found_by_center = {k: [] for k in range(config.parameters.num_centers)}
    for r in tqdm(range(config.parameters.num_rounds)):
        gradient_w = list_updates[r][0].numpy()
        gradient_bias = list_updates[r][1].numpy()

        for idx in range(config.parameters.num_hidden_neurons):
            if np.abs(gradient_bias[idx]) > 1e-5:
                if np.max(np.abs(gradient_w[idx])) > 1e-5:
                    candidate = gradient_w[idx] / gradient_bias[idx]
                    if dataset.oracle(candidate):
                        if np.any(
                            dataset.compare_candidate(
                                candidate,
                                list_recovered_samples,
                            )
                        ):
                            continue
                        is_candidate_matched = False
                        for center in range(dataset.num_centers):
                            if np.any(
                                dataset.compare_candidate(
                                    candidate,
                                    dataset.list_data_per_center[center],
                                )
                            ):
                                is_candidate_matched = True
                                list_recovered_samples.append(
                                    dataset.project_candidate(candidate)
                                )
                                list_truth_centers.append(center)
                                if (
                                    len(
                                        list_of_samples_activating_neuron_per_round[r][
                                            idx
                                        ]
                                    )
                                    != 1
                                ):
                                    logger.info(
                                        " Warning, Recovered exan is not an Exan"
                                    )
                                idx_sample = np.argwhere(
                                    dataset.compare_candidate(
                                        candidate,
                                        dataset.list_data_per_center[center],
                                    )
                                ).ravel()[0]
                                if idx_sample not in [
                                    k[2]
                                    for k in list_of_samples_activating_neuron_per_round[
                                        r
                                    ][
                                        idx
                                    ]
                                ]:
                                    raise ValueError
                                list_samples_found_by_center[center].append(idx_sample)
                                list_idx_recovered_samples.append((center, idx_sample))
                                break
                        if not is_candidate_matched:
                            raise ValueError(
                                "Candidate passed the oracle but is not"
                                "part of any dataset"
                            )

        logger.info(f"Rounds {r}: {len(list_recovered_samples)} samples recovered")

    logger.info(f"We recovered {len(list_recovered_samples)} samples")
    for center in range(dataset.num_centers):
        logger.info(f"Samples found from center {center}")
        logger.info(np.sort(list_samples_found_by_center[center]))

    del list_updates
    del list_of_samples_activating_neuron_per_round
    gc.collect()
    return list_recovered_samples, list_truth_centers, list_idx_recovered_samples
