import gc
import os
import pickle
import warnings

import numpy as np
from loguru import logger
from sklearn.linear_model import OrthogonalMatchingPursuit
from tqdm import tqdm


def support_match(list_candidate, gradient):
    return np.argwhere(
        [
            np.all((list_candidate[:, k] != 0) <= (gradient != 0))
            for k in range(list_candidate.shape[1])
        ]
    ).ravel()


def run_sample_matching(
    config,
    list_recovered_samples,
    dataset,
    temp_dir,
):
    list_samples_active_per_neurons_per_round_per_training = []
    found_all_samples_per_round_per_training = []
    list_sample_used_during_round_per_round_per_training = []

    for idx_training in range(config.parameters.num_trainings):
        logger.info(
            f"Sample matching of training {idx_training} / {config.parameters.num_trainings}"
        )
        list_samples_active_per_neurons_per_round_per_training.append([])
        found_all_samples_per_round_per_training.append([])
        list_sample_used_during_round_per_round_per_training.append([])
        with open(
            os.path.join(temp_dir, f"list_updates_{idx_training}.pkl"), "rb"
        ) as file:
            list_updates = pickle.load(file)

        for round in tqdm(range(config.parameters.num_rounds)):
            gradient_w = list_updates[round][0].numpy()
            gradient_b = list_updates[round][1].numpy()

            (
                list_samples_active_per_neurons,
                found_all_samples,
                list_sample_used_during_round,
            ) = run_sample_matching_for_one_training_one_round(
                list_recovered_samples,
                gradient_w,
                gradient_b,
                config.parameters.num_hidden_neurons,
                config.parameters.max_sample,
                config.parameters.abs_detection_treshold,
                config.parameters.rel_detection_treshold,
                dataset,
            )
            list_samples_active_per_neurons_per_round_per_training[idx_training].append(
                list_samples_active_per_neurons
            )
            found_all_samples_per_round_per_training[idx_training].append(
                found_all_samples
            )
            list_sample_used_during_round_per_round_per_training[idx_training].append(
                list_sample_used_during_round
            )
        del list_updates
        gc.collect()

    return (
        list_samples_active_per_neurons_per_round_per_training,
        found_all_samples_per_round_per_training,
        list_sample_used_during_round_per_round_per_training,
    )


def run_sample_matching_for_one_training_one_round(
    list_recovered_samples,
    gradient_w,
    gradient_b,
    num_hidden_neurons,
    max_sample,
    abs_detection_treshold,
    rel_detection_threshold,
    dataset,
):
    """
    Check which samples from list_recovered_samples was used during a training, and
    which neuron it activated.
    Parameters
    ----------
    list_recovered_samples :
    gradient_w :
    num_hidden_neurons :
    max_sample :

    Returns
    -------
    list_samples_active_per_neuron : List[List[int]]
        List of the samples that activated each neurons
    found_all_samples : List[bool]
        List of whether we found all the samples activating a given neuron
    list_sample_used_during_round :
        List of all the samples that we are sure were used during during the round

    """

    warnings.simplefilter(action="ignore", category=FutureWarning)
    warnings.filterwarnings(action="ignore", category=RuntimeWarning)
    list_samples_active_per_neurons = {}
    found_all_samples = {idx: False for idx in range(num_hidden_neurons)}

    if dataset.name == "mock_mm":
        np_list_samples = np.array(
            [x.ravel()[: dataset.binary_input] for x in list_recovered_samples]
        ).transpose()
        gradient_w = gradient_w[:, : dataset.binary_input]
    else:
        np_list_samples = np.array(
            [x.ravel() for x in list_recovered_samples]
        ).transpose()

    for idx in range(num_hidden_neurons):

        list_samples_active_per_neurons[idx] = []
        if np.max(np.abs(gradient_w[idx])) > 1e-6:
            idx_coeff_1 = support_match(np_list_samples, gradient_w[idx])
            if len(idx_coeff_1) > 0:
                omp = OrthogonalMatchingPursuit(
                    normalize=False, n_nonzero_coefs=min(20, len(idx_coeff_1))
                )
                omp.fit(np_list_samples[:, idx_coeff_1], gradient_w[idx])

                coeffs = omp.coef_
                idx_coeffs = idx_coeff_1[np.argwhere(coeffs != 0).ravel()]

                y_on_coeffs, residual, rank, _ = np.linalg.lstsq(
                    np_list_samples[:, idx_coeffs], gradient_w[idx], rcond=1e-5
                )
                y = np.zeros(len(np_list_samples.transpose()))
                y[idx_coeffs] = y_on_coeffs

                if residual < 1e-10:
                    y_max = np.max(np.abs(y))
                    detection_threshold = max(
                        abs_detection_treshold, y_max * rel_detection_threshold
                    )
                    list_samples_active_per_neurons[idx] = np.argsort(np.abs(y))
                    list_samples_active_per_neurons[
                        idx
                    ] = list_samples_active_per_neurons[idx][
                        np.abs(y[list_samples_active_per_neurons[idx]])
                        > detection_threshold
                    ]

                    list_samples_active_per_neurons[
                        idx
                    ] = list_samples_active_per_neurons[idx][:max_sample]
                    reconstruction = np.sum(
                        np.array(
                            [
                                y[idx_r] * np_list_samples[:, idx_r]
                                for idx_r in list_samples_active_per_neurons[idx]
                            ]
                        ),
                        axis=0,
                    )  #
                    if np.allclose(
                        reconstruction,
                        gradient_w[idx],
                        rtol=rel_detection_threshold,
                        atol=0.0,
                    ):
                        if np.allclose(
                            np.sum(y),
                            gradient_b[idx],
                            rtol=rel_detection_threshold,
                            atol=0.0,
                        ):
                            found_all_samples[idx] = True
                        else:
                            list_samples_active_per_neurons[idx] = []
                    else:
                        list_samples_active_per_neurons[idx] = []

    list_sample_used_during_round = list(
        set(
            np.concatenate(list(list_samples_active_per_neurons.values()))
            .ravel()
            .astype(int)
        )
    )

    return (
        list_samples_active_per_neurons,
        found_all_samples,
        list_sample_used_during_round,
    )
