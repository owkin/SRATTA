import gc
import os
import pickle

import numpy as np
import torch
from tqdm import tqdm


def construction_overbar_A(
    config,
    model,
    list_recovered_samples,
    list_idx_recovered_samples,
    list_sample_used_during_round_per_round_per_training,
    temp_dir,
):
    sample_activations_before_the_round_per_round_per_training = []
    for idx_training in range(config.parameters.num_trainings):
        with open(
            os.path.join(
                temp_dir,
                f"list_of_samples_activating_neuron_per_round_{idx_training}.pkl",
            ),
            "rb",
        ) as file:
            list_of_samples_activating_neuron_per_round = pickle.load(file)
        with open(
            os.path.join(temp_dir, f"list_models_weights_{idx_training}.pkl"), "rb"
        ) as file:
            list_models_weights = pickle.load(file)

        sample_activations_before_the_round_per_round_per_training.append([])
        for round in tqdm(range(config.parameters.num_rounds)):
            for idx, m in enumerate(model.parameters()):
                m.data = torch.clone(list_models_weights[round][idx].data)

            th_list_sample_used_during_rounds = (
                list_of_samples_activating_neuron_per_round[round].values()
            )
            th_list_sample_used_during_rounds = list(
                set(
                    [
                        (item[0], item[2])
                        for sublist in th_list_sample_used_during_rounds
                        for item in sublist
                    ]
                )
            )
            samples_used_mapped = set(
                [
                    list_idx_recovered_samples[k]
                    for k in list_sample_used_during_round_per_round_per_training[
                        idx_training
                    ][round]
                ]
            )
            if len(samples_used_mapped - set(th_list_sample_used_during_rounds)) != 0:
                raise ValueError(
                    f"{samples_used_mapped} not in  {samples_used_mapped}:"
                    f" diff is"
                    f"{samples_used_mapped - set(th_list_sample_used_during_rounds)}"
                )

            sample_activations_before_the_round = np.array(
                [
                    (model.linear_1(torch.from_numpy(s.ravel())).detach().numpy()) > 0.0
                    for s in list_recovered_samples
                ]
            )
            sample_activations_before_the_round_per_round_per_training[
                idx_training
            ].append(sample_activations_before_the_round)
        del list_of_samples_activating_neuron_per_round
        del list_models_weights
        gc.collect()

    return sample_activations_before_the_round_per_round_per_training
