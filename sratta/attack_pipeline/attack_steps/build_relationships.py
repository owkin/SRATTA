import ast

from loguru import logger


def build_relationships(
    config,
    found_all_samples_per_round_per_training,
    list_recovered_samples,
    sample_activations_before_the_round_per_round_per_training,
    list_sample_used_during_round_per_round_per_training,
    list_samples_active_per_neurons_per_round_per_training,
):
    """
    Mapping between the tex notation and the code:
    S_1 = list_activation_not_modified
    S_2 = list_new_activation
    S_3 = list_new_deactivation
    """
    list_relationship = []
    for idx_training in range(config.parameters.num_trainings):
        for round in range(config.parameters.num_rounds):
            list_samples_active_per_neurons = (
                list_samples_active_per_neurons_per_round_per_training[idx_training][
                    round
                ]
            )
            sample_activations_before_the_round = (
                sample_activations_before_the_round_per_round_per_training[
                    idx_training
                ][round]
            )
            list_sample_used_during_round = (
                list_sample_used_during_round_per_round_per_training[idx_training][
                    round
                ]
            )
            found_all_samples = found_all_samples_per_round_per_training[idx_training][
                round
            ]
            for idx_neuron in range(config.parameters.num_hidden_neurons):
                if (
                    len(list_samples_active_per_neurons[idx_neuron]) > 1
                    and found_all_samples[idx_neuron]
                ):

                    list_new_activation = []
                    list_new_deactivation = []
                    list_activation_not_modified = []
                    for recovered_sample_idx in range(len(list_recovered_samples)):
                        if (
                            recovered_sample_idx
                            in list_samples_active_per_neurons[idx_neuron]
                            and not sample_activations_before_the_round[
                                recovered_sample_idx
                            ][idx_neuron]
                        ):
                            list_new_activation.append(recovered_sample_idx)
                        if (
                            recovered_sample_idx in list_sample_used_during_round
                            and recovered_sample_idx
                            not in list_samples_active_per_neurons[idx_neuron]
                            and sample_activations_before_the_round[
                                recovered_sample_idx
                            ][idx_neuron]
                        ):
                            list_new_deactivation.append(recovered_sample_idx)
                        if (
                            recovered_sample_idx
                            in list_samples_active_per_neurons[idx_neuron]
                            and sample_activations_before_the_round[
                                recovered_sample_idx
                            ][idx_neuron]
                        ):
                            list_activation_not_modified.append(recovered_sample_idx)

                    list_relationship.append(
                        {
                            "list_new_activation": list_new_activation,  # S_1
                            "list_new_deactivation": list_new_deactivation,  # S_2
                            "list_activation_not_modified": list_activation_not_modified,  # S_3
                        }
                    )
    list_relationship = remove_duplicate(list_relationship)
    return list_relationship


def remove_duplicate(element):
    len_init = len(element)
    str_parsed = sorted(list(set([str(m) for m in element])))
    result = [ast.literal_eval(k) for k in str_parsed]
    logger.info(f"Reduce list_relationship from {len_init} to {len(result)}.")
    return result
