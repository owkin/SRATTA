import numpy as np
from loguru import logger

from sratta.attack_pipeline.attack_steps.graph_connection import (
    connect_components_from_list,
)


def exploit_relationship(
    list_relationship,
    list_truth_centers,
    list_idx_recovered_samples,
):
    list_relationship = np.array(list_relationship)
    list_edges = []
    list_idx_to_keep = []
    for single_node in range(len(list_idx_recovered_samples)):
        list_edges.append([single_node])
    for idx, relationship in enumerate(list_relationship):
        list_activation_not_modified = relationship["list_activation_not_modified"]
        list_new_activation = relationship["list_new_activation"]
        list_new_deactivation = relationship["list_new_deactivation"]
        # We remove the use of S_3 in the part of the attack as it leads to
        # False negative. We don't know why yet
        list_modified_y = list_new_activation  # + list_new_deactivation

        if len(list_activation_not_modified) == 1 and len(list_modified_y) > 0:
            logger.info("------------------")
            edge = np.sort(list_activation_not_modified + list_modified_y)
            list_edges.append(edge)
            logger.info(f"{edge} are from the same centers")
            if len(set([list_truth_centers[c] for c in (edge)])) > 1:
                logger.info("Error in this prediction")
                logger.info(
                    f"True centers are: " f"{[list_truth_centers[c] for c in (edge)]}"
                )
                logger.info(f"list_new_activation: {list_new_activation}")
                logger.info(f"list_new_deactivation: {list_new_deactivation}")
                logger.info(
                    f"list_activation_not_modified:" f"{list_activation_not_modified}"
                )
                raise ValueError
            logger.info([list_idx_recovered_samples[c] for c in edge])
            logger.info(f"list_new_activation: {list_new_activation}")
            logger.info(f"list_new_deactivation: {list_new_deactivation}")
            logger.info(
                f"list_activation_not_modified:" f"{list_activation_not_modified}"
            )
        else:
            list_idx_to_keep.append(idx)

    if len(list_relationship) > 0 and len(list_idx_to_keep) > 0:
        list_relationship = list_relationship[np.array(list_idx_to_keep)]

        should_continue = True
        repeat = 1
        while should_continue:
            list_idx_to_keep = []
            clustered_centers = list(connect_components_from_list(list_edges))
            logger.info(f"Repeat {repeat}")
            should_continue = False
            for idx, relationship in enumerate(list_relationship):
                should_keep_idx = True
                list_activation_not_modified = relationship[
                    "list_activation_not_modified"
                ]
                list_new_activation = relationship["list_new_activation"]
                list_new_deactivation = relationship["list_new_deactivation"]
                # We remove the use of S_3 in the part of the attack as it leads to
                # False negative. We don't know why yet
                list_modified_y = list_new_activation  # + list_new_deactivation

                if len(list_activation_not_modified) > 1 and len(list_modified_y) > 0:
                    for cluster in clustered_centers:
                        delta = set(list_activation_not_modified) - set(cluster)
                        if len(delta) == 0:
                            should_keep_idx = False
                            if len(set(list_modified_y) - set(cluster)) > 0:
                                logger.info(f"------*** {repeat} ***---------")
                                logger.info(f"{edge} are from the same centers")
                                logger.info(
                                    f"list_new_activation: {list_new_activation}"
                                )
                                logger.info(
                                    f"list_new_deactivation: {list_new_deactivation}"
                                )
                                logger.info(
                                    f"list_activation_not_modified:"
                                    f"{list_activation_not_modified}"
                                )

                                should_continue = True
                                edge = np.sort(
                                    list_activation_not_modified + list_modified_y
                                )
                                list_edges.append(edge)
                if should_keep_idx:
                    list_idx_to_keep.append(idx)

            list_relationship = list_relationship[np.array(list_idx_to_keep)]
            if len(list_relationship) == 0:
                should_continue = False
            repeat += 1
    else:
        list_edges = []

    clustered_centers = list(connect_components_from_list(list_edges))
    return clustered_centers
