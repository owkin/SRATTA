import numpy as np
from loguru import logger
from sklearn.metrics.cluster import v_measure_score

import mlflow


def print_and_log_result(clustered_centers, list_truth_centers, idx_exp_repeat, config):
    for idx_c, c in enumerate(clustered_centers):
        logger.info(f"Cluster {idx_c} ({len(c)}): {c}")

    list_len_clusters = [len(c) for c in clustered_centers if len(c) > 1]
    list_len_clusters = sorted(list_len_clusters)
    sum_len_clusters = np.sum(list_len_clusters)

    v_measure_recovered, v_measure_all = compute_v_measure_score(
        clustered_centers, list_truth_centers, config
    )
    logger.info(f"V measure score recovered: {v_measure_recovered}")
    logger.info(f"V measure score all: {v_measure_all}")
    mlflow.log_metric(
        f"v_measure_score_recovered/{idx_exp_repeat}", v_measure_recovered
    )
    mlflow.log_metric(f"v_measure_score_all/{idx_exp_repeat}", v_measure_all)

    ratio_datasamples_clustered = sum_len_clusters / (
        config.parameters.dataset_size * config.parameters.num_centers
    )

    mlflow.log_metric(f"num_clusters/{idx_exp_repeat}", len(clustered_centers))
    mlflow.log_metric(f"num_datasamples_linked/{idx_exp_repeat}", sum_len_clusters)
    mlflow.log_metric(
        f"ratio_datasamples_linked/{idx_exp_repeat}", ratio_datasamples_clustered
    )

    mean_biggest_clusters = np.mean(list_len_clusters[-config.parameters.num_centers :])
    mlflow.log_metric(f"mean_biggest_clusters/{idx_exp_repeat}", mean_biggest_clusters)
    for idx, len_cluster in enumerate(list_len_clusters):
        mlflow.log_metric(f"list_len_clusters/{idx_exp_repeat}", len_cluster, step=idx)


def compute_v_measure_score(clustered_centers, list_truth_centers, config):
    list_y_pred_recovered = []
    list_y_true_recovered = []

    list_y_pred_not_recovered = []
    list_y_true_not_recovered = []
    for idx_cluster, cluster in enumerate(clustered_centers):
        for sample in cluster:
            list_y_pred_recovered.append(idx_cluster)
            list_y_true_recovered.append(list_truth_centers[sample])

    idx_cluster = len(clustered_centers) + 1
    for center in range(config.parameters.num_centers):
        nb_recoverd_samples_from_center = np.sum(
            np.array(list_truth_centers) == (center)
        )
        nb_not_recoverd_samples_from_center = (
            config.parameters.dataset_size - nb_recoverd_samples_from_center
        )
        list_y_pred_not_recovered += range(
            idx_cluster, idx_cluster + nb_not_recoverd_samples_from_center
        )
        idx_cluster += nb_not_recoverd_samples_from_center

        list_y_true_not_recovered += [(center)] * nb_not_recoverd_samples_from_center

    v_measure_recovered = v_measure_score(list_y_pred_recovered, list_y_true_recovered)
    v_measure_all = v_measure_score(
        list_y_pred_recovered + list_y_pred_not_recovered,
        list_y_true_recovered + list_y_true_not_recovered,
    )
    return v_measure_recovered, v_measure_all
