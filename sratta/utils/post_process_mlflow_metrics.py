import numpy as np

import mlflow


def post_process_mlflow_metrics(config):
    client = mlflow.tracking.MlflowClient()
    run_id = mlflow.active_run().info.run_id
    log_mean_fl_acc(client, run_id, config)
    compute_normalized_v_score(client, run_id, config)
    log_mean_and_std_of_metrics(client, run_id, config)


def log_mean_fl_acc(client, run_id, config):
    list_acc_fl = []
    for idx_exp_repeat in range(config.parameters.num_exp_repeat):
        for idx_training in range(config.parameters.num_trainings):
            list_acc_fl.append(
                sorted(
                    list(
                        client.get_metric_history(
                            run_id, f"fl_val_metric/{idx_exp_repeat}/{idx_training}"
                        )
                    ),
                    key=lambda x: x.key,
                )[-1].value
            )
    mlflow.log_metric("fl_val_metric/mean", np.mean(list_acc_fl))
    mlflow.log_metric("fl_val_metric/std", np.std(list_acc_fl))


def compute_normalized_v_score(client, run_id, config):
    for idx_exp_repeat in range(config.parameters.num_exp_repeat):
        v_score = client.get_metric_history(
            run_id, f"v_measure_score_recovered/{idx_exp_repeat}"
        )[0].value
        frac_sample_recover = client.get_metric_history(
            run_id, f"frac_sample_recover/{idx_exp_repeat}"
        )[0].value

        v_measure_score_recovered_normalized = v_score * frac_sample_recover

        mlflow.log_metric(
            f"v_measure_score_recovered_normalized/{idx_exp_repeat}",
            v_measure_score_recovered_normalized,
        )


def log_mean_and_std_of_metrics(client, run_id, config):
    for metric_name in [
        "frac_sample_recover",
        "mean_biggest_clusters",
        "num_clusters",
        "num_datasamples_linked",
        "num_sample_recover",
        "ratio_datasamples_linked",
        "v_measure_score_all",
        "v_measure_score_recovered",
        "v_measure_score_recovered_normalized",
    ]:
        list_metric_values = []
        for idx_exp_repeat in range(config.parameters.num_exp_repeat):
            list_metric_values.append(
                client.get_metric_history(run_id, f"{metric_name}/{idx_exp_repeat}")[
                    0
                ].value
            )

        mlflow.log_metric(f"{metric_name}/mean", np.mean(list_metric_values))
        mlflow.log_metric(f"{metric_name}/std", np.std(list_metric_values))
