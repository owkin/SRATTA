import gc
import os
import pickle

import lifelines
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

import mlflow
from sratta.utils.FCNet import FCNet
from sratta.utils.hooks import (
    DetectAbsoluteRiskyNeuronHooks,
    DetectLambdasPerGradientStepHook,
)


def perform_fl_trainings(dataset, config, idx_exp_repeat, temp_dir):

    if config.parameters.lr.type == "constant":
        learning_rates = [
            config.parameters.lr.value for _ in range(config.parameters.num_trainings)
        ]
    elif config.parameters.lr.type == "log":
        learning_rates = np.logspace(
            config.parameters.lr.min,
            config.parameters.lr.max,
            config.parameters.num_trainings,
        )
    elif config.parameters.lr.type == "lin":
        learning_rates = np.linspace(
            config.parameters.lr.min,
            config.parameters.lr.max,
            config.parameters.num_trainings,
        )
    else:
        raise NotImplementedError

    for idx_training, lr in zip(range(config.parameters.num_trainings), learning_rates):
        logger.info(f"Training {idx_training}  / {config.parameters.num_trainings}")
        (
            list_updates,
            model,
            list_models_weights,
            list_of_samples_activating_neuron_per_round,
        ) = run_one_FL_training(dataset, lr, config, idx_exp_repeat, idx_training)

        with open(
            os.path.join(temp_dir, f"list_updates_{idx_training}.pkl"), "wb"
        ) as file:
            pickle.dump(list_updates, file)
        with open(
            os.path.join(temp_dir, f"list_models_weights_{idx_training}.pkl"), "wb"
        ) as file:
            pickle.dump(list_models_weights, file)
        with open(
            os.path.join(
                temp_dir,
                f"list_of_samples_activating_neuron_per_round_{idx_training}.pkl",
            ),
            "wb",
        ) as file:
            pickle.dump(list_of_samples_activating_neuron_per_round, file)
        del list_updates
        del list_models_weights
        del list_of_samples_activating_neuron_per_round
        gc.collect()
    return model


def run_one_FL_training(
    dataset,
    learning_rate,
    config,
    idx_exp_repeat,
    idx_training,
):
    input_dim = dataset.input_dim
    output_dim = dataset.output_dim
    num_hidden_neurons = config.parameters.num_hidden_neurons
    criterion = dataset.criterion
    (
        list_dataloader,
        loader_test,
        list_iterator,
        list_dataset,
        dataset_test,
    ) = dataset.get_data(
        config.parameters.batch_size,
    )

    model = FCNet(input_dim, num_hidden_neurons, output_dim)

    if config.parameters.prun_risky_rel_lambda_threshold > 0:
        lambda_hook = DetectLambdasPerGradientStepHook(model)

    if config.parameters.prun_risky_update_threshold > 0:
        detection_hook = DetectAbsoluteRiskyNeuronHooks(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    model.train()

    list_updates = []
    list_models_weights = [[torch.clone(m).detach() for m in model.parameters()]]

    list_of_samples_activating_neuron_per_round = []
    list_metric = []
    for r in range(config.parameters.num_rounds):
        list_neuron_activated = {k: [] for k in range(model.num_hidden_neurons)}

        # metrics evaluation on test set
        model.eval()
        metric = 0
        if dataset.task == "survival":
            list_y_test = []
            list_y_cens = []
            list_y_pred = []
        for idx, (X_test, y_test) in loader_test:
            if dataset.task == "classification":
                y_pred = torch.argmax(model(X_test), dim=1)
                metric += torch.sum(y_pred == y_test).detach().numpy()
            elif dataset.task == "regression":
                y_pred = model(X_test)
                metric += torch.sum((y_pred - y_test) ** 2).detach().numpy()
            elif dataset.task == "survival":
                list_y_test.append(y_test[:, 1].cpu().detach().numpy())
                list_y_cens.append(y_test[:, 0].cpu().detach().numpy())
                list_y_pred.append(model(X_test).cpu().detach().numpy())

        if dataset.task == "survival":
            metric = lifelines.utils.concordance_index(
                np.concatenate(list_y_test),
                -np.concatenate(list_y_pred),
                np.concatenate(list_y_cens),
            )
        else:
            metric = metric / len(dataset_test)
        mlflow.log_metric(
            f"fl_val_metric/{idx_exp_repeat}/{idx_training}", metric, step=r
        )
        list_metric.append(metric)

        # training
        model.train()
        old_weights = [torch.clone(m).detach() for m in model.parameters()]
        list_gradients = []
        for center in range(config.parameters.num_centers):
            if config.parameters.prun_risky_update_threshold > 0:
                samples_per_neuron_this_update = torch.zeros(
                    (model.num_hidden_neurons,), dtype=np.int
                )
            if config.parameters.prun_risky_rel_lambda_threshold > 0:
                lambdas_per_sample = {}
                # keys will be id_sample,
                # and values are sums over the update of
                # lambdas for the sample, for all neurons

            # local training
            for update_idx in range(config.parameters.num_updates):
                try:
                    idx, (X, y) = next(list_iterator[center])
                except StopIteration:
                    list_iterator[center] = iter(list_dataloader[center])
                    idx, (X, y) = next(list_iterator[center])
                idx = idx.cpu().detach().numpy()

                # getting activation ground truth for sample reconstruction/sets construction
                for idx_item in range(len(X)):
                    list_activated_neurons_by_sample = (
                        F.relu(model.linear_1(X[idx_item].reshape(-1))) > 1e-8
                    )
                    list_activated_neurons_by_sample = np.argwhere(
                        list_activated_neurons_by_sample.cpu().detach().numpy()
                    ).ravel()

                    for neuron in list_activated_neurons_by_sample:
                        list_neuron_activated[neuron].append(
                            (center, update_idx, idx[idx_item])
                        )

                # performing local update
                optimizer.zero_grad()
                y_pred = model(X)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                if config.parameters.prun_risky_rel_lambda_threshold > 0:
                    for idx_within_batch, sample_idx in enumerate(idx):
                        current_sample_lambda = lambdas_per_sample.get(
                            sample_idx,
                            torch.zeros(
                                (model.num_hidden_neurons,), dtype=torch.float32
                            ),
                        )
                        lambdas_per_sample[sample_idx] = (
                            current_sample_lambda
                            + lambda_hook.lambdas_per_samples_per_neurons[
                                idx_within_batch
                            ]
                        )

                if config.parameters.prun_risky_update_threshold > 0:
                    samples_per_neuron_this_update += (
                        (detection_hook.activations > 0.0) & (detection_hook.dL_dz != 0)
                    ).sum(axis=0)

            new_weights = [torch.clone(m).detach() for m in model.parameters()]
            center_updates = [
                new_weights[idx] - old_weights[idx] for idx in range(len(old_weights))
            ]

            if config.parameters.prun_risky_rel_lambda_threshold > 0:
                # construct the matrix of lambdas, for each neuron and each sample
                lambdas = torch.stack(tuple(lambdas_per_sample.values()), axis=1)
                # risky neurons are the one for which the relative abs lambda of one sample is above the threshold
                risky_neurons = (
                    (lambdas.abs() / lambdas.abs().sum(axis=1, keepdim=True))
                    > config.parameters.prun_risky_rel_lambda_threshold
                ).any(axis=1)
                # censor these neurons
                center_updates[0][risky_neurons, :] = 0
                # log the number of neurons censored.
                mlflow.log_metric(
                    f"fl_nb_neurons_pruned_on_rel_lambda/{idx_exp_repeat}/{idx_training}",
                    float(risky_neurons.sum().detach().numpy()),
                    step=r,
                )
            if config.parameters.prun_risky_update_threshold > 0:
                risky_neurons = (0 < samples_per_neuron_this_update) & (
                    samples_per_neuron_this_update
                    <= config.parameters.prun_risky_update_threshold
                )

                center_updates[0][risky_neurons, :] = 0
                mlflow.log_metric(
                    f"fl_nb_neurons_pruned_on_update/{idx_exp_repeat}/{idx_training}",
                    float(risky_neurons.sum()),
                    step=r,
                )

            list_gradients.append(center_updates)

            for idx, m in enumerate(model.parameters()):
                m.data = torch.clone(old_weights[idx])

        list_updates.append(
            [
                sum(
                    [
                        list_gradients[k][idx]
                        for k in range(config.parameters.num_centers)
                    ]
                )
                / config.parameters.num_centers
                for idx in range(len(list_gradients[0]))
            ]
        )

        for idx, m in enumerate(model.parameters()):
            m.data = torch.clone(old_weights[idx] + list_updates[-1][idx])

        list_models_weights.append(
            [torch.clone(m).detach() for m in model.parameters()]
        )
        list_of_samples_activating_neuron_per_round.append(list_neuron_activated)

    logger.info(f"metric: {list_metric}")
    if config.parameters.prun_risky_rel_lambda_threshold > 0:
        lambda_hook.close()

    return (
        list_updates,
        model,
        list_models_weights,
        list_of_samples_activating_neuron_per_round,
    )
