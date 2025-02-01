import json
from logging import INFO, WARNING
from typing import Optional, Union

import torch
from datasets import load_dataset
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate, aggregate_inplace
from torch.utils.data import DataLoader

import wandb
from src.models import ModelConfig, get_model_transforms, set_weights
from src.settings import settings
from src.task import create_run_dir, test

PROJECT_NAME = "FLOWER-attacks-defences"


class CustomFedAvg(FedAvg):
    """A class that behaves like FedAvg but has extra functionality.

    This strategy:
    (1) saves results to the filesystem,
    (2) saves a checkpoint of the global model when a new best is found,
    (3) logs results to W&B if enabled.
    """

    def __init__(self, model_config: ModelConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Create a directory where to save results from this run
        self.save_path, self.run_dir = create_run_dir()
        self.model_config = model_config
        self.model = model_config.model
        # Extract test loader based on configuration server_dataset_percentage
        global_test_set = load_dataset(model_config.dataset)["test"]
        global_test_subset_size = int(len(global_test_set) * settings.defence.server_dataset_percentage)
        global_test_subset = global_test_set.shuffle(seed=settings.general.random_seed).select(
            range(global_test_subset_size)
        )
        self.test_loader = DataLoader(
            global_test_subset.with_transform(get_model_transforms(model_config, "test")),
            batch_size=settings.server.batch_size,
        )
        # Initialise W&B if set
        if settings.general.use_wandb:
            self._init_wandb_project()

        # Keep track of best acc
        self.best_acc_so_far = 0.0

        # A dictionary to store results as they come
        self.results = {}

    def _init_wandb_project(self):
        if settings.attack.type is not None:
            wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp-{settings.attack.type}")
        else:
            wandb.init(project=PROJECT_NAME, name=f"{str(self.run_dir)}-ServerApp-No attack")

    def _store_results(self, tag: str, results_dict) -> None:
        """Store results in dictionary, then save as JSON."""
        # Update results dict
        if tag in self.results:
            self.results[tag].append(results_dict)
        else:
            self.results[tag] = [results_dict]

        # Save results to disk.
        # Note we overwrite the same file with each call to this function.
        # While this works, a more sophisticated approach is preferred
        # in situations where the contents to be saved are larger.
        with open(f"{self.save_path}/results.json", "w", encoding="utf-8") as fp:
            json.dump(self.results, fp)

    def _update_best_acc(self, server_round: int, accuracy, parameters: Parameters) -> None:
        """
        Determines if a new best global model has been found.
        If so, the model checkpoint is saved to disk.
        :param server_round: current server round.
        :param accuracy: the accuracy of the global model.
        """
        if accuracy > self.best_acc_so_far:
            self.best_acc_so_far = accuracy
            log(INFO, "💡 New best global model found: %f", accuracy)
            # You could save the parameters object directly.
            # Instead we are going to apply them to a PyTorch
            # model and save the state dict.
            # Converts flwr.common.Parameters to ndarrays
            ndarrays = parameters_to_ndarrays(parameters)
            model = self.model
            set_weights(model, ndarrays)
            # Save the PyTorch model
            file_name = f"model_state_acc_{accuracy}_round_{server_round}.pth"
            torch.save(model.state_dict(), self.save_path / file_name)

    def store_results_and_log(self, server_round: int, tag: str, results_dict) -> None:
        """A helper method that stores results and logs them to W&B if enabled."""
        # Store results
        self._store_results(tag=tag, results_dict={"round": server_round, **results_dict})

        if settings.general.use_wandb:
            # Log centralized loss and metrics to W&B
            wandb.log(results_dict, step=server_round)

    def evaluate(self, server_round: int, parameters: Parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        loss, metrics = super().evaluate(server_round, parameters)

        # Save model if new best central accuracy is found
        self._update_best_acc(server_round, metrics["centralized_accuracy"], parameters)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="centralized_evaluate",
            results_dict={"centralized_loss": loss, **metrics},
        )
        return loss, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate results from federated evaluation."""
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        # Store and log
        self.store_results_and_log(
            server_round=server_round,
            tag="federated_evaluate",
            results_dict={"federated_evaluate_loss": loss, **metrics},
        )
        return loss, metrics

    def test_on_server_model(self, server_model, parameters: Parameters) -> float:
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        server_model.eval()
        set_weights(server_model, parameters_ndarrays)
        loss, _ = test(server_model, self.test_loader, device=settings.server.device, model_config=self.model_config)
        return loss

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results and failures:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        if self.inplace:
            # Does in-place weighted average of results
            if settings.defence.activation_round != 0 and server_round >= settings.defence.activation_round:
                server_model = self.model_config.model
                updated_results = []
                for client_proxy, fit_res in results:
                    parameters = fit_res.parameters
                    loss = self.test_on_server_model(server_model, parameters)
                    updated_results.append((loss, client_proxy, fit_res))
                updated_results = sorted(updated_results, key=lambda x: x[0])  # Sort by loss
                updated_results = updated_results[: settings.defence.k]  # Select the k items with the smallest loss
                updated_results = [updated_result[1:] for updated_result in updated_results]  # Remove loss value
                aggregated_ndarrays = aggregate_inplace(updated_results)
            else:
                aggregated_ndarrays = aggregate_inplace(results)
        else:
            if settings.defence.activation_round != 0 and server_round >= settings.defence.activation_round:
                updated_weights_results = []
                for _, fit_res in results:
                    parameters = fit_res.parameters
                    num_examples = fit_res.num_examples
                    loss = self.test_on_server_model(parameters)
                    updated_weights_results.append((loss, parameters, num_examples))
                updated_weights_results = sorted(updated_weights_results, key=lambda x: x[0])  # Sort by loss
                updated_weights_results = updated_weights_results[
                    : settings.defence.k
                ]  # Select the k items with the smallest loss
                updated_weights_results = [
                    updated_weights_result[1:] for updated_weights_result in updated_weights_results
                ]  # Remove loss value
                aggregated_ndarrays = aggregate(updated_weights_results)
            else:
                # Convert results
                weights_results = [
                    (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples) for _, fit_res in results
                ]
                aggregated_ndarrays = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated
