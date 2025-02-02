from typing import Dict

from datasets import load_dataset
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from torch.utils.data import DataLoader

from src.models import (
    MODELS,
    ModelConfig,
    get_model_transforms,
    get_weights,
    set_weights,
)
from src.settings import settings
from src.strategy import CustomFedAvg
from src.task import test


def gen_evaluate_fn(model_config: ModelConfig):
    """Generate the function for centralized evaluation."""

    model = model_config.model
    global_test_set = load_dataset(model_config.dataset)["test"]
    test_loader = DataLoader(
        global_test_set.with_transform(get_model_transforms(model_config, "test")),
        batch_size=settings.server.batch_size,
    )

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate global model on centralized test set."""
        set_weights(model, parameters_ndarrays)
        loss, accuracy = test(model, test_loader, device=settings.server.device, model_config=model_config)
        return loss, {"centralized_accuracy": accuracy}

    return evaluate


def on_fit_config(server_round: int):
    """
    Construct `config` that clients receive when running `fit()`
    :param server_round: server round
    """
    # Activate attack on configurable server round
    attack_activated = False
    if settings.attack.activation_round != 0 and server_round >= settings.attack.activation_round:
        attack_activated = True
    lr = settings.model.learning_rate
    return {"attack_activated": attack_activated, "lr": lr}


# Define metric aggregation function
def weighted_average(metrics) -> Dict[str, float]:
    """
    Calculate the federated evaluation accuracy based on the sum of weighted accuracies
    from each client divided by the sum of all examples.
    :param metrics: List of client metrics to calculate average across all clients.
    :return: Dictionary of federated evaluation accuracy
    """
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"federated_evaluate_accuracy": sum(accuracies) / sum(examples)}


def get_server_fn():
    def server_fn(context: Context):
        # Read from config
        model_name = settings.model.name
        if model_name not in MODELS:
            raise ValueError(f"Invalid model name: {model_name}")
        model_config = MODELS[model_name]

        # Initialize model parameters
        ndarrays = get_weights(model_config.model)
        parameters = ndarrays_to_parameters(ndarrays)

        # Define strategy
        strategy = CustomFedAvg(
            model_config=model_config,
            fraction_fit=settings.server.fraction_fit,
            fraction_evaluate=settings.server.fraction_eval,
            initial_parameters=parameters,
            on_fit_config_fn=on_fit_config,
            evaluate_fn=gen_evaluate_fn(model_config),
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        config = ServerConfig(num_rounds=settings.server.num_rounds)
        return ServerAppComponents(strategy=strategy, config=config)

    server = ServerApp(server_fn=server_fn)
    return server
