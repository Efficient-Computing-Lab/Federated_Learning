from flwr.client import Client, ClientApp
from flwr.common import Context

from src.flowerClient import FlowerClient
from src.models import MODELS
from src.settings import settings
from src.task import load_data


# Construct a FlowerClient with its own data set partition.
def get_client_fn():
    """
    The VirtualClientEngine will execute this function whenever a client is sampled by
    the strategy to participate.
    """
    model_name = settings.model.name

    def client_fn(context: Context) -> Client:
        # Load model and data
        if model_name not in MODELS:
            raise ValueError(f"Invalid model name: {model_name}")
        model_config = MODELS[model_name]
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        train_loader, val_loader = load_data(model_config, partition_id, num_partitions)
        # Set client type and state
        client_state = context.state
        if partition_id < num_partitions * settings.attack.fraction_malicious_clients:
            client_type = "Malicious"
        else:
            client_type = "Honest"
        client_instance = FlowerClient(model_config, client_type, client_state, train_loader, val_loader).to_client()
        return client_instance

    client = ClientApp(client_fn=client_fn)
    return client
