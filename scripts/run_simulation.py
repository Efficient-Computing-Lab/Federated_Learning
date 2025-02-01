import time

from flwr.simulation import run_simulation
from loguru import logger

from src.client_app import get_client_fn
from src.server_app import get_server_fn
from src.settings import settings


def simulate() -> None:
    try:
        start_time = time.time()
        # Start simulation
        run_simulation(
            server_app=get_server_fn(),
            client_app=get_client_fn(),
            num_supernodes=settings.client.num_clients,
            backend_config=settings.backend.__dict__,
        )
        end_time = time.time()
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time.__round__(2)} sec")

    except Exception as e:
        logger.error(f"Error in {settings.model.name} Federated Scenario, processing: {str(e)}")
        raise
