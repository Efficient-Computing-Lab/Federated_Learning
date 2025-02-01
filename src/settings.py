import os
from pathlib import Path

from pydantic import BaseModel, ValidationInfo, field_validator
from yaml import safe_load


class Server(BaseModel):
    service_device: str
    fraction_fit: float
    fraction_eval: float
    num_rounds: int
    differential_privacy: bool
    batch_size: int

    @field_validator("fraction_fit", "fraction_eval")
    def validate_percentages(cls, value, info):
        """
        Validate individual percentage fields are between 0.0 and 1.0.
        :param value: Field validator
        :param info: Instance of Server class
        :return: Validated fields are between 0.0 and 1.0 or raises exception.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Under server configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value

    @field_validator("batch_size", "num_rounds")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Server class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under server configuration: {info.field_name} must be positive. Got {value}")
        return value


class Client(BaseModel):
    num_clients: int
    batch_size: int
    local_epochs: int
    differential_privacy: bool
    partitioner: str

    @field_validator("num_clients", "batch_size", "local_epochs")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Client class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under client configuration: {info.field_name} must be positive. Got {value}")
        return value


class Model(BaseModel):
    name: str
    learning_rate: float

    @field_validator("learning_rate")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Model class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under model configuration: {info.field_name} must be positive. Got {value}")
        return value


class Attack(BaseModel):
    activation_round: int = 0
    fraction_malicious_clients: float = 0.0
    type: str = None
    # For label-flip attack parameters
    num_labels_to_flip: int = 0

    @field_validator("fraction_malicious_clients")
    def validate_percentages(cls, value, info):
        """
        Validate individual percentage fields are between 0.0 and 1.0.
        :param value: Field validator
        :param info: Instance of Attack class
        :return: Validated fields are between 0.0 and 1.0 or raises exception.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value

    @field_validator("activation_round", "num_labels_to_flip")
    def validate_positive(cls, value, info):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Attack class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be positive. Got {value}")
        return value

    @field_validator("type")
    def validate_attack_type(cls, value: str, info: ValidationInfo):
        """
        Validate attack type is either Label Flip, Byzantine Attack or no attack at all (None).
        :param value: Field validator
        :param info: Instance of Attack type
        :return: Validated attack type or raises exception.
        """
        attack_types = ["Label Flip", "Byzantine Attack"]
        if value not in attack_types + [None]:
            raise ValueError(f"Under attack configuration: {info.field_name} must be in {attack_types}. Got {value}")
        return value


class Defence(BaseModel):
    activation_round: int = 0
    k: int = 0
    server_dataset_percentage: float = 1.0

    @field_validator("server_dataset_percentage")
    def validate_percentages(cls, value: float, info: ValidationInfo):
        """
        Validate individual percentage fields are between 0.0 and 1.0.
        :param value: Field validator
        :param info: Instance of Defence class
        :return: Validated fields are between 0.0 and 1.0 or raises exception.
        """
        if value < 0.0 or value > 1.0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be between 0.0 and 1.0. Got {value}")
        return value

    @field_validator("activation_round", "k")
    def validate_positive(cls, value: int, info: ValidationInfo):
        """
        Validate individual fields are positive.
        :param value: Field validator
        :param info: Instance of Defence class
        :return: Validated fields are positive values or raises exception.
        """
        if value <= 0:
            raise ValueError(f"Under attack configuration: {info.field_name} must be positive. Got {value}")
        return value


class General(BaseModel):
    use_wandb: bool
    random_seed: int


class Backend(BaseModel):
    client_resources: dict[str, float]


class Config(BaseModel):
    server: Server
    client: Client
    model: Model
    attack: Attack
    defence: Defence
    general: General
    backend: Backend
    config_path: Path

    def __init__(self, config_path: Path) -> None:
        if config_path.is_file():
            with open(config_path) as f:
                config = safe_load(f)
                if "attack" not in config:
                    config.update({"attack": Attack()})
                if "defence" not in config:
                    config.update({"defence": Defence()})
            config.update({"config_path": config_path})
            super().__init__(**config)
        else:
            raise FileNotFoundError("Error: yaml config file not found.")

    @field_validator("defence")
    def validate_defence_clients(cls, value, info):
        """
        Validate that the number of client (Defence attribute) is valid.
        It should not exceed the total number of all FL clients.
        :param value: Instance of Defence class
        :param info: Instance of Config class
        :return: Validated defence number of clients or raise exception
        """
        total_clients = info.data["client"].num_clients
        k = value.k
        if total_clients < k:
            raise ValueError(
                f"Number of k smallest losses cannot be more than "
                f"the total number of clients ({total_clients}). "
                f"Got k={k}"
            )
        return value

    @field_validator("attack", "defence")
    def validate_activation_round(cls, value: Attack | Defence, info: ValidationInfo):
        """
        Validate that the activation round (Attack or Defence attribute) is valid.
        It should not exceed the total number of FL rounds.
        :param value: Instance of Attack or Defence class
        :param info: Instance of Config class
        :return: Validated activation round or raise exception
        """
        num_rounds = info.data["server"].num_rounds
        activation_round = value.activation_round
        if num_rounds < activation_round:
            raise ValueError(
                f"Activation round for '{info.field_name}' cannot exceed total number of FL rounds ({num_rounds}). "
                f"Got activation round={activation_round}"
            )
        return value


FOLDER_DIR = Path(__file__).parent.parent
config_file = FOLDER_DIR / f"configs/{os.getenv('config_file_name', 'config')}.yaml"
settings = Config(config_file)
