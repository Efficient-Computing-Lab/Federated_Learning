import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner
from torch.utils.data import DataLoader

from src.models import ModelConfig, get_model_transforms
from src.settings import settings


def train(
    model, train_loader, client_type: str, lr: float, device, model_config: ModelConfig, attack_activated: bool
) -> float:
    """
    Train the model on the training set.
    :param model: Model for training.
    :param train_loader: DataLoader for training.
    :param client_type: Type of client
    :param lr: Learning rate.
    :param device: Device for evaluation.
    :param model_config: Model configuration.
    :param attack_activated: Defines if attack is activated.
    :return: Train loss.
    """
    model.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(settings.client.local_epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch[model_config.target_image]
            labels = batch[model_config.target_label]
            if attack_activated and client_type == "Malicious" and settings.attack.type == "Label Flip":
                try:
                    labels = flip_labels(labels, settings.attack.num_labels_to_flip, model_config.num_classes)
                except KeyError:
                    raise KeyError("'num_labels_to_flip' must be specified in config file.")
            optimizer.zero_grad()
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()

            if attack_activated and client_type == "Malicious" and settings.attack.type == "Byzantine Attack":
                # Randomize gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad = param.grad * torch.randn_like(param.grad)  # Randomize gradients

            optimizer.step()
            running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss


def test(model: nn.Module, test_loader: DataLoader, device: str, model_config: ModelConfig) -> Tuple[float, float]:
    """
    Validate the model on the test set.
    :param model: Model for evaluation.
    :param test_loader: DataLoader for test set.
    :param device: Device for evaluation.
    :param model_config: Model configuration.
    :return: Testing loss, accuracy.
    """
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    loss, total, correct = 0.0, 0.0, 0.0
    all_labels = []
    all_predicted = []
    with torch.no_grad():
        for batch in test_loader:
            images = batch[model_config.target_image].to(device)
            labels = batch[model_config.target_label].to(device)

            # Inference
            outputs = model(images)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

            # Append the batch results
            all_labels.extend(labels.cpu().view_as(pred_labels).numpy())
            all_predicted.extend(pred_labels.cpu().numpy())

    accuracy = correct / total * 100
    loss = loss / len(test_loader)
    return loss, accuracy


def load_data(model_config: ModelConfig, partition_id: int, num_partitions: int) -> Tuple[DataLoader, DataLoader]:
    """
    Load partition data.
    :param model_config: model configuration
    :param partition_id: partition id
    :param num_partitions: Total Partitions
    :return: Train and test dataloaders
    """
    match settings.client.partitioner:
        case "IidPartitioner":
            partitioner = IidPartitioner(num_partitions=num_partitions)
        case "DirichletPartitioner":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions,
                partition_by=model_config.target_label,
                alpha=1.0,
                seed=settings.general.random_seed,
            )
        case _:
            raise ValueError(f"Invalid data partition type: {settings.client.partitioner}")
    fds = FederatedDataset(
        dataset=model_config.dataset,
        partitioners={"train": partitioner},
    )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=settings.general.random_seed)

    train_partition = partition_train_test["train"].with_transform(get_model_transforms(model_config, "train"))
    test_partition = partition_train_test["test"].with_transform(get_model_transforms(model_config, "test"))
    batch_size = settings.client.batch_size
    train_loader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_partition, batch_size=batch_size)
    return train_loader, test_loader


def create_run_dir() -> tuple[Path, str]:
    """Create a directory where to save results from this run."""
    # Create output directory given current timestamp
    current_time = datetime.now()
    run_dir = current_time.strftime("%Y-%m-%d/%H-%M-%S")
    # Save path is based on the current directory
    save_path = Path.cwd() / f"outputs/{run_dir}"
    save_path.mkdir(parents=True, exist_ok=False)
    shutil.copy(settings.config_path, save_path)

    return save_path, run_dir


def static_flip(model_labels: int, labels_to_flip: int, seed: int) -> dict[int, int]:
    """
    Deterministically swaps 'labels_to_flip' numbers out of 'model_labels'.
    :param model_labels: Number of model labels
    :param labels_to_flip: Number of labels to flip
    :param seed: random seed
    :return: A dictionary mapping old labels to new labels.
    """
    distinct_values = list(range(model_labels))
    if labels_to_flip > len(distinct_values):
        raise ValueError("Flipping labels cannot be greater than the number of model labels.")
    # Set the random seed for reproducibility
    random.seed(seed)
    # Select unique indices equal to 'num_to_shuffle' for swapping
    indices = random.sample(range(len(distinct_values)), labels_to_flip)
    # Shuffle the selected indices
    shuffled_indices = indices[:]
    random.shuffle(shuffled_indices)
    # Create mapping (old value → new value)
    mapping = {distinct_values[indices[i]]: distinct_values[shuffled_indices[i]] for i in range(labels_to_flip)}
    return mapping


def flip_labels(labels: torch.tensor, labels_to_flip: int, model_labels: int) -> torch.tensor:
    """
    Flips the labels from a train batch.
    :param labels: Labels from a train batch
    :param labels_to_flip: Number of labels to flip
    :param model_labels: Number of model labels
    :return: Flipped labels tensor
    """
    shuffle_dict = static_flip(model_labels, labels_to_flip, seed=settings.general.random_seed)
    flipped_tensor = torch.tensor([shuffle_dict.get(item, item) for item in labels.tolist()])
    return flipped_tensor
