import json
import os
from pathlib import Path

import click
import torch
import torch.nn as nn
from datasets import load_dataset
from loguru import logger
from torch.utils.data import DataLoader

from src.models import MODELS, ModelConfig, get_model_transforms


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: str,
    model_config: ModelConfig,
) -> tuple[nn.Module, float]:
    """
    Trains and saves the model
    :param model: Model architecture
    :param train_loader: Training data loader
    :param val_loader: Validation data loader
    :param epochs: Training epochs
    :param lr: Learning rate
    :param device: Training device (cpu or gpu)
    :param model_config: Configuration for model architecture
    :return: A tuple with model architecture and validation accuracy
    """
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            images = batch[model_config.target_image].to(device)
            labels = batch[model_config.target_label].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute average training loss for the epoch
        train_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()  # Set model to evaluation mode
        val_accuracy, val_loss, total, correct = 0.0, 0.0, 0.0, 0.0
        all_labels = []
        all_predicted = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch[model_config.target_image].to(device)
                labels = batch[model_config.target_label].to(device)

                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                val_loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)

                # Append the batch results
                all_labels.extend(labels.cpu().numpy())
                all_predicted.extend(pred_labels.cpu().numpy())

        val_accuracy = correct / total * 100
        val_loss /= len(val_loader)

        # Print progress
        logger.info(
            f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation Accuracy: {val_accuracy:.2f}%"
        )

    return model, val_accuracy


@click.command()
@click.argument("model_name", required=True)
@click.option("--delta", help="Server dataset percentage", default=1.0)
@click.option("--epochs", help="Number of training epochs", default=100)
@click.option("--batch-size", help="Batch size", default=64)
@click.option("--lr", help="Learning rate", default=0.001)
@click.option("--device", help="Device to use", default="cpu")
@click.option("--save_path", help="Save model path", default=Path(__file__).parent.parent / "models")
def main(model_name: str, delta: float, epochs: int, batch_size: int, lr: float, device: str, save_path: Path) -> None:
    """
    This script is responsible for:
     - Checking if model is valid from a list of models,
     - Loading the dataset,
     - Splitting the dataset to train and validation sets,
     - Training and saving both the model and its training hyperparameters.
    :param model_name: Model name
    :param delta: Percentage of dataset to use for training
    :param epochs: Training epochs
    :param batch_size: Batch size
    :param lr: Learning rate
    :param device: Training device (cpu or gpu)
    :param save_path: Path to save model
    """
    if model_name not in MODELS:
        raise ValueError(f"Invalid model name: {model_name}")
    model_config = MODELS[model_name]
    model = model_config.model

    dataset = load_dataset(model_config.dataset)["test"]
    subset = dataset.shuffle(seed=42).select(range(int(len(dataset) * delta)))

    train_valid_split = subset.train_test_split(test_size=0.1, seed=42)
    train_partition = train_valid_split["train"].with_transform(get_model_transforms(model_config, "train"))
    valid_partition = train_valid_split["test"].with_transform(get_model_transforms(model_config, "test"))

    train_loader = DataLoader(train_partition, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_partition, batch_size=batch_size)

    # Train and save the model
    trained_model, val_accuracy = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=valid_loader,
        epochs=epochs,
        lr=lr,
        device=device,
        model_config=model_config,
    )

    train_dir = Path(save_path) / model_name / str(delta)
    os.makedirs(train_dir, exist_ok=True)
    # Save the training hyperparameters
    train_metadata = {
        "model_name": model_name,
        "delta": delta,
        "validation_accuracy": val_accuracy,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "device": device,
    }
    with open(f"{train_dir}/train_metadata.json", "w") as json_file:
        json.dump(train_metadata, json_file, indent=4)

    # Save the trained model
    torch.save(trained_model.state_dict(), f"{train_dir}/model.pth")
    logger.info(f"Model saved to {save_path}")
