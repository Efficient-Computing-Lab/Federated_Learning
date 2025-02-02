from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)


class CIFAR_Net(nn.Module):
    """Model (simple CNN adapted for CIFAR-10)"""

    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self._num_classes = num_classes

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MNIST_Net(nn.Module):
    """Model (simple CNN adapted for Fashion-MNIST)"""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ModelConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataset: str
    model: nn.Module
    target_image: str
    target_label: str
    num_classes: int
    eval_transforms: Compose
    train_transforms: Compose


MODELS = {
    "Fashion-MNIST": ModelConfig(
        dataset="zalando-datasets/fashion_mnist",
        model=MNIST_Net(),
        target_image="image",
        target_label="label",
        num_classes=10,
        eval_transforms=Compose([ToTensor(), Normalize(*((0.1307,), (0.3081,)))]),
        train_transforms=Compose(
            [
                RandomCrop(28, padding=4),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(*((0.1307,), (0.3081,))),
            ]
        ),
    ),
    "CIFAR10": ModelConfig(
        dataset="uoft-cs/cifar10",
        model=CIFAR_Net(num_classes=10),
        target_image="img",
        target_label="label",
        num_classes=10,
        eval_transforms=Compose([ToTensor(), Normalize(*((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))]),
        train_transforms=Compose(
            [
                RandomHorizontalFlip(),
                RandomCrop(32, padding=4),
                ToTensor(),
                Normalize(*((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))),
            ]
        ),
    ),
    "CIFAR100": ModelConfig(
        dataset="uoft-cs/cifar100",
        model=CIFAR_Net(num_classes=100),
        target_image="img",
        target_label="fine_label",
        num_classes=10,
        eval_transforms=Compose([ToTensor(), Normalize(*((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))]),
        train_transforms=Compose(
            [
                RandomHorizontalFlip(),
                RandomCrop(32, padding=4),
                ToTensor(),
                Normalize(*((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))),
            ]
        ),
    ),
}


def get_weights(model):
    """Extract parameters from a model.

    Note this is specific to PyTorch. You might want to update this function if you use
    a more exotic model architecture or if you don't want to extrac all elements in
    state_dict.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, parameters):
    """Copy parameters onto the model.

    Note this is specific to PyTorch. You might want to update this function if you use
    a more exotic model architecture or if you don't want to replace the entire
    state_dict.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_model_transforms(model_config: ModelConfig, type_transform: str) -> Compose:
    """Return a function that apply standard transformations to images."""

    def apply_train_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[model_config.target_image] = [
            model_config.train_transforms(img) for img in batch[model_config.target_image]
        ]
        return batch

    def apply_eval_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch[model_config.target_image] = [
            model_config.eval_transforms(img) for img in batch[model_config.target_image]
        ]
        return batch

    match type_transform:
        case "train":
            return apply_train_transforms
        case "test":
            return apply_eval_transforms
        case _:
            raise ValueError(f"Invalid transformation type: {type_transform}")
