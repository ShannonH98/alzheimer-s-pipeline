import torch
import torch.nn as nn
from torchvision import models


def get_model(num_classes=4):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # Freeze all layers except the final classifier
    for param in model.parameters():
        param.requires_grad = False

    # Replace final layer for our 4 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


if __name__ == "__main__":
    model = get_model()
    print(model.fc)
