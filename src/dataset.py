import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

AUGMENTED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "AugmentedAlzheimerDataset")

CLASSES = ["NonDemented", "VeryMildDemented", "MildDemented", "ModerateDemented"]
NUM_CLASSES = len(CLASSES)

IMG_SIZE = 224
BATCH_SIZE = 32


def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


def get_loaders(data_dir=AUGMENTED_DIR, batch_size=BATCH_SIZE):
    dataset = datasets.ImageFolder(root=data_dir, transform=get_transforms())

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Classes: {dataset.classes}")
    print(f"Total: {len(dataset)} | Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    return train_loader, val_loader, test_loader, dataset.classes


if __name__ == "__main__":
    get_loaders()