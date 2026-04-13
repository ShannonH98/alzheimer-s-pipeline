import torch
import torch.nn as nn
import torch.optim as optim
import os
import csv
import matplotlib.pyplot as plt
from dataset import get_loaders
from model import get_model

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 20
LR = 1e-4
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "model.pth")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def train():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUT_DIR, exist_ok=True)

    train_loader, val_loader, _, classes = get_loaders()
    model = get_model(num_classes=len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)

    best_val_acc = 0.0
    train_losses, train_accs, val_accs = [], [], []

    # CSV log
    csv_path = os.path.join(OUT_DIR, "training_log.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_acc"])

    for epoch in range(EPOCHS):
        # --- Train ---
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = running_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_correct, val_total = 0, 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        scheduler.step(val_acc)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved best model (val acc: {val_acc:.4f})")

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{val_acc:.4f}"])

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

    # Save training curves
    epochs_range = range(1, EPOCHS + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs_range, train_losses, marker="o")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    ax2.plot(epochs_range, train_accs, marker="o", label="Train")
    ax2.plot(epochs_range, val_accs, marker="o", label="Val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"))
    print("Saved training curves to outputs/training_curves.png")


if __name__ == "__main__":
    train()
