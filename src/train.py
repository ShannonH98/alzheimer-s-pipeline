import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_loaders
from model import get_model

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS = 10
LR = 1e-3
SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "model.pth")


def train():
    print(f"Using device: {DEVICE}")

    train_loader, val_loader, _, classes = get_loaders()
    model = get_model(num_classes=len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    best_val_acc = 0.0

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

        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  -> Saved best model (val acc: {val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
