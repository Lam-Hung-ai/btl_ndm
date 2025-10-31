import os
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from resnet import ResNet18

# --- 1. Dataset ---
batch_size = 512
train = CIFAR10(root="/home/lamhung/code/ndm/cifar_10/", train=True, transform=ToTensor(), download=False)
test = CIFAR10(root="/home/lamhung/code/ndm/cifar_10/", train=False, transform=ToTensor(), download=False)

train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=3)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=3)

# --- 2. Model setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 3. Training parameters ---
epochs = 60
loss_info = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}

# --- 4. Directory for saving weights ---
os.makedirs("weights", exist_ok=True)
best_acc = 0.0

# --- 5. Training loop ---
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record metrics
        total_train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = total_train_loss / total_train
    train_accuracy = correct_train / total_train

    # --- Evaluation ---
    model.eval()
    total_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_test_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct_test += (predicted == labels).sum().item()
            total_test += labels.size(0)

    avg_test_loss = total_test_loss / total_test
    test_accuracy = correct_test / total_test

    # --- Save info ---
    loss_info["train_loss"].append(avg_train_loss)
    loss_info["train_acc"].append(train_accuracy)
    loss_info["test_loss"].append(avg_test_loss)
    loss_info["test_acc"].append(test_accuracy)

    print(f"[Epoch {epoch+1:03d}/{epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    # --- Save weights ---
    torch.save(model.state_dict(), "weights/last.pth")

    if test_accuracy > best_acc:
        best_acc = test_accuracy
        torch.save(model.state_dict(), "weights/best.pth")
        print(f"✅ New best model saved with accuracy {best_acc:.4f}")

# --- 6. Summary ---
print("\n🎉 Training complete!")
print(f"Best Test Accuracy: {best_acc:.4f}")

# --- 7. Plot loss and accuracy ---
epochs_range = range(1, epochs + 1)

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss_info["train_loss"], label="Train Loss")
plt.plot(epochs_range, loss_info["test_loss"], label="Test Loss")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_info["train_acc"], label="Train Accuracy")
plt.plot(epochs_range, loss_info["test_acc"], label="Test Accuracy")
plt.title("Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("weights/training_curves.png", dpi=200)
plt.show()
