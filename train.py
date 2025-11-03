import json
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from tqdm import tqdm

from early_stop import EarlyStopping
from resnet import ResNet152, ResNet18, ResNet34, ResNet50, ResNet101

# --- 1. Tập dữ liệu ---
batch_size = 512
train = CIFAR10(root="cifar_10/", train=True, transform=ToTensor(), download=False)
test = CIFAR10(root="cifar_10/", train=False, transform=ToTensor(), download=False)

train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=3)
test_dataloader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=3)

# --- 2. Thiết lập mô hình ---
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet152().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# --- 3. Thông số huấn luyện ---
epochs = 60
loss_info = {
    "train_loss": [],
    "train_acc": [],
    "test_loss": [],
    "test_acc": []
}

# --- 4. Thư mục lưu trọng số ---
os.makedirs("weights", exist_ok=True)
os.makedirs("images", exist_ok=True)
early_stopper = EarlyStopping(patience=10, verbose=False)
best_acc = 0.0

# --- 5. Vòng lặp huấn luyện ---
for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Đang huấn luyện", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        # lan truyền xuôi (forward)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # lan truyền ngược (backward)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ghi nhận chỉ số huấn luyện
        total_train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct_train += (predicted == labels).sum().item()
        total_train += labels.size(0)

    avg_train_loss = total_train_loss / total_train
    train_accuracy = correct_train / total_train

    # --- Đánh giá mô hình ---
    model.eval()
    total_test_loss = 0.0
    correct_test = 0
    total_test = 0

    with torch.no_grad():
        for images, labels in tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{epochs} - Đang kiểm tra", leave=False):
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

    # --- Lưu thông tin huấn luyện ---
    loss_info["train_loss"].append(avg_train_loss)
    loss_info["train_acc"].append(train_accuracy)
    loss_info["test_loss"].append(avg_test_loss)
    loss_info["test_acc"].append(test_accuracy)

    print(f"[Epoch {epoch+1:03d}/{epochs}] "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
          f"Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.4f}")

    # --- Lưu trọng số ---
    torch.save(model.state_dict(),f"weights/{model.name}_last.pth")

    if test_accuracy > best_acc:
        best_acc = test_accuracy
        torch.save(model.state_dict(), f"weights/{model.name}_best.pth")
        print(f"✅ Mô hình tốt nhất mới được lưu với độ chính xác {best_acc:.4f}")

    early_stopper(avg_test_loss)
    if early_stopper.should_stop:
        print("🛑 Dừng sớm được kích hoạt!")
        break

# --- 6. Tổng kết ---
print("\n🎉 Quá trình huấn luyện hoàn tất!")
print(f"Độ chính xác kiểm tra tốt nhất: {best_acc:.4f}")

# --- 7. Vẽ biểu đồ mất mát và độ chính xác ---
epochs_done = len(loss_info["train_loss"])
epochs_range = range(1, epochs_done + 1)

plt.figure(figsize=(12, 5))

# Mất mát (Loss)
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss_info["train_loss"], label="Mất mát huấn luyện")
plt.plot(epochs_range, loss_info["test_loss"], label="Mất mát kiểm tra")
plt.title("Biểu đồ mất mát qua các epoch")
plt.xlabel("Epoch")
plt.ylabel("Mất mát (Loss)")
plt.legend()
plt.grid(True)

with open("accuracy.json", "a") as file:
    json.dump({model.name: best_acc}, file, indent=4)

# Độ chính xác (Accuracy)
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss_info["train_acc"], label="Độ chính xác huấn luyện")
plt.plot(epochs_range, loss_info["test_acc"], label="Độ chính xác kiểm tra")
plt.title("Biểu đồ độ chính xác qua các epoch")
plt.xlabel("Epoch")
plt.ylabel("Độ chính xác (Accuracy)")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(f"images/{model.name}_training_curves.png", dpi=300)
plt.show()