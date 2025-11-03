# predict_cifar10_all_resnets.py
import argparse
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToTensor

from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152

WEIGHTS_DIR = "weights"

# Nhãn CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Danh sách tất cả các model cần dự đoán
MODEL_CTORS = [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]


def load_png_as_tensor(path: str) -> torch.Tensor:
    """
    Đọc ảnh PNG/JPG..., chuyển về RGB, resize 32x32 nếu cần,
    trả về tensor (1, C, H, W) trong [0,1] để khớp với code train (ToTensor() không Normalize).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {path}")
    img = Image.open(path).convert("RGB")
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.BILINEAR)
    tensor = ToTensor()(img).unsqueeze(0)  # (1,3,32,32)
    return tensor


def load_model_with_best_weights(model_ctor, device: str) -> torch.nn.Module:
    """
    Khởi tạo model và nạp trọng số *_best.pth theo thuộc tính model.name.
    """
    model = model_ctor().to(device)
    ckpt = os.path.join(WEIGHTS_DIR, f"{model.name}_best.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Thiếu trọng số: {ckpt}")
    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_top1(model: torch.nn.Module, x: torch.Tensor) -> Tuple[int, float]:
    """
    Trả về (class_idx, confidence) top-1 với softmax.
    """
    logits = model(x)
    probs = F.softmax(logits, dim=1)
    conf, idx = probs.max(dim=1)
    return idx.item(), conf.item()


def main():
    parser = argparse.ArgumentParser(description="Dự đoán CIFAR-10 bằng tất cả ResNet (18/34/50/101/152)")
    parser.add_argument("--img_path", type=str, required=True, help="Đường dẫn ảnh đầu vào (PNG/JPG).")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = load_png_as_tensor(args.img_path).to(device)

    print(f"\nẢnh: {args.img_path}")
    print(f"Thiết bị: {device}")
    print("\n=== DỰ ĐOÁN TỪNG MÔ HÌNH ===")

    for ctor in MODEL_CTORS:
        try:
            model = load_model_with_best_weights(ctor, device)
            idx, conf = predict_top1(model, x)
            print(f"{model.name:<12} -> {CIFAR10_CLASSES[idx]:<12} (idx={idx}, conf={conf:.4f})")
        except FileNotFoundError as e:
            # Nếu thiếu file trọng số nào đó, chỉ thông báo và tiếp tục model khác
            print(f"{ctor.__name__:<12} -> BỎ QUA ({e})")


if __name__ == "__main__":
    main()
