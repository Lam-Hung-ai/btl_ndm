# main.py
import io
import os
import threading
from typing import Dict, List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor

from resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152


app = FastAPI(title="CIFAR-10 ResNet Predictor", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # hoặc chỉ định origin của bạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INFER_LOCK = threading.Lock()  # đảm bảo an toàn khi nhiều request chạy song song GPU/CPU

# Danh sách label CIFAR-10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Khởi tạo constructors theo thứ tự mong muốn
MODEL_CTORS = [ResNet18, ResNet34, ResNet50, ResNet101, ResNet152]
# Bộ nhớ cache model đã nạp: {model_name: torch.nn.Module}
MODELS: Dict[str, torch.nn.Module] = {}


def _load_single_model(ctor) -> None:
    """
    Khởi tạo 1 model từ ctor, tìm file *_best.pth tương ứng, nạp trọng số và đưa vào cache MODELS.
    """
    model = ctor().to(DEVICE)
    name = getattr(model, "name", model.__class__.__name__)  # fallback nếu resnet.py không set .name
    ckpt = os.path.join(WEIGHTS_DIR, f"{name}_best.pth")

    if not os.path.isfile(ckpt):
        print(f"[WARN] Thiếu trọng số cho {name}: {ckpt}")
        return

    state = torch.load(ckpt, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    MODELS[name] = model
    print(f"[OK] Loaded {name} from {ckpt}")


def preprocess_image_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Tiền xử lý ảnh: RGB -> resize 32x32 -> ToTensor() -> (1,3,32,32).
    KHÔNG normalize để khớp với code train.
    """
    img = img.convert("RGB")
    if img.size != (32, 32):
        img = img.resize((32, 32), Image.BILINEAR)
    tensor = ToTensor()(img).unsqueeze(0)
    return tensor.to(DEVICE)


def predict_all_models(x: torch.Tensor) -> List[dict]:
    """
    Chạy dự đoán top-1 qua tất cả model đã nạp.
    Trả về list dict: {model, pred_idx, pred_class, confidence}.
    """
    results = []
    with torch.no_grad():
        # Dùng lock để tránh xung đột GPU/CPU khi nhiều request đồng thời
        with INFER_LOCK:
            for name, model in MODELS.items():
                logits = model(x)  # (1,10)
                probs = F.softmax(logits, dim=1).squeeze(0)  # (10,)
                conf, idx = torch.max(probs, dim=0)
                idx = idx.item()
                conf = float(conf.item())
                results.append({
                    "model": name,
                    "pred_idx": idx,
                    "pred_class": CIFAR10_CLASSES[idx],
                    "confidence": conf
                })
    # Sắp xếp theo thứ tự ResNet18 -> ... -> ResNet152 (nếu đủ)
    order = {"ResNet18": 0, "ResNet34": 1, "ResNet50": 2, "ResNet101": 3, "ResNet152": 4}
    results.sort(key=lambda r: order.get(r["model"], 999))
    return results


# ========= SỰ KIỆN KHỞI ĐỘNG APP =========
@app.on_event("startup")
def startup_event():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    for ctor in MODEL_CTORS:
        _load_single_model(ctor)
    if not MODELS:
        # Nếu không nạp được model nào, dừng app với thông báo rõ ràng
        raise RuntimeError(
            "Không nạp được model nào! Hãy kiểm tra thư mục 'weights/' có các file *_best.pth khớp với resnet.py."
        )
    print(f"[READY] Device: {DEVICE}. Loaded models: {list(MODELS.keys())}")


# ========= ROUTES =========
@app.get("/", response_class=HTMLResponse)
def serve_index():
    """Trả về file index.html (UI một file, không CDN)."""
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.isfile(index_path):
        return HTMLResponse("<h1>index.html chưa tồn tại</h1>", status_code=500)
    with open(index_path, "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html, status_code=200)


@app.get("/health", response_class=JSONResponse)
def health():
    """Thông tin sẵn sàng của server."""
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": list(MODELS.keys())
    }


@app.post("/predict", response_class=JSONResponse)
async def predict(file: UploadFile = File(...)):
    """Nhận ảnh từ form-data, trả về dự đoán top-1 của tất cả ResNet đã nạp."""
    if file.content_type not in {"image/png", "image/jpeg", "image/jpg"}:
        raise HTTPException(status_code=400, detail="Chỉ hỗ trợ PNG/JPG.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File rỗng hoặc không đọc được.")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise HTTPException(status_code=400, detail="Không thể mở ảnh. Hãy kiểm tra định dạng.")

    x = preprocess_image_to_tensor(img)
    results = predict_all_models(x)

    return {
        "device": DEVICE,
        "num_models": len(results),
        "results": results
    }
