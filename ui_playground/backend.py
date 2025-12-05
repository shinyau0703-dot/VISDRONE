# ui_playground/backend.py
import os
import sys
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ========= 路徑與 PostgreSQL 工具 =========
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTGRESQL_DIR = PROJECT_ROOT / "PostgreSQL"
sys.path.append(str(POSTGRESQL_DIR))

try:
    from db_utils import insert_raw_image, write_log
except ImportError:
    insert_raw_image = None
    write_log = None


def safe_log(level, source, message, run_id=None, detail=None):
    """寫 log 進 DB，失敗就只印在 console，不讓整個 app 掛掉。"""
    if write_log is None:
        return
    try:
        write_log(level, source, message, run_id, detail)
    except Exception as e:
        print(f"[LOG ERROR] {e}")


# ========= YOLO 權重與類別 =========
MODELS_DIR = PROJECT_ROOT / "models"
if (MODELS_DIR / "best.pt").exists():
    WEIGHTS_PATH = MODELS_DIR / "best.pt"
elif (MODELS_DIR / "yolov8n.pt").exists():
    WEIGHTS_PATH = MODELS_DIR / "yolov8n.pt"
else:
    WEIGHTS_PATH = PROJECT_ROOT / "ui_playground" / "yolov8n.pt"

# 強制用 CPU，避免本機 CUDA 相容問題（要用 GPU 再把這行拿掉）
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# VisDrone / YOLO 類別對照（前後端共用）
CLASS_MAP = {
    0: "people", # 原本 pedestrian
    1: "people",
    2: "bicycle",
    3: "car", 
    4: "car",  # 原本 van
    5: "car",   # 原本 truck
    6: "bicycle",   # 原本 tricycle
    7: "bicycle",   # 原本 awning-tricycle
    8: "car",   # 原本 bus
    9: "motor",  
}


def save_raw_image(img_bytes, filename, content_type, width, height):
    """
    將一張原始圖片寫進 raw_images 表，回傳 image_id。
    若沒有 db_utils 或發生錯誤，回傳 None。
    """
    if insert_raw_image is None:
        print("[DB] insert_raw_image 未匯入，略過 raw_images 寫入。")
        return None

    try:
        image_id = insert_raw_image(
            img_bytes=img_bytes,
            filename=filename,
            content_type=content_type,
            width=width,
            height=height,
        )
        safe_log("INFO", "backend.py", f"raw_images 寫入成功 image_id={image_id}, file={filename}")
        return image_id
    except Exception as e:
        print(f"[DB] 寫入 raw_images 失敗：{e}")
        safe_log("ERROR", "backend.py", f"raw_images 寫入失敗 file={filename}", detail=str(e))
        return None


@lru_cache(maxsize=1)
def load_model():
    """載入 YOLO 模型（用 lru_cache 讓整個程式共用同一個 instance）"""
    print(f"[MODEL] loading weights from: {WEIGHTS_PATH}")
    return YOLO(str(WEIGHTS_PATH))


def run_inference(img: Image.Image, imgsz: int, conf: float, filename: str, classes_ids):
    """
    單張圖片的 YOLO 推論：
    傳入 PIL.Image，回傳 (畫好框的 PIL.Image, bounding boxes 的 DataFrame)
    """
    model = load_model()
    img_np = np.array(img)

    results = model.predict(
        source=img_np,
        imgsz=imgsz,
        conf=conf,
        device="cpu",
        classes=classes_ids,  # None = 不過濾
        verbose=False,
        save=False,
    )
    r = results[0]

    plotted = r.plot()
    plotted_rgb = Image.fromarray(plotted[..., ::-1])

    rows = []
    if r.boxes is not None:
        for box in r.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls)
            rows.append(
                {
                    "file": filename,
                    "cls": cls_id,
                    "label": CLASS_MAP.get(cls_id, str(cls_id)),
                    "conf": float(box.conf),
                    "xmin": float(xyxy[0]),
                    "ymin": float(xyxy[1]),
                    "xmax": float(xyxy[2]),
                    "ymax": float(xyxy[3]),
                }
            )
    df = pd.DataFrame(rows)
    return plotted_rgb, df
