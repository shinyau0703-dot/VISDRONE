from pathlib import Path
import sys
import traceback

from ultralytics import YOLO

# ==== 把 PostgreSQL/db_utils.py 加到匯入路徑 ====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTGRESQL_DIR = PROJECT_ROOT / "PostgreSQL"
sys.path.append(str(POSTGRESQL_DIR))

from db_utils import insert_train_run, update_train_run_finished, write_log

# ====== 這裡填你這次訓練的設定 ======
MODEL_NAME = "yolov8n.pt"  # 或者使用你已有的 best.pt 當起始
DATA_YAML = PROJECT_ROOT / "config" / "visdrone.yaml"  # TODO: 改成你自己的 data.yaml 路徑
EPOCHS = 50
IMGSZ = 640
BATCH = 16
LR0 = 0.01
TRAIN_IMGS = None  # 如果你知道訓練集張數可以填，先留 None 也可以
VAL_IMGS = None
NOTES = "baseline training from Colab settings"  # 備註，可自由填

# 權重輸出位置，會變成 runs/train/run_xxx/weights/best.pt
RUNS_DIR = PROJECT_ROOT / "runs" / "train"


def safe_log(level, source, message, run_id=None, detail=None):
    try:
        write_log(level, source, message, run_id, detail)
    except Exception as e:
        print(f"[LOG ERROR] {e}")


def main():
    # 1. 在 DB 裡建立一筆訓練紀錄，拿到 run_id
    run_id = insert_train_run(
        model_name=MODEL_NAME,
        data_yaml=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        lr0=LR0,
        train_imgs=TRAIN_IMGS,
        val_imgs=VAL_IMGS,
        notes=NOTES,
    )
    safe_log("INFO", "train_visdrone.py", f"開始訓練 run_id={run_id}", run_id=run_id)

    try:
        # 2. 建 YOLO 模型 & 開始訓練
        model = YOLO(MODEL_NAME)
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            imgsz=IMGSZ,
            batch=BATCH,
            lr0=LR0,
            project=str(RUNS_DIR),
            name=f"run_{run_id}",  # 每次訓練的資料夾用 run_id 區分
        )

        # 3. 推測 best.pt 的路徑
        weights_path = RUNS_DIR / f"run_{run_id}" / "weights" / "best.pt"

        # （進階）如果你有從 results 裡抓 mAP，也可以填進來
        best_map50 = None
        best_map5095 = None

        # 4. 更新 DB，標記這次訓練已完成
        update_train_run_finished(
            run_id=run_id,
            weights_path=str(weights_path),
            best_map50=best_map50,
            best_map5095=best_map5095,
        )

        safe_log(
            "INFO",
            "train_visdrone.py",
            f"訓練完成 run_id={run_id}, weights={weights_path}",
            run_id=run_id,
        )

    except Exception as e:
        err_msg = str(e)
        err_detail = traceback.format_exc()
        safe_log("ERROR", "train_visdrone.py", err_msg, run_id=run_id, detail=err_detail)
        raise  # 讓錯誤照樣丟出來，你在 Terminal 可以看到


if __name__ == "__main__":
    main()
