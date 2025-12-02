CREATE TABLE train_runs (
    id            SERIAL PRIMARY KEY,
    started_at    TIMESTAMPTZ DEFAULT now(),
    finished_at   TIMESTAMPTZ,
    model_name    TEXT,          -- 起始模型yolov8n.pt
    data_yaml     TEXT,          -- 使用的 data.yaml 路徑
    epochs        INTEGER,
    imgsz         INTEGER,
    batch         INTEGER,
    lr0           REAL,          -- 初始 learning rate
    train_imgs    INTEGER,       -- 訓練集張數
    val_imgs      INTEGER,       -- 驗證集張數
    best_map50    REAL,          -- 最佳 mAP@0.50
    best_map5095  REAL,          -- 最佳 mAP@0.50:0.95
    weights_path  TEXT,          -- 最終輸出的 best.pt 路徑
    notes         TEXT           -- 備註
);
