CREATE TABLE app_logs (
    id        SERIAL PRIMARY KEY,
    ts        TIMESTAMPTZ DEFAULT now(),  -- log 時間，預設現在
    level     TEXT        NOT NULL,       -- 'INFO', 'ERROR' 等
    source    TEXT        NOT NULL,       -- 來源，例如 'app.py', 'yolo_infer'
    run_id    INTEGER,                    -- 選填，之後可以對應到 runs.id
    message   TEXT        NOT NULL,       -- 簡短訊息
    detail    TEXT                        -- 詳細內容（例如完整 traceback）
);
