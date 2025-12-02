import os
import psycopg2
from dotenv import load_dotenv
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("PGHOST", "localhost"),
    "port": int(os.getenv("PGPORT", "5432")),
    "dbname": os.getenv("PGDATABASE", "VISDRONE"),
    "user": os.getenv("PGUSER", "postgres"),
    "password": os.getenv("PGPASSWORD", ""),
}

import psycopg2  # 你前面其實已經有這行，如果有就不用再加

def insert_raw_image(img_bytes, filename, content_type, width, height):
    """
    將一張原始圖片（binary）寫進 raw_images 表，回傳 image_id。
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO raw_images (filename, content_type, width, height, bytes)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (filename, content_type, width, height, psycopg2.Binary(img_bytes)),
    )
    image_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return image_id

def get_conn():
    """
    建立一個新的 PostgreSQL 連線。
    呼叫端記得用完要關閉：cur.close(); conn.close()
    """
    return psycopg2.connect(**DB_CONFIG)


def write_log(level, source, message, run_id=None, detail=None):
    """
    把一筆 log 寫進 app_logs 表。

    需要先在資料庫建表：
    CREATE TABLE app_logs (
        id        SERIAL PRIMARY KEY,
        ts        TIMESTAMPTZ DEFAULT now(),
        level     TEXT NOT NULL,
        source    TEXT NOT NULL,
        run_id    INTEGER,
        message   TEXT NOT NULL,
        detail    TEXT
    );
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO app_logs (level, source, run_id, message, detail)
        VALUES (%s, %s, %s, %s, %s)
        """,
        (level, source, run_id, message, detail),
    )
    conn.commit()
    cur.close()
    conn.close()


def insert_train_run(
    model_name,
    data_yaml,
    epochs,
    imgsz,
    batch,
    lr0,
    train_imgs=None,
    val_imgs=None,
    notes=None,
):
    """
    建立一筆訓練紀錄，回傳 run_id。

    需要先在資料庫建表：
    CREATE TABLE train_runs (
        id            SERIAL PRIMARY KEY,
        started_at    TIMESTAMPTZ DEFAULT now(),
        finished_at   TIMESTAMPTZ,
        model_name    TEXT,
        data_yaml     TEXT,
        epochs        INTEGER,
        imgsz         INTEGER,
        batch         INTEGER,
        lr0           REAL,
        train_imgs    INTEGER,
        val_imgs      INTEGER,
        best_map50    REAL,
        best_map5095  REAL,
        weights_path  TEXT,
        notes         TEXT
    );
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO train_runs
        (model_name, data_yaml, epochs, imgsz, batch, lr0, train_imgs, val_imgs, notes)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
        """,
        (model_name, data_yaml, epochs, imgsz, batch, lr0, train_imgs, val_imgs, notes),
    )
    run_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return run_id


def update_train_run_finished(run_id, weights_path=None, best_map50=None, best_map5095=None):
    """
    訓練完成後更新 train_runs：
    - finished_at 設為 now()
    - 若有提供 weights_path / mAP，則一併更新
    """
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE train_runs
        SET
            finished_at = now(),
            weights_path = COALESCE(%s, weights_path),
            best_map50 = COALESCE(%s, best_map50),
            best_map5095 = COALESCE(%s, best_map5095)
        WHERE id = %s;
        """,
        (weights_path, best_map50, best_map5095, run_id),
    )
    conn.commit()
    cur.close()
    conn.close()


if __name__ == "__main__":
    # 簡單連線測試
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    print("DB 連線測試結果：", cur.fetchone())
    cur.close()
    conn.close()
    print("DB_CONFIG(host, dbname, user)：", DB_CONFIG["host"], DB_CONFIG["dbname"], DB_CONFIG["user"])
