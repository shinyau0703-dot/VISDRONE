import os
import psycopg2

# ===== 1. 路徑設定 =====
# 資料集的根目錄
DATA_ROOT = r"D:\Sandy\VisDrone\datasets"

# 要掃描的四個資料夾（你剛剛給的那四個）
DIR_CONFIGS = [
    # split,  file_type,   folder_path
    ("train", "image",      r"D:\Sandy\VisDrone\datasets\VisDrone2019-DET-train\images"),
    ("train", "annotation", r"D:\Sandy\VisDrone\datasets\VisDrone2019-DET-train\annotations"),
    ("val",   "image",      r"D:\Sandy\VisDrone\datasets\VisDrone2019-DET-val\images"),
    ("val",   "annotation", r"D:\Sandy\VisDrone\datasets\VisDrone2019-DET-val\annotations"),
]

# ===== 2. PostgreSQL 連線設定 =====
DB_CONFIG = {
    "host": "localhost",      # 本機
    "port": 5432,             # 預設5432
    "dbname": "VISDRONE",   
    "user": "postgres",       # postgres帳號
    "password": "***"  # 密碼
}


def get_conn():
    """建立資料庫連線"""
    return psycopg2.connect(**DB_CONFIG)


def insert_visdrone_files():
    """掃描資料夾，寫入 visdrone_files 表"""
    conn = get_conn()
    cur = conn.cursor()

    total = 0  # 計算總共處理多少檔案

    for split, file_type, folder in DIR_CONFIGS:
        print(f"\n=== 處理 {split} / {file_type} / {folder} ===")

        # os.walk 會走訪資料夾底下所有檔案（包含子資料夾）
        for root, dirs, files in os.walk(folder):
            for fname in files:
                # 根據類型簡單過濾副檔名
                if file_type == "image":
                    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                elif file_type == "annotation":
                    if not fname.lower().endswith((".txt", ".xml")):
                        continue

                # 絕對路徑
                abs_path = os.path.join(root, fname)
                # 從 DATA_ROOT 開始的相對路徑
                rel_path = os.path.relpath(abs_path, DATA_ROOT)
                # 把 Windows 的 \ 換成 /，看起來比較一致
                rel_path = rel_path.replace("\\", "/")

                # 印出要寫進去的資訊（讓你看過程）
                print(f"插入：{split} | {file_type} | {rel_path}")

                # 寫進資料庫
                cur.execute(
                    """
                    INSERT INTO visdrone_files (split, file_type, filename, rel_path, abs_path)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (split, file_type, rel_path) DO NOTHING
                    """,
                    (split, file_type, fname, rel_path, abs_path)
                )

                total += 1

    conn.commit()
    cur.close()
    conn.close()
    print(f"\n✅ 完成，嘗試插入 {total} 筆資料（重複的會被跳過）")


if __name__ == "__main__":
    insert_visdrone_files()
