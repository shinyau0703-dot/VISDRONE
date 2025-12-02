# Project structure


```markdown
VisDrone/
├─ datasets/                          # 訓練/驗證用原始資料
│  ├─ VisDrone2019-DET-train/
│  │  ├─ annotations/                # 官方標註 txt
│  │  └─ images/                     # 訓練影像 jpg/png
│  └─ VisDrone2019-DET-val/
│     ├─ annotations/                # 驗證標註 txt
│     └─ images/                     # 驗證影像 jpg/png
│
├─ docs/                             
│  ├─ project_structure.md          
│  └─ workflow.md                    
│
├─ logs/                             
│  └─ logs\yolov8n_exp20251125
│
├─ models/                           # 對外提供推論用的權重
│  └─ best.pt
│
├─ notebooks/                        # Jupyter 筆記本（EDA/train model）
│  ├─ CheckDataset.ipynb
│  └─ VisDrone.ipynb
│
├─ PostgreSQL/                       # DB 結構與工具
│  ├─ db_utils.py                    # get_conn / insert_raw_image / write_log 等
│  ├─ load_visdrone_to_pg.py         # 一次性載入 VisDrone 檔名/路徑
│  ├─ app_logs.sql                   # 建立 app_logs 表的 SQL
│  ├─ raw_images.sql                 # 建立 raw_images 表的 SQL
│  ├─ train_runs.sql                 # 建立 train_runs 等表的 SQL
│  ├─ SQL_create.sql                 # 初始化所有表的總整理（可選）
│  ├─ export_last_raw_image.py       # 從 raw_images 匯出最新一張圖片
│  └─ visdrone_db.png                # pgAdmin 結構截圖
│
├─ results/                          # 分析或 demo 的輸出結果
│  ├─ detections_pixel.csv           # 推論結果（pixel 座標）
│  ├─ detections_fake_geo.csv        # 測試用假地理座標
│  └─ from_db_*.jpg                  # 從資料庫還原出的圖片（驗證用）
│
├─ runs/                             # Ultralytics 預設的訓練輸出目錄
│  └─ train/
│     ├─ run_1/
│     │  ├─ weights/
│     │  │  └─ best.pt
│     │  └─ args.yaml
│     └─ run_2/
│        ├─ weights/
│        │  └─ best.pt
│        └─ args.yaml
│
├─ ui_playground/                    # 前端 / 後端應用程式
│  ├─ app.py                         # Streamlit 前端（UI + 使用者互動）
│  ├─ backend.py                     # 模型推論 + raw_images 寫入 + logging
│  └─ train_visdrone.py              # 單獨訓練腳本（呼叫 YOLO train）
│
├─ config/                           # 設定檔
│  └─ visdrone.yaml                  # YOLO dataset 設定（train/val 路徑等）
│
├─ .env                              # 本機環境變數（DB 連線等）※ 已加入 .gitignore
├─ .gitignore                        # 忽略 .env、中間檔、暫存資料等
└─ README.md                         # 專案總說明
```










