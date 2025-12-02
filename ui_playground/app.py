# 使用方式（PowerShell）：
# cd D:\Sandy\VisDrone\ui_playground
# & "C:/Users/Sandy/AppData/Local/Programs/Python/Python311/python.exe" -m streamlit run app.py

import os
import io
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# ==== 把 PostgreSQL/db_utils.py 加到匯入路徑 ====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POSTGRESQL_DIR = PROJECT_ROOT / "PostgreSQL"
sys.path.append(str(POSTGRESQL_DIR))

from db_utils import write_log  # 從 db_utils 匯入寫 log 的函式


def safe_log(level, source, message, run_id=None, detail=None):
    """
    安全寫 log：就算寫入 DB 失敗也不會讓整個 app 掛掉
    """
    try:
        write_log(level, source, message, run_id, detail)
    except Exception as e:
        print(f"[LOG ERROR] {e}")


# 強制用 CPU，避免本機 CUDA 相容問題
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 專案根目錄（app.py 在 ui_playground/ 底下）
WEIGHTS_PATH = PROJECT_ROOT / "models" / "best.pt"

# VisDrone / YOLO 類別對照
CLASS_MAP = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}

st.set_page_config(
    page_title="VisDrone YOLOv8",
    layout="wide",
    page_icon="🛰️",
)

# ================== CSS ==================
st.markdown(
    """
    <style>
    body{
        font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        background:#020617;
    }
    .hero{
        background:linear-gradient(120deg,#020617,#020617,#020617);
        background-size:200% 200%;
        padding:1.1rem 1.5rem;
        border-radius:1.1rem;
        margin-bottom:1.1rem;
        color:#e5e7eb;
        border:1px solid #1f2937;
        box-shadow:0 0 28px rgba(56,189,248,0.55);
        display:flex;
        align-items:center;
        justify-content:space-between;
    }
    .hero-title{
        font-size:2.2rem;
        font-weight:900;
        letter-spacing:0.07em;
        text-transform:uppercase;
        background:linear-gradient(90deg,#22c55e,#38bdf8,#a855f7,#f97316);
        background-size:300% 100%;
        -webkit-background-clip:text;
        background-clip:text;
        color:transparent;
        text-shadow:0 0 18px rgba(56,189,248,0.6);
        animation:titleGlow 3.5s ease-in-out infinite;
        display:flex;
        align-items:center;
        gap:0.6rem;
    }
    .hero-title span.emoji{
        font-size:2.4rem;
        filter:drop-shadow(0 0 12px rgba(250,250,250,0.7));
    }
    .hero-sub{
        font-size:0.85rem;
        color:#9ca3af;
        text-align:right;
    }
    @keyframes titleGlow{
        0%{
            background-position:0% 50%;
            text-shadow:0 0 10px rgba(56,189,248,0.5);
        }
        50%{
            background-position:100% 50%;
            text-shadow:0 0 26px rgba(251,191,36,0.9);
        }
        100%{
            background-position:0% 50%;
            text-shadow:0 0 10px rgba(56,189,248,0.5);
        }
    }
    .card{
        background:#020617;
        padding:1rem 1.1rem;
        border-radius:0.9rem;
        border:1px solid #1f2937;
        box-shadow:0 0 14px rgba(15,23,42,0.9);
        animation:cardFloat 0.6s ease-out;
    }
    @keyframes cardFloat{
        0%{opacity:0;transform:translateY(6px);}
        100%{opacity:1;transform:translateY(0);}
    }
    .card-title{
        font-size:1rem;
        font-weight:600;
        margin-bottom:0.4rem;
        color:#e5e7eb;
    }
    .small-muted{
        font-size:0.8rem;
        color:#9ca3af;
    }
    .stButton>button{
        width:100%;
        padding:1rem 1.4rem;
        font-size:1.2rem;
        font-weight:800;
        border-radius:999px;
        border:none;
        background:radial-gradient(circle at 0% 0%,#22c55e,#16a34a);
        color:white;
        box-shadow:0 0 24px rgba(34,197,94,0.7);
        cursor:pointer;
        animation:pulseButton 1.6s ease-in-out infinite;
    }
    .stButton>button:hover{
        box-shadow:0 0 30px rgba(34,197,94,1);
        transform:translateY(-2px) scale(1.03);
    }
    @keyframes pulseButton{
        0%{transform:scale(1);}
        50%{transform:scale(1.05);}
        100%{transform:scale(1);}
    }
    .img-frame{
        border-radius:0.8rem;
        padding:0.35rem;
        background:conic-gradient(from 180deg,#22c55e33,#0ea5e933,#22c55e33);
        animation:imgGlow 5s linear infinite;
    }
    @keyframes imgGlow{
        0%{box-shadow:0 0 10px rgba(34,197,94,0.2);}
        50%{box-shadow:0 0 24px rgba(34,197,94,0.6);}
        100%{box-shadow:0 0 10px rgba(34,197,94,0.2);}
    }
    .param-label{
        font-size:0.9rem;
        font-weight:600;
        color:#e5e7eb;
        margin-bottom:0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== Header ==================
st.markdown(
    """
    <div class="hero">
      <div class="hero-title">
        <span class="emoji">🛰️</span>
        <span>VisDrone YOLOv8 物件偵測</span>
      </div>
      <div class="hero-sub">
        多張影像一次偵測 · YOLOv8n · VisDrone2019-DET
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ================== 模型 ==================
@st.cache_resource
def load_model():
    return YOLO(str(WEIGHTS_PATH))


def run_inference(img: Image.Image, imgsz: int, conf: float, filename: str, classes_ids):
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


# ================== 上方：上傳 + 參數 ==================
left, right = st.columns([2, 1])

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📤 上傳影像（可多張）</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "選擇一張或多張影像 (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        first_bytes = uploaded_files[0].getvalue()
        first_img = Image.open(io.BytesIO(first_bytes)).convert("RGB")
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(first_img, caption=f"預覽：{uploaded_files[0].name}", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption(f"已選擇 {len(uploaded_files)} 張影像。")
    else:
        st.info("尚未上傳影像。")

    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🎚️ 參數與類別</div>', unsafe_allow_html=True)

    # 類別選擇（空 = 全部）
    st.markdown('<div class="param-label">想看的類別</div>', unsafe_allow_html=True)
    all_class_names = list(CLASS_MAP.values())
    selected_names = st.multiselect(
        "classes",
        options=all_class_names,
        default=all_class_names,
        label_visibility="collapsed",
    )
    if selected_names:
        selected_ids = [cid for cid, name in CLASS_MAP.items() if name in selected_names]
    else:
        selected_ids = []

    st.markdown('<div class="param-label">輸入影像尺寸 (imgsz)</div>', unsafe_allow_html=True)
    imgsz = st.slider("imgsz", 320, 1280, 640, 160, label_visibility="collapsed")
    st.caption("送進模型前會 resize 到這個大小。")

    st.markdown('<div class="param-label">信心閾值 (conf)</div>', unsafe_allow_html=True)
    conf = st.slider("conf", 0.1, 0.9, 0.25, 0.05, label_visibility="collapsed")
    st.caption("只保留置信度 ≥ conf 的框。")

    st.markdown("</div>", unsafe_allow_html=True)

# ================== 中間：按鈕 + 進度 ==================
st.markdown("")
btn_col, prog_col = st.columns([1, 3])

with btn_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    run_button = st.button("🚀 開始偵測", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with prog_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    progress_text = st.empty()
    progress_bar = st.progress(0)
    st.markdown("</div>", unsafe_allow_html=True)

# ================== 下方：結果（左圖右表） ==================
result_col, table_col = st.columns([2, 1])

if run_button and uploaded_files:
    # 按下開始偵測 + 有上傳檔案：寫一筆 run 開始的 log
    n_files = len(uploaded_files)
    safe_log(
        "INFO",
        "app.py",
        f"開始偵測：images={n_files}, imgsz={imgsz}, conf={conf}, classes={selected_names}",
    )

    try:
        all_rows = []
        results_images = []

        progress_text.markdown("⏱️ 正在處理影像…")
        progress_bar.progress(0)

        classes_ids = None if len(selected_ids) == 0 else selected_ids

        for idx, f in enumerate(uploaded_files):
            img_bytes = f.getvalue()
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            plotted_img, df_boxes = run_inference(img, imgsz, conf, f.name, classes_ids)
            results_images.append((f.name, plotted_img))

            num_boxes = 0
            if not df_boxes.empty:
                all_rows.append(df_boxes)
                num_boxes = len(df_boxes)

            # 每張影像偵測完成後寫一筆 log
            safe_log(
                "INFO",
                "app.py",
                f"單張偵測完成：file={f.name}, boxes={num_boxes}, imgsz={imgsz}, conf={conf}",
            )

            pct = int((idx + 1) / n_files * 100)
            progress_bar.progress((idx + 1) / n_files)
            progress_text.markdown(f"✅ 已完成 {idx + 1}/{n_files} 張影像（{pct}%）")

        total_boxes = 0
        df_all = None
        if all_rows:
            df_all = pd.concat(all_rows, ignore_index=True)
            total_boxes = df_all.shape[0]

        # run 結束總結 log
        safe_log(
            "INFO",
            "app.py",
            f"偵測完成：images={n_files}, total_boxes={total_boxes}, imgsz={imgsz}, conf={conf}",
        )

        # 左邊：所有偵測影像
        with result_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📸 偵測結果影像（全部）</div>', unsafe_allow_html=True)

            if results_images:
                for i in range(0, len(results_images), 2):
                    cols = st.columns(2)
                    for j in range(2):
                        idx2 = i + j
                        if idx2 < len(results_images):
                            name, img_pred = results_images[idx2]
                            with cols[j]:
                                st.markdown('<div class="img-frame">', unsafe_allow_html=True)
                                st.image(img_pred, caption=name, use_container_width=True)
                                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.write("沒有任何影像產生偵測結果。")

            st.markdown("</div>", unsafe_allow_html=True)

        # 右邊：bbox 表
        with table_col:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Bounding Boxes（所有影像彙整）</div>', unsafe_allow_html=True)

            if df_all is not None and not df_all.empty:
                st.dataframe(df_all, hide_index=True, use_container_width=True)

                csv_buf = io.StringIO()
                df_all.to_csv(csv_buf, index=False)
                st.download_button(
                    "下載偵測結果 CSV",
                    data=csv_buf.getvalue(),
                    file_name="detections_pixel_multi_image.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.write("偵測結果為空，可能是 conf 太高或選到的類別在畫面裡太少。")

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        # 偵測流程出錯：寫 ERROR log + 在 UI 顯示
        err_msg = str(e)
        err_detail = traceback.format_exc()
        safe_log("ERROR", "app.py", err_msg, detail=err_detail)

        progress_text.markdown("❌ 偵測過程發生錯誤，詳情請查看後端 log。")
        progress_bar.progress(0)
        st.error(f"偵測過程發生錯誤：{err_msg}")

elif run_button and not uploaded_files:
    # 有按按鈕但沒上傳檔案：寫一筆 WARNING log
    safe_log("WARNING", "app.py", "按下開始偵測但沒有上傳影像")
    progress_text.markdown("⚠️ 請先上傳至少一張影像再開始偵測。")
    progress_bar.progress(0)
