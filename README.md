# VISDRONE – YOLOv8 物件偵測

## 1. 專案總覽

這個專案使用 **VisDrone2019-DET** 資料集，搭配 **Ultralytics YOLOv8n** 進行多類別物件偵測練習（行人、車輛、機車、腳踏車等），目標是：

- 熟悉 **YOLOv8 + PyTorch** 的訓練與推論流程  
- 練習從 **模型輸出 → CSV → 後處理（經緯度）** 的完整 pipeline  
- 把整個過程整理成一個乾淨的範例專案（含檔案架構與流程圖）

---

## 2. 環境與版本說明

### 2.1 主要運算環境（訓練與推論）

> 模型訓練與推論在雲端 GPU 環境完成（Colab）

- OS：Linux（雲端環境預設）
- Python：**3.12.12**
- GPU：**NVIDIA Tesla T4**
- CUDA：**12.6（cu126 build）**
- PyTorch：**2.9.0+cu126**
- Ultralytics：**8.3.231**
- 主要套件：
  - `torch`
  - `torchvision`
  - `ultralytics`
  - `numpy`
  - `pandas`
  - `matplotlib`

### 2.2 本機開發環境（版控與文件）

> 本機主要用來整理程式碼、notebook 與文件，**不再跑訓練**。

- 編輯器：VS Code
- OS：Windows 10
- Python（本機）：3.11.9  
- 功能：
  - 管理 Git / GitHub repo
  - 編寫與維護 `notebooks/`、`docs/`、`README.md`
  - 保存從雲端下載的 `models/best.pt` 與偵測結果 CSV

---

## 3. 資料集說明（VisDrone2019-DET）

本專案目前使用：

- **VisDrone2019-DET-train**：訓練用影像與標註  
- **VisDrone2019-DET-val**：驗證用影像與標註  
Dataset: https://github.com/VisDrone/VisDrone-Dataset.git 

```text
datasets/
  VisDrone2019-DET-train/
    images/   # 訓練影像 JPG
    labels/   # YOLO 格式標註 TXT
  VisDrone2019-DET-val/
    images/   # 驗證影像 JPG
    labels/   # YOLO 格式標註 TXT
```

---

## 4. 專案檔案架構與流程圖

- **專案檔案架構圖**  
[docs/project_structure.md](docs/project_structure.md)  

- **模型與訓練流程圖**  
[docs/workflow.md](docs/workflow.md)  

