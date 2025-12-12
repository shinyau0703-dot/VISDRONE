## VisDrone Docker 啟動說明

專案位置：`D:\Sandy\VisDrone\ui_playground`  
Image 名稱：`visdrone-app`  
Container 名稱：`visdrone-demo`

---

#### 步驟 1：
**啟動 Docker Desktop**

---
#### 步驟 2：
**啟動 VisDrone 容器**

```powershell
docker start visdrone-demo
```
(如果看到輸出： "visdrone-demo" 代表 container 已成功啟動)

確認狀態：
```powershell
docker ps
```
---
#### 步驟 3：
打開瀏覽器，在網址列輸入：
http://localhost:8501

---
## VisDrone Docker 關閉說明
```powershell
docker stop visdrone-demo
```
(這會「停止容器」)
