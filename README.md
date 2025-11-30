# 5cm DSM + Depth Pro 誤差矯正工程方案

本文件總結了一套以 **5cm 傾斜攝影 DSM** 為幾何真值、結合 **已知相機內外參** 與 **Depth Pro** 輸出的深度圖，將單目深度做高精度校正的完整工程流程。所有深度定義均在相機座標系下的光軸方向 \(z_c\)。

---

## 1. 坐標系與輸入假設

### 1.1 世界座標系 \(W\)（DSM 坐標系）
- \((X_w, Y_w)\)：投影平面座標（UTM 等）。
- \(Z_w\)：DSM 高程（5cm），代表可見地表最高點。
- DSM 提供網格索引 \((i, j)\) 與仿射變換，將索引映射到 \((X_w, Y_w)\)，並給出對應的 \(Z_w\)。

### 1.2 相機座標系 \(C\)
- 原點在相機光心 \(C_w\)，軸定義：\(z_c\) 向前、\(x_c\) 向右、\(y_c\) 向下。
- 世界到相機：\(X_c = R (X_w - C_w)\)，其中 \(R\) 為 \(W\to C\) 的旋轉矩陣。
- 深度定義：使用 \(z_c\)（沿光軸前向距離）。

### 1.3 像素座標系 \(I\)
- 內參 \(K\) 為三角矩陣 \(\begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}\)。
- 投影：\([u, v, 1]^T \sim K [x_c/z_c, y_c/z_c, 1]^T\)。
- Depth Pro 輸出的 metric depth 與 \(z_c\) 對齊或保持線性比例。

---

## 2. 離線幾何：DSM 生成真值深度圖 \(D_{dsm}\)

### 2.1 相機標定與外參精化
1. **內參標定**：棋盤格 + OpenCV 取得 \(K\) 與畸變參數。
2. **初始 PnP**：使用若干圖像特徵點與 DSM/正射中的對應 \((X_w, Y_w, Z_w)\)，透過 `solvePnP` 得到 \(R_{init}, t_{init}\)，再推回 \(C_w\)。
3. **稠密 DSM refine（推薦）**：
   - 從 DSM 隨機採樣 3D 點投影到影像，評估與真實邊緣的重投影誤差。
   - 最小化重投影誤差，優化 \((\Delta R, \Delta t)\)，將誤差壓到 1 像素級。

### 2.2 射線–DSM 求交獲得 \(D_{dsm}\)
對每個像素 \((u,v)\)：
1. **像素 → 相機射線**：`ray_dir_cam = normalize(K^{-1} [u,v,1]^T)`。
2. **轉到世界系**：\(X_w(s) = C_w + R^T (s\,\text{ray_dir_cam})\)，參數 \(s>0\) 與 \(z_c\) 對應。
3. **光線步進與二分 refine**：
   - 以 2–5cm 步長沿 \(s\) 前進，將 \((x_w,y_w)\) 轉到 DSM 索引，雙線性插值取得 \(z_{dsm}\)。
   - 若與射線點的 \(z\) 出現符號翻轉，則在區間內二分搜尋精確交點，求得 \(s_{hit}\)。
4. **相機深度**：\(D_{dsm}(u,v) = (s_{hit} \cdot \text{ray_dir_cam})_z\)。若射線無交點則置 \(\text{NaN}\)。

---

## 3. DSM 矯正 Depth Pro 深度的算法流程

### 3.1 Depth Pro 推理
對對齊的 RGB 圖進行推理得到 \(D_{dp}\)（單位米、對應 \(z_c\)）。保持與 DSM raycast 相同的幾何（避免未同步的裁剪/縮放）。

### 3.2 構造可靠區域掩膜
- `ground_mask`：由 DSM/正射識別道路、廣場等平地並投影到影像。
- `roof_mask`：由建築 footprint + DSM 推得屋頂區域。
- 其他（植被、天空）視需求決定是否信任 DSM。
- `reliable_mask = (ground_mask | roof_mask) & isfinite(D_dsm)`。

### 3.3 誤差模型：線性校正 + 分塊 + 低頻
1. **全局線性校正**：在 `reliable_mask` 上回歸 \(d_{dsm} \approx a d_{dp} + b\)，得到 \(D_{dp}^{lin}=aD_{dp}+b\)。
2. **分塊校正（位置相關誤差）**：
   - 將影像分成 \(G\times G\) 小塊，對每塊擬合 \(d_{dsm}\approx a_{ij} d_{dp}+b_{ij}\)，不足樣本時用全局值或鄰塊插值。
   - 將 \(a_{ij}, b_{ij}\) 雙線性插值成全圖場 \(A(u,v), B(u,v)\)，得到 \(D_{dp}^{corr1}=A\odot D_{dp}+B\)。
3. **低頻殘差平滑（可選）**：
   - 計算 \(R = D_{dp}^{corr1}-D_{dsm}\)（僅在可靠區域）。
   - 對 \(R\) 做大尺度 Gaussian 模糊得 \(R_{smooth}\)。
   - \(D_{dp}^{corr2}=D_{dp}^{corr1}-R_{smooth}\)，保留高頻細節並對齊 DSM 的低頻。

### 3.4 區域融合：DSM 做地基，Depth Pro 補細節
- **地面**：直接用 \(D_{dsm}\)；若需細微起伏，可 `D = 0.8 D_{dsm} + 0.2 D_{dp}^{corr2}`。
- **屋頂**：\(D_{base}=D_{dsm}\)；提取 \(D_{detail}=D_{dp}^{corr2}-\text{blur}(D_{dp}^{corr2})\)；`D = D_base + alpha * D_detail`（\(\alpha \approx 0.3-0.5\)）。
- **立面**：用地面 + 屋頂高度構造平面模型求得 \(D_{facade\_plane}\) 作為基礎，再疊加 Depth Pro 的高頻細節：`D = D_facade_plane + beta * D_detail`。
- **植被**：按需求在 DSM 與校正後深度間加權；天空/DSM 外部區域置 \(\text{NaN}\) 或大值。

### 3.5 多幀融合
對多張無動態的影像重覆上述流程得到 \(D_{bg\_final,k}\)，再逐像素取中位數：\(D_{bg\_static} = \text{median}_k D_{bg\_final,k}\)，可強化穩定性並去除偶發噪聲。

---

## 4. 工程模組劃分（Python 建議）
- `geo/dsm_io.py`：DSM/GeoTIFF 讀寫，索引↔世界座標變換。
- `geo/camera_model.py`：管理 \(K, R, C_w\)，提供 `pixel_to_ray`、`cam_to_world`/`world_to_cam` 等。
- `geo/dsm_raycaster.py`：射線與 DSM 交點計算，生成 \(D_{dsm}\)。
- `depth/depth_pro_wrapper.py`：封裝 Depth Pro 推理，輸出 \(D_{dp}\)。
- `fusion/region_masks.py`：生成地面/屋頂/立面/植被/天空掩膜。
- `fusion/error_model.py`：全局與分塊線性校正、低頻殘差平滑。
- `fusion/depth_fusion.py`：依區域融合 DSM 與校正後深度，輸出單幀 \(D_{bg\_final}\)。
- `fusion/multiframe.py`：多幀融合（中位數/均值）。
- `build_static_background_depth.py`：主流程腳本，串起讀取、射線投射、深度推理、校正、融合與保存。

---

此方案以 5cm DSM 為幾何真值，Depth Pro 提供細節補全，並在相機座標系 \(z_c\) 下統一定義深度，可直接作為高精度背景深度生成的工程藍本。
