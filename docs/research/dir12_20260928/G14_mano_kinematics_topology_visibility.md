# G14：MANO 運動學、拓撲與可見性

> 2026-09-28。子代理 G14。只讀倉庫程式，CPU 數值實驗寫在 `.experiments/dir12_20260928/G14/`。未改 `model/`、`semkine/`、`configs/`、`outputs/`，未使用 GPU。
> 讀碼時的 `HEAD`：`85d76a1ec3c942118699e82377de834e98764998`（工作區其餘檔案有未提交改動；下面 sha256 對的是當時磁碟上的程式）。

本組回答四件事：（a）腕樞軸與手掌質心樞軸下，輪廓法向對 \((v,\omega)\) 的資訊矩陣；（b）點濺剪影加閉運算，與 S6 真三角遮擋輪廓 SDF 的差；（c）連續兩個 50 ms GT 狀態之間，路由頂點、可見性、SDF 改變的比例；（d）這些幾何事實支持或反駁方案 A/B/C 的哪一條前提。

## 0. 檢索記錄

日期皆為 2026-09-28。未宣稱遍歷全部文獻。

| 檢索式 / 作法 | 來源 | 命中與篩選 |
|---|---|---|
| 約 60 條書目查詢（MANO、NIMBLE、HTML、HALO、LISA、Handy、DART 2022、Spurr、Zhou、SoftRas、nvdiffrast、PyTorch3D、HaMeR、WiLoR、EventHands、MS-MANO、Koenderink、Cipolla、SMPL、NASA、HybrIK-X 等） | Crossref REST `api.crossref.org/works` | 多數第一名即目標論文；AMVUR、FastMETRO、E-3DPSM、EvHandPose 正文、骨軸 twist 的口語查詢命中錯誤條目或 0 筆。篩選：題名與作者對得上種子或 2023–2026 手部網格／生物力學／世界座標手運動 |
| 指定 arXiv id 的 abs 頁（搜尋端點 `export.arxiv.org/api` 與 `arxiv.org/search` 回 429） | arXiv abs HTML | 核到的 id 見候選表。猜錯的 id（如 `2201.02667` 並非 MANO）已丟棄 |
| `ar5iv.labs.arxiv.org/html/<id>` | ar5iv | 取得 HTML 全文後再摘公式。WebFetch 不可用，改以本機 `curl --noproxy` |
| CVF Open Access 論文頁與 PDF | `openaccess.thecvf.com` | HTML 頁只有摘要與 PDF 連結；公式來自 `pdftotext` 或 ar5iv |
| Unpaywall | `api.unpaywall.org` | IEEE DOI 回 `is_oa=false`，未用來下載 |
| Semantic Scholar、OpenAlex | 公開 API | 429，未採用 |
| GitHub 倉庫首頁 | `github.com` | 連線逾時，**未讀官方碼** |
| OpenReview notes search | `api2.openreview.net` | 「DART hand」「HALO」未返回目標筆記 |

**覆蓋缺口。** 未取得全文、因此不引用其公式：NIMBLE（TOG 2022）、HTML（ECCV 2020）、HALO（3DV 2021）、DART 手模型（NeurIPS 2022，Gao 等；與 2014 年 Schmidt 的追蹤器 DART 不是同一篇）、Koenderink 1984、Cipolla & Blake 1992、EvHandPose、E-3DPSM、Dyn-HaMR、EasyHOI、FoundHand、PromptHMR。2023–2026 沒有做逐屆 CVPR/ICCV/ECCV 目錄掃描。中科院分區表本次未打開官方 Excel，會議論文不套期刊分區。

**分區口徑。** CCF 欄寫的是《中國計算機學會推薦國際學術會議和期刊目錄》裡這些容器的通行等級（CVPR、ICCV、ECCV、NeurIPS、ACM TOG 為 A 類）。3DV、SIGGRAPH 課程、預印本不寫成 A 類論文。不確定處標「待核」。

## 1. 候選目錄

| 標題 | 作者 / 機構 | 容器 | 年 | 發表狀態 | 分區口徑 | 官方代碼（本次未開啟） | 標籤 |
|---|---|---|---|---|---|---|---|
| Embodied Hands | Romero, Tzionas, Black / MPI | ACM TOG（SIGGRAPH Asia） | 2017 | 正式發表；arXiv:2201.02610 為 2022 公開版 | CCF-A（TOG） | `mano.is.tue.mpg.de`；本倉庫 `model/mano_layer.py` | 種子、LBS、根樞軸 |
| SMPL-X | Pavlakos 等 / MPI | CVPR | 2019 | 正式發表；arXiv:1904.05866 | CCF-A | — | 手連到身體、腕 |
| On the Continuity of Rotation Representations | Zhou, Barnes, Lu, Yang, Li | CVPR | 2019 | 正式發表；arXiv:1812.07035 | CCF-A | — | 6D 旋轉 |
| An Analysis of SVD for Deep Rotation Estimation | Levinson 等 | NeurIPS | 2020 | 正式發表；arXiv:2006.14616 | CCF-A | — | SVD 正交化 |
| Soft Rasterizer | Liu, Li, Chen, Li | ICCV | 2019 | 正式發表；arXiv:1904.01786 | CCF-A | `ShichenLiu/SoftRas` | 可微可見性 |
| Modular Primitives for High-Performance Differentiable Rendering | Laine, Hellsten, Karras 等 / NVIDIA | ACM TOG | 2020 | 正式發表；arXiv:2011.03277 | CCF-A（TOG） | `nvlabs/nvdiffrast` | 光柵、反鋸齒 |
| Accelerating 3D Deep Learning with PyTorch3D | Ravi, Reizenstein, Johnson 等 / FAIR | SIGGRAPH Asia Courses | 2020 | 課程，不是會議論文；arXiv:2007.08501 | 不適用 | `facebookresearch/pytorch3d` | 可微渲染庫 |
| Weakly Supervised 3D Hand Pose via Biomechanical Constraints | Spurr, Iqbal, Molchanov, Hilliges, Kautz | ECCV | 2020 | 正式發表；arXiv:2003.09282 | CCF-A | — | 骨長、掌結構、關節角 |
| LISA | Corona, Hodan, Vo 等 / Facebook Reality Labs | CVPR | 2022 | 正式發表；arXiv:2204.01695 | CCF-A | 論文頁 `iri.upc.edu/.../lisa` | 逐骨 SDF、LBS 融合 |
| Handy | Potamias, Ploumpis, Moschoglou, Triantafyllou, Zafeiriou / Imperial | CVPR | 2023 | 正式發表 | CCF-A | — | 外觀、引用 NIMBLE/HTML |
| HaMeR | Pavlakos, Shan, Radosavovic 等 | CVPR | 2024 | 正式發表；arXiv:2312.05251 | CCF-A | `geopavlakos/hamer` | Transformer 回歸 MANO |
| WiLoR | Potamias, Zhang, Deng 等 | CVPR | 2025 | 正式發表；arXiv:2409.12259 | CCF-A | — | 定位加重建，仍輸出 MANO |
| MS-MANO | Xie, Xu, Tang, Yu, Lu | CVPR | 2024 | 正式發表 | CCF-A | — | 肌肉骨骼約束 |
| HaWoR | Zhang, Deng, Ma 等 | CVPR | 2025 | 正式發表；arXiv:2501.02973 | CCF-A | — | 世界座標手運動 |
| EventHands | Rudnev, Golyanik, Wang 等 | ICCV | 2021 | 正式發表 | CCF-A | — | 事件、LNES、毫秒 |
| HandOccNet | Park, Oh, Moon | CVPR | 2022 | 正式發表 | CCF-A | — | 遮擋特徵 |
| HybrIK-X | Li, Bian, Xu, Chen, Yang, Lu | arXiv:2304.05690 | 2023 | **預印本**；文中寫明擴展 CVPR 2021 的 HybrIK | 預印本，不計 A 類 | — | twist / swing |
| NASA | Deng 等 / Google | arXiv:1912.03207 | 2020 | 預印本（本文讀到的是 ar5iv；容器未在摘錄首段寫成會議名，**會議狀態待核**） | 待核 | — | 關節佔有、蒙皮查詢 |
| Motion Representations for Articulated Animation | Siarohin, Ren, Chai, Tulyakov | arXiv:2104.11280 | 2021 | 預印本；會議歸屬待核 | 待核 | — | 部件運動表示 |
| RealisticHands | — / arXiv:2108.13995 | 2021 | 只完整讀了摘要與開篇 | 預印本或會議待核 | 待核 | — | 參數模型加可微渲染優化 |
| VIBE | Kocabas 等 | CVPR | 2020 | 正式發表；arXiv:1912.05656 | CCF-A | — | 運動先驗（只讀到摘要與開篇） |
| NIMBLE | Li, Zhang, Qiu 等 / ShanghaiTech | ACM TOG | 2022 | 正式發表；DOI `10.1145/3528223.3530079` | CCF-A（TOG） | 未取得 | 種子；**未讀全文** |
| HTML | Qian, Wang, Mueller 等 | ECCV | 2020 | 正式發表 | CCF-A | 未取得 | 種子；**未讀全文** |
| HALO | Karunratanakul, Spurr, Fan 等 | 3DV | 2021 | 正式發表 | 3DV 不按 CCF-A 計；中科院待核 | 未取得 | 種子；**未讀全文** |
| DART（手模型） | Gao, Xiu, Li 等 | NeurIPS | 2022 | 正式發表；DOI `10.52202/068431-2685` | CCF-A（NeurIPS） | 未取得 | 種子；**未讀全文**。不是 2014 Schmidt 追蹤器 |
| Koenderink, What does the occluding contour tell us | Perception | 1984 | 只讀到 Crossref 著錄 | 期刊分區待核 | — | 遮擋輪廓理論；**無公式** |
| Cipolla & Blake, Surface shape from apparent contours | IJCV | 1992 | 只讀到 Crossref 著錄 | 待核 | — | **無公式** |
| Dyn-HaMR、EasyHOI、FoundHand、PromptHMR | 各 CVPR 2025 條目 | CVPR | 2025 | 正式發表（Crossref） | CCF-A | — | 只著錄，未深讀 |

## 2. 深讀

深讀 20 篇：下面 1–18 讀到方法段與公式或等價的機制句；19–20 讀到摘要與開篇，公式未逐式抄出，在該條標明。未讀全文的種子放在第 1 節，不在這裡冒充深讀。

### 2.1 MANO（Romero 等，TOG 2017；讀 arXiv:2201.02610 的 ar5iv）

- **原問題。** 用少數參數生成可擬合掃描、可放進圖形引擎的手網格。
- **關鍵公式**（ar5iv 公式，與 SMPL 同構）：
  \[
  M(\vec\beta,\vec\theta)=W\big(T_P(\vec\beta,\vec\theta),\,J(\vec\beta),\,\vec\theta,\,\mathcal{W}\big),
  \quad
  T_P=\boldsymbol{\bar T}+B_S(\vec\beta)+B_P(\vec\theta)
  \]
  \[
  B_P(\vec\theta;\mathcal{P})=\sum_{n=1}^{9K}\big(R_n(\vec\theta)-R_n(\vec\theta^*)\big)\mathbf{P}_n,
  \quad
  B_S=\sum_n \beta_n \mathbf{S}_n.
  \]
  \(W\) 是線性混合蒙皮，\(K\) 是部位數，\(R_n\) 是部位旋轉矩陣的元素。原文寫：pose blend shape 用來修正線性混合蒙皮，並表現手指自然彎曲。
- **觀測與假設。** 姿態空間用 LBS「for simplicity」，再加自動學到的修正形變。腕在採集架上，模型的根是腕關節。
- **代碼證據。** `model/mano_layer.py` 的 `forward`：`J = J_regressor @ v_shaped`，根變換的平移列是 \(j_0\) 而不是 \(R j_0\)，頂點再加 `transl`。`semkine/lie.py` 把這寫成
  \[
  X=R_g(X_{\mathrm{rest}}-j_0)+j_0+t,
  \]
  故 \(R_g\) 在 \(t\) 固定時繞相機系點 \(c=j_0+t\) 轉。有限差分（本組實驗）中位誤差 \(2.0\times10^{-8}\) m。
- **對本任務。** 狀態裡的 `global_orient` 就是繞腕的 \(R_g\)。換樞軸若只改網路輸出的座標、不改這條 FK，MANO 資產本身不用重訓。

### 2.2 SMPL-X（Pavlakos 等，CVPR 2019；ar5iv）

- **原問題。** 單張影像同時回歸身體、臉與手。
- **關鍵公式。** 手部直接嵌 MANO；身體參數 \(\vec\theta_b\) 收到腕為止，手指是另一組 \(\vec\theta_h\)（ar5iv 正文：body pose parameters up to and including the wrist, excluding the fingers）。
- **觀測與假設。** 腕是身體鏈與手鏈的介面。手的全局旋轉在身體模型裡是腕關節，不是手掌質心。
- **代碼證據。** 本倉庫沒有 SMPL-X；根的定義以 `lie.py` 的 \(j_0\) 為準。
- **對本任務。** 單目手追蹤沿用「繞腕」是 MANO/SMPL-X 的介面，不是為輪廓可觀測性選的樞軸。

### 2.3 Zhou 等，CVPR 2019，6D 旋轉（ar5iv:1812.07035）

- **原問題。** 歐氏空間裡哪種旋轉表示對網路是連續的。
- **關鍵結論**（摘要與貢獻列表，原文）：三維旋轉在四維及以下的歐氏表示全部不連續，因此四元數與歐拉角不連續；5D 與 6D 連續。若在網路裡做 Gram–Schmidt 正交化，就得到他們的 6D 表示。
- **觀測與假設。** 不連續性是表示的拓撲性質，與手的形狀無關。
- **代碼證據。** 本倉庫根與 15 個關節仍是軸角（`mano_full_axis_angle`）。`lie.py` 禁止把軸角相加當成旋轉複合，並用 `so3_exp` / `so3_log`。
- **對本任務。** 把根頭從軸角改成 6D，修的是參數化的連續性，不增加輪廓對 \(\omega_z\) 的資訊。可當「換表示」對照，不能當成可觀測性修復。

### 2.4 Levinson 等，NeurIPS 2020（ar5iv:2006.14616）

- **原問題。** 用 SVD 把網路輸出的矩陣投回 SO(3)，與 6D 表示相比如何。
- **關鍵機制。** 對稱正交化（SVD）把估計拉回旋轉；文中比較的是 Procrustes / 旋轉平均一類已有的投影，不是新的觀測。
- **對本任務。** 與 Zhou 相同：輸出頭的正交化不改變 \(J_r\)。

### 2.5 SoftRas（Liu 等，ICCV 2019；ar5iv:1904.01786）

- **原問題。** 硬 z-buffer 的可見性對頂點位置不可微。
- **關鍵公式**：
  \[
  \mathcal{D}_j^i=\mathrm{sigmoid}\Big(\delta_j^i\frac{d^2(i,j)}{\sigma}\Big),\quad
  \delta_j^i=\begin{cases}+1&p_i\in f_j\\-1&\text{否則}\end{cases}
  \]
  \[
  w_j^i=\frac{\mathcal{D}_j^i\exp(z_j^i/\gamma)}{\sum_k\mathcal{D}_k^i\exp(z_k^i/\gamma)+\exp(\epsilon/\gamma)},\quad
  I^i=\sum_j w_j^i C_j^i+w_b^i C_b.
  \]
  \(\sigma\to 0\) 時概率圖收斂到三角形的精確形狀。
- **觀測與假設。** 每個三角形對每個像素都有軟佔有；深度衝突用溫度 \(\gamma\) 的 softmax 融合，而不是單一勝者。
- **對本任務。** S6 的 `kssf.rasterize` 是硬 z-buffer 加精確线段距離，不是 SoftRas。方案 B 若用點濺閉運算，連 SoftRas 那種「三角形內外用連續距離」都沒有：閉運算的支撐是頂點圓盤的並，再填洞。SoftRas 的 \(\sigma\) 會把剪影塗寬，和閉運算一樣會吞掉細的指縫，只是可微。

### 2.6 nvdiffrast（Laine 等，TOG 2020；ar5iv:2011.03277）

- **原問題。** 高解析度可微光柵，並給出可用的可見性梯度。
- **關鍵機制**（正文）：反鋸齒讓覆蓋率成為頂點位置的連續函數；只在輪廓相對其他幾何體位於前方時，輪廓才提供可見性梯度。點被抽樣到的內部不產生輪廓梯度。
- **對本任務。** 事件殘差 \(n^\top(u-\pi(X))\) 用的就是這條輪廓梯度。硬光柵的內部像素對頂點的法向位移梯度為 0。點濺沒有三角形邊，輪廓法向只能事後從掩膜的距離變換估計。

### 2.7 PyTorch3D（Ravi 等，SIGGRAPH Asia 2020 課程；ar5iv:2007.08501）

- **原問題。** 把光柵做成可微、模組化的運算元。
- **關鍵句**（摘要）：包含對網格與點雲的可微渲染器，用於 analysis-by-synthesis。
- **對本任務。** 點雲渲染器在概念上接近本倉庫 `model.py` 的 `_render_chunk`（頂點撒到像素）。課程沒有主張點雲剪影等於遮擋輪廓。

### 2.8 Spurr 等，ECCV 2020（ar5iv:2003.09282）

- **原問題。** 弱監督下單目深度歧義；用生物力學軟約束縮小它。
- **關鍵公式**：
  \[
  \mathbf{b}_i=\mathbf{j}^{3D}_{i+1}-\mathbf{j}^{3D}_{p(i+1)},\quad
  \mathcal{L}_{\mathrm{BL}}=\frac1{20}\sum_{i=1}^{20}\mathcal{I}(\|\mathbf{b}_i\|_2;b_i^{\min},b_i^{\max})
  \]
  掌結構用四根指根的係數 \(c_i\) 與夾角 \(\phi_i\)：
  \[
  \mathcal{L}_{\mathrm{RB}}=\frac14\sum_{i=1}^{4}\big(\mathcal{I}(c_i;c_i^{\min},c_i^{\max})+\mathcal{I}(\phi_i;\phi_i^{\min},\phi_i^{\max})\big).
  \]
  第三項是拇指與其餘手指的關節角範圍。\(\mathcal{I}\) 是區間外的懲罰。原文寫：這些量直接從預測的 21 個關節抽出，不在測試時另做模型擬合。
- **觀測與假設。** 約束是訓練損失，作用在關節位置與骨向量上。2D 標註加 BMC 比只加 2D 更能降深度誤差（摘要：只加 2D 約降 15%，加 BMC 約降 50%）。這組百分比是他們的 RGB 弱監督設定，不是本數據集。
- **對本任務。** BMC 是先驗。它不觀測繞骨軸的 twist（骨向量 \(\mathbf{b}_i\) 對繞自身的旋轉不變）。放進方案 C2 的 \(\Lambda_{\mathrm{prior}}\) 是重複使用歷史先驗，事件沒有因此多出一條測量。

### 2.9 LISA（Corona 等，CVPR 2022；ar5iv:2204.01695）

- **原問題。** 同時表示手的形狀、外觀與對應，並能從影像動畫。
- **關鍵公式**：
  \[
  \mathbf{v}_i=\sum_{j=1}^{n_b} w_{i,j}\,\mathbf{T}_j\bar{\mathbf{v}}^r_i,\qquad
  \mathbf{x}_j=\mathbf{R}_j^{-1}(\mathbf{x}-\mathbf{t}_j)
  \]
  \[
  s=\sum_{j=1}^{n_b} w_j s_j,\quad \mathbf{c}=\sum_{j=1}^{n_b} w_j\mathbf{c}_j.
  \]
  每個查詢點變到各骨局部座標，網路預測該骨的有符號距離 \(s_j\) 與顏色，再用蒙皮權重混合。\(n_b\) 與 MANO 的 16 個關節同一量級（文中 \(n_j=16\)）。
- **觀測與假設。** 佔有是各骨 SDF 的凸組合，不是像素上的 z-buffer。自遮擋（指壓在掌上）若兩骨的 SDF 都為負，混合後仍為負，**內部遮擋邊界不會出現在零等值面的拓撲裡**，除非權重本身隨深度切換。
- **對本任務。** 方案 B 的 SDF 是影像平面上到剪影的距離，不是 LISA 的三維逐骨 SDF。用 LISA 式融合不能自動得到遮擋輪廓法向。

### 2.10 Handy（Potamias 等，CVPR 2023；CVF PDF）

- **原問題。** 高保真手形與外觀，並縮小合成到真實的域差。
- **關鍵句**（第 3 節附近）：MANO 之後，Li 等的 NIMBLE 建模骨骼與肌肉等內部結構；HTML 做參數化外觀。Handy 自己仍從掃描建 LBS。
- **對本任務。** NIMBLE 的骨頭是體積結構。輪廓仍由外表面產生。把狀態從 MANO 表皮換成 NIMBLE，不自動分開 \(\omega_z\) 與橫向平移。未讀 NIMBLE 全文，此句只轉述 Handy 的引用，不引用 NIMBLE 的公式。

### 2.11 HaMeR（Pavlakos 等，CVPR 2024；ar5iv:2312.05251 與 CVF PDF）

- **原問題。** 野外單張 RGB 的手網格。
- **關鍵機制。** ViT 骨幹加 transformer 解碼器，回歸 MANO 參數。正文寫：直接回歸網格頂點常常對得更齊，但他們仍回歸 MANO，以便姿態可解釋、可動畫。
- **觀測與假設。** 單張圖的絕對估計。沒有事件時間，沒有 50 ms 增量，沒有把幾何乘在殘差上。
- **對本任務。** 方案 C3 的低頻絕對路徑在結構上接近 HaMeR（整圖 → MANO），但 HaMeR 吃的是 RGB 裁塊，不是事件包。把它的頭接到事件池化向量上，只是換輸入，不構成新觀測模型。

### 2.12 WiLoR（Potamias 等，CVPR 2025；ar5iv:2409.12259）

- **原問題。** 野外整圖裡先找到手，再重建。
- **關鍵機制。** 卷積定位器出手區，transformer 在緊裁塊上回歸 MANO。正文寫先前方法假設手已經被裁好。
- **對本任務。** zgz 的手在 240×180 裡約佔數十像素見方（本組框面積中位 3410 px²，約 58×58）。定位不是當前誤差的主項；WiLoR 不處理 \(\omega_z\) 與平移的共線。

### 2.13 MS-MANO（Xie 等，CVPR 2024；CVF PDF）

- **原問題。** 關節驅動的手會做出生理上不像的運動。
- **關鍵機制**（摘要）：把肌肉骨骼系統接到可學習的 MANO 上，用肌肉與肌腱的激勵產生力矩軌跡；BioPR 再用 MLP 在模擬迴圈裡修正初值姿態。評估分成模型精度（對 MyoSuite）與修正功效（兩個公開數據集）。
- **觀測與假設。** 約束來自肌肉動力學模擬，不是來自輪廓。
- **對本任務。** 與 Spurr 一樣，是先驗精煉。50 ms 事件包裡它不提供 \(n^\top\delta u\)。

### 2.14 HaWoR（Zhang 等，CVPR 2025；CVF PDF）

- **原問題。** 現有方法在相機系裡做單幀手，不管世界系運動與相機軌跡。
- **關鍵句**（摘要）：補上序列中缺失的幀，同時估手運動與世界系相機軌跡。
- **對本任務。** 本協議的 RA-MPJPE 在相機系、只減腕點、不對齊旋轉。世界系軌跡是另一個狀態，不能解釋相機系裡 \(\omega_z\) 與 \(v_x,v_y\) 的共線。

### 2.15 EventHands（Rudnev 等，ICCV 2021；CVF PDF）

- **原問題。** 用單顆事件相機即時估 3D 手。
- **關鍵機制**（正文）：輸入是局部歸一化事件表面（LNES）一類的幀堆積，利用毫秒時間解析度；速度來自 GPU 事件模擬器與幀式網路。文中把 MANO、HTML 列為形狀或紋理模型，網路本身回歸的是由事件幀得到的手姿態。
- **觀測與假設。** 事件被積成幀。非同步時刻沒有進入每事件的雅可比。
- **對本任務。** S37 的事件圖保留了包內時刻（token 含 t 與 SAE 年齡），但讀出仍是整包一個 \(\Delta\)。EventHands 說明「事件幀 + 前饋」已經能做手姿態，卻留下本組要的增量可觀測性問題。

### 2.16 HandOccNet（Park 等，CVPR 2022；CVF PDF）

- **原問題。** 遮擋區域的特徵不該被丟掉。
- **關鍵機制。** 遮擋感知的數據增強、時間資訊、空間注意力；並學一張必要性圖，讓遮擋區仍有特徵。
- **觀測與假設。** 遮擋是 RGB 特徵的缺失。他們用注意力把缺失區填上，填的是圖像先驗。
- **對本任務。** 方案 A 若把被掌遮住的指頂點仍連進圖，連的是外推，不是新測量。事件只在亮度變化的像素出現；被穩定遮住的表面沒有事件。

### 2.17 HybrIK-X（Li 等，arXiv:2304.05690，預印本）

- **原問題。** 三維關節位置到部位旋轉時，繞骨軸的 twist 沒有被位置決定。
- **關鍵句**（開篇）：swing 由三維關節解析解出，twist 由視覺線索經網路得到。HybrIK-X 把這套分解擴到全身。文中寫早期版本收於 CVPR 2021。
- **觀測與假設。** 關節位置的觀測模型對 twist 是零。視覺線索（輪廓、紋理、陰影）才可能看見 twist。
- **代碼證據。** `semkine/jacobian.py` 檔頭：關節位置看不到繞骨的 twist，所以狀態必須帶旋轉，而不能只帶關節座標。同檔寫明 pose blendshape 對雅可比的第二項是手部追蹤文獻的剛體 LBS 公式所沒有的；舊 SemKine 階段日誌記錄該項中位約為雅可比範數的 0.28%（那是倉庫舊測量，日誌已於 2026-09-29 刪除，本組未重測）。
- **對本任務。** 根的 \(\omega_z\) 不是指骨 twist，但同一個零空間解釋了為什麼手指頭即使用對了路由，繞指軸的分量仍然吃不到輪廓法向。方案 B 的槓桿臂 \(L_j\) 對純 twist 為 0（點繞自身軸的線速度在軸上為 0，在輪廓生成元上常與法向垂直）。

### 2.18 NASA（Deng 等，arXiv:1912.03207）

- **原問題。** 用神經佔有表示關節物體，並能查詢空間中一點是否在體內。
- **關鍵機制**（摘要與圖說）：傳統蒙皮模型若要查佔有，需要加速結構；NASA 用神經指示函數，查詢前把點用部位變換送進典範空間。
- **對本任務。** 佔有查詢給的是內外，不是影像平面上的遮擋輪廓法向。內部摺疊（指縫、指壓掌）在三維佔有裡可以是同一個連通內部，零等值面不必經過遮擋邊界。

### 2.19 RealisticHands（arXiv:2108.13995；讀摘要與開篇）

- **原問題。** 參數手穩但表達力有限，無模型方法相反；自遮擋與自相似使網格難估。
- **關鍵句**（摘要）：網路初值再加可微渲染優化。未逐式抄錄其渲染損失，故不寫成公式。
- **對本任務。** 「可微渲染優化」就是方案 B/S38 那一類 analysis-by-synthesis。他們的穩健性來自參數模型當先驗，優化仍受單目輪廓的零空間限制。本組第 3 節把這個零空間算出來了。

### 2.20 VIBE（Kocabas 等，CVPR 2020；讀摘要與開篇）

- **原問題。** 影片上的人體運動不自然，因為缺少成對的三維運動真值。
- **關鍵機制。** 時間姿態回歸器，加上用 AMASS 分辨真運動與生成運動的對抗判別器。未抄錄損失公式。
- **對本任務。** 這是「普通運動先驗」。它約束軌跡像人，不提高單包輪廓對 \(\omega_z\) 的秩。與 Spurr/MS-MANO 同屬先驗，只是載體從骨長變成運動判別器。

## 3. 可觀測性、資訊來源與失效條件

狀態 \(x=[t,\,R_g,\,\{\text{15 個局部軸角}\}]\)，\(R_g\) 繞腕。相機系剛體運動的測量只用輪廓法向，與 `kssf.py`、`jacobian.py` 一致：

\[
r = n^\top\big(u-\pi(X)\big),\qquad
\frac{\partial r}{\partial\theta}=-n^\top\frac{\partial\pi}{\partial X}\frac{\partial X}{\partial\theta}.
\]

\(n\) 取自投影邊的外法向（離開正面三角形的質心）。\(\pi\) 用 `RENDER_SCALE=0.375`，內參來自 `zgz_*_aux.npz` 的 `camera_K`（\(f_x\approx 603.45\)，主點約 \((325.1,\,242.1)\)），渲染解析度 240×180，與 S37 配置相同。

### 3.1 樞軸：推導

左乘 \(R_g\leftarrow\exp(\omega)R_g\)、平移增量 \(v\) 時，

\[
\delta X = v+\omega\times(X-c),\qquad c_{\mathrm{wrist}}=j_0+t.
\]

有限差分把 \(\omega=(0,0,10^{-4})\) 左乘到第 0 幀的 \(R_g\) 上，778 個頂點的位移與 \(\omega\times(X-c)\) 的中位差是 \(2.0\times10^{-8}\) m，最大 \(7.7\times10^{-8}\) m。由位移反解的 \((c_x,c_y)\) 中位與 \(j_0+t\) 一致（約 \((0.0325,\,0.0424)\) m，深度 0.654 m）。所以 MANO 的 `global_orient` 在相機系的旋轉中心就是腕，不是相機原點，也不是手掌質心。

把手掌質心定義為 **argmax 蒙皮權重為關節 0 的頂點** 的相機系均值 \(c_{\mathrm{palm}}\)。同一剛體場改寫為繞 \(c_{\mathrm{palm}}=c_{\mathrm{wrist}}+d\)：

\[
v_{\mathrm{palm}}=v_{\mathrm{wrist}}+\omega\times d.
\]

因此「繞腕的純 \(\omega_z\)」等於「繞掌心的同一個 \(\omega_z\)」加上一份與 \(X\) 無關的平移 \(\omega\times d\)。這份平移完全落在 \(v_x,v_y,v_z\) 的列空間裡。消去平移之後，\(\omega_z\) 列的殘餘 \(\|(I-P)J_{\omega_z}\|\) 對兩種樞軸必須相同。實驗裡兩者相對差的中位是 **0**（2623 幀）。

\(d\) 的長度在 zgz 上幾乎是常數：中位 **38.43 mm**（p10 38.37，p90 38.51）。形狀 \(\beta\) 兩條序列相同，掌相對腕的槓桿不隨這組姿態明顯變化。腕深中位 **607 mm**（p10 541，p90 734）。

### 3.2 資訊矩陣（2623 幀，zgz_global 1387 + zgz_local 1236）

每幀取遮擋輪廓邊（`KSSF.occluding_contour`：相鄰面正面性不同，或腕環上的邊界邊）的 \(t=0.25,0.5,0.75\) 樣本，法向殘差行按邊長平方根加權。\(\Lambda=J_r^\top J_r\)。條件數在「1 mm 平移、0.02 rad 旋轉」的列縮放下計算，使兩者在 50 mm 力臂上位移同量級。

\(g=\|(I-P)J\|/\|J\|\) 是該旋轉列裡、無法被指定平移（及其他旋轉）解釋的比例。數字是中位（p10 / p90）。

| 量 | 繞腕 | 繞掌心 |
|---|---|---|
| \(g(\omega_z\mid v_x,v_y)\) | 0.433（0.397 / 0.487） | 0.607（0.553 / 0.742） |
| \(g(\omega_z\mid v_x,v_y,v_z)\) | 0.431（0.396 / 0.485） | 0.603（0.552 / 0.737） |
| \(g(\omega_x\mid v)\) | 0.655（0.458 / 0.823） | 0.847（0.643 / 0.962） |
| \(g(\omega_y\mid v)\) | 0.450（0.400 / 0.574） | 0.656（0.606 / 0.799） |
| \(g(\omega_z\mid v_x,v_y,\omega_x,\omega_y)\) | 0.340（0.207 / 0.450） | 0.487（0.301 / 0.673） |
| \(\|(I-P_{v})J_{\omega_z}\|\)（px / rad，加權） | 576（363 / 692） | **相同** |
| \(\mathrm{cond}_3(v_x,v_y,\omega_z)\)，每毫米 | 32.4（18.7 / 42.4） | 10.9（5.6 / 14.1） |
| \(\mathrm{cond}_6\)，每毫米 | 1274（849 / 1851） | 740（479 / 1044） |
| 無阻尼偽逆對純 \(\omega_z\) 的增益 | 1 | 1 |
| 課程先驗 \(\sigma_t=50\) mm、\(\sigma_r=0.3\) rad 的維納增益 | 0.9999 | 0.9999 |
| 較緊先驗 5 mm / 0.05 rad | 0.997 | 0.997 |
| 各向同性 LM，\(\lambda=10^{-2}\overline{\mathrm{diag}}\)，**米與弧度混用** | 0.440（0.333 / 0.487） | 0.439（0.333 / 0.487） |
| 同一 LM，改在每毫米座標 | 0.975（0.952 / 0.987） | 0.984（0.969 / 0.992） |

輪廓邊中位 197 條，其中摺疊邊（兩面正面性不同）佔 0.919；腕環邊界邊中位 **16** 條，與 MANO 腕部開孔的邊數一致，幀間幾乎不變。

**怎麼讀 S38 的「rot_z 增益約 0.2」。** 稠密理想輪廓上，無阻尼最小二乘把純 \(\omega_z\) **完整收回**（增益 1）。課程噪聲那種很寬的先驗也被輪廓資訊壓過（增益 0.9999），因為一行殘差已經是每弧度數百像素，而 \(1/\sigma_r^2\) 只有約 11。0.2 對得上的是另一個量：消去橫向平移以及 \(\omega_x,\omega_y\) 之後，\(\omega_z\) 列剩下的比例，其 **p10 是 0.207**（中位 0.34）。也就是說，約兩成的幀上，繞腕的 \(\omega_z\) 有八成列能量與「橫向平移 + 另外兩個轉軸」共線。這不是維納增益，也不是偽逆增益。

換到掌心後，\(g(\omega_z\mid v_x,v_y)\) 從 0.43 升到 0.61，三維條件數從 32 降到 11。升的是**座標對齊**：列範數變小（中位 1336 → 952 px/rad），殘餘像素訊號不動。各向同性、米／弧度混用的 LM 增益在兩種樞軸都是 0.44；把單位改成每毫米之後，增益回到 0.98。所以 S38 若在未縮放的 \((v,\omega)\) 上做對角阻尼，看到的「增益約 0.2–0.5」主要是**單位與阻尼**，其次才是掌腕槓桿。換樞軸會改善條件數，不會增加消去平移後的資訊，也不會把已經為 1 的無阻尼增益再抬高。

剩餘的 \(1-g\)：掌心樞軸下仍有約 39% 的 \(\omega_z\) 列與橫向平移平行。來自輪廓法向只看見徑向分量（圓形剪影對繞質心的旋轉法向殘差為 0），以及透視（\(f/Z\) 隨深度變，均勻三維平移在影像上不是均勻位移）。這部分換樞軸消不掉。

**失效條件。** 手的剪影越接近以掌心投影為圓心的圓，\(g\) 越低。腕深變化（0.54–0.73 m）改變像素／毫米，不改變上述不變殘餘。指骨 twist 對這組根部 6 列是另一個零空間（第 2.17 節），沒有算進這張 6×6 裡。

### 3.3 點濺剪影與遮擋輪廓（16 幀，按 50 ms 根旋轉分位數抽取）

點濺是 `model.py` `_render_chunk` 的佔有：頂點四捨五入撒到像素。閉運算是半徑 \(r\) 的方核先膨脹再腐蝕。對照是 `KSSF.rasterize` 的硬三角 z-buffer，以及到遮擋輪廓**線段**的有符號距離（內負外正）。

| 量 | 中位（p10 / p90） |
|---|---|
| 光柵前景佔畫面 | 0.052（0.047 / 0.062） |
| 光柵面積 / 點濺面積 | 3.39（3.14 / 3.99） |
| 點濺與光柵的 IoU | 0.248（0.210 / 0.266） |
| 點濺落在光柵外的比例 | 0 |
| 閉運算 \(r=2\) 的 IoU / 多填比例 / 仍缺的洞 | 0.870 / 0.100 / 0.046 |
| 閉運算 \(r=4\) 的 IoU / 多填 / 洞 | 0.835 / 0.186 / 0.011 |
| 閉運算 \(r=2\) 的 EDT 與 KSSF SDF，\(\lvert s\rvert<12\) px 的 MAE | 1.31 px（1.00 / 1.44） |
| 同上，16 次 3×3 池化距離 | 1.55 px（1.32 / 1.67） |
| 內部像素 MAE（光柵內、離外輪廓 >3 px、且 \(\lvert s_{\mathrm{KSSF}}\rvert<4\)） | 3.01 px（2.37 / 3.75）；\(r=4\) 為 5.29 px |
| 帶內符號一致率，\(r=2\) | 0.945（0.929 / 0.951） |
| 帶內法向餘弦（EDT 梯度 vs KSSF），\(r=2\) | 0.682（0.624 / 0.760） |
| 被 \(r=2\) 填上、但光柵為背景的像素 | 229（197 / 251） |
| 這些縫上 KSSF SDF 中位 / EDT 中位 | **+0.45 px / −2 px** |
| 輪廓邊：外 / 內（邊中點在掩膜內且離邊界 >2 px）/ 膨脹 2 px 後仍在外 | 178 / 14 / 0 |
| 正面且 z-buffer 可見的頂點 | 416 |
| 只過點濺 z-buffer、法向背對相機的頂點（`front_px=1`） | 173（145 / 197） |
| 同上，`front_px=3`（7×7） | 78（66 / 95） |
| 正面但被 z-buffer 擋住 | 3.5 |
| 指頂點落在掌主導三角上且更近 5 mm 的個數 | **0**（這 16 幀） |
| 超寬三角被丟棄 | 0 |

點濺是光柵的子集（落在光柵外的比例為 0），面積只有約 30%。閉運算 \(r=2\) 把 IoU 抬到 0.87，代價是約 10% 的多填，集中在指縫：那裡真輪廓 SDF 仍為正（縫在背景），閉運算 EDT 已為負。符號在 \(\pm12\) px 帶內仍有 94.5% 一致，因為帶的主體是外輪廓；不一致集中在縫和內部摺疊。內部像素的距離誤差是帶內的兩倍多，法向餘弦 0.68，表示方案 B 若把 \(n\) 取自閉運算 EDT，約有 \(\arccos 0.68\approx 47^\circ\) 量級的典型夾角，而不是亞像素法向。

`occluding_contour` 標的是**摺疊輪廓**（正面性翻轉）加腕開孔，不是「兩個都朝向相機的表面誰更近」的 T 形遮擋邊。這 16 幀上指壓掌的嚴格计数是 0，所以 zgz 這批樣本的剪影誤差主要是**點太稀造成的洞與指縫**，不是指腹蓋住掌心。腕開孔的 16 條邊界邊始終在輪廓集合裡，腕附近的 SDF 有一部分量的是網格的洞，不是皮膚對背景的邊界。

點濺 z-buffer 在 `front_px=1` 時約 173 個背向頂點成為像素勝者，因為前表面沒有被頂點蓋滿。`mesh_graph.visible_vertices` 再乘上 `facing_camera` 之後，這些背向頂點会被拿掉；S37 路由讀出 **沒有** 呼叫這兩個函數，它在 8 個最近頂點裡、於 3 px 內取 z 最小（`routed_readout.route_front_vertex_lbs`）。z 最小能在 3 px 內挑到前表面；洞大於 3 px 時，最近頂點可以是背向的。本組沒有另計「路由結果中背向頂點的比例」。

### 3.4 50 ms 快取（2585 對，步長 50 ms，落在 valid run 內）

`zgz_global` 69334 ms、1 段、1386 對；`zgz_local` 65284 ms、90 段、1199 對。可見性是 `facing_camera` 且 `zbuffer_visible`（`front_px=1`）。路由頂點用 `route_front_vertex_lbs`，像素步長 4（45×60），帶寬 16 px，`front_k=8`。`nearest_node_lut` 是全解析度、只含可見頂點。

50 ms 真實運動：根旋轉測地中位 **1.32°**（p10 0.58，p90 2.62），平移 **5.74 mm**（1.12 / 11.7），15 個關節測地角的平均 **4.57°**（2.81 / 7.66）。同時可見頂點的像素位移中位 **2.76 px**（0.44 / 6.32）。

| 改變比例 | 中位（p10 / p90） |
|---|---|
| 778 頂點可見位翻轉 | 0.022（0.012 / 0.036） |
| 上一幀可見、這一幀不可見 | 0.019（0.010 / 0.036） |
| 路由頂點 id 改變（任一時刻被路由的像素） | **0.656**（0.260 / 0.879） |
| 路由 id 改變（步長 4 的全部像素） | 0.108（0.039 / 0.153） |
| LUT 頂點 id 改變（16 px 帶內） | 0.645（0.228 / 0.866） |

按根旋轉分箱，頂點可見位翻轉幾乎不動（<1°：0.019；1–3°：0.023；≥3°：0.027，該箱只有 164 對）。路由 id 的全圖像素改變從 0.076 升到 0.153。可見**集合**穩，**哪個頂點擁有這個像素**不穩。頂點間距與 2.8 px 的典型位移同量級，最近頂點身份每 50 ms 換掉約三分之二。

KSSF SDF 只算了 8 對（按根旋轉分位數，不是普查）：

| 根旋轉 | 帶內 \(\lvert\Delta\mathrm{SDF}\rvert>1\) px | 帶內符號翻轉 | 可見掩膜翻轉的像素 |
|---|---|---|---|
| 0.07° | 0.010 | 0.017 | 0.002 |
| 0.67° | 0.257 | 0.031 | 0.004 |
| 0.94° | 0.007 | 0.007 | 0.001 |
| 1.18° | 0.667 | 0.093 | 0.014 |
| 1.45° | 0.749 | 0.122 | 0.019 |
| 1.78° | 0.822 | 0.166 | 0.027 |
| 2.32° | 0.771 | 0.111 | 0.017 |
| 11.2° | 0.819 | 0.231 | 0.034 |

根旋轉單獨解釋不了 SDF 是否過期：0.94° 那對帶內幾乎沒變，0.67° 已有 26%。平移方向與轉軸決定剪影怎麼動。根旋轉 \(\ge 1.2^\circ\) 的五對裡，帶內過 1 px 的比例都在 0.67 以上。中位運動是 1.32° 與 2.8 px，因此**典型 50 ms 不能沿用上一包的 SDF 或路由 id**；可見位可以，誤差約 2% 的頂點。

### 3.5 資訊從哪裡來

| 來源 | 實際給了什麼 | 失效 |
|---|---|---|
| 輪廓法向 | 根的 \(v\) 與 \(\omega_x\) 較強；\(\omega_z\) 在消去平移後仍有不可約的像素殘餘（掌心座標下約 61% 的列） | 剪影近圓、法向與切向運動垂直、透視把平移場扭成可被旋轉部分解釋的形狀 |
| 事件時刻 | 50 ms 內根只轉約 1.3°、平移約 6 mm，一階常數扭量是合理的局部模型 | 單一端點的雅可比代表不了包內起點：路由 id 已變 66% |
| 蒙皮權重 / 槓桿臂 | 把像素殘差分到關節；掌腕 38 mm 槓桿解釋腕座標下 \(\omega_z\) 與 \(v\) 的共線 | 純 twist 的槓桿為 0；權重是模型，不是測量 |
| 生物力學、AMASS、肌肉力矩 | 縮小關節角與深度的先驗體積 | 殘差為 0 時它們仍會拉狀態，除非放進先验而在殘差項為零時貢獻為零 |
| 上一包的可見性 | 2% 頂點翻轉 | 不能當成路由表或 SDF 的快取 |

## 4. 可證偽假說

**H1（樞軸最小修復）。** 在不改事件特徵、不改損失的條件下，只把根更新的旋轉中心從 \(c_{\mathrm{wrist}}\) 改成 \(c_{\mathrm{palm}}\)（\(v\) 隨 \(v\leftarrow v+\omega\times(c_{\mathrm{palm}}-c_{\mathrm{wrist}})\) 改寫），閉環根旋轉誤差會下降。

最強的反對已經在第 3.2 節：消去平移後的像素資訊不變；無阻尼與合理先驗下增益已經是 1；條件數下降是座標條件，不是新行。混用單位的 LM 增益也不隨樞軸變（0.440 對 0.439）。H1 若被說成「可觀測性修復」，與這組恆等式衝突。它仍可能作為**對角阻尼頭的預條件**有一點用，那要靠第 6 節的對照，而不是靠條件數本身。

**H2（剪影近似污染根新息）。** 方案 B 的 119 維若用點濺閉運算的 EDT，而不是 KSSF 線段 SDF，則指縫像素的符號是反的（中位 +0.45 px 對 −2 px），內部摺疊的距離誤差中位 3.0 px、法向餘弦 0.68。在這些像素上，\(n\,s\) 指向錯誤的剛體方向。

反對：帶內符號仍有 94.5% 一致，MAE 1.3 px，外輪廓佔輪廓邊的 178/192。若根新息主要由外輪廓支撐，H2 對**根**的傷害可能小於對手指的傷害。指壓掌在這 16 幀是 0，不能拿來主張 zgz 上大量 T 形內輪廓。H2 應收斂成：指縫與內部摺疊，而不是「整張 SDF 都不可用」。

本組認為 **H1 更可能被否定**：不變殘餘已經測到相對差 0，閉環只換樞軸很難把 RA 裡的根旋轉降下去。

## 5. 最小可遷移機制

數學上只做座標對齊，不加新測量。根切空間仍是相機系 \((v,\omega)\)，但網絡或求解器使用掌心座標：

\[
\begin{bmatrix}v_{\mathrm{palm}}\\ \omega\end{bmatrix}
=
\begin{bmatrix}v_{\mathrm{wrist}}+\omega\times(c_{\mathrm{palm}}-c_{\mathrm{wrist}})\\ \omega\end{bmatrix},\quad
c_{\mathrm{palm}}=\mathrm{mean}\{X_i:\arg\max_j W_{ij}=0\}.
\]

MANO 的 `global_orient` 與 `transl` 存儲不變；複合仍是 \(R\leftarrow\exp(\omega)R\)，\(t\) 用 `lie.py` 的 \(p=t+(I-R)j_0\) 對應回去。

**接入點。** 方案 B 的 \(M(L_j)\) 若已把槓桿寫成相對腕的 \(X-c_{\mathrm{wrist}}\)，把根塊的參考點改成 \(c_{\mathrm{palm}}\) 就是這一對照。不要在 `prev_mlp` 或 `root_head` 裡加一項與殘差無關的 \(W_r c\)。檔案：`semkine/jacobian.py` 的 `ROOT_ROT` 與 `pivot[:,0]`（現在固定為 `p_root`，對應腕），以及 `semkine/routed_readout.py` 不存幾何。SDF 本身若要做對照，接 `semkine/kssf.py` 的 `sdf` / `sdf_normal`，不要接 `_render_chunk` 的點濺。

**成本。** 樞軸改寫是每幀一次叉積，0 個新參數，MACs 可忽略，延遲遠小於 0.1 ms。KSSF 全圖才是成本：本機 CPU 上單幀光柵加 SDF 可以在互動實驗裡跑（16 幀加在 241 s 的總時間裡，不是熱點；熱點是 2623 幀的輪廓行與 2585 次路由）。延遲不在這裡報訓練表。

快取策略（由第 3.4 節直接得出，仍是 0 參數）：可見位可沿用一包；路由 id 與 SDF 每包重算。

## 6. 最小判別實驗

三個對照都用現成 zgz、兩種子的協議外的**凍結特徵**或**無訓練**求解，避免和「再訓一個大模型」纏在一起。

1. **只換樞軸。** 用同一批理想輪廓行，在腕座標與掌心座標各做一次每毫米 LM，比較 \(\omega_z\) 增益——本組已經做了，增益分別是 0.975 與 0.984。下一步若要閉環：在 S37 的根更新上只改複合中心，特徵與權重不變。預期 RA-MPJPE 變化小於選點噪聲（兩種子之差目前是 19.23 對 22.26 那個量級裡的零點幾毫米）。**否定 H1：** 兩種子的遞推根旋轉誤差都下降，且降幅大於只重跑評估的重現誤差。
2. **只換 SDF。** 凍結 S37 特徵，在 zgz 上用同一嶺回歸比較兩套 119 維：KSSF 線段 SDF，與 \(r=2\) 閉運算 EDT。預期差異集中在指縫像素比例高的包；外輪廓包上兩者接近（帶內 MAE 已只有 1.3 px）。**否定 H2：** 兩套特徵的根旋轉嶺回歸誤差差小於 0.3°，且指縫包也不差。
3. **與最近的簡單修復分開。** 同一回歸再加三行對照：6D 根表示（Zhou）、常數增益 \(\delta\)-trust（倉庫舊結論：0.5 的信任已拿走大部分可實現增益）、以及把 SDF 換成上一包的 SDF（快取）。預期：6D 不改善嶺回歸（標籤仍可由軸角線性讀出）；快取 SDF 在根旋轉 \(\ge 1.2^\circ\) 的包上變差（第 3.4 節已有 67% 以上的帶內像素過 1 px）。**否定「必須用線段 SDF」：** 快取 SDF 與當前 SDF 的嶺回歸誤差相同。

這些都還沒有跑成主行，不能寫進 `tools/report_table.py` 的表。

## 7. 六個創新問題

這裡的「機制」指：掌心座標只做對照；真正要遷的是「法向殘差 × 縮放後的關節雅可比」，SDF 用遮擋線段而不是點濺閉運算，並且每包重算路由與 SDF。

1. **此前未解決的矛盾。** S37 的根頭從絕對影像特徵回歸一個增量，幾何若以加項進入就會在殘差為零時仍推動狀態（S38a/b 的教訓，見 `00_CONTEXT.md`）。輪廓法向殘差在殘差為零時貢獻為零，這和契約一致。矛盾的另一半是：繞腕的 \(\omega_z\) 列有一半以上與橫向平移平行，對角頭會把這份能量算進錯誤座標。掌心座標把這份平行分量從 \(\omega_z\) 列裡減掉，條件數從 32 降到 11，但資訊量不變。
2. **新資訊從哪裡來。** 新的行只有事件位置相對**當前**剪影的法向距離。槓桿臂、蒙皮權重、骨長與關節極限都是 MANO 與 Spurr 式先驗的重複使用。上一包可見性只貢獻約 2% 的翻轉，不是新測量。
3. **事件的時間與稀疏性。** 50 ms 內根 1.32°、平移 5.7 mm、頂點 2.8 px。稀疏事件若被整包池化，估到的是這段的平均位移。路由 id 在兩端已有 66% 不同，所以包內時刻若仍去查**起點**的頂點，關聯是錯的。時間的貢獻是把 \(\delta\xi\) 與 \(\dot\xi\) 拆開，前提是每個事件用自己的 \(J(t_i)\)，而不是一份快取 SDF。
4. **相對最近的方法，變化在哪。** 相對 HaMeR/WiLoR：觀測從 RGB 裁塊的絕對 MANO 回歸，改成輪廓法向殘差。相對 EventHands：關聯從整幀 LNES 改成每事件的 \((s,n)\)。相對 SoftRas：狀態更新是線性化的 \(J^\top W r\)，不是把軟光柵放進訓練損失。相對 HybrIK：twist 仍留給有法向分量的像素，不從關節位置硬解。
5. **為什麼更簡單的修復不夠。** 可見性修復（乘上 `facing_camera`）只改 2% 量級的頂點集合，不改 \(\omega_z\) 與 \(v\) 的列空間。普通時序濾波（VIBE 式運動先驗或常數 \(\delta\)-trust）在殘差為零時仍收縮狀態，或把增益凍成常數；稠密輪廓上先驗增益已經是 1，常數信任修不了稀疏事件的方向。6D 表示修的是軸角不連續，第 3.2 節的 \(g\) 與表示維度無關。
6. **什麼結果否定它。** 第 6 節：只換樞軸就顯著降低兩種子的遞推根旋轉誤差，則「樞軸只是預條件」不成立。指縫包上閉運算 SDF 與線段 SDF 的根回歸誤差相同，則 H2 不成立。快取一包的 SDF 與重算的 SDF 在 \(\ge 1.2^\circ\) 的包上誤差相同，則「必須每包重算幾何」不成立。

## 8. 限制與誠實聲明

- 未讀全文、未引用公式：NIMBLE、HTML、HALO、NeurIPS 2022 的 DART 手模型、Koenderink 1984、Cipolla & Blake 1992、EvHandPose、E-3DPSM、Dyn-HaMR、EasyHOI、FoundHand、PromptHMR。Handy 對 NIMBLE/HTML 的一句話只是轉述。
- VIBE、RealisticHands、HaWoR、EventHands、HandOccNet、MS-MANO、運動表示那篇，公式完整度不如 MANO / Spurr / LISA / SoftRas。事件與遮擋機制依 PDF 文字，沒有打開他們的訓練程式。
- HybrIK-X 按預印本處理。NASA 與「Motion Representations for Articulated Animation」的會議狀態標待核。
- GitHub 逾時，nvdiffrast / PyTorch3D / SoftRas / HALO 的官方碼沒有讀。可見性的實作證據來自本倉庫 `kssf.py`、`mesh_graph.py`、`model.py` `_render_chunk`。
- 資訊矩陣是**理想輪廓**、無事件噪聲、無路由錯誤。它解釋不了 S37 裡「事件糾正增益只有 0.01–0.20」——那個數是稀疏事件對固定特徵的回歸，不是稠密輪廓的維納增益。本組的 0.9999 不能拿去覆蓋 prereg §9.2。
- SDF 對照 16 幀、SDF 時間差 8 對，都是分位數樣本。指壓掌為 0 只在這 16 幀成立。路由比例是像素步長 4；與全解析度 LUT 的 0.645 對 0.656 互相支持，但不是每個像素的路由。
- 掌心用「關節 0 權重最大的頂點均值」，不是面積加權，也不是輪廓質心。槓桿 38.4 mm 對這個 \(\beta\) 穩，換受試者會變。
- `g` 的加權含 \(\sqrt{\text{邊長}}\)，絕對像素列範數依賴這個權。樞軸之間的不變性不依賴權，因為兩套行只差一個與點無關的平移列。
- 實驗命令（CPU，約 241 s）：

```bash
CUDA_VISIBLE_DEVICES="" /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python -u \
  .experiments/dir12_20260928/G14/probe_g14.py \
  --kssf-frames 16 --route-stride 4 \
  --out .experiments/dir12_20260928/G14/probe_g14.json
```

輸出：`.experiments/dir12_20260928/G14/probe_g14.json`，日誌 `probe_g14.log`。程式 sha256：`mano_layer.py` `f54872b63b0d9284…`，`lie.py` `6bb7cd626e64d2f9…`，`jacobian.py` `37fa28bc14e748c7…`，`kssf.py` `849ca9c8b24f91f6…`，`routed_readout.py` `373f34435c5efcb1…`，`mesh_graph.py` `f6ddfaf87bb25e0d…`（完整 64 位在 json 的 `files_sha256`）。

## 對方案 A/B/C 的幾何判斷

- **方案 A**（778 頂點與事件同一張圖，邊上是相對偏移）。幾何前提「上一狀態的投影頂點是事件的空間錨」只在頂點身份穩時成立。50 ms 內路由 id 變 66%，可見位只變 2%。邊可以每包重算，不能快取。背向頂點在點濺洞裡會成為最近鄰；`facing_camera` 沒有進路由函數。圖本身不增加 \(\omega_z\) 的秩，只改變關聯。
- **方案 B**（SDF 新息 × 槓桿）。稠密正確輪廓在縮放座標下對根是可解的（偽逆增益 1，每毫米 LM 增益約 0.98）。點濺加 \(r=2\) 閉運算在外輪廓上 MAE 1.3 px，在指縫上符號反了。腕樞軸把 \(\omega_z\) 的條件數抬到 32；掌心降到 11，殘餘資訊不變。契約「殘差為零則更新為零」與這組線性化相容；閉運算製造的假縫內殘差會在真殘差為零時仍給出更新。
- **方案 C1。** 兩端狀態差已大到路由表不能共用，支持包內用 \(t_i\)。不支持把一份 prev SDF 用滿 50 ms。
- **方案 C2。** 增益應隨 \(\Lambda\) 變。稠密輪廓上先驗幾乎不起作用；自適應的價值在稀疏事件，不在「理想輪廓上 \(\omega_z\) 增益只有 0.2」這句——那句與無阻尼稠密實驗不符，與未縮放 LM（增益 0.44）和 \(g\) 的 p10（0.21）更接近。
- **方案 C3。** 低頻絕對路徑可以少算可見性，不能少算 SDF 與路由。可見性修復、6D、常數信任都碰不到第 3.2 節那個與樞軸無關的法向殘餘。
