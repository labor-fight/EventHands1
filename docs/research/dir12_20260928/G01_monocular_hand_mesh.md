# G01 單目手部 mesh：全局旋轉、殘差更新與事件觀測

> 2026-09-28。子代理 G01。只寫本檔，未改程式、配置、測試或 `outputs/`，未使用 GPU。
> 共用脈絡：`docs/research/dir12_20260928/00_CONTEXT.md`。
> 深讀 25 篇（方法節與評測協議已對原文 HTML 或 CVF PDF 逐段核對）。其餘進候選目錄，標明未深讀。
> MANO 原文 PDF 本次未取得；其公式只引自後續論文的轉述，並在第 8 節聲明。

## 0. 檢索記錄

日期皆為 2026-09-28。未宣稱遍歷全部文獻。

| 檢索式 / 操作 | 來源 | 命中 | 篩選 |
|---|---|---|---|
| `monocular 3D hand mesh reconstruction`，`from_publication_date:2023-01-01,to_publication_date:2026-09-28` | OpenAlex works API | 約 3275（含大量不相關） | 只留手部 mesh/姿態、事件手、殘差回饋、全局/世界系旋轉 |
| 種子標題逐條 `search=`（HaMeR、WiLoR、HaWoR、Dyn-HaMR、HOLD、EventHands、EvHandPose、Ev2Hands、EvRGBHand、EventEgoHands、H2ONet、Deformer、HandOccNet、MobRecon、METRO、Mesh Graphormer、I2L-MeshNet、PyMAF、PyMAF-X、HMR、CLIFF、Hamba、SimpleHand、ACR、InterWild、HARP、Hand4Whole、IntagHand、FrankMocap、HandBooster、CameraHMR、E-3DPSM、contrast maximization 等） | OpenAlex | 每條 top-3；其後 **429，當日額度用盡** | 用 DOI / arXiv PDF 連結核對正式版本與預印本是否為同一題 |
| `query.title=`：H2ONet、Deformer、ReFit、Zhou 2020、EventPointMesh | Crossref | 第一筆即目標論文；`total-results` 被寬查詢放大，不可當命中數 | 取 `container-title`、DOI、作者姓氏 |
| CVF Open Access 索引 `CVPR2023/2024/2025/2026`、`ICCV2023/2025` `?day=all` | openaccess.thecvf.com | 論文數：CVPR 2353 / 2716 / 2871 / 4042；ICCV 2156 / 2701。標題含 hand/mano/event 後再人工收窄 | 保留單目手 mesh/姿態、事件手或事件人體、SDF/迭代/世界系；去掉 humanoid locomotion 等誤擊 |
| `https://www.ecva.net/papers.php` | ECVA | HTTP 200，約 3.6 MB，靜態 HTML 幾乎沒有論文連結 | **ECCV 2024/2026 目錄未展開**，ECCV 覆蓋有缺口 |
| `https://export.arxiv.org/api/query` | arXiv API | HTTP 406，空正文 | 改抓 `arxiv.org/abs` 與 `arxiv.org/html` |
| `arxiv.org/html/<id>` 與 `arxiv.org/abs/<id>` | arXiv HTML | 成功取得 31 篇 HTML（見第 2 節）；`1804.01306` HTML 只有 21 KB，對比度最大化未當深讀 | 公式取自 HTML 中的 `application/x-tex` |
| CVF `papers/*.pdf` + `pdftotext -layout` | CVF Open Access | H2ONet、ReFit、HandR2N2、DIR、NVF、HOISDF、gSDF、HandOS、EgoWorld、EventEgo3D、MeMaHand、MS-MANO、Tradeoff、PriorTemp、TokenHand、MaskHand、PhysHand、PAD-Hand、GraphFreq、HHMR 均 HTTP 200 | 深讀只用其中方法已被讀到的；其餘只入目錄 |
| Semantic Scholar paper match / GitHub search | S2、GitHub API | 多數 429 或 rate limit | 只保留已返回的 Deformer arXiv `2303.04991`，以及已打開的 `geopavlakos/hamer` `mano_wrapper.py` 開頭 |
| MANO 原文 | `ps.is.tue.mpg.de`、`files.is.tue.mpg.de`、`mano.is.tue.mpg.de` | SSL 自簽或 404 | **未讀到 SIGGRAPH Asia 2017 全文** |

篩選規則：2023-01-01 至 2026-09-28 的單目 3D 手 mesh/姿態（RGB、深度、事件），加上定義旋轉表示、迭代回饋、重投影優化或事件觀測模型所必需的更早工作。雙目/多視角只在「不能憑空變成第二視角」時點名排除。預印本不寫成已發表。

覆蓋缺口：ECCV 2024/2026 靜態目錄未打開；OpenAlex/S2/GitHub/arXiv API 中途限流；中文期刊與 workshop 未系統檢索；CVPR/ICCV 索引裡還有 TokenHand、MaskHand、PAD-Hand、物理擴散手運動、MS-MANO、DIR、MeMaHand、圖頻率手形等，只入目錄、未逐式深讀。不把這次檢索說成完備。

分區口徑（全文統一）：

- 會議 **CVPR / ICCV / ECCV / NeurIPS**，期刊 **TPAMI / TOG**：按通行的 **CCF 推薦目錄 2022** 視為 A 類。本次沒有重新下載目錄 PDF 逐條核對，標「CCF 2022 通行認定」。
- **中科院分區表針對期刊**。TPAMI 在近年升級版通常列為一區 Top，具體版本年份 **待核**。會議不套中科院期刊分區。
- **3DV、ICIP**：不在上述 A 類清單裡。社區常把它們放在 CCF C，本次標 **待核**。
- **IEEE Access**：非 CCF-A。中科院分區 **待核**。

## 1. 候選目錄

「深讀」列標本次是否讀完方法節。代碼連結來自論文正文或 arXiv 頁，未逐一 clone。

| 標題 | 作者 / 機構（讀到的） | 會議/期刊 | 年 | 發表狀態 | 分區口徑 | 代碼 | 關聯標籤 |
|---|---|---|---|---|---|---|---|
| Embodied Hands (MANO) | Romero, Tzionas, Black；MPI | SIGGRAPH Asia / TOG | 2017 | 正式（DOI `10.1145/3130800.3130883`） | CCF 2022：TOG/SIGGRAPH Asia 為 A | mano.is.tue.mpg.de | 狀態定義；**原文 PDF 未取得** |
| End-to-end Recovery of Human Shape and Pose (HMR) | Kanazawa 等；Berkeley | CVPR | 2018 | 正式；arXiv `1712.06584` | CCF-A | — | 迭代誤差回饋；軸角殘差相加 |
| I2L-MeshNet | Moon, Lee | ECCV | 2020 | 正式；arXiv `2008.03713` | CCF-A | github.com/mks0601/I2L-MeshNet_RELEASE | lixel 絕對 3D；**未深讀** |
| Monocular Real-time Hand Shape and Motion Capture | Zhou 等；清华/MPI | CVPR | 2020 | 正式；arXiv `2003.09572` | CCF-A | — | 根相對關節 + 閉式深度；IK 一次前向，不是迭代擬合 |
| METRO | Lin, Wang, Liu；Microsoft | CVPR | 2021 | 正式；arXiv `2012.09760` | CCF-A | github.com/microsoft/MeshTransformer | 關節/頂點 query 與影像特徵同序列 |
| Mesh Graphormer | Lin, Wang, Liu | ICCV | 2021 | 正式；arXiv `2104.00272` | CCF-A | github.com/microsoft/MeshGraphormer | 方案 A 最近鄰：網格頂點與影像格點同一 encoder |
| PyMAF | Zhang 等 | ICCV | 2021 | 正式；arXiv `2103.16507` | CCF-A | — | 網格對齊特徵回饋；身體 |
| EventHands | Rudnev 等；MPI | ICCV | 2021 | 正式；arXiv `2012.06475` | CCF-A | 論文頁；倉庫本次未打開 | LNES；絕對回歸 + 常速度 Kalman |
| MobRecon | Chen 等 | CVPR | 2022 | 正式；arXiv `2112.02753` | CCF-A | github.com/SeanChenxy/HandMesh | 像素對齊 lifting + spiral；**方法總述，未逐式** |
| HandOccNet | Park 等 | CVPR | 2022 | 正式；arXiv `2203.14564` | CCF-A | github.com/namepllet/HandOccNet | FIT/SET 特徵注入；**未逐式** |
| CLIFF | Li 等；Huawei | ECCV | 2022 | 正式；arXiv `2208.00571` | CCF-A | github.com/huawei-noah/noah-research | 裁剪丟掉全局旋轉；全身 |
| FrankMocap | Rong 等 | ICCV W | 2021 | 正式；arXiv `2008.08324` | workshop，非 A 類主會 | — | 手部回歸基線；**未深讀** |
| PyMAF-X | Zhang 等 | TPAMI | 2023 | 正式；arXiv `2207.06400` | CCF-A 期刊；中科院一區 Top **待核年份** | — | 全身延伸；**未深讀** |
| H2ONet | Xu, Wang, Tang, Fu | CVPR | 2023 | 正式 DOI `10.1109/cvpr52729.2023.01635`；本次未找到 arXiv | CCF-A | — | 規範姿態與全局朝向解耦；6D；多幀 ΔR |
| Deformer | Fu, Liu, Xu, Niebles, Kitani；CMU | **ICCV**（種子寫 CVPR，以 DOI 為準） | 2023 | 正式；arXiv `2303.04991` | CCF-A | — | 時序融合；軸角差分當運動 |
| ReFit | Wang, Daniilidis；UPenn | ICCV | 2023 | 正式 DOI `10.1109/iccv51070.2023.01346` | CCF-A | yufu-wang.github.io/refit | 重投影採樣特徵 + 每關節 GRU；身體 |
| HandR2N2 | Cheng 等（CVF 檔名） | ICCV | 2023 | 正式 | CCF-A | — | 深度點雲上的殘差 RNN；關節座標不是 MANO 根旋 |
| Neural Voting Field | Huang 等；Buffalo/Microsoft | CVPR | 2023 | 正式 | CCF-A | linhuang17.github.io/NVF | 相機系關節；三維 SDF + voting，不是影像 SDF |
| gSDF | Chen 等 | CVPR | 2023 | 正式 | CCF-A | — | 手-物三維 SDF；**未深讀** |
| ACR / InterWild / HARP / IntagHand | Yu / Moon / Karunratanakul / Li 等 | CVPR 2023 / CVPR 2022 | 2022–2023 | 正式；arXiv 見檢索 | CCF-A | 各論文 GitHub | 雙手/個性化；**未逐式** |
| HaMeR | Pavlakos 等；Berkeley | CVPR | 2024 | 正式；arXiv `2312.05251` | CCF-A | github.com/geopavlakos/hamer | 絕對回歸 θ,β,π；評測幾乎全是 PA |
| SimpleHand（正式題：A Simple Baseline for Efficient Hand Mesh Reconstruction） | Zhou 等 | CVPR | 2024 | 正式；arXiv `2403.01813` | CCF-A | 論文稱將公開 | 直接回歸頂點；明文 PA 不含全局旋轉 |
| Hamba | Dong 等 | NeurIPS | 2024 | 正式 DOI `10.52202/079017-0069`；arXiv `2407.09646` | CCF-A | — | 21 關節圖 + Mamba；仍是單幀絕對回歸 |
| EvRGBHand | Jiang 等 | CVPR | 2024 | 正式；arXiv `2403.07346` | CCF-A | — | 事件+RGB；頂點回歸；ConvLSTM/時序注意 |
| HOLD | Fan 等；ETH/MPI | CVPR | 2024 | 正式；arXiv `2311.18448` | CCF-A | github.com/zc-alexfan/hold | 測試期渲染比較 + 規範空間 SDF |
| HOISDF | Qi 等 | CVPR | 2024 | 正式 | CCF-A | github.com/amathislab/HOISDF | 預測的手/物 SDF 引導特徵，不是剪影殘差 |
| EvHandPose | Jiang 等 | TPAMI | 2024 | 正式 DOI `10.1109/tpami.2024.3380648`；arXiv `2303.02862` | CCF-A；中科院一區 Top **待核年份** | — | IWE/對比度最大化 + 手邊緣；Conv-GRU；仍回歸絕對姿態 |
| Ev2Hands | Millerdurai 等 | 3DV | 2024 | 正式 DOI `10.1109/3dv62453.2024.00008`；arXiv `2312.14157` | 3DV **待核**（非 CCF-A 清單） | — | 事件點雲；絕對 R、t；無遞推狀態 |
| EventEgo3D | Millerdurai 等 | CVPR | 2024 | 正式 | CCF-A | — | 自我中心事件**人體**；LNES；**只讀摘要** |
| WiLoR | Potamias 等 | CVPR | 2025 | 正式 DOI `10.1109/cvpr52734.2025.01143`；arXiv `2409.12259` | CCF-A | 論文頁；倉庫本次限流未打開 | 全圖檢測 + 網格對齊 Δθ；仍報 PA |
| HaWoR | Zhang 等 | CVPR | 2025 | 正式；arXiv `2501.02973` | CCF-A | — | 相機系手 + SLAM 世界系；Φ 與 Θ 分開 |
| Dyn-HaMR | Yu, Zafeiriou, Birdal | CVPR | 2025 | 正式；arXiv `2412.12861` | CCF-A | — | 重投影魯棒損失；先優化根旋/平移 |
| HandOS | Chen 等 | CVPR | 2025 | 正式 | CCF-A | — | 單階段；報 PA-MPJPE；**只讀摘要** |
| EgoAllo（題：Estimating Body and Hand Motion in an Ego-sensed World） | Yi 等 | CVPR | 2025 | 正式 | CCF-A | — | 世界/異心座標系的身體+手；**只讀開頭** |
| EventEgoHands | Hara 等 | ICIP | 2025 | 正式 DOI `10.1109/icip55913.2025.11084751`；arXiv `2505.19169` | ICIP **待核** | — | 自我中心事件；分割後點雲；絕對回歸 |
| EventPointMesh | Hori 等 | TVCG | 2025 | 正式 DOI `10.1109/tvcg.2024.3462816` | TVCG 為 CCF-A 期刊（2022 口徑） | — | 事件點雲人體 mesh；**未讀全文** |
| EventEgoHands++ | — | IEEE Access | 2026 | 正式 DOI `10.1109/access.2026.3735008` | 非 CCF-A；中科院 **待核** | — | **未讀** |
| E-3DPSM | Deshmukh 等；MPI | — | 2026 | **預印本** arXiv `2604.08543`。CVPR 2026 OA 索引無此標題。共用脈絡寫 CVPR 2026，與本次 OA 目錄不一致，此處不寫成已發表 | 預印本，無分區 | 4dqv.mpi-inf.mpg.de/E-3DPSM/ | 人體；可學習但推理期常數 Q、R 的 Kalman |
| EgoEV-HandPose | — | — | 2026 | 預印本 arXiv `2605.12297` | 無 | github.com/ZJUWang01/EgoEV-HandPose | **雙目**事件；不作為單目證據 |
| TokenHand / MaskHand / PAD-Hand / 直覺物理手運動 / DIR / MeMaHand / MS-MANO | 見 CVF 檔名 | CVPR/ICCV 2023–2026 | 2023–2026 | 正式（OA 索引在列） | CCF-A | — | 已下載 PDF，**未深讀** |

## 2. 深讀

每篇：原問題；關鍵公式（原文記號，出處為論文節號或式號）；觀測與假設；代碼；對本任務。未讀到的公式不寫。

### 2.1 MANO（轉述，非原文）

Zhou 等 CVPR 2020 §3.2 把 MANO 寫成：形狀 \(\beta\in\mathbb{R}^{10}\)，姿態 \(\theta\in\mathbb{R}^{21\times 3}\) 為軸角；

\[
\mathcal{T}(\beta,\theta)=\bar{T}+\mathcal{B}_s(\beta)+\mathcal{B}_p(\theta),\quad
\mathcal{M}(\theta,\beta)=W(\mathcal{T}(\theta,\beta),\theta,\mathcal{W},\mathcal{J}(\theta)).
\]

EvHandPose §III-A 式 (1) 寫 \(M(\beta,\theta)=W(T_P(\beta,\theta),J(\beta),\theta,\mathcal{W})\)，輸出 778 頂點。HaMeR §3.1 把輸入寫成 \(\theta\in\mathbb{R}^{48}\)（含全局朝向 3 維 + 15 關節）與 \(\beta\in\mathbb{R}^{10}\)。

觀測：MANO 是蒙皮模型，不是觀測模型。全局旋轉是繞腕的剛體因子，與 15 個局部軸角分開。假設：LBS 與形狀 PCA 足夠表示手。

代碼：HaMeR 倉庫 `hamer/models/mano_wrapper.py` 繼承 `smplx.MANOLayer`（本次只讀到檔案開頭）。

對本任務：S37 的 51 維就是平移 3 + 全局軸角 3 + 15×3。Zhou 寫明旋轉是軸角；後面多篇把增量直接加在這個向量上。

### 2.2 HMR（CVPR 2018）

原問題：單張 RGB 直接回歸 SMPL 與弱透視相機。

§3.1：姿態 \(\theta\in\mathbb{R}^{3K}\) 為軸角；全局旋轉 \(R\) 也用軸角；\(\hat{\mathbf{x}}=s\Pi(RX(\theta,\beta))+t\)。

§3.2 迭代誤差回饋：回歸器吃影像特徵 \(\phi\) 與當前 \(\Theta_t\)，輸出 \(\Delta\Theta_t\)，

\[
\Theta_{t+1}=\Theta_t+\Delta\Theta_t,\quad \Theta_0=\bar{\Theta}.
\]

他們明確不把當前估計渲染回影像，只在潛空間拼接 \([\phi,\Theta]\)。重投影損失 \(L_{\mathrm{reproj}}\) 只在最後一次迭代上監督。對抗先驗在每次迭代上都加。

觀測：2D 關節 + 可選 3D。假設：弱透視；裁剪後的人；迭代加性修正能處理旋轉。

代碼：未核對官方實現。

對本任務：這是 `prev_mlp` 與「參數空間相加」的祖先。\(\Delta\Theta\) 依賴 \(\Theta_t\) 本身，不依賴「投影與影像的差是否為零」。殘差為零時更新不必為零。軸角相加正是本倉庫 `semkine/lie.py` 禁止的運算。HMR 的初始值是訓練均值 \(\bar{\Theta}\)，和 S37 的 `prev_mlp` 把根往均值拉是同一類收縮。

### 2.3 PyMAF（ICCV 2021）

原問題：HMR 的 IEF 每次都用同一全局特徵，不隨當前網格對齊狀況改變。

§3.2 式 (3)：

\[
\Theta_{t+1}=\Theta_t+\mathcal{R}_t\big(\Theta_t,\mathcal{F}(\phi_s^t,\Pi(\tilde{M}_t))\big),\quad t>0.
\]

\(t=0\) 時 \(\Theta_0\) 仍是訓練均值。採樣點是當前網格下採樣後的投影。輔助任務是渲染的 IUV，不是剪影距離。

觀測：空間特徵圖。假設：投影點上的外觀特徵能表示「網格與影像是否對齊」。他們自己寫：相對旋轉的小誤差會造成投影與影像證據的大錯位。

代碼：未打開。

對本任務：這是 WiLoR 的直接前身，也是「網格對齊回饋」的定義。它採的是特徵，不是 \(r=\Pi(M)-I\)。特徵在正確位置上不是零向量，所以沒有「證據為零則更新為零」。幾何只決定採樣位置，不與殘差相乘。

### 2.4 CLIFF（ECCV 2022）

原問題：自上而下流程先裁剪，裁剪後的圖看起來一樣，全局旋轉卻不同。在裁剪圖上算重投影，會用關節姿態去補償全局旋轉誤差。

做法（§1，無另編新狀態方程）：除了裁剪圖特徵，再餵入框在全圖中的位置與大小；2D 重投影改在全圖、原始相機系上計算。

觀測：全圖透視（文中舉對角 FOV 55°）。假設：框的位置足以恢復相對原始相機的全局旋轉。

對本任務：DAVIS 240×180 是固定內參、不裁剪的全圖，這一部分比手部 CNN 的緊框更接近 CLIFF 想要的條件。S37 的事件 token 帶絕對像素座標，理論上含有這項資訊。問題是 root 頭把池化後的絕對特徵回歸成**增量**，而不是相機系絕對朝向，同時 `prev_mlp` 又不看事件。CLIFF 說明：全局旋轉監督必須在原始相機平面上做；只在根對齊或 PA 空間裡訓練，會把旋轉誤差藏進姿態。

### 2.5 METRO（CVPR 2021）

原問題：用 Transformer 同時回歸網格頂點與關節。

§3 / Fig.2：CNN 影像特徵與模板網格每個頂點的 \((x_i,y_i,z_i)\) 拼接，作為 query 的位置編碼。輸入是關節 query 與頂點 query。相機參數由 encoder 輸出上的線性層得到，再做 2D 重投影損失。

觀測：單張 RGB。假設：模板座標能把頂點身份送進注意力。

代碼：`github.com/microsoft/MeshTransformer`（論文頁給出；本次未讀原始碼）。

對本任務：頂點與影像特徵在同一條序列裡，但是影像被收成一個向量再拼到每個頂點上，不是「每個事件一個節點、每條跨邊一個偏移」。輸出是絕對座標，不是對 prev 的新息。

### 2.6 Mesh Graphormer（ICCV 2021）— 方案 A 的最近鄰

原問題：自注意力抓長程，抓不好網格的局部；全局 2048 維影像向量沒有局部細節。

§3.2：每個 block 含 MHSA 與 Graph Residual Block（圖卷積殘差，沿用 GraphCMR 的網格鄰接）。

§4.1–4.2：\(7\times7\times1024\) 格點特徵收成 **49 個影像 token**；另有 **14 個關節 query 與 431 個頂點 query**。三層 encoder 的 token 數保持為 \(49+14+431\)。粗網格再線性上採樣到 SMPL。訓練是頂點、關節與 2D 投影的 L1，另加 Masked Vertex Modeling。3DPW 表同時有 MPJPE 與 PA-MPJPE（身體，不是手）。

觀測：單張 RGB 格點。假設：讓頂點自由注意全部格點，就能把局部影像用到頂點座標上。

代碼：`github.com/microsoft/MeshGraphormer`。

對本任務：這是「778 頂點與觀測在同一張圖」的已發表形態，只是觀測是 RGB 格點而不是事件，輸出是絕對 3D 而不是 \(\Delta\xi\)。S37 的 `mesh_graph` 只有 1-ring，沒有這條長程注意，和本篇的失敗模式（根旋沒有長程通路）一致。但 Graphormer 沒有「跨邊特徵 = 事件相對投影頂點的偏移」，也沒有殘差為零則輸出為零：沒有影像 token 時模型仍可從模板位置編碼與頂點自注意產生座標。把方案 A 說成 Graphormer 的手部事件版，必須多證明兩件事：跨邊在無事件時消失，以及讀出的是增量而不是絕對座標。只換 token 種類不能自動算創新。

### 2.7 Zhou 等，單目實時手（CVPR 2020）

原問題：從單張 RGB 得到可驅動 MANO 的關節角，而不是只出 3D 關節。

§3.1：DetNet 出根相對、按參考骨長歸一化的 3D 關節，以及 2D。根是中指 MCP。若已知 \(K\) 與參考骨長 \(l_{\mathrm{ref}}\)，根深度 \(z_r\) 由式 (5) 閉式解出，再回推出 \(x,y\) 平移。

§3.2 式 (6)(7)：MANO 軸角與 LBS，見 §2.1。

§3.3：IKNet 是**一次前向**，把 3D 關節映射到四元數旋轉。他們對比的是迭代模型擬合 [23, 29]，並說自己更快，因為不迭代。訓練用無噪聲 MoCap，再用 DetNet 的有噪預測做第二套配對，經 FK 用 3D 關節監督。

觀測：RGB；深度尺度用已知骨長與內參解開。假設：手指姿態可獨立插值；IK 可學。

對本任務：手部文獻裡「先根相對、再單獨恢復全局平移」很常見，全局旋轉往往留在根相對誤差裡，不單獨報告。IKNet 不是 PyMAF 式殘差回饋。閉式 \(z_r\) 依賴已知骨長與內參；事件沒有 RGB 關節熱圖，不能直接搬。

### 2.8 MobRecon、HandOccNet（方法總述）

MobRecon（CVPR 2022）§1：2D 堆疊編碼、深度可分離 spiral 卷積、MapReg / pose pooling / pose-to-vertex lifting，把像素對齊特徵抬到頂點。目標含精度、速度與時序一致性。代碼 `github.com/SeanChenxy/HandMesh`。本次未逐條核對損失與是否報告未對齊的全局旋轉。

HandOccNet（CVPR 2022）：FIT 按手與遮擋區的相關把手工特徵注入遮擋區，SET 再自注意。代碼 `github.com/namepllet/HandOccNet`。這是特徵增強，不是投影殘差，也沒有零殘差契約。

對本任務：兩者都是「把影像特徵送到手上」的加性或注意力融合。S38 已表明加性幾何會被閉環當成先驗。它們不支持方案 B。

### 2.9 H2ONet（CVPR 2023）— 少數把全局朝向拆出來的手部網路

原問題：手物遮擋下，全局朝向是病態的；把網格與朝向綁在一次回歸裡不穩。

§3.2–3.3（PDF）：先在**規範姿態**回歸頂點 \(M^c\)，朝向另走一支。旋轉用 **6D**（文中引 Zhou 等連續表示 [66]）再轉成矩陣 \(R\)。多幀時用手級遮擋 \(\Omega^h\) 挑選未遮擋幀的特徵 \(G'\)，再

\[
R_{t-ik}=f(G_{t-ik}),\quad
\Delta R=g(\mathrm{cat}[G',G_t]),\quad
R_t=\Delta R\, R_t.
\]

式 (7) 的 PDF 排版是 \(\Delta R\) 與 \(R_t\) 相乘。規範姿態與旋轉之後的網格都有 L1。他們寫明：規範損失**不包含全局旋轉**。

觀測：多幀 RGB。假設：最近的未遮擋幀能提供當前幀缺失的朝向；6D 比直接回歸矩陣穩。

對本任務：這是手部文獻裡最接近「全局旋轉單獨處理」的學習式做法。\(\Delta R\) 來自特徵拼接，不是剪影殘差，也不是槓桿臂。遮擋門 \(\Omega^h\) 是可見性修復，不是新息。若當前幀特徵為零，\(g(\mathrm{cat}[G',G_t])\) 仍可從其他幀產出非零 \(\Delta R\)。他們同時監督旋轉前與旋轉後，等於承認 PA/規範誤差會掩蓋朝向。S37 的 root 頭沒有這條分開的朝向支路，也沒有 6D/矩陣複合。

### 2.10 Deformer（ICCV 2023）

原問題：單幀在遮擋和模糊下不穩；RNN 受時間距離限制。

§3.1：空間 Transformer 出每幀緊湊特徵，時序 Transformer 出姿態 \(\hat\theta_t\) 與全序列一個形狀 \(\hat\beta\)。動態融合預測置信度 \(\hat c_t\) 與前向/後向運動。運動真值是 MANO 參數差分：

\[
d\theta_t^{fw}=\theta_{t+1}-\theta_t,\quad d\theta_t^{bw}=\theta_t-\theta_{t-1}.
\]

再把幀 \(i\) 的姿態沿這些差分加到幀 \(j\)，用置信度融合。

觀測：RGB 序列。假設：鄰幀更清楚時，可以用參數空間的運動把網格搬過來。

對本任務：運動被定義成軸角向量相減，複合也是相加。這和 SO(3) 複合不是一回事。融合權重是學出來的幀置信度，不是投影殘差，也不是事件新息。遮擋時它靠其他幀的絕對預測，不靠「本幀殘差 × 雅可比」。

### 2.11 ReFit（ICCV 2023）— 身體上的 PyMAF/擬合回饋

原問題：回歸要大量 3D；經典擬合又容易局部最小。

摘要與 §1：每步把模型關節重投影到特徵圖上開一個窗，窗被說成帶有一階資訊（例如 2D flow，即 L2 目標的導數）。每個參數組一個 GRU，更新規則分開，使一側手的對齊誤差不去推動另一側。迭代直到對齊。GRU 記憶對應一階法的動量。

觀測：RGB 特徵圖，不是幾何殘差圖。假設：重投影位置上的特徵窗足以扮演能量函數的梯度。

對本任務：這是種子裡「ReFit 類殘差回饋」的正文。手部對應物是 WiLoR 的多尺度頂點採樣，不是方案 B。窗特徵不是有符號距離；GRU 有隱藏狀態，隱藏狀態不為零時，即使窗變了，更新也不被契約強制為零。他們把「一階」當作動機，實現仍是學習的特徵到 \(\Delta\) 的映射。若要遷移，必須把窗換成 \(r\)，並讓 \(\psi(0)=0\)，否則只是又一個 PyMAF。

### 2.12 HandR2N2（ICCV 2023）

原問題：深度點雲上手部關節的迭代精化，希望推理迭代次數可與訓練不同。

輸入是**手部點雲**（深度圖乘內參），不是 RGB，也不是事件。RRU 在上一輪關節周圍採局部點，更新該關節的隱藏狀態與座標，再用通道向 GCN 傳遞運動學依賴。初始化模組給出全局初值。評測 ICVL、MSRA、NYU，指標是平均關節誤差（毫米）。

觀測：深度點雲，三維點是真實測量。假設：上一輪關節附近的點能提供位移。

對本任務：這是手部上最像「圍繞上次估計採樣、再出位移」的迴圈，但狀態是關節座標，不是相機系根旋；觀測是深度點，不是事件。不能把深度點的局部殘差當成單目事件的第二個視角。多個歷史深度假設也不等於多次獨立測量。它不回答全局旋轉。

### 2.13 Neural Voting Field（CVPR 2023）

原問題：多數方法只做根相對姿態，相機系全局根與尺度要另做第二階段。

摘要與引言：在相機視錐的三維查詢點上，MLP 回歸 (i) 到手表面的有符號距離；(ii) 對每個關節的 4D 偏移（1 維權重 + 3 維方向）。近表面點做加權投票，直接得到相機系關節。FreiHAND 上報相機系任務；HO3D 上也可做根相對。

觀測：RGB，像素對齊特徵。假設：三維密集局部證據能同時解開根與姿態。SDF 是網路預測的手表面場，不是「把 prev 網格投影成剪影再量像素距離」。

對本任務：相機系評測這件事本身稀少，值得對照。投票是 \(\sum w_i d_i\)，權重來自分類/回歸，不是槓桿臂雅可比。單目事件沒有 RGB 體素特徵，不能直接在視錐裡查詢。

### 2.14 HOISDF（CVPR 2024）

原問題：手物遮擋下，顯式點雲/網格只描述當前估計的鄰域。

摘要：先從影像特徵回歸手與物體的 SDF，再用 SDF 採樣與特徵增強去回歸姿態。SDF 的角色是給編碼器隱式形狀、編碼交互、引導採樣。DexYCB 與 HO3Dv2。代碼 `github.com/amathislab/HOISDF`。

對本任務：這裡的 SDF 是三維重建體上的形狀場，用來改善特徵。方案 B 的 \(s_i\) 是二維影像平面上、事件到 **prev 投影剪影** 的有符號距離，無學習參數。兩者同名不同物。用 HOISDF 不能證明方案 B。

### 2.15 HaMeR（CVPR 2024）

原問題：用大 ViT 與 270 萬樣本做單目手網格。

§3.1–3.2：\(f(I)=\Theta=\{\theta,\beta,\pi\}\)，\(\pi\) 是平移 \(t\in\mathbb{R}^3\)，固定內參 \(K\)，\(x=\Pi_K(X+t)\)。

§3.3：ViT-H，一個 token 交叉注意影像 token，直接回歸 \(\Theta\)。沒有迭代，沒有網格對齊回饋。

§3.4 式 (1)(2)：有 3D 標註時

\[
\mathcal{L}_{\mathrm{3D}}=\|\theta-\theta^*\|_2^2+\|\beta-\beta^*\|_2^2+\|X-X^*\|_1,
\]

另加 \(\mathcal{L}_{\mathrm{2D}}=\|x-x^*\|_1\) 與關節級對抗。\(\theta\) 的 48 維按 MANO 慣例含全局軸角，但本節**沒有**寫 6D，也沒有把全局旋轉與手指分開監督。

§5：FreiHAND/HO3D 類指標是 **PA-MPJPE、PA-MPVPE、AUC、F@5、F@15**。HInt 用投影後的 2D PCK。本次讀到的實驗節沒有未做 Procrustes 的相機系全局旋轉誤差。

代碼：`github.com/geopavlakos/hamer`，`mano_wrapper.py` 已確認包一層 MANO。頭裡的旋轉參數化本次因 API 限流沒有往下讀。

對本任務：HaMeR 是強絕對回歸器，不是跟蹤器。它的數字不能和 S37 的遞推 RA-MPJPE 比：PA 去掉了旋轉、平移與尺度。HInt 的 2D PCK 對全局旋轉敏感一些，但仍是投影誤差。沒有遞推狀態。

### 2.16 SimpleHand（CVPR 2024）

原問題：手網格解碼器裡哪些結構是必需的。

§3：token 生成器在上採樣特徵圖上按 21 個 2D 關鍵點採樣；mesh 回歸器把 token 從 21 級聯上採到 778，每級 MetaFormer。位置編碼是加性 \(X_k=X_k+emb_k\)。損失是頂點、3D 關節、2D 關節的 L1，權重 10、10、1。關節由 \(J_{3d}=J V_{3d}\) 線性回歸。**不回歸 MANO 姿態。**

§4.4 原文：「PA-MPJPE 與 PA-MPVPE 是 Procrustes 對齊之後的誤差。這兩個指標不考慮全局旋轉和尺度的影響。」FreiHAND 摘要數字是 PA-MPJPE 5.8 mm、PA-MPVPE 6.1 mm；DexYCB 同時報 MPJPE 與 MPVPE。正文沒有再定義非 PA 的 MPJPE 是否只減腕點，因此不能把它等同於本任務的 RA-MPJPE。

對本任務：作者自己寫出 PA 會掩蓋全局旋轉。他們的非 PA 指標仍是單幀、根的處理未在該段說清，且沒有時間遞推。加性位置編碼 \(emb_k\) 與證據無關，違反本任務「幾何只能乘殘差」的契約。

### 2.17 Hamba（NeurIPS 2024）

原問題：用很少的關節 token，加上圖與狀態空間，做單視圖手網格。

§3.2 式 (4)：\(f(I)=\{\theta,\beta,\pi\}\)，\(\theta\in\mathbb{R}^{48}\)，\(\pi\in\mathbb{R}^3\) 是相機平移。先由 Joints Regressor 出 \(\hat\theta,\hat\beta,\hat\pi\)，MANO 得 3D 關節，再用固定焦距 \(F_{\mathrm{focal}}=5000\) 投回 2D，在特徵圖上雙線性採樣 21 個 token。GCN 的鄰接是 MANO 關節骨架，不是 778 頂點。Mamba 掃描的是**關節空間序列**，式 (1) 的 \(t\) 是關節下標，不是時間。損失含 2D/3D 關節、姿態、形狀與對抗；\(\lambda_\theta\) 的說明寫「global orientation and hand pose」。評測仍是 PA-MPJPE、AUC、F@。附錄把 PA 寫成 Procrustes，並誤稱為 non-rigid；以正文「Procrustes」為準。

對本任務：圖是 21 關節不是 778 頂點；狀態空間是空間掃描不是時間遞推；採樣位置依賴第一次絕對回歸。沒有事件，沒有新息契約。它說明「先投影再取局部 token」在 RGB 上有用，但評測不暴露全局旋轉。

### 2.18 WiLoR（CVPR 2025）— 手上的網格對齊回饋

原問題：野外全圖裡先找到手，再重建，並希望投影對齊。

§4.1：DarkNet + PANet，錨點自由檢測，損失含框、分類與關鍵點。

§4.2：裁剪圖上，ViT 用影像 token 加 pose/shape/camera 三個 token，先回歸粗 \(\theta^c,\beta^c\) 與 \(K_{cam}=\{t_{cam},s_{cam}\}\)。然後把粗網格用弱透視投到多尺度特徵圖，對**每個頂點**採樣：

\[
\mathbf{f}^v_0=\pi(v,K_{cam}),\quad
\Delta\beta=\mathrm{MLP}_\beta(\square_v f^v_0),\quad
\Delta\theta=\mathrm{MLP}_\theta(\square_v f^v_0).
\]

\(\square\) 是 mean/max/sum 之類的聚合。損失是頂點 L1、重投影 L1、MANO 參數 L2、對抗。

評測：FreiHAND/HO3D 仍是 PA-MPJPE、F@、AUC。另報幀間 MPFVE/MPFJE、jerk，以及腕的 Root Translation Error（幀間位移，不是相對 GT 的全局旋轉角）。他們強調沒有時序模組，時序穩來自檢測穩。

對本任務：這是 PyMAF 在手上的對應，也是方案 A/B 之間的中間物：778 個投影頂點去採特徵，但採到的是外觀，聚合成一個全局向量再出 \(\Delta\theta\)。\(\Delta\theta\) 加在粗參數上（正文寫 residuals），沒有寫 SO(3) 複合。聚合 \(\square_v\) 會把不同手指的槓桿方向平均掉，和 S37 根頭把 16 個部位池化後線性讀出是同一類「先混合再回歸」。零殘差契約不成立。RTE 只看平移抖動，不評全局旋轉。

### 2.19 HaWoR（CVPR 2025）

原問題：自我中心視頻裡，手在世界座標系的運動，而不只是相機系單幀。

§3：每隻手的世界系狀態分開寫成局部姿態 \(\Theta^i_t\in\mathbb{R}^{15\times 3}\)、形狀、全局朝向 \(\Phi^i_t\in\mathbb{R}^{3}\)、根平移 \(\Gamma^i_t\in\mathbb{R}^{3}\)。

三步：(i) WiLoR 骨幹 + 時序 Image/Pose Attention，在**相機系**回歸 \(\widetilde\Phi^{c_t}\)、\(\widetilde\Gamma^{c_t}\) 與 MANO；(ii) 用手掩膜擋住 DROID-SLAM 的動態手，再用 Metric3D 在中間深度帶上對齊尺度 \(\alpha\)，式 (4) 是 German-McClure；(iii) 缺幀用 Transformer infiller。式 (5) 把相機系朝向變到規範系：

\[
\Phi^{cano,i}_t=R^{c_t 2 cano,i}\times\Phi^{c_t,i}_t.
\]

\(\Phi\) 在正文宣告為 \(\mathbb{R}^3\)。因此寫出來的乘積不是顯然的 \(R_{\mathrm{new}}=R_{\mathrm{cam}} R_{\mathrm{hand}}\)。世界系損失分別懲罰 \(\Gamma\) 與 \(\Phi\)（L1）。

觀測：自我中心 RGB 視頻 + 單目 SLAM + 度量深度網路。假設：手是動態前景，背景可用於 SLAM；度量深度在中距離可靠。

對本任務：全局旋轉被當成世界系問題，解法是「相機系絕對回歸 + 外部相機軌跡」，不是事件新息。SLAM 需要靜態背景；本任務相機固定、背景也可以產生事件，不能把 HaWoR 的相機軌跡搬進來冒充第二視角。Infiller 是整段序列的非自回歸 Transformer，不是 50 ms 遞推狀態。他們至少把 \(\Phi\) 和 15 個局部關節分開，這點 H2ONet 與 Dyn-HaMR 一致，S37 的 51 維向量 MSE 沒有這層分開。

### 2.20 Dyn-HaMR（CVPR 2025）— 重投影殘差優化根

原問題：動態相機下兩隻交互手的世界系 4D 運動。

§3 式 (1)(2)：\(q^h_t=\{\theta^h_t\in\mathbb{R}^{3\times 15},\,\beta,\,\phi^h_t\in\mathbb{R}^3,\,\tau^h_t\in\mathbb{R}^3\}\)，\(\phi\) 為軸角。形狀沿時間常數。

階段 I：現成雙手網路給相機系初值，HMP/NeMF 補全。階段 II 式 (7)(8)：

\[
\mathcal{L}_{2d}=\sum_t\sum_{h\in\{l,r\}}\rho\big(C^h_t(\tilde J^h_t-\hat J^h_t)\big),
\]

\(\tilde J=\Pi({}^w J, R_t, \omega, \tau^c, K)\)，\(\rho\) 為 Geman-McClure。實現上**先用 20 步只優化世界系根朝向與平移**，再 60 步加入局部姿態、尺度與相機外參。另有平滑、相機平滑、\(\|J\|^2\) 姿態先驗與 \(\|\beta\|^2\)。階段 III 再加 HMP 潛變量、穿刺與生物力學。優化器是 L-BFGS。

評測：局部 MPJPE/MPVPE/加速度是**根對齊之後**。全局用 128 幀片段、以前兩幀對齊的 G-MPJPE，或整段對齊的 GA-MPJPE。

對本任務：這是手部文獻裡最接近「用重投影殘差更新根」的方法。幾何通過可微投影進梯度，等價於殘差乘雅可比，但是在優化器裡，不是一個 \(\psi(0)=0\) 的小頭。殘差為零時 \(\mathcal{L}_{2d}\) 的梯度為零，**先驗與平滑項仍會推動狀態**，契約不成立。2D 觀測是 RGB 關鍵點，不是事件。他們承認弱透視不夠用於動態相機，改透視 \(\Pi(\cdot,K)\)。S37 的固定 DAVIS 內參更接近透視而不是弱透視；方案 B 若只用 2D 槓桿臂，會重演弱透視與 \(rot_z\)/平移共線的問題。G-MPJPE 用片段起點對齊，仍會吃掉一段全局旋轉，和本任務「只減腕、不對齊旋轉」不是同一指標。

### 2.21 HOLD（CVPR 2024）

原問題：單目視頻、類別無關的手與物體表面。

§3.1：手姿態用現成估計器，含 \(R_h\in SO(3)\)、\(t_h\)。物體用掩膜後的 HLoc/SfM，尺度靠接觸與 2D 重投影優化。

§3.2：手在規範空間是 SDF+顏色 MLP \(f_h(x)\mapsto(d,c)\)。觀測空間的點用逆 LBS 映回規範空間，骨變換 \(\{B_i\}\) 來自 \(\theta\) 的 FK，蒙皮權重用 MANO 頂點的 K 近鄰。訓練是體渲染。§3.3 再用交互約束精化姿態。

觀測：單目 RGB 視頻的光度與幾何。假設：一段視頻裡形狀固定，姿態已知到足以做逆 LBS；物體可被 SfM。

代碼：`github.com/zc-alexfan/hold`。

對本任務：這是 render-and-compare，但優化的是神經 SDF 與外觀，成本遠高於 50 ms 包。SDF 在三維規範空間，不是影像平面距離。光度殘差為零時，接觸與正則項仍在。不能當成在線新息頭。

### 2.22 EventHands（ICCV 2021）

原問題：事件流上的實時 3D 手姿態。

§5.1 式 (4)：LNES。窗內事件按時間順序寫入，正負極性分通道，

\[
\mathcal{I}(x_i,y_i,p_i)=\frac{t_i-t_0}{L}.
\]

同像素同極性後來的事件覆蓋先前的。實驗窗長 100 ms，重疊 99 ms，等效 1 ms 一步。

§5.2：\(\theta=[t,R,\alpha]\in\mathbb{R}^{12}\)，\(R\in\mathbb{R}^3\) 軸角，\(\alpha\in\mathbb{R}^6\) 為 MANO PCA。ResNet-18 回歸 \(\theta\)。損失是各塊 L2，\(\lambda_t=500\)，\(\lambda_R=1/3\)。假設恆定光照、背景相對相機靜止，事件來自手和前臂。

§5.3：輸出再套常速度 Kalman。低速 \(W=\omega(0.1), v=5\)；高速 \(W=\omega(3), v=1\)。噪聲不隨新息改變。

§6.1：有 3D 時報根對齊 3D-PCK/AUC（閾值 0–100 mm）。真實數據只有 2D-PCKp。沒有單獨的全局旋轉角，也沒有「不對齊旋轉」的遞推 MPJPE。

對本任務：觀測模型是短窗時間面，不是點、不是 IWE。窗內時間被壓成每像素一個標量，後事件覆蓋前事件，包內運動與姿態誤差混在一起。狀態更新是「每窗絕對回歸，再線性 KF」。KF 是真正的遞推狀態，但是在 \(\mathbb{R}^{12}\) 上，軸角被當成歐氏座標，Q/R 只按速度分兩檔，屬於常數增益。靜態背景假設在手持或自我中心時不成立；本任務相機固定，但桌面邊緣仍會產生事件。

### 2.23 EvHandPose（TPAMI 2024）— 事件手的觀測模型

原問題：事件只記錄變化，稀疏 3D 標註下如何估計手姿態。

§III-B：靜止部位在恆定光照下不產生事件。一段 \([t_{n-1},t_n]\) 的事件可對應多個絕對姿態（motion ambiguity）。他們點名相對姿態 \(\Delta\theta_{t_n}\) 的方法（EventHPE）**不能**解除這個歧義，因為同樣的事件增量仍依賴絕對姿態。

對策不是改成新息更新，而是：Conv-GRU 編碼更長歷史，每段仍預測絕對 \(\varphi=(\beta,\theta)\)。

§IV-A：邊緣用 LNES 式 (3)。網格流：相鄰 MANO 做 slerp / 線性插值，頂點投影位移式 (7)，再按重心座標鋪成稠密流式 (8)(9)。FlowNet 用這個網格流做 EPE 監督，因此流是手而不是背景。

式 (14)–(18) 對比度最大化：用網格流把事件扭到段首和段尾，IWE 的方差取負當損失。式 (19)–(22) 手邊緣損失：IWE 上累積最多的像素，在 \(12\times12\) 鄰域裡找頂點，權重是法向與相機射線的夾角 \(w^o\) 以及該頂點的投影位移 \(w^m\)，懲罰投影點到 IWE 像素的距離。平滑項懲罰參數向量的幀間 L2。推理是前向網路；這些損失只在訓練。

§VI-A：3D 序列用**根對齊 MPJPE**、PA-MPJPE、0–10 cm 的 AUC。他們給出固定姿態序列 MPJPE 19.82 mm，並引用約 20 mm 的人工標註精度。快動作只有根對齊 2D-MPJPE。

對本任務：

- 觀測上，事件是運動證據。靜止的錯誤姿態可以不產生事件。方案 B 若只用「當前剪影 vs 事件」的空間 SDF，在手幾乎不動、但 prev 姿態是錯的時候，事件可能很稀，殘差通道會空，這和本任務「乾淨樣本上解碼器仍注入 3.9–5°」是同一類現象的反面：沒有事件時不該更新，有事件時事件也不編碼絕對姿態。
- 手邊緣損失是事件文獻裡最接近方案 B 的幾何項：投影距離 × 與輪廓/法向有關的權重。它沒有槓桿臂，也不在推理時更新 \(\xi\)。
- 他們反對純 \(\Delta\theta\)。S37 的 \(\Delta=F(events)+G(prev)\) 正落在他們批評的相對更新上，而且 \(G\) 不看事件。
- Conv-GRU 是特徵遞推，輸出仍是絕對 MANO，不是「狀態 = 上一幀 MANO，更新 = 新息」。

### 2.24 Ev2Hands（3DV 2024）

原問題：單目事件、兩隻交互手，輸出每隻手的形狀、姿態與全局位姿。

§3.2 式 (2)：時間窗內同像素合成一個點 \((x,y,t_{\mathrm{avg}},P,N)\)，得 \(\mathbf{E}\in\mathbb{R}^{M\times5}\)。時間被平均，不是逐事件時間戳。

§3.3：MANO PCA \(\theta\in\mathbb{R}^6\)，\(\beta\in\mathbb{R}^{10}\)，每隻手另有 \(t\in\mathbb{R}^3\)、\(R\in\mathbb{R}^3\)。PointNet++ 出點特徵，左右手 MLP，分割 logits 當 attention 的 key，式 (3)。每隻手一個小 PointNet+MLP 回歸 \(\{\theta,\beta,R,t\}\)。

式 (5) 對 \(R\) 用 L2，對 \(t\) 用 L1。另有左右手相對關節、相對平移、錐距離場碰撞、PCA 的 Tikhonov。真實數據上去掉 3D 關節損失，改為合成相機與真實相機各自投影的 2D 損失。沒有跨窗隱藏狀態。

評測：根對齊的 R-PCK/R-AUC；RR-PCK 把整組雙手對到右手腕，用來評相對位置。沒有單獨的全局旋轉角。

對本任務：事件是點雲，MANO 頂點是解碼結果，不是圖節點。沒有「事件—頂點」跨邊。全局旋轉是每窗絕對回歸的 3 維向量，用歐氏 L2 監督。點雲把同像素事件合成一點，包內時間結構被平均掉，不利於把「prev 誤差」和「包內運動」分開（方向 1 的 C1）。分割 attention 是左右手歸屬，不是幾何新息。

### 2.25 EvRGBHand（CVPR 2024）

原問題：事件與 RGB 互補，做遮擋、過曝、運動模糊下的手網格。

§3.2：事件用最近 \(N\) 個事件堆成 LNES 式時間面，式 (2)。ConvLSTM 從兩模態特徵預測可變形卷積偏移 \(\Delta P\)，從而對齊，不估計相對位姿。互補融合後，Transformer encoder 出 token，時序注意看過去 \(S\) 步。解碼器輸入是可學習的 21 關節 token 與 195 頂點 token，上採樣到 778。損失式 (9) 是頂點 L1 與關節 L2，**不回歸全局旋轉參數**。

觀測：配準的 RGB + 事件。假設：兩模態偏移可學習；事件補運動，RGB 補外觀。

對本任務：不是純事件。頂點 token 與融合特徵做交叉注意，接近 METRO/FastMETRO，不是與原始事件建圖。ConvLSTM 與時序注意是特徵狀態，不是 MANO 狀態。沒有殘差契約。

### 2.26 EventEgoHands（ICIP 2025）

原問題：自我中心、相機在動，背景事件淹沒手。

§3：LNES 幀進 U-Net 出手掩膜，再用掩膜過濾 Event Cloud（5 維，同 Ev2Hands）。PointNet++ 後用左右手交叉注意，解碼 MANO。正文寫姿態 \(\theta\in\mathbb{R}^{15}\)、形狀 \(\beta\in\mathbb{R}^{10}\)、\(t\in\mathbb{R}^3\)、\(R\in\mathbb{R}^3\)。\(\mathbb{R}^{15}\) 與常見的 \(15\times3\) 或 PCA-6 不一致，論文沒有再解釋，此處只照錄。式 (7) 的 MANO 損失只含 \(\theta,\beta\)，不含 \(R,t\)；\(R,t\) 通過關節與頂點 L1 間接受監督。沒有時序遞推。

他們報告相對 EventHands/Ev2Hands，R-AUC、MPJPE（約 4.5 cm、43%）、MPVPE（約 2.2 cm、34%）。數據是 HOT3D 視頻經 v2e 合成的 N-HOT3D，不是真實事件閉環。

對本任務：自我中心的背景事件必須先分割，否則點雲特徵被相機運動主導。本任務相機固定，背景事件較少，但不能假設為零。掩膜是可見性/相關性修復，不是新息。絕對回歸、無遞推狀態。合成事件上的 MPJPE 不能外推到 zgz 遞推。

### 2.27 E-3DPSM（arXiv 2026-04，預印本；人體，不是手）

原問題：自我中心事件下的 3D 人體，要連續、少漂移。

§方法：每步直接回歸 \(\mathbf{P}^D_t\)，再回歸位移 \(\mathbf{P}^\Delta_t=\mathrm{MLP}_\Delta([F_t;E_{t-1}])\)。樸素融合 \(\mathbf{P}_t=\mathbf{P}^D_{t-1}+\mathbf{P}^\Delta_t\) 會漂移（式 11 與附錄 B）。他們改成 Kalman 式融合，狀態 \(X_t\in\mathbb{R}^{48}\)，

\[
X_t=A X_{t-1}+B P^\Delta_t,\quad
K_t=\Sigma H^\top(H\Sigma H^\top+R)^{-1},\quad
P_t=X_t+K_t(P^D_t-HX_t).
\]

正文寫明：\(A,B,H\) 固定為單位陣，\(\mathbf{Q},\mathbf{R}\) **訓練後在推理期為常數**，不隨幀變化。附錄試過用特徵預測 Q、R，那是額外消融。狀態是 16 個關節的 3D 位置，不是 MANO，也不是根旋李代數。

對本任務：共用脈絡已指出常數 Q、R 融合會收斂成常數增益，不能當創新。本篇正文把這點寫死了。樸素「上一狀態 + delta」會漂移，和 S37 閉環裡更新方向餘弦只有 0.05–0.11、但每步仍在動，是同一結構問題的不同表現。他們的直接分支是絕對錨，delta 分支才是運動；S37 沒有絕對分支。這支持方向 1 的 C3，不支持把 Kalman 本身當貢獻。對象是人體關節，不是手，也沒有槓桿臂。

## 3. 可觀測性、資訊來源與失效條件

對象：單目 DAVIS 類事件，50 ms 一包，非剛體手，狀態為相機系 MANO \((t,R_{\mathrm{root}},\{\theta_j\}_{j=1}^{15})\)。

資訊從哪裡來：

1. **事件只在對數亮度變化超過對比閾值時出現**（EvHandPose 式 2，Ev2Hands 式 1）。恆定光照下，靜止表面不產生事件。因此一包事件不是該時刻手的完整剪影，而是這 50 ms 裡動過的邊緣。EvHandPose 把這稱為 motion ambiguity：同一事件段可對應多個絕對姿態。
2. **全局旋轉沒有被根對齊或 PA 指標觀察到。** HaMeR、WiLoR、Hamba、SimpleHand 的主表是 PA-MPJPE。SimpleHand §4.4 寫明 PA 不含全局旋轉與尺度。EvHandPose、EventHands、Ev2Hands 的 3D 指標是根對齊 PCK/MPJPE，旋轉誤差仍在關節誤差裡，但不分解成根旋角度。H2ONet 把規範網格和旋轉後網格分開監督，因為他們認為朝向是病態的。Dyn-HaMR 的局部數字在根對齊之後，全局數字還要先做片段對齊。
3. **單目透視下，繞光軸的旋轉與橫向平移在輪廓上接近共線。** 這不是本批論文的新公式，而是 Dyn-HaMR 放棄弱透視、CLIFF 堅持在全圖上重投影的原因。本倉庫 S38 在理想輪廓上已測到 \(rot_z\) 增益約 0.2。文獻沒有一篇用二維剪影雅可比宣稱解開了手的相機系 \(rot_z\)。
4. **深度不是被第二次測量的。** NVF、HOISDF、HOLD 的 SDF 要麼是 RGB 回歸出的形狀場，要麼是視頻優化出的規範場。事件流沒有提供獨立深度。EvHandPose 的網格流用的是模型自己的相鄰姿態，不是測得的深度。把多個時間假設堆進狀態，不會變成多視角。
5. **非剛體使「剛體雅可比」只對根的瞬時運動成立。** 15 個手指在同一 50 ms 內也在動。Dyn-HaMR 因此先優化根、再優化局部，並加生物力學與穿刺。EvHandPose 用 slerp 而不是單個剛體流。方案 B 若把全部事件殘差都乘手腕槓桿臂，會把手指運動誤記到根上。
6. **時間被表示法扔掉。** LNES 每像素只留最後時間（EventHands 式 4）；Ev2Hands 把同像素事件合成一點並取平均時間。兩者都觀測不到 \(r_i\approx J_i(\delta\xi+(t_i-t_0)\dot\xi)\) 這種包內分解。EvHandPose 保留段內扭曲，但是訓練損失，推理不輸出 \(\dot\xi\)。
7. **遞推狀態幾乎不存在。** 真正把上一狀態寫進濾波器的是 EventHands 的 \(\mathbb{R}^{12}\) 常速度 KF，以及 E-3DPSM 的關節位置 Kalman。兩者的噪聲在推理期都是常數（或只按速度分檔）。HaWoR/Dyn-HaMR 是整段優化或滑窗，不是 50 ms 因果包。WiLoR、HaMeR、Hamba、Ev2Hands、EventEgoHands 是每窗絕對回歸。

失效條件（方案 A 或 B 都會碰到）：

- 包內事件少（本任務中位約 652 事件/50 ms，手指更稀）時，跨邊或 SDF 池化接近空，若頭仍有偏置或 `prev_mlp`，就會在沒有證據時更新。
- 手相對 prev 的投影幾乎不動、但絕對姿態是錯的：事件殘差可以很小，絕對誤差很大。EvHandPose 的歧義段落就是這個情況。
- 背景或前臂事件（EventHands 承認前臂會產生事件；EventEgoHands 用掩膜去掉）會把剪影 SDF 拉偏。
- 根旋繞光軸、與平移耦合時，2D 槓桿臂的列接近共線，\(M(L)\) 無法把增益做大而不把平移帶跑。

## 4. 可證偽假說

### 假說 H1（方案 B；最可能被否定）

在刪掉 `prev_mlp` 的根 6 維、旋轉改為相機系 \(R\leftarrow\exp(\omega)R_{\mathrm{prev}}\) 的前提下，用無學習的剪影 SDF 新息 \(q_j\) 與槓桿臂 \(L_j\)，

\[
\Delta\xi=\sum_j M(L_j)\,\psi(q_j),\quad \psi(0)=0,\ \psi\text{ 無偏置},
\]

能使 zgz 兩種子的遞推 RA-MPJPE 明顯低於「只做上述刪除與 SO(3) 複合、根頭仍是現在的線性層」的對照，並且擾動樣本上根旋更新與所需方向的餘弦明顯高於 S37 現在的 0.05–0.11。

最強反對證據（已有，不是猜測）：

- 資訊探針（共用脈絡 §9.2）：119 維 2D 剪影殘差的 ridge/MLP 是 7.69° / 7.08°，相對「保持」9.21° 只降約 1.5–2°；乘 3D 槓桿臂後是 7.09° / 6.96°。加上 oracle 深度才到 5.30° / 4.51°。乾淨樣本上所有解碼器仍注入 3.9–5.0°。
- S38 理想輪廓：\(rot_z\) 增益約 0.2，且與橫向平移近共線；\(rot_x/rot_y\) 約 0.5。
- 文獻裡用重投影殘差更新根的 Dyn-HaMR，同時保留先驗；他們的全局指標還要片段對齊。沒有一篇手部論文用二維剪影雅可比作為在線根更新並報告未對齊的全局旋轉。
- EvHandPose 寫明：事件段對應的是運動歧義，不是唯一絕對姿態。SDF 新息在「姿態錯但這 50 ms 幾乎沒動」時可以接近零。

### 假說 H2（方案 A；比 H1 較不容易被現有探針直接否定，但仍可能錯）

把 prev 的 778 個三維頂點與當前包的事件放在同一張圖裡，跨邊特徵只用事件相對該頂點投影的偏移（以及 \(dt\)），根的增量只從這些跨邊上讀出。無事件或無跨邊時，這條支路貢獻為零。相對 S37 現在的「路由後丟棄偏移、root 頭讀絕對池化特徵 + `prev_mlp`」，遞推 RA-MPJPE 下降，且兩種子不再像 `mesh_graph` 那樣在根旋上分裂。

反對證據：

- S37 `mesh_graph` 已失敗：778 頂點、1-ring、LBS 池化，兩種子 20.48 / 29.30，不穩在根旋，原因是沒有長程通路。方案 A 若只加短程跨邊、根仍靠局部池化，會再犯一次。
- Mesh Graphormer 的長程來自影像 token 與頂點的 MHSA，輸出是絕對座標。事件節點比 7×7 格點更稀，長程注意可能只看到手臂或桌沿。
- S38a 把幾何當加性特徵，TF 最好、閉環放大最高。方案 A 只要把頂點座標或槓桿臂拼進節點特徵，就回到那條失敗路徑。
- Hamba/WiLoR 的「投影後採樣」在 PA 指標上有效，不能推出遞推根旋會好。

## 5. 最小可遷移機制

不把方案 A 或方案 B 整包當成已經被文獻證明的模組。文獻支持的最小形式是下面這一條，而不是再加一個 GRU 或再加一項損失。

數學形式（根，6 維李代數 \(\xi=(v,\omega)\)）：

\[
q_j=\mathrm{Pool}_j(s_i,n_i,a_{ij})\in\mathbb{R}^{7},\quad
\Delta\xi=\sum_{j=0}^{16} M(L_j)\,\psi(q_j),
\]

\(M(L_j)\in\mathbb{R}^{6\times d}\) 只依賴該部位相對腕的 2D/3D 槓桿臂，\(\psi\) 無偏置且 \(\psi(0)=0\)。然後

\[
R\leftarrow\exp(\omega)\,R_{\mathrm{prev}},\quad t\leftarrow t_{\mathrm{prev}}+\text{由 }v\text{ 與 }R\text{ 決定的平移更新}.
\]

平移更新必須繞腕關節 \(J_0\) 複合，不能把 \(\omega\) 加進軸角再讓平移單獨加。`prev_mlp` 的根 6 維輸出置零。空包走現有 `ZERO_EVENT_GATE`。

這條形式同時吸收：Dyn-HaMR 的「殘差經投影幾何作用在根上」、ReFit/WiLoR 的「在投影位置取證據」、H2ONet 的「旋轉用矩陣複合而不是軸角相加」、EvHandPose 的「沒有事件就不該有運動證據」。它不吸收他們的先驗項、對抗項、常數 Q/R 或外觀特徵。

S37 接入點：

- 新息：渲染 prev 的投影剪影。倉庫已有 `semkine/kssf.py`（到遮擋輪廓的有符號距離與法向）。池化權重用現有路由 \(a_{ij}\)（`model/model.py` 的 `_route_nodes` 與 `semkine/routed_readout.py`）。
- 槓桿臂：`semkine/jacobian.py` 的解析投影雅可比，或路由時已算的 FK。只允許出現在 \(M(L)\) 里，不作為 root 頭的加性輸入。
- 讀出：根頭旁路。現有 `nn.Linear(512+16\times257\to6)` 可保留為絕對支路，但必須與 \(\Delta\xi\) 分開消融，不能一開始就加在一起（那會回到 E-3DPSM 的常數融合）。
- `prev_mlp`：`model/model.py` 裡對根的 6 維輸出置零。

成本：共用脈絡對方案 B 的估計是約 1.3k 新參數（根頭現約 27.7k），SDF 用池化近似時延遲約 +0.2–0.5 ms，**未實測**。\(\psi\) 若是 7→32→16 的無偏置 MLP，參數量在這個量級。不把未測延遲寫成結果。

方案 A 的最小形式若要做，只能是跨邊特徵 \((\Delta u,\Delta v,\Delta t)\)，消息在沒有事件鄰居時為零；禁止把頂點三維座標或 \(L_j\) 加進節點。這部分沒有現成的零殘差頭，計算上是一張含 ≤2048 事件節點與 778 頂點的圖，貴於方案 B 的 119 維池化。

## 6. 最小判別實驗

目的：把「殘差 × 槓桿臂」和可見性修復、普通時序濾波、常數增益、換表示分開。全部沿用 zgz、種子 3407/3408、遞推 RA-MPJPE。不在這裡報訓練數字。

對照（同一數據、同一事件編碼，只改根的更新）：

| 代號 | 做法 | 它代表什麼 |
|---|---|---|
| C0 | 現 S37 | 絕對池化特徵回歸增量 + `prev_mlp` |
| C1 | C0 但 `prev_mlp` 根 6 維為零，且 \(R\leftarrow\exp(\omega)R\) | 去掉均值回拉與軸角相加 |
| C2 | C1 加上按事件數或可見頂點比例縮放根增量 | 可見性修復 / 常數信任係數（舊 δ-trust） |
| C3 | C1 的根增量再套常速度 Kalman 或固定 Q、R | EventHands / E-3DPSM 式濾波 |
| C4 | C1 但根頭改讀 WiLoR 式投影點特徵的均值，不乘 \(L_j\) | 換表示的網格對齊回饋 |
| T | C1 + 第 5 節的 \(M(L)\psi(q)\)，\(\psi(0)=0\) | 待判機制 |

預期（若 H1 為真）：

- 契約：\(q=0\) 時 T 的根增量範數為 0，C4 不必為 0。
- 擾動 prev、固定事件包：T 的根旋方向餘弦高於 C0 的 0.05–0.11，且高於 C1–C3。
- 遞推 RA-MPJPE：T 優於 C1 的幅度大於 C2、C3、C4 相對 C1 的幅度。兩種子同號。

否定 H1 的條件（任一即否定）：

- T 的遞推 RA 不能優於 C1，或只和 C2/C3 落在同一條常數增益折衷上。
- \(q=0\) 時根增量不為零。
- 只在合成理想輪廓上 \(rot_z\) 增益仍約 0.2 且與平移共線，真實 zgz 上方向餘弦仍 ≤0.15。
- 增益主要來自 oracle 深度才能出現。那說明 2D SDF × 2D 槓桿臂不是資訊來源。

方案 A 的判別要另加一臂：跨邊偏移圖，但根頭不讀絕對座標池化。若它只在 PA 式根對齊誤差上變好、遞推 RA 不變，則否定「建圖能解決全局旋轉」。

## 7. 六個創新問題

1. **此前未解決的矛盾是什麼？** 單幀手網格用 PA-MPJPE 變得好（HaMeR、SimpleHand、Hamba、WiLoR），遞推且不對齊旋轉時根旋仍是主誤差（S37 診斷）。文獻要麼不評這個量，要麼用整段 SLAM/優化（HaWoR、Dyn-HaMR）而不是 50 ms 因果包。矛盾是：相機系根旋需要「當前投影與證據的差」，而現有手部網路把對齊做成特徵採樣或把指標做成 PA。

2. **新資訊從哪裡來，哪些是舊先驗？** 新資訊只可能是本包事件相對 prev 投影的空間與時間差。MANO 的 LBS、訓練均值、`prev_mlp`、對抗姿態先驗、HMP、常數 Q/R，都是歷史先驗的重複使用。H2ONet 的多幀特徵拼接是在重複使用其他時刻的絕對預測，不是新的幾何測量。

3. **事件的異步、時間或稀疏性貢獻了什麼？** 到目前的手部模型裡，貢獻大多被表示法消掉：LNES 覆蓋、點雲平均時間、固定窗 CNN。EvHandPose 的 IWE 是少數把時間用於扭曲的地方，且只在訓練。稀疏性目前被用來做分割或路由（EventEgoHands、S37），沒有變成「可觀測方向才更新」。若創新成立，時間戳必須進入殘差，而不只是 token 的一個通道。

4. **相對最近鄰，變化在觀測、關聯、狀態更新還是計算？** 相對 WiLoR/PyMAF：觀測從外觀特徵改成有符號幾何殘差，更新改成與槓桿臂相乘。相對 Dyn-HaMR：計算從 L-BFGS 整段優化改成一次線性形狀的 \(M(L)\psi(q)\)，並去掉先驗項以滿足契約。相對 EventHands/E-3DPSM：狀態更新不再是常數增益 KF。相對 Mesh Graphormer：若做方案 A，變化在關聯（事件—頂點跨邊），不在損失函數。

5. **為什麼可見性修復、普通濾波或換表示不夠？** C2 只縮放步長，不改變方向；S37 的方向餘弦已經接近 0。C3 是 EventHands 與 E-3DPSM 已經做過的常數增益，共用脈絡寫明它不是創新，舊 S7 的 δ-trust 也只從 19.26 到 18.73。C4 換到網格對齊特徵，WiLoR 已經說明這種特徵不為零，閉環會把錯誤投影再採一次樣。H2ONet 的遮擋門是可見性修復，他們仍要單獨的朝向頭，而且沒有在遞推根旋上證明契約。

6. **什麼結果會否定假說？** 見第 6 節。最直接的一條：在 C1 已經去掉 `prev_mlp` 根回拉並改成 SO(3) 複合之後，再加上 SDF×槓桿臂，遞推 RA 沒有進一步下降。

## 8. 限制與誠實聲明

- **沒有讀到全文的：** MANO 2017 原文 PDF；I2L-MeshNet、MobRecon、HandOccNet、PyMAF-X、FrankMocap、ACR、InterWild、HARP、IntagHand 的公式細節（其中 MobRecon 與 HandOccNet 只讀了方法主張）；gSDF、DIR、MeMaHand、MS-MANO、HandOS、EgoAllo、EventEgo3D 的方法後段（EventEgo3D、HandOS、EgoAllo、HOISDF 的摘要或開頭已讀）；TokenHand、MaskHand、PAD-Hand、ICCV 2025 物理手運動、Prior-aware temporal、圖頻率手形、HHMR、Tradeoff 的 PDF 已下載但未讀；EventPointMesh、EventEgoHands++、EgoEV-HandPose 未讀正文。對比度最大化原文（Gallego CVPR 2018）HTML 不完整，IWE 公式以 EvHandPose 的引用實現為準。
- **Deformer 的 venue：** 種子寫 CVPR 2023；Crossref 與 arXiv 均為 ICCV 2023。
- **E-3DPSM：** 只作為預印本引用。CVPR 2026 Open Access 標題索引中沒有這篇。
- **EvHandPose 式 (23) 的平滑損失** 在 arXiv HTML 裡有斷行，閾值的排版不完整；權重 \(\lambda_\beta=0.2\)、\(\lambda_\theta=1.0\)、\(b_{\mathrm{smooth}}=0.5\) 是讀到的。
- **EventEgoHands 的 \(\theta\in\mathbb{R}^{15}\)** 照錄，可能是記法不完整，沒有改成 \(15\times 3\)。
- **HaWoR 式 (5) 與 Dyn-HaMR 式 (4)** 把 \(\mathbb{R}^3\) 的軸角和旋轉矩陣寫成直接相乘。沒有把它們改寫成共軛 \(R_{\mathrm{cam}}R_{\mathrm{hand}}R_{\mathrm{cam}}^\top\)。
- **Hamba 附錄** 把 Procrustes 說成 non-rigid，與正文矛盾；指標解釋以正文為準。
- **代碼：** 除 HaMeR 的 `mano_wrapper.py` 開頭外，沒有審計各倉庫的旋轉實現。GitHub API 在本次會話被限流。
- **推測（已標在相應節）：** 方案 A 的長程注意能否在稀事件上穩定兩種子；2D 槓桿臂能否在真實 zgz 上超過 C1。這兩條都還沒有新實驗。
- **未做：** 沒有訓練，沒有改 S37，沒有把文獻數字填進 `tools/report_table.py` 的表。那些 PA-MPJPE 與本任務的遞推 RA-MPJPE 不是同一協議。
