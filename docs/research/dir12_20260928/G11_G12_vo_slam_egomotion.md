# G11+G12：事件／視覺里程計、SLAM 與自運動補償

> 2026-09-28。子代理只寫本檔。未改倉庫程式、配置、測試或 `outputs/`。未使用 GPU。
> 權威任務約束見 `docs/research/dir12_20260928/00_CONTEXT.md` §6–§7。
> 深讀計數：**G11（事件／視覺 VO）16 篇**，**G12（SLAM 與自運動補償）18 篇**。兩組都未湊滿 20。缺口與原因在 §0 與 §8，未把只讀到摘要或未取回全文的種子計入深讀。

## 0. 檢索記錄

日期：2026-09-28。本子代理的 `WebSearch`／`WebFetch` 不可用。arXiv HTML 搜尋與 `export.arxiv.org/api` 在連發後回 HTTP 429；Semantic Scholar、OpenAlex、Crossref 429；DBLP 逾時。下列是實際打通的來源，**不宣稱遍歷全部**。

| 檢索式／入口 | 來源 | 命中與篩選 |
|---|---|---|
| 已知 arXiv id 的 PDF（`arxiv.org/pdf/<id>`）與 abs 頁標題核對 | arXiv | 成功下載並 `pdftotext` 的 id 見 §1。篩選：標題與種子一致才進入深讀。猜錯的 id（粒子物理等）丟棄。 |
| `2108.10869` e-print TeX | arXiv | DROID-SLAM `main.tex` 方法節全文。 |
| `HKUST-Aerial-Robotics/ESVO`、`princeton-vl/DPVO`、`tum-vision/DEVO`、`tub-rip/cmax_slam`、`rmurai0610/MASt3R-SLAM`、`youmi-zym/GO-SLAM` 的 README | GitHub raw | 用來核對 arXiv id、會議與代碼位置。 |
| `git clone --depth 1 https://github.com/NAIL-HNU/ESVO2` | GitHub | 成功。HEAD `9381fe9dda5f5dd528eb63a8af2e3a49a1db3a63`。追蹤殘差與 LM 以這份代碼為準。 |
| `rpg.ifi.uzh.ch/docs/RAL16_EVO.pdf`、`RAL16_Gallego.pdf`、`CVPR18_Gallego.pdf`、`BMVC16_Rebecq.pdf`、`BMVC17_Rebecq.pdf`、`TRO18_Mueggler.pdf`、`IROS16_Kueng.pdf` | RPG 公開 PDF | 200 且首頁標題相符。`RAL17_Rebecq.pdf`、`IJCV20_Gehrig.pdf` 為 404。 |
| CVF `Peng_Globally-Optimal_Contrast_Maximisation_..._CVPR_2020_paper.pdf` | CVF Open Access | **404**。未改用記憶中的公式。 |
| `arclab-hku/EVI-SAM`、`arclab-hku/PL-EVIO`、`HKUST-Aerial-Robotics/ESVO2` 的 README | GitHub raw | 404。ESVO2 官方位置是 `NAIL-HNU/ESVO2`（由 ESVO README 指向）。 |
| `all:Canny-EVT`、`all:EVI-SAM`、`ti:EKLT` 的 arXiv API | arXiv | 空回應或 429。未取得這三篇全文。 |

覆蓋缺口（如實）：

- **Peng 等，Globally-Optimal Contrast Maximisation，CVPR 2020**：CVF 路徑 404，所試 arXiv id 均非該文。只在目錄中保留，**深讀數不含它**。
- **EKLT / EKLT-VIO**、**EVI-SAM**、**Canny-EVT**：本次未取回可核對的全文或官方 README。不編造公式。
- 未打開 IEEE Xplore 付費頁。期刊「正式發表」若 PDF 首頁或官方 README 寫了接受／DOI 才採用；否則標預印本。
- CCF 2022 目錄本次未重新下載官方 PDF。會議 A 類（CVPR／ICCV／ECCV／NeurIPS）與 TPAMI 為 A，依通行的 CCF 2022 目錄記憶並標「待與官方 PDF 复核」。TRO、RA-L、BMVC、3DV、ICRA、IROS 的分區一律標待核，不寫成已核實的中科院年份。

## 1. 候選目錄

「深讀」欄為本次是否讀到方法節並能引用原文公式。分區口徑見 §0。

### 1.1 G11 種子與擴充（事件／視覺 VO）

| 標題 | 作者／機構 | 場合 | 年 | 發表狀態 | 分區口徑 | 代碼 | 標籤 | 深讀 |
|---|---|---|---|---|---|---|---|---|
| EVO | Rebecq, Horstschaefer, Gallego, Scaramuzza／UZH | RA-L | 2017（2016-12 接受） | 正式。DOI 10.1109/LRA.2016.2645143 | RA-L，CCF A 待核（不在 A 類會議） | `uzh-rpg/rpg_dvs_evo_open` | 追蹤／建圖分離、邊緣對齊 | 是 |
| EMVS | Rebecq, Gallego, Scaramuzza／UZH | BMVC | 2016 | 正式會議論文（RPG PDF） | BMVC，CCF 待核 | EVO 倉庫沿用 | 光線計數建圖 | 是 |
| Semi-Dense 3D Reconstruction with a Stereo Event Camera | Zhou, Gallego, Rebecq, Kneip, Li, Scaramuzza | ECCV | 2018 | 正式。arXiv:1807.07429 | ECCV＝CCF 2022 A（待官方 PDF 复核） | 後續 ESVO | 時間表面、立體深度 | 是 |
| Event-based Stereo Visual Odometry | Zhou, Gallego, Shen／HKUST, TU Berlin | IEEE TRO | 2021 | 正式。DOI 10.1109/TRO.2021.3062252。arXiv:2007.15548 | TRO：CCF 2022 記憶為 B，待核 | `HKUST-Aerial-Robotics/ESVO`；實現細節見 ESVO2 同源註冊 | 追蹤對時間表面距離場 | 是 |
| ESVO2 | Niu, Zhong, Lu, Shen, Gallego, Zhou | IEEE TRO | 2025 | 官方 README 寫 T-RO 2025。arXiv:2410.09374。本次 PDF 首頁未見接受句，**以倉庫聲明為準，未核 Xplore** | 同上 TRO 待核 | `NAIL-HNU/ESVO2` @ `9381fe9` | OS-TS、IMU 先驗、後端不優化位姿 | 是 |
| IMU-Aided Event-based Stereo Visual Odometry | Niu, Zhong, Zhou | ICRA | 2024 | 官方 README 寫 ICRA 2024。arXiv:2405.04071。PDF 首頁未印接受句 | ICRA，CCF 待核 | 同上倉庫的前身敘述 | 陀螺先驗進 ESVO 追蹤 | 是 |
| Event-aided Direct Sparse Odometry (EDS) | Hidalgo-Carrió, Gallego, Scaramuzza | CVPR | 2022 | 正式。arXiv:2204.07640 | CVPR＝CCF 2022 A（待复核） | 論文寫 Ceres；本次未克隆代碼 | 事件生成模型、關鍵幀 PBA | 是 |
| Deep Event Visual Odometry (DEVO) | Klenk, Motzet, Koestler, Cremers／TUM | 3DV | 2024 | 正式會議（倉庫與 PDF 首頁均寫 3DV 2024）。arXiv:2312.09800 | 3DV 非 CCF A，待核 | `tum-vision/DEVO` | 事件塊、DBA、分數圖 | 是 |
| ES-PTAM | Ghosh, Cavinato, Gallego／TU Berlin | ECCV Workshops | 2024 | 正式研討會。DOI 10.1007/978-3-031-92460-6_5。arXiv:2408.15605 | **不是** ECCV 主會 A | `tub-rip/ES-PTAM` | 邊緣圖對齊、獨立運動失效 | 是 |
| Ultimate SLAM | Vidal, Rebecq, Horstschaefer, Scaramuzza | RA-L | 2018 | 正式。arXiv:1709.06310，DOI 前綴 10.1109/LRA.2018 | RA-L 待核 | 論文敘述，本次未克隆 | 事件＋幀＋IMU、關鍵幀窗 | 是 |
| PL-EVIO | Guan, Chen, Xie, Lu | — | 2023 | **預印本** arXiv:2209.12160v2（2023-09-26）。首頁無接受聲明，不寫成已發表 | 不適用 | 本次 README 路徑 404 | 點線殘差、IMU 扭曲事件 | 是 |
| Real-time Visual-Inertial Odometry for Event Cameras | Rebecq, Horstschaefer, Scaramuzza | BMVC | 2017 | 正式會議（RPG PDF） | BMVC 待核 | 未取代碼 | 運動補償事件幀、關鍵幀 | 是 |
| Continuous-Time Visual-Inertial Odometry for Event Cameras | Mueggler, Gallego, Rebecq, Scaramuzza | IEEE TRO | 2018 | PDF 印 accepted June 2018 | TRO 待核 | 未取代碼 | 已知地圖上的連續時間重投影 | 是 |
| Low-Latency Visual Odometry using Event-based Feature Tracks | Kueng, Mueggler, Gallego, Scaramuzza | IROS | 2016 | 正式會議（RPG PDF） | IROS 待核 | 未取代碼 | 局部 ICP、邊緣權重 | 是 |
| Direct Sparse Odometry (DSO) | Engel, Koltun, Cremers | IEEE TPAMI | 2018 | 正式。arXiv:1607.02565 | TPAMI＝CCF 2022 A（待复核） | 經典開源，本次未克隆 | 直接法、Schur | 是 |
| Deep Patch Visual Odometry (DPVO) | Teed, Lipson, Deng／Princeton | NeurIPS | 2023 | 正式（倉庫 BibTeX 寫 NeurIPS 2023）。arXiv:2208.04726 | NeurIPS＝CCF 2022 A（待复核） | `princeton-vl/DPVO` | 稀疏流殘差、置信度 BA | 是 |
| EKLT / EKLT-VIO | Gehrig 等／UZH | IJCV 一線 | 2019–2020 | 本次未取回全文 | — | `IJCV20_Gehrig.pdf` 404 | 種子，未深讀 | 否 |
| EVI-SAM | — | — | — | 本次未取回全文 | — | 所試 README 404 | 種子，未深讀 | 否 |
| Canny-EVT | — | — | — | 本次未取回全文 | — | arXiv API 無命中 | 種子，未深讀 | 否 |

### 1.2 G12 種子與擴充（SLAM 與自運動補償）

| 標題 | 作者／機構 | 場合 | 年 | 發表狀態 | 分區口徑 | 代碼 | 標籤 | 深讀 |
|---|---|---|---|---|---|---|---|---|
| Accurate Angular Velocity Estimation with an Event Camera | Gallego, Scaramuzza | RA-L | 2017（2016-12 接受） | 正式。RPG `RAL16_Gallego.pdf` | RA-L 待核 | 後續 CMax 倉庫 | 純旋轉、對比度 | 是 |
| A Unifying Contrast Maximization Framework | Gallego, Rebecq, Scaramuzza | CVPR | 2018 | 正式。RPG `CVPR18_Gallego.pdf` | CVPR＝CCF 2022 A（待复核） | `tub-rip` 後續 | IWE 方差、流／深度／運動 | 是 |
| DROID-SLAM | Teed, Deng／Princeton | NeurIPS | 2021 | 正式。arXiv:2108.10869 | NeurIPS＝CCF 2022 A（待复核） | `princeton-vl/DROID-SLAM` | DBA、置信度、前後端 | 是 |
| DPV-SLAM | Lipson, Teed, Deng | ECCV | 2024 | 倉庫 BibTeX 寫 ECCV 2024。arXiv:2408.01654。PDF 首頁未印接受句 | ECCV＝CCF 2022 A（待复核） | 與 DPVO 同倉庫 | 鄰近閉環、DBoW、Sim(3) | 是 |
| MASt3R-SLAM | Murai, Dexheimer, Davison／Imperial | 使用者指定 CVPR 2025 | 2024 預印本日期 | **本次 PDF 首頁與所讀 README 片段未印 CVPR**。arXiv:2412.12392。不把預印本寫成已發表 | 若確為 CVPR 2025 則 CCF A，待核 | `rmurai0610/MASt3R-SLAM` | 射線誤差、檢索閉環、GN | 是 |
| GO-SLAM | Zhang, Tosi, Mattoccia, Poggi／Bologna | ICCV | 2023 | PDF 印 ICCV 2023。arXiv:2309.02436 | ICCV＝CCF 2022 A（待复核） | `youmi-zym/GO-SLAM` | 共視閉環、全 BA | 是 |
| ORB-SLAM3 | Campos 等／Zaragoza | IEEE TRO | 2021 | 正式。DOI 10.1109/TRO.2021.3075644。arXiv:2007.11898 | TRO 待核 | 開源，本次未克隆 | Atlas、DBoW2、重定位 | 是 |
| VINS-Mono | Qin, Li, Shen／HKUST | IEEE TRO | 2018 | 正式期刊論文（arXiv:1708.03852；倉庫 `HKUST-Aerial-Robotics/VINS-Mono`） | TRO 待核 | 同上 | 預積分、4 DoF 位姿圖、關鍵幀視差 | 是 |
| GlORIE-SLAM | Zhang, Sandström, Zhang, Patel, Van Gool, Oswald | arXiv | 2024 | **預印本** arXiv:2403.19549。首頁未印接受句 | 不適用 | 未取 | 外部追蹤器＋地圖重錨 | 是 |
| RAFT | Teed, Deng | ECCV | 2020 | 正式。arXiv:2003.12039 | ECCV＝CCF 2022 A（待复核） | 官方倉庫，本次未逐行讀碼 | 相關體積、循環更新 | 是 |
| DUSt3R | Wang 等／Naver | CVPR | 2024 | arXiv:2312.14132 已讀方法。PDF 片段未再核對首頁會議行 | CVPR＝CCF 2022 A（待复核） | 未取 | 點圖、置信度損失 | 是 |
| iMAP | Sucar, Liu, Ortiz, Davison／Imperial | ICCV | 2021 | arXiv:2103.12352 | ICCV＝CCF 2022 A（待复核） | 未取 | 關鍵幀重放 | 是 |
| CMax-SLAM | Guo, Gallego／TU Berlin | IEEE TRO | 2024 | 正式。DOI 10.1109/TRO.2024.3378443。arXiv:2403.08119 | TRO 待核 | `tub-rip/cmax_slam` | 旋轉樣條 BA、前後端 | 是 |
| Secrets of Event-based Optical Flow | Shiba, Aoki, Gallego | ECCV | 2022 | 正式。arXiv:2207.10022。期刊擴展 TPAMI 2024 只讀到倉庫聲明，未讀 TPAMI 全文 | ECCV＝CCF 2022 A（待复核） | `tub-rip/event_based_optical_flow` | 方差目標、事件坍縮 | 是 |
| Event Collapse in Contrast Maximization Frameworks | Shiba, Aoki, Gallego | Sensors | 2022 | 正式。DOI 10.3390/s22145190。arXiv:2207.04007 | 非 CCF A | `tub-rip/event_collapse` | 坍縮是對比度的不良極大 | 是 |
| Motion-prior Contrast Maximization | Hamann, Wang, Asmanis, Chaney, Gallego, Daniilidis | ECCV | 2024 | 正式。DOI 10.1007/978-3-031-72646-0_2。arXiv:2407.10802 | ECCV＝CCF 2022 A（待复核） | `tub-rip/MotionPriorCMax` | 非線性軌跡先驗；編號損失式未完整抽出 | 是（見 §2.2 聲明） |
| Event-based Mosaicing Bundle Adjustment (EMBA) | Guo, Gallego | ECCV | 2024 | PDF 印 accepted ECCV 2024。arXiv:2409.07365 | ECCV＝CCF 2022 A（待复核） | `tub-rip/emba` | 旋轉光度 BA | 是 |
| Event-based Photometric Bundle Adjustment (EPBA) | Guo, Gallego | IEEE TPAMI | 2025 | PDF 印 accepted TPAMI 2025。DOI 10.1109/TPAMI.2025.3586497。arXiv:2412.14111 | TPAMI＝CCF 2022 A（待复核） | `tub-rip/epba` | 後端精化旋轉與全景 | 是 |
| A Fast Geometric Regularizer… Event Collapse | Shiba, Aoki, Gallego | Advanced Intelligent Systems | 2022 | 正式。DOI 10.1002/aisy.202200251。arXiv:2212.07350 | 非 CCF A | `tub-rip/event_collapse` | 只讀到動機，正則公式未抽出 | **否** |
| Globally-Optimal Contrast Maximisation | Peng, Gao, Wang, Kneip | CVPR | 2020 | 正式會議（使用者種子）。**全文未取回** | CVPR＝CCF 2022 A（待复核） | 未核 | 種子，未深讀 | 否 |

## 2. 深讀

每篇只寫本次從 PDF／TeX／代碼讀到的內容。公式用原文記號。

### 2.1 G11（16）

#### G11-1. EVO（RA-L 2017）

- **原問題**：只用事件做 6 自由度平行追蹤與建圖。
- **公式**（§III-A，式 (1)–(4)）：給定半稠密地圖投影 \(M\) 與事件圖 \(I\)，逆合成 Lucas-Kanade 求增量 \(\Delta T\)
  \[
  \sum_u \big(M(W(u;\Delta T)) - I(W(u;T))\big)^2,\quad
  W(u;T)=\pi\big(T\cdot\pi^{-1}(u,d_u)\big),\quad
  T\leftarrow T\cdot(\Delta T)^{-1}.
  \]
  雅可比用地圖梯度 \(\nabla M\) 與交互矩陣 \(W'\)（式 (4)）。建圖是 EMVS：把事件反投影成射線，在 DSI 上數交點，取高置信局部極大（§III-B）。關鍵幀：當前位姿離上一關鍵幀的距離除以平均場景深度超過約 15% 就新建局部圖（§III-C）。
- **假設**：場景邊緣靜止，相機剛體運動；事件圖抓到的是與視運動不平行的邊緣。地圖深度在追蹤時視為已知。
- **代碼**：倉庫 `uzh-rpg/rpg_dvs_evo_open` 的 README 指向該文。本次未逐行讀 C++。
- **對本任務**：這就是「把 3D 邊緣地圖投影到當前事件圖再對齊」。手的 prev 網格可以扮演 \(M\)，但 \(M\) 在 EVO 裡是靜態場景邊緣，不是會隨關節改變的表面。

#### G11-2. EMVS（BMVC 2016）

- **原問題**：已知相機運動時，從事件恢復半稠密深度。
- **公式**（§2.1–2.2）：空間掃描把邊緣像素反投影成射線，DSI 體素累計穿過的射線數。事件版對每個 \(e_k=(x_k,y_k,t_k,p_k)\) 用時刻 \(t_k\) 的位姿反投影。高射線密度處是 3D 邊緣。
- **假設**：MVS 假設位姿已知；邊緣是靜止場景結構。極性不參與投票。
- **代碼**：本次未讀。
- **對本任務**：建圖與追蹤被拆開。手沒有「場景邊緣在多視角下靜止」這一條；MANO 深度是狀態的函數，不是獨立的多視測量。

#### G11-3. 立體事件半稠密重建（ECCV 2018）

- **原問題**：立體事件相機上，用時間表面的時間一致性估逆深度。
- **公式**（§2.1–2.2，式 (1)–(4)）：
  \[
  \mathcal{T}(x,t)=\exp\big(-(t-t_{\mathrm{last}}(x))/\delta\big),\quad \delta\approx 30\,\mathrm{ms},
  \]
  \[
  \rho^\star=\arg\min_\rho C(x,\rho),\quad
  C=\frac{1}{|S_{\mathrm{RV}}|}\sum_s\big\|\tau^s_{\mathrm{left}}(x_1(\rho))-\tau^s_{\mathrm{right}}(x_2(\rho))\big\|_2^2.
  \]
  \(x_1,x_2\) 由已知位姿與外參把參考視圖的射線投到左右時間表面。補丁 \(w=25\)。
- **假設**：標定與左相機位姿已知（文中說可來自追蹤器）。極性不用。
- **代碼**：未讀；ESVO 論文寫明映射沿用這條線並改成單次立體觀測加融合。
- **對本任務**：時間表面是「最近事件的年齡場」，不是歐氏 SDF。立體是第二個真實視角。本任務是單目，不能把歷史深度假設當成另一次獨立測量。

#### G11-4. ESVO（TRO 2021）——方案 B／C 的最近鄰

- **原問題**：立體事件、實時、追蹤與建圖並行。建圖產半稠密逆深度；追蹤把該地圖對齊到當前時間表面。
- **時間表面當距離場**（§III-A 式 (1)，§V-A 式 (14)）：
  \[
  \mathcal{T}(x,t)=\exp\big(-(t-t_{\mathrm{last}}(x))/\eta\big),\quad \eta\text{ 實驗約 }30\,\mathrm{ms},
  \]
  \[
  \bar{\mathcal{T}}(x,t)=1-\mathcal{T}(x,t).
  \]
  原文：「large values … ramp on one side … cliff on the other … anisotropic distance field」。負片把當前邊緣變成小值，斜坡變成到邊緣的距離場。為加大收斂盆，對負片做 5 像素高斯模糊（§V-C）。
- **追蹤**（§V-B 式 (15)–(18)）：參考幀支撐 \(S_{F_{\mathrm{ref}}}\) 上有逆深度 \(\rho\)。只使用左時間表面。
  \[
  \theta^\star=\arg\min_\theta\sum_{x\in S}\bar{\mathcal{T}}_{\mathrm{left}}\big(W(x,\rho;\theta),k\big)^2,
  \]
  \[
  W(x,\rho;\theta)=\pi_{\mathrm{left}}\big(T(\pi_{\mathrm{ref}}^{-1}(x,\rho),G(\theta))\big).
  \]
  \(G(\theta):\mathbb{R}^6\to SE(3)\)，旋轉用 Cayley 參數 \(c\)，平移 \(t\)。正向合成 LK：每次對 \(\Delta\theta=0\) 線性化，再複合增量。穩健化：Huber + IRLS。效率：LM，每步隨機 \(N_p=300\) 個點，每批只做一次 LM 迭代，約 5 次收斂，因為初值通常靠近最優（§V-D）。
- **建圖**（§IV 式 (2)–(7)）：左右時間表面補丁殘差 \(r_i=\mathcal{T}_L-\mathcal{T}_R\)，高斯–牛頓 \(\Delta\rho=-(J^\top r)/\|J\|^2\)。深度再以 Student-t 融合。追蹤用的地圖是這條線的產物，不是網路回歸的深度。
- **假設**：剛體靜態場景；立體已整流；邊緣在 3D 中靜止，只有相機在動。式 (15) 的盆地在真值附近光滑且局部唯一（Fig. 9），作者強調初值要近。
- **代碼**：見 G11-5。ESVO 論文的 \(N_p=300\)、Huber、5 像素模糊與 ESVO2 的 `tracking_rpg_AA.yaml` 同量級。
- **對本任務**：若 prev 的 778 頂點是「地圖」，最近鄰操作是：把這些 3D 點用候選姿態投影，在距離場上讀殘差，LM 解 6 維。距離場在 ESVO 裡是**事件時間表面負片**，支撐在地圖點上。方案 B 把支撐反過來：事件是點，場是渲染剪影的 SDF。二者都是「投影結構對齊一個邊緣場」，但資料項打在不同的點集上。見 §5。

#### G11-5. ESVO2（TRO 2025，倉庫聲明）與代碼

- **原問題**：ESVO 映射太貴、追蹤在俯仰／偏航退化；要做到 VGA、CPU 實時，並用 IMU。
- **公式**（§IV，式 (10)–(12)）：追蹤目標與 ESVO 相同，寫成
  \[
  \theta^\star=\arg\min_\theta\sum_{x_i\in S}\mathcal{T}_{\mathrm{left}}\big(W(x_i,\rho_i;\theta)\big),
  \]
  其中 \(\mathcal{T}\) 在正文裡先定義了負片 \(\bar{\mathcal{T}}=1-\mathcal{T}\)，式 (10) 印刷時沒有平方。代碼實現的是最小二乘（下款）。高斯模糊會把邊緣盆地平移，造成配準偏差（Fig. 6）。OS-TS（Alg. 2）：原時間表面為 0 的一側填模糊值，非 0 一側保留原值，再取負片。IMU 預積分 \(\alpha,\beta,\gamma\)（式 (12)）只用來給式 (10) 初值；追蹤率高，位置先驗用 \((t_{i+1}-t_i)v\)，忽略加速度。
- **後端**（§V）：作者寫明，幾何法沒有顯式時空數據關聯，滑窗裡再優化相機位姿不會提高追蹤，因為局部點雲本身就是用這些位姿融合出來的，「並非」獨立約束。後端只優化線速度與 IMU 偏置，滑窗長度 5，LM。
- **代碼**（`NAIL-HNU/ESVO2` @ `9381fe9`）：
  - `esvo2_core/src/core/RegProblemLM.cpp` 的 `thread()`：重投影成功則 `ri.residual_[index] = tau1(y,x)`，`tau1` 來自 `TsObs.TS_negative_left_` 的雙線性插值；失敗則殘差設 255。
  - 同檔 `df()`：雅可比必須在零增量處計算（否則直接退出）。負片的 Sobel 梯度除以 8 作為 \(\partial \bar{\mathcal{T}}/\partial u\)。
  - Huber 分支：`residual(0) > huber_threshold` 時權重為閾值／殘差。
  - `RegProblemSolverLM::solve_analytical()`：`Eigen::LevenbergMarquardt`，每迭代 `setStochasticSampling(batch)` 後 `minimizeOneStep`，再 `addMotionUpdate`。
  - `cfg/tracking/tracking_rpg_AA.yaml`：`BATCH_SIZE: 300`，`MAX_ITERATION: 20`，`LSnorm: Huber`，`huber_threshold: 50`，`patch_size` 1×1，`kernelSize: 5`，`MAX_REGISTRATION_POINTS: 2000`，`USE_IMU: False`（該配置）。`esvo2_Tracking.cpp` 預設批量 200、迭代 10，可被 yaml 覆蓋。
- **假設**：同 ESVO，另加「前向立體、時間立體極線與基線不平行」用於映射。俯仰／偏航退化時要 IMU 初值。
- **對本任務**：三條可直接核對的事實。(1) 殘差就是投影點上的時間表面負片取值，LM 解 SE(3)。(2) 模糊距離場會偏置最優點；只在空的一側補梯度（OS-TS）才保住邊緣位置。(3) 由位姿生成的點雲不能再當成滑窗裡的獨立測量。prev 的 MANO 網格正是這種點雲。

#### G11-6. IMU-aided 立體事件 VO（ICRA 2024，倉庫聲明）

- **原問題**：ESVO 追蹤對偏航不敏感；映射只做靜態立體不完整。
- **公式**（§II-C 式 (2)–(4)）：追蹤與 ESVO 相同，
  \[
  \theta^\star=\arg\min_\theta\sum_{x\in S}\mathcal{T}_{\mathrm{left}}(W(x,\rho;\theta)).
  \]
  陀螺 \(\tilde\omega_b=\omega_b+b_g+n_g\) 的預積分 \(\gamma\) 只提供 Cayley 初值 \(c_0\)。偏置在該文中初始化後不更新。映射用自適應累積（AA）的對比度（方差）決定每個塊累積多久。
- **假設**：前向立體；陀螺偏置在一次運行中常數。
- **代碼**：未單獨克隆；敘述與 ESVO2 的預積分初值同一條線。
- **對本任務**：IMU 在這裡是追蹤的初值，不是第二套姿態觀測的常數增益融合。本任務沒有手部 IMU。若沒有近的初值，式 (2) 的非凸配準沒有這條先驗。

#### G11-7. EDS（CVPR 2022）

- **原問題**：單目 6 自由度，幀與事件在前端用事件生成模型融合，後端做光度束調整。
- **公式**（§3.1–3.3，式 (1)、(4)–(9)）：
  \[
  \Delta L(u_k,t_k)=p_k C,\qquad v(u)=J(u,d_u)\dot T,
  \]
  \[
  \Delta\hat L(u)\approx -\nabla\hat L(u)\cdot J(u,d_u)\dot T\,\Delta t,
  \]
  \[
  (\delta T^\star,\dot T^\star)=\arg\min_{\delta T,\dot T}\Big\|\frac{\Delta\hat L}{\|\Delta\hat L\|_2}-\frac{\Delta L}{\|\Delta L\|_2}\Big\|_\gamma.
  \]
  關鍵幀像素 \(u_f\) 用當前相對位姿投到事件圖再比較。後端式 (9) 是關鍵幀之間的光度誤差，Huber，滑窗 7 個關鍵幀，約 2000–8000 點，Ceres。關鍵幀條件：選中點少 20–30%，或相對旋轉超閾。
- **假設**：亮度對數變化等於極性乘對比閾值 \(C\)；輪廓處才產生事件；深度由關鍵幀維護。需要幀（DAVIS 或分光）。作者寫明純事件單目仍缺這一條，所以才引入幀。
- **代碼**：論文寫 Ceres 自動微分。本次未克隆。
- **對本任務**：殘差是「預測的亮度增量對上事件的亮度增量」，雅可比裡有 \(J(u,d)\)，也就是投影對運動的敏感度。這和方案 B 的槓桿臂同一幾何，但觀測是亮度變化而不是剪影距離。沒有幀時 \(\nabla L\) 不存在。手的運動不是單一剛體 \(\dot T\)。

#### G11-8. DEVO（3DV 2024）

- **原問題**：只用單目事件、在真實基準上做 VO，不依賴 IMU／立體／幀。
- **公式**（§3，式 (1)–(2)）：事件體素 \(E_t\in\mathbb{R}^{H\times W\times 5}\)。分數圖 \(S_t\) 選補丁。更新算子預測流修正 \(\Delta\hat f\) 與 DBA 權重 \(\omega\)。分數損失
  \[
  L_{\mathrm{score}}=\frac{1}{|E|}\sum_{(k,j)} s_k r_{kj}(1-\alpha\ln\omega_{kj})-\ln S_P.
  \]
  總損失 \(L=0.05 L_{\mathrm{score}}+0.1 L_{\mathrm{flow}}+10 L_{\mathrm{pose}}\)。推斷時關鍵幀閾值是平均光流（資料集而異，5／15／25 像素）。
- **假設**：訓練在 TartanAir 上用 ESIM 模擬事件，對比閾值隨機；場景按剛體相機運動生成光流監督。分數高的地方應是流殘差小且 DBA 權重大的地方。
- **代碼**：`tum-vision/DEVO` README 確認 DBA 與 \(\omega\)。本次未讀訓練迴圈原始碼。
- **對本任務**：學習的是「哪些事件塊值得進 BA」和「BA 裡的權重」，求解仍是可微束調整。權重隨殘差與置信一起被監督，增益不是常數。監督來自剛體場景的光流。手的非剛體運動沒有這個光流生成模型。

#### G11-9. ES-PTAM（ECCV Workshops 2024）

- **原問題**：立體事件上把 MC-EMVS 建圖和邊緣對齊追蹤接成並行系統。
- **公式**（§2.3 式 (4)）：
  \[
  \min_{T\in SE(3)}\sum_p\big(E(p;T,M)-B(p)\big)^2.
  \]
  \(E\) 是半稠密點雲 \(M\) 在候選位姿下的投影，\(B\) 是少量事件累成的二值圖。逆合成 LK。對投影圖做高斯模糊。建圖慢（文中例 5 張圖／秒），追蹤快（50–150 位姿／秒）。優化：批量 1000、2 層金字塔、最多 150 迭代，追蹤 6–20 ms（表中）。
- **假設**：靜態世界。DSEC 有軌電車佔滿視野時，「our assumption of a static world becomes invalid, causing tracking failure」。ESVO 靠重新初始化才跑完該段。
- **代碼**：`tub-rip/ES-PTAM` README 確認輸入是多相機事件，輸出位姿與置信圖。本次未讀求解原始碼。
- **對本任務**：獨立運動佔主導時，這條對齊直接失敗。手部序列的事件主要來自手自己的運動，正是這個失效條件，不是邊角情況。

#### G11-10. Ultimate SLAM（RA-L 2018）

- **原問題**：HDR 與高速下，事件、標準幀、IMU 一起做視覺 SLAM。
- **機制**（正文所讀段落）：特徵在事件幀與灰度幀上追蹤，再用關鍵幀非線性優化與 IMU 融合（風格同 OKVIS／預積分）。關鍵幀集合大小 \(M\)，另有最近 \(K\) 幀的滑窗；幀間用 IMU 傳播。優化器是 Ceres。事件窗的事件數 \(N\) 要按紋理調（四旋翼實驗用 \(N=20000\)）。關燈後灰度幀失效，事件仍能維持特徵軌跡。
- **假設**：特徵對應靜態場景點；IMU 提供幀間先驗。論文把系統定位成關鍵幀 VIO，不是純事件直接法。
- **代碼**：本次未讀。
- **對本任務**：多感測器是真的第二路測量（IMU、灰度）。本任務沒有它們。關鍵幀的作用是有界資訊，不是把 69 秒全部放進一個圖。

#### G11-11. PL-EVIO（arXiv:2209.12160v2，預印本）

- **原問題**：單目事件＋灰度＋IMU，點與線，攻擊性飛行。
- **公式**（所讀 §E）：事件點／線與灰度點的重投影殘差，以及 IMU 預積分，放進關鍵幀圖優化。事件在同一事件流裡用該窗的平均角速度與線加速度扭到第一個事件的時間。閉環用額外的事件角點＋BRIEF。線段用 LSD 打在事件表示上。
- **假設**：人造場景裡的直線；剛體相機；IMU 可用。閉環用來消累積漂移。
- **代碼**：本次未取回。
- **對本任務**：點線重投影假設 3D 結構跨關鍵幀不變。手指輪廓不是穩定的無限直線。把事件扭到窗起點，等價於用一個剛體速度解釋窗內所有事件。

#### G11-12. 事件相機的關鍵幀 VIO（BMVC 2017）

- **原問題**：事件特徵不好提，因為固定窗要麼太稀要麼運動模糊。
- **公式**（§4.1.1 式 (3)）：窗 \(W_k\) 含 \(N\) 個事件。運動補償
  \[
  x'_j=\pi\big(T_{t^f_k,t_j}(Z(x_j)\,\pi^{-1}(x_j))\big),
  \]
  增量位姿來自 IMU 積分，深度用當前路標的中位數即可。FAST 打在補償後的事件幀上，金字塔 LK 追蹤，兩點 RANSAC 剔外點。關鍵幀：特徵數過低，或與上一關鍵幀的距離／中位深度超閾。
- **假設**：短窗內 IMU 積分與一個深度足以把事件對齊到同一邊緣；場景點剛體。
- **代碼**：未讀。
- **對本任務**：補償用的是相機剛體運動。若改成用 prev 的 MANO 運動去扭事件，那是在檢驗「這組關節速度能否解釋事件」，不是從事件裡獨立測出第二個姿態。

#### G11-13. 連續時間事件 VIO（TRO 2018）

- **原問題**：地圖已知（點或線段）時，用事件與 IMU 估連續軌跡。
- **公式**（式 (13)–(19)）：事件獨立、圖像座標高斯誤差時，
  \[
  F=\frac{1}{N}\sum_k\frac{1}{\sigma_e^2}\|e_k-\hat e_k(x(t_k),M)\|^2
  +\frac{1}{M}\sum_j\frac{1}{\sigma_\omega^2}\|\omega_j-\hat\omega_j\|^2
  +\frac{1}{M}\sum_j\frac{1}{\sigma_a^2}\|a_j-\hat a_j\|^2.
  \]
  \(\hat e_k\) 是地圖元素在 \(t_k\) 的投影（點的重投影，或點到投影線段的距離）。軌跡用 B 樣條控制位姿，問題變成有限維，GN／LM。另外優化 IMU 偏置、尺度、重力方向的滾轉俯仰。
- **假設**：地圖給定；數據關聯已知；時間戳誤差可忽略。
- **代碼**：未讀。
- **對本任務**：這是「地圖已知的連續時間重投影」，和方案 C1 的時間戳殘差同構，但地圖必須獨立於待估姿態。用 prev 網格當 \(M\) 時，關聯誤差與姿態誤差綁在一起。

#### G11-14. 事件特徵軌跡的低延遲 VO（IROS 2016）

- **原問題**：DAVIS 幀上檢測特徵，用隨後的事件做低延遲追蹤，再做 VO。
- **公式**（式 (1)–(2)）：每個特徵維護模型點集與資料點集（最近事件）。
  \[
  \arg\min_{R,t}\sum b_i\|R p_i+t-m_i\|^2,
  \]
  權重 \(b_i\) 與 3×3 鄰域裡同時發生的事件數成正比。求解是加權 ICP。另用兩張局部直方圖的交檢漂移，平移在 ±3 像素內搜索。
- **假設**：兩事件之間特徵幾乎剛體（\(R,t\) 是 2D 歐氏變換）；特徵是邊緣所以鄰域會同時觸發。
- **代碼**：未讀。
- **對本任務**：這是局部、短基線、近似剛體的數據關聯。50 ms 內一根手指的外觀可以轉出特徵塊，ICP 的「模型點集不變」會斷。權重來自事件的空間聚集，是一種置信，但是 2D 剛體權重，不是 6 維根的資訊矩陣。

#### G11-15. DSO（TPAMI 2018）

- **原問題**：稀疏直接法，在圖像上最小化光度誤差，聯合優化逆深度與位姿。
- **公式**（式 (8) 附近）：光度誤差對所有幀、點、觀測求和 \(E_{\mathrm{photo}}=\sum_i\sum_{p\in P_i}\sum_{j\in\mathrm{obs}(p)} E_{pj}\)。點只有一個參數（參考幀逆深度）。Schur 補消點之後，位姿塊的稀疏性和間接法相同。作者寫明先驗會帶來偏差，所以不用平滑先驗，改為在圖像上均勻採樣。
- **假設**：光度恆定（含曝光與漸暈標定）；靜態場景；點的主幀深度在窗口內一致。
- **代碼**：未讀。EDS 的後端就是這條 PBA。
- **對本任務**：直接法的增益來自光度（或距離）殘差的正規方程，殘差為零則該點不貢獻。這滿足「證據為零則輸出貢獻為零」。DSO 的觀測模型是灰度，不是事件，也不是非剛體。

#### G11-16. DPVO（NeurIPS 2023）

- **原問題**：用稀疏補丁做實時 VO，作為 DROID 的稀疏前端。
- **公式**（§3.1 式 (4)–(6)）：補丁 \(P_k\) 含像素與逆深度。重投影 \(\omega_{ij}(T,P_k)\)。因子頭對每條邊預測 2D 修正 \(\delta_{kj}\) 與置信 \(\Sigma_{kj}\in(0,1)^2\)（sigmoid）。目標
  \[
  \sum_{(k,j)\in E}\big\|\hat\omega_{ij}(T,P_k)-[\hat P'_{kj}+\delta_{kj}]\big\|^2_{\Sigma_{kj}}.
  \]
  兩次高斯–牛頓，Schur 補，只優化位姿與逆深度。關鍵幀：最近 3 幀必留；\(t-5\) 與 \(t-3\) 的光流小於 64 px 就刪 \(t-4\)。優化窗只開最後 10 個關鍵幀的位姿。
- **假設**：補丁在源幀是正面平行小平面；訓練光流由剛體位姿與深度生成；場景可被靜態補丁軌跡解釋。
- **代碼**：`princeton-vl/DPVO`。本次讀 README 與論文，未逐行讀 CUDA BA。
- **對本任務**：置信度進馬氏距離，等價於每個殘差分量有自己的 \(W\)。低置信的流修正幾乎不拉動位姿。這是「增益隨置信變化」的顯式形式。補丁隨機取樣在稠密圖像上成立；事件圖大片是空的，DEVO 才加了分數圖。

### 2.2 G12（18）

#### G12-1. 角速度的對比度最大化（RA-L 2017）

- **原問題**：旋轉中的事件相機，不重建強度、不先算光流，直接估 \(\omega\)。
- **公式**（§III-B–E，式 (1)–(4)）：短窗事件
  \[
  I(x)=\sum_k \pm_k\delta(x-x_k),\qquad
  x'_k=W(x_k;\omega,t_k-t_0),\qquad
  I(x;\omega)=\sum_k\pm_k\delta(x-x'_k(\omega)).
  \]
  旋轉 \(R(t)=\exp(\hat\omega t)\)，每個事件的轉角是 \((t_k-t_0)\omega\)，不能所有事件共用一個角。對比度用 \(I(\omega)\) 的某種範數；正文給出 \(C_p\propto\int |I-\mu|^p\)，\(p\ge 1\)。最大化對比度，優化器是 Fletcher–Reeves 共軛梯度。Dirac 用高斯近似以便亞像素。
- **假設**：純旋轉，場景深度不出現在軌跡裡；短窗內 \(\omega\) 常數；亮度變化由邊緣相對感測器運動引起。
- **代碼**：未讀 2017 年代碼。CMax-SLAM 倉庫把本文寫成 CMax-\(\omega\)。
- **對本任務**：相機固定、手在動時，事件不是「靜態邊緣被相機掃過」。單一 \(\omega\) 解釋的是整個視野的剛體旋轉。腕關節旋轉只是手部事件的一部分。

#### G12-2. 統一對比度最大化（CVPR 2018）

- **原問題**：用同一目標處理運動、深度與光流。
- **公式**（§2，式 (1)–(3)）：局部流恆定，
  \[
  x'_k=x_k-(t_k-t_{\mathrm{ref}})\theta,\qquad
  H(x;\theta)=\sum_k b_k\delta(x-x'_k),\qquad
  f(\theta)=\sigma^2(H)=\frac{1}{N_p}\sum_{ij}(h_{ij}-\mu_H)^2.
  \]
  高斯近似的 \(\delta\) 使事件之間的影響隨扭曲後的歐氏距離衰減，數據關聯是軟的。深度是沿參考視圖射線的一維族軌跡。
- **假設**：參數維度遠小於事件數且可觀測；短時空鄰域內流恆定；照明恆定。
- **代碼**：未讀 2018 代碼。
- **對本任務**：對比度對 \(\theta\) 的峰，就是「這組運動參數把同一邊緣的事件疊在一起」。多個獨立運動（各手指）沒有一個全域 \(\theta\)。軟關聯也不提供第二個相機。

#### G12-3. DROID-SLAM（NeurIPS 2021）

- **原問題**：單目／立體／RGB-D 上，學習對應，再用可微束調整出位姿與深度。
- **公式**（TeX §3，式 (1) 與標為 `eqn:objective` 的目標）：相關體積
  \[
  C^{ij}_{u_1v_1u_2v_2}=\langle g_\theta(I_i)_{u_1v_1}, g_\theta(I_j)_{u_2v_2}\rangle.
  \]
  用當前位姿與逆深度把 \(p_i\) 投到 \(j\) 得 \(p_{ij}\)，再在相關金字塔上查找。GRU 不直接出位姿，而出流修正 \(r_{ij}\) 與置信 \(w_{ij}>0\)，以及深度塊上的阻尼 \(\lambda\)（softplus）。修正後對應 \(p^\star_{ij}=r_{ij}+p_{ij}\)。
  \[
  E(G',d')=\sum_{(i,j)\in\mathcal{E}}\big\|p^\star_{ij}-\Pi_c(G'_{ij}\circ\Pi_c^{-1}(p_i,d'_i))\big\|^2_{\Sigma_{ij}},\quad \Sigma_{ij}=\mathrm{diag}\,w_{ij}.
  \]
  高斯–牛頓，Schur 補，深度塊 \(C\) 對角可逆，\(\lambda\) 加在深度塊上。更新是 SE(3) 的指數映射。
- **系統**（§3.4）：前端對新幀與 3 個近鄰做局部 BA，線性運動模型做初值，固定前兩個位姿去掉規範自由度。後端在全部關鍵幀上重建幀圖：時間相鄰邊必加，再按光流從小到大採樣長邊，Chebyshev 距離 2 以內的鄰邊抑制。長邊就是回到舊區域時的閉環。非關鍵幀只做 motion-only BA。
- **假設**：靜態場景的共視；單目有相似變換規範（訓練時固定前兩幀真值）。大塊獨立運動會破壞對應，網路用全局池化隱藏狀態去拒絕它們，這是學習到的外點抑制，不是生成模型。
- **代碼**：公式來自 `arxiv.org/e-print/2108.10869` 的 `main.tex`。推理用自訂 CUDA 核做塊稀疏 Schur。本次未逐行讀該核。
- **對本任務**：學習殘差（流修正）與顯式求解器的介面就是 \(p^\star\) 與 \(w\)。\(w\) 小則該像素在正規方程裡的權小，增益下降。\(\lambda\) 阻止深度在弱觀測處亂走。閉環不是定時器，是共視／小光流的長邊。手部沒有「回到舊房間」這種共視。

#### G12-4. DPV-SLAM（ECCV 2024）

- **原問題**：DPVO 會漂；要在單 GPU 上加閉環與全局 BA。
- **公式**（式 (3)–(6)）：與 DPVO 相同的加權重投影目標，\(\Sigma_{ikj}=\mathrm{diag}(w_{ikj})\)。鄰近閉環：只存舊幀的補丁特徵，單向邊指向仍在記憶體裡的新幀，按相機空間鄰近插入長邊，再做全局 BA。經典閉環（DPV-SLAM++）：DBoW2 檢索，連續多幀才接受；用現成檢測器在檢索幀與時間鄰幀上三角化，再 RANSAC+Umeyama 估 Sim(3) 漂移；位姿圖
  \[
  r_i=\log_{\mathrm{Sim}(3)}(\Delta S_{i,i+1}^{-1} S_i^{-1} S_{i+1}),\quad
  r_{jk}=\log_{\mathrm{Sim}(3)}(\Delta S^{\mathrm{loop}}_{jk} S_j^{-1} S_k).
  \]
  高置信邊的門檻例子：\(w>0.5\) 才算參與優化的補丁。
- **假設**：回訪在幾何上靠近，或外觀能被 ORB 詞袋認出。尺度漂移用 Sim(3) 而不是 SE(3)。DPVO 的補丁不是可重複關鍵點，所以閉環匹配不能直接用追蹤補丁。
- **代碼**：`princeton-vl/DPVO` README 同時指向兩篇。本次未讀閉環 CUDA。
- **對本任務**：兩種觸發都依賴「再次看到同一靜態結構」。手的 69 秒是同一相機看運動的手，外觀一直在變，詞袋閉環沒有對應的正樣本定義。置信門檻說明全局 BA 只該吸收高權重因子。

#### G12-5. MASt3R-SLAM（arXiv:2412.12392）

- **原問題**：用預訓練點圖先驗做實時稠密 SLAM，並處理尺度不一致與閉環。
- **公式**（§3.2–3.5，式 (1)–(9)）：位姿用 Sim(3)。匹配先把點圖變成單位射線，對每個點用 LM 最小化射線角（10 次迭代量級）。追蹤最小化
  \[
  E_r=\sum_{m,n}\big\|\psi(\tilde X^{kk}_n)-\psi(T_{kf} X^{ff}_m)\big\|_{w(q,\sigma_r^2),\rho},
  \]
  權重 \(w(q,\sigma^2)=\sigma^2/q\) 當匹配置信 \(q>q_{\min}\)，否則無窮大（剔掉）。高斯–牛頓 \(J^\top W J\,\tau=-J^\top W r\)。關鍵幀：有效匹配數或唯一像素低於 \(\omega_k\)。閉環：ASMK 檢索，分數過 \(\omega_r\) 才解碼，匹配數過 \(\omega_l\) 才加邊。後端對所有邊的射線誤差做稀疏 Cholesky 的 GN，最多 10 次，固定第一個 7 維位姿。
- **假設**：中心相機（所有射線過一個光心）。點圖網路在靜態多視數據上訓練。純旋轉要加一項距離一致性，否則射線誤差退化。
- **代碼**：`rmurai0610/MASt3R-SLAM`。匹配與 GN 在 CUDA。本次未讀核函數。
- **對本任務**：置信進 \(W\)，低置信匹配權重為無窮即不進方程。閉環由檢索分數與匹配數觸發，不是時鐘。點圖先驗是在大量靜態場景上學的絕對幾何；手的 MANO 已經是更強的形狀先驗，再套一個場景點圖不會自動給出腕關節旋轉。

#### G12-6. GO-SLAM（ICCV 2023）

- **原問題**：神經隱式地圖會把漂移烤進幾何；要在線閉環和全 BA。
- **公式**（§3.1 式 (1)）：前端用 RAFT 式更新算子。新關鍵幀條件是與上一關鍵幀的平均光流大於 \(\tau_{\mathrm{flow}}\)。共視矩陣用反投影的平均剛體流，流大於 \(\tau_{\mathrm{co}}\) 的邊丟掉。閉環要連續三個候選且平均流低於 \(\tau_{\mathrm{co}}\)。局部圖上的目標與 DROID 相同，權重是置信對角 \(\Sigma_{ij}=\mathrm{diag}\,w_{ij}\)，阻尼 GN。全 BA 在另一線程，對歷史關鍵幀按共視再建圖。映射用哈希編碼的 SDF。
- **假設**：共視由剛體流定義；靜態場景。Fig. 3 顯示沒有閉環與全 BA 時 ATE 隨幀數累積。
- **代碼**：`youmi-zym/GO-SLAM`。本次讀 README 與論文。
- **對本任務**：他們的全局路徑是「光流小且連續出現的回訪邊」，加上後台全 BA。觸發量是幾何共視，不是 200 ms 定時。沒有回訪時這條路徑不啟動，前端仍會漂。本任務的診斷是沒有漂移、有逐包地板，所以這條路徑對症的是另一種誤差。

#### G12-7. ORB-SLAM3（TRO 2021）

- **原問題**：單目／立體／RGB-D／魚眼，視覺與視覺慣性，以及跟丟之後的多地圖。
- **機制**（§I）：三種數據關聯。短期：追蹤。中期：共視窗口裡的點，進局部 BA。長期：DBoW2，要連續三個關鍵幀命中同一區域才接受。長期匹配用於閉環、重定位、地圖合併；跟丟就重定位，重定位失敗就在 Atlas 裡開新圖。精度來自把舊觀測拉回 BA 或位姿圖，把漂移清掉。
- **假設**：同一靜態地點會以可重複的 ORB 詞再次出現。視覺慣性時漂移主要在 4 個自由度（平移與繞重力的偏航），這是 IMU 可觀測性，不是單目事件的可觀測性。
- **代碼**：未讀。
- **對本任務**：重定位的定義是「跟丟了，再認回舊地圖」。S37 的失敗模式是每步都給出一個帶偏的根旋轉，軌跡並沒有發散到跟丟。詞袋也沒有手部姿態的地點語義。

#### G12-8. VINS-Mono（TRO 2018）

- **原問題**：單目加 IMU 的完整系統：里程計、閉環、重定位、全局一致。
- **機制**（所讀 §）：IMU 預積分在流形上，偏置可後驗修正。關鍵幀兩條件之一是與上一關鍵幀的平均視差超閾。單目 VINS 的漂移在 4 DoF，全局位姿圖只優化這 4 維。重定位把回環幀的特徵和滑窗緊耦合，而不是只做鬆耦合的位姿圖。
- **假設**：激勵足夠時尺度、滾轉、俯仰可觀；偏航與位置會漂，要靠回環。
- **代碼**：`HKUST-Aerial-Robotics/VINS-Mono`（論文腳註）。本次未讀。
- **對本任務**：視差觸發關鍵幀，是「資訊有增加才加節點」。4 DoF 全局圖的前提是 IMU 已經釘死滾轉俯仰與尺度。本任務沒有這個釘。把 VINS 的位姿圖搬過來，優化的自由度與手的根旋轉不是同一組。

#### G12-9. GlORIE-SLAM（arXiv:2403.19549，預印本）

- **原問題**：RGB-only 稠密 SLAM 裡，神經點雲在 BA／閉環更新位姿後要跟著變形。
- **公式**（式 (1)–(6)）：點存錨定幀、像素、深度。位姿更新後
  \[
  p'_i=\omega'_{k_i} D'_i K^{-1}[u_i,v_i,1]^\top.
  \]
  沒有新深度時用最小二乘縮放 \(D'=sD\)。代理深度用多視一致性過濾。追蹤器是帶閉環和在線全局 BA 的光流系統（文中指向 DROID 一類），映射吃它的位姿與噪聲深度。
- **假設**：閉環給的是剛體位姿更新；場景靜態所以舊點可以剛體重錨。
- **代碼**：未取。
- **對本任務**：地圖點必須能按更新後的剛體位姿重錨。MANO 頂點還依賴 15 個關節，只重錨根部會把手指誤差留在「地圖」裡，下一幀再當成真值邊緣。

#### G12-10. RAFT（ECCV 2020）

- **原問題**：光流不要粗到細的金字塔，而要在全解析度上迭代更新。
- **機制**（§1 與所讀相關段落）：所有像素對的相關體積，池化成多尺度，GRU 更新算子反覆讀取相關並修正單一高解析度流場。權重在迭代間共享，所以可以跑很多步而不必加層。相關體積不隨著當前流扭曲，查找是在固定體積上取局部格點。
- **假設**：兩幀之間的外觀相關能表示對應；訓練用合成光流。
- **代碼**：未讀。DROID／GO-SLAM 寫明更新算子來自 RAFT。
- **對本任務**：DROID 的 \(w\) 與流修正是這個迭代相關查找的輸出。相關是學習的外觀相似度，不是幾何殘差。事件沒有灰度外觀，DEVO 改用體素。空事件區域的相關沒有定義良好的峰。

#### G12-11. DUSt3R（arXiv:2312.14132）

- **原問題**：一對圖像直接回歸兩個在同一座標系的點圖與置信圖。
- **公式**（式 (2)–(3)）：回歸損失是歸一化點圖的歐氏距離。置信度感知的訓練讓不確定像素（天空、半透明）權重下降。焦距由置信加權的閉式／Weiszfeld 迭代從點圖恢復。相對位姿可用 Procrustes。
- **假設**：兩幅靜態圖像共視；尺度在一對點圖內部對齊。
- **代碼**：未讀。MASt3R-SLAM 的前端是它的後繼。
- **對本任務**：置信圖是網路輸出，用來給後續幾何求解加權。它不創造第二個視角。手的單目事件包沒有第二幅圖像。

#### G12-12. iMAP（ICCV 2021）

- **原問題**：用一個神經隱式地圖做 RGB-D SLAM，跟蹤與建圖共用該表示。
- **機制**（所讀段落）：關鍵幀集合隨場景增長；損失引導的隨機重放，讓地圖更新時重新看到舊區域。光度損失加地圖網路的插值。跟蹤與自動關鍵幀選擇並行。
- **假設**：RGB-D；關鍵幀集合代表已見靜態表面。
- **代碼**：未讀。
- **對本任務**：重放舊關鍵幀是為了對抗遺忘，不是為了觀測新事件。本任務每 50 ms 都有新事件，缺的是這些事件對根旋轉的資訊，不是一張會遺忘的神經地圖。

#### G12-13. CMax-SLAM（TRO 2024）

- **原問題**：純旋轉事件相機的長時軌跡與全景，用對比度最大化做束調整，而不是每包獨立估 \(\omega\)。
- **公式**（§II–III，式 (3)、(8)、(10)–(13)）：Gallego 等 2017 的局部形式是
  \[
  \omega^\star=\arg\max_\omega\mathrm{Var}\big(I(\omega,E)\big).
  \]
  本文把軌跡寫成線性或三次 B 樣條 \(R(t)\)，
  \[
  \arg\max_{R(t)} f\big(I(R(t);E)\big),\qquad X'_k=R(t_k)X_k,
  \]
  全景 IWE 由所有事件按自己的時間投影累加。控制位姿少於事件數；樣條局部支撐。前端產局部運動，後端精化控制位姿。附錄主張旋轉模型不會事件坍縮；正文寫 6 自由度 CMax 目標「似乎不適定」（ill-posed），所以系統留在 SO(3)。
- **假設**：相機相對靜態場景做旋轉；照明恆定；不需要對比閾值 \(C\)，也不用極性。
- **代碼**：`tub-rip/cmax_slam`。本次讀 README 與論文，未讀樣條導數實現。
- **對本任務**：全局路徑在這裡是「低頻的旋轉樣條 BA」，前端是短窗對比度。觸發是滑窗前移，不是地點識別。模型類被限制在純旋轉，因為 6 自由度會坍縮。手的根是 6 維（平移加旋轉）再加 45 維關節，比他們主動避開的 6 自由度更大。

#### G12-14. Secrets of Event-based Optical Flow（ECCV 2022）

- **原問題**：對比度最大化估稠密流時會過擬合、遮擋、收斂差。
- **公式**（式 (3)、(9)）：IWE 方差
  \[
  \mathrm{Var}(I(x;\theta))=\frac{1}{|\Omega|}\int(I-\mu_I)^2\,dx.
  \]
  他們改用梯度幅值作為對比，並加全變分，
  \[
  \theta^\star=\arg\min_\theta\big(1/f(\theta)+\lambda R(\theta)\big).
  \]
  多參考時間：只在窗的一端尖銳、在另一端糊掉的流被排除。
- **假設**：同一邊緣的事件應在任何參考時間都對齊；流沿流線大約恆定。
- **代碼**：`tub-rip/event_based_optical_flow`。本次未跑。
- **對本任務**：方差最大不一定是對的運動。不良極大存在。加正則是在承認目標本身會把事件疊到錯誤的尖峰上。

#### G12-15. Event Collapse（Sensors 2022）

- **原問題**：對比度最大化的損失面上有一種退化極值。
- **機制**（§2.1 與 Fig. 1）：迭代地扭曲事件再計算對齊目標。坍縮是事件被扭進少數像素或線，方差仍然很大，但不是場景邊緣。Fig. 1 用 \(h_z\) 掃描方差損失，標出坍縮 IWE 與期望 IWE。
- **假設**：討論的是 CMax 的目標幾何，不限於某一數據集。
- **代碼**：`tub-rip/event_collapse`。
- **對本任務**：若用對比度去估腕部旋轉，優化可以停在「所有手部事件疊成一條」的解上，該解的對比度高，旋轉卻是錯的。

#### G12-16. Motion-prior CMax（ECCV 2024）

- **原問題**：0.1–0.3 s 的稠密連續時間運動，線性扭曲不夠，純對比度會進坍縮。
- **機制**（摘要與 §1）：損失把對比度最大化與像素級非線性軌跡先驗合在一起。先驗用來避開事件坍縮。自監督，在 DSEC 上與 EV-FlowNet 一類 U-Net 比較。作者寫明先前基準的時間間隔多在 0.022–0.1 s，且運動較均一。
- **聲明**：編號損失式在本次抽出的頁面裡沒有完整出現。下面不引用具體係數。
- **假設**：運動場比獨立的每事件軌跡低維，可以用網路先驗約束。
- **代碼**：`tub-rip/MotionPriorCMax`。
- **對本任務**：50 ms 已經落在他們認為「短、運動較單一」的區間邊緣。先驗若是剛體或平滑流場，手指的獨立運動會被當成應被抑制的坍縮或外點。

#### G12-17. EMBA（ECCV 2024）

- **原問題**：旋轉事件相機上，同時精化軌跡與全景梯度圖。
- **公式**（所讀首節）：用線性化事件生成模型，把光度束調整寫成相機運動與全景梯度圖的正則化非線性最小二乘。事件稀疏，所以只更新被觀測到的像素，圖是半稠密梯度。線性化模型給出塊對角稀疏，用來做高效求解。需要前端提供初始軌跡。
- **假設**：旋轉為主（全景、星追蹤、衛星）。靜態場景。
- **代碼**：`tub-rip/emba`。本次未讀求解器。
- **對本任務**：後端精化的是旋轉軌跡加一張靜態全景。手沒有靜態全景。初始軌跡若已把根旋轉估錯，光度 BA 會在錯誤關聯上把地圖一併改壞。

#### G12-18. EPBA（TPAMI 2025）

- **原問題**：把旋轉軌跡與全景強度一起當後端變數。
- **機制**（Fig. 2 與所讀段落）：前端產出 \(R(t)\) 與全景 \(M\)，EPBA 聯合精化二者。難點是噪聲與數據關聯：每個事件資訊少，要判斷哪些事件來自同一場景點。應用場景仍然是旋轉主導。
- **假設**：同 EMBA。全文後段的正規方程本次沒有逐式抄出，不補係數。
- **代碼**：`tub-rip/epba`。
- **對本任務**：數據關聯是後端能不能幫前端的前提。prev 網格與當前事件的關聯若錯在根旋轉，聯合優化沒有額外觀測可用來拆開這兩個變數。

## 3. 可觀測性、資訊來源與失效條件

對象：單目事件，240×180，50 ms 一包，狀態是相機系 MANO 51 維，非剛體手。下面每條都對應已讀機制。

1. **根旋轉的圖像測量是一條輪廓的運動，不是一個剛體場景的重投影。** ESVO／EVO／ES-PTAM 的資訊來自「靜態 3D 邊緣在已知深度下的投影是否落在當前邊緣場上」。手的邊緣場同時包含腕的剛體運動和 15 個關節的運動。一個 6 維 \(\theta\) 的正規方程把這些來源加在一起。ES-PTAM 在獨立運動佔滿視野時追蹤失敗，是同一條件。

2. **深度不是新測量。** ESVO 的 \(\rho\) 來自立體時間一致性，和追蹤位姿是兩條感測路徑（左右事件）。本任務的頂點深度來自 prev 的 FK。ESVO2 §V 寫明：這種點雲再放進位姿優化不構成額外約束。連續時間 VIO（TRO 2018）把地圖 \(M\) 當成給定。把 prev 網格叫做「地圖」時，必須把它當成帶誤差的初值，而不是第二隻眼睛。

3. **時間表面是各向異性年齡場，不是剪影 SDF。** 式 (1) 與負片 (14) 只在邊緣的一側有斜坡。ESVO 用 5 像素模糊換盆地寬度，ESVO2 Fig. 6 顯示模糊把盆地平移。OS-TS 只填零側。方案 B 的有符號距離是渲染輪廓的歐氏場，兩側都有梯度，零水平集是模型輪廓而不是事件邊緣。兩者的雅可比 \(\nabla\mathrm{field}\cdot J_{\mathrm{proj}}\) 形式相同，零水平集的物理意義不同。

4. **短窗剛體旋轉在 CMax 裡可觀測，是因為軌跡不依賴深度。** RA-L 2017 與 CMax-SLAM 都把模型限制在 SO(3)。CMax-SLAM 寫 6 自由度對比度目標不適定，Secrets 與 Event Collapse 給出不良極大。根狀態有平移。S38 在理想輪廓上已經看到 \(r_z\) 與橫向平移近共線。對比度最大化不會自動補上這條幾何退化。

5. **50 ms 與事件密度。** 半稠密論文的時間常數 \(\delta\approx 30\) ms，與包長同量級。包內運動和 prev 誤差在時間盲的池化裡混在一起（上下文 C1）。EDS 與 BMVC 2017 用速度項或 IMU 積分把窗內運動補償掉，補償模型是單一剛體。手指在 50 ms 內的位移相對腕可以不小，補償殘差會被當成根的新息。

6. **置信度改變的是正規方程的權，不創造秩。** DROID 的 \(\Sigma=\mathrm{diag}\,w\)、DPVO 的 \(\Sigma_{kj}\)、MASt3R-SLAM 的 \(w(q,\sigma^2)\)、DSO 的 Huber，都是 \(J^\top W J\) 裡的 \(W\)。\(W\to 0\) 的方向不更新。若輪廓對 \(r_z\) 與 \(t_y\) 的列近共線，再大的 \(W\) 也只是在這條退化子空間裡分配步長。DEVO 的分數圖決定哪些塊進入 BA，空的事件區域不應拿高權；手指事件中位很少，高權區域可能根本不覆蓋能分開根旋轉的槓桿。

7. **漂移與閉環。** GO-SLAM Fig. 3、DPV-SLAM、ORB-SLAM3、VINS-Mono 的全局層解決的是累積誤差，觸發是回訪或視差。上下文 §3 寫明 69 s 內無漂移，閉環誤差等於逐包重估的 13–17 mm。閉環沒有被積出來的誤差可清。`semkine/anchor.py` 的 `outside_silhouette_fraction` 註釋寫明：該信號對 1 cm 平移仍可為 0，要到數厘米才上升。它抓的是大位移，不是數度的根旋轉地板。

8. **空包與契約。** ESVO 在事件太少時時間表面沒有邊緣。本任務已有 `ZERO_EVENT_GATE`。任何從 ESVO 搬來的殘差都必須在場或事件為空時給出零新息，否則會重新引入 `prev_mlp` 那種與證據無關的回拉。

## 4. 可證偽假說

### H1. 時間表面（或剪影場）上的加權 LM 能給根一個隨證據變化的增益

**陳述。** 把 prev 的前表面頂點當作 ESVO 的 \(S_{F_{\mathrm{ref}}}\)。殘差取投影位置上的距離場（時間表面負片，或渲染剪影 SDF，兩種要分開做）。用現有 `semkine/jacobian.py` 的投影雅可比組成 \(J\)。更新
\[
\Delta\xi=(J^\top W J+\Lambda)^{-1} J^\top W r,
\]
\(W\) 由事件是否落在場的斜坡上、以及覆蓋決定，\(\Lambda\) 只加在近共線的方向上。\(r=0\) 時 \(\Delta\xi=0\)。根的 `prev_mlp` 行保持為零。

**目前最強的反對。** (1) ESVO §V-D：5 次 LM 夠用是因為初值靠近最優；TF 上根旋轉誤差已是 6–8°，而 50 ms 真實轉動只有 1.6–3.0°（上下文 §3）。(2) ES-PTAM：獨立運動佔滿視野時靜態地圖對齊失敗。(3) ESVO2 §V：由位姿生成的點雲再參與位姿優化不增加約束；深度來自同一 prev。(4) 舊線常數 \(\delta\)-trust 0.5 已把可實現增益從 19.26 mm 收到 18.73 mm（上下文 §4）。倉庫裡 `semkine/gn.py` 的 `lm_solve` 與 `GNLMRefiner` 已經是 KSSF 殘差上的阻尼法方程。若 H1 只是把這次 LM 再跑一遍剪影 SDF，它不是新機制。

### H2. 方向 2 的全局路徑應當是 SLAM 式閉環，定時或回訪時重錨，用來消除長序列漂移

**陳述。** 每 200–300 ms，或在回到相似事件外觀時，跑一次更長窗口的絕對估計並覆蓋局部追蹤。

**目前最強的反對，也是更可能被否定的一條。** 上下文 §3：69 s 內沒有漂移，誤差是逐包重估地板。ORB-SLAM3／DPV-SLAM／GO-SLAM 的觸發是舊地點再次被看到。手部序列沒有這個事件。ESVO2 的後端故意不優化位姿。`AnchorTrigger` 的輪廓外比例對小誤差不敏感。定時重錨若用的仍是同一個退化殘差，只是把地板再估一次。

H2 若改成「僅在 LM 殘差不降、且輪廓外事件比例持續高時，用多個時間戳重解一次」，那是另一條假說（下面的最小實驗裡的對照），不再是 SLAM 閉環。

## 5. 最小可遷移機制

只遷移同時滿足三條的部分：殘差乘幾何、證據為零則貢獻為零、不把剛體場景地圖當成獨立測量。

**數學。** 記前表面頂點 \(X_j\)（prev 的 FK，深度固定在這一次線性化裡）。候選根增量 \(\xi\in\mathbb{R}^6\)。
\[
u_j(\xi)=\pi\big(\exp(\xi^\wedge) X_j\big),\qquad
r_j=\phi\big(u_j(\xi)\big).
\]
\(\phi\) 二選一，實驗必須拆開：

- \(\phi_{\mathrm{TS}}\)：當前包時間表面負片在 \(u_j\) 的取值（ESVO 式 (15)，代碼 `TS_negative_left_`）。
- \(\phi_{\mathrm{SDF}}\)：事件位置上的剪影有符號距離（方案 B；`KSSF`／`outside_silhouette_fraction` 已有場）。

\[
J_j=\nabla\phi\cdot \frac{\partial\pi}{\partial X}\,[X_j]_\times\text{ 與平移列},\qquad
\Delta\xi=(J^\top W J+\Lambda)^{-1} J^\top W r.
\]
\(W\) 為對角。事件落在場的平坦區（時間表面的懸崖外側，或 SDF 截斷之外）時對應權為 0。\(\Lambda\) 用 `semkine/jacobian.py` 的 `fisher`／`schur_eliminate` 看哪些根方向在該包的 \(J^\top W J\) 裡近零，那些方向步長為 0。旋轉更新寫在相機系，\(R\leftarrow\exp(\omega^\wedge)R_{\mathrm{prev}}\)，與方案 B 一致。

不遷移：立體匹配、DBoW2、Sim(3) 回環、全景 IWE、對比度最大化的無約束 \(\arg\max\mathrm{Var}\)、把點雲與位姿同時放進滑窗（ESVO2 已否定）、`prev_mlp` 的根部加性項。

**接入點。**

- 頂點與投影：`model/model.py` 的 `forward_packet` 在路由之後、`_decode_active` 之前。根頭現在是 `root_head` 的線性層，讀絕對特徵。
- 場：剪影距離用 `semkine/kssf.py` 的 `KSSF`；像素鄰域可用 `semkine/mesh_graph.py` 的 `nearest_node_lut`。時間表面需要在包內按事件時間戳累積，現有路由把偏移丟掉了，這一步是新的、但是無參數的。
- 雅可比：`semkine/jacobian.py` 的 `query_jacobian`／`project_jacobian`。
- 求解：`semkine/gn.py` 的 `lm_solve` 已經是 \((J^\top J+\lambda I)\) 在保留座標上的最小二乘。要改的是把標量阻尼換成對角 \(W\) 與近零方向的 \(\Lambda\)，而不是再寫一個求解器。
- 觸發：`semkine/anchor.py` 的 `AnchorTrigger` 與 `outside_silhouette_fraction`。只在 H1 的局部 LM 不下降時才允許第二次、更長窗的求解。不要接 ORB 重定位。
- 訓練：根的 `prev_mlp` 輸出 6 維置零。\(W\) 若學習，只能乘殘差，零殘差時輸出為零。不要新增一條與 \(r\) 相加的頭。

**成本量級。** ESVO2 追蹤配置是 300 點、最多 20 次 LM，YAML 裡追蹤率 100 Hz 的設計目標。本任務 50 ms 預算、6×6 的正規方程，算術量遠小於現有 0.827 GMAC 的 `forward_packet`。延遲的風險在時間表面的散射累加與 KSSF，不在 6×6 求解。參數：若 \(W\) 不用網路，新增參數為 0。若 \(W\) 用一個共享的無偏置小 MLP，參數量級是幾千以內，仍遠小於根頭現有的 2.7 萬。這些是結構估計，不是實測。

## 6. 最小判別實驗

不要先訓練完整模型。在 zgz 兩條序列、種子 3407／3408 的現成 prev 上做推理期探針。主指標仍只走 `tools/report_table.py`；下面的分項只寫進本研究目錄，不進報告表。

| 臂 | 做什麼 | 預期若 H1 成立 | 否定 H1 |
|---|---|---|---|
| T0 | 現有 S37 遞推 | 基線 RA 20.74（19.23／22.26） | — |
| T1 | 常數 \(\delta\)-trust（舊線已測的那種縮放） | 只吃掉一點增益 | — |
| T2 | `lm_solve` + 剪影 SDF，\(W=I\)，固定阻尼，3–5 次 | 根旋轉步長與殘差同號，空場時步長為 0 | 根的 RA 分量不優於 T1 |
| T3 | T2 但 \(W\) 只在 \(\lvert\phi\rvert\) 處於斜坡內時為正 | 平坦區不再拉根 | 與 T2 無差異 |
| T4 | 殘差改為時間表面負片在投影頂點上的值，其餘同 T3 | 與 T2 不同，且 OS-TS 版本的偏差小於全圖模糊 | 兩者都不優於 T1，或模糊版與 OS-TS 版相同 |
| T5 | 關節凍結為該幀真值，只估根 | 若失敗主因是非剛體，T5 應明顯好於 T2 | T5 仍不優於 T1，則場本身對根不可觀，與關節無關 |
| G0 | 每 4 包定時用同一殘差重解並覆蓋 | 與逐包 T3 幾乎相同 | 若 G0 明顯優於 T3，才支持「全局重估」而不是閉環 |
| G1 | 僅當 LM 殘差不降且 `outside_silhouette_fraction` 超過 `AnchorTrigger.outside_on` 時重解 | 大誤差段被拉回，小誤差段不被打擾 | 觸發次數接近 0，或觸發後根旋轉地板不變 |

否定 H2 的讀法：G0／G1 相對 T3 沒有額外收益。這與「69 s 無漂移」一致，實驗只是把它和 SLAM 閉環分開。

否定「只換表示」：T4 若只把 SDF 換成時間表面、求解器與 \(W\) 不變，卻得到與 T2 相同的根誤差，則表示本身不是資訊來源。否定「只加 LM」：T2 不優於 T1。

## 7. 六個創新問題

1. **此前未解決的矛盾。** S37 的根更新是絕對影像特徵的增量，增益不隨 prev 與事件的分歧變化；`prev_mlp` 在根上是與事件無關的收縮。ESVO 的 LM 與 DROID 的 \(J^\top W J\) 把增益放進殘差的正規方程。矛盾在於：這套增益依賴靜態剛體地圖，而手的地圖會隨關節變。

2. **新資訊從哪裡來。** 時間表面負片的斜坡方向是事件時間戳相對邊緣的一側，剪影 SDF 沒有這側資訊。立體、IMU、第二幀是文獻裡的真資訊，本任務沒有。MANO 深度、上一幀位姿、詞袋、場景點圖先驗都是歷史狀態或別的數據域的先驗，不是這一包事件的新測量。

3. **事件的異步與稀疏貢獻了什麼。** ESVO 的 \(\eta\approx 30\) ms 與 50 ms 包長同階，時間表面把「多早以前的邊緣」編成空間梯度。BMVC 2017 與 TRO 2018 用每個事件自己的時間戳做補償或重投影，從而把窗內運動和參考時刻的位姿分開。稀疏的後果是 DEVO 必須選補丁：大片像素沒有事件，不該進 \(W\)。

4. **相對最近鄰，變化在哪一層。** 相對 ESVO，觀測場可以換成剪影 SDF，關聯從「地圖點對場」換成「事件對場」或維持地圖點對場，狀態更新從 SE(3) 相機變成相機系根增量，求解器仍是 LM。相對 DROID，沒有學習的稠密流，\(W\) 若保留也只是殘差權。相對 CMax，不做無約束的方差最大化。相對 ORB-SLAM3，沒有重定位。

5. **為何可見性修復、普通濾波、換表示不夠。** 可見性只改哪些頂點進入路由，不改變「增量頭的增益是常數」。普通時序濾波假設誤差是可積的漂移；診斷是逐包地板。把特徵從座標換成 SDF 或時間表面，若後面仍是線性頭或固定 \(\delta\)-trust，增益仍不隨殘差範數與 \(J\) 的秩變化。`lm_solve` 已在倉庫裡，所以「加上 LM」本身要和 T1 比。

6. **什麼結果直接否定。** T2 與 T4 的根旋轉都不優於常數 trust；或 T5 把關節凍成真值後仍然不優於 T1。任一條成立，H1 失敗。G0 不優於 T3，則定時全局路徑失敗。

## 8. 限制與誠實聲明

- G11 深讀 16，G12 深讀 18。未湊滿 20。EKLT、EVI-SAM、Canny-EVT、Peng 等 CVPR 2020 的全局最優對比度最大化，全文沒有取回，目錄裡標為未深讀，正文沒有他們的公式。
- Motion-prior CMax 的編號損失、EPBA 的正規方程係數、Ultimate SLAM 的完整代價函式，沒有從 PDF 裡完整抄出。相應段落只寫讀到的機制。
- MASt3R-SLAM、DPV-SLAM、ESVO2、IMU-aided、PL-EVIO 的「正式發表」只在來源寫明時採用。PL-EVIO 按預印本處理。MASt3R-SLAM 不寫成已發表的 CVPR 論文。
- CCF／中科院分區除「CVPR、ICCV、ECCV、NeurIPS、TPAMI 在 CCF 2022 中為 A」這一條通行事實外，TRO、RA-L、BMVC、3DV、ICRA、IROS 均為待核。本次沒有下載 CCF 官方 PDF。
- DROID 的公式來自 arXiv TeX，不是來自 CUDA 原始碼逐行核對。ESVO 追蹤的代碼證據來自 ESVO2 倉庫 `9381fe9`，不是 2021 年 ESVO 倉庫的逐行diff。論文式 (15) 有平方，ESVO2 式 (10) 印刷無平方，代碼是負片取值的最小二乘。三處不一致已在 G11-4／G11-5 寫明。
- 延遲與參數是結構估計，沒有在本機計時。
- 沒有跑 zgz，沒有改任何實驗輸出。§6 是實驗設計，不是結果。
