# G06+G07：事件雙目與單目事件深度

> 2026-09-28。研究子代理交付。只讀文獻與 S37 預註冊，未改程式、配置或 `outputs/`，未用 GPU。
> 權威數字：S37 預註冊 `docs/S37_ROUTED_READOUT_PREREG.md` §9.2；本檔公式與誤差均來自當日實際打開的 PDF/HTML。讀不到全文者不編造公式。

## 0. 檢索記錄

日期皆為 2026-09-28。未宣稱遍歷全部文獻。

| 檢索式 | 來源 | 命中與篩選 |
|---|---|---|
| `title.search`：event-based stereo visual odometry；dense event-based deep stereo；temporal event stereo；event-based multi-view stereo；contrast maximization event camera；ESVO2；DSEC stereo event；event-based depth estimation 等 30 條 | OpenAlex works API | 成功批次約 70 篇去重。後續 DOI 批次與第二輪標題檢索回 HTTP 429，未再翻頁。篩選：題名含 event 且與 stereo / monocular depth / contrast 有關；去掉地球物理、推薦系統等誤命中 |
| `query.bibliographic`：SE-CFF、Selection and Cross Similarity、Secrets of Event-Based Optical Flow、Depth AnyEvent、ZEST、ADES、DTC、DERD、EMVS、E2Depth、RAMNet、EReFormer 等 | Crossref | 鬆散匹配的 `total-results` 常達數十萬，不可當命中數。人工只留題名相符的前 1–3 條 |
| `id_list` / `search_query` | arXiv API `export.arxiv.org` | HTTP 406，API 不可用。改走 `arxiv.org/pdf`、`arxiv.org/html`、`arxiv.org/abs` |
| 標題檢索 EReFormer、SE-CFF、Depth AnyEvent | `arxiv.org/search` | 多數 HTTP 429。一次回了 HTML 但抽不出題名 |
| DOI → OA PDF | Unpaywall | 取得 Depth AnyEvent `2509.15224`、Focus、EMVS（ZORA 頁本身失敗，改 RPG 鏡像）、Xie、Ghosh 綜述等。多篇 IEEE 正式版無 OA PDF |
| 直接 PDF | arXiv、CVF Open Access、Frontiers、Springer OA、RPG 文件頁 | 見下方深讀。ZORA、MDPI、Wiley、IEEE Xplore 多數被擋成 HTML 錯誤頁 |
| GitHub API / 倉庫首頁 | `api.github.com`、`github.com` | 搜尋偏到一般雙目倉庫；之後首頁逾時。代碼證據只採用論文正文印出的 URL，不臆造檔名與函式 |

**SE-CFF 對不上題名。** arXiv 標題檢索 0 篇；Crossref 前幾條與事件雙目無關。Ghosh 與 Gallego，TPAMI 2025，Table 2 裡 ECCV 2022 的事件–影像雙目是 **SCS-Net**（Cho 與 Yoon，*Selection and Cross Similarity for Event-Image Deep Stereo*），不是名為 SE-CFF 的論文。同表鄰近項是 Conc-Net（CVPR 2022，Nam 等，*Concentrate and Focus on the Future*）與 DTC-SPADE（CVPR 2022）。SCS-Net 全文未取得，不列入深讀；其數字只當綜述轉引。

**覆蓋缺口。** 未打開 ccf.org.cn，分區用通行口徑並標待核。未讀 SCS-Net、ZEST（NeurIPS 2024）、Shiba 等 TPAMI 2024 對比度最大化長文、TSES（ECCV 2018）全文、Ye 等 IROS 2020、跨模態一致性（IROS 2023）、Distil-E2D（NeurIPS 2025）、EMDepth、事件塌縮（Sensors 2022）全文。單目事件深度的 2023–2026 一區全文，付費牆與 API 限流之後，手上能逐頁讀到的少於 20 篇；不把綜述表格改寫成深讀。

**分區口徑（全文共用，不在每行重複升格）。** CVPR、ICCV、ECCV、NeurIPS、AAAI、TPAMI、IJCV：通行視為 CCF 推薦目錄 A 類（本次未逐條打開官網，標「通行口徑，待官網复核」）。TRO、RA-L、TCSVT、ICRA、IROS、3DV：通行多不在該 A 類清單或層級更低，標待核，不寫成 CCF-A。Sensors、Frontiers in Neuroscience、IET Computer Vision、Expert Systems with Applications、Advanced Intelligent Systems、ECCV Workshops：不升格。中科院分區年份未逐刊核對，一律不寫「一區 Top」除非上面已標通行口徑的那一組會議/期刊。

## 1. 候選目錄

「深讀」欄：全文或會議相機版 PDF/HTML，且下面第 2 節有獨立條目。綜述轉引不計入深讀篇數。

### G06 事件雙目

| 標題 | 作者 / 機構 | 發表處 | 年 | 狀態 | 代碼 | 標籤 | 深讀 |
|---|---|---|---|---|---|---|---|
| Semi-Dense 3D Reconstruction with a Stereo Event Camera | Zhou, Gallego, Rebecq, Kneip, Li, Scaramuzza；ANU / UZH | ECCV | 2018 | 正式 | 論文未在正文給出現今倉庫 | 雙目建圖、時間面 | 是 |
| Realtime Time Synchronized Event-based Stereo (TSES) | Zhu, Chen, Daniilidis；UPenn | ECCV | 2018 | 正式 | 未取得全文 | 時間同步雙目 | 否 |
| Learning an Event Sequence Embedding for Dense Event-Based Deep Stereo (DDES) | Tulyakov, Fleuret, Kiefel, Gehler, Hirsch；EPFL / Amazon | ICCV | 2019 | 正式 | 未核到官方倉庫 | 學習雙目 | 是 |
| Event-Based Stereo Visual Odometry (ESVO) | Zhou, Gallego, Shen；HKUST / TU Berlin | TRO | 2021 | 正式 | 正文未印 URL；GitHub 本次逾時 | 雙目 VO 建圖 | 是 |
| DSEC | Gehrig 等；UZH | RA-L | 2021 | 正式 | 未取得全文 | 駕駛雙目數據 | 否（數字經他文） |
| Event-Intensity Stereo (EIS) | Mostafavi, Yoon, Choi；GIST / KAIST | ICCV | 2021 | 正式 | 未核 | 事件+影像雙目 | 是 |
| Deep Event Stereo Leveraged by Event-to-Image Translation (EIT-Net) | Ahmed 等 | AAAI | 2021 | 正式 | 未取得對應 PDF（一次 AAAI 下載下錯文） | 事件轉影像再雙目 | 否 |
| Feature-based Event Stereo Visual Odometry | Hadviger 等；U. Zagreb | arXiv / 後有 Advanced Robotics 版本 | 2021 | 預印本為深讀底本 | 未核 | 特徵雙目 VO | 是 |
| Stereo Depth from Events Cameras: Concentrate and Focus on the Future (Conc-Net) | Nam, Mostafavi, Yoon, Choi | CVPR | 2022 | 正式 | 未核 | 未來事件蒸餾 | 是 |
| Discrete Time Convolution for Fast Event-Based Stereo (DTC-SPADE) | Zhang, Che, Leng 等；SUSTech / Huawei | CVPR | 2022 | 正式 | 未核 | 事件立體匹配 | 是 |
| Selection and Cross Similarity (SCS-Net) | Cho, Yoon；KAIST | ECCV | 2022 | 正式 | 未取得全文 | 事件–影像雙目；種子「SE-CFF」未對上 | 否 |
| Event-Based Stereo VO with Native Temporal Resolution (CT-GP) | Wang, Gammell | RA-L | 2023 | 正式；深讀 arXiv `2306.01188` | 未核 | 連續時間雙目 | 是 |
| ESVIO | Chen, Guan, Lu | RA-L | 2023 | 正式；深讀 arXiv `2212.13184` | 未核 | 雙目+IMU | 是 |
| Learning Adaptive Dense Event Stereo from the Image Domain (ADES) | Cho, Cho, Yoon；KAIST | CVPR | 2023 | 正式 | 未核 | 影像域遷移到事件雙目 | 是 |
| T-ESVO | Liu 等 | Adv. Intell. Syst. | 2023 | 正式 | Wiley PDF 未下到 | 自適應時間面 | 否 |
| IMU-Aided Event-based Stereo VO | Niu, Zhong, Zhou | ICRA | 2024 | 正式；深讀 arXiv `2405.04071` | 未核 | 加速 ESVO 建圖 | 是 |
| Temporal Event Stereo via Joint Learning with Stereoscopic Flow | Cho, Kang, Yoon；KAIST | ECCV | 2024 | 正式；深讀 arXiv `2407.10831` | 論文印 `github.com/mickeykang16/TemporalEventStereo` | 時間雙目+立體流 | 是 |
| ES-PTAM | Ghosh, Cavinato, Gallego | ECCV Workshops | 2024 | 正式 workshop；arXiv `2408.15605` | 未核 | 雙目 PTAM、融合 DSI | 是 |
| Edge-guided fusion … event-image stereo | Zhao, Zhou, Xiong | ECCV | 2024 | 正式 | 未取得全文 | 事件–影像 | 否 |
| LiDAR-event stereo fusion with hallucinations | Bartolomei 等；Bologna | ECCV | 2024 | 正式 | 未取得全文 | 事件+LiDAR | 否 |
| Zero-shot Event-Intensity Asymmetric Stereo (ZEST) | Lou, Liang, Shi 等 | NeurIPS | 2024 | 正式 | 未取得全文 | 測試單目事件、訓練用影像 | 否 |
| Event-based Stereo Depth Estimation: A Survey | Ghosh, Gallego；TU Berlin | TPAMI | 2025 | 正式（PDF 頁首 accepted）；預印本 `2409.17680` | 綜述頁有表格連結，本次未打開 | 雙目總表 | 是 |
| ESVO2 | Niu, Zhong, Lu, Shen, Gallego, Zhou | TRO | 2025 | 正式；深讀 arXiv `2410.09374` | 未核 | 雙目 VIO | 是 |
| DERD-Net | Hitzges, Ghosh, Gallego；TU Berlin | NeurIPS | 2025 | PDF 頁首寫 accepted | 論文稱提供代碼，URL 未在所讀頁核到 | 由 DSI 學深度；單目與雙目 | 是 |
| Voxel-ESVIO | Zhang 等 | arXiv `2506.23078` | 2025 | **預印本** | 未核 | 體素地圖雙目 VIO | 是 |
| Continuous-time 3D detection with stereo events | Kang, Cho, Yoon | arXiv `2508.02288` | 2025 | **預印本** | 未核 | 立體掃體積，任務是檢測 | 是 |
| Event-Based Stereo Depth Estimation Using Belief Propagation | Xie, Chen, Orchard | Frontiers in Neuroscience | 2017 | 正式 | 未核 | 早期訊息傳遞雙目 | 是 |
| Asynchronous Event-based Cooperative Stereo Matching | Firouzi, Conradt | Neural Processing Letters | 2015 | 正式 OA | 未核 | 合作網路、每事件更新 | 是 |

### G07 單目事件深度與時間結構

| 標題 | 作者 / 機構 | 發表處 | 年 | 狀態 | 代碼 | 標籤 | 深讀 |
|---|---|---|---|---|---|---|---|
| EMVS | Rebecq, Gallego, Mueggler, Scaramuzza；UZH | IJCV（BMVC 2016 為短版，深讀期刊版） | 2018 | 正式 | RPG 頁；GitHub 逾時 | 已知位姿、靜態、光線密度 | 是 |
| A Unifying Contrast Maximization Framework | Gallego, Rebecq, Scaramuzza | CVPR | 2018 | 正式 | 未核 | 聚焦 / 對比度；旋轉與深度 | 是 |
| Unsupervised Event-based Learning of Optical Flow, Depth, and Egomotion | Zhu, Yuan, Chaney, Daniilidis；UPenn | CVPR | 2019 | 正式；深讀 arXiv `1812.08156` | 未核 | 無監督；尺度靠雙目損失 | 是 |
| Focus Is All You Need | Gallego, Gehrig, Scaramuzza | CVPR | 2019 | 正式 | 未核 | 22 個聚焦損失；純旋轉 | 是 |
| Learning Monocular Dense Depth from Events (E2Depth) | Hidalgo-Carrió, Gehrig, Scaramuzza | 3DV | 2020 | 正式；深讀 arXiv `2010.08350` | 未核到倉庫 | 監督單目 | 是 |
| Unsupervised Learning of Dense Optical Flow, Depth and Egomotion… | Ye, Mitrokhin, Fermüller, Aloimonos | IROS | 2020 | 正式 | 未取得全文 | 單目學習 | 否 |
| RAM-Net | Gehrig, Rüegg, Gehrig, Hidalgo-Carrió, Scaramuzza | RA-L | 2021 | 正式；深讀 arXiv `2102.09320` | 未核 | 事件+影像 RNN | 是 |
| Event-based Monocular Dense Depth Estimation with Recurrent Transformers (EReFormer) | Liu, Li, Fan, Tian 等 | 預印本 2022；期刊 TCSVT 2024 | 2024 期刊 | 深讀的是 arXiv `2212.02791`，不是 IEEE 排版 | 未核 | 事件 Transformer | 是 |
| Deep Learning for Event-based Vision: A Comprehensive Survey and Benchmarks | Zheng 等（HTML 作者欄未逐一核） | arXiv `2302.08890` | 2023 | **預印本** | — | 只深讀單目深度專節 | 是（專節） |
| Self-Supervised Event-Based Monocular Depth Estimation Using Cross-Modal Consistency | Zhu, Liu, Jiang, Wen | IROS | 2023 | 正式 | 未取得全文 | 跨模態自監督 | 否 |
| EVEN | Shi 等 | ROBIO | 2023 | 正式 | 未取得全文 | 夜間單目 | 否 |
| Improved Event-Based Dense Depth via Optical Flow Compensation | Shi, Jing, Li, Liu | ICRA | 2023 | 正式 | 未取得全文 | 光流補償深度 | 否 |
| Secrets of Event-Based Optical Flow, Depth and Ego-Motion by Contrast Maximization | Shiba, Klose, Aoki, Gallego | TPAMI | 2024 | 正式 | 未取得全文 | 對比度最大化的退化 | 否 |
| EMDepth | Lin 等 | ICNC | 2024 | 正式 | 未取得全文 | 自監督單目 | 否 |
| Multi-Modal Fusion of Event and RGB… | Devulapally 等 | CVPRW | 2024 | 正式 | 未取得全文 | 事件+RGB | 否 |
| Depth AnyEvent | Bartolomei, Mannocci, Tosi, Poggi, Mattoccia；Bologna | ICCV | 2025 | 正式；arXiv `2509.15224` | 論文頁 `bartn8.github.io/depthanyevent` | 基礎模型蒸餾 | 是 |
| Depth Any Event Stream (EventDAM) | Zhu, Pan, Cao, Liu, Kwok, Xiong；HKUST(GZ) / HKUST | ICCV | 2025 | 正式 OA | 未核 | Depth Anything 遷移 | 是 |
| DERD-Net 的單目列 | 同上 | NeurIPS | 2025 | accepted | 同上 | 已知位姿下的單目 DSI | 是（與 G06 同一 PDF，單目結果另讀） |
| Distil-E2D | Lee, Lee | NeurIPS | 2025 | 正式（Crossref） | 未取得全文 | 影像到深度先驗蒸餾 | 否 |
| Event Camera Monocular Depth… State-Space | Wang 等 | IEEE TIM | 2026 | Crossref 有 DOI | 未取得全文 | 狀態空間單目 | 否 |

DERD-Net 同時出現在兩組：G06 讀雙目 DSI 融合，G07 讀作者自己寫的「單目事件深度是 ill-posed」以及單目誤差列。計數分開，不把同一段話算兩次。

## 2. 深讀

### G06 事件雙目（21 篇全文）

**G06-01. Zhou 等，ECCV 2018，半稠密雙目重建。**
原問題：一對時間同步的事件相機，在已知內外參與左相機位姿下，恢復參考視圖的逆深度。
公式（§2.1–2.2）：時間面 \(\mathcal{T}(\mathbf{x},t)=\exp(-(t-t_{\mathrm{last}}(\mathbf{x}))/\delta)\)，實驗 \(\delta=30\,\mathrm{ms}\)。
\(\rho^\star=\arg\min_\rho C(\mathbf{x},\rho)\)，
\(C=\frac{1}{|S_{RV}|}\sum_s\|\tau^s_{\mathrm{left}}(\mathbf{x}_1(\rho))-\tau^s_{\mathrm{right}}(\mathbf{x}_2(\rho))\|_2^2\)。
\(\mathbf{x}_1=\pi(T_{sr}\pi^{-1}(\mathbf{x},\rho))\)，\(\mathbf{x}_2=\pi(T_E T_{sr}\pi^{-1}(\mathbf{x},\rho))\)，\(T_E\) 是左右外參，常數。極性不用。
觀測與假設：兩台 DAVIS240，240×180，FOV \(62.9^\circ\)，基線 **14.7 cm**，運動捕捉給位姿。深度只長在有事件的邊緣。Table 1：simulation 三平面深度範圍 2.76 m，均值誤差 0.03 m、中位 0.01 m；indoor flying1 範圍 4.96 m，均值 **0.13 m**、中位 **0.05 m**；flying3 範圍 5.74 m，均值 **0.33 m**、中位 **0.11 m**。文中 relative error 是均值誤差除以**場景深度範圍**，不是 \(|\hat Z-Z|/Z\)。
代碼：未核到函式。
意義：厘米級中位誤差出現在數米遠的室內飛行，而且要第二台相機、同步、已知位姿與靜態（或剛體鏡頭運動）場景。0.05 m 已是 S37「每部位 1 cm」的五倍，場景還比手容易。

**G06-02. ESVO，Zhou、Gallego、Shen，TRO 2021（arXiv `2007.15548`）。**
原問題：雙目事件 VO 的建圖。同時刻左右時間面的時空一致性，加上跨時刻融合。
公式（§IV-A）：\(\rho^\star=\arg\min_\rho C\)，\(C=\sum_i r_i^2(\rho)\)，
\(r_i(\rho)=\mathcal{T}_{\mathrm{left}}(\mathbf{x}_{1,i},t)-\mathcal{T}_{\mathrm{right}}(\mathbf{x}_{2,i},t)\)。
\(\mathbf{x}_2\) 用常數 \(^{\mathrm{right}}T_{\mathrm{left}}\) 與該事件時刻的鏡頭運動。Gauss–Newton：\(\Delta\rho=-(J^\top r)/\|J\|^2\)。初值是極線上的整數視差 ZNCC，再做非線性修正。融合用 Student-t，不是把多次歷史深度當成獨立高斯測量。
觀測與假設：作者寫明，理想模型是「同一 3D 邊緣在兩台相機的對應極線上同時觸發事件」，但像素級同時性因延遲與不匹配並不成立，所以比的是時間面鄰域。實驗用兩台 DAVIS346，基線 **7.5 cm**。建圖對比在**真值位姿**下做。Table III 合成三平面：Student-t 均值深度誤差 **2.15 cm**（標準差 1.29 cm）。Table IV，upenn flying1/3 深度範圍 5.48 / 6.03 m：本方法均值 **0.16 / 0.19 m**，中位 **0.12 / 0.09 m**。可視化近距約 0.55 m 起，沒有 0.5 m 手的定量實驗。結構平行於基線時重建不出來。
代碼：GitHub 逾時，不引用函式。
意義：建圖要的是第二台已標定相機，不是單目時間序列。2 cm 級只出現在合成平行平面；真實飛行序列是分米級。

**G06-03. DDES，Tulyakov 等，ICCV 2019。**
原問題：把左右事件序列嵌成描述子，直接回歸稠密視差。
公式：\(\hat D=\mathrm{Net}(E^l,E^r\mid\Theta,d_{\max})\)。視差
\(\hat D_{y,x}=\sum_{j:|\hat\jmath-j|\le\delta} d(j)\,\mathrm{softmin}_j(C_{j,y,x})\)，\(d(j)=2j\)。
損失是視差上的 Laplace 交叉熵。每像素 FIFO 存最近 \(\kappa\) 個事件的極性與時間戳。
觀測與假設：MVSEC indoor flying，稀疏協議用最近 15 000 個事件位置。Table 4：DDES 均值深度誤差 **13.6±0.2 / 18.0±0.2 / 18.4±0.5 cm**（三個 split），與融合多視角、且用已知相機運動的 Semi-Dense 3D 同一量級，好過單視角的 TSES（36/44/36 cm）與 CopNet（61/100/64 cm）。作者寫：網路感受野變大之後，加大時間佇列幾乎無幫助；**在動態序列裡，空間上下文比時間上下文更可靠**。
意義：學習出來的雙目，在飛行數據上仍是十厘米級。時間維度並不能替代第二台相機的空間視差。動態場景下，模型還會主動少用時間。

**G06-04. EIS，Mostafavi、Yoon、Choi，ICCV 2021。**
原問題：事件堆疊與強度圖一起做雙目，補事件在靜止區不觸發的洞。
公式：軟 argmin 回歸視差；訓練用視差 L1（end-point error）。事件堆在強度幀之前。正文寫：只有事件時，靜態場景細節會缺。
觀測：MVSEC 與他們的立體事件數據；強度與事件要空間對齊。定量表在雙欄抽取裡與圖混排，本條不另造 MAE 數字；Ghosh 綜述 Table 4 轉引 EIS（2E）DSEC MAE 0.529 px、（2E+2F）0.396 px，標為轉引。
意義：事件–影像雙目仍是兩台（或事件加對齊的強度相機）。靜止手掌內部沒有事件，融合影像也只填強度能看見的紋理；皮膚低紋理時兩邊都稀。

**G06-05. Conc-Net，Nam 等，CVPR 2022。**
原問題：不同密度的事件堆疊，用注意力挑「堆得好」的那一層；並用未來事件蒸餾，推理時只用過去。
公式：\(L=L_{\mathrm{sl1}}(D,\hat D_{\mathrm{past}})+L_{\mathrm{sl1}}(D,\hat D_{\mathrm{both}})+L_{\mathrm{sim}}(b_{\mathrm{both}},b_{\mathrm{past}})\)。
Table 1，DSEC 視差：事件-only 基線 MAE **0.576** px；E-Stereo 0.529；他們的事件-only **0.519** px（1PE 9.583%）；事件+強度 **0.364** px（1PE 4.844%）。解析度 346×260 / 640×480。
意義：最好的數字是駕駛場景上約半個像素的視差，而且加強度更好。這不是 0.6 m 處手部的深度，也不能在只有一台事件相機時復現。

**G06-06. DTC-SPADE，Zhang 等，CVPR 2022。**
原問題：用離散時間卷積編碼不規則事件，再進立體匹配，避免固定時間箱把時間抹平。
公式：事件先變成堆疊，DTC/CTC 做時空編碼，再以 SPADE 調制並預測視差。文中編碼消融（MVSEC 協議一類的 MDE）：手工聚合 17.9±0.6 cm，event queue 16.9±1.0 cm，DTC **15.4±0.1 cm**，CTC 15.1±0.3 cm。Ghosh Table 4 轉引完整方法：MVSEC 均值 13.5 / — / 17.1 cm，DSEC MAE **0.526** px。
意義：把時間卷積做得更細，深度誤差仍在十厘米（飛行數據）或半像素視差（駕駛）。時間編碼改善的是匹配特徵，不創造第二個光心。

**G06-07. ADES，Cho、Cho、Yoon，CVPR 2023。**
原問題：事件雙目缺少稠密真值時，從影像雙目域（如 KITTI）遷移。
做法：極線方向對齊特徵；smudge-aware 自監督模仿事件在時間上被塗開的效應；運動不變的一致性。推理可以只有事件，訓練用了影像域。
Ghosh Table 4 把 ADES 標成無監督、模態 `2E (+2F train)`，DSEC MAE **0.771** px、1PE 18.37%，差於全監督事件雙目。正文一處寫 Zurich City 上某誤差自 16.1 降到 7.0，欄位在雙欄抽取中未對齊，不把 7.0 寫成深度厘米。
意義：測試時看起來像「只用事件」，尺度與對應其實在訓練時由影像雙目教過。這不支持「單目時間序列自行長出 1 cm 深度」。

**G06-08. Temporal Event Stereo，Cho、Kang、Yoon，ECCV 2024（arXiv `2407.10831`）。**
原問題：事件在時間上連續，立體匹配應同時學「立體流」，用視差真值去監督流，再把過去的匹配沿流送到當前。
做法：立體匹配網路與 stereoscopic flow 一起訓練；流的真值不另標，由視差推。時間聚合是級聯，而不是把歷史深度當獨立觀測。代碼 URL 印在論文中。
Ghosh Table 4 以 StereoFlow-Net 轉引：MVSEC 均值深度 **13 / — / 15 cm**，DSEC MAE **0.493** px，是表中事件-only 最好的一檔之一。
意義：2024 的時間雙目把時間用在**已有左右相機**的匹配上。時間幫助的是對應與遮擋，不是取代基線。

**G06-09. Xie、Chen、Orchard，Frontiers in Neuroscience 2017。**
原問題：一對事件相機的視差，用信念傳播，全事件驅動。
觀測：左右事件要落在極線上；訊息在視差節點間傳遞。全文讀了摘要、引言與問題設定；定量表未完整抽出，故不引用厘米數。
意義：早期方法已經把「兩台感測器 + 極線」寫成問題定義。沒有第二視圖就沒有視差節點。

**G06-10. Firouzi 與 Conradt，Neural Processing Letters 2015。**
原問題：每個新事件異步更新視差，不先把事件收成幀。
機制：合作網路；跨視差唯一性、同一視差連續性。讀了摘要與問題設定，沒有可用的厘米表。
意義：異步只降低匹配延遲。幾何仍是雙目合作，不是單目重放。

**G06-11. Hadviger 等，arXiv `2107.04921`（特徵雙目 VO）。**
原問題：時間面上的角點，用重投影誤差做雙目 VO。
公式：\(\pi(X;R,t)\) 標準針孔投影；在候選點上最小化重投影。文中一組感測器 346×260、基線 **10 cm**；另一處提到基線 **60 cm** 的設置。匹配還用到事件時間面的歸一化互相關一類線索。
意義：特徵法同樣要求左右外參。10 cm 基線在近處理論上視差大，但手的低紋理邊緣很少，角點前提不成立。

**G06-12. Wang 與 Gammell，RA-L 2023，連續時間高斯過程（arXiv `2306.01188`）。**
原問題：每個特徵有自己的時間戳，不能再假設一個立體對共用一個快門時刻。
公式：軌跡段上常速度 \(\,T\approx I+\Delta t\,\varpi^\wedge\)，把重投影誤差線性化後與高斯過程加速度先驗一起解。特徵要通過左右時序追蹤，且視差小的點被丢掉（文中有小於 2 px 視差的篩選）。
意義：它把**雙目剛體軌跡**寫成連續時間。手不是剛體鏡頭，50 ms 包內的非剛性運動不能套這條高斯過程。

**G06-13. ESVIO，Chen、Guan、Lu，RA-L 2023（arXiv `2212.13184`）。**
原問題：雙目事件角點加 IMU。
觀測：兩台相機剛性連接，基線 **6.0 cm**。HDR 飛行軌跡 RMSE **0.17 m**；滾轉與俯仰誤差在約 6° 以內的量級（文中對 VICON 的敘述）。攻擊性飛行 RMSE 0.26 m。深度是路標三角化的副產品，論文主指標是軌跡，不是每部位 1 cm。
意義：即使用 IMU 與 6 cm 基線，系統誤差仍是分米級軌跡。近距手的 1 cm 相對深度不在其設計點。

**G06-14. Niu、Zhong、Zhou，ICRA 2024，IMU 輔助 ESVO（arXiv `2405.04071`）。**
原問題：ESVO 直接法的瓶頸是建圖太慢、跟蹤精度有限。
做法：仍是左右事件的生成模型與隱式對應，用 IMU 幫運動，並改建圖的計算路徑。深讀集中在問題陳述與相對 ESVO 的定位，未另抽一張新的厘米深度表。
意義：後續工作優化的是**已有雙目**的速度與跟蹤，沒有改成單目可模擬。

**G06-15. ESVO2，Niu 等，TRO 2025（arXiv `2410.09374`）。**
原問題：把雙目事件直接法擴到 VGA、接 IMU，在校園尺度上比 ESVO 更少漂移。
觀測：仍是左右事件時間一致性建圖；圖 1 的深度是半稠密邊緣。主指標是軌跡相對 RTK，不是近距絕對深度。
意義：2025 年的雙目 VIO 前沿仍預設兩台事件相機。尺度來自主體基線與 IMU，不是單目學習先驗。

**G06-16. ES-PTAM，Ghosh、Cavinato、Gallego，ECCV Workshop 2024（arXiv `2408.15605`）。**
原問題：雙目事件的平行跟蹤與建圖，用融合後的光線密度（DSI）而不是逐對匹配。
公式：兩台相機的 DSI 在三維點上融合（正文給出融合值由兩台 DSI 決定）；跟蹤是把地圖投到候選位姿，最小化雙模事件圖的光度式誤差。文中相對 ESVO，RPG 上軌跡誤差降約 45%，另一組約 61%（摘要）。
意義：多相機融合的是**空間視差**。單目 EMVS 在同表裡更差（見 G07-01 與 DERD）。Workshop 不是主會，但是同一幾何。

**G06-17. Voxel-ESVIO，Zhang 等，arXiv `2506.23078`（預印本）。**
原問題：事件噪聲讓地圖點質量差，用體素地圖濾點，服務雙目視覺–慣性里程計。
觀測：仍是立體事件加 IMU。主指標 ATE。不報告手部相對深度。
意義：地圖管理不增加觀測。預印本，未當正式期刊結果。

**G06-18. Kang、Cho、Yoon，arXiv `2508.02288`（預印本）。**
原問題：雙目事件在沒有 RGB 的時間縫裡做連續時間三維檢測。
公式：事件生成 \(L(u,v,t)-L(u,v,t-\Delta t)\ge p\,C\)；幾何支路用 plane-sweep 體積，把左右事件投到深度平面。檢測頭輸出框的殘差，不是逐手部深度。
意義：2025 年要用事件的時間，仍然先有左右兩台相機，再掃深度。任務是車載物體框。預印本。

**G06-19. Ghosh 與 Gallego，TPAMI 2025 綜述（accepted PDF；預印本 `2409.17680`）。**
原問題：把事件雙目分成瞬時 / 長時間基線、模型 / 學習、稀疏 / 稠密，並在 MVSEC 與 DSEC 上對齊數字。
公式（§2）：正規雙目 \(Z=(b\cdot f)/\Delta x\)。事件模型 \(L(\mathbf{x},t_k)-L(\mathbf{x},t_k-\Delta t_k)=p_k C\)。
Table 4（學習法，數字註明來自原文）：事件-only 在 MVSEC indoor flying 的均值深度大約 **11–20 cm**；DSEC 視差 MAE 大約 **0.49–0.58 px**。加上強度後 DSEC MAE 可到約 **0.36–0.39 px**。無監督明顯更差。
Table 5（模型法，MVSEC，約 200 s）：EMVS **單目** 均值 39.37 / 31.42 / 30.54 cm，中位 14.35 cm；ESVO 雙目均值 23.39 / 20.42 / 24.29 cm，中位 9.83 cm；MC-EMVS 均值 22.53 / 18.20 / 19.49 cm，中位 9.53 cm。
Table 6（DSEC zurich city 04a，最大距離 50 m，0.2 s 窗）：EMVS 單目均值 **517 cm**、中位 **99 cm**；最好的雙目模型法均值仍是 **290 cm** 量級。
§5.2 作者原話的要點：雙目相對單目（時間）立體的好處是精度、建圖速度、剔除外點、以及**絕對尺度**；代價是兩倍事件、更嚴的標定。瞬時雙目較能處理獨立運動物體；長時間基線方法假設場景在融合窗內可對齊，動態物體會破壞時間恆常。低事件數像素（駕駛時影像中心）仍是開放問題。
數據表：RPG stereo 基線 14.7 cm；文中另有多組手持雙目基線約 6–30 cm，以及更大基線的駕駛平台。作者建議基線要按深度範圍選。
意義：這張表是「單目時間立體 ≠ 雙目」的直接對照。單目 EMVS 在同一協議下系統性更差，而且駕駛遠距誤差到米。

**G06-20. DERD-Net，Hitzges、Ghosh、Gallego，NeurIPS 2025（arXiv `2504.15863`）。**
原問題：已知位姿把事件反投成 DSI，用小網路從 DSI 子塊讀出深度，單目與雙目都做。
公式：位姿已知時，事件 \(e_k\) 從移動光心穿過像素打成射線；深度軸 \(D\) 層，體素計數射線。雙目時兩個 DSI 按體素融合（如調和平均，沿用 MC-EMVS）。網路輸入子塊 \(100\times 1\times 7\times 7\)，3D 卷積加沿深度的 GRU，輸出深度。
假設（§2 原文要點）：這類方法**假設靜態世界與已知相機運動**，用來在更長時間裡累積視差。作者寫：單目事件深度是 **ill-posed**，學習框架很難得到高精度；靜態場景沒有事件。
Table 11，MVSEC indoor flying 平均，單位 cm：EMVS 單目均值 **33.78**、中位 **14.35**；ESVO 雙目（獨立 1 s）均值 22.70、中位 9.83；MC-EMVS 均值 20.07、中位 9.53；DERD 雙目均值 **11.69**、中位 **5.50**；DERD **單目**均值 **23.68**、中位 **11.55**。單目學習 DSI 仍比雙目差約一倍，中位 11.6 cm，不是 1 cm。
意義：即使用 2025 年的網路去讀光線密度，單目列也到不了 S37 的 1 cm 門。而且輸入裡已經有位姿；位姿若來自錯誤的 prev 手姿態，DSI 會沿錯誤射線投票。

**G06-21. 對照用的幾何極限（不是新論文，服務 G06-01 的感測器參數）。**
Zhou 等 2018 的 DAVIS240：寬 240 px、水平 FOV \(62.9^\circ\)。焦距換算 \(f_x\approx (240/2)/\tan(31.45^\circ)\approx 196\,\mathrm{px}\)（論文給 FOV，不直接給 \(f_x\)）。基線 \(b=14.7\,\mathrm{cm}\)、\(Z=0.6\,\mathrm{m}\) 時視差 \(d=bf_x/Z\approx 48\,\mathrm{px}\)，\(\lvert\mathrm{d}Z/\mathrm{d}d\rvert=Z^2/(b f_x)\approx 1.25\,\mathrm{cm/px}\)。要每部位 1 cm，匹配須穩在約 0.8 px 以內，而且只在有事件的邊緣上。駕駛數據上 0.5 px 的 MAE 不能搬到手：解析度、基線、紋理、剛性都不同。本項目沒有第二台相機，這條極限用不上。

### G07 單目事件深度與時間結構（12 篇；未滿 20）

未滿 20 的原因：OpenAlex 後段 429、arXiv API 406、arXiv 搜尋 429、IEEE/Wiley/MDPI 全文下載失敗。2023–2026 的單目事件深度裡，ICCV 2025 兩篇蒸餾、NeurIPS 2025 DERD、以及 2018–2021 的基礎工作能讀到全文；IROS/ICRA/TIM/NeurIPS 的 Distil-E2D、跨模態一致性、Shiba TPAMI 2024、Ye 等 IROS 2020 只有題名或綜述一句話。不把這些編成深讀。下面 12 篇已足夠回答 1 cm 門是否站得住。

**G07-01. EMVS，Rebecq 等，IJCV 2018。**
原問題：一台移動的事件相機、已知視點，重建靜態場景的半稠密三維邊緣。
公式（§5）：事件 \(e_k=(x_k,y_k,t_k,p_k)\) 按 \(t_k\) 的相機位姿反投成射線。DSI \(f(X)\) 是穿過體素的射線條數。沿每條視線取射線密度的局部極大，得到深度與置信度。沒有顯式對應。場景點是多條視線的交匯；作者強調：感測器與場景都靜止時沒有事件，因此**必須移動相機**。輸出是邊緣，不是稠密表面。
Table 1 合成：Dunes / 三平面 / 三牆，深度範圍 3.00 / 1.30 / 7.60 m，均值誤差 **0.14 / 0.15 / 0.52 m**，相對誤差（均值/範圍）4.63% / 11.31% / 6.86%。
Table 2，DAVIS 240×180 對紋理牆，滑軌已知運動，不做後處理：距離 **23.1 cm** 均值誤差 **1.22 cm**（相對 5.29%）；距離 **58.5 cm** 均值 **2.01 cm**（恆定光照，相對 4.33%）與 **1.87 cm**（HDR，相對 3.44%）。高速實驗，牆距 **40.5 cm**，均值 **1.26 cm**（相對 4.84%）。
意義：文獻裡最接近「0.5–0.7 m、厘米級」的數字，是**靜態紋理牆 + 已知平移 + 單目運動視差**。0.58 m 處均值約 **2 cm**，已經落在 S37 掃描的 2 cm 格（只比無深度好 0.8–0.9°），而且不是非剛性、低紋理的手。射線密度的許多深度層是同一批事件的互斥假設，不是多次獨立測深。

**G07-02. Gallego、Rebecq、Scaramuzza，CVPR 2018，對比度最大化。**
原問題：用一個運動參數 \(\theta\) 把事件扭到參考時刻，使扭曲事件圖（IWE）的對比度最大，從而估計運動、深度或光流。
公式：對比度是 IWE 的方差。深度應用：在已知相機運動下沿視線掃深度，對比度最大的深度即 EMVS 式聚焦。旋轉應用：短窗內常角速度，圖像運動不依賴深度。平面場景則參數化 \(\theta=\{\omega, v/d, n\}\)，平移與深度只以 \(v/d\) 出現。文中圖示基線從 1.9 cm 增到 85 cm，深度圖才變清楚。
意義：聚焦恢復的是「哪個深度讓邊緣對齊」，前提是扭轉所根據的運動已知且場景點剛性。純旋轉可估計角速度，因為該分量與深度無關；繞腕的出平面旋轉不是相機純旋轉，點的像速度依賴槓桿與深度。\(v/d\) 表明平移與深度不可分。

**G07-03. Focus Is All You Need，Gallego、Gehrig、Scaramuzza，CVPR 2019。**
原問題：運動補償裡，哪一種聚焦損失最能衡量事件沿軌跡對齊。
公式：事件按 \(\theta\) 扭成 IWE，損失是 IWE 的方差、梯度幅值、局部方差等二十餘種。旋轉實驗用來比較損失，因為該任務的幾何相對乾淨。
意義：時間結構的可用形式是「事件應被某個運動模型對齊」。損失本身不提供米制深度。出平面旋轉若要寫進 \(\theta\)，仍要一個三維運動模型；用 prev 網格當該模型，深度就回到循環量。

**G07-04. Zhu 等，CVPR 2019（arXiv `1812.08156`）。**
原問題：只從事件學光流、深度與自運動，無光度真值。
公式：事件體積保持時間分布；深度–自運動網路用重投影與時間戳損失。尺度來自**雙目視差損失**（訓練時用立體事件相機的左右一致性），不是單目事件自帶米制。正文寫網路必須記住度量尺度。E2Depth Table 3 轉引 Zhu 等在 outdoor day1 的平均絕對深度誤差：**2.72 / 3.84 / 4.40 m**（截斷 10 / 20 / 30 m）。Zhu 自己的表在 outdoor night 一類設置給出 3.13 / 4.02 / 4.89，與 E2Depth 表中 Zhu 的 night1 行一致，單位是米。
意義：無監督單目事件深度的尺度是訓練時用第二台相機灌進去的。測試誤差在駕駛場景是數米。自運動不好時深度一起壞。

**G07-05. E2Depth，Hidalgo-Carrió、Gehrig、Scaramuzza，3DV 2020。**
原問題：用循環網路從事件體素預測稠密度量深度。
公式：事件體素
\(E_k(u_k,t_n)=\sum_i p_i\,\delta(u_i-u_k)\max(0,1-\lvert t_n-t_i^\star\rvert)\)。
監督是對數深度的尺度不變損失加梯度項，\(\lambda=0.5\)。先合成後真實（MVSEC outdoor day2 訓練）。
Table 2，outdoor day1，最好的 \(S^\star\to(S+R)\)：**Abs Rel 0.346**，RMSE **8.564 m**。Table 3 平均絕對誤差（米）：outdoor day1 在 10/20/30 m 截斷為 **1.85 / 2.64 / 3.13**。night 序列更差，約 3–4 m（10 m 截斷）。
意義：這是學習先驗，不是幾何測深。Abs Rel 0.35 若錯誤地線性搬到 0.6 m，也有約 20 cm，仍比 1 cm 大一個量級；而且駕駛的誤差結構（地面、車輛、天空）與手無關。靜止區域沒有事件，網路只能靠先驗填。

**G07-06. RAM-Net，Gehrig 等，RA-L 2021。**
原問題：事件與影像異步融合的單目深度。
公式：與 E2Depth 同類的體素
\(V(x,y,t)=\sum_i p_i\,\delta(x-x_i,y-y_i)\max\{0,1-\lvert t-t_i^\star\rvert\}\)，
循環網路在影像幀之間用事件更新。
觀測：影像提供靜止區與尺度，事件提供高動態。作者指出只看影像的基線會在 LiDAR 無效區出偽影，事件減輕但沒有取消米級場景。定量以相對 E2Depth / Zhu 的曲線為主，工作距離仍是 MVSEC 的 10–30 m 截斷。
意義：要在單目事件上得到可用稠密深度，論文的做法是**再加影像**。本項目沒有對齊的強度幀。即便有，RAM-Net 的誤差尺度也不是手上的 1 cm。

**G07-07. EReFormer，Liu 等，arXiv `2212.02791`（期刊 TCSVT 2024，深讀預印本）。**
原問題：用循環 Transformer 從事件估單目稠密深度。
公式：當前幀特徵 \(f_t\) 與歷史 \(h_{t-1}\) 做線性注意，
\(Q_t=f_t W_Q^f+h_{t-1}W_Q^h\)，\(K,V\) 同形。骨幹 Swin-T，ImageNet 預訓練。
觀測：仍是 MVSEC 駕駛深度，指標 Abs Rel、RMSE log、10/20/30 m 絕對誤差。預印本寫相對基線 Abs Rel 約有 8.2% 的相對改進，10 m 截斷的平均絕對誤差約 14.8% 的相對改進。雙欄表未完整對齊，絕對米數以 Depth AnyEvent 對 EReFormer 的重評為準（下一條），避免把擠壓後的數字寫成原文。
意義：更大的時序模型沒有改變問題的不適定。參數量遠大於 S37 根頭上 1.3 k 的預算。

**G07-08. Zheng 等，arXiv `2302.08890`，2023 綜述的單目深度專節（HTML，非通讀 276 條參考文獻）。**
專節原話要點：單目深度可分事件-only 與事件加影像。事件-only 以 E2Depth 為代表，用循環網路讀網格化事件；**從事件做單目深度是 ill-posed，難以高精度**；場景靜止時沒有事件，深度預測會失敗。事件加影像以 RAM-Net 為代表，用影像補事件。立體方法才使用左右事件的空間線索。
意義：2023 年的事件視覺綜述與 2025 年 DERD 的措辭一致。這不是實現細節問題。

**G07-09. Depth AnyEvent，Bartolomei 等，ICCV 2025（arXiv `2509.15224`）。**
原問題：把 Depth Anything v2 的影像深度先驗蒸餾給只吃事件的學生網路。
公式：學生吃事件歷史；教師是 DAv2 ViT-L，只在蒸餾時看影像。代理標籤替代稀疏真值。評估前**對預測做尺度與平移對齊**再算 Abs Rel 等（§5.2 原文：scale and shift to align predictions with the ground-truth）。
Table 3，合成預訓練、**未**對齊前的協議仍是同一套對齊後指標：MVSEC 上 DepthAnyEvent Abs Rel **0.466**、RMSE **7.824 m**；E2Depth 0.527 / 7.894 m；EReFormer 0.518 / 8.423 m。DSEC 上 DepthAnyEvent-R 合成 Abs Rel **0.276**、RMSE **10.942 m**。蒸餾之後 DSEC Abs Rel 可到 0.191–0.226，RMSE 仍 **8.6–9.3 m**。
意義：基礎模型遷移改善的是對齊之後的相對形狀，作者自己拿掉了全局尺度。RMSE 仍是數米到十米。這不能當作 0.6 m 處 1 cm 的度量深度。影像只在訓練出現，測試的事件學生沒有第二視角。

**G07-10. EventDAM / Depth Any Event Stream，Zhu 等，ICCV 2025。**
原問題：把 Depth Anything 適配到事件流，用稀疏性加權的特徵混合與稠密到稀疏的蒸餾，訓練可不依賴事件深度真值。
公式：總損失 \(L=L_{\mathrm{SFM}}+\vartheta_{\mathrm{SFD}}L_{\mathrm{SFD}}+\vartheta_{\mathrm{SCM}}L_{\mathrm{SCM}}\)，\(\vartheta\) 取 2 與 0.2。深度用仿射不變損失與尺度不變對數損失；預測會被縮放、平移到零平移與單位尺度。
定量：文中 10/20/30 m 平均深度誤差，EventDAM-S 事件輸入約 **1.64 / 2.46 / 3.36 m**；未混合時約 0.65 / 1.92 / 2.99（另一組消融的米制截斷，與 Abs Rel 表並存，此處只引用帶 10/20/30 m 字樣的那組）。影像教師 DAM-S 在同樣截斷約 1.33 / 2.03 / 3.06 m，仍然是米。
意義：2025 年的基礎模型遷移把駕駛場景的 10 m 內誤差從數米降到約 1.6 m。相對 10 m 約 16%。即便這個相對誤差能搬到 0.6 m（不能，分布不同），也約是 10 cm，比 1 cm 寬一個數量級。損失明示尺度不可觀測。

**G07-11. DERD-Net 的單目列（同一 PDF，讀單目假設與 Table 11 單目行）。**
見 G06-20。補充這一側的結論：在已知位姿、靜態場景、MVSEC 室內飛行上，單目 DERD 中位誤差 **11.55 cm**（均值 23.68 cm），單目 EMVS 中位 **14.35 cm**。作者把「單目事件深度 ill-posed」寫在相關工作裡，並用實驗對上這句話。手上沒有已知的獨立相機運動，連這個 11 cm 的前提都不成立。

**G07-12. E2Depth / RAM-Net / Zhu 共同的數據前提（作為交叉核對，不另計新論文）。**
三篇的測試場景是車或無人機，深度截斷 10–30 m，真值來自 LiDAR 或立體。沒有一篇在 0.5–0.7 m、非剛性、低紋理的手上報告每部位相對深度。因此「先驗可能在手上突然變到 1 cm」沒有文獻支撐，只有外推，而外推方向是更難而不是更容易。

## 3. 可觀測性：單目事件、50 ms、非剛性手、MANO

狀態是相機系下的 MANO：平移、繞腕的根旋轉、15 個局部軸角。每 50 ms 一個事件包。手距約 0.5–0.7 m。S37 §9.2：在已有 2D 剪影殘差 × 3D 槓桿的特徵上，再加入**真值事件深度相對 prev 頂點深度的殘差**，單包根旋轉從 7.09° / 6.96°（ridge / MLP）降到 5.30° / 4.51°。把每部位池化後的深度殘差加高斯噪聲：

| 每部位深度誤差 \(\sigma\) | 0 | 1 cm | 2 cm | 3 cm | 5 cm |
|---|---|---|---|---|---|
| 比無深度好（ridge / MLP） | 1.8° / 2.5° | **1.2° / 1.6°** | 0.8° / 0.9° | 0.6° / 0.6° | 0.4° / 0.3° |

prev 提供的深度與 prev 深度誤差相關 0.74–0.98，殘差恆為零。1 cm 深度差在 8 cm 槓桿上約等於 7° 旋轉，和旋轉本身同一難度。

### (a) 單目事件深度還剩多少

兩類方法，前提都不會在這隻手上成立。

1. **幾何聚焦（EMVS、對比度最大化、DERD 的 DSI）。** 需要靜態場景與已知相機運動，讓同一三維點的射線交在一起。最好的近距數字是紋理牆、滑軌已知運動：0.585 m 均值約 **1.9–2.0 cm**（EMVS Table 2）。MVSEC 室內飛行、同樣假設下，單目中位約 **12–14 cm**（DERD Table 11）。手是非剛性的；相機通常固定，運動在手上，射線不交於固定點。皮膚低紋理，事件只在輪廓。50 ms 內的平移基線若只有 1 cm 量級，視差比 14.7 cm 的雙目小一個數量級，深度方差按 \(Z^2/B\) 放大。
2. **學習先驗（Zhu、E2Depth、RAM-Net、EReFormer、Depth AnyEvent、EventDAM）。** 監督或蒸餾自駕駛 / 合成駕駛。度量誤差是米；2025 年的蒸餾還在評估前做尺度與平移對齊，或在損失裡使用仿射不變項。Zheng 等 2023 與 DERD 2025 都把單目事件深度寫成 ill-posed，並指出靜止區沒有事件。

手上剩下的，不是「差一點的 1 cm」，而是**沒有一個獨立的度量深度通道**。學習模型至多提供一個與訓練手姿態先驗糾纏的形狀，其誤差預期遠大於 1 cm，且與根旋轉估計用的是同一批事件，不是新資訊。

### (b) 雙目要什麼，為何單目時間序列模擬不了

雙目要的是：第二個光學中心、已知基線與外參、時間同步（或連續時間模型裡各自的時間戳加上剛體軌跡）、以及極線約束 \(Z=(b f)/\Delta x\)。事件版把光度誤差換成時間面或事件嵌入的一致性（ESVO 式 (4)、DDES 的左右描述子）。瞬時雙目不要求場景剛性，所以比較能看獨立運動的手；長時間融合則要求對應點在融合窗內仍是同一靜態點。

單目時間序列模擬不了，因為：

- 只有一條光心軌跡。歷史上的深度假設是沿**同一條視線**的互斥解釋（EMVS 的 DSI 層），選出極大值仍是一次估計。
- 時間立體的極線由未知的相對位姿決定。該位姿正是根旋轉與平移，用它來測深度再回頭修旋轉，是循環。
- 非剛性使不同時刻的事件不對應同一表皮點。DDES 在動態序列裡觀察到時間上下文比空間上下文更不可靠。
- 手的平移很小，有效基線遠小於文獻雙目的 6–15 cm，\(Z^2/B\) 把視差噪聲放大。純旋轉不提供視差（Gallego 等 2018 的 \(v/d\)）。

ADES、ZEST 這類「測試時只有事件」的網路，訓練時用了影像或第二模態（ADES 已讀；ZEST 只見綜述 Table 2 的 `1E (+1F train)`，未深讀）。那是把雙目或影像深度蒸餾進權重，不是在 50 ms 包裡做了一次測量。

### (c) 文獻是否支持 ≤1 cm，方案 B 的深度擴展

不支持。把已讀到的最好近距結果對上 S37 的掃描：

- EMVS 在 58.5 cm 紋理靜態牆、已知運動：均值約 2 cm。S37 在 \(\sigma=2\,\mathrm{cm}\) 只多 0.8° / 0.9°，且該掃描還偏樂觀（覆蓋率通道未加噪）。
- 同一感測器在 23 cm 才到約 1.2 cm，距離比手更近，場景仍是平面紋理，不是手。
- 雙目在數米飛行場景的中位最好約 5 cm（DERD 雙目），單目約 12 cm。DSEC 上單目 EMVS 中位約 1 m。
- 學習單目在 10 m 截斷仍是 1.6 m 以上，或在對齊尺度之後才報告 Abs Rel。

因此 **方案 B 不應增加深度通道**：既不要單目事件深度頭，也不要把 prev 的頂點深度當作測量。prev 深度是循環量，殘差構造出來也是零。1 cm 的 oracle 增益不能當成可實現的資訊。

### (d) 不靠深度、靠事件時間的替代

存在一種**弱**的替代，它約束的是像面運動，不是度量深度。

事件的時間戳給出邊緣越過像素的時刻。對比度最大化把這些時刻扭到與運動參數一致（Gallego 等 2018、2019）。在輪廓上，可觀測的是沿圖像梯度的法向流，不是完整光流。相機純旋轉的像運動與深度無關，所以角速度可以由聚焦估計。繞腕的出平面旋轉會讓輪廓產生膨脹與剪切，這部分**可以**寫進一個以根旋轉為參數的扭轉模型，而不必先輸出一張深度圖。

它仍然用到形狀：扭轉的雅可比依賴頂點的三維位置。若位置來自當前假設的 MANO，深度是姿態的函數，應放在待估變量裡一起動，而不是把 prev 深度當成額外特徵。殘差為零時更新必須為零，否則又回到 S38 禁止的、與證據無關的加性幾何項。

這條資訊的強度有限。S37 上下文已記錄：理想輪廓上 rot_z 增益約 0.2，且與橫向平移近共線；rot_x / rot_y 約 0.5。50 ms 手指事件中位只有約 652 個。乾淨樣本上現有解碼器仍注入約 3.9–5.0°。文獻沒有證明時間聚焦能在這種事件密度下再挖出 1° 以上的獨立根旋轉。Shiba 等關於對比度最大化退化（事件塌縮）的 TPAMI 2024 全文本次未讀，不引用其公式；只把它留在缺口裡，當作該方向的已知風險，而不是證據。

## 4. 可證偽假說

**H1（應關閉）。** 在 0.5–0.7 m 的非剛性手上，單目事件能提供每部位相對深度誤差 ≤1 cm 的獨立測量，使 S37 式單包根旋轉再降 1.2–1.6°。

最強反對：EMVS Table 2 在更容易的靜態紋理牆、0.585 m、已知運動上均值已約 2 cm；DERD Table 11 單目中位 11.6 cm；Depth AnyEvent 與 EventDAM 的度量誤差是米，且評估或損失去掉了尺度。prev 深度與自身誤差相關 0.74–0.98。H1 與已讀文獻衝突。方案 B 的深度擴展關閉。

**H2（仍可做最小實驗，預期偏弱）。** 包內事件時間（輪廓法向流或對比度對根旋轉的聚焦）在不新增深度通道的前提下，能把單包根旋轉降到低於「2D 殘差 × 3D 槓桿」的 6.96–7.09°，且降幅不能被常數增益或純可見性修復解釋。

反對：出平面旋轉與平移在輪廓上近共線（項目內 S38 理想輪廓）；事件只提供法向分量；動態場景下 DDES 觀察到時間上下文被空間上下文蓋過；乾淨樣本的注入底約 4°。H2 可能只在擾動大、輪廓事件多的包上有小增益，閉環不一定留下。

## 5. 最小可遷移機制

不遷移深度網路。E2Depth / EReFormer / Depth AnyEvent 是循環 U-Net 或 Swin 級骨幹，參數與延遲都超出方案 B 的約 1.3 k 參數與 +0.2–0.5 ms 的量級，而且測量達不到 1 cm。

可保留的形式是方案 B 已有的乘法結構，不加深度：

\[
\Delta\xi=\sum_j M(L_j)\,\psi(q_j),\qquad \psi(0)=0.
\]

\(q_j\) 只用 2D 有符號距離、法向、覆蓋，與現有路由權重。\(L_j\) 用 prev 的 FK 槓桿，乘在殘差上；殘差為零則該項為零。接入點是現有路由與根更新（上下文中的 `semkine/routed_readout.py`、`model/model.py` 的包前向、`semkine/jacobian.py` 的投影雅可比、`semkine/kssf.py` 的輪廓距離）。成本仍是方案 B 已估算的量級，不另加深度頭。

若要試 H2，只在 \(q_j\) 裡增加**無學習參數**的時間矩：例如包內事件相對 \(t_0\) 的法向流一階矩，仍按部位用 \(a_{ij}\) 池化，維度增加個位數，\(\psi\) 仍無偏置。不把 prev 的 \(z\) 拼進特徵。

## 6. 最小判別實驗

沿用 §9.2 的探針（固定 S37 特徵、zgz、ridge 與小 MLP，不訓練完整跟蹤器），四個特徵，同一批包：

1. 2D 殘差 × 3D 槓桿（已有，7.09° / 6.96°）。
2. 上一項再加一個在手上訓練的單目事件深度頭的每部位殘差。深度頭只許用事件，不許用 prev 深度當輸入。
3. 同一項加 oracle 深度，\(\sigma=1\,\mathrm{cm}\) 與 \(2\,\mathrm{cm}\)（已有掃描，作上界）。
4. 2D 殘差 × 3D 槓桿，再加包內時間矩，不加任何深度。

預期：

- (2) 相對 (1) 的降幅小於 1.2°，或降幅不大於把 (2) 的深度殘差打亂後的降幅。則 H1 在本數據上被否定，深度擴展維持關閉。
- (4) 相對 (1) 的降幅小於 0.3°，或只出現在擾動子集而乾淨子集變差（常數增益式的交換）。則 H2 被否定。
- (4) 的增益若與「只加可見性或覆蓋率、不加時間」相同，則時間沒有額外貢獻。
- (3) 在 \(\sigma=2\,\mathrm{cm}\) 已只有約 0.8–0.9°。若 (2) 的有效 \(\sigma\) 測出來大於 2 cm，與 EMVS 近距實驗一致。

否定 H1 不需要新的閉環訓練。閉環只在探針顯示存在**獨立於 prev 深度**的增益之後才值得做。

## 7. 六個創新問題

針對「單目事件深度擴展」這條被關掉的路，以及唯一還值得探針的時間矩：

1. **此前未解決的矛盾。** 出平面根旋轉需要沿視線的資訊，而單目輪廓對該方向很弱。深度看起來能補，但獨立深度在這個距離與場景裡不可測，prev 深度又是循環的。關掉深度擴展，是承認這個矛盾不能靠再加一個深度頭化解。
2. **資訊從哪來。** 雙目的資訊來自第二個光心與極線，文獻表裡單目對雙目的差距就是這份資訊。學習深度的資訊來自駕駛數據或影像教師，是先驗的重複使用。事件時間戳是同一批邊緣的法向運動，不是第三種感測器。
3. **異步、時間、稀疏性貢獻了什麼。** 在雙目裡，時間面與立體流改善的是對應（ESVO、Temporal Event Stereo），稀疏性使深度只存在於邊緣。在單目裡，時間使聚焦成為可能，但只在運動模型正確且場景點剛性時對齊；稀疏性使手掌內部沒有觀測。
4. **相對最近的方法，變化在哪。** 若只加深度頭，變化在輸入特徵，觀測模型仍是「從事件回歸一個量」，和 E2Depth 同一類，沒有新的關聯或更新律。時間矩若做成 \(\psi(0)=0\) 的乘法殘差，變化在觀測（法向流）與更新（增益乘殘差），不是新網路。
5. **為何可見性修復、時序濾波、換表示不夠，也不自動構成創新。** 可見性只說明事件在不在剪影上，不區分出平面旋轉與平移。普通時序濾波是常數增益，項目裡已經測過。把事件換成體素或 Transformer（EReFormer、Depth AnyEvent）不改變 ill-posed。所以這些都替代不了 1 cm 深度，也替代不了一個有識別力的時間實驗。
6. **什麼結果直接否定假說。** H1：探針 (2) 不比 (1) 好過 1.2°，或有效深度噪聲大於 2 cm。H2：時間矩相對無時間的 2D 殘差沒有超出乾淨樣本注入底的增益，或與不加時間的覆蓋率特徵不可區分。

## 8. 限制與誠實聲明

- 深讀篇數：**G06 為 21 篇全文**；**G07 為 12 篇**（其中 Zheng 等 2023 只讀了 HTML 的單目深度專節，DERD 的單目列與 G06 共用 PDF）。G07 未滿 20，原因見第 2 節開頭，沒有用目錄充數。
- SCS-Net、ZEST、Shiba TPAMI 2024、TSES 全文、Ye 等 IROS 2020、Distil-E2D、跨模態 IROS 2023、EMDepth、事件塌縮全文未讀。綜述 Table 4–6 的轉引已標明。
- EReFormer 讀的是 2022 預印本，不是 2024 期刊排版。Voxel-ESVIO 與 Kang 等 3D 檢測是預印本。
- 雙欄 PDF 抽取會把表打亂。凡欄位沒對齊的數字都沒有寫進結論；結論用到的厘米、米、Abs Rel、視差 MAE 均能在上文對到表號或章節。
- \(f_x\approx 196\,\mathrm{px}\) 是由 Zhou 等 2018 的寬度與 FOV 換算的，不是論文給出的標定值。1.25 cm/px 只說明該雙目平台的數量級。
- GitHub 連線逾時，沒有函式級代碼證據。
- 未跑探針、未訓練。第 6 節是給後續的判別設計，不是新的實驗結果。
