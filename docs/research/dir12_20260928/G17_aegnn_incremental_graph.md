# G17　AEGNN 與增量圖維護

> 2026-09-28。研究子代理 G17。只寫本檔；未改倉庫程式、配置、測試或 `outputs/`。未使用 GPU。
> 網格鄰域大小是本次在 `assets/mano_right.npz` 上的 CPU 計算。判別實驗是設計，本次未跑。
> 權威程式是倉庫原始碼；文獻公式只寫實際讀到的原文。讀不到全文的條目在 §1 與 §8 標明，不計入深讀。

**提要（對應任務 (a)–(d)）**

- **(a)** 消息傳遞層數為 \(L\) 時，與整圖重算等價的充要條件是：只重算「輸入或入邊發生變化的節點」在訊息方向上的 \(L\) 跳閉包，且聚合對未變節點逐位不變。因果有向邊（只從較早事件指向較晚事件、且舊節點的鄰域不含新事件）把這個閉包收成**新節點本身**；無向／雙向半徑圖則要重算整個 \(L\) 跳球。滑窗刪點、特徵空間重算 k-NN、按度歸一、圖上 BatchNorm，都會把等價性打破或改成近似。
- **(b)** S37 事件圖只依賴 \((x,y,t)\)，同一包事件換 `prev` 時節點特徵可整段快取；狀態只進讀出。方案 A 把 778 個頂點與事件放進同一張圖、跨邊依賴 `prev`：根旋轉會移動全部頂點，髒節點集合在第 0 層已是全部頂點，跨邊再把事件節點一併弄髒，增量 \(\approx\) 全圖重算。替代是：**三層 EdgeConv 保持狀態無關；狀態只進已有的廉價讀出**（路由距離／偏移／LBS），不要把跨邊送回訊息傳遞。
- **(c)** 異步的已證實貢獻是少做「每個新事件都重算整圖」的冗餘，用來在幀與幀之間提早出檢測。本任務的決策節拍是 50 ms 一包、整包一次 `forward_packet`（延遲 7.72 ms）。等價的增量排程不增加觀測、也不把 7.72 ms 變成追蹤延遲的主項。
- **(d)** 方案 A 不該用「增量會變便宜」來辯護。若要做，它是表示實驗，每次狀態改變都要付全圖代價，並重新面對「狀態進入圖拓撲」這條已測過的風險。最小判別是 CPU 上的等價容差與髒節點比例，不是再訓一個完整模型。

---

## 0. 檢索記錄

日期皆為 **2026-09-28**。Crossref 的 `total-results` 是寬鬆計數（常達數百萬），**不當成精準命中數**；下表「留下」是實際點開並核過書目的條目。arXiv 檢索頁、OpenAlex、Semantic Scholar、Unpaywall、GitHub Search 本次失敗，不把失敗寫成已覆蓋。

| # | 檢索式 | 來源 | 結果 | 篩選後留下 |
|---|---|---|---|---|
| 1 | 直接 `arxiv.org/html/2203.17149`、`/abs/2203.17149` | arXiv HTML | HTTP 200 | AEGNN，CVPR 2022 |
| 2 | `query.bibliographic=Graph-based asynchronous event processing Li` | Crossref | 寬鬆總數不可用；第 1 條即目標 | Li、Zhou、Yang、Zhang，ICCV 2021，DOI `10.1109/iccv48922.2021.00097`，題名是 **Recognition** 不是 Detection |
| 3 | `query.bibliographic=Low-latency automotive vision with event cameras Gehrig` | Crossref | 第 1 條 | Gehrig & Scaramuzza，*Nature* 629:1034–1040，2024，DOI `10.1038/s41586-024-07409-w` |
| 4 | `query.bibliographic=EvGNN event-driven graph neural network` | Crossref | 第 1 條相關 | Yang, Kneip, Frenkel，*IEEE TCASAI*，書目年 2025-03，DOI `10.1109/tcasai.2024.3520905`。**無摘要、無 OA** |
| 5 | `query.bibliographic=EGSST event graph` | Crossref + OpenReview | Crossref 容器名 *NeurIPS 37*；OpenReview `venue=NeurIPS 2024` | Wu, Sheng, Feng, Hu。PDF：`proceedings.neurips.cc/.../da733d44e4be3902d952d6c1ffcb7db6-Paper-Conference.pdf` |
| 6 | `query.title` 一批（DGCNN、RVT、SSM、GET、NeutronStream、Bi ICCV/TIP、EventNet） | Crossref | 各取第 1 條相關；中間有 HTTP 429 | 見 §1。GET、NeutronStream、TIP 只留書目 |
| 7 | 已知 arXiv id 的 `/abs/` 標題核對 | arXiv | 200（429 出現在檢索 API，不在單篇 abs） | DGCNN `1801.07829`、SplineCNN `1711.08920`、GraphSAGE `1706.02216`、MPNN `1704.01212`、GCN `1609.02907`、TGN `2006.10637`、GNNAutoScale `2106.05609`、EvolveGCN `1902.10191`、ECC `1704.02901`、Bi `1908.06648`、RVT `2212.05598`、GIN `1810.00826`、DeepGCN `1904.03751`、TGL `2203.14883`（後者 HTML/PDF 406，未深讀） |
| 8 | CVF Open Access HTML → PDF | CVF | Li、EventNet 的 HTML 只是摘要頁；PDF 全文成功 | Li ICCV 2021 PDF；EventNet CVPR 2019 PDF |
| 9 | `rpg.ifi.uzh.ch/docs/ECCV20_Messikommer.pdf`、`CVPR2024_Zubic.pdf`；Nature 文章 HTML | 實驗室頁／Nature | AsyNet PDF、Zubic PDF、DAGr HTML 全文。`Nature24_Gehrig.pdf` 為 404，改讀 Nature HTML | 三篇全文 |
| 10 | `api.github.com/repos/uzh-rpg/aegnn/git/trees/master?recursive=1` 與 raw 檔 | GitHub | 樹 103 檔；讀了 `asyncronous/conv.py`、`batch_norm.py`、`__init__.py`、`base.py`、`graph_res.py`、README | 官方異步實現。GitHub **Search** API 403，未做全庫代碼搜尋 |
| 11 | `query.title=AEGNN` 的第 2 條 | Crossref | DDCLS 2023「Event-Based Object Detection using Graph Neural Networks」 | 只入目錄，非 CCF-A，未深讀 |

**失敗因而未當成覆蓋的來源：** arXiv `/search/` 與 `export.arxiv.org/api`（429）、OpenAlex（429）、Semantic Scholar（429）、Unpaywall（422）、GitHub repository search（403）、IEEE Xplore 正文（EvGNN DOI 跳轉後空正文）。Google Scholar、ACM DL 全文本次沒有可用通路。

**篩選規則：** 優先 2023–2026 的 CCF-A 或中科院期刊一區，加上題目點名的早期基礎（AEGNN、Li、DGCNN、SplineConv、異步稀疏卷積、消息傳遞、增量／歷史嵌入）。加速器與研討會可以入目錄，但不把預印本寫成已發表，也不把讀不到的摘要補完。

**覆蓋缺口（如實）：**

- 沒有遍歷 2023–2026 全部事件 GNN。Crossref 寬鬆檢索不可當作普查。
- **EvGNN 全文與摘要都沒讀到**，因此不引用本倉庫 2026-08-29 舊調查裡的「16 µs／事件」（該調查已於 2026-09-29 刪除）。那句本次無法核實。
- 舊調查寫的「EGSST / eGSMV (2025)」：EGSST 核實為 **NeurIPS 2024**；**eGSMV 沒有獨立命中**，不採用。
- TGL（arXiv:2203.14883）與 NeutronStream（PVLDB，DOI `10.14778/3632093.3632108`）只核到書目，PDF/HTML 未取到，不深讀。
- SplineCNN 的 arXiv PDF 回 406，改讀 ar5iv HTML。Bi 的 TIP 2020 擴展（DOI `10.1109/tip.2020.3023597`）與 ICCV 2019 不是同一題名，本次只深讀 ICCV／arXiv 版。

---

## 1. 候選目錄

分區寫法：CCF 指中國計算機學會推薦目錄的通行 **2022** 口徑，**本次沒有下載目錄 PDF**，故凡寫 CCF 都帶「待目錄複核」。中科院分區同理，未核對 2025 升級版表則標待核。研討會、新刊、只讀到書目者分開放，避免把數量當成理解。

### 1.1 深讀（§2，共 20 篇）

| 標題 | 作者／機構 |  venue | 年 | 發表狀態 | 分區口徑 | 官方代碼（本次核到的連結） | 標籤 |
|---|---|---|---|---|---|---|---|
| AEGNN: Asynchronous Event-based Graph Neural Networks | Schaefer, Gehrig, Scaramuzza；蘇黎世大學 | CVPR | 2022 | 正式。DOI `10.1109/cvpr52688.2022.01205`；arXiv:2203.17149 | CCF-A（CVPR，2022 口徑待複核） | https://github.com/uzh-rpg/aegnn ；專案頁 `uzh-rpg.github.io/aegnn` | 異步 GNN、半徑圖、SplineConv |
| Graph-based Asynchronous Event Processing for Rapid Object Recognition（SlideGCN） | Li, Zhou, Yang, Zhang, Cui, Bao, Zhang；浙江大學 CAD&CG | ICCV | 2021 | 正式。DOI `10.1109/iccv48922.2021.00097` | CCF-A（ICCV，待複核） | 本次 PDF／HTML **未看到**官方庫連結，不編 | 滑窗增量卷積、像素隊列半徑搜尋 |
| Low-latency automotive vision with event cameras（DAGr） | D. Gehrig, Scaramuzza；蘇黎世大學 | Nature 629:1034–1040 | 2024-05-29 | 正式。DOI `10.1038/s41586-024-07409-w` | 非 CCF 會議。*Nature* 通行為中科院綜合 1 區 Top，**2025 升級版表待核** | https://github.com/uzh-rpg/dagr （Nature／RPG 出版物頁寫出） | 有向異步 GNN、50 ms 幀間、LUT-Spline |
| Event-based Asynchronous Sparse Convolutional Networks（AsyNet） | Messikommer, Gehrig, Loquercio, Scaramuzza | ECCV | 2020 | 正式（PDF 首頁寫 accepted ECCV 2020）。DOI `10.1007/978-3-030-58598-3_25` | CCF 2022 通行口徑 **B**（待目錄複核） | PDF 寫 https://github.com/uzh-rpg/rpg_asynet | 卷積版的局部等價更新 |
| EventNet: Asynchronous Recursive Event Processing | Sekikawa, Hara, Saito | CVPR | 2019 | 正式（CVF） | CCF-A（待複核） | 本次未在摘錄中核到庫 | 淺層遞迴、非層次 GNN |
| Dynamic Graph CNN for Learning on Point Clouds（EdgeConv） | Y. Wang 等 | ACM TOG | 2019 | 正式。DOI `10.1145/3326362`；arXiv:1801.07829 | TOG：CCF-A 期刊（待複核） | 本次 HTML 片段未摘到庫，不編 | 動態特徵 k-NN、max EdgeConv |
| SplineCNN | Fey, Lenssen, Weichert, Müller；TU Dortmund | 書目作 CVPR 2018 | 2018 | arXiv:1711.08920 已讀。CVPR 2018 依 AEGNN 參考文獻 [13] 轉引，**待 CVF 頁複核** | 若屬 CVPR 則 CCF-A（待複核） | 原文寫 https://github.com/rusty1s/pytorch_geometric | B-樣條核、偽座標 |
| Graph-Based Object Classification for Neuromorphic Vision Sensing | Bi, Chadha, Abbas, Bourtsoulatze, Andreopoulos；UCL | ICCV | 2019 | 正式。DOI `10.1109/iccv.2019.00058`；arXiv:1908.06648 | CCF-A（待複核） | 未核 | 事件半徑圖、非異步 |
| Inductive Representation Learning on Large Graphs（GraphSAGE） | Hamilton, Ying, Leskovec | NeurIPS／NIPS | 2017 | 正文寫投 NIPS 2017；arXiv:1706.02216。主會頁碼本次未再核 | CCF-A（待複核） | 未核 | \(K\) 跳聚合、鄰域採樣 |
| Neural Message Passing for Quantum Chemistry | Gilmer 等；Google Brain | 通行作 ICML 2017 | 2017 | arXiv:1704.01212 已讀。會議頁本次 HTML 未見，**待核** | 若屬 ICML 則 CCF-A（待複核） | 未核 | 消息傳遞模板 |
| Semi-Supervised Classification with GCNs | Kipf, Welling | 通行作 ICLR 2017 | 2017 | arXiv:1609.02907 已讀。會議頁待核 | 若屬 ICLR 則 CCF-A（待複核） | 未核 | 度歸一使局部更新不封閉 |
| How Powerful are GNNs（GIN） | Xu, Hu, Leskovec, Jegelka | 通行作 ICLR 2019 | 2019 | arXiv:1810.00826 已讀。會議頁待核 | 同上 | 未核 | sum 可增量、mean 非單射 |
| Temporal Graph Networks for Deep Learning on Dynamic Graphs | Rossi 等；Twitter | ICML **Workshop** | 2020 | arXiv:2006.10637。AEGNN 書目標 ICMLW 2020。**不是主會** | 研討會，不作 CCF-A | 未核 | 節點記憶，不是事件相機 |
| GNNAutoScale | Fey, Lenssen 等；TU Dortmund | 通行作 ICML 2021 | 2021 | arXiv:2106.05609 已讀。HTML 頁首未見 proceedings 題名，**會議待核** | 若屬 ICML 則 CCF-A（待複核） | 文中 PyGAS／PyG | 歷史嵌入是**近似**，不是等價 |
| Dynamic Edge-Conditioned Filters（ECC） | Simonovsky, Komodakis | 書目作 CVPR 2017 | 2017 | arXiv:1704.02901。CVPR 依 AEGNN [51] 轉引，待 CVF 複核 | 若屬 CVPR 則 CCF-A | 未核 | 邊標籤生成濾波器；\(1/|N(i)|\) |
| EvolveGCN | Pareja 等；MIT-IBM／IBM | 通行作 AAAI 2020 | 2020 | arXiv:1902.10191 已讀。會議頁待核 | 若屬 AAAI 則 CCF-A（待複核） | 未核 | 演化的是權重，不是 k-hop 重算 |
| Recurrent Vision Transformers for Object Detection with Event Cameras | M. Gehrig, Scaramuzza | CVPR | 2023 | 正式。DOI `10.1109/cvpr52729.2023.01334`；arXiv:2212.05598 | CCF-A（待複核） | 未核 | 延遲對照，不是 GNN |
| EGSST: Event-based Graph Spatiotemporal Sensitive Transformer | Wu, Sheng, Feng, Hu | NeurIPS | 2024 | 正式（OpenReview venue + proceedings PDF） | CCF-A（NeurIPS，待複核） | 本次只核到 proceedings PDF，未核代碼庫 | 無向半徑圖 + GCN／GAT，批處理 |
| State Space Models for Event Cameras | Zubic, M. Gehrig, Scaramuzza | CVPR | 2024 | 正式（PDF 首頁：accepted CVPR 2024） | CCF-A（待複核） | RPG：`docs/CVPR2024_Zubic.pdf` | 非 GNN；點名 GNN 次採樣的延遲代價 |
| DeepGCNs: Can GCNs Go as Deep as CNNs? | Li, Müller, Thabet, Ghanem（作者以 HTML 為準） | 預印本已讀；會議收錄待核 | 2019 | arXiv:1904.03751v2 | 待核 | 文中專案頁 `sites.google.com/view/deep-gcns` | 殘差 \(\mathcal{G}_{l+1}=\mathcal{F}+\mathcal{G}_l\)；動態 k-NN 開銷 |

### 1.2 看到書目、未深讀

| 標題 | 書目 | 為何不深讀 |
|---|---|---|
| EvGNN: An Event-Driven Graph Neural Network Accelerator for Edge Vision | Yang, Kneip, Frenkel；*IEEE TCASAI*；DOI `10.1109/tcasai.2024.3520905`；Crossref 發行日期 2025-03 | 新刊，CCF／中科院 **待核**。IEEE 頁無正文、Crossref 無摘要。**不能**把它寫成已讀，也不能沿用舊調查的 16 µs |
| Graph-Based Spatio-Temporal Feature Learning for Neuromorphic Vision Sensing | Bi 等；IEEE TIP 2020；DOI `10.1109/tip.2020.3023597` | 與 ICCV 2019 題名不同，疑為擴展。TIP 為 CCF-A（待複核）、中科院分區待核。全文未取 |
| GET: Group Event Transformer for Event-Based Vision | ICCV 2023；DOI `10.1109/iccv51070.2023.00555` | 只核書目。Zubic 表裡拿它與 RVT、SSM 比 20 Hz 檢測，但是 Transformer 不是 GNN |
| NeutronStream | PVLDB；DOI `10.14778/3632093.3632108`；Crossref 年 2023 | 滑窗動態 GNN **訓練**系統。PVLDB 通行 CCF-A（待複核）。PDF 未取 |
| TGL | arXiv:2203.14883 | HTML/PDF 406。不深讀 |
| Event-Based Object Detection using GNNs | DDCLS 2023；DOI `10.1109/ddcls58216.2023.10166491` | 非 CCF-A／一區 Top，只作缺口記錄 |

---

## 2. 深讀

每篇：原問題、關鍵公式（原文記號）、觀測與假設、代碼、對本任務。公式出處寫我讀到的章節。

### 2.1 AEGNN（CVPR 2022）

**原問題：** 事件 GNN 若每來一個事件就重算整圖，稀疏性與時間解析度就浪費了。能否同步訓練、測試時改成異步，輸出仍與同步前向相同？

**圖（§4.1）：** 事件 \(e_i=(x_i,y_i,t_i,p_i)\)。均勻次採樣因子 \(K=10\)。節點位置 \(\mathbf{X}_i=(x_i,y_i,t_i^*)\)，\(t_i^*=\beta t_i\)。\(\|\mathbf{X}_i-\mathbf{X}_j\|\le R\) 則連邊，且 \(|\mathcal{N}(i)|\le D_{max}\)。初始特徵 \(\mathbf{x}_i=p_i\)，邊特徵為相對位置除以 \(R\)。

**消息傳遞（§3，式 (1)(2)）：**

\[
\mathbf{z}_i=\sum_{j\in\mathcal{N}(i)}\psi_\Theta(\mathbf{x}_i,\mathbf{x}_j,\mathbf{e}_{ij}),\qquad
\hat{\mathbf{x}}_i=\gamma_\Theta(\mathbf{x}_i,\mathbf{z}_i).
\]

聚合可換成對稱函數（max／min／sum）。他們實際用 SplineConv（附錄式 (8)）：

\[
(g_n * f)(i)=\frac{1}{|N(i)|}\sum_{l=1}^{M_{in}}\sum_{j\in N(i)} f_l(j)\, g_{n,l}(u(i,j)).
\]

**異步規則（§4.2，式 (6)(7)）：** 新事件 \(i'\) 只讓其 1 跳子圖在該層重算；第 \(N\) 層要更新的是 \(N\) 跳子圖 \(\mathcal{H}_N(i')\)。正文寫：只處理這個子圖，得到的激活與式 (2) 的整圖結果相同。池化是體素 max（體素 \(12\times 16\times 16\)），只重算被碰到的體素。

**觀測：** N-Cars 準確率 0.945、0.47 MFLOP／事件；N-Caltech101 0.668、7.31 MFLOP／事件（表 1）。摘要聲稱相對當時異步方法最高約 11 倍 FLOPs、相對標準 GNN 約 8 倍計算延遲。識別網 Spline 核 \(k=2\)，檢測網 \(k=8\)（附錄 §8.1）。每塊是 SplineConv + ELU + **BatchNorm**。

**代碼（`uzh-rpg/aegnn`，本次讀 raw）：**

- `aegnn/asyncronous/conv.py` 的 `__graph_processing`：新事件用 `torch.cdist(pos_all, pos_new) <= radius` 找鄰域，再把邊**和反向邊拼在一起**（雙向）。更新集合是新節點的鄰居加上特徵已變節點的 1 跳（`k_hop_subgraph(..., num_hops=1)`）。所以論文的「1 跳子圖」在實現裡是**雙向半徑圖**，不是「只算新節點」。
- 同檔 `__check_support`：帶歸一化的 `GCNConv`、帶 bias 或 root weight 的 `SplineConv` **直接 NotImplemented**。度歸一與根項會讓「只改一條訊息」不封閉。
- `batch_norm.py` 註解寫明：新節點會改變全體特徵分佈，因此異步 BN **用初始化時的 mean／var 近似**，前提是新事件遠少於初始事件。這不是逐位等價。
- `graph_res.py`：識別網通道 `(1,8,16,16,16,32,32,32,32)`，與附錄一致；`bias=False, root_weight=False`。
- 我讀到的 `conv.py` **只有插入、沒有滑窗刪點**。論文 §4.2 提到舊事件離開窗口，這份處理函式沒有對應路徑。

**對本任務：** AEGNN 證明的是計算排程，不是手部姿態的新觀測。等價依賴「變了的節點的 \(L\) 跳」加上若干算子限制。S37 的因果邊比 AEGNN 的雙向半徑更窄，見 §5。S37 沒有 BN，這一點比 AEGNN 的部署更接近逐位等價。

### 2.2 Li 等 SlideGCN（ICCV 2021）

**原問題：** 半徑圖 GNN 若用滑窗，每進一個事件就要重算窗內所有節點。如何只傳播真正改過的特徵，並讓半徑搜尋在插入／刪除下仍然便宜？

**圖（§3.1）：** 與 Bi 相同的半徑鄰域。時間軸先縮放。度數上限 \(D_{max}\)。邊屬性是相對笛卡爾座標。

**卷積（§3.2 式 (3)，§4.1 式 (4)(5)）：**

\[
(f\otimes g)(i)=\sum_{j\in E(i)} f(j)\, h_\theta,\quad h_\theta=h_\theta(f(i),f(j),e_{ij}).
\]

\[
f_{n+1}(i)=\sum_{j\in N(i)} f_n(j)\, h_\theta.
\]

增量：

\[
f_{n+1}^{t+1}(i)=f_{n+1}^{t}(i)+\Delta_{n+1}(i),\qquad
\Delta_{n+1}(i)=\sum_{(j,i)\in E_{n+1}}\bigl(f_n^{t+1}(j)-f_n^{t}(j)\bigr) h_\theta.
\]

\(E_{n+1}\) 是指向「被修改節點」的有向邊。節點分成刪除 \(V^{del}\)、新增 \(V^{add}\)、以及落在二者感受野裡的 \(V^{up}\)。式 (6)(7) 用集合運算把這三類從第 \(n\) 層推到第 \(n+1\) 層。池化會改拓撲（圖 2），所以不能只在第 0 層的新節點上停住。

**半徑搜尋（§4.2）：** 事件在像素網格上，不在一般 3D 連續空間。他們用像素隊列做兩段式半徑搜尋，並寫插入／刪除代價為 \(O(1)\)，相對 k-d tree 在頻繁插刪下會失衡。這是**動態圖維護**的直接設計，不是 GNN 層的數學。

**觀測與假設：** 相對「每次滑窗都重算全圖」，他們稱計算複雜度最高約降 100 倍，識別精度與批式圖方法相當，並用穩定狀態做提早識別。假設是 \(h_\theta\) 對未變的 \((f(j),e_{ij})\) 不變，因此差分和等於重算和。若 \(h_\theta\) 含 softmax 或 \(1/|N(i)|\) 且度數變了，這個差分不成立；他們的式 (5) 是把 \(h_\theta\) 乘在**特徵差**上，度數歸一沒有出現在該式裡。

**代碼：** 本次 CVF PDF 未給官方庫，無代碼核對。

**對本任務：** 式 (5) 是「和式聚合 + 只加髒邊」的等價模板，和 S37 的 **mean**（除以邊數）不完全相同：邊數不變時 mean 的差分才封閉。滑窗**刪除**在 Li 的公式裡是一等公民；S37 現在的 `_edges` 是每包重建，沒有這條路徑。像素隊列依賴「事件落在離散像素、時間有序」，手部 50 ms 包滿足；它不解決「頂點座標隨姿態連續變動」的跨邊。

### 2.3 DAGr（Nature 2024）

**原問題：** 車載 RGB 的幀率造成 22–33 ms 以上的盲區。事件若先堆成幀再用 CNN，計算延遲把感測器的時間解析度吃掉。能否用低幀率影像的特徵，加上事件上的異步 GNN，在幀與幀之間更新檢測，且異步輸出與整圖前向相同？

**圖（Methods，式 (4)）：** 節點 \(\mathbf{n}_p^i=(\hat{\mathbf{x}}_i,\beta t_i)\)，\(\beta=10^{-6}\)，特徵為極性。連邊條件是

\[
(i,j)\in E \quad\text{若}\quad \|\mathbf{n}_p^i-\mathbf{n}_p^j\|_\infty < R \ \text{且}\ t_i<t_j.
\]

\(R=0.01\)，每節點最多 16 個鄰居。邊特徵只用 \(xy\) 相對位置，映到 \([0,1]^2\)。**時間有序使圖有向。**

**Spline（式 (5)）：**

\[
\mathbf{n}_f^{\prime i}=W\mathbf{n}_f^i+\sum_{(j,i)\in E} W(e_{ij})\,\mathbf{n}_f^j.
\]

\(W(e_{ij})\) 是 \(d=1\)、\(k=5\) 的二維 B-樣條。部署時改成查表 \(W_{ij}=\mathrm{LUT}(dx,dy)\)（式 (13)(14)），因為事件在網格上、鄰域有界，相對位移只有有限種。BN 在快取後融進權重，從計數裡去掉。

**有向與雙向（「Update propagation」「Directed event graph」）：** 有向圖每一層要更新的訊息數保持常數，「不再把更新擴散到原先沒碰到的節點」。雙向邊會讓 \(k\) 跳子圖逐層變大。普通體素池化會把有向邊併成雙向（式 (7)）；他們的 directed voxel pooling（式 (8)(9)）用時間 max 當節點時間、並丟掉時間逆序的邊。正文寫：有向時除了罕見的邊翻轉，**每一層最多更新一個節點**。

**剪枝：** max-pool 的每個輸出通道最多只由一個輸入節點決定（式 (15)–(17)）。若髒輸入不在這個argmax 集合、位置四捨五入也不變，就停止向下傳。他們在 Gen1 上看到約 **73%** 的更新被剪掉。

**延遲數字（同一節 Timing）：** 50,000 個事件，Quadro RTX 4000 上稠密 GNN 30.8 ms，異步 8.46 ms，**牆鐘只快 3.7 倍**。討論寫明：FLOPs 上相對稠密方法可以差四個數量級，牆鐘沒有同比例下降。建圖：50,000 節點 1.75 ms，單事件插入 0.3 ms（同一 GPU）。混合模型最快一檔 DAGr-S+ResNet-50 為 9.6 ms；並寫 **MFLOPS／事件與運行時間在這個量級上不相關**（PyG 的批處理開銷）。

**精度與有向的代價：** 純事件、directed pooling 的 mAP 降到 18.35，計算 0.31 MFLOPS／事件；他們對 SOTA 比較時改回非此配置。加上影像後，有向邊把計算降約 91%，mAP 只降約 2 個百分點（「Using images and events」）。訓練標籤在影像時刻與 50 ms 事件窗末端；GNN 學的是更新影像分支的檢測。

**對本任務：** DAGr 把「因果有向 ⇒ 每層只碰新節點」寫清楚了，這正是 S37 `_edges` 的方向。他們的 50 ms 與本協議的包長相同，但用途是**填兩幀之間的盲區並多次輸出**。本協議一包只呼叫一次 `forward_packet`。DAGr 也給出方案 A 的反例結構：一旦邊變成雙向，或節點**位置**變了（式 (13) 下位置一變就要重算該節點的整個和，而不只是一條訊息），髒集合就不再是單節點。姿態一變，778 個頂點的位置一起變，不屬於「插入一個事件」。

### 2.4 AsyNet（ECCV 2020）

**原問題：** 把事件收成類影像張量再用 CNN，空間與時間稀疏性都被丟掉。能否把訓練好的同步網路變成異步網路，輸出相同、計算嚴格更少？

**輸入遞推（式 (2)）：** 新事件只在少數位置加 \(\Delta_i(c)\)。他們把能這樣更新的表示叫 sparse recursive representation；直方圖、事件隊列、時間影像每個新事件只改一個像素。

**局部更新（式 (5)–(9)）：** 感受野 \(\mathcal{F}_n\) 與 rulebook 逐層擴張，只含仍活躍的位置。單事件引起的激活差沿 rulebook 加上去，再過非線性。補充材料裡證明：逐事件處理 \(N\) 個事件等價於一次處理全部。新變活躍的位置改走完整稀疏卷積，變不活躍的位置置 0。

**複雜度：** 稠密卷積的更新區域隨深度按核寬**二次**長；稀疏活躍點的增長用分形維 \(\gamma<2\) 描述（式 (11)）。相對高延遲網路，識別／檢測上計算最高約降 20 倍。

**對本任務：** 這是 GNN 增量規則的卷積前身，AEGNN §2 明確說自己在補它的兩點（層次學習、保留時間而不是直方圖）。等價的前提是：非線性作用在**更新後的預激活**上（先加 \(\Delta\) 再 \(\sigma\)），不是把 \(\sigma\) 分配進和式。S37 的 `relu` 在每條邊上，殘差加在外面，和這個「先聚合再非線性」不同，但「未變鄰域 ⇒ 中心不變」仍然成立。AsyNet **不**處理非剛體姿態狀態。

### 2.5 EventNet（CVPR 2019）

**原問題：** 在事件率上遞迴處理數萬個因果事件，而不是每次重跑 PointNet 式的整批 MLP。

**讀到的機制：** PDF 正文寫：用時間編碼把對因果事件序列的依賴寫成遞迴；對每個新事件用 LUT 更新狀態；max 與一個對時間差大致局部常數的函數 \(c\) 複合後可以遞迴。圖中的公式是嵌入圖片，**我沒有把那些公式當成可抄的原文**。Li 等 §2 對它的轉述（近似、無層次結構、預先按空間座標與極性算節點特徵）與 AEGNN 對 [49] 的評語一致：不是可堆疊的 GNN。

**對本任務：** 它說明「因果 + 預計算」可以讓單事件更新很便宜，但也說明淺層遞迴不等於 S37 這種三層 EdgeConv。不能把 EventNet 的提早輸出直接當成手部閉環會更好。

### 2.6 DGCNN／EdgeConv（TOG 2019）

**原問題：** 點雲沒有固定拓撲。在特徵空間裡每層重算近鄰，能否讓感受野變大仍保持稀疏？

**公式（§3，他們的 (9) 附近）：**

\[
e'_{ijm}=\mathrm{ReLU}\bigl(\boldsymbol{\theta}_m\cdot(\mathbf{x}_j-\mathbf{x}_i)+\boldsymbol{\phi}_m\cdot\mathbf{x}_i\bigr),\qquad
x'_{im}=\max_{j:(i,j)\in\mathcal{E}} e'_{ijm}.
\]

§3.2：每一層用該層特徵空間的 k 近鄰重算 \(\mathcal{G}^{(l)}\)。動態圖使感受野可以到點雲直徑。

**對本任務：** S37 的 `EdgeConv` **借用了** \(\mathbf{h}_j-\mathbf{h}_i\)，但有三處不同，這三處決定增量是否等價：

1. 聚合是 **mean**，不是 max（`event_gnn.py` 第 81–83 行）。
2. 邊特徵多了顯式 \((dx,dy,dt)\)，圖在輸入座標上建一次，**三層共用同一個 `idx`**（第 219–222 行）。不是 DGCNN 的逐層動態圖。
3. 殘差是 \(\mathbf{h}\leftarrow\mathbf{h}+\mathrm{mean}(m)\)，不是把中心項 \(\phi\cdot x_i\) 放進 max。

若改回特徵空間 k-NN，新事件的特徵會改變其他節點的鄰域，因果快取失效。DeepGCN 附錄也把動態 k-NN 的運行時間單獨當成開銷（§2.20）。

### 2.7 SplineCNN（arXiv:1711.08920）

**原問題：** 在不規則幾何上做卷積，核要隨相對位置連續變化，計算又不要隨核尺寸爆炸。

**公式：** 偽座標 \(\mathbf{u}(i,j)\in[0,1]^d\)。核

\[
g_l(\mathbf{u})=\sum_{\mathbf{p}\in\mathcal{P}} w_{\mathbf{p},l}\, B_{\mathbf{p}}(\mathbf{u}),\qquad
B_{\mathbf{p}}(\mathbf{u})=\prod_{i=1}^{d} N_{i,p_i}^{m}(u_i).
\]

ar5iv 抽出的卷積式 (3) 是 \(\sum_{j\in\mathcal{N}(i)} f_l(j)\cdot g_l(\mathbf{u}(i,j))\)。**該片段沒有 \(1/|N(i)|\)**。AEGNN 附錄式 (8) 寫的是帶 \(1/|N(i)|\) 的形式。兩處不一致時，以各自原文為準：Fey 的這段是不歸一的和；AEGNN 實現採用歸一化，因此**度數一變，該節點的輸出就要整段重算**，不能只加一條訊息。B-樣條局部支撐使每個鄰居只碰到 \((m+1)^d\) 個控制點。

**對本任務：** S37 用線性層吃 \([\mathbf{h}_j-\mathbf{h}_i;\mathrm{dp}]\)，是把「核依賴相對位置」做成邊 MLP，而不是 B-樣條。增量條件相同：\(\mathrm{dp}\) 或鄰域成員一變，該中心的訊息就失效。

### 2.8 Bi 等（ICCV 2019）

**原問題：** 把一段事件收成時空圖再做殘差圖卷積，能否比稠密 CNN 更省、分類更好？

**圖（式 (1)(2)）：** \(\{e_i\}=\{x_i,y_i,t_i,p_i\}\)，非均勻網格採樣得到 \(M\ll N\)。連邊

\[
d_{i,j}=\sqrt{\alpha(|x_i-x_j|^2+|y_i-y_j|^2)+\beta|t_i-t_j|^2}\le R.
\]

補充材料把 \(R\) 試到 \(\{1.5,3,4.5,6\}\)，\(R=3\) 之後精度不再升、計算量明顯升；\(D_{max}=32\)。這是**整段事件一次建圖**，沒有異步更新規則。

**對本任務：** S37 的「時間窗內 k-NN」是這個半徑圖的截斷：只在前 `window=32` 個事件裡取 \(k=8\)，代價 \(O(N\cdot\mathrm{window})\) 而不是全對半徑。`event_gnn.py` 寫明：只有真實鄰居落在這個時間窗內時，它才與全搜尋精確相同。Bi 的圖不因果，新事件可以成為舊事件的鄰居。

### 2.9 GraphSAGE（NeurIPS 2017）

**公式（算法 1）：** 對 \(k=1\ldots K\)，

\[
\mathbf{h}_{\mathcal{N}(v)}^{k}=\mathrm{aggregate}_k\bigl(\{\mathbf{h}_u^{k-1}:u\in\mathcal{N}(v)\}\bigr),\qquad
\mathbf{h}_v^{k}=\sigma\bigl(\mathbf{W}^k\cdot\mathrm{concat}(\mathbf{h}_v^{k-1},\mathbf{h}_{\mathcal{N}(v)}^{k})\bigr),
\]

再做 \(\ell_2\) 歸一。\(K\) 層的輸出只依賴 \(K\) 跳鄰域。測試時可以對沒見過的節點套同一組聚合。

**對本任務：** 「只重算受影響的 \(k\) 跳」的感受野定義來自這裡，不是來自事件相機。採樣聚合是近似；S37 在窗內是精確 top-k，沒有再採樣。\(\ell_2\) 歸一是逐節點的，不破壞「鄰域沒變則輸出沒變」。

### 2.10 MPNN（arXiv:1704.01212）

**公式（§2 式 (1)(2)(3)）：**

\[
m_v^{t+1}=\sum_{w\in N(v)} M_t(h_v^t,h_w^t,e_{vw}),\qquad
h_v^{t+1}=U_t(h_v^t,m_v^{t+1}),\qquad
\hat y=R(\{h_v^T\}).
\]

\(R\) 必須對節點置換不變。

**對本任務：** S37 的 EdgeConv 是這個模板的一個例子：\(M\) 是 \(\mathrm{ReLU}(W[\mathbf{h}_j-\mathbf{h}_i;\mathrm{dp}])\)，\(U\) 是殘差加 mean，\(R\) 在讀出側是 mean\(\|\)max 再加路由池化。等價增量維護的是 \(m_v\) 與 \(h_v\)，不是 \(R\) 的全局統計。全局 mean 在節點集合只增一個、舊 \(h\) 不變時有精確更新 \((n\bar h+h_{new})/(n+1)\)；max 要保留當前最大值。S37 每次前向都重算這兩個池化（第 226–230 行），沒有維護運行中的 mean／max。

### 2.11 GCN（arXiv:1609.02907）

**公式（式 (8)）：** 重歸一化

\[
Z=\tilde D^{-\frac12}\tilde A\tilde D^{-\frac12} X\Theta,\qquad \tilde A=A+I.
\]

\(\tilde D_{ii}=\sum_j \tilde A_{ij}\)。複雜度 \(\mathcal{O}(|\mathcal{E}|FC)\)。

**對本任務：** 一條新邊同時改兩個端點的度，於是**所有**經過這兩個端點的歸一化係數都變，不是一條訊息的事。AEGNN 代碼拒絕 `normalize=True` 的 GCNConv，就是這個原因。S37 的 mean 用的是該節點自己的邊掩碼和，不是對稱度歸一；舊節點邊集不變時，這個標量不變。

### 2.12 GIN（arXiv:1810.00826）

**引理 5：** 在可數全集、有界多重集上，存在 \(f\) 使 \(h(X)=\sum_{x\in X} f(x)\) 對每個多重集唯一；任意多重集函數可寫成 \(\phi(\sum f(x))\)。**mean 聚合不是多重集上的單射函數。**

**對本任務：** 和式聚合在「加上一個新鄰居、舊項不變」時可以精確增量；mean 額外依賴基數。S37 選擇 mean 是為了與邊數無關的尺度，增量時必須同時維護邊數。這不妨礙等價，只是更新量是 \((\mathrm{sum}+\Delta)/(\mathrm{count}+\delta)\)，不是把 \(\Delta\) 加到舊 mean 上。

### 2.13 TGN（ICML Workshop 2020）

**公式（Message Function）：** 互動 \(\mathbf{e}_{ij}(t)\) 產生

\[
\mathbf{m}_i(t)=\mathrm{msg}_s\bigl(\mathbf{s}_i(t^-),\mathbf{s}_j(t^-),\Delta t,\mathbf{e}_{ij}(t)\bigr),
\]

目標節點有對稱的一條。記憶 \(\mathbf{s}_i\) 隨事件更新。附錄有刪除事件。

**對本任務：** AEGNN 相關工作把 TGN 當成「動態圖上的淺層嵌入、沒有端任務層次」。TGN 的狀態是**每個節點的記憶**，新事件按定義要改參與節點的記憶。這和 S37「事件圖狀態無關」相反，也和方案 A「頂點狀態進圖」更像：狀態一進節點，下次聚合就不能當快取。RVT 補充材料列了「與 TGNN 的關係」，但是檢測主幹仍是卷積 + 循環，不是這套記憶。

### 2.14 GNNAutoScale（arXiv:2106.05609）

**機制：** 歷史嵌入 \(\tilde{\mathbf{h}}_v^{(\ell)}\) 用來近似計算圖的整棵子樹，從離線儲存裡取，而不是每次重算。單批 GPU 記憶體 \(\mathcal{O}(|\bigcup_{v\in\mathcal{B}}\mathcal{N}(v)\cup\{v\}|\cdot L)\)，隨層數線性而不是指數。正文明確把它和「不用歷史近似的精確嵌入」分開。

**對本任務：** 這是**訓練時**少算 \(k\) 跳的近似，不是 AEGNN／Li 那種與整圖重算逐位相同的推理規則。不能把「歷史嵌入」當成方案 A 的快取：姿態一變，歷史裡的頂點特徵就是錯的，誤差會留在未重算的跳上。判別實驗若用近似維護，必須單測漂移；精確因果維護不應該有隨事件數增長的系統漂移。

### 2.15 ECC（arXiv:1704.02901，式 (1)）

\[
X^l(i)=\frac{1}{|N(i)|}\sum_{j\in N(i)} F^l\bigl(L(j,i);w^l\bigr)\, X^{l-1}(j)+b^l
=\frac{1}{|N(i)|}\sum_{j\in N(i)}\Theta_{ji}^l X^{l-1}(j)+b^l.
\]

\(\Theta_{ji}^l\) 由邊標籤動態生成，不是固定卷積核。

**對本任務：** 方案 A 的跨邊若把「事件相對 prev 投影頂點的偏移」當成邊標籤 \(L(j,i)\)，則 \(\Theta\) 隨 `prev` 變，所有跨邊的訊息同時失效。這和 ECC 的表達力是同一件事，也和快取失效是同一件事。\(1/|N(i)|\) 使度數變化時中心節點必須整段重算。

### 2.16 EvolveGCN（arXiv:1902.10191）

**機制：** 圖序列 \((A_t,\ldots)\) 上，用 RNN 演化 **GCN 的權重**，而不是演化節點隱藏態（-H 與 -O 兩種把權重放進 RNN 的方式）。節點集合允許隨時間變。

**對本任務：** 這不是「只重算 \(k\) 跳」。把它搬進手部追蹤，會變成另一個隨時間改濾波器的模型，不解決 50 ms 包裡的等價重算，也碰不到根旋轉的觀測。不採用。

### 2.17 RVT（CVPR 2023）

**原問題：** 事件檢測的精度已高，但推理常常超過 40 ms。能否把循環視覺主幹的推理時間降一個數量級同時保住精度？

**機制：** 每級用卷積先驗、局部與膨脹全局自注意力、循環時間聚合。事件仍先變成張量，不是圖。摘要：推理時間約降到原先的 1/6。

**對本任務：** 用來對照 (c)。RVT 的延遲改進來自**主幹在固定頻率的張量上變輕**，不是來自與整圖重算等價的異步圖。Zubic 表 1 在 **20 Hz**（50 ms）上比較 RVT 與 SSM。本任務的 7.72 ms 已經落在 50 ms 以內；再把圖改成異步，對齊的是 RVT 想壓的那類「主幹比幀間隔還長」的問題，而 S37 目前不是那個問題。

### 2.18 EGSST（NeurIPS 2024）

**原問題：** 事件稀疏、異步，幀式檢測不合適。用圖保留時間與空間，再用 Transformer 做檢測。

**圖（式 (1)(2)）：** \(event_i=(x_i,y_i,t_i,p_i)\)，\(t_i^*=\beta(t_i-t_0)\)。無向半徑

\[
e_{ij}=1 \quad\text{若}\quad \|c_i-c_j\|\le R.
\]

沒有 \(t_i<t_j\)。連通子圖按節點數過濾做下採樣；10,000 個事件約留下 73%（他們在所用數據上的初步測試）。GCN 打在子圖上。SSM 用 \(f(x,y)=x/y\)：

\[
\mathrm{EGM}=f(N,\Delta t^*),\qquad \mathrm{ELM}_k=f(n_k,\delta t_k^*),\qquad h_k=\mathrm{ELM}_k/\mathrm{EGM}.
\]

再對子圖代表點做 k-NN 與 GAT。TAC 用全局特徵產生 Query（式 (9)）。

**對本任務：** 2024 的頂會事件圖工作**沒有**走 AEGNN 的異步等價更新；它是另一種批式圖。半徑圖無向，和因果快取不相容。EGM／ELM 是事件計數除以時間跨度，不是手部剛體速度，也不能當成第二視角或深度。圖是檢測用的輪廓分團，不是 MANO 狀態。

### 2.19 Zubic 等 SSM（CVPR 2024）

**原問題：** 事件檢測裡的 RNN／Transformer 訓練慢，而且換推理頻率就要重來。連續時間線性狀態空間能否在變頻率下保持檢測？

**公式：** 連續時間線性 SSM，參數 \(A\in\mathbb{R}^{P\times P}\)、\(B\)、\(C\)；離散化成線性遞迴（S4／S4D／S5）。事件先變成張量。他們在相關工作裡寫 GNN 的困難：要在大時空體裡傳訊息，大而慢的物體尤其難；為了低延遲而猛烈次採樣會丟掉事件。表 1 是 Gen1 與 1 Mpx 的 **20 Hz** 檢測，不是逐事件延遲。

**對本任務：** 獨立來源支持兩點。(1) 事件 GNN 的低延遲往往靠次採樣，次採樣是資訊損失，不是免費的。(2) 和本任務對齊的時間尺度仍是數十毫秒的輸出節拍（20 Hz），不是微秒級逐事件閉環。SSM 的隱藏態是時間遞迴，換 `prev` 不會自動保持「事件編碼與姿態無關」。

### 2.20 DeepGCN（arXiv:1904.03751）

**公式（式 (3)）：**

\[
\mathcal{G}_{l+1}=\mathcal{F}(\mathcal{G}_l,\mathcal{W}_l)+\mathcal{G}_l.
\]

殘差是整張圖的頂點加法。他們把 ResNet／DenseNet／膨脹卷積搬到 GCN，因為堆多層圖卷積不容易訓（當時 SOTA 常常不超過 3 層）。附錄有動態 k-NN 的運行時間。

**對本任務：** S37 的殘差在節點上：\(\mathbf{h}\leftarrow\mathbf{h}+\mathrm{mean}(m)\)，與式 (3) 同類。殘差**不**縮小髒集合：只要 \(\mathcal{F}\) 的輸入髒了，輸出就髒，再加上一份舊 \(\mathcal{G}\)。它讓深層好訓，不讓增量更便宜。DAGr 用殘差把 GNN 加深後，才更需要有向邊與剪枝，否則 \(L\) 一大，雙向圖的髒集合會鋪滿。

---

## 3. 可觀測性、資訊來源與失效條件

對象是單目事件、50 ms 包、非剛性手、MANO 51 維。這裡只談圖維護帶來或帶不來的資訊，不把已測的根旋轉診斷重寫一遍。

**事件圖上看得到的：** 像素 \((x,y)\)、極性、包內時間、以及 S37 的 7 維 token（含對數間隔與兩個 SAE 年齡）。邊是這些量的差。它們是亮度變化的時空樣本，不是深度，也不是第二台相機。Bi、AEGNN、DAGr、EGSST 的圖都是這一類。

**事件圖上看不到、方案 A 想從 `prev` 借的：** 頂點在相機系的三維位置與投影。那是上一次狀態的前向運動學，是歷史估計，不是新的獨立測量。多個頂點共享同一次 FK，不能當成多次獨立觀測。手不是剛體場景；DAGr 的有向事件圖假設的是「事件之間的時間序」，不是「手是剛體」。

**失效條件（增量規則本身）：**

| 條件 | 後果 |
|---|---|
| 邊雙向或無向（AEGNN 代碼、EGSST 式 (2)、Bi 式 (2)） | 新事件進入舊節點的鄰域，髒集合至少是 1 跳，逐層變 \(L\) 跳 |
| 只保留時間正向的邊（DAGr 式 (4)、S37 `_edges`） | 舊節點的入鄰域不含新事件；插入時只有新節點要算 |
| 滑窗把舊事件刪掉（Li 式 (6) 的 \(V^{del}\)；AEGNN 正文有、`conv.py` 未實現） | 曾以它為鄰居的節點變髒，再沿訊息方向傳 \(L\) 層 |
| 特徵空間每層重算 k-NN（DGCNN §3.2） | 鄰域依賴全體特徵，單點插入可以改很多人的邊 |
| \(1/|N(i)|\) 或 GCN 的 \(\tilde D^{-1/2}\)（ECC 式 (1)、GCN 式 (8)、AEGNN 附錄式 (8)） | 度數變了就要重算該節點的整個和；對稱歸一還會牽連另一端的所有邊 |
| 圖上 BatchNorm（AEGNN 每塊都有；官方異步 BN 凍結初始統計） | 與同步前向只是近似，誤差隨新事件變多而偏離「新事件 \(\ll\) 初始事件」的前提 |
| max 池化（DAGr 式 (15)） | 若 argmax 沒換，輸出可以不變，傳播可以停；若最大值節點被刪，必須重算該體素，不能只減一個數 |
| 全局 mean／max 讀出（S37 第 226–230 行） | 節點特徵可以不變，圖級向量仍然變。mean 有精確增量式；max 要保存冠軍 |
| 次採樣依賴**整包**事件數（S37 `_sample`：步長 \(=\mathrm{count}/\mathrm{max\_nodes}\)） | 多來一個原始事件，步長就變，節點集合不是前綴。與「逐事件插入、舊節點身份不變」不相容。AEGNN 的固定因子 \(K=10\)（每 \(K\) 個留一個）才是在線穩定的 |
| 頂點座標或跨邊標籤依賴 `prev`（方案 A；ECC 的 \(L(j,i)\)） | 一次姿態變化使所有相關邊同時失效，不是單事件插入 |

**非剛性與單目：** 增量等價說的是「同一張圖、同一組權重，少算一些節點，輸出仍相同」。它不增加可觀測量。手部自遮擋、厚度方向、根旋轉與橫向平移的近共線，不會因為改成異步更新而消失。Zubic 寫的「慢速大物體在事件圖裡難把訊息傳遠」對掌心這類低事件區同樣適用；那是圖連通性問題，不是排程問題。

**(c) 異步相對 50 ms 批處理多出來的是什麼**

多出來的是**何時付錢、何時可以輸出**，不是 50 ms 窗內事件集合以外的資訊。

- 若增量規則與整圖重算等價（AEGNN §4.2、AsyNet 補充材料、DAGr Methods），則包末的輸出與一次 `forward_packet` 相同。精度差必須為 0（容差內）。異步沒有新的資訊源。
- DAGr 的使用方式是：20 fps 影像之間用事件連續改檢測。那是把輸出率提高到高於幀率。本協議的遞推已經規定每 50 ms 一步。除非改協議、在包內提早輸出，否則異步不縮短決策等待。
- 牆鐘證據反對「FLOPs 下降＝延遲下降」：DAGr 稠密 30.8 ms 對異步 8.46 ms（50,000 事件），只快 3.7 倍；並寫明 MFLOPS／事件與運行時間不相關。AEGNN 摘要的 8 倍是相對「每個事件都重算整張標準 GNN」，不是相對「50 ms 一次批處理」。
- S37 主行（`docs/S37_ROUTED_READOUT_PREREG.md` §6）：延遲 7.72 ms，0.827 G，0.73 M。S36 同 FLOPs 為 7.04 ms，預註冊把多出來的約 0.68 ms 記成每包一次 FK + 路由。7.72 ms 已經小於 50 ms。計算不是這一步的主等待；包長才是。
- RVT 要壓的是「推理 > 40 ms」的檢測主幹。S37 不在那個區間。
- 包內提早輸出（Li 的穩定狀態、AEGNN 圖 4 在 2,500／10,000 事件時精度已高）會用**更少的事件**做決定，輸出與包末批處理**不相等**。那是另一個估計量，必須單獨測閉環，不能說成等價加速。

---

## 4. 可證偽假說與反對證據

**H1（關於現有算子，預期成立）。** 在 S37 的 `EdgeConv` 上：圖由 `_edges` 一次建成、三層共用、無 BatchNorm、無對稱度歸一、節點按時間下標排列且邊只指向更早下標。則對「在序列末尾插入一個新節點、舊節點的 `idx` 與 `dp` 不變」這件事，只更新新節點的 \(L\) 層特徵，與把增大後的圖整段重算，節點特徵最大絕對差應停在 float32 捨入（建議門檻 \(10^{-4}\)），且不隨插入次數系統性變大。

最強的反對證據不是手部數據，而是算子邊界：AEGNN 的 BN 實現自承是近似；DGCNN 的逐層 k-NN 與 GCN 的 \(\tilde D^{-1/2}\) 使「只更新新節點」為假；S37 的 `_sample` 若隨總事件數改步長，舊節點身份會變，H1 的前提不成立。H1 只在**凍結的節點集合、只在末尾追加**時有意義。

**H2（關於方案 A，預期被否定）。** 把 778 個頂點與當前事件放進同一張訊息傳遞圖，跨邊特徵依賴 `prev` 的投影。則姿態小擾動（例如根旋轉 \(10^{-3}\) rad）之後，需要重算的節點比例仍然很小，增量維護明顯省於全圖重算。

**已有的反對證據：**

1. 根旋轉經 FK 移動**所有**頂點。髒種子在第 0 層就是 778 個頂點，不是一個事件。DAGr 寫明：節點**位置**變化要重算該節點發出的全部訊息（式 (13) 下的討論），不是一條邊。
2. 本次在 `assets/mano_right.npz` 的面片上做無向 1-環 BFS（1538 面、778 頂點、無向邊 2315、度 3–8、平均 5.95、偏心率最大 28）。**單個**頂點的 3 跳球平均 38.0 個頂點，約 4.9%。所以「\(L=3\) 只碰局部」只在種子很小時成立。種子是全部頂點時，3 跳限制不再省任何頂點。
3. 只要跨邊進入 EdgeConv，每個帶跨邊的事件的聚合都依賴那些頂點。事件側的髒比例跟著變成 \(O(1)\)。
4. 倉庫已有的設計律（`semkine/event_gnn.py` 模組說明，以及 `docs/GNN_ARMS_ARCHIVE_20260828.md` §3、§5.2）：證據的**結構**保持與狀態無關；KEG 把 prev 送進圖結構後，閉環譜增益到 0.91 左右，而單步誤差可以更好。方案 A 的跨邊集合由 prev 的投影決定，結構再次依賴狀態。這和 KEG 的「可學習路由」不是同一個模組，但是同一類回路：下一步的圖取決於上一步的輸出。
5. 共用脈絡 §4 記錄的網格圖臂（778 頂點、1-環、沒有事件–頂點跨邊）已經在根旋轉上兩種子分裂。方案 A 加上跨邊，可能補上長程通路，這**沒有**被那次實驗直接否定；被否定的是「因此增量會便宜」。

H2 是我認為**最可能被否定**的一條。否定它不需要訓練，見 §6 的 E4。

---

## 5. 最小可遷移機制

### 5.1 (a) 與整圖重算等價的條件

設第 \(\ell+1\) 層

\[
h_i^{(\ell+1)}=U\Bigl(h_i^{(\ell)},\ \mathrm{Agg}_{j\in\mathcal{N}(i)} M(h_i^{(\ell)},h_j^{(\ell)},e_{ij})\Bigr).
\]

若 \(\mathcal{N}(i)\)、\(\{h_j^{(\ell)}\}_{j\in\mathcal{N}(i)}\)、\(\{e_{ij}\}\) 與 \(h_i^{(\ell)}\) 都與上次相同，則 \(h_i^{(\ell+1)}\) 相同。因此只需重算「第 \(\ell\) 層特徵變了的節點」的出鄰域（誰把這些節點當成鄰居）。從輸入層的髒集合 \(S_0\) 出發，\(L\) 層之後的髒集合是訊息方向上的 \(L\) 跳閉包。這就是 AEGNN §4.2 與 AsyNet 式 (5)(7) 的內容。GraphSAGE 的 \(K\) 跳給了感受野的上界，但上界只有在 \(S_0\) 很小時才省計算。

**因果邊把閉包收成單點。** S37（`_edges`，第 149–176 行）：節點按時間排序，節點 \(i\) 的候選是 \(i-1,\ldots,i-\mathrm{window}\)，\(\mathrm{window}=32\)，再取 \(k=8\)。邊從較早事件連到較晚的中心。新事件是目前最大的下標，沒有任何舊中心把它收進 \(\mathcal{N}\)。於是 \(S_0=\{\text{新節點}\}\) 的每一層仍然只有它自己：它讀的是舊節點**已經算好**的上一層特徵。三層共用同一 `idx`（第 219–222 行）是必要的；若第 2 層改在特徵空間連邊，舊節點的 \(\mathcal{N}\) 會變。

**滑窗近鄰的前提：**

- 窗內 top-k 等於全域時空近鄰，僅當真實鄰居沒有落在「早於 32 個事件」之外。`event_gnn.py` 自己寫了這句。手快速劃過、事件在時間上拉得很開時，這個截斷是近似圖，不是 AEGNN 的半徑圖。
- 刪除窗內最老事件時，下標在其後、且回看距離 \(\le 32\) 的節點可能改 top-k。髒集合沿時間向前，每層最多再蓋 `window` 個下標，\(L=3\) 時上界約 \(L\times 32=96\) 個節點，而不是 2048。這是上界，不是測過的平均。
- 等距次採樣（第 127–146 行）用整包長度當步長。它不是 AEGNN 的固定 \(K=10\)。**在現在的採樣器下，不能宣稱逐原始事件插入與批式 `forward` 逐位相同**，因為節點集合會重抽。

**S37 相對 AEGNN 更乾淨的一點：** 沒有 BatchNorm。AEGNN 的異步 BN 是凍結統計的近似。S37 的 mean 除以該節點自己的邊數；舊節點邊數不變則除數不變。

全局 mean\(\|\)max 仍要更新，但那是 \(O(C)\)，不是再跑三層。

### 5.2 (b) 什麼讓快取失效，以及方案 A 的複雜度

配置（`configs/semkine/s37_routed_s3407.yaml`）：`ENCODER_HIDDEN=128`，`ENCODER_LAYERS=3`，`ENCODER_K=8`，`ENCODER_MAX_NODES=2048`，`ENCODER_WINDOW=32`，`PREV_RENDER=false`，`ROUTED_READOUT=true`。

`forward_packet` 在路由分支（`model/model.py` 約 1299–1333 行）：先 `event_encoder(..., return_nodes=True)`，此時 `extra` 只有在 `prev_render` 時才有，S37 為空；然後 `_route_nodes` 在 `no_grad` 裡做 FK 與 `route_front_vertex_lbs`；再 `pool_joint_evidence`；手指頭讀該關節證據和 prev 的 3 維角；另外 `prev_mlp(prev)` 加到 51 維上。事件圖的 `idx`、`dp`、`h` **不讀 prev**。

因此同一包事件、只改 `prev`（閉環迭代 \(x\leftarrow f(\text{事件}_k,x)\)，或路由探針）時，\(h\) 可以整段復用。狀態依賴的是路由與頭。預註冊 §6：FLOPs 仍記 0.827 G，延遲從 S36 的 7.04 ms 增到 7.72 ms。與下面的 MAC 分解一致：大頭在 EdgeConv，路由是小頭，但 FK 仍反映在牆鐘上。

**EdgeConv 的乘加（與實現一致）：** 每條邊的線性層輸入維度 \(128+3\)，輸出 128。一層

\[
2048\times 8\times 131\times 128=274{,}726{,}912\ \text{MACs},\quad \times 3=824{,}180{,}736\approx 0.824\ \text{G}.
\]

嵌入 \(2048\times 7\times 128\approx 1.8\times 10^6\)。主行 0.827 G 與此同量級（計數細節以 `tools/make_s36_row.py` 的跡為準，這裡只解釋數量級）。

**讀出：** 每個活節點對 778 個投影頂點算平方距離，約 \(2048\times 778\times 2\approx 3.2\times 10^6\) MACs \(\approx 0.0032\) G，約為三層 EdgeConv 的 **1/260**。`route_front_vertex_lbs` 已在 top-8 裡做前表面與 16 px 門，並返回距離與頂點 id。上下文寫明：過門之後，節點到頂點的偏移被丟掉。

**方案 A：** 節點數 \(N+V=2048+778\)。事件邊約 \(2048\times 8\)。網格雙向 1-環約 4630 條有向邊（本次由面片統計）。若每個事件再連 \(m=8\) 條跨邊，跨邊約 \(1.6\times 10^4\) 條，總邊數約為現在事件圖的 2 倍量級。跨邊特徵依賴投影，投影依賴 `prev`。

一次非零姿態變化：

- 若頂點輸入含座標，或跨邊 \(e_{ij}\) 含偏移，則 \(S_0\) 包含全部 778 個頂點，以及每一條跨邊的事件端。
- \(L=3\) 的 1-環局部性（單種子約 5% 頂點）**節省為 0**，因為種子已經是全部頂點。
- 事件節點只要聚合了一條變過的跨邊，該層輸出就變；因果的事件–事件邊不能保護它們。
- 因此增量重算的 MAC 與全圖重算同階，約 \(\ge 0.82\) G **乘上**每次狀態改變，而不是 0.003 G。閉環若對同一包迭代 \(I\) 次，方案 A 約 \(I\) 次全圖，S37 是 1 次圖 + \(I\) 次讀出。

這不是實現不小心，是髒集合的定義。DAGr 的「每層一個節點」只適用於**插入一個時間上最新的事件**，且邊保持時間有向、舊節點位置不變。

**替代：只讓廉價讀出依賴狀態。**

保持 `EventGNN.forward` 不接收 prev。狀態只出現在 \(h\) 已經算完之後：

1. 繼續用現在的 `route_front_vertex_lbs`（已有距離與頂點 id）。把丟掉的像素偏移、或到輪廓的量，當成證據通道乘進池化，而不是變成 EdgeConv 的邊。幾何為零時這條通道為零，才符合「證據為零則幾何貢獻為零」。跨邊若在圖裡，幾何會改變訊息，即使事件特徵為零，殘差項仍可能非零，契約不自動成立。
2. 不要在三層訊息傳遞裡建跨邊。若堅持要一層學習的事件–頂點交互，把它放在**快取的 \(h\) 之後、且不寫回 \(h\)**。一層、\(m=8\) 的同類線性層仍約 0.275 G，比三層便宜，但比 0.003 G 的路由貴兩個數量級。真正便宜的是不再跑第二個 MLP，只用已算好的距離與 LBS。
3. 事件圖的鄰接繼續只由 \((x,y,t)\) 決定。這同時保住快取，並避開 KEG 那類「拓撲依賴 prev」的回路。方案 A 若把跨邊放進圖內，這兩條一起失去。

接入點：`semkine/event_gnn.py` 的 `forward`（保持）；`semkine/routed_readout.py` 的 `route_front_vertex_lbs` 返回值（偏移目前未進 `pool_joint_evidence`）；`model.py` 的 `forward_packet` 路由分支在 `pool_joint_evidence` 之前。不要改 `_edges`。

參數與延遲：讀出側多幾個標量通道，頭的輸入維度增加，參數是每關節頭的線性層寬度，數量級為數千而不是新的 0.82 G。延遲量級應接近現在 FK+路由的零點幾毫秒，而不是再加一輪三層圖。這是數量級，不是新的剖析。

### 5.3 不遷移的部分

- AEGNN 的雙向半徑 + 凍結 BN：不等價，而且 `cdist` 是他們自己在 S36 時代已否定的全對代價（見 `event_gnn.py` 模組說明）。
- EGSST 的無向半徑與連通子圖：批式檢測，無增量等價，也無 MANO。
- EvolveGCN 的權重遞迴、TGN 的節點記憶：狀態進了會被下次聚合讀取的地方。
- GNNAutoScale 的歷史嵌入：近似，會在姿態變化時留下過期特徵。

---

## 6. 最小判別實驗

全部可以在 CPU、隨機初始化、不訓練的前提下做。通過與否與 zgz 的 RA 無關；RA 不能代替等價測試。建議容差：隱藏維 128 上 \(\|h_{\mathrm{inc}}-h_{\mathrm{full}}\|_\infty\le 10^{-4}\)（float32）。累積漂移：每 200 次插入與「從頭全圖重算」比一次，看這個無窮範數是否隨插入序號上升。上升且超過 \(10^{-3}\) 即認為不是舍入。

**E1　因果插入，預期通過（支持 H1）。** 按 `_edges` 的規則建圖。全圖重算對照：對前綴 \(1..t\) 跑三層。增量：只算下標 \(t\) 的三層，舊節點用快取。比較所有節點。預期最大絕對差 \(\le 10^{-4}\)，到 \(t=2048\) 仍不爬升。

**E2　故意破壞前提，預期失敗。** (i) 只更新新節點，但邊改成 AEGNN 那種雙向半徑：差應遠大於 \(10^{-3}\)。改成每層更新 1 跳閉包後，在**沒有 BN** 時應回到 \(10^{-4}\)。(ii) 加上 AEGNN 那種凍結均值的 BN：差應非零，且隨插入變多而變大（他們的近似前提）。(iii) 每層用特徵 k-NN：只更新新節點應失敗。

**E3　次採樣，預期與批式圖不一致。** 用 `_sample` 的整包步長建 2048 節點；再模擬「事件數增加、步長改變」。舊快取對新節點集合沒有身份對應。這否定「現在的 S37 已經是 AEGNN 運行時」。若要做在線等價，採樣必須改成固定步長或只追加，且這個改動本身要先證明不改變訓練分佈；那是另一個實驗，不是本判別的通過條件。

**E4　方案 A 的髒集合，預期否定 H2。** 778 個頂點用 1-環，另加數百個事件節點；每個事件連到若幹頂點，邊特徵含頂點座標或投影偏移。三層與 S37 同型的 EdgeConv。把所有頂點座標做一個小旋轉。統計 \(|\Delta h|>10^{-5}\) 的節點比例，以及若只重算髒閉包所需的邊數除以全圖邊數。預期兩者都接近 1。對照：事件圖不含頂點，同一旋轉，事件節點的 \(\Delta h=0\)。

**E5　漂移。** E1 的精確路徑：無窮範數應停在舍入，不需要週期性全圖刷新。若有人改用過期的 k-NN 或凍結的全局 BN，同一曲線應隨 \(t\) 上升。否定「增量可以長期不對齊」：任一前綴超過 \(10^{-3}\)。不必跑 69 s 閉環；嵌入已經漂了，後面的 RA 沒有解釋價值。

**E6　延遲是否真的變小。** 同一 2048 節點：一次批式前向的牆鐘，對「從空圖插入到 2048」的牆鐘和。兩者都與 50 ms 比。預期（來自 DAGr 的 3.7 倍而非數量級，以及 S37 已是 7.72 ms）：插入總和不會明顯短於一次批式前向，且兩者都小於 50 ms。否定「異步降低本任務遞推延遲」：包末才輸出時，等待仍是 50 ms 量級。若實驗改成每 10 ms 輸出一次，那是新估計量，必須預先寫閉環門檻；它**不能**用來宣稱與 50 ms 批處理等價。

**和簡單修復、最近方法的區分：**

- 可見性修復、常數增益 \(\delta\)-trust、普通時序濾波、換一種事件表示：都不改變 E1–E4 的髒集合。它們可以同時做，但通過 E4（髒比例接近 1）只否定 H2，不評價那些修復。
- 最近的異步方法是 DAGr 的有向單節點更新。E1 通過且 E4 失敗，說明該機制適用於 S37 的事件圖插入，不適用於方案 A 的姿態更新。
- 不需要與 EGSST 比精度：EGSST 沒有這條等價規則。

本次沒有跑 E1–E6。網格的 3 跳統計只支持 E4 的「單種子很小、全頂點種子則不再小」這一半，不能代替帶跨邊的前向。

---

## 7. 六個創新問題

**1. 真正解決了什麼此前沒充分解決的矛盾？**  
AEGNN／Li／AsyNet／DAGr 解決的是「每個新事件都重算整網」與「激活其實只局部變」之間的矛盾，而且把同步訓練與異步部署的輸出對齊。本任務裡這個矛盾並不尖銳：一包一次前向，7.72 ms 對 50 ms。方案 A 想解決的是根旋轉與事件的幾何聯繫；增量文獻**沒有**證明把 prev 頂點放進事件圖就能觀測到根旋轉。那是另一個假說，而且和「少算」互相衝突。

**2. 新的資訊或約束從哪裡來？哪些只是歷史先驗再用一次？**  
等價更新不帶來新資訊。事件節點的資訊仍是單目亮度變化。方案 A 的跨邊把上一次 MANO 狀態再投影一次；778 個頂點是同一次 FK 的相關樣本，不是新測量。DAGr 的影像分支是另一個感測器；本任務沒有第二台相機，不能把 `prev` 的渲染說成 DAGr 的 RGB。

**3. 事件的異步、時間或稀疏性貢獻了什麼？**  
在這些論文裡，異步貢獻的是計算與「幀間可以更早輸出」。時間進入圖的方式是節點座標或有向約束（DAGr 的 \(t_i<t_j\)），稀疏性讓半徑圖的邊數遠小於像素網格。對已經按 50 ms 聚合成包、再抽 2048 個節點的追蹤器，這些性質已經用在批式 `EventGNN` 裡了。再改成逐事件排程，不增加稀疏性，只改變計算順序。包內時間若要用來分開「prev 誤差」和「包內運動」，那是時間戳回歸（方向 1 的 C1），不是圖的增量維護。

**4. 相對最接近的方法，變化在觀測、關聯、狀態更新，還是計算？**  
相對 AEGNN／DAGr，若只把 S37 改成逐事件更新，變化只在**計算**。相對 DAGr 的有向圖，S37 的 `_edges` 已經是時間因果的。方案 A 的變化在**關聯**：邊的存在或邊特徵改由 prev 決定。那不是 DAGr 的計算技巧，DAGr 的邊只由事件的時空座標決定（式 (4)）。狀態更新公式（殘差 EdgeConv、讀出頭）可以完全不變，快取卻已經失效。

**5. 為什麼更簡單的可見性修復、普通時序濾波或換表示不夠？**  
對「增量是否等價、方案 A 是否省計算」這兩問，那些修復既不構成證明也不構成反證：它們不動鄰域。若問題是根旋轉，增量文獻沒有說必須用跨邊圖；現有讀出已經能接觸投影後的頂點（`route_front_vertex_lbs`），缺的是把幾何乘進證據而不是改拓撲。換表示（體素、EGSST 的連通子圖、SSM 的張量）同樣不提供第二視角。

**6. 什麼結果會直接否定創新假說？**  
否定 H1：E1 的無窮範數 \(>10^{-3}\)，或 E5 隨插入上升。否定 H2：E4 的髒節點比例或 MAC 比 \(>0.5\)（預期會接近 1）。否定「異步改進本協議的延遲或精度」：E6 裡包末增量的牆鐘不短於一次批式前向，且與批式前向的輸出差在 \(10^{-4}\) 內（精度相同、延遲不降）。若有人把「包內提早輸出」當成創新，則否定條件是：預註冊的閉環指標不高於同一模型在包末輸出，或它可以被「更短的批窗 + 一次前向」複製——那就仍然是窗長，不是圖的增量機制。

---

## 8. 限制與誠實聲明

- **深讀 20 篇**，列在 §1.1。EvGNN 全文與摘要未讀，不算。TGL、NeutronStream、GET、Bi 的 TIP 版、DDCLS 2023 只核書目。
- EventNet 的關鍵公式在 PDF 裡是圖片，§2.5 只採用能讀到的散文，不抄那些圖。
- SplineCNN 式 (3) 的 ar5iv 片段沒有 \(1/|N(i)|\)；AEGNN 附錄有。兩處都已寫明，沒有合併成一個公式。
- GCN、GIN、MPNN、GraphSAGE、EvolveGCN、GNNAutoScale、ECC、SplineCNN、DeepGCN 的**會議收錄**，有的只在 arXiv HTML 裡讀到正文，proceedings 頁或 DOI 本次沒逐一打開，表中標了待核。內容公式來自讀到的 HTML。
- CCF 2022 與中科院 2025 分區表本次沒有下載，所有分區都是通行口徑加「待複核／待核」。
- AEGNN 官方 `conv.py` 我讀的是插入路徑；不能從這份檔案證明他們實現了滑窗刪除。論文正文寫了滑窗。
- DAGr 的 mAP、毫秒、MFLOPS 來自 Nature HTML 正文，不是從圖裡估的。圖本身（Extended Data）沒有逐格重測。
- S37 的 7.72 ms、0.827 G、0.73 M 來自 `docs/S37_ROUTED_READOUT_PREREG.md` §6，不是本次重跑。0.824 G 是按 `EdgeConv` 線性層尺寸手算的乘加，用來解釋數量級，與主行差在計數約定（偏置、池化、嵌入是否計入）。
- 網格 3 跳是 CPU 上對 `f` 的 BFS，不是論文數字，也沒有把事件跨邊算進去。
- E1–E6 沒有執行。因此 H1 仍是假說，雖然與代碼結構一致；H2 的否定目前是結構論證加網格 BFS，還缺 E4 的一次前向。
- 2026-08-29 的舊調查（已刪）把 AEGNN 概括成「只重算 K-hop」、把 EvGNN 寫成 16 µs、把 EGSST 與 eGSMV 並列到 2025。本次對 AEGNN 的修正是：雙向半徑才是 K-hop，因果有向可以收成單節點，BN 不是等價的。後兩條本次不採信。
- 沒有改任何程式、配置、測試或 `outputs/`。沒有用 GPU。沒有訓練。
