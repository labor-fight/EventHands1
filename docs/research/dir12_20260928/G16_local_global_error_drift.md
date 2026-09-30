# G16 局部—全局誤差及漂移

> 2026-09-28。研究子代理 G16。只讀倉庫與外部文獻；未改程式、配置或 `outputs/`；未使用 GPU。
> 權威診斷數字來自已讀的 `docs/S37_ROUTED_READOUT_PREREG.md` §8–§9，以及 `semkine/anchor.py`、`semkine/filter.py`、`semkine/eval_track.py`、`model/model.py` 的原文。文獻公式只寫實際讀到的原文；讀不到全文的條目不計入深讀、不補公式。

## 0. 檢索記錄

日期：2026-09-28。未宣稱遍歷全部文獻。

| 檢索式 / 端點 | 來源 | 結果 |
|---|---|---|
| `search_query=ti:HaWoR` 與 `search_query=all:ORB-SLAM3` | arXiv export API | HTTP 429 |
| arXiv HTML 搜尋（`/search/?query=HaWoR+hand+motion` 等） | arxiv.org | HTTP 429 |
| `works?search=HaWoR` | OpenAlex | HTTP 429 |
| `paper/search?query=HaWoR` | Semantic Scholar | HTTP 429 |
| `search/publ/api?q=HaWoR` | DBLP | HTTP 200 但為 Anubis 驗證頁，無書目 |
| `query.title=` 共 36 條（Sturm、ORB-SLAM2/3、VINS-Mono、Cadena、TLD、scheduled sampling、DAgger、Lohmiller、GLAMR、SLAHMR、WHAM、TRAM、HaWoR、GVHMR、Dyn-HaMR、Mahony、IEKF、Bailey、Huang FEJ、ByteTrack、OC-SORT、DeepSORT、Professor Forcing、ReFit、TRACE、DEVO、BoT-SORT、Hybrid-SORT、ScoreHMR、EventHands、E-3DPSM、KITTI 等） | Crossref | 部分 HTTP 429。成功時 `total-results` 為 10^5–10^7，是分詞寬匹配，**不當命中數**。只用第一筆標題是否對上該篇 |
| 已知 arXiv id 的 `/abs/` 與 `ar5iv.labs.arxiv.org/html/` | arXiv / ar5iv | 多數 HTTP 200。猜錯的 id（如 `2106.01923`、`0809.2949`、`1206.5174`）會下到無關論文，已丟棄、不引用 |
| CVF OpenAccess HTML（HaWoR、Dyn-HaMR、WHAM、OnlineHMR、ReFit、GLAMR） | openaccess.thecvf.com | HTTP 200，正文約 6 KB，只有贊助商頁，**不是全文**。全文改走 ar5iv |
| `q=HaWoR+hand+motion` | GitHub repository search | `total_count=2`；官方倉庫 `ThunderVVV/HaWoR`，README 指向 arXiv:2501.02973 |
| `https://rpg.ifi.uzh.ch/docs/IROS18_Zhang.pdf` | 蘇黎世大學 RPG | PDF 495 KB，`pdftotext` 後讀 §I–§IV |
| `http://www.cvlibs.net/publications/Geiger2012CVPR.pdf` | cvlibs.net | PDF 875 KB，讀到 §3.3 視覺里程計評測 |
| Sturm IROS 2012、TLD TPAMI 2012、Bailey ICRA 2006、Huang FEJ、Mahony TAC 2008、Lohmiller & Slotine Automatica 1998 的若干 PDF URL | 作者頁 / arXiv PDF | 404、401、406 或憑證失敗。**未讀全文** |

篩選：保留與「絕對估計 vs 相對積分、漂移與重錨、RPE/ATE、檢測+跟蹤、人體/手的世界軌跡、遞推訓練穩定性、濾波一致性、事件手/人體」直接相關者。2023–2026 優先，早期基礎工作只在深讀或目錄中單獨標出。猜中無關 arXiv 的不入目錄。

覆蓋缺口（如實）：

- 未打開 CCF 官網與中科院分區表。下文「分區口徑」對 CVPR / ICCV / ECCV / NeurIPS 標通行口徑 CCF-A 並註「官網待核」；其餘會議期刊與全部中科院分區標**待核**。
- Sturm、TLD、Mahony、Bailey、Huang、Lohmiller 1998 原文未取得，不把它們算進 20 篇深讀，也不寫它們的公式。
- arXiv 搜尋 429，候選不是從一次完整檢索清單篩出，而是由題名（Crossref）、已知 id、GitHub README 與引用鏈彙總。
- Dyn-HaMR、ReFit、DEVO、OnlineHMR、EventHands、Hybrid-SORT 只有書目或空 HTML，不深讀。
- Sola 誤差狀態 Kalman 報告（arXiv:1711.02508）已下載，但未讀到可引用的 NIS/NEES 定義，不計入深讀。Forster 預積分（arXiv:1512.02363，abs 頁 DOI `10.1109/TRO.2016.2597321`）只核到書目，不計入深讀。

## 1. 候選目錄

分區欄：`CCF-A*` = 通行口徑且官網待核。`待核` = 本次未核對目錄。發表狀態只寫本次核到的證據。

| 標題 | 作者 / 機構（讀到的） | 會議 / 期刊 | 年 | 發表狀態 | 分區 | 官方代碼（本次看到的） | 標籤 |
|---|---|---|---|---|---|---|---|
| A Tutorial on Quantitative Trajectory Evaluation for Visual(-Inertial) Odometry | Zhang, Scaramuzza；蘇黎世大學 RPG | IROS（檔名 `IROS18_Zhang.pdf`；正文摘錄未再印會議名） | 2018 | 正式 PDF | 待核 | `github.com/uzh-rpg/rpg_trajectory_evaluation`（PDF 內寫明） | ATE/RPE、對齊規範 |
| A benchmark for the evaluation of RGB-D SLAM systems | Sturm, Engelhard, Endres, Burgard, Cremers | IROS | 2012 | Crossref DOI `10.1109/iros.2012.6385773`；**全文未取得** | 待核 | — | ATE/RPE 源頭，未深讀 |
| Are We Ready for Autonomous Driving? The KITTI Vision Benchmark Suite | Geiger 等（PDF） | CVPR（檔名） | 2012 | 正式 PDF | CCF-A* | cvlibs.net | 分段相對誤差 |
| ORB-SLAM3 | Campos 等 | IEEE TRO | 2021 | 全文頁首 DOI `10.1109/TRO.2021.3075644`，©2021 | 待核 | 論文稱開源；本次未克隆倉庫 | 跟丟、重定位、閉環 |
| ORB-SLAM2 | Mur-Artal, Tardós | IEEE TRO | 2017 | abs DOI `10.1109/TRO.2017.2705103` | 待核 | 論文稱開源；本次未核對函式 | 定位模式、`t_abs`/`t_rel` |
| VINS-Mono | Qin, Li, Shen | IEEE TRO | 2018 | abs DOI `10.1109/TRO.2018.2853729` | 待核 | 本次未開倉庫 | 4 DoF 漂移、回環因子 |
| Direct Sparse Odometry | Engel, Koltun, Cremers | 預印本 arXiv:1607.02565；期刊版待核 | 2016 提交 | **預印本**（abs 無 DOI） | 待核 | 本次未開倉庫 | 滑窗里程計、無閉環 |
| Past, Present, and Future of SLAM | Cadena, Carlone 等 | IEEE TRO | 2016 | abs DOI `10.1109/TRO.2016.2624754` | 待核 | — | MAP / 因子圖 |
| On-Manifold Preintegration | Forster 等 | IEEE TRO | 2016 | abs DOI `10.1109/TRO.2016.2597321`；**未深讀** | 待核 | — | IMU 偏置狀態 |
| The invariant extended Kalman filter as a stable observer | Barrau, Bonnabel | 預印本 arXiv:1410.1465；TAC 版待核 | 2014 提交 | **預印本** | 待核 | — | 軌跡無關的誤差方程 |
| Contraction Theory … A Tutorial Overview | Tsukamoto, Chung, Slotine | Annual Reviews in Control（頁首） | arXiv 2021-10-01 | 頁首標期刊；卷期待核 | 待核 | — | 增量指數穩定、擾動球 |
| On Contraction Analysis for Non-linear Systems | Lohmiller, Slotine | Automatica | 1998 | Crossref DOI `10.1016/s0005-1098(98)00019-3`；**全文未取得** | 待核 | — | 收縮原論文，未深讀 |
| Scheduled Sampling for Sequence Prediction with RNNs | Bengio, Vinyals, Jaitly, Shazeer | 預印本 arXiv:1506.03099；NIPS 2015 為通行出處，abs 無 DOI | 2015 | **預印本**；會議版待核 | CCF-A* 若屬 NeurIPS，待核 | — | teacher forcing 與閉環脫節 |
| A Reduction of Imitation Learning … to No-Regret Online Learning | Ross, Gordon, Bagnell；CMU | 依 scheduled sampling 引用 [10] 為 AISTATS 2011；讀到的是 arXiv:1011.0686 | 2011 | 預印本 HTML；會議版待核 | 待核 | — | \(T^2\epsilon\) 誤差複利 |
| Professor Forcing | Lamb 等 | 預印本 arXiv:1610.09038 | 2016 | **預印本**；NeurIPS 版待核 | 待核 | — | scheduled sampling 有偏 |
| GLAMR | Yuan 等 | 預印本 arXiv:2112.01524；CVPR 2022 通行出處待核 | 2021 提交 | **預印本**（abs 無 DOI） | CCF-A* 若屬 CVPR，待核 | 本次未開倉庫 | 自我中心增量再累積 |
| Decoupling Human and Camera Motion（SLAHMR） | Ye, Pavlakos, Malik, Kanazawa | CVPR | 2023 | Crossref DOI `10.1109/cvpr52729.2023.02033` | CCF-A* | 本次未開倉庫 | 相機系姿態 × SLAM，再平滑 |
| WHAM | Shin, Kim, Halilaj, Black（abs） | 預印本 arXiv:2312.07531（online date 2024-04-18）；全文開頭未見 CVPR 字樣 | 2023/2024 | **不把預印本寫成已發表** | 待核 | 摘要寫 `wham.is.tue.mpg.de` | 根速度積分 + 接觸修正 |
| TRAM | Wang 等（ar5iv） | 預印本 arXiv:2403.17346 | 2024 | **預印本**（abs 無 DOI） | 待核 | 本次未開倉庫 | 掩手 SLAM + 尺度 + 相機系人體 |
| HaWoR | Zhang, Deng, Ma, Potamias；上海交大 / ICL | CVPR | 2025 | Crossref DOI `10.1109/cvpr52734.2025.00175`；arXiv:2501.02973 為預印本 | CCF-A* | `github.com/ThunderVVV/HaWoR` | 手：相機系絕對 + 掩手 SLAM |
| GVHMR | Shen 等（ar5iv 頁首） | SIGGRAPH Asia 2024 | 2024 | 頁首會議名 + DOI `10.1145/3680528.3687565` | 待核 | 本次未開倉庫 | 重力-視線座標、速度 rollout |
| Dyn-HaMR | Yu, Zafeiriou, Birdal | CVPR | 2025 | Crossref DOI `10.1109/cvpr52734.2025.02581`；CVF HTML 為空頁 | CCF-A* | — | 目錄only |
| E-3DPSM | ar5iv；abs 2026-04-09 | 倉庫脈絡稱 CVPR 2026 | 2026 | **預印本** arXiv:2604.08543；abs 無 DOI，全文開頭無錄用句 | 若屬 CVPR 則 CCF-A*，待核 | 本次未開倉庫 | direct+delta、常數 Q,R |
| ByteTrack | Zhang, Sun, Jiang, Yu | ECCV（LNCS） | 2022 | Crossref DOI `10.1007/978-3-031-20047-2_1` | CCF-A* | 本次未開倉庫 | 檢測框 + Kalman 關聯 |
| OC-SORT | Cao 等（ar5iv） | 預印本 arXiv:2203.14360；ECCV 2022 通行出處，Crossref 未取到該 DOI | 2022 | **預印本**；會議版待核 | 待核 | 本次未開倉庫 | 估計中心的 \(T^2\) 放大、ORU |
| Tracking-Learning-Detection | Kalal, Mikolajczyk, Matas | TPAMI | 2012 | **全文未取得**（PDF 404/SSL） | 待核 | — | 目錄only |
| EvHandPose | ar5iv | 預印本 arXiv:2303.02862；期刊版待核 | 2023 提交 | **預印本** | 待核 | 本次未開倉庫 | 事件運動歧義 |
| Ev2Hands | ar5iv | 預印本 arXiv:2312.14157；3DV 版待核 | 2023 提交 | **預印本** | 待核 | 本次未開倉庫 | 事件點雲的第三維是時間 |
| EventHands | Rudnev 等 | ICCV | 2021 | Crossref DOI `10.1109/iccv48922.2021.01216`；未深讀 | CCF-A* | — | 目錄only |
| ReFit | Wang, Daniilidis | ICCV | 2023 | Crossref DOI `10.1109/iccv51070.2023.01346`；CVF HTML 空頁 | CCF-A* | — | 目錄only |
| DEVO | Klenk, Motzet, Koestler, Cremers | 3DV | 2024 | Crossref DOI `10.1109/3dv62453.2024.00036`；未深讀 | 待核 | — | 目錄only |
| Hybrid-SORT | Yang, Han, Yan, Zhang | AAAI | 2024 | Crossref DOI `10.1609/aaai.v38i7.28471`；未深讀 | CCF-A* 通行、官網待核 | — | 目錄only |
| Quaternion kinematics for the ESKF | Solà | arXiv:1711.02508 | 2017 | 已下載未深讀 | — | — | 不計深讀 |
| Consistency of the EKF-SLAM / FEJ | Bailey 等；Huang, Mourikis, Roumeliotis | ICRA 2006 / ISER | 2006 / 2009 | PDF 失敗。Huang 有 Crossref DOI `10.1007/978-3-642-00196-3_43` | 待核 | — | 一致性文獻缺口 |
| Nonlinear Complementary Filters on SO(3) | Mahony, Hamel, Pflimlin | IEEE TAC 通行出處 | 2008 | **全文未取得** | 待核 | — | 目錄only |

深讀 24 篇見 §2。未列入上表的猜錯 arXiv 已丟棄。

## 2. 深讀（24 篇）

### 2.1 Zhang & Scaramuzza，IROS 2018 教程

- 原問題：估計軌跡與真值不在同一參考系，且軌跡是高維時間序列，如何壓成可比較的誤差。
- 關鍵公式（PDF §IV，原文記號）：單狀態誤差 \(\Delta x_i=\{\Delta R_i,\Delta p_i,\Delta v_i\}\)，且 \(R_i=\Delta R_i\hat R_i^0\)，\(p_i=\Delta R_i\hat p_i^0+\Delta p_i\)。整軌 ATE 為對齊後的 RMSE，
  \[
  \mathrm{ATE}_{\mathrm{pos}}=\Bigl(\frac1N\sum_i\|\Delta p_i\|^2\Bigr)^{1/2}.
  \]
  相對誤差先按準則選子軌跡對 \(d_k=\{\hat x_s,\hat x_e\}\)，**只用起點狀態**做對齊，再量終點：
  \[
  \delta p_k=\|p_e-\delta R_k\hat p_e^0\|_2.
  \]
  對齊類型由不可觀規範決定：單目相似變換，雙目剛體，視覺慣性是繞重力的 4 自由度（§III，式 (12)）。
- 觀測與假設：ATE 對「誤差發生的時刻」敏感——同樣的旋轉誤差放在軌跡開頭比放在結尾更大。相對誤差可換子軌跡長度，短的看局部一致性，長的看長期精度。兩者高度相關（他們把這句歸於 Sturm 等），仍建議同時報告。用全部狀態做一次最小二乘對齊會把 ATE 壓低；只用起點對齊才看得到誤差隨時間長大。
- 代碼：PDF 給出 `uzh-rpg/rpg_trajectory_evaluation`。本次未讀該倉庫原始碼。
- 對本任務：`zgz_local` 更像「子軌跡長度約 0.7 s、起點對齊（而且起點是 GT）」的相對誤差；`zgz_global` 的 69 s 更像長時絕對誤差，但本倉庫的 RA **不是**他們的 ATE（見 §3）。

### 2.2 Geiger 等，KITTI，CVPR 2012 PDF §3.3

- 原問題：車載立體視覺里程計怎麼報誤差。
- 讀到的說法：比較的五種方法都不用閉環。Fig. 5 把平移與旋轉誤差畫成子序列長度與車速的函數。VISO2-S 平均平移誤差 2.2%、旋轉 0.016 deg/m。增量或滑窗方法會慢慢漂；低速時相對影響最大，因為同樣的漂移率除以很小的位移。他們寫：檢測閉環、更完整的束調整、用訓練資料擬合參數，還能再提高。
- 假設：評測時不把閉環算進里程計本身，才能看見漂移率。
- 代碼：本次未讀 KITTI devkit。
- 對本任務：69 s 上若誤差不隨 5 s 分桶增長，就不是 KITTI 意義上的里程計漂移。低速（手幾乎不動的 local）會把每步亂跳放大成相對誤差，與「原地不動已經只有 4.41 mm」一致。

### 2.3 ORB-SLAM3，TRO 2021，arXiv:2007.11898

- 原問題：視覺 / 視覺慣性在跟丟、回環、多地圖下如何保持實時與全局一致。
- 讀到的機制（式子在 HTML 轉文本時有殘缺，只複述讀全的句子）：純視覺跟丟後，用與相機模型解耦的 MLPnP 在 Atlas 裡重定位；視覺慣性在跟蹤點少於 15 時進入 short-term lost，用 IMU 傳播並在大窗口搜點，5 秒仍失敗則新建地圖。閉環不是每幀混合：DBoW2 候選經過幾何驗證後，在焊接窗口做 BA，再用本質圖把修正傳到全圖，最後在獨立執行緒做全局 BA。評測是 RMS ATE，單目用 \(\mathrm{Sim}(3)\) 對齊，其餘用 \(\mathrm{SE}(3)\)。
- 假設：短程關聯靠把地圖點投影到當前幀附近幾個像素；長程關聯不能從當前位姿的初值出發，必須用位置識別。誤閉環會毀圖，所以精度優先、召回其次。
- 代碼：論文稱開源庫。本次未打開 `Tracking.cc` 核對函式名。
- 對本任務：重定位是「跟丟才做的另一條絕對求解」，不是每個 50 ms 用常數 \(\alpha\) 把絕對頭拌進遞推。S37 沒有跟丟狀態，也沒有第二條求解。

### 2.4 ORB-SLAM2，TRO 2017，arXiv:1610.06475

- 原問題：立體 / RGB-D 下尺度可觀，閉環用剛體而不是相似變換。
- 讀到的：定位模式關掉建圖與閉環，同時使用「上一幀立體點的視覺里程計匹配」（會積漂移）和「地圖點匹配」（對已有地圖無漂）。KITTI 同時報 Sturm 的絕對平移 RMSE \(t_{abs}\) 與 Geiger 的相對平移 \(t_{rel}\)、相對旋轉 \(r_{rel}\)。遠點主要約束朝向，近點才約束平移。
- 對本任務：一條系統裡可以並存「會漂的相對匹配」和「無漂的地圖匹配」，但後者只在能重新看到地圖時生效。事件手沒有靜態地圖；把靜態 SLAM 的閉環直接搬過來，會把非剛性的手當成要被掩掉的動態物體（見 TRAM / HaWoR）。

### 2.5 VINS-Mono，TRO 2018，arXiv:1708.03852

- 原問題：單目視覺慣性的滑窗會在邊緣化後累積漂移。
- 讀到的（§VI-G、§VII、§VIII）：漂移發生在全局位置 \((x,y,z)\) 與繞重力的 yaw，一共 4 個自由度；roll / pitch 可觀，所以位姿圖只優化 4 DoF。失敗檢測是三條內部訊號：最新幀跟蹤特徵數過低、相鄰兩次輸出的位置或旋轉不連續、偏置或外參突變。失敗後回到初始化，並新開一段位姿圖。重定位把回環幀位姿當常數，把檢索到的特徵殘差加進滑窗代價 (26)，維度不增加。IMU 可以在視覺更新之間前向傳播，得到更高頻的狀態，視覺殘差負責把漂移拉回來。
- 假設：回環因子是稀疏的絕對約束，不是每步固定增益。
- 對本任務：失敗檢測對應 `anchor.py` 的「何時」，而不是 `prev_mlp` 的「每步拉一點」。S37 沒有特徵數崩潰或位姿跳變這種跟丟；它的誤差從一開始就在，而且保持平的。

### 2.6 DSO，arXiv:1607.02565（預印本）

- 原問題：直接法稀疏光度誤差的滑窗視覺里程計。
- 讀到的：在最近幀窗口上連續優化光度誤差；舊位姿與離開視野的點被邊緣化。作者寫幾何先驗會在 Hessian 裡加入幾何-幾何相關，實時聯合優化做不到統計一致；他們還發現先驗會引入偏差，從而降低而非提高長期、大尺度精度。
- 對本任務：純滑窗是跟蹤器。把幾何當成與殘差無關的加性項，正是本倉庫契約要禁止的（證據為零時幾何貢獻必須為零）。DSO 這段是反對「加性幾何先驗」的旁證，不是事件手的方法。

### 2.7 Cadena 等，TRO 2016 綜述，arXiv:1606.05830

- 原問題：現代 SLAM 的後端在估什麼。
- 關鍵公式（§II）：\(z_k=h_k(\mathcal X_k)+\epsilon_k\)，MAP
  \[
  \mathcal X^\star=\arg\min_{\mathcal X}\sum_{k=0}^{m}\|h_k(\mathcal X_k)-z_k\|^2_{\Omega_k}.
  \]
  高斯、獨立測量時，運動模型與觀測模型都只是因子。他們寫：線性高斯下 Kalman 濾波與 MAP 給出同一估計，一般情況則否。
- 對本任務：常數增益融合只有在線性高斯、噪聲統計平穩時才與 MAP 重合。事件數在 `zgz_local` 與 `zgz_global` 之間差一個數量級，50 ms 真實轉動只有 1.6°–3.0°，課程噪聲卻是 0.3 rad，統計不平穩，固定 \(Q,R\) 或固定 \(\alpha\) 不是這個問題的 MAP。

### 2.8 Barrau & Bonnabel，arXiv:1410.1465（預印本）

- 原問題：Lie 群上的 IEKF 何時是沿任意軌跡局部收斂的觀測器。
- 讀到的（§2.3）：對滿足式 (12) 的系統，兩條任意遠的軌跡之間的左/右不變誤差 \(\eta_t^i\) 等於 \(\exp(\xi_t^i)\)，而 \(\xi\) 服從線性方程 \(\dot\xi_t^i=A_t^i\xi_t^i\)，對任意大的初始誤差都成立。誤差方程不依賴真實軌跡。
- 對本任務：若將來做流形濾波，誤差必須放在切空間（`filter.py`、`anchor.py` 已這樣寫）。S37 根更新是軸角座標相加。prereg §9.1 已排除「左雅可比用錯」是主因（朝向軸只散開 3°–6°）。IEKF 解釋不了 6°–8° 的正交更新。

### 2.9 Tsukamoto, Chung, Slotine，收縮理論教程

- 原問題：非自治非線性系統的軌跡彼此是否指數靠攏。
- 讀到的（Theorem 2.1，§2）：存在一致正定的 \(M(x,t)=\Theta^\top\Theta\)，使微分 Lyapunov 函數 \(V=\delta x^\top M\delta x\) 沿微分動力學下降，則所有解軌跡以速率 \(\alpha\) 指數收斂到同一條軌跡（增量指數穩定）。反之亦然。擾動系統（Theorem 2.4 的推導段）滿足
  \[
  \dot V_\ell(t)\le -\alpha V_\ell(t)+\sup\|\Theta(q,t)\,d(\xi_1,t)\|,
  \]
  比較引理給出有界球；球的半徑由擾動上界與 \(\alpha\) 決定。HTML 未把式 (2.33) 的係數展開，此處不補係數。他們還給出反例：系統可以對平衡點不穩定，卻仍然收縮到唯一的（發散的）特解。收縮不要求存在穩定平衡點。
- 對本任務：同包迭代從 GT 或遠點收到同一點，是收縮映射的固定點，不是積分器。69 s 誤差不長大，符合「持久擾動下停在球內」，不符合隨機遊走。一次狀態替換若沒有改寫映射，會被收縮忘掉。

### 2.10 Bengio 等，Scheduled Sampling，arXiv:1506.03099

- 原問題：訓練時條件於真值上一詞，推理時條件於自己的輸出，早期錯誤被放大。
- 關鍵公式：訓練目標 \(\sum_t\log P(y_t\mid y_1^{t-1},X)\)，狀態 \(h_t=f(h_{t-1},y_{t-1})\)。Scheduled sampling 以概率 \(\epsilon_i\) 喂真值、以 \(1-\epsilon_i\) 喂模型自身；\(\epsilon_i\) 從 1 衰到接近 0（線性、指數或逆 sigmoid）。梯度沒有穿過採樣決策。
- 觀測：Always Sampling（一直喂自己）在圖像描述上 BLEU-4 從 28.8 掉到 11.2。語音實驗裡，teacher forcing 的下一步幀錯誤 15.0，解碼錯誤 46.0；Always Sampling 下一步 34.6、解碼 35.8。下一步指標與閉環解碼可以反向。
- 對本任務：這套理論解釋的是「閉環比單步差，因為狀態走到訓練沒見過的地方」。S37 的 global 閉環停在同包信念上，69 s 不漂，不符合錯誤複利。local 手指在段內 0.5–2 s 從約 10–11 mm 升到 18–24 mm，才比較像這類問題，但之後又落到 14–18 mm，也不是單調複利。

### 2.11 Ross, Gordon, Bagnell，DAgger，arXiv:1011.0686

- 原問題：模仿學習裡，專家分佈上錯誤率 \(\epsilon\) 的策略，在自己誘導的狀態分佈上可以付出 \(T^2\epsilon\)。
- 讀到的：\(\mathbb E_{s\sim d_{\pi^\*}}[\ell(s,\pi)]=\epsilon\) 時 \(J(\pi)\le J(\pi^\*)+T^2\epsilon\)。若在**自己的**狀態分佈上 \(\epsilon\)，且從錯誤恢復的額外代價 \(\le u\)，則 \(J(\pi)\le J(\pi^\*)+uT\epsilon\)。DAgger 在當前策略下採軌跡、聚合數據、再擬合，使訓練分佈靠近執行分佈。
- 對本任務：課程噪聲是在 GT prev 上加的獨立噪聲，不是閉環狀態分佈。若部署映射真的在積分，DAgger 式的閉環採樣能把 \(T^2\) 降到 \(T\)。S37 global 的失敗不是這條界：誤差有上界且等於信念。把 DAgger 當成 global 的下一臂，是在修一個沒有出現的故障。

### 2.12 Lamb 等，Professor Forcing，arXiv:1610.09038

- 原問題：teacher forcing 與自由運行的隱狀態佔用不同區域，小誤差在條件上下文裡複利。
- 讀到的：他們引用 Huszár，指出 scheduled sampling 即使容量與樣本趨於無窮也可能收斂不到正確模型，是有偏估計。Professor Forcing 用判別器迫使「被真值鉗住」與「自己生成」兩種行為序列不可區分。短序列上他們報告過負結果。
- 對本任務：就算要做閉環訓練，scheduled sampling 本身不是無偏修復。S37 的 TF 單步已經比原地不動差 2.7–3.4 倍，問題首先是單步更新方向不對，而不是只在閉環才暴露。

### 2.13 GLAMR，arXiv:2112.01524（預印本）

- 原問題：動態相機、有遮擋的影片裡，人的世界軌跡。
- 讀到的（§3.2）：軌跡網路先出自我中心增量 \(\psi_t=(\delta x_t,\delta y_t,z_t,\delta\phi_t,\eta_t)\)，其中 \((\delta x_t,\delta y_t)\) 是朝向座標下的相鄰平移差，\(\delta\phi_t\) 是朝向角差；再 `EgoToGlobal` 累加。起點 \((\delta x_0,\delta y_0,\delta\phi_0)\) 在推理時任意，因為全局平移與朝向有規範自由度。之後用全局優化把軌跡對到影片的 2D 證據，並解出起點。
- 假設：長軌跡不該直接回歸一個很大的全局偏移；相對增量更好學，但必須再被圖像證據錨住，否則漂。
- 對本任務：他們把「身體姿態」和「根軌跡」拆開。增量只放在根軌跡，而且後面有非常數的優化錨。S37 把 51 維一起做成沒有新息的增量。

### 2.14 SLAHMR，CVPR 2023，arXiv:2302.12827

- 原問題：把相機運動和人的運動從野外影片裡拆開。
- 讀到的（§3.1–3.2）：相機系姿態 \(\hat{\mathcal P}\) 乘 SLAM 的 \(\hat R_t,\hat T_t\) 得到世界系初值，尺度 \(\alpha\) 初值為 1。數據項是關節重投影，Geman-McClure 魯棒核。第一階段只優化世界系根朝向與平移。接著用平滑項 \(E_{\mathrm{smooth}}=\sum\|J_t-J_{t+1}\|^2\) 把人和相機的貢獻拆開，否則重投影欠定。
- 假設：每幀相機系估計是絕對觀測；世界軌跡是另一條狀態，靠重投影和運動先驗約束，不是把上一幀姿勢加一個網路增量。
- 對本任務：單目重投影欠定（尺度、以及相機運動與人的平移可互換）。事件包更欠定。不能把「多個深度假設」當成多次獨立測量。

### 2.15 WHAM，arXiv:2312.07531（預印本）

- 原問題：世界座標下的人體，不要假設平地，且要能在線。
- 讀到的（§3.2）：相機系 SMPL 由運動解碼器從 2D 關鍵點與圖像特徵回歸。全局軌跡解碼器另出根朝向 \(\Gamma_0^{(t)}\) 與根速度 \(v_0^{(t)}\)，輸入裡拼上相機角速度 \(\omega^{(t)}\)（SLAM 或陀螺儀）。接觸概率高的腳速度被減掉以壓滑步，再經精煉網路。平移是 rollout
  \[
  \tau^{(t)}=\sum_{i=0}^{t-1}\Gamma^{(i)}v^{(i)}.
  \]
- 評測（§4）：相機系用 MPJPE / PA-MPJPE。世界系沿用把序列切成 100 幀、用前兩幀或整段對齊的 W-MPJPE\(_{100}\) / WA-MPJPE\(_{100}\)。他們明文寫：這些指標給不出真實的長時圖像，因為不測長序列漂移；所以另報整段剛體對齊後、按位移歸一化的 RTE，以及世界系抖動與腳滑。
- 對本任務：WHAM 自己認為「短段、起點對齊」會美化結果。這正是 `zgz_local`（每段從 GT 起、平均 0.7 s）相對 `zgz_global`（69 s）的結構差。他們的根是速度積分，身體是另一條回歸；S37 沒有這條分裂。

### 2.16 TRAM，arXiv:2403.17346（預印本）

- 原問題：野外影片裡人的 \(SE(3)\) 根軌跡與相機系身體運動。
- 讀到的：\(\{H_t\}=\{G_t\circ T_t\}\)，\(G\) 是相機軌跡，\(T\) 是人相對相機。DROID-SLAM 的稠密 BA 是
  \[
  E(G,d)=\sum_{(i,j)}\|p_{ij}-\Pi(G_{ij}\circ\Pi^{-1}(p_i,d_i))\|^2_{\Sigma_{ij}},\quad \Sigma_{ij}=\mathrm{diag}\,w_{ij}.
  \]
  人佔畫面很大時，置信度不夠，必須把人從圖像和置信度裡掩掉，否則 BA 用到違反靜態假設的像素。尺度 \(\alpha\) 用 ZoeDepth 的米制深度對 SLAM 深度做 German-McClure 魯棒對齊，逐關鍵幀求解再取中位數；他們認為這比從人體運動模型反推尺度更可靠。VIMO 在相機系做時序平滑，不負責世界積分。
- 剛體假設：靜態場景的 SLAM 不能直接用於非剛性手。HaWoR 把同一件事用在手上（下一條）。雙目不會憑空出現。

### 2.17 HaWoR，CVPR 2025，arXiv:2501.02973

- 原問題：自我中心影片裡，手在世界座標的運動。
- 讀到的（§3）：先做相機系手運動 \(\mathcal M\)（WiLoR 特徵 + 時間注意力 + 姿態注意力），損失是 3D/2D 關節與 MANO，**不是**對 prev 的增量。相機軌跡用掩掉手的 DROID-SLAM；式 (2) \(\hat I_t=(1-M_t)I_t\)，\(\hat w_t=(1-M_t)w_t\)。尺度用 Metric3D，在手以外、深度介於 \([D_{\min},D_{\max}]\) 的點上做 German-McClure 對齊，式 (4)。出畫的手用填充網路，先變到規範座標（段首旋轉平移為零）再補。世界系指標是 W-MPJPE、WA-MPJPE、RTE、加速度；相機用 ATE 與使用估計尺度的 ATE-S。
- 代碼：`ThunderVVV/HaWoR` README 指向同一 arXiv。本次未讀訓練迴圈。
- 對本任務：2024–2025 的手部世界軌跡仍是「相機系絕對手姿」加「把手当動態物體掩掉之後的場景 SLAM」。沒有「用 50 ms 事件增量積分 51 維 MANO」這條主幹。S37 既沒有相機系絕對手姿（根頭不讀 \(r_{\mathrm{prev}}\)，完美路由下 \(R^2=0\)），也沒有一條獨立的場景錨。

### 2.18 GVHMR，SIGGRAPH Asia 2024，arXiv:2409.06662

- 原問題：世界系人體朝向繞重力有規範自由度。
- 讀到的（§3.1）：每幀在重力與相機視線定義的 GV 座標裡預測朝向 \(\Gamma_{GV}\)。相鄰 GV 之間的旋轉只繞重力。靜態相機時
  \[
  \tau_w^t=\sum_{i=0}^{t-1}\Gamma_w^i v_{\mathrm{root}}^i\quad(t>0).
  \]
  運動相機時用相對旋轉的連乘把各幀 \(\Gamma_{GV}^t\) 送回第 0 幀。他們寫這避免了重力方向的累積誤差，而且不需要自回歸初始化。相對相機旋轉用 GT 陀螺儀或 DPVO，表 1 上兩者接近。
- 對本任務：又一次把「每幀絕對朝向（在一個可觀的座標裡）」和「速度積分出的平移」拆開。繞不可觀軸的相對旋轉才被積分。S37 把可觀與不可觀的根旋轉放進同一個加性 \(\Delta r\)。

### 2.19 E-3DPSM，arXiv:2604.08543（預印本；倉庫稱 CVPR 2026，本次未核到 DOI）

- 原問題：自我中心事件相機上，人體 3D 關節如何又跟得上運動又不漂。
- 關鍵公式（§4.2，原文）：直接姿態 \(\mathbf P_t^{\mathrm D}=\mathrm{MLP}_{\mathrm{Direct}}(\mathbf F_t)\)。增量 \(\mathbf P_t^\Delta=\mathrm{MLP}_\Delta([\mathbf F_t;\mathbf E_{t-1}])\)，\(\mathbf E_{t-1}\) 來自上一姿勢。樸素融合
  \[
  \mathbf P_t=\mathbf P_{t-1}^{\mathrm D}+\mathbf P_t^\Delta
  \]
  會累積誤差（附錄 B、Fig. 6）。他們的融合是 Kalman 形式：\(\mathbf A,\mathbf B,\mathbf H\) 固定為單位陣，
  \[
  \mathbf K_t=\boldsymbol\Sigma_{t|t-1}\mathbf H^\top(\mathbf H\boldsymbol\Sigma_{t|t-1}\mathbf H^\top+\mathbf R)^{-1},
  \]
  \[
  \mathbf P_t=\mathbf X_t+\mathbf K_t(\mathbf P_t^{\mathrm D}-\mathbf H\mathbf X_t),
  \]
  協方差用 Joseph 形式。\(\mathbf Q,\mathbf R\) 訓練時學一次，推理時常數、不隨幀變。
- 觀測：去掉融合、只做式 (11)，EE3D-R 的 MPJPE 141.22，是表 3 最差。只有直接姿態則平滑誤差 17.22。靜態（非學習）融合 88.31 / 平滑 9.93；完整模型 84.45 / 8.40。附錄 E.5：用 MLP 讓 \(Q,R\) 依賴輸入/狀態，MPJPE 91.15，差於全局常數 \(Q,R\) 的 84.45；他們解釋為過擬合。附錄 E.2：每 40 幀重置 SSM 或 Kalman 狀態沒有好處，連續演化最好（表 8 的數字在轉文本時沒有留下，不編）。表 10：訓練 20 ms，推理改 50 ms，MPJPE 87.45 對 84.45，平滑 11.90 對 8.40——拉長窗口主要傷平滑，不太傷這套直接姿態的精度。
- 對本任務：這就是「絕對頭 + 增量頭 + 固定 \(Q,R\) 的 Kalman」。增益 \(\mathbf K_t\) 仍隨 \(\Sigma\) 變，但 \(A,H,Q,R\) 時不變時 Riccati 會收到常數增益。倉庫已規定這不能當創新點。它能工作，是因為直接頭每幀都是絕對觀測，增量頭才是運動。S37 沒有這樣的直接頭。把同一套常數 \(Q,R\) 套到 S37 上，是在融合一個不存在的絕對觀測。

### 2.20 ByteTrack，ECCV 2022

- 原問題：遮擋時低分檢測框不該直接丟掉。
- 讀到的（§3）：檢測器每幀出框，按分數分成高、低。Kalman 只用來預測軌跡在當前幀的位置，以便與框做 IoU / Re-ID 關聯。高分框先配；沒配上的軌跡再與低分框配。沒配上的低分框當背景刪掉。新軌跡只從沒配上的高分框誕生。
- 假設：絕對觀測是檢測框（每幀、近似獨立噪聲）；運動模型是關聯的預測，不是姿態本身。
- 對本任務：檢測+跟蹤的最優組合是「框負責絕對位置，濾波器負責在框之間插一段運動」，而且低分框是第二次關聯，不是每步用固定 \(\alpha\) 混合兩個姿態網路。

### 2.21 OC-SORT，arXiv:2203.14360（預印本）

- 原問題：SORT 的 Kalman 在遮擋加非線性運動時為什麼崩。
- 關鍵公式（§3.1–3.2）：預測 \(\hat x_{t|t-1}=F\hat x\)，\(P\leftarrow FPF^\top+Q\)；更新
  \[
  K_t=P_{t|t-1}H^\top(HP_{t|t-1}H^\top+R)^{-1},\quad
  \hat x_{t|t}=\hat x_{t|t-1}+K_t(z_t-H\hat x_{t|t-1}).
  \]
  沒有觀測時的 dummy update 把先驗直接當後驗。速度由差分估計時，方差帶 \(1/(\Delta t)^2\)。遮擋 \(T\) 步後位置噪聲方差帶 \(T^2\sigma^2\)（式 (5)）。MOT17 上人的幀間位移只有 1.93 / 0.65 像素，一個像素的位置噪聲就足以讓速度估計和速度本身一樣大。
- 修復：觀測中心而不是估計中心。軌跡重新匹配上之後，用上次與這次的真實觀測拉一條虛擬軌跡，把中間的 Kalman 重更新（ORU，式 (6)–(7)）。方向一致性也用觀測而不是濾波狀態。
- 對本任務：這是「跟蹤器在沒有絕對觀測時方差按 \(T^2\) 長」的直接公式。S37 global 的 69 s 曲線不是這個形狀。若未來把根做成真正的速度積分，就必須有 OC-SORT 這種「重新看到絕對觀測就重錨」的路徑；dummy update（空包只複製 prev、協方差卻不長大）正是 `filter.py` 批評 `ZERO_EVENT_GATE` 的地方。

### 2.22 EvHandPose，arXiv:2303.02862（預印本）

- 原問題：事件只記錄亮度變化，靜止的手在恆定光照下不產生事件。
- 讀到的（§III-B，式 (2)）：同一段事件 \(E_{t_{n-1},t_n}\) 可以對應多個絕對手姿，因為靜止部位可以處於不同姿態，只要相對變化相同。他們把這叫做 motion ambiguity。EventHPE 一類方法預測相對 \(\Delta\theta\)，仍然解不開，因為相對姿態本身還依賴絕對姿態。他們用 Conv-GRU 帶上更長的事件歷史，並用預測的網格流去扭轉事件（對比度最大化），作為弱監督，不是在線積分器。
- 對本任務：事件同時是「哪裡在動」的絕對圖像座標和「動了多少」的差分。只做相對增量，會把不同的絕對姿態混成同一段事件。S37 用絕對圖像特徵回歸增量，正好卡在這個歧義的中間，兩邊都不是。

### 2.23 Ev2Hands，arXiv:2312.14157（預印本）

- 原問題：單目事件、兩隻交互的手。
- 讀到的（§3.1–3.3）：事件 \((x,y,t,p)\)。他們反對把時間投成 2D 圖像，改成點雲 \(\mathbf E_k=(x_k,y_k,t_k,P_k,N_k)\)，用 PointNet++ 回歸窗口**結束時刻**的 MANO、平移與旋轉。第三維是時間，不是深度。
- 對本任務：再次確認不能把事件的時間維說成深度測量。窗口端點的絕對回歸是全局估計器；他們沒有把上一窗口的姿態積分進來。

### 2.24 本倉庫模組（不是論文，但是 (c) 的原文）

`semkine/anchor.py` 文件字串：無差別地把遞推跟蹤器與凍結絕對預測每步混合，歷史上 19.26 mm 到 19.32 mm，接近中性。觸發不用 GT，三條內部訊號：NIS 持續高於卡方帶、後驗不確定度（實現為協方差跡）、剪影外事件比例。遲滯與冷卻避免每步觸發退回常數混合。融合是流形上兩個高斯的最大似然，線性化後
\[
\delta=(P^{-1}+J^{-\top}\Sigma^{-1}J^{-1})^{-1}J^{-\top}\Sigma^{-1}d,\quad x^+=x_{\mathrm{trk}}\operatorname{Exp}(\delta).
\]
預設根度量是 \(R^3\times SO(3)\) 而不是 \(SE(3)\)，因為 \(SE(3)\) 測地中點會把 0.6 rad 的旋轉分歧耦合成 5.9 mm 的平移，而本問題的誤差是頂點毫米。`AnchorTrigger` 預設：NIS 開閾值 \(2.0\times 51\)、關 \(1.2\times 51\)，跡 \(>0.05\)，剪影外比例開 0.20 / 關 0.10，冷卻 20 步，連續 2 步才開火。剪影外在真實幀、4 px 裕量上的標定：正確狀態 0.00，平移漂移 1 cm 仍 0.00，2 cm 為 0.23，4 cm 為 0.36，8 cm 為 0.65。它抓的是粗丟失，不是毫米級漂移。

`semkine/filter.py`：狀態加身體速度，\(v^-=e^{-\gamma\Delta t}v^+\)，\(\gamma=8\,\mathrm{s}^{-1}\)，否則手停了、事件也停了，外推停不下來。測量是切空間 \(\mathrm{Log}((X^-)^{-1}\otimes X^{\mathrm{net}})\)，\(H=[I\ 0]\)。\(\Sigma\) 來自該包 Fisher 資訊，而不是再學一個不確定度頭（文件寫 S9 時那種頭會在直接回歸旁邊變懶）。無事件則只傳播、\(P\) 增長；`ZERO_EVENT_GATE` 逐位回 prev 且不降低置信，只在極短空窗成立。NIS 是 \(\nu^\top S^{-1}\nu\)，期望等於新息維度。更新用 Joseph 形式，避免千步之後協方差失去正定、增益變號。通過標準是校準（NIS 覆蓋、標準差與誤差的秩相關），不是精度本身。

`eval_track.py` 的 `root_align` 是每幀減去第 0 個關節（腕）的座標，不對齊旋轉。`model.py` 的 `forward_packet`：關節頭讀該關節證據和 prev 的 3 維角；根頭讀池化特徵與 16 行證據，不讀 prev 的根；`prev_mlp(prev)` 加到 51 維輸出上；`predict_delta` 時輸出是 `delta+prev`。

## 3. 可觀測性、資訊來源與失效條件

對象：單目事件、240×180、50 ms 包、非剛性手、MANO 51 維，相機系平移與繞腕的 global_orient，15 個關節局部軸角。

資訊實際從哪裡來：

- 事件是對數亮度變化超過對比度閾值才觸發（EvHandPose 式 (2)，E-3DPSM 式 (1)）。恆定光照下，靜止部位不產生事件。一段事件可以對應多個絕對姿態（EvHandPose 的 motion ambiguity）。
- 事件的 \((x,y)\) 是圖像上的絕對座標；\(t\) 是時間，不是深度（Ev2Hands）。所以同一包裡既有「輪廓在圖像哪裡」的絕對訊息，也有「這 50 ms 怎麼動」的差分訊息。
- 單目投影把沿視線的平移和一部分旋轉混在一起。SLAHMR 的重投影在相機運動與人的平移之間欠定。Zhang 教程：單目有相似變換規範，視覺慣性仍剩繞重力的 yaw 與平移。本問題沒有第二個相機，不能把歷史上的深度假設當成新的獨立視角。
- prereg §9.2：完美路由下，S37 的 4624 維根輸入對所需旋轉 \(R^2=0\)。根旋轉的訊息只從「路由錯了」漏進來。119 維剪影殘差的資訊不低於這 4624 維。真實事件深度把誤差從約 7° 降到 4.5°–5.3°，但每部位 1 cm 的深度誤差在 8 cm 槓桿上約等於 7°，與估旋轉同難；prev 提供的深度與 prev 的深度誤差相關 0.74–0.98，殘差恒為零。深度不是免費的第二測量。
- `zgz_local` 事件最稀（中位 652 / 50 ms）。手指在遠起點的同包迭代不走向 GT。這是觀測不足，不是濾波器參數問題。

失效條件：

- 把剛體 SLAM 用在手上：TRAM / HaWoR 必須先把手從 BA 裡掩掉。手本身才是要估的非剛體，掩掉之後剩下的場景運動不是 MANO 狀態。
- 把 50 ms 的事件增量當 IMU：真實根旋轉只有 1.6°（global）或 3.0°（local），而 S37 的 TF 更新是 5.8°–7.6° 且與所需方向餘弦 0.05–0.11。增量的噪聲大於訊號。KITTI 已說明，位移很小時相對誤差會被漂移或亂跳主導。
- 空包：`ZERO_EVENT_GATE` 回 prev 但不增大不確定度。`filter.py` 寫這在毫秒級成立，在一秒未觀測時不成立。OC-SORT 的 dummy update 把位置方差放大成 \(T^2\sigma^2\)。
- 常數增益：只在 \(Q,R\) 與真實噪聲一致、且系統時不變時，Kalman 增益收到常數（E-3DPSM 把 \(Q,R\) 學成全局常數，並發現讓它們隨輸入變反而更差——那是因為他們每幀都有直接姿態）。S37 的課程把乾淨、小噪聲、大噪聲混在一起，Wiener 斜率是這三檔的平均；測試時 prev 接近真值，最優增益接近 0（prereg §9.1）。

### (a) 全局估計器、跟蹤器、以及最優組合

全局估計器：\(\hat x_k=g(y_k)\)，無記憶或只在窗口內看當前觀測。誤差不沿時間積分。若 \(g\) 穩定，誤差有界，方差等於測量噪聲，高頻抖。Ev2Hands 的窗口端點回歸、HaWoR 的相機系 MANO、E-3DPSM 的 \(\mathbf P_t^{\mathrm D}\)、ByteTrack 的檢測框，都是這一類。同包反復呼叫 \(g\)，結果就是 \(g\)，不會越走越遠。

跟蹤器：\(\hat x_k=\hat x_{k-1}\oplus u(y_k,\hat x_{k-1})\)，\(u\) 估的是增量。OC-SORT 式 (5)：沒有新觀測時，位置誤差的方差隨遮擋長度 \(T^2\) 增長。KITTI：增量法會漂，低速時相對誤差最大。E-3DPSM 式 (11) 與附錄 B：把增量加到上一狀態上，MPJPE 隨時間上升。WHAM / GVHMR 只對根平移做這種求和，並且用接觸或重力方向約束可觀子空間。噪聲可以很小，時間長了沒有絕對測量就沒有上界；若增量有非零均值，漂移是線性偏置而不是隨機遊走。

最優組合，在讀到的公式裡是新息形式的 Kalman，而不是固定 \(\alpha\)：

\[
\hat x\leftarrow \hat x^-+K(z-H\hat x^-),\quad
K=PH^\top(HPH^\top+R)^{-1}.
\]

（OC-SORT 式 (1)，E-3DPSM 式 (14)–(15)。）\(z-H\hat x\) 為零則更新為零。\(P\) 小或 \(R\) 大時 \(K\) 小，跟蹤器保持自己的低噪聲；\(P\) 大或絕對測量準時 \(K\) 變大。\(A,H,Q,R\) 都是常數時，\(P\) 收到穩態，\(K\) 變成常數——這就是 E-3DPSM 推理期的形態，也是 prereg §9.1 寫的 Wiener 增益 \(- \Sigma_n(\Sigma_r+\Sigma_n)^{-1}\)。它只在噪聲統計與設計點一致時最優。

頻率上的同一件事：VINS 用高頻 IMU 傳播、低頻視覺殘差、更低頻的回環因子。ORB-SLAM3 的跟蹤是局部重投影，重定位/閉環是偶爾的絕對約束，然後用位姿圖把修正傳開，不是每幀混合。ByteTrack 的檢測框每幀都到，Kalman 只預測關聯。人體線（WHAM、TRAM、HaWoR、GVHMR、GLAMR、SLAHMR）把相機系絕對姿態和世界系根軌跡分成兩個狀態：前者不積分，後者積分並被接觸、尺度或重投影錨住。

因此「最優」不是一個更漂亮的常數 \(\alpha\)。常數 \(\alpha\) 是平穩線性高斯的穩態特例。事件密度、遮擋、prev 是否已經準，都在變，增益必須能變成 0，並且絕對測量必須和跟蹤器不是同一個有偏信念。

### (b) S37 為什麼兩者都不是

根的一步（prereg §9.1，與 `model.py` 一致）：

\[
r_{\mathrm{out}}=r_{\mathrm{prev}}+W_r[f;e_0,\ldots,e_{15}]+b+g_{\mathrm{rot}}(\mathrm{prev}).
\]

\(W_r\) 不讀 \(r_{\mathrm{prev}}\)。\(g=\mathrm{prev\_mlp}\) 不看事件。可加可分，只能表示增益不隨「prev 與事件分歧」變化的融合。

它不是跟蹤器：

- 同包迭代 \(x\leftarrow f(\text{事件}_k,x)\) 從 GT 或從約 27 mm 外都收到 13–17 mm（global），等於閉環的 15.66 / 15.97。積分器在固定包上會反覆加上同一段運動，離開任何有界集。收縮教程的語言：映射在把軌跡拉向同一條特解。
- 69 s 按 5 s 分桶，RA 在 11.6–22.4 mm 之間波動，不隨時間增長。沒有 KITTI / OC-SORT / E-3DPSM 式 (11) 的漂移形狀。
- TF 旋轉更新 5.8°–7.6°，真實轉動 1.6° / 3.0°，方向餘弦 0.05–0.11。更新幾乎垂直於所需運動。手指方向餘弦 0.25，幅值 0.55 對 0.56 rad，像一個方向很差的增量，仍然不是低噪聲積分。
- TF 的 RA 是原地不動的 2.7–3.4 倍。好的跟蹤器在 prev 為 GT 時，輸出誤差應接近這 50 ms 的真實運動（原地不動那一欄），而不是比「什麼都不做」更差。

它不是全局估計器：

- 全局估計器輸出 \(g(y_k)\)，不把 \(r_{\mathrm{prev}}\) 加回去。S37 的單步不是同包不動點：TF 6.6–8.3 mm（global），迭代 8 次 13.4 / 16.1 mm。閉環坐在不動點上，單步停在半路上。
- 完美路由使 4624 維特徵對所需旋轉資訊為零。絕對圖像座標並沒有變成絕對旋轉測量；它們只在路由錯時洩漏 prev 誤差。
- 不動點誤差約 15 mm（global RA），差於本表基線的 global 10.99。信念本身就吵，而且吵在根旋轉：每步把根旋轉換成 GT，合併 RA 20.74→12.4。

它實際是什麼：一個收縮的逐包映射。線性圖像 \(e_k=\alpha e_{k-1}+b\)，穩態 \(e_\infty=b/(1-\alpha)\)，與收縮教程裡「擾動下停在球內」一致。\(\alpha\) 主要來自分不看事件的 `prev_mlp`（根斜率約 \(-0.24\) 到 \(-0.64\)，對上課程噪聲的 Wiener 係數）；事件只貢獻糾正增益 0.01–0.20。prereg 已把舊的「放大率 2.12」改讀成穩態信念誤差與 TF 單步之比，不是回路增益。local 上從遠點迭代不靠近 GT（3408 到 36.1），手指信念幾乎是空的；段內 0.5–2 s 的上升更像弱觀測通道上的短期走偏，不是 69 s 那種穩態。

### (c) 何時刷新，以及如何不再引入常數增益

三條觸發各自的失效：

- NIS。`filter.py`：期望等於新息維度，前提是新息零均值且 \(S\) 校準。S37 沒有協方差。若把 \(R\) 設成實測的 6°–8° 噪聲底，NIS 看起來正常，永遠不觸發——偏置被吸進噪聲模型。若把 \(R\) 設成真實運動的 1.6°，則每步都超閾，`anchor.py` 寫明這會退回已經測過接近中性的常數混合。NIS 發現的是「自報不確定度與殘差不符」，不是「信念有一個穩定的 10° 偏置」。
- 剪影外事件比例。標定顯示 1 cm 平移仍是 0，2 cm 才到 0.23，閾值 0.20。10° 根旋轉在 8 cm 槓桿上約 1.4 cm，落在這條曲線的死區附近。它適合 ORB-SLAM「點少於 15」或 VINS「輸出不連續」那種跟丟，不適合 S37 這種從第一個 5 s 桶就在的穩態誤差。
- 後驗跡 / log-det。只在濾波器承認自己無知時變大。沒有真實的 \(P\) 就沒有這條訊號。過度自信的 \(P\)（一致性失敗；本次未讀到 Bailey / Huang 原文，不引用他們的定理）會讓這條觸發沉默。`filter.py` 已警告：把不確定度頭和直接回歸並列，頭會變懶。

怎樣刷新才不會把常數增益帶回來：

1. 不要每步融合。`anchor.py` 的連續 2 步、冷卻 20 步就是為了這個。ORB-SLAM3 與 VINS 都是檢測到回環或跟丟才加入絕對約束。
2. 權重用資訊，不用固定 \(\alpha\)。`fuse()` 與 Kalman 的 \(K(P,R)\) 在該方向沒有資訊時增益為 0。E-3DPSM 的全局常數 \(Q,R\) 已佔住「學一個穩態增益」；本任務若再做一遍，沒有新資訊。
3. 絕對測量必須是另一個估計器。更長窗口、或不以 prev 為條件的檢測。與當前信念同源的第二次前向，誤差相關，資訊相加是重複計數。
4. 若映射本身收縮到壞信念，只改狀態不夠。收縮教程的擾動球：外加的一次替換會被 \(\alpha\) 忘掉，穩態仍由擾動 \(b\)（這裡是每步注入的信念誤差）決定。E-3DPSM 附錄 E.2 的負結果是：他們的濾波器已經會自我調節，週期重置沒有幫助。那不能推出「重置能修好一個錯誤的吸引子」。要改吸引子，必須讓更新在殘差為零時為零，並去掉不看事件的根回拉。

### (d) `zgz_local` 與 `zgz_global` 各測什麼，協議該補什麼

`eval_track.py`：每個 valid run 從該段起點 GT 加初始化噪聲出發，然後遞推。RA 每幀只減腕點，旋轉誤差全額留下，腕平移漂移從 RA 裡消失。

`zgz_global`：1 段 69 s。測的是長時閉環的穩態，不是漂移率。證據是 5 s 分桶平坦，以及同包不動點等於閉環誤差。它接近 Zhang 的長時絕對誤差，但不是：沒有做一次 \(SE(3)\) 或 \(\mathrm{Sim}(3)\) 對齊，而且每幀減掉了腕平移。腕每步被 `prev_mlp` 推 42–130 mm 這件事，RA 看不見（local 的 abs 89–110 mm 對保持段首 27.9 mm 才看得見）。

`zgz_local`：90 段，平均 0.7 s，每段從 GT 起。33% 的幀在初始化後 1 s 內。測的是短時、帶真值重初始化的行為，接近 Zhang 的相對誤差（子軌跡用起點對齊）和 WHAM 自己批評的 W-MPJPE\(_{100}\)（短段、前兩幀對齊，不反映長漂）。它不測 69 s 漂移。主表把兩條序列合成一個 RA，會把「常常被 GT 重新放下」算成跟蹤能力。prereg 已寫：3408 的 local 閉環 29.49 差於保持段首 28.2。

應寫進每個臂、且與主表分開的參照（prereg §8 已測，此處只說明它們對應文獻裡的哪一條零假設）：

| 參照 | 它固定的零假設 | 不過線代表什麼 |
|---|---|---|
| 原地不動（copy-prev） | 50 ms 更新沒有比「輸出 prev」更接近當前 GT | 增量的噪聲大於這 50 ms 的運動。S37 的 TF 已在這條線下方（更差） |
| 保持段首 | 整段不更新，誤差只來自分段內的真實運動 | 閉環沒有在跟。local 至少要兩種子都贏過它 |
| 同包迭代收斂點 | 閉環誤差是不是逐包信念，而不是積分出來的 | 閉環貼著它，就還在測信念精度，不是測跟蹤動力學 |

另外，按 Zhang §IV 與 WHAM 的長/短分離，補一組相對誤差：\(\Delta\in\{50\mathrm{ms},250\mathrm{ms},1\mathrm{s},5\mathrm{s}\}\)，每個窗口只在起點對齊（或只減起點腕、仍不對齊旋轉），報終點 RA。短 \(\Delta\) 看單步注入，長 \(\Delta\) 看是否真的開始漂。global 與 local 分開報，不要合成一個均值。腕平移只出現在 abs 列；RA 繼續不計平移，但診斷必須並列 abs，否則 local 的亂移根被隱藏。

### (e) 哪部分多餘、哪部分要優化、哪部分要增加

多餘，有文獻與本倉庫測量雙重支持：

- 根上的 `prev_mlp`。它是課程噪聲下的平均 Wiener 收縮，不看事件。測試時 prev 接近真值，最優增益為 0，它卻每步注入約 0.3 的信念誤差（prereg §9.1）。K0/K1/K2 已落在同一條常數增益權衡上。E-3DPSM 佔住了「再學一個常數 \(Q,R\)」。
- 用 4624 維絕對池化特徵回歸根增量。完美路由下資訊為零。加大這個頭是換表示，不增加新息。
- 把 local 與 global 合成一個 RA 再選點。Zhang 與 WHAM 都寫明短段相對誤差和長時絕對誤差回答不同問題。
- 為了 69 s global 去上 scheduled sampling 或 DAgger。那條文獻修的是 \(T^2\) 複利；global 沒有這條曲線。Professor Forcing 還指出 scheduled sampling 有偏。

要優化的是信念本身，而且是方向不是幅值：

- 根旋轉的更新方向（餘弦 0.05–0.11）。119 維無學習剪影殘差已經不差於 4624 維 \(z\)（ridge 7.69° 對 8.04°，MLP 7.08° 對 8.44°）。優化對象是殘差與槓桿的乘積，不是再加一條加性幾何。
- 手指更新方向（餘弦 0.25，幅值已對上）。這是關聯/方向，不是缺一個 GRU。local 事件稀，遠起點迭代不收斂，先承認該通道資訊不夠，再談時序。
- 協議參照線。不改網路也能阻止把「重初始化」讀成「在跟蹤」。

要增加的，文獻裡還沒有被本倉庫佔住、而且 E-3DPSM 的常數融合不算：

- 新息：殘差為零則該項為零。幾何只作為把殘差映到根切空間的乘法因子（DSO 對加性先驗的警告，加上本倉庫契約）。這是觀測模型的改變，不是再一個損失項。
- 與 50 ms 信念不同源的絕對估計，低頻或按幾何一致性觸發，資訊加權，且觸發後要改的是吸引子而不只是狀態。HaWoR/TRAM 的「另一條路徑」是場景 SLAM，剛體假設不能直接搬；能搬的是結構：絕對路徑不讀遞推狀態。
- 不確定度只在新息近零均值之後才用來做 NIS。在那之前，剪影外比例只當作粗跟丟，不當作 10° 偏置的檢測器。

## 4. 可證偽假說

假說 H1（本文認為最可能被否定的一條）。在不改 S37 的收縮映射（根上的 `prev_mlp` 與非新息根頭）的前提下，用 NIS、剪影外事件比例或後驗跡觸發全局刷新，就能把 `zgz_global` 的閉環 RA 降到同包迭代信念（約 13–17 mm）以下，並維持 5 s 分桶不隨時間上升。

反對證據已經很強：分桶是平的，沒有可被回環消去的累積漂移；剪影外閾值對 1 cm 不敏感；沒有校準的 \(S\) 時 NIS 要麼不響、要麼每步都響；收縮映射會把一次狀態替換拉回原信念；E-3DPSM 附錄 E.2 在他們自己的調節良好的濾波器上，週期重置沒有收益；本倉庫常數混合歷史上接近中性。

假說 H2（建設性的一條）。把根更新改成新息 \(\Delta\xi=(J^\top WJ+\Lambda)^{-1}J^\top W r\)，\(r=0\Rightarrow\Delta\xi=0\)，並把 `prev_mlp` 的根 6 維置零。則在 prev = GT 的單步上，根旋轉誤差的方向餘弦相對 S37 的 0.05–0.11 上升，且誤差不大於原地不動的根旋轉（global 真實轉動 1.6° 那一檔），同時 69 s 分桶斜率仍接近 0。

最強的反對證據是 prereg §9.2：乾淨子集上「保持 prev」只有 2.25°，所有解碼器包括 119 維殘差仍注入 3.9°–5.0°。所以「殘差為零則更新為零」是必要條件，但現有殘差特徵在乾淨樣本上仍然比不動更差。H2 若只換特徵、不保證乾淨點的增益為 0，會被這一格否定。

## 5. 最小可遷移機制

數學：只改根的 6 維。令 \(r\in\mathbb R^{n}\) 為 prev 投影剪影上的事件殘差（prereg 已用的 119 維是一個現成的 \(r\)），\(J\) 為路由時已有的投影槓桿把殘差映到根切空間的矩陣。更新
\[
\Delta\xi=(J^\top W J+\Lambda)^{-1}J^\top W r,\qquad W=W(f_{\mathrm{event}}),
\]
並要求實現上 \(r=0\Rightarrow W r=0\Rightarrow\Delta\xi=0\)（權重可以依賴事件特徵，但必須與 \(r\) 相乘，不能另有 \(W_r r_{\mathrm{prev}}\) 或 `prev_mlp` 的根項）。\(\Lambda\) 只放真正不可觀的方向（例如沿視線的平移與 prereg 已指出的弱方向），不是把狀態拉向訓練均值。旋轉在相機系複合 \(R\leftarrow\exp(\omega)R_{\mathrm{prev}}\)，與 `anchor.py` / `filter.py` 的右不變更新一致，避免軸角向量平均。

絕對刷新不進這個每步式子。若另有一條不讀 prev 的長窗口估計 \((x_{\mathrm{abs}},\Sigma)\)，只在 `AnchorTrigger` 開火時呼叫已有的 `fuse()`。開火條件在 H1 被否定之前，不應指望它降低 69 s 誤差；它只處理剪影外比例已經超過 0.20 的粗跟丟。

接入點：`model/model.py` 的 `forward_packet`，在 `prev_mlp` 加到輸出之後、`delta+prev` 之前，把根的 6 維從 `prev_mlp` 裡減掉，並用上面的 \(\Delta\xi\) 替換 `root_head` 對根旋轉的貢獻。關節頭先不動。空包仍走 `ZERO_EVENT_GATE`。不在此步引入新的 Kalman 狀態；`filter.py` 要等新息近零均值再接，否則 NIS 無意義。

成本：119 維殘差與 \(6\times 6\) 求解相對 S37 整步 0.827 GMACs 是小項。精確 MACs 與延遲本次未測，不報數字。可學習權重若限制在殘差之後的小映射，參數量級遠小於現在根頭的 \(6\times 4624\)。SDF 的具體實現與 +0.2–0.5 ms 是脈絡檔裡的估計，不是本次測量。

## 6. 最小判別實驗

不訓練完整模型。固定 S37 的兩個 checkpoint，在 zgz 上、與 prereg 同一 50 ms 協議，只替換根更新。對照必須包含：

- A：S37 原樣。
- B：只把 `prev_mlp` 根 6 維置零（簡單修復；prereg 已寫 `no_prevmlp` 閉環會散到 RA 81–101，這裡預期復現，用來證明不能只刪回拉）。
- C：常數 \(\delta\)-trust（`eval_track.py` 已有 `delta_trust`），例如 0.5。預期落在已測過的中性附近，不低於同包信念。
- D：普通時序平滑，例如對根旋轉做固定增益指數滑動，不看殘差。預期把抖動和偏差一起抹平，TF 仍差於原地不動。
- E：本文的新息更新，\(r=0\) 時用有限差分或零殘差輸入檢查 \(\Delta\xi=0\)（契約）。
- F：E 再加每步 `fuse` 一個凍結的絕對頭（常數增益的最近鄰，對齊 E-3DPSM 的穩態）。預期不高於 E，用來證明融合不是來源。

每個臂報告 global 與 local 分開的閉環 RA，並附三條參照：原地不動、保持段首、同包迭代 8 次。另附根旋轉的 TF 方向餘弦與度數。主表仍只用 `tools/report_table.py`，本實驗若未跑 `make_s36_row.py` 就不作為結果上報。

預期：E 的 global 同包不動點不高於 A，TF 根旋轉方向餘弦高於 0.11，且 5 s 分桶斜率仍接近 0。F 不優於 E。B 發散或明顯差於 A。

否定 H2：E 在乾淨 prev（TF）上的根旋轉誤差仍大於原地不動的根旋轉，或方向餘弦仍 \(\le 0.11\)，或零殘差輸入下 \(\|\Delta\xi\|\) 不為 0。否定 H1：若在 A 上只加觸發式 `fuse`、不改根頭，global RA 就降到迭代信念以下——那會推翻本文對「刷新現有 S37 無用」的判斷。

最近鄰區隔：E 的每步式子裡沒有第二個直接姿態頭，也沒有常數 \(Q,R\) 的 48 維 Kalman（那是 E-3DPSM）。F 故意做成那個最近鄰，必須差於或等於 E，否則創新點不成立。

## 7. 六個創新問題

真正要解決的矛盾：全局估計器誤差有界但吵，跟蹤器吵聲小但會漂；S37 的穩態誤差等於一個吵的信念，單步又比不動更吵，所以兩頭的好處都沒拿到。矛盾不是「還缺一個時序模組」，而是根更新既不是 \(g(y)\) 也不是運動增量。

新增的資訊或約束從哪裡來：從事件相對 prev 投影剪影的殘差，以及已有 FK 的槓桿（乘法）。不從第二個相機來，不從把歷史深度再當一次獨立測量來，也不從 `prev_mlp` 的訓練均值來。訓練均值是歷史先驗的重複使用，prereg 已把它識別成 Wiener 收縮。

事件的異步、時間或稀疏性貢獻了什麼：在本機制裡，時間首先用來定義 50 ms 包內的殘差；稀疏性決定 \(\Lambda\) 與是否觸發，而不是再學一個密度歸一化。Ev2Hands 說明時間不是深度。E-3DPSM 表 10 說明把 20 ms 改成 50 ms 主要改變平滑而不是他們的直接姿態精度，所以「把窗口加長」本身不是創新。

相對最近的方法，變化在哪：相對 E-3DPSM，變化在觀測模型（新息殘差，且沒有每幀直接姿態頭）而不是再做一個常數 \(Q,R\) 融合。相對 WHAM/HaWoR，不引入場景 SLAM（剛體假設不成立），變化在狀態更新：根切空間的資訊形式新息，而不是根速度的開環求和。相對 OC-SORT，不在 2D 框上做 Kalman，而是要求 3D 根更新在殘差為零時為零。

為什麼更簡單的修復不夠：只修可見性（完美路由）已使根特徵資訊歸零。普通時序濾波不看殘差，會把 6°–8° 的正交更新平滑成另一個偏置。常數 \(\delta\)-trust 是穩態 Kalman 的標量版，歷史上接近中性。把 4624 維 \(z\) 換成別的絕對池化，仍是「絕對座標回歸增量」，EvHandPose 的運動歧義還在。

什麼結果會否定假說：見 §4 與 §6。H1 被否定的條件是只加觸發刷新就降到信念以下。H2 被否定的條件是乾淨 prev 上新息更新仍差於原地不動，或零殘差契約失敗。

## 8. 限制與誠實聲明

- 深讀 24 篇，見 §2。Sturm、TLD、Mahony、Bailey、Huang FEJ、Lohmiller 1998 的 PDF 未取得，目錄裡標明，沒有把它們的公式寫進來。ATE/RPE 的可引用定義來自 Zhang & Scaramuzza 教程；他們把提出歸於 Sturm，本文未核對 Sturm 原文用詞。
- NEES 的標準定義本次沒有讀到原文。NIS 的操作型定義來自 `filter.py` 與 OC-SORT / E-3DPSM 寫出的新息。不把 NEES 寫成已讀公式。
- E-3DPSM、WHAM、TRAM、GLAMR、OC-SORT、EvHandPose、Ev2Hands、scheduled sampling、DAgger、Professor Forcing、IEKF、DSO 在本次 abs 頁沒有 DOI 或全文開頭沒有錄用句，正文裡按預印本處理。倉庫稱 E-3DPSM 為 CVPR 2026，本文未核實。
- 收縮教程式 (2.33) 的係數、E-3DPSM 表 8 的重置數字，HTML 轉文本時不完整，未編造。
- ORB-SLAM3、WHAM、HaWoR 的官方倉庫除 HaWoR README 與 WHAM 項目頁 URL 外，沒有逐函式核對。代碼證據限於本倉庫讀過的 `anchor.py`、`filter.py`、`eval_track.py`、`model.py`。
- CCF 與中科院分區未打開官方表。CVPR/ICCV/ECCV/NeurIPS 的 A 類是通行口徑，標了待核；其餘分區一律待核。
- §5 的算量是量級判斷，不是 profiler 輸出。
- H1/H2 尚未跑 §6 的對照。本文是文獻與已有探針的判讀，不是新的主行結果。
