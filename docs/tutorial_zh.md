# SHAP-Compass 使用教學

> **SHAP-Compass**：基於單位圓投影的 SHAP 歸因方向性分群框架（GeoAI 解釋性方法）
> 版本：0.2.0 | 授權：MIT

---

## 目錄

1. [什麼是 SHAP-Compass？](#1-什麼是-shap-compass)
2. [安裝](#2-安裝)
3. [核心概念](#3-核心概念)
4. [快速入門（3 行完成分析）](#4-快速入門)
5. [完整分析流程](#5-完整分析流程)
6. [理解輸出結果](#6-理解輸出結果)
7. [DCI 方向一致性指數](#7-dci-方向一致性指數)
8. [進階設定](#8-進階設定)
9. [常見問題](#9-常見問題)
10. [引用格式](#10-引用格式)

---

## 1. 什麼是 SHAP-Compass？

### 解決什麼問題？

SHAP 是目前最主流的模型解釋方法，但標準 SHAP 分析只能告訴你「**哪些特徵重要**」（全局排名），無法回答更深層的問題：

> **同一個特徵在不同地理環境下，模型的歸因方向是否會系統性改變？**

例如：「降雨量」在某個沿海農業區可能促進地下水汙染蓄積（正向歸因），在另一個山區水文背景下卻造成稀釋（負向歸因）。這種**機制切換**是 SHAP Summary Plot 無法揭示的。

### SHAP-Compass 的做法

SHAP-Compass 將每個樣本的**特徵值**與**歸因值**編碼為一個方向角 θ，投影到單位圓上，使所有特徵在距離計算中平等參與，然後透過 SOM + Ward 聚類找出**歸因方向相似的歸因機制分區**（attribution regimes）。

```
特徵值 (X) ──┐
              ├── Z 標準化 ── θ = arctan2(Z^S, Z^F) ── (cos θ, sin θ)
歸因值 (S) ──┘                                              │
                                                       SOM 降維
                                                            │
                                                      Ward 階層聚類
                                                            │
                                                  k 個歸因機制分區
                                                            │
                                                   DCI 排名 + 品質指標
```

---

## 2. 安裝

### 方法一：從 GitHub 安裝（推薦）

```bash
git clone https://github.com/GeoAI-Sustainability-Lab/SHAP-Compass.git
cd SHAP-Compass
pip install -e ".[shap]"
```

`[shap]` 是選用 extras——只有範例腳本需要它（用來計算 SHAP 歸因值）。SHAP-Compass 套件本身不依賴 shap。

### 方法二：直接安裝（PyPI 發布後）

```bash
pip install "shap-compass[shap]"
```

### 依賴套件

SHAP-Compass 會自動安裝以下依賴：

| 套件 | 用途 |
|------|------|
| numpy | 數值計算 |
| pandas | 資料表操作 |
| scipy | 圓周統計 |
| scikit-learn | StandardScaler、Ward 聚類、品質指標 |
| minisom | 自組織映射 |
| matplotlib | 視覺化 |

### 驗證安裝

```python
import shap_compass
print(shap_compass.__version__)
# 應輸出: 0.2.0
```

---

## 3. 核心概念

在使用 SHAP-Compass 之前，理解以下概念有助於正確解讀結果。

### 3.1 Z^F 與 Z^S

| 符號 | 意義 | 計算方式 |
|------|------|---------|
| Z^F | 標準化特徵值 | (特徵值 - 全體平均) / 全體標準差 |
| Z^S | 標準化歸因值 | (SHAP 值 - 全體平均) / 全體標準差 |

Z^F 代表「環境條件相對於全體的位置」，Z^S 代表「模型歸因相對於全體的位置」。

### 3.2 方向角 θ

```
θ = arctan2(Z^S, Z^F)
```

θ 編碼了**特徵值與歸因值的耦合關係**：

| θ 範圍 | Z^F | Z^S | 意義 |
|--------|-----|-----|------|
| 0°–90° | 正 | 正 | 特徵偏高，歸因正向（直覺一致） |
| 90°–180° | 負 | 正 | 特徵偏低，歸因卻正向（非線性放大） |
| -180°–-90° | 負 | 負 | 特徵偏低，歸因負向（直覺一致） |
| -90°–0° | 正 | 負 | 特徵偏高，歸因卻負向（交互抑制） |

### 3.3 單位圓投影

```
(cos θ, sin θ)
```

投影到單位圓上**移除了信號強度 r**，只保留方向資訊。這確保所有特徵在分群過程中平等參與，不受特徵間相關性 ρ 的影響。

### 3.4 DCI（方向一致性指數）

```
DCI = R̄_axial = √(mean(cos 2θ)² + mean(sin 2θ)²)
```

DCI 量化某特徵在不同機制分區間的**歸因方向一致程度**，值域為 [0, 1]：

- **DCI 高（≥0.75）**：歸因方向跨分區一致（**通用型驅動因子**，適合全區通用指標）
- **DCI 低（<0.25）**：歸因方向隨環境情境劇烈變化（**情境依賴型**，需分區管理）

注意：DCI 採用 2θ 軸向轉換，θ 與 θ + π 視為同一軸，因此衡量的是「方向軸」而非「方向向量」的一致性。

---

## 4. 快速入門

只需 **3 行程式碼**即可完成完整分析：

```python
from shap_compass import SHAPCompass

# 第 1 行：初始化（輸入特徵矩陣 + 歸因矩陣）
compass = SHAPCompass(features, shap_values, feature_names=names, target=target)

# 第 2 行：執行分析
results = compass.fit(som_grid=(9, 9), n_regimes=6)

# 第 3 行：查看結果
results.summary()
```

### 輸入資料格式

| 參數 | 型態 | 形狀 | 說明 |
|------|------|------|------|
| `features` | numpy array | (n_samples, n_features) | 原始特徵值矩陣 |
| `attributions` | numpy array | (n_samples, n_features) | SHAP 歸因值矩陣（或任何歸因方法） |
| `feature_names` | list of str | (n_features,) | 特徵名稱（選填，預設 F1, F2...） |
| `target` | numpy array | (n_samples,) | 目標變數（選填，用於分區依降序重新編號） |

### 最小可運行範例

```python
import numpy as np
from shap_compass import SHAPCompass

# 模擬資料
np.random.seed(42)
features = np.random.randn(200, 5)
shap_values = np.random.randn(200, 5)
target = np.random.rand(200) * 10  # 例如汙染濃度

# 執行
compass = SHAPCompass(
    features,
    shap_values,
    feature_names=["氣溫", "海拔", "降雨", "坡度", "NDVI"],
    target=target,
)
results = compass.fit(som_grid=(5, 5), n_regimes=3)
results.summary()
```

---

## 5. 完整分析流程

以下示範從原始資料到完整分析的標準流程。

### 5.1 準備資料

#### 方式 A：直接使用套件內建的 CONUS 公開資料集

套件已內建美國地質調查所（USGS）公開的 CONUS 地下水硝酸鹽資料集
（Ransom et al. 2021，公共領域），共 12,082 口井、29 個特徵：

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
import shap
from shap_compass import SHAPCompass, load_conus_nitrate, CONUS_FEATURE_NAMES

# 載入內建資料集
df = load_conus_nitrate()
features = df[CONUS_FEATURE_NAMES].values    # (12082, 29)
target = df["NO3"].values                    # (12082,) NO3 in mg/L

# 訓練模型 + 計算 SHAP
model = GradientBoostingRegressor(random_state=42).fit(features, np.log1p(target))
shap_values = shap.TreeExplainer(model).shap_values(features)
```

#### 方式 B：使用你自己的資料

```python
import pandas as pd
from shap_compass import SHAPCompass

# 載入資料
df = pd.read_csv("your_data.csv")
shap_df = pd.read_csv("your_shap_values.csv")

feature_names = ["Temperature", "Elevation", "Precipitation",
                 "Slope", "Forest_Cover", "Farmland_Ratio"]

features = df[feature_names].values          # (n, p)
shap_values = shap_df[feature_names].values  # (n, p)
target = df["your_target_column"].values     # (n,) 選填
```

### 5.2 初始化與執行

```python
compass = SHAPCompass(
    features=features,
    attributions=shap_values,
    feature_names=feature_names,
    target=target,
)

results = compass.fit(
    som_grid=(9, 9),        # SOM 網格大小（建議 n_samples^0.5 附近）
    n_regimes=6,            # 機制分區數（可透過 η² 拐點法決定）
    use_som=True,           # 使用 SOM 降維（建議 True）
    som_sigma=1.5,          # SOM 鄰域半徑
    som_lr=0.5,             # SOM 學習率
    som_iterations=10000,   # SOM 迭代次數
    random_state=42,        # 隨機種子（可復現性）
)
```

### 5.3 查看結果摘要

```python
results.summary()
```

輸出範例：
```
============================================================
  SHAP-Compass Analysis Results
============================================================
  Samples:    2375
  Features:   17
  Regimes:    6
  eta^2 (target): 0.3510

  Regime sizes:
    R1:   340 (14.3%)
    R2:   381 (16.0%)
    R3:   388 (16.3%)
    R4:   639 (26.9%)
    R5:   298 (12.5%)
    R6:   329 (13.9%)

  DCI ranking (top 5):
    1. Leaching_Risk        DCI=0.978 (high)
    2. Rainfall             DCI=0.952 (high)
    3. Temperature          DCI=0.881 (high)
    4. Agri_Ratio           DCI=0.795 (high)
    5. Clay_Sand_Ratio      DCI=0.732 (medium)
============================================================
```

---

## 6. 理解輸出結果

### 6.1 機制分區標籤

```python
# 每個樣本的分區歸屬（1-indexed）
labels = results.labels  # np.array, shape (n_samples,)

# 加入原始資料
df["Regime"] = results.labels
```

分區按目標變數降序編號：R1 為目標值最高的分區。

### 6.2 方向角與信號強度

```python
# 每個樣本在每個特徵上的方向角（弧度）
theta = results.theta  # (n_samples, n_features)

# 信號強度（距原點的距離）
r = results.r  # (n_samples, n_features)

# 分區質心方向角
group_theta = results.group_theta  # (n_regimes, n_features)
```

### 6.3 DCI 表格

```python
dci_df = results.dci
print(dci_df)
```

| 欄位 | 說明 |
|------|------|
| `feature` | 特徵名稱 |
| `DCI` | 方向一致性指數（0–1，越高跨分區越一致） |
| `rank` | DCI 由高至低排名 |
| `band` | 等級：high / medium / low / context-dependent |

### 6.4 標準化值

```python
ZF = results.ZF  # 標準化特徵值 (n_samples, n_features)
ZS = results.ZS  # 標準化歸因值 (n_samples, n_features)
```

### 6.5 分區級分析

```python
# 查看某分區的特徵
r1_mask = results.labels == 1
r1_mean_ZF = results.ZF[r1_mask].mean(axis=0)
r1_mean_ZS = results.ZS[r1_mask].mean(axis=0)

# 哪些特徵的 Z^F 和 Z^S 異號？（非線性歸因行為）
opposite_sign = (r1_mean_ZF * r1_mean_ZS) < 0
print(f"R1 異號特徵: {np.array(feature_names)[opposite_sign]}")
```

---

## 7. DCI 方向一致性指數

DCI 是 SHAP-Compass 的核心產出之一，提供**因地制宜 vs 一刀切**的決策依據。

### 7.1 解讀方式

```python
dci_df = results.dci

# 高 DCI 特徵（band == "high"）→ 通用型驅動因子，適合全區指標
universal = dci_df[dci_df["band"] == "high"]
print("跨分區一致的通用驅動因子：")
print(universal[["feature", "DCI"]].to_string(index=False))

# context-dependent 特徵 → 機制隨分區劇烈變化，需分區管理
context_dep = dci_df[dci_df["band"] == "context-dependent"]
print("情境依賴特徵（需分區管理）：")
print(context_dep[["feature", "DCI"]].to_string(index=False))
```

### 7.2 等級分級

DCI 採用**固定四級分區**（非依資料分位數）：

| 等級 (band) | DCI 範圍 | 特徵行為 | 管理意涵 |
|---|---|---|---|
| **high** | ≥ 0.75 | 歸因方向跨分區高度一致 | 適合作為全區通用監測指標 |
| **medium** | 0.50–0.75 | 歸因方向中度一致 | 多數分區一致，少數分區有偏差 |
| **low** | 0.25–0.50 | 歸因方向跨分區明顯分歧 | 可考慮針對特定分區微調管理 |
| **context-dependent** | < 0.25 | 歸因方向幾乎隨分區任意變化 | 必須依分區分別制定管理標準 |

---

## 8. 進階設定

### 8.1 不使用 SOM（直接 Ward 聚類）

適用於小樣本（< 500）或不需要拓撲保序的情境：

```python
results = compass.fit(
    n_regimes=4,
    use_som=False,    # 跳過 SOM，直接在樣本層級 Ward 聚類
)
```

### 8.2 k 值選擇

使用 η² 拐點法找最佳分區數：

```python
from shap_compass.clustering import eta_squared_scan
from shap_compass.transform import full_transform

tx = full_transform(features, shap_values)
eta_results = eta_squared_scan(tx["cossin"], target, k_range=range(2, 13))

for k, eta in eta_results.items():
    print(f"k={k}: η²={eta:.4f}")
```

觀察 η² 增量開始趨緩的拐點即為建議的 k 值。

### 8.3 SOM 超參數調整

| 參數 | 預設值 | 建議範圍 | 說明 |
|------|--------|---------|------|
| `som_grid` | (9, 9) | (5,5) 至 (20,20) | 約為 √n_samples |
| `som_sigma` | 1.5 | 1.0–3.0 | 鄰域半徑，越大越平滑 |
| `som_lr` | 0.5 | 0.1–1.0 | 學習率 |
| `som_iterations` | 10000 | 5000–50000 | 迭代次數 |

### 8.4 使用非 SHAP 歸因方法

SHAP-Compass 不限於 SHAP，任何能產生**逐樣本 × 逐特徵歸因矩陣**的方法皆可使用：

```python
# 使用 LIME 歸因值
compass = SHAPCompass(features, lime_attributions, feature_names=names)

# 使用 Integrated Gradients
compass = SHAPCompass(features, ig_attributions, feature_names=names)

# 使用 Permutation Importance（逐樣本版本）
compass = SHAPCompass(features, perm_importance, feature_names=names)
```

### 8.5 存取中間結果

```python
# SOM 神經元層級資料
results.neuron_theta    # (n_neurons, n_features) 神經元方向角
results.neuron_r        # (n_neurons, n_features) 神經元信號強度
results.neuron_labels   # (n_neurons,) 神經元分區標籤
results.neuron_sizes    # (n_neurons,) 每個神經元的樣本數
```

### 8.6 品質指標 M01–M21

```python
from shap_compass import compute_all_metrics

metrics = compute_all_metrics(
    results, target=target,
    features_raw=features, attributions_raw=shap_values,
)
# metrics 是一個 dict，每個 M01–M21 都會獨立輸出
```

關鍵指標：
- **M05 == 1.0**：所有 SOM 神經元都有樣本（無 dead neuron）
- **M19 min fraction ≥ 0.03**：最小分區占比 ≥ 3%
- 核心三選一：**M13**（自助 ARI 穩定性）、**M18**（低濃度區段 η²）、**M20**（相鄰分區目標差距）

---

## 9. 常見問題

### Q1: 我的資料適合用 SHAP-Compass 嗎？

**適合的條件：**
- 有一個預測模型（任何類型）
- 能計算逐樣本的歸因值（SHAP、LIME 等）
- 樣本數 ≥ 100（建議 ≥ 500）
- 特徵數 ≥ 3

**不適合的情境：**
- 僅有全局特徵重要性排名（無逐樣本歸因）
- 樣本數 < 50（SOM 難以訓練）

### Q2: 分區數 k 該怎麼選？

建議使用 η² 拐點法（見 §8.2）。經驗法則：
- 小資料集（< 500 樣本）：k = 3–5
- 中資料集（500–3000 樣本）：k = 4–8
- 大資料集（> 3000 樣本）：k = 5–10

### Q3: DCI 的等級是固定還是相對的？

DCI 採用**固定閾值**（0.25 / 0.50 / 0.75），與資料分布無關，便於跨案例比較。

也可以自行設定：
```python
dci_df = results.dci
my_universal = dci_df[dci_df["DCI"] >= 0.80]  # 自訂更嚴格的「通用」閾值
```

### Q4: SHAP-Compass 是因果推論工具嗎？

**不是。** SHAP-Compass 是描述性分析框架，刻畫模型歸因行為的統計規律。分區間的機制差異反映**模型學習到的統計關聯**，不能直接推論為物理因果關係。

### Q5: 可以用在深度學習模型嗎？

可以，只要你能產出逐樣本的歸因矩陣。例如：
- CNN + Grad-CAM（展平為向量）
- Transformer + Attention Weights
- 任何模型 + SHAP DeepExplainer

### Q6: SOM 和直接 Ward 哪個好？

| 方法 | 優點 | 缺點 | 建議使用情境 |
|------|------|------|------------|
| SOM + Ward | 拓撲保序、抗噪、可視化 | 需調超參數 | n > 500，需要 SOM 網格視覺化 |
| 直接 Ward | 簡單、無超參數 | 受異常值影響、無拓撲結構 | n < 500，快速探索 |

### Q7: 為什麼分區編號用 R / TG / UG 不一樣？

`regime_prefix` 是繪圖選項，預設 `R`（Regime），你可以改成任何字串（例如區域代號 `Z`、`G`、`Cluster_` 等），會反映在 SOM 網格、bilayer 熱力圖、unit-circle 子圖等的標籤上。資料本身不受影響。

---

## 10. 引用格式

如果您在研究中使用了 SHAP-Compass 套件，請參考 repo 根目錄的 `CITATION.cff`
取得最新的引用資訊。GitHub 會自動依此檔案產生「Cite this repository」按鈕。

若使用套件內建的 CONUS 公開資料集，請同時引用原始資料來源：

> Ransom, K.M., Nolan, B.T., Stackelberg, P.E., Belitz, K., Fram, M.S.
> (2021). *Machine learning predictions of nitrate in groundwater used
> for drinking supply in the conterminous United States: data release*.
> U.S. Geological Survey. <https://doi.org/10.5066/P9PQ622D>

---

## 附錄：數學公式速查

| 符號 | 公式 | 說明 |
|------|------|------|
| Z^F | (x - μ_x) / σ_x | 標準化特徵值 |
| Z^S | (s - μ_s) / σ_s | 標準化歸因值 |
| θ | arctan2(Z^S, Z^F) | 方向角 |
| r | √(Z^S² + Z^F²) | 信號強度 |
| COSSIN | (cos θ, sin θ) | 單位圓投影（2p 維） |
| R̄ | √(mean(cos θ)² + mean(sin θ)²) | 平均合成長度 |
| R̄_axial | √(mean(cos 2θ)² + mean(sin 2θ)²) | 軸線版（θ 與 θ+π 視為同軸） |
| **DCI** | **R̄_axial（跨分區質心）** | **方向一致性指數，值域 [0, 1]** |
| η² | SS_between / SS_total | 目標變數區分力 |

---

*SHAP-Compass — 讓 SHAP 從「哪些特徵重要」走向「模型如何因地制宜」*
