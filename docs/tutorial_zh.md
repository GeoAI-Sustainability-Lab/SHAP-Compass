# D-Profile 使用教學

> **D-Profile**：基於單位圓投影的 SHAP 歸因方向性分群框架
> 版本：0.1.0 | 授權：MIT

---

## 目錄

1. [什麼是 D-Profile？](#1-什麼是-d-profile)
2. [安裝](#2-安裝)
3. [核心概念](#3-核心概念)
4. [快速入門（3 行完成分析）](#4-快速入門)
5. [完整分析流程](#5-完整分析流程)
6. [理解輸出結果](#6-理解輸出結果)
7. [CDI 情境依賴指數](#7-cdi-情境依賴指數)
8. [進階設定](#8-進階設定)
9. [常見問題](#9-常見問題)
10. [引用格式](#10-引用格式)

---

## 1. 什麼是 D-Profile？

### 解決什麼問題？

SHAP 是目前最主流的模型解釋方法，但標準 SHAP 分析只能告訴你「**哪些特徵重要**」（全局排名），無法回答更深層的問題：

> **同一個特徵在不同地理環境下，模型的歸因方向是否會系統性改變？**

例如：「降雨量」在屏東平原促進汙染蓄積（正向歸因），在中央山脈卻稀釋汙染（負向歸因）。這種**機制切換**是 SHAP Summary Plot 無法揭示的。

### D-Profile 的做法

D-Profile 將每個樣本的**特徵值**與**歸因值**編碼為一個方向角 θ，投影到單位圓上，使所有特徵在距離計算中平等參與，然後透過 SOM + Ward 聚類找出**歸因方向相似的群體**。

```
特徵值 (X) ──┐
              ├── Z 標準化 ── θ = arctan2(Z^S, Z^F) ── (cos θ, sin θ)
歸因值 (S) ──┘                                              │
                                                       SOM 降維
                                                            │
                                                      Ward 階層聚類
                                                            │
                                                    k 個方向性群組
                                                            │
                                                   CDI 排名 + 品質指標
```

---

## 2. 安裝

### 方法一：從 GitHub 安裝（推薦）

```bash
git clone https://github.com/YOUR_USERNAME/dprofile.git
cd dprofile
pip install -e .
```

### 方法二：直接安裝（PyPI 發布後）

```bash
pip install dprofile
```

### 依賴套件

D-Profile 會自動安裝以下依賴：

| 套件 | 用途 |
|------|------|
| numpy | 數值計算 |
| pandas | 資料表操作 |
| scipy | 圓周統計 |
| scikit-learn | StandardScaler、Ward 聚類 |
| minisom | 自組織映射 |
| matplotlib | 視覺化 |

### 驗證安裝

```python
import dprofile
print(dprofile.__version__)
# 應輸出: 0.1.0
```

---

## 3. 核心概念

在使用 D-Profile 之前，理解以下概念有助於正確解讀結果。

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

### 3.4 CDI（圓周情境依賴指數）

```
CDI = √(-2 ln R̄)
```

CDI 量化某特徵在不同群組間的**歸因方向分散程度**：
- **CDI 高**：歸因方向隨環境情境劇烈變化（需分區管理）
- **CDI 低**：歸因方向跨群組穩定（適合全區通用指標）

---

## 4. 快速入門

只需 **3 行程式碼**即可完成完整分析：

```python
from dprofile import DProfile

# 第 1 行：初始化（輸入特徵矩陣 + 歸因矩陣）
dp = DProfile(features, shap_values, feature_names=names, target=target)

# 第 2 行：執行分析
results = dp.fit(som_grid=(9, 9), n_clusters=6)

# 第 3 行：查看結果
results.summary()
```

### 輸入資料格式

| 參數 | 型態 | 形狀 | 說明 |
|------|------|------|------|
| `features` | numpy array | (n_samples, n_features) | 原始特徵值矩陣 |
| `shap_values` | numpy array | (n_samples, n_features) | SHAP 歸因值矩陣（或任何歸因方法） |
| `feature_names` | list of str | (n_features,) | 特徵名稱（選填，預設 F1, F2...） |
| `target` | numpy array | (n_samples,) | 目標變數（選填，用於群組排序） |

### 最小可運行範例

```python
import numpy as np
from dprofile import DProfile

# 模擬資料
np.random.seed(42)
features = np.random.randn(200, 5)
shap_values = np.random.randn(200, 5)
target = np.random.rand(200) * 10  # 例如汙染濃度

# 執行
dp = DProfile(
    features,
    shap_values,
    feature_names=["氣溫", "海拔", "降雨", "坡度", "NDVI"],
    target=target,
)
results = dp.fit(som_grid=(5, 5), n_clusters=3)
results.summary()
```

---

## 5. 完整分析流程

以下示範從原始資料到完整分析的標準流程。

### 5.1 準備資料

```python
import pandas as pd
import numpy as np
from dprofile import DProfile

# 載入資料
df = pd.read_csv("your_data.csv")
shap_df = pd.read_csv("your_shap_values.csv")

# 特徵名稱
feature_names = ["Temperature", "Elevation", "Precipitation",
                 "Slope", "Forest_Cover", "Farmland_Ratio"]

# 提取矩陣
features = df[feature_names].values          # (n, p)
shap_values = shap_df[feature_names].values  # (n, p)
target = df["NO3_concentration"].values      # (n,)  選填
```

### 5.2 初始化與執行

```python
dp = DProfile(
    features=features,
    attributions=shap_values,
    feature_names=feature_names,
    target=target,
)

results = dp.fit(
    som_grid=(9, 9),        # SOM 網格大小（建議 n_samples^0.5 附近）
    n_clusters=6,           # 群組數（可透過 η² 拐點法決定）
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
  D-Profile Analysis Results
============================================================
  Samples:    2375
  Features:   17
  Groups:     6
  η² (target): 0.3510

  Group sizes:
    G1:   340 (14.3%)
    G2:   381 (16.0%)
    G3:   388 (16.3%)
    G4:   639 (26.9%)
    G5:   298 (12.5%)
    G6:   329 (13.9%)

  CDI ranking (top 5 most context-dependent):
    1. Leaching Risk         CDI=1.979
    2. Slope                 CDI=1.925
    3. Precipitation         CDI=1.907
    4. Infiltration          CDI=1.844
    5. Temperature           CDI=1.672
============================================================
```

---

## 6. 理解輸出結果

### 6.1 群組標籤

```python
# 每個樣本的群組歸屬（1-indexed）
labels = results.labels  # np.array, shape (n_samples,)

# 加入原始資料
df["DProfile_Group"] = results.labels
```

群組按目標變數降序編號：G1 為目標值最高的群組。

### 6.2 方向角與信號強度

```python
# 每個樣本在每個特徵上的方向角（弧度）
theta = results.theta  # (n_samples, n_features)

# 信號強度（距原點的距離）
r = results.r  # (n_samples, n_features)

# 群組質心方向角
group_theta = results.group_theta  # (n_groups, n_features)
```

### 6.3 CDI 表格

```python
cdi_df = results.cdi
print(cdi_df)
```

| 欄位 | 說明 |
|------|------|
| `feature` | 特徵名稱 |
| `CDI` | 圓周情境依賴指數（越高越不穩定） |
| `R_axial` | 軸線平均合成長度（越低越分散） |
| `rank` | CDI 排名 |
| `level` | 高 / 中 / 低 |

### 6.4 標準化值

```python
ZF = results.ZF  # 標準化特徵值 (n_samples, n_features)
ZS = results.ZS  # 標準化歸因值 (n_samples, n_features)
```

### 6.5 群組級分析

```python
# 查看某群組的特徵
g1_mask = results.labels == 1
g1_mean_ZF = results.ZF[g1_mask].mean(axis=0)
g1_mean_ZS = results.ZS[g1_mask].mean(axis=0)

# 哪些特徵的 Z^F 和 Z^S 異號？（非線性歸因行為）
opposite_sign = (g1_mean_ZF * g1_mean_ZS) < 0
print(f"G1 異號特徵: {np.array(feature_names)[opposite_sign]}")
```

---

## 7. CDI 情境依賴指數

CDI 是 D-Profile 的核心產出之一，提供**因地制宜 vs 一刀切**的決策依據。

### 7.1 解讀方式

```python
cdi_df = results.cdi

# 高 CDI 特徵（前 25%）→ 需分區管理
high_cdi = cdi_df[cdi_df["level"] == "high"]
print("需分區管理的特徵：")
print(high_cdi[["feature", "CDI"]].to_string(index=False))

# 低 CDI 特徵（後 25%）→ 適合全區通用指標
low_cdi = cdi_df[cdi_df["level"] == "low"]
print("適合全區通用的特徵：")
print(low_cdi[["feature", "CDI"]].to_string(index=False))
```

### 7.2 管理意涵

| CDI 等級 | 特徵行為 | 管理建議 |
|---------|---------|---------|
| **高** | 歸因方向隨環境情境劇烈變化 | 需依區域分別制定管理標準 |
| **中** | 歸因方向有中度變化 | 可針對特定區域微調 |
| **低** | 歸因方向跨區域穩定 | 適合作為全區通用監測指標 |

---

## 8. 進階設定

### 8.1 不使用 SOM（直接 Ward 聚類）

適用於小樣本（< 500）或不需要拓撲保序的情境：

```python
results = dp.fit(
    n_clusters=4,
    use_som=False,    # 跳過 SOM，直接在樣本層級 Ward 聚類
)
```

### 8.2 k 值選擇

使用 η² 拐點法找最佳群組數：

```python
from dprofile.clustering import eta_squared_scan
from dprofile.transform import full_transform

tx = full_transform(features, shap_values)
eta_results = eta_squared_scan(tx["cossin"], target, k_range=range(2, 13))

for k, eta in eta_results.items():
    print(f"k={k}: η²={eta:.4f}")
```

觀察 η² 增量開始趨緩的拐點即為建議的 k 值。

### 8.3 SOM 超參數調整

| 參數 | 預設值 | 建議範圍 | 說明 |
|------|--------|---------|------|
| `som_grid` | (9, 9) | (5,5) 至 (15,15) | 約為 √n_samples |
| `som_sigma` | 1.5 | 1.0–3.0 | 鄰域半徑，越大越平滑 |
| `som_lr` | 0.5 | 0.1–1.0 | 學習率 |
| `som_iterations` | 10000 | 5000–50000 | 迭代次數 |

### 8.4 使用非 SHAP 歸因方法

D-Profile 不限於 SHAP，任何能產生**逐樣本 × 逐特徵歸因矩陣**的方法皆可使用：

```python
# 使用 LIME 歸因值
dp = DProfile(features, lime_attributions, feature_names=names)

# 使用 Integrated Gradients
dp = DProfile(features, ig_attributions, feature_names=names)

# 使用 Permutation Importance（逐樣本版本）
dp = DProfile(features, perm_importance, feature_names=names)
```

### 8.5 存取中間結果

```python
# SOM 神經元層級資料
results.neuron_theta    # (n_neurons, n_features) 神經元方向角
results.neuron_r        # (n_neurons, n_features) 神經元信號強度
results.neuron_labels   # (n_neurons,) 神經元群組標籤
results.neuron_sizes    # (n_neurons,) 每個神經元的樣本數
```

---

## 9. 常見問題

### Q1: 我的資料適合用 D-Profile 嗎？

**適合的條件：**
- 有一個預測模型（任何類型）
- 能計算逐樣本的歸因值（SHAP、LIME 等）
- 樣本數 ≥ 100（建議 ≥ 500）
- 特徵數 ≥ 3

**不適合的情境：**
- 僅有全局特徵重要性排名（無逐樣本歸因）
- 樣本數 < 50（SOM 難以訓練）

### Q2: 群組數 k 該怎麼選？

建議使用 η² 拐點法（見 §8.2）。經驗法則：
- 小資料集（< 500 樣本）：k = 3–5
- 中資料集（500–3000 樣本）：k = 4–8
- 大資料集（> 3000 樣本）：k = 5–10

### Q3: CDI 的閾值怎麼定？

CDI 沒有絕對的閾值，本套件使用**四分位數**自動分級：
- 高：> Q75（前 25%）
- 低：< Q25（後 25%）
- 中：介於兩者之間

你也可以自行設定：
```python
cdi_df = results.cdi
my_high = cdi_df[cdi_df["CDI"] > 1.5]  # 自訂閾值
```

### Q4: D-Profile 是因果推論工具嗎？

**不是。** D-Profile 是描述性分析框架，刻畫模型歸因行為的統計規律。群組間的機制差異反映**模型學習到的統計關聯**，不能直接推論為物理因果關係。

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

---

## 10. 引用格式

如果您在研究中使用了 D-Profile，請引用：

```bibtex
@article{dprofile2026,
  title   = {D-Profile: A Unit-Circle Projection Framework for Directional
             SHAP Attribution Clustering --- A Case Study of Groundwater
             Nitrate in Taiwan},
  author  = {[Authors]},
  journal = {ISPRS Journal of Photogrammetry and Remote Sensing},
  year    = {2026},
  note    = {Under review}
}
```

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
| R̄(2θ) | √(mean(cos 2θ)² + mean(sin 2θ)²) | 軸線版（消除直徑對稱） |
| CDI | √(-2 ln R̄) | 圓周情境依賴指數 |
| η² | SS_between / SS_total | 目標變數區分力 |

---

*D-Profile — 讓 SHAP 從「哪些特徵重要」走向「模型如何因地制宜」*
