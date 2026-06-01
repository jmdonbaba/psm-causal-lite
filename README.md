# psm-causal-lite · 轻量级倾向得分匹配因果推断工具

[English](#english) | [中文](#中文)

---

## English

A lightweight **Propensity Score Matching** toolkit for causal inference. Built on `numpy, pandas, scikit-learn, scipy, matplotlib` only.

### Install

```bash
pip install psm-causal-lite
# or from source
git clone https://github.com/jmdonbaba/psm-causal-lite.git
cd psm-causal-lite
pip install -e .
```

### Quick Start

```python
from psm_causal import PSMatcher
import pandas as pd

# Data: X covariates, treatment (0/1), outcome
X = pd.DataFrame({"edu": [12, 16, 14, 18, 13], "exp": [5, 8, 3, 10, 6]})
treatment = [0, 1, 0, 1, 0]
outcome = [30, 55, 35, 60, 32]

# Six steps for complete PSM analysis
matcher = PSMatcher()
matcher.fit(X, treatment, outcome)   # Step 1: estimate propensity scores
matcher.match()                       # Step 2: nearest-neighbor matching
matcher.estimate_effect()             # Step 3: compute ATT + CI
matcher.balance_check()               # Step 4: SMD balance table
matcher.model_summary()               # Step 5: model diagnostics (AUC, etc.)
matcher.summary()                     # Step 6: print full report
matcher.plot()                        # Generate diagnostic figure
```

Example output:

```
ATT = 3.0421  [2.5834, 3.5008]

Covariate Balance (SMD):
 feature     SMD_before   SMD_after   improved   flag
 edu_years        0.5234      0.0412       True     OK
 work_exp         0.3102      0.0521       True     OK
```

### API

| Method | Description |
|--------|-------------|
| `fit(X, treatment, outcome)` | Estimate propensity scores via logistic regression |
| `match(method='nearest', k=1, with_replacement=False)` | Nearest-neighbor matching on logit(PS) scale; supports 1:k matching and caliper |
| `estimate_effect(alpha=0.05, bootstrap=False)` | Compute ATT with confidence interval; bootstrap SE available |
| `balance_check(thresholds=(0.1, 0.2))` | Return SMD before/after matching with configurable thresholds |
| `model_summary()` | Print propensity model diagnostics (AUC, sample breakdown, extreme score warnings) |
| `summary()` | Print full analysis report |
| `plot()` | Generate 4-panel diagnostic figure |

#### Key parameters

- **`caliper`** (constructor): caliper threshold in logit(PS) standard-deviation units. Recommended: 0.2 (Austin 2011). `None` disables caliper.
- **`k`**: number of controls matched to each treated unit (1:k matching). Use `with_replacement=True` if `k` exceeds available controls.
- **`bootstrap`**: bootstrap standard error and percentile CI for ATT (recommended when $k > 1$).

### Why PSM?

In observational studies, treatment and control groups are often not directly comparable:

> Studying "effect of training on income" — people with more education are more likely to get training AND already earn more. Simple mean comparison would overestimate the training effect.

PSM matches individuals with similar propensity scores, creating pseudo-experimental conditions to isolate the true causal effect.

### Workflow

```
Raw data (with selection bias)
    │
    ▼
Step 1: Estimate P(T=1 | X)    ← Logistic Regression
    │
    ▼
Step 2: Match on logit(PS)     ← Nearest Neighbors + caliper
    │
    ▼
Step 3: Compute ATT            ← mean(Y_treated - Y_control)
    │
    ▼
Step 4–6: Diagnose & Report    ← SMD balance + AUC + CI
```

### Key Metrics

- **ATT** (Average Treatment Effect on the Treated): average effect on treated units vs. what would have happened had they not been treated
- **SMD** (Standardized Mean Difference): `|SMD| < 0.1` after matching indicates good covariate balance; `< 0.2` is fair
- **AUC**: propensity model discrimination; 0.7–0.8 is good, $< 0.6$ suggests an inadequate model, $> 0.8$ may signal limited common support

### Dependencies

- Python ≥ 3.8
- numpy ≥ 1.20, pandas ≥ 1.3, scikit-learn ≥ 1.0, scipy ≥ 1.7, matplotlib ≥ 3.4

### License

MIT

---

## 中文

轻量级**倾向得分匹配** (Propensity Score Matching) 因果推断工具。基于 `numpy, pandas, scikit-learn, scipy, matplotlib`。

### 安装

```bash
pip install psm-causal-lite
# 或从源码安装
git clone https://github.com/jmdonbaba/psm-causal-lite.git
cd psm-causal-lite
pip install -e .
```

### 快速开始

```python
from psm_causal import PSMatcher
import pandas as pd

# 准备数据: X 协变量, treatment 处理变量(0/1), outcome 结果变量
X = pd.DataFrame({"edu": [12, 16, 14, 18, 13], "exp": [5, 8, 3, 10, 6]})
treatment = [0, 1, 0, 1, 0]
outcome = [30, 55, 35, 60, 32]

# 六步完成完整 PSM 分析
matcher = PSMatcher()
matcher.fit(X, treatment, outcome)   # 步骤1: 估计倾向得分
matcher.match()                       # 步骤2: 最近邻匹配
matcher.estimate_effect()             # 步骤3: 计算 ATT 及置信区间
matcher.balance_check()               # 步骤4: SMD 平衡性检验
matcher.model_summary()               # 步骤5: 模型诊断 (AUC 等)
matcher.summary()                     # 步骤6: 打印完整报告
matcher.plot()                        # 生成诊断图
```

示例输出：

```
ATT = 3.0421  [2.5834, 3.5008]

协变量平衡性 (SMD):
 feature     SMD_before   SMD_after   improved   flag
 edu_years        0.5234      0.0412       True     OK
 work_exp         0.3102      0.0521       True     OK
```

### API 概览

| 方法 | 说明 |
|------|------|
| `fit(X, treatment, outcome)` | 用 logistic 回归估计倾向得分 P(T=1\|X) |
| `match(method='nearest', k=1, with_replacement=False)` | logit(PS) 尺度最近邻匹配，支持 1:k 匹配和 caliper 阈值 |
| `estimate_effect(alpha=0.05, bootstrap=False)` | 计算 ATT 及置信区间，支持 bootstrap 标准误 |
| `balance_check(thresholds=(0.1, 0.2))` | 返回匹配前后 SMD 对比表，阈值可配 |
| `model_summary()` | 打印倾向模型诊断 (AUC、样本分布、极端得分告警) |
| `summary()` | 打印完整分析报告 |
| `plot()` | 生成 4 面板诊断图表 |

#### 关键参数

- **`caliper`**（构造函数）：caliper 阈值，以 logit(PS) 标准差倍数表示。推荐 0.2（Austin 2011）。`None` 表示不使用 caliper。
- **`k`**：每个处理单元匹配的对照单元数量（1:k 匹配）。若 k 超过可用对照数量，可启用 `with_replacement=True`。
- **`bootstrap`**：bootstrap 法估计标准误和百分位置信区间（$k > 1$ 时推荐启用）。

### 为什么需要 PSM？

观察性研究中，处理组和对照组通常不可直接比较：

> 研究"培训对收入的影响"——受教育高的人更可能参加培训，而他们本身就收入更高。简单比较两组均值会高估培训效应。

PSM 通过匹配倾向得分相近的个体，构造出"准随机实验"的条件，从而分离出真正的因果效应。

### 工作流

```
原始数据 (有选择偏差)
    │
    ▼
步骤1: 估计倾向得分 P(T=1 | X)  ← Logistic 回归
    │
    ▼
步骤2: logit(PS) 尺度匹配       ← 最近邻搜索 + caliper
    │
    ▼
步骤3: 计算 ATT                ← mean(Y_处理组 - Y_对照组)
    │
    ▼
步骤4–6: 诊断 & 报告            ← SMD 平衡 + AUC + 置信区间
```

### 评估指标

- **ATT** (Average Treatment Effect on the Treated): 处理组相对于"如果他们未接受处理"的平均因果效应
- **SMD** (Standardized Mean Difference): 匹配后 `|SMD| < 0.1` 表示协变量平衡良好，`< 0.2` 可接受
- **AUC**: 倾向模型区分能力；0.7–0.8 良好，$< 0.6$ 模型不足，$> 0.8$ 可能存在 common support 问题

### 环境依赖

- Python ≥ 3.8
- numpy ≥ 1.20, pandas ≥ 1.3, scikit-learn ≥ 1.0, scipy ≥ 1.7, matplotlib ≥ 3.4

### 开源协议

MIT
