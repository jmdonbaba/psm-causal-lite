# psm-causal-lite · 轻量级倾向得分匹配因果推断工具

[English](#english) | [中文](#中文)

---

## English

A lightweight **Propensity Score Matching** toolkit for causal inference. Zero extra dependencies — `numpy, pandas, scikit-learn, matplotlib` only.

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
matcher.summary()                     # Step 5: print full report
matcher.plot()                        # Step 6: diagnostic figure
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
| `match(method='nearest', k=1)` | Nearest-neighbor matching on propensity scores, supports caliper |
| `estimate_effect(alpha=0.05)` | Compute ATT with 95% confidence interval |
| `balance_check()` | Return SMD (Standardized Mean Difference) before/after matching |
| `summary()` | Print full analysis report |
| `plot()` | Generate 4-panel diagnostic figure |

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
Step 2: Match on propensity    ← Nearest Neighbors
    │
    ▼
Step 3: Compute ATT            ← mean(Y_treated - Y_control)
    │
    ▼
Step 4 & 5: Diagnose & Report   ← SMD balance check
```

### Key Metrics

- **ATT** (Average Treatment Effect on the Treated): average effect on treated units vs. what would have happened had they not been treated
- **SMD** (Standardized Mean Difference): `|SMD| < 0.1` after matching indicates good covariate balance

### Dependencies

- Python ≥ 3.8
- numpy ≥ 1.20, pandas ≥ 1.3, scikit-learn ≥ 1.0, matplotlib ≥ 3.4

### License

MIT

---

## 中文

轻量级**倾向得分匹配** (Propensity Score Matching) 因果推断工具。零额外依赖 — 仅需 `numpy, pandas, scikit-learn, matplotlib`。

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
matcher.summary()                     # 步骤5: 打印完整报告
matcher.plot()                        # 步骤6: 生成诊断图
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
| `match(method='nearest', k=1)` | 倾向得分最近邻匹配，支持 caliper 阈值 |
| `estimate_effect(alpha=0.05)` | 计算 ATT 及 95% 置信区间 |
| `balance_check()` | 返回匹配前后 SMD 对比表 |
| `summary()` | 打印完整分析报告 |
| `plot()` | 生成 4 面板诊断图表 |

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
步骤2: 倾向得分匹配             ← 最近邻搜索
    │
    ▼
步骤3: 计算 ATT                ← mean(Y_处理组 - Y_对照组)
    │
    ▼
步骤4 & 5: 诊断 & 报告          ← SMD 平衡性检验
```

### 评估指标

- **ATT** (Average Treatment Effect on the Treated): 处理组相对于"如果他们未接受处理"的平均因果效应
- **SMD** (Standardized Mean Difference): 匹配后 `|SMD| < 0.1` 表示协变量平衡良好，`< 0.2` 可接受

### 环境依赖

- Python ≥ 3.8
- numpy ≥ 1.20, pandas ≥ 1.3, scikit-learn ≥ 1.0, matplotlib ≥ 3.4

### 开源协议

MIT
