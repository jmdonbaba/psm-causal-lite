# psm-causal-lite

轻量级倾向得分匹配 (Propensity Score Matching) 因果推断工具。

**零额外依赖** — 仅需 `numpy`, `pandas`, `scikit-learn`, `matplotlib`，pip install 即用。

## 安装

```bash
pip install psm-causal-lite
```

或直接从源码安装：

```bash
git clone https://github.com/YOUR_USERNAME/psm-causal-lite.git
cd psm-causal-lite
pip install -e .
```

## 快速开始

```python
from psm_causal import PSMatcher
import pandas as pd

# 数据: X 协变量, treatment 处理变量(0/1), outcome 结果变量
X = pd.DataFrame({"edu": [12, 16, 14, 18, 13], "exp": [5, 8, 3, 10, 6]})
treatment = [0, 1, 0, 1, 0]
outcome = [30, 55, 35, 60, 32]

# 四步完成 PSM 分析
matcher = PSMatcher()
matcher.fit(X, treatment, outcome)   # Step 1: 估计倾向得分
matcher.match()                       # Step 2: 最近邻匹配
matcher.estimate_effect()             # Step 3: 计算 ATT
matcher.summary()                     # Step 4: 打印报告
matcher.plot()                        # Step 5: 诊断图
```

输出示例：

```
ATT = 3.0421  [2.5834, 3.5008]

协变量平衡性 (SMD):
 feature   SMD_before   SMD_after   improved   flag
   edu_years     0.5234      0.0412       True      ✓
   work_exp      0.3102      0.0521       True      ✓
```

## API 概览

| 方法 | 说明 |
|------|------|
| `fit(X, treatment, outcome)` | 用 logistic 回归估计倾向得分 P(T=1\|X) |
| `match(method='nearest', k=1)` | 倾向得分最近邻匹配，支持 caliper |
| `estimate_effect(alpha=0.05)` | 计算 ATT 及 95% 置信区间 |
| `balance_check()` | 返回匹配前后 SMD 对比表 |
| `summary()` | 打印完整分析报告 |
| `plot()` | 生成 4 面板诊断图 |

## 方法原理

### 为什么需要 PSM？

观察性研究中，处理组和对照组通常不可比。比如：

> 研究"培训对收入的影响"——受教育高的人更可能参加培训，而他们本身就收入更高。简单比较两组均值会高估培训效应。

PSM 通过匹配倾向得分相近的个体，构造出"准随机实验"的条件，从而分离出真正的因果效应。

### 工作流

```
原始数据 (有选择偏差)
    │
    ▼
Step 1: 估计倾向得分 P(T=1 | X)  ← Logistic Regression
    │
    ▼
Step 2: 倾向得分匹配             ← Nearest Neighbors
    │
    ▼
Step 3: 计算 ATT                 ← mean(Y_treated - Y_matched_control)
    │
    ▼
Step 4 & 5: 诊断 & 报告          ← SMD 平衡性检验
```

### 评估指标

- **ATT** (Average Treatment Effect on the Treated): 处理组相对于"如果他们未接受处理"的平均效应
- **SMD** (Standardized Mean Difference): 匹配后 `|SMD| < 0.1` 表示协变量平衡良好

## 在你的项目中使用

```python
# 场景: 评估某项政策/干预的因果效应
matcher = PSMatcher(caliper=0.2)  # caliper: 倾向得分标准差倍数
matcher.fit(X, treatment, outcome)
matcher.match(method="nearest", k=1, with_replacement=False)
matcher.estimate_effect(alpha=0.05)
matcher.summary()

# 获取数值结果
print(f"ATT: {matcher.att_:.4f}")
print(f"95% CI: {matcher.att_ci_}")
balance_df = matcher.balance_check()
```

## 依赖

- Python ≥ 3.8
- numpy ≥ 1.20
- pandas ≥ 1.3
- scikit-learn ≥ 1.0
- matplotlib ≥ 3.4

## License

MIT
