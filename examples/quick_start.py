"""
PSM Quick Start Example — Synthetic data with known treatment effect

Scenario: Estimate causal effect of "training" on "income"
- Covariates: edu_years, work_exp
- Treatment: trained (0/1)
- Outcome: income

True ATT = 3.0 (known from data generation)
"""

import numpy as np
import pandas as pd
import sys
import os
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Use non-interactive backend to avoid blocking on plt.show()
import matplotlib
matplotlib.use("Agg")

from psm_causal import PSMatcher

# ================================================================
# 1. Generate synthetic data
# ================================================================
print("=" * 60)
print("1. Generating synthetic data (true ATT = 3.0)")
print("=" * 60)

np.random.seed(42)
n = 500

edu_years = np.random.normal(14, 3, n).clip(8, 22)
work_exp = np.random.normal(8, 5, n).clip(0, 35)

# Selection bias: people with more education are more likely to get training
p_trained = 1 / (1 + np.exp(-(0.3 * edu_years + 0.1 * work_exp - 4.5)))
trained = np.random.binomial(1, p_trained)

# Outcome: income = baseline + training_effect(3) + edu + exp + noise
income = (
    5.0
    + 3.0 * trained
    + 1.5 * edu_years
    + 0.3 * work_exp
    + np.random.normal(0, 2, n)
)

X = pd.DataFrame({"edu_years": edu_years, "work_exp": work_exp})

print(f"Samples: {n}")
print(f"Treatment (trained=1): {trained.sum()} ({trained.mean()*100:.1f}%)")
print(f"Control (trained=0): {(1-trained).sum()} ({(1-trained).mean()*100:.1f}%)")
print()

# ================================================================
# 2. Naive comparison (biased)
# ================================================================
print("=" * 60)
print("2. Naive comparison (ignores selection bias)")
print("=" * 60)

naive_effect = income[trained == 1].mean() - income[trained == 0].mean()
print(f"Naive estimate (T_mean - C_mean): {naive_effect:.4f}")
print(f"True ATT: 3.0000")
print(f"Bias: {naive_effect - 3.0:.4f}  <- education confounds the effect")
print()

# ================================================================
# 3. PSM Analysis
# ================================================================
print("=" * 60)
print("3. PSM Analysis")
print("=" * 60)

matcher = PSMatcher(random_state=42)
matcher.fit(X, trained, income)
matcher.match()
matcher.estimate_effect()

print(f"\nPSM ATT: {matcher.att_:.4f} (true: 3.0000)")
print(f"PSM bias: {abs(matcher.att_ - 3.0):.4f}")
print(f"vs naive bias: {abs(naive_effect - 3.0):.4f}")
print()

# ================================================================
# 4. Full report
# ================================================================
matcher.summary()

# ================================================================
# 5. Visualization
# ================================================================
fig = matcher.plot()
fig.savefig("psm_report.png", dpi=150, bbox_inches="tight")
print("\nReport figure saved to psm_report.png")
