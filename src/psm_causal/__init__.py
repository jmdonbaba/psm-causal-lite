"""
psm-causal-lite: 轻量级倾向得分匹配 (Propensity Score Matching) 因果推断工具

核心 API
--------
PSMatcher    — 一步式 PSM 分析器 (拟合 → 匹配 → 估计 → 诊断)

用法
----
from psm_causal import PSMatcher

matcher = PSMatcher()
matcher.fit(X, treatment, outcome)
matcher.match()
matcher.estimate_effect()
matcher.balance_check()
matcher.summary()
matcher.plot()
"""

from .matcher import PSMatcher

__version__ = "0.2.0"
__all__ = ["PSMatcher"]
