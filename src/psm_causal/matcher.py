"""
PSMatcher — 一步式倾向得分匹配 (Propensity Score Matching)

工作流
------
1. fit(X, treatment, outcome)     → 估计倾向得分
2. match(method, k, caliper)      → 匹配处理组/对照组
3. estimate_effect()              → 计算 ATT
4. balance_check()                → 协变量平衡性诊断
5. summary() / plot()             → 报告与可视化
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .viz import plot_psm_report


class PSMatcher:
    """倾向得分匹配器

    Parameters
    ----------
    random_state : int, default=42
        随机种子，保证结果可复现
    caliper : float or None, default=None
        caliper 阈值 (倾向得分标准差倍数)。None 表示不使用 caliper
    """

    def __init__(self, random_state=42, caliper=None):
        self.random_state = random_state
        self.caliper = caliper

        # Internal state (set during fit/match/estimate)
        self.X_ = None
        self.feature_names_ = None
        self.treatment_ = None
        self.outcome_ = None
        self.propensity_scores_ = None
        self.scaler_ = None
        self.ps_model_ = None

        self.matched_treated_ = None
        self.matched_control_ = None
        self.is_matched_ = None

        self.att_ = None
        self.att_se_ = None
        self.att_ci_ = None
        self._is_fitted = False
        self._is_matched = False

    # ================================================================
    # Step 1: 估计倾向得分
    # ================================================================
    def fit(self, X, treatment, outcome):
        """估计倾向得分 P(T=1 | X)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            协变量矩阵 (未标准化也可，内部自动标准化)
        treatment : array-like of shape (n_samples,)
            二值处理变量 (1 = 处理组, 0 = 对照组)
        outcome : array-like of shape (n_samples,)
            结果变量 (连续值)

        Returns
        -------
        self : PSMatcher
        """
        # 在转换前提取特征名
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        elif hasattr(X, "name"):
            self.feature_names_ = [getattr(X, "name", "X0")]
        else:
            self.feature_names_ = None

        X = self._to_array(X)
        treatment = np.asarray(treatment, dtype=int)
        outcome = np.asarray(outcome, dtype=float)

        if self.feature_names_ is None:
            self.feature_names_ = [f"X{i}" for i in range(X.shape[1])]

        self._validate_inputs(X, treatment, outcome)

        self.X_ = X
        self.treatment_ = treatment
        self.outcome_ = outcome

        # 标准化
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # 逻辑回归估计倾向得分
        self.ps_model_ = LogisticRegression(
            max_iter=1000,
            random_state=self.random_state,
        )
        self.ps_model_.fit(X_scaled, treatment)
        self.propensity_scores_ = self.ps_model_.predict_proba(X_scaled)[:, 1]

        self._is_fitted = True
        self._is_matched = False
        return self

    # ================================================================
    # Step 2: 匹配
    # ================================================================
    def match(self, method="nearest", k=1, with_replacement=False):
        """执行倾向得分匹配

        Parameters
        ----------
        method : str, default="nearest"
            匹配方法。目前仅支持 "nearest" (最近邻匹配)
        k : int, default=1
            每个处理单元匹配的对照单元数量
        with_replacement : bool, default=False
            是否允许重复匹配 (有放回)

        Returns
        -------
        self : PSMatcher
        """
        self._check_fitted()

        treated_idx = np.where(self.treatment_ == 1)[0]
        control_idx = np.where(self.treatment_ == 0)[0]

        if len(treated_idx) == 0:
            raise ValueError("No treatment units (treatment all 0)")
        if len(control_idx) == 0:
            raise ValueError("No control units (treatment all 1)")

        ps = self.propensity_scores_

        self.matched_treated_ = []
        self.matched_control_ = []
        self.is_matched_ = np.zeros(len(self.treatment_), dtype=bool)

        if method == "nearest":
            self._match_nearest(treated_idx, control_idx, ps, k, with_replacement)
        else:
            raise ValueError(f"Unsupported matching method: {method}")

        self._is_matched = True
        matched_n = np.sum(self.is_matched_)
        print(
            f"Matched: {matched_n} / {len(treated_idx)} treated units matched"
            + (f" (caliper 过滤了 {len(treated_idx) - matched_n} 个)" if self.caliper else "")
        )
        return self

    def _match_nearest(self, treated_idx, control_idx, ps, k, with_replacement):
        ps_treated = ps[treated_idx].reshape(-1, 1)
        ps_control = ps[control_idx].reshape(-1, 1)

        nbrs = NearestNeighbors(n_neighbors=min(k, len(control_idx)), algorithm="ball_tree")
        nbrs.fit(ps_control)
        distances, indices = nbrs.kneighbors(ps_treated)

        caliper_val = self.caliper * np.std(ps) if self.caliper else None
        used_control = set() if not with_replacement else None

        for i, t_idx in enumerate(treated_idx):
            for j in range(k):
                c_local = indices[i, j]
                dist = distances[i, j]
                c_idx = control_idx[c_local]

                if caliper_val is not None and dist > caliper_val:
                    continue
                if not with_replacement and c_idx in used_control:
                    continue

                self.matched_treated_.append(t_idx)
                self.matched_control_.append(c_idx)
                self.is_matched_[t_idx] = True
                self.is_matched_[c_idx] = True
                if not with_replacement:
                    used_control.add(c_idx)
                break

        if len(self.matched_treated_) == 0:
            raise RuntimeError(
                "No matches found. Try increasing caliper or use with_replacement=True"
            )

    # ================================================================
    # Step 3: 估计处理效应
    # ================================================================
    def estimate_effect(self, alpha=0.05):
        """计算平均处理效应 (ATT = Average Treatment Effect on the Treated)

        ATT = E[Y(1) - Y(0) | T=1]
            = mean(outcome_treated - outcome_matched_control)

        Parameters
        ----------
        alpha : float, default=0.05
            显著性水平，用于构建 (1-alpha) 置信区间

        Returns
        -------
        self : PSMatcher
        """
        self._check_matched()

        y_t = self.outcome_[self.matched_treated_]
        y_c = self.outcome_[self.matched_control_]
        diffs = y_t - y_c

        n = len(diffs)
        self.att_ = np.mean(diffs)
        self.att_se_ = np.std(diffs, ddof=1) / np.sqrt(n)
        z = 1 - alpha / 2
        # 使用正态近似
        from scipy.stats import norm
        ci_lo = self.att_ - norm.ppf(z) * self.att_se_
        ci_hi = self.att_ + norm.ppf(z) * self.att_se_
        self.att_ci_ = (ci_lo, ci_hi)

        print(f"ATT = {self.att_:.4f}  [{ci_lo:.4f}, {ci_hi:.4f}]")
        return self

    # ================================================================
    # Step 4: 平衡性诊断
    # ================================================================
    def balance_check(self):
        """协变量平衡性检查 — 匹配前后的标准化均值差异 (SMD)

        SMD = (M_t - M_c) / sqrt((S_t^2 + S_c^2) / 2)

        Returns
        -------
        balance_df : pd.DataFrame
            每列: feature, SMD_before, SMD_after, improved
        """
        self._check_matched()

        names = self.feature_names_ or [f"X{i}" for i in range(self.X_.shape[1])]
        rows = []

        t_all = self.treatment_ == 1
        c_all = self.treatment_ == 0
        t_m = self.is_matched_ & (self.treatment_ == 1)
        c_m = np.zeros(len(self.treatment_), dtype=bool)
        c_m[self.matched_control_] = True

        for j, name in enumerate(names):
            x = self.X_[:, j]
            smd_before = self._smd(x[t_all], x[c_all])
            smd_after = self._smd(x[t_m], x[c_m])
            rows.append({
                "feature": name,
                "SMD_before": round(smd_before, 4),
                "SMD_after": round(smd_after, 4),
                "improved": abs(smd_after) < abs(smd_before),
            })

        balance_df = pd.DataFrame(rows)
        balance_df["flag"] = balance_df["SMD_after"].apply(
            lambda s: "OK" if abs(s) < 0.1 else ("~" if abs(s) < 0.2 else "!")
        )
        return balance_df

    @staticmethod
    def _smd(x1, x2):
        """标准化均值差异 (Standardized Mean Difference)"""
        v1, v2 = np.var(x1, ddof=0), np.var(x2, ddof=0)
        denom = np.sqrt((v1 + v2) / 2)
        return (np.mean(x1) - np.mean(x2)) / denom if denom > 1e-10 else 0.0

    # ================================================================
    # Step 5: 综合报告
    # ================================================================
    def summary(self):
        """打印完整的 PSM Analysis Report"""
        self._check_matched()

        n_total = len(self.treatment_)
        n_treated = np.sum(self.treatment_ == 1)
        n_control = np.sum(self.treatment_ == 0)
        n_matched = np.sum(self.is_matched_)

        print("=" * 60)
        print("PSM Analysis Report")
        print("=" * 60)
        print(f"  Total samples:        {n_total}")
        print(f"  Treatment (T=1):    {n_treated}")
        print(f"  Control (T=0):    {n_control}")
        print(f"  Matched:        {n_matched}")
        print()

        if self.att_ is not None:
            print(f"  ATT: {self.att_:.4f}")
            if self.att_ci_ is not None:
                print(f"  CI (95%):     [{self.att_ci_[0]:.4f}, {self.att_ci_[1]:.4f}]")
                print(f"  SE:             {self.att_se_:.4f}")
            print()

        balance = self.balance_check()
        print("Covariate Balance (SMD):")
        print(balance.to_string(index=False))
        print()
        print("  SMD < 0.1: good (OK)")
        print("  SMD < 0.2: fair (~)")
        print("  SMD >= 0.2: poor (!)")
        print("=" * 60)

    # ================================================================
    # Step 6: 可视化
    # ================================================================
    def plot(self, figsize=(12, 8)):
        """生成 PSM 诊断图

        包含:
        1. 倾向得分分布 (匹配前)
        2. 协变量平衡性对比 (SMD before vs after)
        3. 处理效应 (ATT) 图示
        """
        self._check_matched()
        return plot_psm_report(
            propensity_scores=self.propensity_scores_,
            treatment=self.treatment_,
            outcome=self.outcome_,
            matched_treated=self.matched_treated_,
            matched_control=self.matched_control_,
            balance_df=self.balance_check(),
            att=self.att_,
            att_ci=self.att_ci_,
            X=self.X_,
            feature_names=self.feature_names_,
            figsize=figsize,
        )

    # ================================================================
    # 内部辅助方法
    # ================================================================
    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError("Call .fit() first")

    def _check_matched(self):
        self._check_fitted()
        if not self._is_matched:
            raise RuntimeError("Call .match() first")

    @staticmethod
    def _to_array(x):
        if hasattr(x, "values"):
            x = x.values
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    @staticmethod
    def _validate_inputs(X, treatment, outcome):
        n = len(X)
        if len(treatment) != n:
            raise ValueError(
                f"treatment length ({len(treatment)}) != X rows ({n})"
            )
        if len(outcome) != n:
            raise ValueError(
                f"outcome length ({len(outcome)}) != X rows ({n})"
            )
        unique_t = np.unique(treatment)
        if not set(unique_t).issubset({0, 1}):
            raise ValueError(f"treatment must be binary (0/1), got: {unique_t}")
        if len(unique_t) < 2:
            raise ValueError(
                f"treatment must contain both 0 and 1, got: {unique_t}"
            )
