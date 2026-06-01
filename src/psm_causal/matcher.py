"""
PSMatcher — 一步式倾向得分匹配 (Propensity Score Matching)

工作流
------
1. fit(X, treatment, outcome)     → 估计倾向得分
2. match(method, k, caliper)      → 匹配处理组/对照组
3. estimate_effect()              → 计算 ATT
4. balance_check()                → 协变量平衡性诊断
5. summary() / plot()             → 报告与可视化
6. model_summary()                → 倾向模型诊断
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm

from .viz import plot_psm_report


class PSMatcher:
    """倾向得分匹配器

    Parameters
    ----------
    random_state : int, default=42
        随机种子，保证结果可复现
    caliper : float or None, default=None
        caliper 阈值（logit倾向得分标准差倍数）。None 表示不使用 caliper。
        推荐值 0.2（Austin 2011），在 logit(PS) 尺度上施加。
    """

    def __init__(self, random_state=42, caliper=None):
        self.random_state = random_state
        self.caliper = caliper

        # Internal state
        self.X_ = None
        self.feature_names_ = None
        self.treatment_ = None
        self.outcome_ = None
        self.propensity_scores_ = None
        self.logit_ps_ = None
        self.scaler_ = None
        self.ps_model_ = None
        self.model_info_ = None

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
            协变量矩阵（未标准化也可，内部自动标准化）
        treatment : array-like of shape (n_samples,)
            二值处理变量（1 = 处理组，0 = 对照组）
        outcome : array-like of shape (n_samples,)
            结果变量（连续值）

        Returns
        -------
        self : PSMatcher
        """
        # 在转换前提取特征名
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
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

        # 计算 logit(PS)，用于 caliper 和匹配距离
        eps = 1e-10
        ps_clipped = np.clip(self.propensity_scores_, eps, 1 - eps)
        self.logit_ps_ = np.log(ps_clipped / (1 - ps_clipped))

        # 记录倾向模型诊断信息
        self.model_info_ = {
            "auc": float(roc_auc_score(treatment, self.propensity_scores_)),
            "n_features": X.shape[1],
            "n_samples": X.shape[0],
            "n_treated": int(np.sum(treatment == 1)),
            "n_control": int(np.sum(treatment == 0)),
        }

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
            匹配方法。目前仅支持 "nearest"（最近邻匹配）
        k : int, default=1
            每个处理单元匹配的对照单元数量（1:k 匹配）
        with_replacement : bool, default=False
            是否允许重复匹配（有放回）。设为 True 时同一对照单元可被多次匹配。

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

        if not with_replacement and k > len(control_idx):
            raise ValueError(
                f"k ({k}) exceeds available control units ({len(control_idx)}). "
                "Use with_replacement=True or reduce k."
            )

        # 匹配前检查 common support
        self._check_common_support(treated_idx, control_idx)

        self.matched_treated_ = []
        self.matched_control_ = []
        self.is_matched_ = np.zeros(len(self.treatment_), dtype=bool)

        if method == "nearest":
            self._match_nearest(treated_idx, control_idx, k, with_replacement)
        else:
            raise ValueError(f"Unsupported matching method: {method}")

        self._is_matched = True
        n_unique_treated = len(set(self.matched_treated_))

        msg = f"Matched: {n_unique_treated}/{len(treated_idx)} treated units"
        if n_unique_treated < len(treated_idx):
            if self.caliper:
                msg += " (some dropped by caliper)"
            else:
                msg += " (insufficient unique controls; try with_replacement=True)"
        if k > 1:
            n_total_pairs = len(self.matched_control_)
            msg += f" × ≤{k} controls each = {n_total_pairs} total pairs"
        print(msg)

        # 超过一半的处理单元未能匹配时告警
        if 0 < n_unique_treated < len(treated_idx) * 0.5:
            dropped = len(treated_idx) - n_unique_treated
            suggestion = (
                "Consider increasing caliper or using with_replacement=True."
                if self.caliper
                else "Consider using with_replacement=True."
            )
            warnings.warn(
                f"More than 50% of treated units ({dropped}) could not be matched. "
                + suggestion,
                UserWarning,
            )

        return self

    def _match_nearest(self, treated_idx, control_idx, k, with_replacement):
        """在 logit(PS) 尺度上执行最近邻匹配"""
        logit_treated = self.logit_ps_[treated_idx].reshape(-1, 1)
        logit_control = self.logit_ps_[control_idx].reshape(-1, 1)

        n_neighbors = min(k, len(control_idx))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="ball_tree")
        nbrs.fit(logit_control)
        distances, indices = nbrs.kneighbors(logit_treated)

        caliper_val = (
            self.caliper * np.std(self.logit_ps_) if self.caliper else None
        )
        used_control = set() if not with_replacement else None

        for i, t_idx in enumerate(treated_idx):
            n_matched = 0
            for j in range(n_neighbors):
                if n_matched >= k:
                    break
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
                n_matched += 1

        if len(self.matched_treated_) == 0:
            raise RuntimeError(
                "No matches found. Try increasing caliper or use with_replacement=True"
            )

        # 检查是否有处理单元匹配数不足 k
        match_counts = {}
        for t_idx in self.matched_treated_:
            match_counts[t_idx] = match_counts.get(t_idx, 0) + 1
        partial = sum(1 for v in match_counts.values() if v < k)
        if partial > 0:
            warnings.warn(
                f"{partial} treated unit(s) got fewer than k={k} match(es). "
                "Consider increasing caliper or using with_replacement=True.",
                UserWarning,
            )

    def _check_common_support(self, treated_idx, control_idx):
        """检查处理组与对照组倾向得分的重叠区域 (common support)"""
        ps_t = self.propensity_scores_[treated_idx]
        ps_c = self.propensity_scores_[control_idx]

        overlap_min = max(ps_t.min(), ps_c.min())
        overlap_max = min(ps_t.max(), ps_c.max())

        treated_outside = int(np.sum((ps_t < overlap_min) | (ps_t > overlap_max)))
        control_outside = int(np.sum((ps_c < overlap_min) | (ps_c > overlap_max)))

        if treated_outside > 0 or control_outside > 0:
            warnings.warn(
                f"Common support violation: {treated_outside} treated + "
                f"{control_outside} control unit(s) outside overlapping PS range "
                f"[{overlap_min:.4f}, {overlap_max:.4f}]. "
                "Estimates may be unreliable for units without comparable counterparts.",
                UserWarning,
            )

    # ================================================================
    # Step 3: 估计处理效应
    # ================================================================
    def estimate_effect(self, alpha=0.05, bootstrap=False, n_bootstrap=1000):
        """计算平均处理效应（ATT = Average Treatment Effect on the Treated）

        ATT = E[Y(1) - Y(0) | T=1]

        对 1:k 匹配，先将每处理单元匹配到的 k 个对照结果取均值，
        再计算处理组与对照组均值之差。

        Parameters
        ----------
        alpha : float, default=0.05
            显著性水平，用于构建 (1-alpha) 置信区间
        bootstrap : bool, default=False
            是否使用 bootstrap 估计标准误和置信区间。
            bootstrap 可更好地反映匹配后数据的变异性，推荐在 k>1 时启用。
        n_bootstrap : int, default=1000
            Bootstrap 重抽样次数

        Returns
        -------
        self : PSMatcher
        """
        self._check_matched()

        diffs = self._compute_matched_diffs()
        n = len(diffs)

        self.att_ = float(np.mean(diffs))

        if bootstrap:
            rng = np.random.RandomState(self.random_state)
            atts = np.zeros(n_bootstrap)
            for b in range(n_bootstrap):
                boot_idx = rng.choice(n, size=n, replace=True)
                atts[b] = np.mean(diffs[boot_idx])
            self.att_se_ = float(np.std(atts, ddof=1))
            self.att_ci_ = (
                float(np.percentile(atts, 100 * alpha / 2)),
                float(np.percentile(atts, 100 * (1 - alpha / 2))),
            )
        else:
            self.att_se_ = float(np.std(diffs, ddof=1) / np.sqrt(n))
            z = 1 - alpha / 2
            ci_lo = self.att_ - norm.ppf(z) * self.att_se_
            ci_hi = self.att_ + norm.ppf(z) * self.att_se_
            self.att_ci_ = (float(ci_lo), float(ci_hi))

        print(f"ATT = {self.att_:.4f}  [{self.att_ci_[0]:.4f}, {self.att_ci_[1]:.4f}]")
        return self

    def _compute_matched_diffs(self):
        """计算每处理单元的匹配结果差异（处理 k:1 匹配）"""
        treated_arr = np.array(self.matched_treated_)
        control_arr = np.array(self.matched_control_)

        unique_treated = np.unique(treated_arr)
        diffs = np.zeros(len(unique_treated))

        for idx, t_idx in enumerate(unique_treated):
            y_t = self.outcome_[t_idx]
            c_indices = control_arr[treated_arr == t_idx]
            y_c_mean = np.mean(self.outcome_[c_indices])
            diffs[idx] = y_t - y_c_mean

        return diffs

    # ================================================================
    # Step 4: 平衡性诊断
    # ================================================================
    def balance_check(self, thresholds=(0.1, 0.2)):
        """协变量平衡性检查 —— 匹配前后的标准化均值差异 (SMD)

        SMD = (M_t - M_c) / sqrt((S_t^2 + S_c^2) / 2)

        Parameters
        ----------
        thresholds : tuple (good, fair), default=(0.1, 0.2)
            |SMD| < good → "OK", good ≤ |SMD| < fair → "~", |SMD| ≥ fair → "!"

        Returns
        -------
        balance_df : pd.DataFrame
            每列: feature, SMD_before, SMD_after, improved, flag
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
        good_t, fair_t = thresholds
        balance_df["flag"] = balance_df["SMD_after"].apply(
            lambda s: "OK" if abs(s) < good_t else ("~" if abs(s) < fair_t else "!")
        )

        n_poor = int((balance_df["flag"] == "!").sum())
        if n_poor > 0:
            bad = balance_df[balance_df["flag"] == "!"]["feature"].tolist()
            warnings.warn(
                f"{n_poor} covariate(s) with poor balance after matching: {bad}. "
                "Consider a richer propensity model or different matching strategy.",
                UserWarning,
            )

        return balance_df

    @staticmethod
    def _smd(x1, x2):
        """标准化均值差异 (Standardized Mean Difference)"""
        v1, v2 = np.var(x1, ddof=0), np.var(x2, ddof=0)
        denom = np.sqrt((v1 + v2) / 2)
        return (np.mean(x1) - np.mean(x2)) / denom if denom > 1e-10 else 0.0

    # ================================================================
    # 倾向模型诊断
    # ================================================================
    def model_summary(self):
        """打印倾向得分模型的诊断信息

        包括 AUC、样本量分布、极端倾向得分等，帮助判断倾向模型是否充分。
        """
        self._check_fitted()

        info = self.model_info_
        print("=" * 50)
        print("Propensity Score Model Diagnostics")
        print("=" * 50)
        print(f"  AUC (discrimination):  {info['auc']:.4f}")
        print(f"  N samples:             {info['n_samples']}")
        print(f"  N treated (T=1):       {info['n_treated']}")
        print(f"  N control (T=0):       {info['n_control']}")
        print(f"  N covariates:          {info['n_features']}")
        print()

        auc = info["auc"]
        if auc < 0.6:
            print("  [!] AUC < 0.6 — propensity model has poor discrimination.")
            print("      Causal estimates may be unreliable.")
        elif auc < 0.7:
            print("  [~] AUC 0.6–0.7 — fair discrimination.")
        elif auc < 0.8:
            print("  [OK] AUC 0.7–0.8 — good discrimination.")
        else:
            print("  [OK] AUC ≥ 0.8 — strong discrimination.")
            print("      Very high AUC may signal limited common support.")

        ps = self.propensity_scores_
        extreme_cutoff = 0.01
        n_extreme = int(
            np.sum((ps < extreme_cutoff) | (ps > 1 - extreme_cutoff))
        )
        if n_extreme > 0:
            print(
                f"\n  [!] {n_extreme} unit(s) with extreme propensity scores "
                f"(< {extreme_cutoff} or > {1 - extreme_cutoff})."
            )
            print("      These units may lack comparable counterparts.")

        print("=" * 50)

    # ================================================================
    # Step 5: 综合报告
    # ================================================================
    def summary(self):
        """打印完整的 PSM Analysis Report"""
        self._check_matched()

        n_total = len(self.treatment_)
        n_treated = int(np.sum(self.treatment_ == 1))
        n_control = int(np.sum(self.treatment_ == 0))
        n_matched_t = len(set(self.matched_treated_))
        n_matched_c = len(set(self.matched_control_))
        n_pairs = len(self.matched_control_)

        print("=" * 60)
        print("PSM Analysis Report")
        print("=" * 60)
        print(f"  Total samples:           {n_total}")
        print(f"  Treatment (T=1):         {n_treated}")
        print(f"  Control (T=0):           {n_control}")
        print(f"  Matched treated units:   {n_matched_t}")
        print(f"  Matched control units:   {n_matched_c}")
        if n_pairs != n_matched_t:
            print(f"  Total matched pairs:     {n_pairs}  (1:k matching)")
        print()

        if self.att_ is not None:
            print(f"  ATT:     {self.att_:.4f}")
            if self.att_ci_ is not None:
                print(f"  CI (95%): [{self.att_ci_[0]:.4f}, {self.att_ci_[1]:.4f}]")
                print(f"  SE:      {self.att_se_:.4f}")
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
        1. 倾向得分分布（匹配前）
        2. 协变量平衡性对比（SMD before vs after）
        3. 处理效应 (ATT) 图示
        4. 首个协变量分布对比
        """
        self._check_matched()

        return plot_psm_report(
            propensity_scores=self.propensity_scores_,
            treatment=self.treatment_,
            outcome=self.outcome_,
            matched_treated=np.unique(self.matched_treated_),
            matched_control=np.unique(self.matched_control_),
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
        """将输入转为 float64 的 2D numpy 数组"""
        if hasattr(x, "values"):
            x = x.values
        x = np.asarray(x, dtype=float)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return x

    @staticmethod
    def _validate_inputs(X, treatment, outcome):
        """验证输入数据的形状、类型和完整性"""
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

        # 检查 NaN / Inf
        for name, arr in [("X", X), ("treatment", treatment), ("outcome", outcome)]:
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN or Inf values")

        # 检查常量协变量列
        if X.shape[1] > 0:
            for j in range(X.shape[1]):
                if np.std(X[:, j]) < 1e-10:
                    raise ValueError(
                        f"Covariate column {j} is constant (zero variance)"
                    )

        if n < 10:
            warnings.warn(
                f"Very small sample size (n={n}). PSM results may be unreliable.",
                UserWarning,
            )
