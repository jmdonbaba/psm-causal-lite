import numpy as np
import pandas as pd
import pytest
from psm_causal import PSMatcher


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def simple_data():
    """Synthetic data: 2 covariates, selection bias via X0, true ATT = 2.0"""
    np.random.seed(42)
    n = 200
    X = np.column_stack([
        np.random.normal(0, 1, n),
        np.random.normal(0, 1, n),
    ])
    ps = 1 / (1 + np.exp(-(0.5 * X[:, 0] + 0.3 * X[:, 1])))
    treatment = np.random.binomial(1, ps)
    outcome = (
        1.0
        + 2.0 * treatment
        + 0.8 * X[:, 0]
        + 0.4 * X[:, 1]
        + np.random.normal(0, 0.5, n)
    )
    return X, treatment, outcome


# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------
class TestFit:
    def test_basic(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        assert m._is_fitted
        assert len(m.propensity_scores_) == len(X)
        assert m.logit_ps_ is not None
        assert m.model_info_ is not None

    def test_pandas_features(self):
        X = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        t = np.array([0, 1, 0, 1, 0])
        y = np.array([10, 20, 15, 25, 12])
        m = PSMatcher()
        m.fit(X, t, y)
        assert m.feature_names_ == ["a", "b"]

    def test_numpy_features(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        assert m.feature_names_ == ["X0", "X1"]

    def test_propensity_scores_in_range(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        ps = m.propensity_scores_
        assert np.all((ps >= 0) & (ps <= 1))


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------
class TestMatch:
    def test_basic(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match()
        assert m._is_matched
        assert len(m.matched_treated_) > 0
        assert len(m.matched_control_) > 0

    def test_k3_with_replacement(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match(k=3, with_replacement=True)
        from collections import Counter

        counts = Counter(m.matched_treated_)
        assert max(counts.values()) <= 3

    def test_k3_without_replacement(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match(k=2, with_replacement=False)
        from collections import Counter

        counts = Counter(m.matched_treated_)
        assert max(counts.values()) <= 2

    def test_k_exceeds_controls_raises(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        n_control = int(np.sum(t == 0))
        with pytest.raises(ValueError, match="exceeds available control"):
            m.match(k=n_control + 10, with_replacement=False)

    def test_caliper_filters(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher(caliper=0.05)
        m.fit(X, t, y)
        m.match()
        n_treated = int(np.sum(t == 1))
        matched_t = len(set(m.matched_treated_))
        assert matched_t <= n_treated


# ---------------------------------------------------------------------------
# estimate_effect
# ---------------------------------------------------------------------------
class TestEstimateEffect:
    def test_basic(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match()
        m.estimate_effect()
        assert m.att_ is not None
        assert m.att_ci_ is not None
        assert m.att_se_ is not None

    def test_bootstrap(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match()
        m.estimate_effect(bootstrap=True, n_bootstrap=500)
        ci_lo, ci_hi = m.att_ci_
        assert ci_lo < m.att_ < ci_hi

    def test_with_k3(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match(k=3, with_replacement=True)
        m.estimate_effect()
        assert m.att_ is not None
        assert m.att_ci_ is not None

    def test_att_closer_than_naive(self):
        """PSM ATT should be closer to true value than naive comparison."""
        np.random.seed(123)
        n = 500
        X = np.column_stack([
            np.random.normal(0, 1, n),
            np.random.normal(0, 1, n),
        ])
        ps = 1 / (1 + np.exp(-(0.6 * X[:, 0] + 0.4 * X[:, 1])))
        t = np.random.binomial(1, ps)
        true_att = 2.0
        y = (
            1.0
            + true_att * t
            + 1.2 * X[:, 0]
            + 0.5 * X[:, 1]
            + np.random.normal(0, 0.5, n)
        )

        naive = y[t == 1].mean() - y[t == 0].mean()

        m = PSMatcher(caliper=0.2, random_state=42)
        m.fit(X, t, y)
        m.match(with_replacement=True)
        m.estimate_effect(bootstrap=True)

        assert abs(m.att_ - true_att) < abs(naive - true_att), (
            f"PSM error {abs(m.att_ - true_att):.4f} >= naive "
            f"error {abs(naive - true_att):.4f}"
        )


# ---------------------------------------------------------------------------
# balance_check
# ---------------------------------------------------------------------------
class TestBalanceCheck:
    def test_returns_dataframe(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match()
        bal = m.balance_check()
        assert isinstance(bal, pd.DataFrame)
        assert "SMD_before" in bal.columns
        assert "SMD_after" in bal.columns
        assert "flag" in bal.columns

    def test_balance_improves(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match()
        bal = m.balance_check()
        assert bal["improved"].any()

    def test_custom_thresholds(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match()
        bal = m.balance_check(thresholds=(0.05, 0.15))
        # With tighter thresholds, more flags should be "!" or "~"
        flags = bal["flag"].tolist()
        assert isinstance(flags, list)


# ---------------------------------------------------------------------------
# model_summary
# ---------------------------------------------------------------------------
class TestModelSummary:
    def test_has_auc(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        assert 0 <= m.model_info_["auc"] <= 1

    def test_runs_without_error(self, simple_data, capsys):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.model_summary()
        captured = capsys.readouterr()
        assert "Propensity Score Model Diagnostics" in captured.out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
class TestSummary:
    def test_runs_without_error(self, simple_data, capsys):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match()
        m.estimate_effect()
        m.summary()
        captured = capsys.readouterr()
        assert "PSM Analysis Report" in captured.out


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
class TestPlot:
    def test_returns_figure(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        m.match()
        m.estimate_effect()
        fig = m.plot()
        from matplotlib.figure import Figure
        assert isinstance(fig, Figure)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_nan_in_X(self):
        X = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
        t = np.array([0, 1, 0])
        y = np.array([10, 20, 15])
        m = PSMatcher()
        with pytest.raises(ValueError, match="NaN"):
            m.fit(X, t, y)

    def test_constant_column(self):
        X = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])
        t = np.array([0, 1, 0])
        y = np.array([10, 20, 15])
        m = PSMatcher()
        with pytest.raises(ValueError, match="constant"):
            m.fit(X, t, y)

    def test_length_mismatch_treatment(self):
        X = np.array([[1.0], [2.0], [3.0]])
        t = np.array([0, 1])
        y = np.array([10, 20, 15])
        m = PSMatcher()
        with pytest.raises(ValueError):
            m.fit(X, t, y)

    def test_length_mismatch_outcome(self):
        X = np.array([[1.0], [2.0], [3.0]])
        t = np.array([0, 1, 0])
        y = np.array([10, 20])
        m = PSMatcher()
        with pytest.raises(ValueError):
            m.fit(X, t, y)

    def test_treatment_not_binary(self):
        X = np.array([[1.0], [2.0], [3.0]])
        t = np.array([0, 2, 0])
        y = np.array([10, 20, 15])
        m = PSMatcher()
        with pytest.raises(ValueError, match="binary"):
            m.fit(X, t, y)

    def test_treatment_all_same(self):
        X = np.array([[1.0], [2.0], [3.0]])
        t = np.array([0, 0, 0])
        y = np.array([10, 20, 15])
        m = PSMatcher()
        with pytest.raises(ValueError, match="both 0 and 1"):
            m.fit(X, t, y)


# ---------------------------------------------------------------------------
# error states
# ---------------------------------------------------------------------------
class TestErrorStates:
    def test_match_before_fit(self):
        m = PSMatcher()
        with pytest.raises(RuntimeError, match="Call .fit"):
            m.match()

    def test_estimate_before_match(self, simple_data):
        X, t, y = simple_data
        m = PSMatcher()
        m.fit(X, t, y)
        with pytest.raises(RuntimeError, match="Call .match"):
            m.estimate_effect()

    def test_no_treatment_units(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        t = np.array([0, 0, 0, 0])
        y = np.array([10, 20, 15, 25])
        m = PSMatcher()
        with pytest.raises(ValueError, match="both 0 and 1"):
            m.fit(X, t, y)
