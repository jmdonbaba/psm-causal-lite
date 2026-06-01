"""PSM diagnostic visualization — matplotlib only, zero extra deps"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _is_interactive():
    """Check if we're in an interactive environment (notebook, IDE, etc.)"""
    try:
        get_ipython().__class__.__name__
        return True
    except NameError:
        return matplotlib.get_backend().lower() in (
            "tkagg", "qtagg", "qt5agg", "macosx",
        )


def plot_psm_report(
    propensity_scores,
    treatment,
    outcome,
    matched_treated,
    matched_control,
    balance_df,
    att,
    att_ci,
    X,
    feature_names,
    figsize=(12, 8),
):
    """4-panel PSM diagnostic report

    Panel 1: Propensity score distribution (before matching)
    Panel 2: Covariate balance SMD (before vs after)
    Panel 3: ATT — violin plots of matched outcomes
    Panel 4: Distribution of first covariate across groups
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # ---- Panel 1: Propensity Score Distribution ----
    ax = axes[0, 0]
    t_mask = treatment == 1
    c_mask = treatment == 0
    ax.hist(
        propensity_scores[t_mask],
        bins=30,
        alpha=0.5,
        color="red",
        label=f"Treatment (n={t_mask.sum()})",
    )
    ax.hist(
        propensity_scores[c_mask],
        bins=30,
        alpha=0.5,
        color="blue",
        label=f"Control (n={c_mask.sum()})",
    )
    ax.set_xlabel("Propensity Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Propensity Score Distribution (Before Matching)")
    ax.legend(fontsize=8)

    # ---- Panel 2: Covariate Balance (SMD) ----
    ax = axes[0, 1]
    n_feat = len(balance_df)
    y_pos = np.arange(n_feat)
    bar_h = 0.35
    ax.barh(
        y_pos - bar_h / 2,
        balance_df["SMD_before"],
        bar_h,
        color="gray",
        alpha=0.6,
        label="Before Matching",
    )
    ax.barh(
        y_pos + bar_h / 2,
        balance_df["SMD_after"],
        bar_h,
        color="green",
        alpha=0.7,
        label="After Matching",
    )
    ax.axvline(
        x=0.1, color="orange", linestyle="--", linewidth=1,
        label="|SMD| = 0.1 (good)",
    )
    ax.axvline(
        x=0.2, color="red", linestyle=":", linewidth=1,
        label="|SMD| = 0.2 (fair)",
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(balance_df["feature"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Standardized Mean Difference (SMD)")
    ax.set_title("Covariate Balance: Before vs After Matching")
    ax.legend(fontsize=7, loc="lower right")

    # ---- Panel 3: Treatment Effect (ATT) ----
    ax = axes[1, 0]
    if att is not None and att_ci is not None:
        y_t = outcome[matched_treated]
        y_c = outcome[matched_control]
        positions = [1, 0]
        ax.violinplot(
            [y_t, y_c], positions=positions, showmeans=True, showmedians=True,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Matched Control", "Treated"])
        ax.set_ylabel("Outcome")
        ci_str = f"  [{att_ci[0]:.4f}, {att_ci[1]:.4f}]"
        ax.set_title(f"ATT = {att:.4f}{ci_str}")

    # ---- Panel 4: First Covariate Distribution ----
    ax = axes[1, 1]
    c_matched = np.zeros(len(treatment), dtype=bool)
    c_matched[matched_control] = True
    t_matched = np.zeros(len(treatment), dtype=bool)
    t_matched[matched_treated] = True

    if n_feat > 0 and X is not None:
        idx = 0
        x_col = X[:, idx]
        fname = feature_names[idx] if feature_names else "X0"

        positions = [
            ("Treated\n(Before)", t_mask),
            ("Control\n(Before)", c_mask),
            ("Treated\n(Matched)", t_matched),
            ("Control\n(Matched)", c_matched),
        ]
        vp_data = [x_col[mask] for _, mask in positions]
        ax.violinplot(vp_data, showmeans=True, showmedians=True)
        ax.set_xticks(range(1, len(positions) + 1))
        ax.set_xticklabels([p[0] for p in positions], fontsize=8)
        ax.set_ylabel(fname)
        ax.set_title(f"Distribution of '{fname}' Across Groups")

    plt.tight_layout()
    return fig
