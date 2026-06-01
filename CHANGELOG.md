# Changelog

## [0.2.0] — 2026-06-02

### Added
- `k:1` matching support (`match(k=3)`)
- Logit-scale caliper (recommended default: 0.2, per Austin 2011)
- Bootstrap standard errors (`estimate_effect(bootstrap=True)`)
- `model_summary()` with AUC diagnostic and extreme PS warnings
- `balance_check()` with SMD (Standardized Mean Difference) tables
- `summary()` full analysis report
- `plot()` 4-panel diagnostic figure
- `trim_common_support` option in `match()` for automatic PS range trimming
- `__repr__` for readable object state display
- Properties: `n_matched_treated_`, `n_matched_control_`, `n_pairs_`
- `caliper` override parameter in `match()`
- Type hints on all public and private methods
- CI/CD via GitHub Actions (Python 3.8–3.13, Ubuntu + Windows)

### Changed
- `_smd()` now uses sample variance (ddof=1) per Cohen's d convention
- `_compute_matched_diffs()` vectorized with `np.bincount` for performance
- Nearest-neighbor algorithm: `ball_tree` → `kd_tree` (better for 1D logit-PS)
- Panel 4 of diagnostic plot now auto-selects the most imbalanced covariate
- `balance_check()` result explicitly passed to `summary()` / `plot()` (no silent caching)

### Fixed
- Removed unused `_is_interactive()` from `viz.py`
- `LogisticRegression` convergence failure now emits a user-visible warning

## [0.1.0] — 2026-05-15

### Added
- Initial release: `PSMatcher` with fit/match/estimate_effect workflow
- Nearest-neighbor 1:1 matching on propensity scores
- ATT with asymptotic normal confidence intervals
- Synthetic data example in `examples/quick_start.py`
