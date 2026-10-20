"""Unit tests for Milestone 19 attribution and calibration mathematics."""

import numpy as np
import pytest
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


def benjamini_hochberg_fdr(p_values: np.ndarray) -> np.ndarray:
    """Compute Benjamini-Hochberg FDR-adjusted q-values."""
    p_vals = np.asarray(p_values, dtype=float)
    n = len(p_vals)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_vals)
    ranked_p = p_vals[order]
    q_vals = np.zeros(n, dtype=float)
    q_vals[-1] = ranked_p[-1]
    for i in range(n - 2, -1, -1):
        q_candidate = ranked_p[i] * n / (i + 1)
        q_vals[i] = min(q_candidate, q_vals[i + 1])
    # Restore original order and clip to [0, 1]
    rev_order = np.argsort(order)
    return np.clip(q_vals[rev_order], 0.0, 1.0)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta non-parametric effect size between two groups.
    
    Returns value in [-1, 1], where 0 indicates identical distributions.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    n_x, n_y = len(x_arr), len(y_arr)
    if n_x == 0 or n_y == 0:
        return 0.0
    # Pairwise comparisons: x > y adds +1, x < y adds -1, x == y adds 0
    greater = np.sum(x_arr[:, None] > y_arr[None, :])
    smaller = np.sum(x_arr[:, None] < y_arr[None, :])
    return float((greater - smaller) / (n_x * n_y))


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins."""
    p_arr = np.asarray(probs, dtype=float)
    y_arr = np.asarray(labels, dtype=int)
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(p_arr)

    for i in range(n_bins):
        low, high = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (p_arr >= low) & (p_arr <= high if i == n_bins - 1 else p_arr < high)
        if np.any(mask):
            bin_size = np.sum(mask)
            bin_conf = np.mean(p_arr[mask])
            bin_acc = np.mean(y_arr[mask])
            ece += (bin_size / n) * np.abs(bin_acc - bin_conf)

    return float(ece)


class PlattScaler:
    """Platt scaling (logistic regression on logits with learned slope w and bias b)."""

    def __init__(self):
        self.w: float = 1.0
        self.b: float = 0.0
        self.is_fitted: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "PlattScaler":
        clf = LogisticRegression(solver="lbfgs", C=1e5, max_iter=1000)
        X = logits.reshape(-1, 1)
        clf.fit(X, labels)
        self.w = float(clf.coef_[0][0])
        self.b = float(clf.intercept_[0])
        self.is_fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("PlattScaler must be fitted before transforming.")
        scaled_z = self.w * logits + self.b
        return 1.0 / (1.0 + np.exp(-scaled_z))


class TemperatureScaler:
    """Temperature scaling (single parameter T > 0)."""

    def __init__(self):
        self.T: float = 1.0
        self.is_fitted: bool = False

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaler":
        z = np.asarray(logits, dtype=float)
        y = np.asarray(labels, dtype=float)

        def nll_loss(t_param):
            t = t_param[0]
            scaled = z / t
            probs = 1.0 / (1.0 + np.exp(-scaled))
            eps = 1e-12
            probs = np.clip(probs, eps, 1.0 - eps)
            return -np.mean(y * np.log(probs) + (1.0 - y) * np.log(1.0 - probs))

        res = minimize(nll_loss, [1.0], bounds=[(0.1, 10.0)], method="L-BFGS-B")
        self.T = float(res.x[0])
        self.is_fitted = True
        return self

    def transform(self, logits: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TemperatureScaler must be fitted before transforming.")
        return 1.0 / (1.0 + np.exp(-logits / self.T))


def derive_threshold_policies(probs: np.ndarray, labels: np.ndarray) -> dict:
    """Derive threshold policies on validation set."""
    p_arr = np.asarray(probs, dtype=float)
    y_arr = np.asarray(labels, dtype=int)
    thresholds = np.linspace(0.001, 0.999, 1000)

    clean_mask = y_arr == 0
    fake_mask = y_arr == 1

    tau_95 = 0.50
    tau_99 = 0.50
    best_youden = -1.0
    tau_youden = 0.50
    best_f1 = -1.0
    tau_f1 = 0.50

    for t in thresholds:
        spec = np.mean(p_arr[clean_mask] < t) if np.any(clean_mask) else 0.0
        sens = np.mean(p_arr[fake_mask] >= t) if np.any(fake_mask) else 0.0
        youden = sens + spec - 1.0
        
        tp = np.sum((p_arr >= t) & fake_mask)
        fp = np.sum((p_arr >= t) & clean_mask)
        fn = np.sum((p_arr < t) & fake_mask)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        if spec >= 0.95 and tau_95 == 0.50:
            tau_95 = t
        if spec >= 0.99 and tau_99 == 0.50:
            tau_99 = t
        if youden > best_youden:
            best_youden = youden
            tau_youden = t
        if f1 > best_f1:
            best_f1 = f1
            tau_f1 = t

    return {
        "tau_95": float(tau_95),
        "tau_99": float(tau_99),
        "tau_youden": float(tau_youden),
        "tau_f1": float(tau_f1),
    }


# =============================================================================
# UNIT TESTS
# =============================================================================

def test_benjamini_hochberg_fdr_invariants():
    """Verify Benjamini-Hochberg preserves monotonicity, bounding, and identity."""
    p_vals = np.array([0.001, 0.01, 0.03, 0.05, 0.20, 0.80])
    q_vals = benjamini_hochberg_fdr(p_vals)

    assert len(q_vals) == len(p_vals)
    assert (q_vals >= 0.0).all() and (q_vals <= 1.0).all()
    assert (q_vals >= p_vals).all()
    assert np.all(np.diff(q_vals) >= 0)


def test_cliffs_delta_effect_size_invariants():
    """Verify Cliff's delta effect size bounds and extreme cases."""
    a = np.array([10, 11, 12, 13])
    b = np.array([1, 2, 3, 4])
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0

    assert cliffs_delta(a, a) == 0.0

    c = np.array([2, 5, 10, 12])
    delta = cliffs_delta(c, b)
    assert -1.0 <= delta <= 1.0
    assert delta > 0.0


def test_ece_bounds_and_extremes():
    """Verify ECE is strictly within [0, 1] and near zero for well-calibrated predictions."""
    probs = np.array([0.1, 0.1, 0.9, 0.9])
    labels = np.array([0, 0, 1, 1])
    ece = compute_ece(probs, labels, n_bins=10)
    assert 0.0 <= ece <= 1.0
    assert ece < 0.15

    inv_probs = np.array([0.9, 0.9, 0.1, 0.1])
    inv_ece = compute_ece(inv_probs, labels, n_bins=10)
    assert inv_ece > 0.70


def test_platt_scaler_learns_bias_and_preserves_monotonicity():
    """Verify Platt scaling learns a non-zero intercept to fix severe offset."""
    rng = np.random.default_rng(42)
    reals_z = rng.normal(loc=3.0, scale=0.5, size=200)
    fakes_z = rng.normal(loc=5.0, scale=0.5, size=200)
    logits = np.concatenate([reals_z, fakes_z])
    labels = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])

    scaler = PlattScaler().fit(logits, labels)
    assert scaler.is_fitted
    assert scaler.b < 0.0
    assert scaler.w > 0.0

    calibrated_probs = scaler.transform(logits)
    assert (calibrated_probs >= 0.0).all() and (calibrated_probs <= 1.0).all()

    real_cal_probs = calibrated_probs[:200]
    fake_cal_probs = calibrated_probs[200:]
    assert np.mean(real_cal_probs) < 0.35
    assert np.mean(fake_cal_probs) > 0.65


def test_temperature_scaler_bounds_and_transform():
    """Verify Temperature scaling fits positive T and transforms probabilities."""
    logits = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    labels = np.array([0, 0, 0, 1, 1])
    scaler = TemperatureScaler().fit(logits, labels)
    assert scaler.is_fitted
    assert scaler.T > 0.0

    probs = scaler.transform(logits)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert np.all(np.diff(probs) >= 0)


def test_derive_threshold_policies_bounds():
    """Verify threshold policy derivation returns valid thresholds."""
    rng = np.random.default_rng(42)
    probs = np.concatenate([rng.uniform(0.01, 0.30, size=100), rng.uniform(0.70, 0.99, size=100)])
    labels = np.concatenate([np.zeros(100, dtype=int), np.ones(100, dtype=int)])

    policies = derive_threshold_policies(probs, labels)
    assert "tau_95" in policies
    assert "tau_99" in policies
    assert "tau_youden" in policies
    assert "tau_f1" in policies

    for k, v in policies.items():
        assert 0.0 < v < 1.0, f"Policy {k} returned invalid threshold {v}"
