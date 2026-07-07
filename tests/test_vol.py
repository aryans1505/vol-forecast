import numpy as np
import pandas as pd
import pytest

from volsig.estimators import garman_klass, parkinson, LN2
from volsig.models import har_features, forward_target, garch_variance_path
from volsig.evaluate import qlike, dm_test, walk_forward_ols


def _ohlc(n=100, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    op = close * np.exp(rng.normal(0, 0.003, n))
    hi = np.maximum(op, close) * np.exp(np.abs(rng.normal(0, 0.005, n)))
    lo = np.minimum(op, close) * np.exp(-np.abs(rng.normal(0, 0.005, n)))
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"open": op, "high": hi, "low": lo, "close": close}, index=idx)


def test_garman_klass_hand_computed():
    df = pd.DataFrame({"open": [100.0], "high": [102.0], "low": [99.0], "close": [101.0]})
    hl = np.log(102.0 / 99.0)
    co = np.log(101.0 / 100.0)
    expected = 0.5 * hl**2 - (2 * LN2 - 1) * co**2
    assert garman_klass(df).iloc[0] == pytest.approx(expected)
    assert parkinson(df).iloc[0] == pytest.approx(hl**2 / (4 * LN2))


def test_har_features_no_lookahead():
    """Features at day t are unchanged when all future rows are deleted."""
    var = garman_klass(_ohlc(120))
    full = har_features(var)
    trunc = har_features(var.iloc[:80])
    pd.testing.assert_frame_equal(full.iloc[:80], trunc)


def test_target_strictly_forward():
    """Target at t must not involve var_t or anything before it."""
    var = pd.Series(np.arange(1.0, 31.0), index=pd.bdate_range("2020-01-01", periods=30))
    tgt = forward_target(var, h=5)
    # at t=0 (var=1), target = mean(var_1..var_5) = mean(2..6) = 4
    assert tgt.iloc[0] == pytest.approx(4.0)
    # changing var at t leaves target at t untouched
    var2 = var.copy()
    var2.iloc[10] = 999.0
    assert forward_target(var2, 5).iloc[10] == tgt.iloc[10]


def test_qlike_minimized_at_truth():
    assert qlike(2.0, 2.0) == pytest.approx(0.0)
    assert qlike(1.0, 2.0) > 0
    assert qlike(4.0, 2.0) > 0


def test_dm_sign():
    """Uniformly smaller losses for A => negative DM stat."""
    rng = np.random.default_rng(1)
    base = np.abs(rng.normal(1, 0.1, 500))
    stat = dm_test(base * 0.5, base, h=5)
    assert stat < -2


def test_walk_forward_trains_only_on_realized_targets():
    """With h=5, the fit at OOS start must exclude rows within h of it."""
    n = 60
    idx = pd.bdate_range("2020-01-01", periods=n)
    X = pd.DataFrame({"x": np.ones(n)}, index=idx)
    y = pd.Series(np.ones(n), index=idx)
    # poison targets in the h-window just before oos_start: if they leak into
    # the fit, the intercept-only prediction moves away from 1.0
    oos_start, h = 40, 5
    y.iloc[oos_start - h + 1 : oos_start + 1] = 1000.0
    preds, _ = walk_forward_ols(X, y, oos_start=oos_start, h=h, refit=1000)
    assert preds.iloc[oos_start] == pytest.approx(1.0, abs=1e-6)


def test_garch_recursion_matches_manual():
    r = pd.Series([0.01, -0.02, 0.015])
    omega, alpha, beta = 1e-6, 0.1, 0.85
    sig2 = garch_variance_path(r, omega, alpha, beta)
    uncond = omega / (1 - alpha - beta)
    s1 = omega + alpha * 0.01**2 + beta * uncond
    s2 = omega + alpha * 0.02**2 + beta * s1
    assert sig2.iloc[0] == pytest.approx(s1)
    assert sig2.iloc[1] == pytest.approx(s2)
