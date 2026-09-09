"""Walk-forward evaluation: expanding-window OLS, QLIKE/RMSE, Diebold-Mariano."""
import numpy as np
import pandas as pd


def qlike(fvar, rvar):
    """QLIKE loss. Robust to noise in the variance proxy."""
    r = rvar / fvar
    return r - np.log(r) - 1.0


def rmse_vol(fvar, rvar):
    """RMSE on annualized vol (%)."""
    fv = np.sqrt(np.asarray(fvar) * 252.0) * 100.0
    rv = np.sqrt(np.asarray(rvar) * 252.0) * 100.0
    return float(np.sqrt(np.mean((fv - rv) ** 2)))


def walk_forward_ols(X, y, oos_start, h, refit=21, log_space=False):
    """Expanding-window OLS, refit every `refit` OOS days.

    Predicting at i, train only on rows j <= i - h, so every training target
    (which spans j+1..j+h) is finished before the forecast date. OLS variance
    forecasts can go negative; floored at the 1st percentile of the training
    target (count returned).

    With log_space=True the regression is log-variance on log-features:
    forecasts are positive by construction (no flooring), mapped back as
    exp(Xb + s2/2) with s2 the training residual variance (lognormal mean).
    """
    Xv = X.to_numpy(dtype=float)
    yv = y.to_numpy(dtype=float)
    if log_space:
        Xv = np.log(np.where(Xv > 0, Xv, np.nan))
        yv = np.log(np.where(yv > 0, yv, np.nan))
    n = len(y)
    preds = np.full(n, np.nan)
    beta = None
    floor = 1e-12
    s2 = 0.0
    n_clip = 0
    for i in range(oos_start, n):
        if (i - oos_start) % refit == 0:
            hi = i - h + 1  # rows 0..hi-1 have j <= i-h
            mask = ~(np.isnan(Xv[:hi]).any(axis=1) | np.isnan(yv[:hi]))
            ytr = yv[:hi][mask]
            A = np.column_stack([np.ones(mask.sum()), Xv[:hi][mask]])
            beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
            s2 = float(np.var(ytr - A @ beta))
            floor = max(float(np.percentile(ytr, 1)), 1e-12)
        if not np.isnan(Xv[i]).any():
            p = beta[0] + Xv[i] @ beta[1:]
            if log_space:
                p = np.exp(p + 0.5 * s2)
            elif p < floor:
                p = floor
                n_clip += 1
            preds[i] = p
    return pd.Series(preds, index=y.index), n_clip


def dm_test(loss_a, loss_b, h):
    """Diebold-Mariano test, H0: equal expected loss. Negative stat => A better.

    Newey-West HAC variance with h-1 lags (forecast overlap), plus the
    Harvey-Leybourne-Newbold small-sample correction.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    d = d[~np.isnan(d)]
    n = len(d)
    dbar = d.mean()
    dc = d - dbar
    gamma = [np.mean(dc * dc)]
    for k in range(1, h):
        gamma.append(np.mean(dc[k:] * dc[:-k]))
    var_dbar = (gamma[0] + 2.0 * sum(gamma[1:])) / n
    if var_dbar <= 0:  # can happen with heavy overlap; fall back to gamma0
        var_dbar = gamma[0] / n
    stat = dbar / np.sqrt(var_dbar)
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(stat * hln)
