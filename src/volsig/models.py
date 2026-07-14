"""HAR, EWMA and GARCH forecasters.

Convention throughout: a value indexed at date t uses information up to and
including t, and forecasts the average daily variance over t+1 .. t+h.
"""
import numpy as np
import pandas as pd


def har_features(var):
    """Daily / weekly (5d) / monthly (22d) trailing means of daily variance."""
    return pd.DataFrame(
        {
            "rv_d": var,
            "rv_w": var.rolling(5).mean(),
            "rv_m": var.rolling(22).mean(),
        }
    )


def forward_target(var, h):
    """Mean daily variance over t+1 .. t+h. Does not include t."""
    return var.rolling(h).mean().shift(-h)


def ewma_variance(returns, lam=0.94):
    """RiskMetrics EWMA. Value at t (uses r_t) is the forecast for t+1 on."""
    return returns.pow(2).ewm(alpha=1.0 - lam, adjust=False).mean()


def garch_variance_path(returns, omega, alpha, beta):
    """GARCH(1,1) variance recursion.

    The value at index t is the one-step forecast made at t (uses r_t and
    sigma2_t). Starts from the unconditional variance.
    """
    r2 = returns.to_numpy() ** 2
    n = len(r2)
    sig2 = np.empty(n)
    uncond = omega / max(1.0 - alpha - beta, 1e-8)
    prev = uncond
    for t in range(n):
        prev = omega + alpha * r2[t] + beta * prev
        sig2[t] = prev  # forecast for t+1, made with info through t
    return pd.Series(sig2, index=returns.index)


def garch_multistep_mean(sig2_next, omega, alpha, beta, h):
    """Average of sigma2_{t+1..t+h} from the one-step forecast (closed form)."""
    phi = alpha + beta
    uncond = omega / max(1.0 - phi, 1e-8)
    ks = np.arange(h)
    weights = phi**ks
    # sigma2_{t+1+k} = uncond + phi^k (sigma2_{t+1} - uncond)
    return uncond + (sig2_next - uncond) * weights.mean()
