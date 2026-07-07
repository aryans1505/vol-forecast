"""End-to-end: SPY realized-vol forecasting, HAR vs baselines, 1/5/22-day horizons."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from volsig.data import load_ohlc, load_close
from volsig.estimators import garman_klass, annualize_vol
from volsig.models import (
    har_features,
    forward_target,
    ewma_variance,
    garch_variance_path,
    garch_multistep_mean,
)
from volsig.evaluate import qlike, rmse_vol, walk_forward_ols, dm_test

HORIZONS = (1, 5, 22)
OOS_START = 2520  # ~10 years burn-in, OOS starts ~2004
REFIT = 21


def main():
    spy = load_ohlc(ROOT / "data" / "spy_daily.csv")
    vix = load_close(ROOT / "data" / "vix_daily.csv")

    var = garman_klass(spy)
    ret = np.log(spy["close"]).diff().dropna()
    var = var.loc[ret.index]  # align
    vix = vix.reindex(var.index).ffill()
    vix_var = (vix / 100.0) ** 2 / 252.0  # VIX -> daily variance units

    print(f"{len(var)} days, {var.index[0].date()} -> {var.index[-1].date()}")
    print(f"OOS from {var.index[OOS_START].date()}, refit every {REFIT} days")
    med_vol = float(np.median(annualize_vol(var)))
    print(f"median annualized GK vol: {med_vol:.1f}%")

    X_har = har_features(var)
    X_vix = X_har.assign(vix_var=vix_var)

    # GARCH(1,1): estimate params on expanding window every REFIT OOS days,
    # then run the variance recursion with fixed params (info through t only)
    from arch import arch_model

    garch_next = pd.Series(np.nan, index=var.index)
    scale = 100.0
    params_log = []
    for i in range(OOS_START, len(var), REFIT):
        res = arch_model(
            ret.iloc[:i] * scale, vol="GARCH", p=1, q=1, mean="Zero"
        ).fit(disp="off", show_warning=False)
        om, al, be = (
            res.params["omega"] / scale**2,
            res.params["alpha[1]"],
            res.params["beta[1]"],
        )
        params_log.append((var.index[i].date(), om, al, be))
        sig2 = garch_variance_path(ret, om, al, be)
        j_end = min(i + REFIT, len(var))
        garch_next.iloc[i:j_end] = sig2.iloc[i:j_end].to_numpy()
    om, al, be = params_log[-1][1:]

    ewma = ewma_variance(ret)

    rows = []
    dm_rows = []
    plot_series = {}
    for h in HORIZONS:
        y = forward_target(var, h)
        valid = slice(OOS_START, len(y) - h)  # last h days have no target

        # baselines: forecasts at t from info <= t
        rw = var.rolling(h).mean()  # trailing h-day mean (stronger than 1-day)
        ew = ewma.reindex(var.index)
        gar = garch_multistep_mean(garch_next, om, al, be, h)

        har_p, clip1 = walk_forward_ols(X_har, y, OOS_START, h, REFIT)
        hvx_p, clip2 = walk_forward_ols(X_vix, y, OOS_START, h, REFIT)

        models = {"rw": rw, "ewma": ew, "garch": gar, "har": har_p, "har_vix": hvx_p}
        losses = {}
        for name, f in models.items():
            fv = f.iloc[valid].to_numpy()
            rv = y.iloc[valid].to_numpy()
            ok = ~(np.isnan(fv) | np.isnan(rv))
            ql = qlike(fv[ok], rv[ok])
            losses[name] = pd.Series(ql, index=y.index[valid][ok])
            rows.append(
                {
                    "horizon": h,
                    "model": name,
                    "qlike": float(np.mean(ql)),
                    "rmse_vol_pct": rmse_vol(fv[ok], rv[ok]),
                    "n": int(ok.sum()),
                }
            )
        for a, b in [
            ("har", "rw"),
            ("har", "ewma"),
            ("har", "garch"),
            ("har_vix", "har"),
        ]:
            la, lb = losses[a].align(losses[b], join="inner")
            dm_rows.append(
                {"horizon": h, "a": a, "b": b, "dm_stat": dm_test(la.to_numpy(), lb.to_numpy(), h)}
            )
        if h == 5:
            plot_series = {
                "realized": y.iloc[valid],
                "har_vix": hvx_p.iloc[valid],
                "rw": rw.iloc[valid],
            }
        if clip1 or clip2:
            print(f"h={h}: clipped negative forecasts HAR={clip1}, HAR+VIX={clip2}")

    met = pd.DataFrame(rows)
    dm = pd.DataFrame(dm_rows)
    (ROOT / "results").mkdir(exist_ok=True)
    met.to_csv(ROOT / "results" / "metrics.csv", index=False)
    dm.to_csv(ROOT / "results" / "dm_tests.csv", index=False)
    print("\nQLIKE (lower is better):")
    print(met.pivot(index="model", columns="horizon", values="qlike").round(4))
    print("\nRMSE on annualized vol (%):")
    print(met.pivot(index="model", columns="horizon", values="rmse_vol_pct").round(2))
    print("\nDiebold-Mariano (negative => first model better):")
    print(dm.round(2).to_string(index=False))

    # variance risk premium sanity check: VIX vs subsequent realized (22d)
    y22 = forward_target(var, 22)
    both = pd.DataFrame({"vix": vix_var, "rv": y22}).dropna()
    prem = float(np.mean(annualize_vol(both["vix"]) - annualize_vol(both["rv"])))
    print(f"\nMean VIX minus subsequent 22d realized vol: {prem:.2f} vol pts (variance risk premium)")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, s in plot_series.items():
        ax.plot(s.index, annualize_vol(s), lw=0.8, label=name, alpha=0.85)
    ax.set_yscale("log")
    ax.set_ylabel("annualized vol (%)")
    ax.set_title("SPY 5-day realized vol: HAR+VIX forecast vs trailing baseline (OOS)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "vol_forecast.png", dpi=130)
    print("wrote results/metrics.csv, dm_tests.csv, vol_forecast.png")


if __name__ == "__main__":
    main()
