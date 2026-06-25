# scripts/compare_with_local_benchmark.py
"""
Compare task1 aggregate NAV with a local benchmark annual NAV file.
Outputs:
  - results/compare_metrics_local.csv
  - results/compare_aggregate_vs_benchmark.png
Usage:
  python scripts/compare_with_local_benchmark.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

def load_nav(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index)
    # ensure tz-naive
    if getattr(df.index, "tz", None) is not None:
        try:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            df.index = df.index.tz_localize(None)
    if "nav" not in df.columns:
        if df.shape[1] == 1:
            df.columns = ["nav"]
        else:
            raise ValueError(f"{path} must contain a 'nav' column or be single-column.")
    return df["nav"]

def calc_metrics(nav_series: pd.Series) -> dict:
    returns = nav_series.pct_change().dropna()
    if returns.empty:
        return {"CAGR": np.nan, "Cumulative": np.nan, "AnnVol": np.nan, "MaxDD": np.nan, "Sharpe": np.nan}
    # For annual series, annualization factor = 1
    days = (nav_series.index[-1] - nav_series.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cumulative = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1.0)
    cagr = float((nav_series.iloc[-1] / nav_series.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    ann_vol = float(returns.std() * np.sqrt(1.0))  # annualized for yearly returns
    roll_max = nav_series.cummax()
    drawdown = (nav_series - roll_max) / roll_max
    max_dd = float(drawdown.min())
    sharpe = float(cagr / ann_vol) if ann_vol and not np.isnan(ann_vol) and ann_vol != 0 else np.nan
    return {"CAGR": cagr, "Cumulative": cumulative, "AnnVol": ann_vol, "MaxDD": max_dd, "Sharpe": sharpe}

def main():
    agg_path = RESULTS / "task1_nav_aggregate.csv"
    bench_path_candidates = [
        RESULTS / "benchmark_0050_annual.csv",
        RESULTS / "benchmark_annual.csv",
        RESULTS / "benchmark_partial_annual.csv",
    ]

    if not agg_path.exists():
        raise FileNotFoundError(f"Aggregate NAV not found: {agg_path}")

    # find an existing benchmark annual file
    bench_path = None
    for p in bench_path_candidates:
        if p.exists():
            bench_path = p
            break

    if bench_path is None:
        raise FileNotFoundError("No benchmark annual CSV found in results/. Run make_benchmark.py first.")

    # load series
    agg = load_nav(agg_path)
    bench = load_nav(bench_path)

    # align on common dates (annual)
    merged = pd.concat([agg.rename("strategy"), bench.rename("benchmark")], axis=1).dropna()
    if merged.empty:
        print("No overlapping dates between aggregate and benchmark (annual).")
        print("Aggregate range:", agg.index.min(), "to", agg.index.max())
        print("Benchmark range:", bench.index.min(), "to", bench.index.max())
        return

    # compute metrics
    strat_metrics = calc_metrics(merged["strategy"])
    bench_metrics = calc_metrics(merged["benchmark"])

    # save metrics table
    rows = [
        {"metric": "CAGR", "strategy": strat_metrics["CAGR"], "benchmark": bench_metrics["CAGR"]},
        {"metric": "Cumulative", "strategy": strat_metrics["Cumulative"], "benchmark": bench_metrics["Cumulative"]},
        {"metric": "AnnVol", "strategy": strat_metrics["AnnVol"], "benchmark": bench_metrics["AnnVol"]},
        {"metric": "MaxDD", "strategy": strat_metrics["MaxDD"], "benchmark": bench_metrics["MaxDD"]},
        {"metric": "Sharpe", "strategy": strat_metrics["Sharpe"], "benchmark": bench_metrics["Sharpe"]},
    ]
    pd.DataFrame(rows).to_csv(RESULTS / "compare_metrics_local.csv", index=False)
    print(f"[info] Saved {RESULTS/'compare_metrics_local.csv'}")

    # plot normalized series
    plt.figure(figsize=(10, 6))
    plt.plot(merged.index, merged["strategy"] / merged["strategy"].iloc[0], label="Strategy (aggregate)", linewidth=2)
    plt.plot(merged.index, merged["benchmark"] / merged["benchmark"].iloc[0], label=f"Benchmark ({bench_path.name})", linewidth=2)
    plt.legend()
    plt.title("Aggregate NAV vs Benchmark (annual, normalized)")
    plt.xlabel("Date")
    plt.ylabel("Normalized NAV")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out_plot = RESULTS / "compare_aggregate_vs_benchmark.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"[info] Saved {out_plot}")

if __name__ == "__main__":
    main()
