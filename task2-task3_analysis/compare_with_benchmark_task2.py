# compare_with_benchmark.py
"""
Generate task2 aggregate NAV from per-round files and compare with benchmark.
Outputs (all in results_of_task2/):
  - task2_nav_aggregate.csv
  - task2_nav_aggregate_annual.csv
  - compare_metrics_local.csv
  - compare_task2_vs_benchmark.png
Usage:
  python compare_with_benchmark.py
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

# OUTPUT DIRECTORY (changed to results_of_task2)
RESULTS = Path("results_of_task2")
RESULTS.mkdir(exist_ok=True)

def _ensure_tz_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        try:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            df.index = df.index.tz_localize(None)
    return df

def build_task2_aggregate(results_dir: Path = RESULTS) -> pd.Series:
    """
    Read per-round CSVs (task2_nav_round*.csv). Each file must have 'date' and 'nav'.
    Aggregate by aligning dates, forward-filling, then taking mean across rounds.
    Save results/task2_nav_aggregate.csv and return the aggregate Series (index = date).
    """
    pattern = str(results_dir.parent / "results" / "task2_nav_round*.csv")
    files = sorted(glob.glob(pattern))
    # fallback to task1 round files if none found in results/
    if not files:
        pattern2 = str(results_dir.parent / "results" / "task1_nav_round*.csv")
        files = sorted(glob.glob(pattern2))
    if not files:
        raise FileNotFoundError("找不到任何 task2_nav_round*.csv 或 task1_nav_round*.csv 於 results/。請先產生 per-round 檔案。")

    rounds = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["date"], index_col="date")
        df = _ensure_tz_naive_index(df)
        # normalize column name to 'nav'
        if "nav" not in df.columns:
            if df.shape[1] == 1:
                df.columns = ["nav"]
            else:
                raise ValueError(f"{f} 必須包含 'nav' 欄或為單欄檔案。")
        s = df["nav"].sort_index()
        rounds.append(s)

    # union index, reindex each series, forward-fill then take mean
    all_index = rounds[0].index
    for s in rounds[1:]:
        all_index = all_index.union(s.index)
    aligned = []
    for s in rounds:
        s2 = s.reindex(all_index).ffill()
        aligned.append(s2)
    agg_df = pd.concat(aligned, axis=1)
    agg_series = agg_df.mean(axis=1)
    agg_series.name = "nav"

    # save full-frequency aggregate (daily/monthly whatever original)
    out_full = results_dir / "task2_nav_aggregate.csv"
    agg_series.to_frame().to_csv(out_full, index_label="date")
    print(f"[info] Saved aggregate (full freq) -> {out_full}")

    # also save annual resampled (year-end last) for direct comparison if needed
    agg_annual = agg_series.resample("A").last().ffill()
    out_annual = results_dir / "task2_nav_aggregate_annual.csv"
    agg_annual.to_frame().to_csv(out_annual, index_label="date")
    print(f"[info] Saved aggregate (annual) -> {out_annual}")

    return agg_series

def load_nav(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df = _ensure_tz_naive_index(df)
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
    days = (nav_series.index[-1] - nav_series.index[0]).days
    years = days / 365.25 if days > 0 else np.nan
    cumulative = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1.0)
    cagr = float((nav_series.iloc[-1] / nav_series.iloc[0]) ** (1.0 / years) - 1.0) if years and years > 0 else np.nan
    ann_vol = float(returns.std() * np.sqrt(1.0))
    roll_max = nav_series.cummax()
    drawdown = (nav_series - roll_max) / roll_max
    max_dd = float(drawdown.min())
    sharpe = float(cagr / ann_vol) if ann_vol and not np.isnan(ann_vol) and ann_vol != 0 else np.nan
    return {"CAGR": cagr, "Cumulative": cumulative, "AnnVol": ann_vol, "MaxDD": max_dd, "Sharpe": sharpe}

def find_benchmark_annual(results_dir: Path = RESULTS) -> Path:
    # look for benchmark files under the original results/ folder first, then results_of_task2/
    candidates = [
        results_dir.parent / "results" / "benchmark_0050_annual.csv",
        results_dir.parent / "results" / "benchmark_annual.csv",
        results_dir.parent / "results" / "benchmark_partial_annual.csv",
        results_dir / "benchmark_0050_annual.csv",
        results_dir / "benchmark_annual.csv",
        results_dir / "benchmark_partial_annual.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    # fallback: any benchmark*_annual.csv in either folder
    for p in list((results_dir.parent / "results").glob("benchmark*_annual.csv")) + list(results_dir.glob("benchmark*_annual.csv")):
        return p
    raise FileNotFoundError("找不到 benchmark annual CSV。請先執行 make_benchmark.py 或放入 results/ 或 results_of_task2/")

def main():
    # 1) build task2 aggregate from per-round files
    try:
        agg_full = build_task2_aggregate(RESULTS)
    except Exception as e:
        print(f"[error] 無法建立 task2 aggregate: {e}")
        return

    # 2) load annual aggregate for comparison (use the annual resampled file)
    agg_annual_path = RESULTS / "task2_nav_aggregate_annual.csv"
    if not agg_annual_path.exists():
        print("[error] 找不到 task2_nav_aggregate_annual.csv，請確認第一步是否成功。")
        return
    agg_annual = load_nav(agg_annual_path)

    # 3) find benchmark annual and load
    try:
        bench_path = find_benchmark_annual(RESULTS)
    except FileNotFoundError as e:
        print(f"[error] {e}")
        return
    bench = load_nav(bench_path)

    # 4) align and compare
    merged = pd.concat([agg_annual.rename("strategy"), bench.rename("benchmark")], axis=1).dropna()
    if merged.empty:
        print("No overlapping dates between task2 aggregate and benchmark (annual).")
        print("Aggregate range:", agg_annual.index.min(), "to", agg_annual.index.max())
        print("Benchmark range:", bench.index.min(), "to", bench.index.max())
        return

    strat_metrics = calc_metrics(merged["strategy"])
    bench_metrics = calc_metrics(merged["benchmark"])

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
    plt.plot(merged.index, merged["strategy"] / merged["strategy"].iloc[0], label="Task2 Aggregate", linewidth=2)
    plt.plot(merged.index, merged["benchmark"] / merged["benchmark"].iloc[0], label=f"Benchmark ({bench_path.name})", linewidth=2)
    plt.legend()
    plt.title("Task2 Aggregate NAV vs Benchmark (annual, normalized)")
    plt.xlabel("Date")
    plt.ylabel("Normalized NAV")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    out_plot = RESULTS / "compare_task2_vs_benchmark.png"
    plt.savefig(out_plot, dpi=150)
    plt.close()
    print(f"[info] Saved {out_plot}")

if __name__ == "__main__":
    main()
