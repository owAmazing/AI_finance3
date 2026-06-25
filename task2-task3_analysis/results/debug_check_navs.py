# scripts/debug_check_navs.py
import pandas as pd
from pathlib import Path
import yfinance as yf

res = Path("results")
agg = pd.read_csv(res/"task1_nav_aggregate.csv", parse_dates=["date"], index_col="date")
print("Aggregate head/tail:")
print(agg.head(), "\n", agg.tail())
print("Aggregate index tz:", getattr(agg.index, "tz", None))

# per-round sample
for p in sorted(res.glob("task1_nav_round*.csv"))[:5]:
    df = pd.read_csv(p, parse_dates=["date"], index_col="date")
    print(p.name, "rows:", len(df), "min/max:", df.index.min(), df.index.max())

# benchmark (if local CSV exists)
bench_csv = res/"benchmark_0050_daily.csv"
if bench_csv.exists():
    b = pd.read_csv(bench_csv, parse_dates=["date"], index_col="date")
    print("Benchmark CSV min/max:", b.index.min(), b.index.max(), "tz:", getattr(b.index, "tz", None))
else:
    # fetch via yfinance to inspect columns and range
    t = yf.Ticker("0050.TW")
    hist = t.history(start=None, interval="1d")
    print("yfinance columns:", hist.columns.tolist())
    if not hist.empty:
        print("yfinance min/max:", hist.index.min(), hist.index.max(), "tz:", getattr(hist.index, "tz", None))
