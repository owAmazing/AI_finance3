# 0050.py (修正版)
from pathlib import Path
import yfinance as yf
import pandas as pd

outdir = Path("results")
outdir.mkdir(exist_ok=True)

t = yf.Ticker("0050.TW")
hist = t.history(start="1998-01-01", interval="1d")

# 檢查欄位並選擇可用的價格欄
if "Adj Close" in hist.columns:
    price_col = "Adj Close"
elif "Close" in hist.columns:
    price_col = "Close"
else:
    raise RuntimeError("yfinance 回傳資料沒有 'Adj Close' 或 'Close' 欄位，請檢查 ticker 或網路連線。")

# 若有 Dividends，仍以 price_col 為基礎計算累積報酬（簡單處理）
hist = hist[[price_col]].dropna().rename(columns={price_col: "price"})
hist["ret"] = hist["price"].pct_change().fillna(0)
hist["nav"] = (1 + hist["ret"]).cumprod()
hist[["nav"]].to_csv(outdir / "benchmark_0050_daily.csv", index_label="date")
print("saved results/benchmark_0050_daily.csv")
