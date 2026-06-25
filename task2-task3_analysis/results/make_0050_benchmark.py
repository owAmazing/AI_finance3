# scripts/make_benchmark.py
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime

RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)

def read_aggregate_nav(path=RESULTS/"task1_nav_aggregate.csv"):
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    df.index = pd.to_datetime(df.index)
    return df["nav"]

def fetch_daily_nav(ticker, start, end):
    t = yf.Ticker(ticker)
    hist = t.history(start=start.strftime("%Y-%m-%d"), end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"), interval="1d", auto_adjust=False)
    if hist.empty:
        return None
    price_col = "Adj Close" if "Adj Close" in hist.columns else ("Close" if "Close" in hist.columns else None)
    if price_col is None:
        return None
    df = hist[[price_col]].rename(columns={price_col: "price"}).dropna()
    df["ret"] = df["price"].pct_change().fillna(0)
    df["nav"] = (1 + df["ret"]).cumprod()
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        try:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        except Exception:
            df.index = df.index.tz_localize(None)
    return df[["nav"]]

def save_daily_and_annual(nav_daily: pd.DataFrame, out_prefix="benchmark"):
    daily_path = RESULTS / f"{out_prefix}_daily.csv"
    nav_daily.to_csv(daily_path, index_label="date")
    nav_annual = nav_daily["nav"].resample("A").last().ffill()
    nav_annual.to_frame().to_csv(RESULTS / f"{out_prefix}_annual.csv", index_label="date")
    return daily_path, RESULTS / f"{out_prefix}_annual.csv"

def main():
    agg_path = RESULTS / "task1_nav_aggregate.csv"
    if not agg_path.exists():
        raise FileNotFoundError(f"找不到 {agg_path}. 請先產生 task1_nav_aggregate.csv")
    agg_nav = read_aggregate_nav(agg_path)
    start = agg_nav.index.min()
    end = agg_nav.index.max()
    print(f"Aggregate period: {start.date()} to {end.date()}")

    # 優先嘗試 0050，若資料不足則 fallback 到 ^TWII
    candidates = ["0050.TW", "^TWII"]
    for tk in candidates:
        print(f"嘗試抓取 {tk} ({start.date()} -> {end.date()}) ...")
        daily = fetch_daily_nav(tk, start, end)
        if daily is None or daily.empty:
            print(f"  {tk}: 無足夠資料，嘗試下一個基準")
            continue
        # 檢查是否覆蓋 aggregate 的最早日期（若 daily 的最早日晚於 aggregate 最早日，視為不覆蓋）
        if daily.index.min() <= start:
            daily_path, annual_path = save_daily_and_annual(daily, out_prefix="benchmark")
            print(f"成功使用 {tk} 作為基準。日度檔: {daily_path}；年終檔: {annual_path}")
            print(f"基準實際範圍: {daily.index.min().date()} -> {daily.index.max().date()}")
            return
        else:
            print(f"  {tk} 的最早資料為 {daily.index.min().date()}，晚於 aggregate 的最早日，視為不覆蓋。")
    # 若所有候選都不覆蓋，仍把最後一個抓到的 daily 存下並提醒
    if 'daily' in locals() and daily is not None and not daily.empty:
        daily_path, annual_path = save_daily_and_annual(daily, out_prefix="benchmark_partial")
        print("所有候選基準都無法完全覆蓋 aggregate 期間。已儲存可用的部分基準資料：")
        print(f"  {daily_path}, {annual_path}")
    else:
        raise RuntimeError("無法從 yfinance 取得任何可用基準資料。請改用本地歷史資料或其他資料來源。")

if __name__ == "__main__":
    main()
