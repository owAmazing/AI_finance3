# scripts/convert_task1_to_nav.py
"""
把 task1_results.xlsx 的「各輪逐年報酬」轉成 NAV（每輪與彙總）
輸出到 results/ 目錄：
  - task1_nav_roundXX.csv    每輪的年度 NAV（index=date, col=nav）
  - task1_nav_aggregate.csv  合併年度 NAV（每年取平均或取最後一輪，依參數）
可用參數：
  --input      : 輸入 xlsx（預設 task1_results.xlsx）
  --outdir     : 輸出資料夾（預設 results）
  --agg-method : aggregate 方法，mean 或 last（預設 mean）
  --per-round  : 若指定，輸出每輪 NAV
"""
from pathlib import Path
import argparse
import pandas as pd
import sys

DEFAULT_INPUT = "task1_results.xlsx"
DEFAULT_OUTDIR = "results"
SHEET_CANDIDATES = ["各輪逐年報酬", "各輪逐年報酬 ", "yearly", "annual_returns"]

def find_sheet_name(xlsx_path: Path) -> str:
    """嘗試找到包含逐年報酬的 sheet 名稱"""
    try:
        xl = pd.ExcelFile(xlsx_path)
    except Exception as e:
        raise RuntimeError(f"無法讀取 Excel: {xlsx_path}  ({e})")
    for s in xl.sheet_names:
        if s in SHEET_CANDIDATES:
            return s
    # 若沒有精確匹配，嘗試包含關鍵字
    for s in xl.sheet_names:
        low = s.lower()
        if "年" in s or "year" in low or "逐年" in s:
            return s
    # fallback: 回傳第一個 sheet
    return xl.sheet_names[0]

def load_yearly_sheet(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    # 嘗試標準化欄位名稱
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    # 常見欄位名稱
    possible_round = [c for c in df.columns if "輪" in c or "round" in c.lower()]
    possible_year  = [c for c in df.columns if "年" in c and "報" not in c]  # 年份欄
    possible_ret   = [c for c in df.columns if "年報酬" in c or "annual" in c.lower() or "Return" in c]
    # 找欄位
    round_col = possible_round[0] if possible_round else ("輪次" if "輪次" in df.columns else None)
    year_col  = possible_year[0]  if possible_year  else ("年份" if "年份" in df.columns else None)
    ret_col   = possible_ret[0]   if possible_ret   else ("年報酬（%）" if "年報酬（%）" in df.columns else ("年報酬" if "年報酬" in df.columns else ("Return" if "Return" in df.columns else None)))
    if year_col is None or ret_col is None:
        raise RuntimeError(f"找不到必要欄位（年份/年報酬）。請檢查 sheet: {sheet_name} 的欄位：{list(df.columns)}")
    # 只保留必要欄位並回傳
    keep = []
    if round_col:
        keep.append(round_col)
    keep += [year_col, ret_col]
    df2 = df[keep].copy()
    # 統一欄位名稱
    rename_map = {}
    if round_col:
        rename_map[round_col] = "輪次"
    rename_map[year_col] = "年份"
    rename_map[ret_col]  = "年報酬（%）"
    df2.rename(columns=rename_map, inplace=True)
    # 轉型
    df2["年份"] = df2["年份"].astype(int)
    df2["年報酬（%）"] = pd.to_numeric(df2["年報酬（%）"], errors="coerce")
    return df2

def nav_per_round(df_yearly: pd.DataFrame, outdir: Path):
    rounds = sorted(df_yearly["輪次"].dropna().unique().astype(int))
    for r in rounds:
        sub = df_yearly[df_yearly["輪次"] == r].sort_values("年份")
        if sub.empty:
            continue
        returns = sub["年報酬（%）"].fillna(0) / 100.0
        nav = (1 + returns).cumprod()
        nav.index = pd.to_datetime(sub["年份"].astype(str) + "-12-31")
        nav_df = nav.rename("nav").to_frame()
        outpath = outdir / f"task1_nav_round{int(r):02d}.csv"
        nav_df.to_csv(outpath, index_label="date")
    return rounds

def nav_aggregate(df_yearly: pd.DataFrame, outdir: Path, method: str = "mean"):
    # 若同一年有多個輪次，依 method 決定合併方式
    if method == "mean":
        yearly = df_yearly.groupby("年份")["年報酬（%）"].mean().sort_index()
    elif method == "last":
        # 取該年輪次最大的那一筆（視為最後一輪）
        yearly = df_yearly.sort_values(["輪次"]).groupby("年份")["年報酬（%）"].last().sort_index()
    else:
        raise ValueError("agg method must be 'mean' or 'last'")
    returns = yearly.fillna(0) / 100.0
    nav = (1 + returns).cumprod()
    nav.index = pd.to_datetime(yearly.index.astype(str) + "-12-31")
    nav_df = nav.rename("nav").to_frame()
    outpath = outdir / "task1_nav_aggregate.csv"
    nav_df.to_csv(outpath, index_label="date")
    return outpath

def main(argv=None):
    parser = argparse.ArgumentParser(description="Convert task1_results.xlsx yearly returns to NAV CSVs")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT, help="task1_results.xlsx 路徑")
    parser.add_argument("--outdir", "-o", default=DEFAULT_OUTDIR, help="輸出資料夾")
    parser.add_argument("--agg-method", "-m", choices=["mean", "last"], default="mean", help="合併多輪同年結果的方法")
    parser.add_argument("--no-per-round", action="store_true", help="不要輸出每輪 NAV（只輸出 aggregate）")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[Error] 找不到輸入檔案：{input_path}", file=sys.stderr)
        sys.exit(1)

    sheet = find_sheet_name(input_path)
    print(f"[Info] 使用 sheet: {sheet}")
    df_yearly = load_yearly_sheet(input_path, sheet_name=sheet)

    # 若沒有輪次欄位，先嘗試填入輪次 = 1（單一輪）
    if "輪次" not in df_yearly.columns:
        df_yearly["輪次"] = 1

    if not args.no_per_round:
        rounds = nav_per_round(df_yearly, outdir)
        print(f"[Info] 已輸出每輪 NAV（共 {len(rounds)} 輪）到 {outdir}")

    agg_path = nav_aggregate(df_yearly, outdir, method=args.agg_method)
    print(f"[Info] 已輸出 aggregate NAV 到 {agg_path}")

if __name__ == "__main__":
    main()
