import numpy as np
import pandas as pd
from data_loader import DATE_COL, RETURN_COL


# ──────────────────────────────────────────
# 1. 選股
# ──────────────────────────────────────────

def select_stocks(
    predictions : np.ndarray,
    test_df     : pd.DataFrame,
) -> pd.DataFrame:
    """
    從測試集中挑出預測為 1 的股票，每年資金平均分配。

    Parameters
    ----------
    predictions : np.ndarray，model.predict() 的輸出（1 或 -1）
    test_df     : pd.DataFrame，測試集原始資料（需含 年月、Return）

    Returns
    -------
    pd.DataFrame，只保留預測為 1 的股票，新增欄位：
        weight — 該年每支股票的資金比例（等權重）
    """
    df = test_df.copy()
    df['prediction'] = predictions

    selected = df[df['prediction'] == 1].copy()

    # 每年計算等權重
    yearly_counts = selected.groupby(DATE_COL)['prediction'].transform('count')
    selected['weight'] = 1.0 / yearly_counts

    print(f"[select_stocks] 共選出 {len(selected)} 筆"
          f"（預測為 1 / 總測試筆數 {len(df)}）")

    # 印出每年選股數量
    yearly_summary = selected.groupby(DATE_COL).size().reset_index(name='選股數')
    print(yearly_summary.to_string(index=False))

    return selected


# ──────────────────────────────────────────
# 2. 計算每年投資組合報酬
# ──────────────────────────────────────────

def calc_annual_returns(selected: pd.DataFrame) -> pd.Series:
    """
    計算每年投資組合的加權報酬率（等權重平均）。

    公式：年報酬 = Σ (weight_i × Return_i)

    Parameters
    ----------
    selected : pd.DataFrame，select_stocks() 的輸出

    Returns
    -------
    pd.Series，index 為年份（int），值為報酬率（%）
    """
    # 年月 YYYYMM → 年份 YYYY
    selected = selected.copy()
    selected['year'] = selected[DATE_COL] // 100

    annual = (
        selected
        .groupby('year')
        .apply(lambda g: (g['weight'] * g[RETURN_COL]).sum())
        .rename('annual_return')
    )

    print("\n=== 每年投資組合報酬 ===")
    for year, ret in annual.items():
        print(f"  {year} 年：{ret:+.2f}%")

    return annual


# ──────────────────────────────────────────
# 3. 績效指標
# ──────────────────────────────────────────

def calc_cumulative_return(annual_returns: pd.Series) -> pd.Series:
    """
    計算逐年累積報酬率。

    公式：Cum_t = Π (1 + r_i/100) - 1，以百分比回傳

    Returns
    -------
    pd.Series，index 為年份，值為累積報酬率（%）
    """
    growth = (1 + annual_returns / 100).cumprod()
    cumulative = (growth - 1) * 100
    return cumulative


def calc_cagr(annual_returns: pd.Series) -> float:
    """
    計算年化複合成長率 CAGR。

    公式：CAGR = (終值/初值)^(1/N) - 1

    Returns
    -------
    float，CAGR（%）
    """
    n = len(annual_returns)
    if n == 0:
        return 0.0
    total_growth = (1 + annual_returns / 100).prod()
    cagr = (total_growth ** (1 / n) - 1) * 100
    return float(cagr)


def calc_max_drawdown(annual_returns: pd.Series) -> float:
    """
    計算最大回撤（Maximum Drawdown）。

    定義：從歷史高點到最低點的最大跌幅。
    公式：MDD = (谷值 - 峰值) / 峰值 × 100%

    Returns
    -------
    float，最大回撤（%，負數）
    """
    growth = (1 + annual_returns / 100).cumprod()
    peak   = growth.cummax()
    dd     = (growth - peak) / peak * 100
    return float(dd.min())


def calc_sharpe_ratio(
    annual_returns  : pd.Series,
    risk_free_rate  : float = 2.0,   # 無風險利率（%），預設 2%
) -> float:
    """
    計算夏普比率（Sharpe Ratio）。

    公式：Sharpe = (平均超額報酬) / (報酬標準差)
    超額報酬 = 年報酬 - 無風險利率

    Parameters
    ----------
    annual_returns : pd.Series，每年報酬率（%）
    risk_free_rate : float，無風險利率（%），預設 2%

    Returns
    -------
    float，夏普比率（無單位）
    """
    excess = annual_returns - risk_free_rate
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std())


# ──────────────────────────────────────────
# 4. 績效報告
# ──────────────────────────────────────────

def print_performance(annual_returns: pd.Series):
    """
    印出完整績效報告。

    Parameters
    ----------
    annual_returns : pd.Series，calc_annual_returns() 的輸出
    """
    cumulative = calc_cumulative_return(annual_returns)
    cagr       = calc_cagr(annual_returns)
    mdd        = calc_max_drawdown(annual_returns)
    sharpe     = calc_sharpe_ratio(annual_returns)

    print("\n" + "=" * 40)
    print("         績效報告")
    print("=" * 40)
    print(f"  測試年數          : {len(annual_returns)} 年")
    print(f"  CAGR（年化報酬）  : {cagr:+.2f}%")
    print(f"  累積報酬          : {cumulative.iloc[-1]:+.2f}%")
    print(f"  最大回撤          : {mdd:.2f}%")
    print(f"  夏普比率          : {sharpe:.4f}")
    print(f"  年均報酬          : {annual_returns.mean():+.2f}%")
    print(f"  報酬標準差        : {annual_returns.std():.2f}%")
    print("-" * 40)
    print("  逐年累積報酬：")
    for year, cum in cumulative.items():
        print(f"    {year} 年：{cum:+.2f}%")
    print("=" * 40)

    return {
        'cagr'            : cagr,
        'cumulative_return': float(cumulative.iloc[-1]),
        'max_drawdown'    : mdd,
        'sharpe_ratio'    : sharpe,
        'annual_mean'     : float(annual_returns.mean()),
        'annual_std'      : float(annual_returns.std()),
    }


# ──────────────────────────────────────────
# 快速測試
# ──────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    from data_loader import load_data, clean_data, split_train_test, get_X_y
    from model import ID3DecisionTree

    # 載入資料
    df            = load_data('/mnt/user-data/uploads/top200.xlsx')
    df            = clean_data(df)
    train_df, test_df = split_train_test(df, test_years=[2006, 2007, 2008])
    X_train, y_train  = get_X_y(train_df)
    X_test,  y_test   = get_X_y(test_df)

    # 訓練（模型固定）
    model = ID3DecisionTree(max_depth=5, min_samples=10, min_gain=1e-4)
    model.fit(X_train, y_train)

    # 預測
    y_pred = model.predict(X_test)

    # 選股
    selected = select_stocks(y_pred, test_df)

    # 每年報酬
    annual_returns = calc_annual_returns(selected)

    # 績效報告
    print_performance(annual_returns)