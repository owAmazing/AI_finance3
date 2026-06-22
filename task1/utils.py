import numpy as np
import pandas as pd


# ──────────────────────────────────────────
# 1. 熵計算
# ──────────────────────────────────────────

def calc_entropy(y: np.ndarray | pd.Series) -> float:
    """
    計算標籤序列的熵 H(S)。
    公式：H = Σ p_i * log2(1/p_i)

    Parameters
    ----------
    y : array-like，值為 1 或 -1

    Returns
    -------
    float：熵值（0 ~ 1 之間）

    Examples
    --------
    >>> calc_entropy(np.array([1, 1, -1, -1]))
    1.0
    >>> calc_entropy(np.array([1, 1, 1, 1]))
    0.0
    """
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return 0.0

    classes, counts = np.unique(y, return_counts=True)
    probs = counts / n

    # p=0 的情況 log2(0) 定義為 0
    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    return float(entropy)


# ──────────────────────────────────────────
# 2. 單一特徵的最佳分裂點搜尋
# ──────────────────────────────────────────

def calc_best_split(
    X_col: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> dict:
    """
    對單一連續特徵，搜尋所有候選分裂點，
    找出資訊增益最大的閾值。

    候選分裂點 = 所有相鄰不同值的中間點。
    例如排序後值為 [0.12, 0.18, 0.35]
    → 候選點為 [0.15, 0.265]

    Parameters
    ----------
    X_col : array-like，單一特徵的數值序列
    y     : array-like，對應的標籤（1 或 -1）

    Returns
    -------
    dict with keys:
        best_threshold : float  — 最佳分裂閾值
        best_gain      : float  — 對應的資訊增益
        n_candidates   : int    — 嘗試過的候選分裂點數量
    """
    X_col = np.asarray(X_col, dtype=float)
    y     = np.asarray(y)
    n     = len(y)

    parent_entropy = calc_entropy(y)

    # 取所有唯一值排序後，計算相鄰中間點作為候選分裂點
    unique_vals = np.unique(X_col[~np.isnan(X_col)])
    if len(unique_vals) < 2:
        return {'best_threshold': None, 'best_gain': 0.0, 'n_candidates': 0}

    candidates = (unique_vals[:-1] + unique_vals[1:]) / 2

    best_gain      = -np.inf
    best_threshold = None

    for threshold in candidates:
        left_mask  = X_col <= threshold
        right_mask = X_col >  threshold

        n_left  = left_mask.sum()
        n_right = right_mask.sum()

        if n_left == 0 or n_right == 0:
            continue

        # 加權熵
        weighted_entropy = (
            (n_left  / n) * calc_entropy(y[left_mask]) +
            (n_right / n) * calc_entropy(y[right_mask])
        )

        gain = parent_entropy - weighted_entropy

        if gain > best_gain:
            best_gain      = gain
            best_threshold = threshold

    return {
        'best_threshold': best_threshold,
        'best_gain'     : float(best_gain) if best_gain != -np.inf else 0.0,
        'n_candidates'  : len(candidates),
    }


# ──────────────────────────────────────────
# 3. 所有特徵排序
# ──────────────────────────────────────────

def rank_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    對所有特徵跑 calc_best_split，依資訊增益由大到小排序。

    Parameters
    ----------
    X     : pd.DataFrame，特徵矩陣（16欄）
    y     : pd.Series，標籤（1 或 -1）
    top_n : int | None，只回傳前 N 個特徵，None 表示全部

    Returns
    -------
    pd.DataFrame，欄位為：
        feature        — 特徵名稱
        best_threshold — 最佳分裂閾值
        info_gain      — 資訊增益
        n_candidates   — 嘗試的候選分裂點數

    Examples
    --------
    >>> result = rank_features(X_train, y_train, top_n=5)
    >>> print(result)
    """
    records = []
    for col in X.columns:
        result = calc_best_split(X[col].values, y.values)
        records.append({
            'feature'       : col,
            'best_threshold': result['best_threshold'],
            'info_gain'     : result['best_gain'],
            'n_candidates'  : result['n_candidates'],
        })

    df_rank = (
        pd.DataFrame(records)
        .sort_values('info_gain', ascending=False)
        .reset_index(drop=True)
    )

    if top_n is not None:
        df_rank = df_rank.head(top_n)

    return df_rank


# ──────────────────────────────────────────
# 快速測試
# ──────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    from data_loader import load_data, clean_data, split_train_test, get_X_y

    df       = load_data('top200.xlsx')
    df       = clean_data(df)
    train_df, test_df = split_train_test(df, test_years=[2006, 2007, 2008])
    X_train, y_train  = get_X_y(train_df)

    # 測試熵
    print("=== 熵測試 ===")
    print(f"全部同類別 entropy: {calc_entropy(np.array([1,1,1,1])):.4f}  (應為 0.0)")
    print(f"五五波 entropy:     {calc_entropy(np.array([1,1,-1,-1])):.4f}  (應為 1.0)")
    print(f"訓練集 entropy:     {calc_entropy(y_train.values):.4f}")

    # 測試單一特徵
    print("\n=== 單一特徵最佳分裂點 ===")
    col = '資產報酬率ROA'
    result = calc_best_split(X_train[col].values, y_train.values)
    print(f"{col}:")
    print(f"  最佳閾值   : {result['best_threshold']:.6f}")
    print(f"  資訊增益   : {result['best_gain']:.6f}")
    print(f"  嘗試候選點 : {result['n_candidates']} 個")

    # 所有特徵排序
    print("\n=== 全部特徵資訊增益排序（Top 8）===")
    ranking = rank_features(X_train, y_train, top_n=8)
    print(ranking.to_string(index=False))