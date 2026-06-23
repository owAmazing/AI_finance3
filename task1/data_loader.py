import pandas as pd


# 所有財務指標欄位（16個輸入特徵）
FEATURE_COLS = [
    '市值(百萬元)',
    '收盤價(元)_年',
    'Unknown masked parameter',
    '股價淨值比',
    '股價營收比',
    'M淨值報酬率─稅後',
    '資產報酬率ROA',
    '營業利益率OPM',
    '利潤邊際NPM',
    '負債/淨值比',
    'M流動比率',
    'M速動比率',
    'M存貨週轉率 (次)',
    'M應收帳款週轉次',
    'M營業利益成長率',
    'M稅後淨利成長率',
]

LABEL_COL   = 'ReturnMean_year_Label'
RETURN_COL  = 'Return'
DATE_COL    = '年月'


def load_data(path: str) -> pd.DataFrame:
    """
    讀入 Excel 檔案，自動判斷副檔名。
    支援 .xlsx / .xls / .csv。

    Parameters
    ----------
    path : str
        檔案路徑，例如 'data/top200.xlsx'

    Returns
    -------
    pd.DataFrame
        原始資料（尚未清理）
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.xls'):
        df = pd.read_excel(path, engine='xlrd')
    else:
        df = pd.read_excel(path)

    print(f"[load_data] 讀入 {len(df)} 筆資料，欄位數：{len(df.columns)}")
    return df


def clean_data(
    df: pd.DataFrame,
    drop_dates: list[int] | None = None,
) -> pd.DataFrame:
    """
    清理資料：
      1. 刪除指定年月的資料（預設刪除 200912）
      2. 刪除特徵欄位全為 NaN 的列
      3. 對特徵欄位用各年度中位數填補剩餘缺值（避免用到跨年資訊）

    Parameters
    ----------
    df : pd.DataFrame
        load_data() 回傳的原始資料

    drop_dates : list[int] | None
        要刪除的年月清單，格式為整數，例如 [200912]。
        傳入 [] 或 None 表示不刪除任何年月。
        預設刪除 [200912]（作業規定）。

    Returns
    -------
    pd.DataFrame
        清理後的資料（index 重置）

    Examples
    --------
    # 使用預設（刪除 200912）
    df_clean = clean_data(df)

    # 不刪除任何年月（例如 task3 用 2026 之後的資料）
    df_clean = clean_data(df, drop_dates=[])

    # 刪除多個年月
    df_clean = clean_data(df, drop_dates=[200912, 202612])
    """
    if drop_dates is None:
        drop_dates = [200912]

    original_len = len(df)

    # 1. 刪除指定年月
    if drop_dates:
        df = df[~df[DATE_COL].isin(drop_dates)].copy()
        print(f"[clean_data] 刪除年月 {drop_dates} 後剩 {len(df)} 筆"
              f"（移除 {original_len - len(df)} 筆）")
    else:
        print(f"[clean_data] 未刪除任何年月，共 {len(df)} 筆")

    # 2. 刪除特徵欄位全為 NaN 的列
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS, how='all')
    if len(df) < before:
        print(f"[clean_data] 刪除全空特徵列 {before - len(df)} 筆")

    # 3. 用各年度中位數填補剩餘缺值（按 DATE_COL 分組，避免跨年資訊洩漏）
    missing_before = df[FEATURE_COLS].isna().sum().sum()
    if missing_before > 0:
        df[FEATURE_COLS] = df.groupby(DATE_COL)[FEATURE_COLS].transform(
            lambda x: x.fillna(x.median())
        )
        missing_after = df[FEATURE_COLS].isna().sum().sum()
        print(f"[clean_data] 填補缺值：{missing_before} → {missing_after} 個缺值")

    df = df.reset_index(drop=True)
    print(f"[clean_data] 清理完成，最終 {len(df)} 筆")
    return df


def split_train_test(
    df: pd.DataFrame,
    test_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按年份切分訓練集與測試集。
    訓練集 = test_years 最早年份之前的所有資料。
    測試集 = test_years 指定年份的資料。

    年月欄位格式為 YYYYMM（如 199712），
    比較時取前四碼（YYYY）作為年份。

    Parameters
    ----------
    df : pd.DataFrame
        clean_data() 回傳的清理後資料

    test_years : list[int]
        測試年份清單，例如 [2006, 2007, 2008]

    Returns
    -------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame

    Examples
    --------
    # 用 2006~2008 當測試期
    train_df, test_df = split_train_test(df, test_years=[2006, 2007, 2008])
    """
    # 從 年月（YYYYMM）取出 YYYY
    year_series = df[DATE_COL] // 100

    min_test_year = min(test_years)

    train_df = df[year_series < min_test_year].copy().reset_index(drop=True)
    test_df  = df[year_series.isin(test_years)].copy().reset_index(drop=True)

    print(f"[split_train_test] 訓練集：{len(train_df)} 筆"
          f"（{year_series[year_series < min_test_year].min()}"
          f" ~ {min_test_year - 1} 年）")
    print(f"[split_train_test] 測試集：{len(test_df)} 筆"
          f"（{test_years} 年）")

    return train_df, test_df


def get_X_y(df: pd.DataFrame):
    """
    從 DataFrame 取出特徵矩陣 X 和標籤 y。

    Returns
    -------
    X : pd.DataFrame  — shape (n, 16)
    y : pd.Series     — 值為 1 或 -1
    """
    X = df[FEATURE_COLS]
    y = df[LABEL_COL]
    return X, y


# ── 快速測試用 ──────────────────────────────────────────────
if __name__ == '__main__':
    df_raw   = load_data('top200.xlsx')
    df_clean = clean_data(df_raw)                          # 預設刪 200912
    # df_clean = clean_data(df_raw, drop_dates=[])         # task3 用：不刪任何年月
    # df_clean = clean_data(df_raw, drop_dates=[202612])   # task3 用：刪 2026 的資料

    train_df, test_df = split_train_test(df_clean, test_years=[2006, 2007, 2008])

    X_train, y_train = get_X_y(train_df)
    X_test,  y_test  = get_X_y(test_df)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test  shape: {X_test.shape}")
    print(f"y_train 分佈:\n{y_train.value_counts()}")