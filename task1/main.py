import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data, clean_data, split_train_test, get_X_y
from utils       import rank_features
from model       import ID3DecisionTree
from portfolio   import (
    select_stocks,
    calc_annual_returns,
    print_performance,
)


# ──────────────────────────────────────────
# 設定區（修改這裡就好）
# ──────────────────────────────────────────

DATA_PATH  = 'top200.xlsx'

# Walk-Forward 設定：自動從 START_YEAR 跑到 END_YEAR
# 每輪訓練期擴一年，測試期為剩下所有年份
START_YEAR = 1997   # 訓練期起始年
END_YEAR   = 2008   # 資料最末年

TREE_PARAMS = {
    'max_depth'  : 5,
    'min_samples': 11,
    'min_gain'   : 1e-4,
}

DROP_DATES = [200912]


# ──────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────

def main():
    print("=" * 50)
    print("  Task 1 — ID3 決策樹選股（Walk-Forward）")
    print("=" * 50)

    # Step 1：載入與清理資料（只做一次）
    print("\n【Step 1】載入與清理資料")
    df = load_data(DATA_PATH)
    df = clean_data(df, drop_dates=DROP_DATES)

    all_summary, all_yearly, all_detail = [], [], []
    total_rounds = END_YEAR - START_YEAR

    for rnd, train_end in enumerate(range(START_YEAR, END_YEAR), start=1):
        test_years = list(range(train_end + 1, END_YEAR + 1))
        print(f"\n{'='*50}")
        print(f"  Round {rnd:02d}/{total_rounds}"
              f"  訓練：{START_YEAR}~{train_end}  測試：{test_years[0]}~{test_years[-1]}")
        print(f"{'='*50}")

        # Step 2：切分
        train_df, test_df = split_train_test(df, test_years=test_years)
        X_train, y_train  = get_X_y(train_df)
        X_test,  y_test   = get_X_y(test_df)

        # Step 3：特徵資訊增益分析
        print("\n【Step 3】特徵資訊增益分析（訓練集）")
        feature_ranking = rank_features(X_train, y_train, top_n=8)
        print(feature_ranking.to_string(index=False))

        # Step 4：訓練決策樹
        print("\n【Step 4】訓練 ID3 決策樹")
        model = ID3DecisionTree(**TREE_PARAMS)
        model.fit(X_train, y_train)
        model.print_tree(max_display_depth=3)

        print("\n--- 特徵重要性（訓練後）---")
        print(model.get_feature_importance().to_string(index=False))

        # Step 5：預測
        print("\n【Step 5】測試集預測")
        y_pred    = model.predict(X_test)
        train_acc = (model.predict(X_train) == y_train.values).mean()
        test_acc  = (y_pred == y_test.values).mean()
        print(f"  訓練集準確率：{train_acc:.4f}")
        print(f"  測試集準確率：{test_acc:.4f}")

        # Step 6：選股與績效
        print("\n【Step 6】選股與績效評估")
        selected       = select_stocks(y_pred, test_df)
        annual_returns = calc_annual_returns(selected)
        metrics        = print_performance(annual_returns)

        cumulative = (1 + annual_returns / 100).cumprod()
        cumulative = (cumulative - 1) * 100

        # 收集各輪結果
        all_summary.append({
            '輪次'           : rnd,
            '訓練期'         : f"{START_YEAR}~{train_end}",
            '測試期'         : f"{test_years[0]}~{test_years[-1]}",
            '測試年數'       : len(annual_returns),
            'CAGR（%）'     : round(metrics['cagr'], 4),
            '累積報酬（%）'  : round(metrics['cumulative_return'], 4),
            '最大回撤（%）'  : round(metrics['max_drawdown'], 4),
            '夏普比率'       : round(metrics['sharpe_ratio'], 4),
            '年均報酬（%）'  : round(metrics['annual_mean'], 4),
            '報酬標準差（%）': round(metrics['annual_std'], 4),
        })

        for year, ret in annual_returns.items():
            all_yearly.append({
                '輪次'         : rnd,
                '訓練期'       : f"{START_YEAR}~{train_end}",
                '年份'         : year,
                '年報酬（%）'  : round(ret, 4),
                '累積報酬（%）': round(cumulative[year], 4),
            })

        detail = selected[['證券代碼', '簡稱', '年月', 'Return', 'weight']].copy()
        detail.insert(0, '訓練期', f"{START_YEAR}~{train_end}")
        detail.insert(0, '輪次', rnd)
        detail.columns = ['輪次', '訓練期', '證券代碼', '簡稱', '年月', '實際報酬（%）', '資金權重']
        all_detail.append(detail)

    # Step 7：輸出 Excel
    print("\n【Step 7】輸出 Excel 報告")
    excel_path = 'task1_results.xlsx'

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        pd.DataFrame(all_summary).to_excel(writer, sheet_name='各輪績效摘要', index=False)
        pd.DataFrame(all_yearly).to_excel(writer,  sheet_name='各輪逐年報酬', index=False)
        pd.concat(all_detail, ignore_index=True).to_excel(writer, sheet_name='各輪選股明細', index=False)

    print(f"  Excel 已儲存至：{excel_path}")
    print(f"  內含三個 Sheet：各輪績效摘要 / 各輪逐年報酬 / 各輪選股明細")


# ──────────────────────────────────────────
# 執行
# ──────────────────────────────────────────

if __name__ == '__main__':
    main()