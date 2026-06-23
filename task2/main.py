# main.py
import openpyxl
from data_loader import DataLoader
from temporal_test import TemporalTest

def main():
    print("正在讀取 top200.xlsx 財報資料集並進行預處理...")
    excel_path = "top200.xlsx"
    
    loader = DataLoader(excel_path)
    data_by_year = loader.load_and_clean_data()
    
    print("\n開始執行 11 輪時間序列驗證 (Temporal Validation)...")
    all_rounds = TemporalTest.run_validation(data_by_year)
    
    # 1. 在終端機打印精美結果表（方便即時查看）
    print("\n" + "="*115)
    print(f"| {'輪次':<2} | {'訓練期':<9} | {'測試期':<9} | {'測試年數':<4} | {'CAGR(%)':<7} | {'累積報酬(%)':<9} | {'最大回撤(%)':<9} | {'夏普比率':<6} | {'年均報酬(%)':<9} | {'標準差(%)':<7} |")
    print("="*115)
    
    for r in all_rounds:
        print(f"| {r['round']:<4} "
              f"| {r['train_period']:<12} "
              f"| {r['test_period']:<12} "
              f"| {r['test_years_count']:<8} "
              f"| {r['cagr']*100:>7.2f}% "
              f"| {r['cum_return']*100:>12.2f}% "
              f"| {-abs(r['max_dd'])*100:>12.2f}% "  # 確保最大回撤顯示為負號
              f"| {r['sharpe']:>8.4f} "
              f"| {r['mean_return']*100:>12.2f}% "
              f"| {r['std_return']*100:>8.2f}% |")
    print("="*115 + "\n")
    
    # 2. 自動將資料寫入 Excel 檔案：task2_results.xlsx
    output_filename = "task2_results.xlsx"
    print(f"正在將量化成果導出至 {output_filename} ...")
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "各輪績效摘要"
    
    # 寫入與 Task 1 一模一樣的欄位表頭
    headers = ["輪次", "訓練期", "測試期", "測試年數", "CAGR（%）", "累積報酬（%）", "最大回撤（%）", "夏普比率", "年均報酬（%）", "報酬標準差（%）"]
    ws.append(headers)
    
    # 逐列寫入數據（百分比與小數點格式皆與原始資料格式完全對齊）
    for r in all_rounds:
        row_data = [
            r['round'],
            r['train_period'],
            r['test_period'],
            r['test_years_count'],
            round(r['cagr'] * 100, 4),           # 轉為百分比數值 (如 8.7277)
            round(r['cum_return'] * 100, 4),     # 轉為百分比數值
            round(-abs(r['max_dd']) * 100, 4),   # 確保為負數百分比
            round(r['sharpe'], 4),
            round(r['mean_return'] * 100, 4),    # 轉為百分比數值
            round(r['std_return'] * 100, 4)      # 轉為百分比數值
        ]
        ws.append(row_data)
        
    # 儲存 Excel 活頁簿
    wb.save(output_filename)
    print(f"導出成功！已在當前目錄下生成：{output_filename}")
    print("你可以直接用 Excel 開啟該檔案，並將其成果完美貼入期末報告第 4 節。")

if __name__ == "__main__":
    main()