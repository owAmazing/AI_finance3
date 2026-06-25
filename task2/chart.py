import pandas as pd
import matplotlib.pyplot as plt
import os

# 1. 設定專業圖表風格與字型（防止中文變亂碼）
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = ['Microsoft JhengHei', 'Arial', 'sans-serif'] # 支援 Windows 微軟正黑體
plt.rcParams['axes.unicode_minus'] = False # 正常顯示負號

# 設定可能出現的兩種檔案名稱
excel_file = "task2_testing_results.xlsx"
csv_file = "task2_testing_results.xlsx - 各輪績效摘要.csv"
df = None

# 2. 自動判斷並讀取數據
if os.path.exists(excel_file):
    try:
        df = pd.read_excel(excel_file)
    except:
        pass

if df is None and os.path.exists(csv_file):
    for enc in ['utf-8-sig', 'cp950', 'big5', 'utf-16']:
        try:
            df = pd.read_csv(csv_file, encoding=enc)
            break
        except:
            continue

if df is None:
    print("❌ 錯誤：找不到資料檔案。請確保 task2_results.xlsx 或其 CSV 檔在同個資料夾。")
    os._exit(0)

# 清洗欄位名稱並建立 X 軸標籤
df.columns = df.columns.str.strip()
df['X_label'] = df.apply(lambda r: f"R{int(r['輪次'])}\n{r['測試期']}", axis=1)

# 判斷資料內使用的是「年均報酬（%）」還是「年均報酬率（%）」
avg_return_col = None
for col in ['年均報酬（%）', '年均報酬率（%）', '年均報酬', '年均報酬率']:
    if col in df.columns:
        avg_return_col = col
        break

# 如果資料裡找不到對應中文，就拿原來的 CAGR 欄位來代替
if avg_return_col is None:
    avg_return_col = 'CAGR（%）' if 'CAGR（%）' in df.columns else 'CAGR'

# 建立獨立圖表儲存目錄
output_dir = "output_testing_charts"
os.makedirs(output_dir, exist_ok=True)

# 3. 定義 4 個效能指標的配置 (圖檔名, 原始欄位名, 顯示標籤名稱, 顏色, 標記點)
metrics_config = [
    ("cumulative_return.png", "累積報酬（%）", "隨機森林選股策略：各輪次累積報酬率 (%)", "#1f77b4", "o"),
    ("annualized_return.png", avg_return_col, "隨機森林選股策略：各輪次年均報酬率 (%)", "#2ca02c", "s"),
    ("max_drawdown.png", "最大回撤（%）", "隨機森林選股策略：各輪次最大回撤 MDD (%)", "#d62728", "v"),
    ("sharpe_ratio.png", "夏普比率", "隨機森林選股策略：各輪次夏普比率 (Sharpe Ratio)", "#ff7f0e", "d")
]

# 4. 迴圈繪製 4 張獨立圖表
print("開始生成個別效能指標圖表...")
for filename, col_name, title, color, marker in metrics_config:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    
    # 決定圖例與軸線要顯示的中文標籤
    label_name = "年均報酬率 (%)" if col_name == avg_return_col else col_name
    
    # 畫出單一指標折線圖
    ax.plot(df['X_label'], df[col_name], marker=marker, linewidth=2.5, color=color, label=label_name)
    
    # 圖表細節客製化
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel('時間序列驗證輪次 (Round) 與盲測期間', fontsize=10, fontweight='bold', labelpad=8)
    ax.set_ylabel(label_name, fontsize=10, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, shadow=True)
    
    # 針對最大回撤（負值）調整 Y 軸方向或樣式以利閱讀
    if col_name == "最大回撤（%）":
        ax.set_ylim(ax.get_ylim()[0] - 5, 0) # 讓上限剛好停在 0%
        
    plt.tight_layout()
    
    # 儲存至資料夾
    full_path = os.path.join(output_dir, filename)
    plt.savefig(full_path, bbox_inches='tight')
    plt.close() # 關閉畫布釋放記憶體
    print(f"🎉 成功生成圖表: {full_path}")

print(f"\n✨ 修正完畢！4 張獨立績效圖表（內含中文「年均報酬率」）已儲存至: ./{output_dir}/")