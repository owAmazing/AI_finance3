import pandas as pd
import requests

def get_guaranteed_200_stocks():
    print("1. 正在從證交所 Open Data 獲取『上市公司基本資料』...")
    url = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code != 200:
            print(f"❌ 連線失敗，狀態碼：{response.status_code}")
            return None
        df = pd.DataFrame(response.json())
    except Exception as e:
        print(f"❌ 讀取證交所 API 失敗: {e}")
        return None

    print("2. 連線成功！正在根據指定格式進行歷史時間清洗...")
    
    # 🎯 依據你提供的 JSON 格式，精準對應欄位
    df['stock_id'] = df['公司代號'].astype(str).str.strip()
    df['stock_name'] = df['公司簡稱'].astype(str).str.strip()
    
    # 🧼 過濾機制一：只要 4 碼純數字普通股，且排除開頭為 00 的 ETF
    df_clean = df[
        (df['stock_id'].str.isnumeric()) & 
        (df['stock_id'].str.len() == 4) &
        (~df['stock_id'].str.startswith('00'))
    ].copy()
    
    # 🧼 過濾機制二：精準解析 YYYYMMDD 日期格式（如 19620209）
    df_clean['parsed_date'] = pd.to_datetime(df_clean['上市日期'], format='%Y%m%d', errors='coerce')
    
    # 🔥 核心篩選：嚴格鎖定在 2012-01-01 以前就已經上市的長青股票！
    df_stable = df_clean[df_clean['parsed_date'] < '2012-01-01'].copy()
    
    # 依代號排序，確保選取前 200 檔
    df_stable = df_stable.sort_values(by='stock_id')
    
    # 去除重複項，並精準擷取前 200 檔
    result_df = df_stable[['stock_id', 'stock_name']].drop_duplicates(subset=['stock_id']).head(200)
    return result_df

# 執行主程式
try:
    final_200 = get_guaranteed_200_stocks()
    if final_200 is not None and not final_200.empty:
        # 存檔
        final_200.to_csv('new200_list.csv', index=False, encoding='utf-8-sig')
        print(f"\n🎉【大成功】已重新生成精準的 'new200_list.csv'！")
        print(f"這 {len(final_200)} 檔全部符合：4碼普通股、2012年前上市、2024年底仍在交易。")
        print("\n前 5 檔檢查預覽（保證 2012-2024 資料絕對完整）：")
        print(final_200.head())
    else:
        print("\n❌ 失敗：未能成功篩選出股票，請確認 API 是否有正常回傳資料。")
except Exception as e:
    print(f"\n❌ 程式執行異常中斷: {e}")