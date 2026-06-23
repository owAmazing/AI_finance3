# data_loader.py
import openpyxl

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_and_clean_data(self):
        data_by_year = {}
        
        # 1. 讀取 Excel 檔案
        wb = openpyxl.load_workbook(self.file_path, data_only=True)
        # 預設讀取第一個工作表
        sheet = wb.active 
        
        # 2. 獲取所有行，並分離出表頭 (第一行)
        rows = list(sheet.iter_rows(values_only=True))
        if not rows:
            return data_by_year
            
        headers = rows[0]
        data_rows = rows[1:]
        
        # 3. 開始解析每一行資料
        for row in data_rows:
            # 確保該行不是空白行，且至少有基本欄位
            if not row or len(row) < 3 or row[0] is None:
                continue
            
            # 年月欄位通常在第 3 欄 (Index 2)
            year_month = str(row[2]).strip()
            
            # 專案規範：剔除 200912 的資料
            if year_month == '200912':
                continue
            
            try:
                year = int(year_month[:4])
                
                # 關鍵修正：將 Return 除以 100，避免複利計算時數值爆表
                raw_return = (float(row[-2]) / 100.0) if row[-2] is not None else 0.0
                label = int(row[-1]) if row[-1] is not None else -1
                
                features = []
                for val in row[3:19]:
                    features.append(float(val) if val is not None else 0.0)
                    
            except (ValueError, TypeError, IndexError):
                # 預防資料有缺漏、非數字或格式損壞，直接跳過該行確保回測不中斷
                continue
            
            if year not in data_by_year:
                data_by_year[year] = []
                
            data_by_year[year].append({
                'code': str(row[0]).strip(),
                'name': str(row[1]).strip() if row[1] else "",
                'features': features,
                'return': raw_return,
                'label': label
            })
            
        return data_by_year