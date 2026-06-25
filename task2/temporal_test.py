# temporal_test.py
from model import RandomForestModel
from portfolio import PortfolioManager
from performance_measurement import PerformanceMeasurement

class TemporalTest:
    @staticmethod
    def run_validation(data_by_year):
        all_rounds_results = []
        
        # 💡 【改動 1】動態抓取新資料庫中的所有可用年份 (例如得到 [2010, 2011, 2012, 2013, 2014, 2015])
        available_years = sorted(list(data_by_year.keys()))
        if not available_years:
            print("錯誤：輸入的資料中沒有找到有效的年份數據！")
            return []
            
        start_year = available_years[0]   # 自動變成 2010
        end_year = available_years[-1]     # 自動變成 2015
        
        # 💡 【改動 2】動態計算總共可以滾動幾輪 
        # 為了留至少一年當測試期，總輪數 = 總年數 - 1 (在這份資料中會跑 5 輪)
        total_rounds = end_year - start_year
        
        # 滾動回測：
        # 輪次 1: 2010 訓, 2011~2015 測
        # 輪次 2: 2010~2011 訓, 2012~2015 測
        # ...
        # 輪次 5: 2010~2014 訓, 2015 測
        for round_idx in range(1, total_rounds + 1):
            last_train_year = start_year + round_idx - 1
            
            # 1. 動態配置該輪的訓練年份與測試年份
            train_years = list(range(start_year, last_train_year + 1))
            test_years = list(range(last_train_year + 1, end_year + 1))
            
            # 組合該輪的訓練資料集 (合併多個年份的股票)
            train_data = []
            for y in train_years:
                train_data.extend(data_by_year.get(y, []))
                
            # 組合該輪的測試資料集 (依年份分組保存)
            test_data_by_year = {y: data_by_year[y] for y in test_years if y in data_by_year}
            
            # 防呆機制：如果沒有測試資料（例如超出範圍），就跳出
            if not train_data or not test_data_by_year:
                continue
                
            # 2. 為該輪初始化並訓練獨立的隨機森林模型 (50 棵樹)
            rf_model = RandomForestModel(num_trees=50)
            rf_model.train(train_data)
            
            # 3. 執行該輪的年度投組回測，計算淨值曲線
            yearly_returns, equity_curve = PortfolioManager.calculate_portfolio_returns(test_data_by_year, rf_model)
            
            # 4. 計算 4 大財務效能指標
            returns_list = list(yearly_returns.values())
            years_count = len(test_years)
            
            cum_return = PerformanceMeasurement.cumulative_return(equity_curve)
            ann_return = PerformanceMeasurement.annualized_return(equity_curve, years_count)
            max_dd = PerformanceMeasurement.max_drawdown(equity_curve)
            sharpe = PerformanceMeasurement.sharpe_ratio(returns_list, risk_free_rate=0.015)
            
            # 計算額外統計欄位，以對齊 Task 1 的表格
            mean_return = sum(returns_list) / len(returns_list) if returns_list else 0.0
            # 呼叫底層計算標準差
            from utils import Utils
            std_return = Utils.std_dev(returns_list)
            
            # 5. 紀錄該輪的結果摘要
            round_summary = {
                'round': round_idx,
                'train_period': f"{train_years[0]}~{train_years[-1]}",
                'test_period': f"{test_years[0]}~{test_years[-1]}",
                'test_years_count': years_count,
                'cagr': ann_return,               # CAGR 欄位
                'cum_return': cum_return,         # 累積報酬
                'max_dd': max_dd,                 # 最大回撤
                'sharpe': sharpe,                 # 夏普比率
                'mean_return': mean_return,       # 年均報酬
                'std_return': std_return          # 報酬標準差
            }
            
            all_rounds_results.append(round_summary)
            print(f"完成 輪次 {round_idx} 回測 ({round_summary['train_period']} 訓練 -> {round_summary['test_period']} 測試)")
            
        return all_rounds_results