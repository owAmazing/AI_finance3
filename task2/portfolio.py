# portfolio.py
class PortfolioManager:
    @staticmethod
    def calculate_portfolio_returns(test_data_by_year, model):
        yearly_returns = {}
        equity_curve = [1.0]
        current_equity = 1.0
        
        for year in sorted(test_data_by_year.keys()):
            stocks = test_data_by_year[year]
            
            # 計算每檔股票在隨機森林中的信心分數
            stock_scores = []
            for s in stocks:
                score = model.predict_score(s['features'])
                stock_scores.append({'return': s['return'], 'score': score})
            
            # 優先篩選出模型明確看好 (score >= 0.5) 的股票
            selected_returns = [item['return'] for item in stock_scores if item['score'] >= 0.5]
            
            # 🔥 風險防禦機制：如果明確看好的股票不足 20 檔，則直接抓分數前 20 高的股票組成投組
            if len(selected_returns) < 20:
                stock_scores.sort(key=lambda x: x['score'], reverse=True)
                selected_returns = [item['return'] for item in stock_scores[:20]]
            
            # 計算等權重年度報酬
            year_return = sum(selected_returns) / len(selected_returns) if selected_returns else 0.0
            
            current_equity *= (1.0 + year_return)
            equity_curve.append(current_equity)
            yearly_returns[year] = year_return
            
        return yearly_returns, equity_curve