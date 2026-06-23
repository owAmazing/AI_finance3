# performance_measurement.py
from utils import Utils
import math

class PerformanceMeasurement:
    @staticmethod
    def cumulative_return(equity_curve):
        if not equity_curve:
            return 0.0
        return equity_curve[-1] - 1.0

    @staticmethod
    def annualized_return(equity_curve, years_count):
        if not equity_curve or years_count <= 0:
            return 0.0
        total_equity = equity_curve[-1]
        if total_equity <= 0:
            return -1.0 # 避免資產歸零錯誤
        return math.pow(total_equity, 1.0 / years_count) - 1.0

    @staticmethod
    def max_drawdown(equity_curve):
        max_dd = 0.0
        peak = 1.0
        for equity in equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd

    @staticmethod
    def sharpe_ratio(yearly_returns, risk_free_rate=0.015):
        mean_ret = Utils.mean(yearly_returns)
        std_val = Utils.std_dev(yearly_returns)
        if std_val == 0:
            return 0.0
        return (mean_ret - risk_free_rate) / std_val