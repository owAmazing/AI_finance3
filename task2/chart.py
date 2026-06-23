# chart.py
import tkinter as tk

class ChartGenerator:
    @staticmethod
    def render_results(metrics, equity_curve, years):
        root = tk.Tk()
        root.title("AIFT Final Project - Random Forest Backtest Results")
        root.geometry("800x500")
        
        canvas = tk.Canvas(root, width=800, height=500, bg="white")
        canvas.pack()
        
        # 1. 繪製四大指標長條圖 (左半邊)
        canvas.create_text(200, 30, text="Performance Metrics (1998-2008)", font=("Arial", 14, "bold"), fill="black")
        
        y_offset = 70
        for name, val in metrics.items():
            display_str = f"{name}: {val*100:.2f}%" if name != 'Sharpe Ratio' else f"{name}: {val:.2f}"
            canvas.create_text(120, y_offset, text=display_str, font=("Arial", 11), anchor="w")
            
            # 畫長條圖
            bar_width = max(10, min(200, abs(val) * 200))
            color = "green" if val >= 0 else "red"
            canvas.create_rectangle(120, y_offset + 15, 120 + bar_width, y_offset + 30, fill=color)
            y_offset += 60
            
        # 2. 繪製資產淨值折線圖 Equity Curve (右半邊)
        canvas.create_text(600, 30, text="Equity Curve (Growth of 1.0)", font=("Arial", 14, "bold"), fill="black")
        canvas.create_line(450, 400, 750, 400, width=2) # X 軸
        canvas.create_line(450, 100, 450, 400, width=2) # Y 軸
        
        # 計算座標映射
        max_eq = max(equity_curve)
        min_eq = min(equity_curve)
        eq_range = (max_eq - min_eq) if max_eq != min_eq else 1.0
        
        num_pts = len(equity_curve)
        x_step = 300 / (num_pts - 1) if num_pts > 1 else 300
        
        points = []
        for i, eq in enumerate(equity_curve):
            x = 450 + (i * x_step)
            # Y 軸向上為小，所以用 400 減
            y = 400 - ((eq - min_eq) / eq_range * 250)
            points.append((x, y))
            
            # 標記起訖與年份
            if i == 0:
                canvas.create_text(x, 415, text="1997", font=("Arial", 8))
            elif i == num_pts - 1:
                canvas.create_text(x, 415, text="2008", font=("Arial", 8))
                canvas.create_text(x, y - 10, text=f"{eq:.2f}", font=("Arial", 9, "bold"), fill="blue")
                
        # 連接線條
        for i in range(len(points) - 1):
            canvas.create_line(points[i][0], points[i][1], points[i+1][0], points[i+1][1], fill="blue", width=3)
            
        root.mainloop()