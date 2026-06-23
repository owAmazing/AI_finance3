import numpy as np
import pandas as pd
from utils import calc_entropy, calc_best_split, rank_features


# ──────────────────────────────────────────
# 樹節點結構
# ──────────────────────────────────────────

class TreeNode:
    """
    決策樹的單一節點。

    分支節點：存放分裂特徵、閾值、左右子節點
    葉節點  ：存放預測結果（1 或 -1）
    """
    def __init__(self):
        # 分支節點屬性
        self.feature    = None   # 分裂特徵名稱
        self.threshold  = None   # 分裂閾值
        self.info_gain  = None   # 此節點的資訊增益
        self.left       = None   # <= threshold
        self.right      = None   # >  threshold

        # 葉節點屬性
        self.is_leaf    = False
        self.prediction = None   # 1 或 -1
        self.n_samples  = None   # 此節點的樣本數
        self.n_pos      = None   # 正類（1）數量
        self.n_neg      = None   # 負類（-1）數量


# ──────────────────────────────────────────
# ID3 決策樹
# ──────────────────────────────────────────

class ID3DecisionTree:
    """
    參考 ID3 精神實作的決策樹，支援連續型特徵。

    每個節點：
      1. 對所有特徵搜尋最佳分裂點（最大資訊增益）
      2. 用最佳特徵 + 閾值切成左右兩個子集
      3. 遞迴建樹直到停止條件

    Parameters
    ----------
    max_depth      : int，樹的最大深度（預防過擬合）
    min_samples    : int，節點最少樣本數，低於此值就變葉節點
    min_gain       : float，資訊增益低於此值就停止分裂
    """

    def __init__(
        self,
        max_depth  : int   = 5,
        min_samples: int   = 10,
        min_gain   : float = 1e-4,
    ):
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.min_gain    = min_gain
        self.root        = None
        self.feature_importance_ = {}   # 各特徵的累積資訊增益

    # ── 訓練 ────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        訓練決策樹。

        Parameters
        ----------
        X : pd.DataFrame，特徵矩陣
        y : pd.Series，標籤（1 或 -1）
        """
        # 初始化特徵重要性
        self.feature_importance_ = {col: 0.0 for col in X.columns}
        self.feature_cols_ = list(X.columns)

        self.root = self._build_tree(
            X.values, y.values, list(X.columns), depth=0
        )
        print(f"[ID3] 訓練完成，樹深度上限={self.max_depth}，"
              f"訓練樣本={len(y)}")
        return self

    def _build_tree(
        self,
        X      : np.ndarray,
        y      : np.ndarray,
        cols   : list[str],
        depth  : int,
    ) -> TreeNode:
        node = TreeNode()
        node.n_samples = len(y)
        node.n_pos     = int((y ==  1).sum())
        node.n_neg     = int((y == -1).sum())

        # ── 停止條件 ────────────────────────
        # 1. 達到最大深度
        # 2. 樣本數不足
        # 3. 所有樣本同一類別
        if (depth >= self.max_depth or
                len(y) < self.min_samples or
                len(np.unique(y)) == 1):
            node.is_leaf    = True
            node.prediction = self._majority_vote(y)
            return node

        # ── 搜尋最佳分裂點 ──────────────────
        best_gain      = -np.inf
        best_col_idx   = None
        best_threshold = None

        for i, col in enumerate(cols):
            result = calc_best_split(X[:, i], y)
            if result['best_gain'] > best_gain:
                best_gain      = result['best_gain']
                best_col_idx   = i
                best_threshold = result['best_threshold']

        # 資訊增益太小，直接變葉節點
        if best_gain < self.min_gain or best_threshold is None:
            node.is_leaf    = True
            node.prediction = self._majority_vote(y)
            return node

        # ── 紀錄特徵重要性 ──────────────────
        best_col_name = cols[best_col_idx]
        self.feature_importance_[best_col_name] += best_gain

        # ── 設定分支節點 ────────────────────
        node.feature   = best_col_name
        node.threshold = best_threshold
        node.info_gain = best_gain

        left_mask  = X[:, best_col_idx] <= best_threshold
        right_mask = X[:, best_col_idx] >  best_threshold

        node.left  = self._build_tree(
            X[left_mask],  y[left_mask],  cols, depth + 1
        )
        node.right = self._build_tree(
            X[right_mask], y[right_mask], cols, depth + 1
        )
        return node

    def _majority_vote(self, y: np.ndarray) -> int:
        """回傳多數類別，平手時回傳 -1（保守策略）"""
        if len(y) == 0:
            return -1
        n_pos = (y ==  1).sum()
        n_neg = (y == -1).sum()
        return 1 if n_pos > n_neg else -1

    # ── 預測 ────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        預測每支股票的 Label（1 或 -1）。

        Parameters
        ----------
        X : pd.DataFrame，特徵矩陣（欄位需與訓練時相同）

        Returns
        -------
        np.ndarray，shape (n,)，值為 1 或 -1
        """
        return np.array([
            self._predict_one(row.values)
            for _, row in X[self.feature_cols_].iterrows()
        ])

    def _predict_one(self, x: np.ndarray) -> int:
        """沿著樹走到葉節點，回傳預測值"""
        node = self.root
        col_idx = {col: i for i, col in enumerate(self.feature_cols_)}

        while not node.is_leaf:
            feat_val = x[col_idx[node.feature]]
            if np.isnan(feat_val):
                # 缺值：走樣本數較多的那邊
                if node.left.n_samples >= node.right.n_samples:
                    node = node.left
                else:
                    node = node.right
            elif feat_val <= node.threshold:
                node = node.left
            else:
                node = node.right

        return node.prediction

    # ── 特徵重要性 ──────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        """
        回傳各特徵的累積資訊增益，由大到小排序。

        Returns
        -------
        pd.DataFrame，欄位：feature, importance
        """
        df = pd.DataFrame([
            {'feature': k, 'importance': v}
            for k, v in self.feature_importance_.items()
        ]).sort_values('importance', ascending=False).reset_index(drop=True)
        return df

    # ── 樹結構印出 ──────────────────────────

    def print_tree(self, max_display_depth: int = 4):
        """
        印出樹的結構（文字版）。

        Parameters
        ----------
        max_display_depth : int，只顯示到第幾層（樹太深會太長）
        """
        print("\n=== 決策樹結構 ===")
        self._print_node(self.root, depth=0, max_depth=max_display_depth)

    def _print_node(self, node: TreeNode, depth: int, max_depth: int):
        indent = "    " * depth + ("└── " if depth > 0 else "")

        if node.is_leaf:
            label = "打敗平均 ✓" if node.prediction == 1 else "輸給平均 ✗"
            print(f"{indent}[葉節點] 預測={label}"
                  f"  樣本={node.n_samples}（+{node.n_pos}/-{node.n_neg}）")
            return

        if depth >= max_depth:
            print(f"{indent}[...已達顯示深度上限，節點樣本={node.n_samples}]")
            return

        print(f"{indent}[{node.feature} <= {node.threshold:.4f}]"
              f"  增益={node.info_gain:.4f}"
              f"  樣本={node.n_samples}")

        self._print_node(node.left,  depth + 1, max_depth)
        self._print_node(node.right, depth + 1, max_depth)


# ──────────────────────────────────────────
# 快速測試
# ──────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, '/mnt/user-data/outputs')
    from data_loader import load_data, clean_data, split_train_test, get_X_y

    # 載入資料
    df       = load_data('/mnt/user-data/uploads/top200.xlsx')
    df       = clean_data(df)
    train_df, test_df = split_train_test(df, test_years=[2006, 2007, 2008])
    X_train, y_train  = get_X_y(train_df)
    X_test,  y_test   = get_X_y(test_df)

    # 訓練
    model = ID3DecisionTree(max_depth=5, min_samples=10, min_gain=1e-4)
    model.fit(X_train, y_train)

    # 印出樹結構
    model.print_tree(max_display_depth=3)

    # 特徵重要性
    print("\n=== 特徵重要性（累積資訊增益）===")
    print(model.get_feature_importance().to_string(index=False))

    # 測試集預測
    y_pred = model.predict(X_test)
    accuracy = (y_pred == y_test.values).mean()
    print(f"\n=== 測試集準確率 ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"預測為 1（買入）的股票數：{(y_pred == 1).sum()} / {len(y_pred)}")