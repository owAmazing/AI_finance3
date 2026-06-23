# model.py
import random
import math

class SimpleDecisionTree:
    def __init__(self):
        self.best_feature_idx = None
        self.best_threshold = None
        self.pred_left = -1
        self.pred_right = -1
        self.is_leaf = False
        self.leaf_value = -1

    def _calculate_entropy(self, labels):
        """純手寫計算 Entropy (熵)"""
        if not labels:
            return 0.0
        total = len(labels)
        pos = sum(1 for l in labels if l == 1)
        neg = total - pos
        
        if pos == 0 or neg == 0:
            return 0.0
            
        p_pos = pos / total
        p_neg = neg / total
        return -(p_pos * math.log2(p_pos) + p_neg * math.log2(p_neg))

    def train(self, samples, feature_subset):
        if not samples:
            self.is_leaf = True
            return
            
        labels = [s['label'] for s in samples]
        # 如果樣本全部都是同一類，直接當作葉節點
        if len(set(labels)) == 1:
            self.is_leaf = True
            self.leaf_value = labels[0]
            return
            
        current_entropy = self._calculate_entropy(labels)
        best_gain = -1.0
        
        # 遍歷隨機特徵子集
        for f_idx in feature_subset:
            f_values = [s['features'][f_idx] for s in samples]
            if not f_values:
                continue
            
            # 使用分位數（或排序後的代表點）來尋找最佳切分點
            sorted_v = sorted(list(set(f_values)))
            if len(sorted_v) <= 1:
                continue
                
            # 隨機抽樣一些切分候選點，維持隨機森林的隨機性
            step = max(1, len(sorted_v) // 10)
            candidates = sorted_v[::step]
            
            for threshold in candidates:
                left_s = [s for s in samples if s['features'][f_idx] <= threshold]
                right_s = [s for s in samples if s['features'][f_idx] > threshold]
                
                if not left_s or not right_s:
                    continue
                
                left_labels = [s['label'] for s in left_s]
                right_labels = [s['label'] for s in right_s]
                
                # 計算切分後的總期望熵 (Weighted Entropy)
                w_entropy = (len(left_s)/len(samples)) * self._calculate_entropy(left_labels) + \
                            (len(right_s)/len(samples)) * self._calculate_entropy(right_labels)
                
                # 資訊增益 (Information Gain)
                gain = current_entropy - w_entropy
                
                if gain > best_gain:
                    best_gain = gain
                    self.best_feature_idx = f_idx
                    self.best_threshold = threshold
                    self.pred_left = 1 if sum(left_labels) >= 0 else -1
                    self.pred_right = 1 if sum(right_labels) >= 0 else -1

        if best_gain <= 0:
            self.is_leaf = True
            self.leaf_value = 1 if sum(labels) >= 0 else -1

    def predict(self, features):
        if self.is_leaf:
            return self.leaf_value
        if self.best_feature_idx is None:
            return -1
        if features[self.best_feature_idx] <= self.best_threshold:
            return self.pred_left
        return self.pred_right


class RandomForestModel:
    def __init__(self, num_trees=50):
        self.num_trees = num_trees
        self.trees = []

    def train(self, train_data):
        # 固定種子，確保實驗可被完全複製
        random.seed(40)
        self.trees = []
        
        num_features = len(train_data[0]['features'])
        subset_size = max(1, int(math.sqrt(num_features)))

        for _ in range(self.num_trees):
            bootstrap_samples = [random.choice(train_data) for _ in range(len(train_data))]
            feature_indices = list(range(num_features))
            feature_subset = random.sample(feature_indices, subset_size)
            
            tree = SimpleDecisionTree()
            tree.train(bootstrap_samples, feature_subset)
            self.trees.append(tree)

    def predict_score(self, features):
        """回傳森林中預測買入(1)的樹佔總數的比例，作為選股排序的分數"""
        votes = sum(1 for tree in self.trees if tree.predict(features) == 1)
        return votes / self.num_trees

    def predict(self, features):
        # 傳統多數決
        score = self.predict_score(features)
        return 1 if score >= 0.5 else -1