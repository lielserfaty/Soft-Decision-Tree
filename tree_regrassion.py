from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from statistics import mean
from sklearn.model_selection import cross_validate



class treeRegrassion(DecisionTreeRegressor):

    def __init__(self, alpha=0.1, n=10):
        super(treeRegrassion, self).__init__()
        self.alpha = alpha
        self.n = n

    def generate_random(self):
        # 1 - as threshold
        return np.random.choice(np.arange(0, 2), p=[self.alpha, 1 - self.alpha])

    def soft_splits_prediction(self, X_test):
        proba = []
        feature = self.tree_.feature
        threshold = self.tree_.threshold
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        values = self.tree_.value

        for sample_id in range(X_test.shape[0]):
            sample_class_lst = []
            for i in range(self.n):
                node_id = 0
                # we dont get leaf node
                while feature[node_id] != -2:

                    # check if value of the split feature for sample 0 is below threshold
                    direction = self.generate_random()
                    left_node = children_left[node_id]
                    right_node = children_right[node_id]
                    next_node_1 = None
                    next_node_0 = None
                    if X_test[sample_id, feature[node_id]] <= threshold[node_id]:
                        next_node_1 = left_node
                        next_node_0 = right_node
                    else:
                        next_node_1 = right_node
                        next_node_0 = left_node
                    if direction == 0:
                        node_id = next_node_0
                    else:
                        node_id = next_node_1
                # we get leaf thus save the value of the prediction
                sample_class_lst.append(values[node_id][0][0])

            avg_value = mean(sample_class_lst)
            proba.append(avg_value)

        proba = np.array(proba)
        proba = proba.reshape(-1, 1)
        return proba

    def predict(self, X, check_input=True):

        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        # proba = self.tree_.predict(X)
        proba = self.soft_splits_prediction(X)
        n_samples = X.shape[0]

        # Classification
        if self.n_outputs_ == 1:
            return proba[:, 0]



#
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
#
#
# df = pd.read_csv('Iris.csv')
# df = df.dropna()
#
# X_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm']
# y_cols = ['PetalWidthCm']
# X = df[X_cols]
# y = df[y_cols]
#
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# decision_tree = treeRegrassion()
# decision_tree = decision_tree.fit(X_train, y_train)
# ours = decision_tree.predict(X_test)
# print(ours)
# print(y_test)
