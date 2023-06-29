from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from statistics import mode
from sklearn.model_selection import cross_validate


class treeSoft(DecisionTreeClassifier):
    def __init__(self, alpha=0.1, n=10):
        super(treeSoft, self).__init__()
        self.alpha = alpha
        self.n = n

    def predict_proba(self, X, check_input=True):
        """Predict class probabilities of the input samples X.
        The predicted class probability is the fraction of samples of the same
        class in a leaf.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32`` and if a sparse matrix is provided
            to a sparse ``csr_matrix``.
        check_input : bool, default=True
            Allow to bypass several input checking.
            Don't use this parameter unless you know what you do.
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes) or list of n_outputs \
            such arrays if n_outputs > 1
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        proba = self.soft_splits_prediction(X)

        if self.n_outputs_ == 1:
            proba = proba[:, : self.n_classes_]
            normalizer = proba.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            proba /= normalizer
            return proba
        else:
            all_proba = []

            for k in range(self.n_outputs_):
                proba_k = proba[:, k, : self.n_classes_[k]]
                normalizer = proba_k.sum(axis=1)[:, np.newaxis]
                normalizer[normalizer == 0.0] = 1.0
                proba_k /= normalizer
                all_proba.append(proba_k)

            return all_proba


    def generate_random(self):
        # 1 - as threshold
        return np.random.choice(np.arange(0, 2), p=[self.alpha, 1 - self.alpha])

    def soft_splits_prediction(self, X_test):

        proba =[]
        feature = self.tree_.feature
        threshold = self.tree_.threshold
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        values = self.tree_.value

        for sample_id in range(X_test.shape[0]):
            sample_class_lst = []
            for i in range(self.n):
                node_id = 0
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
                sample_class_lst.append(values[node_id][0])

            sample_class_lst_avg = np.array(sample_class_lst)
            proba.append(np.average(sample_class_lst_avg, axis=0))

        proba = np.array(proba)
        return proba


    def predict(self, X, check_input=True):

        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        # proba = self.tree_.predict(X)
        proba = self.soft_splits_prediction(X)
        n_samples = X.shape[0]

        # Classification
        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)


# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# decision_tree = treeSoft()
# decision_tree = decision_tree.fit(X_train, y_train)
# # ours = decision_tree.predict(X_test)
# ours = decision_tree.predict_proba(X_test)
# from sklearn.metrics import roc_auc_score
# auc_regular = roc_auc_score(y_test, ours, multi_class='ovr')
# print(auc_regular)
# # scores = cross_validate(decision_tree, iris.data, iris.target, scoring=('accuracy'), cv=5, n_jobs=-1)
# # print(scores)