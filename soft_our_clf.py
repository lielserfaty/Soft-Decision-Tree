from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from statistics import mode
from sklearn.model_selection import cross_validate
import random



class treeSoftImprove(DecisionTreeClassifier):
    def __init__(self, alpha=0.1, n=100, alpha_left_wight=0.8, tresh_epsilon_up=0.4, tresh_epsilon_down=0.05):
        super(treeSoftImprove, self).__init__()
        self.alpha = alpha
        self.n = n
        self.alpha_left_wight = alpha_left_wight
        self.tresh_epsilon_up = tresh_epsilon_up
        self.tresh_epsilon_down = tresh_epsilon_down

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

        return random.uniform(self.tresh_epsilon_down, self.tresh_epsilon_up)

    def generate_direction(self):
        # 1 - as threshold
        return np.random.choice(np.arange(0, 2), p=[self.alpha, 1 - self.alpha])


    def soft_splits_prediction(self, X_test):

        proba =[]
        for sample_id in range(X_test.shape[0]):
            sample_class_lst = []

            for i in range(self.n):
                node_id = 0
                predict = self.sub_tree(X_test, node_id, sample_id)
                sample_class_lst.append(predict)

            sample_class_lst_avg = np.array(sample_class_lst)
            proba.append(np.average(sample_class_lst_avg, axis=0))

        proba = np.array(proba)
        return proba

    def sub_tree(self, X_test, node_id, sample_id):

        proba = []
        feature = self.tree_.feature
        threshold = self.tree_.threshold
        children_left = self.tree_.children_left
        children_right = self.tree_.children_right
        values = self.tree_.value

        while feature[node_id] != -2:

            epsilon_alpha = self.generate_random()
            epsilon_beta = self.generate_random()

            left_node = children_left[node_id]
            right_node = children_right[node_id]

            distance = X_test[sample_id, feature[node_id]] - threshold[node_id]

            # wight the subtree left and right
            direction = self.generate_direction()
            res = []

            val_node = X_test[sample_id, feature[node_id]]
            up_bond = threshold[node_id] + epsilon_alpha
            down_bond = threshold[node_id] - epsilon_beta

            if up_bond > val_node > down_bond:

                # we go not the right direction
                if direction == 0:
                    if distance <= 0:
                        alpha_left = 1 - self.alpha_left_wight

                    else:
                        alpha_left = self.alpha_left_wight
                # we go to the right direction
                else:
                    if distance <= 0:
                        alpha_left = self.alpha_left_wight
                    else:
                        alpha_left = 1 - self.alpha_left_wight

                right_val = self.sub_tree(X_test, right_node, sample_id)
                left_val = self.sub_tree(X_test, left_node, sample_id)

                res.append((1-alpha_left) * right_val)
                res.append(alpha_left * left_val)

                # save the output of the child
                sample_class_lst = np.array(res)
                # save the avg prob output of the child
                proba = np.array(np.average(sample_class_lst, axis=0))
                return proba

            if val_node >= up_bond:
                # we go not the right direction
                if direction == 1:
                    node_id = right_node

                # we go to the right direction
                else:
                    node_id = left_node

            elif val_node <= down_bond:
                # we go to the right direction
                if direction == 1:
                    node_id = left_node

                # we not go to the right direction
                else:
                    node_id = right_node

        # we get leave
        return values[node_id][0]




    def predict(self, X, check_input=True):

        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        # proba = self.tree_.predict(X)
        proba = self.soft_splits_prediction(X)
        n_samples = X.shape[0]

        # Classification
        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

#
# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
# iris = load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
#
# decision_tree = treeSoftImprove()
# decision_tree = decision_tree.fit(X_train, y_train)
# ours = decision_tree.predict(X_test)
#
# scores = cross_validate(decision_tree, iris.data, iris.target, scoring=('accuracy'), cv=5, n_jobs=-1)
# print(scores)