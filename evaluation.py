from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
import time
from sklearn.model_selection import cross_validate
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from multiscorer import *
import numpy as np
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from numpy import average


def model_accuracy_clf(prediction_soft_improve, prediction_soft, prediction_regular, y_test):
    accuracy_regular = accuracy_score(y_test, prediction_regular)
    accuracy_soft = accuracy_score(y_test, prediction_soft)
    accuracy_soft_improve = accuracy_score(y_test, prediction_soft_improve)
    return accuracy_regular, accuracy_soft, accuracy_soft_improve


def model_auc_clf(prediction_soft_improve, prediction_soft, prediction_regular, y_test):
    if len(np.unique(y_test)) > 2:
        auc_regular = roc_auc_score(y_test, prediction_regular, multi_class='ovr')
        auc_soft = roc_auc_score(y_test, prediction_soft, multi_class='ovr')
        auc_soft_improve = roc_auc_score(y_test, prediction_soft_improve, multi_class='ovr')

    else:
        auc_regular = roc_auc_score(y_test, prediction_regular[:,1])
        auc_soft = roc_auc_score(y_test, prediction_soft[:,1])
        auc_soft_improve = roc_auc_score(y_test, prediction_soft_improve[:,1])

    return auc_regular, auc_soft, auc_soft_improve


def model_evaluation_reg(prediction_soft_improve, prediction_soft, prediction_regular, y_test):

    r2_regular = r2_score(y_test, prediction_regular)
    r2_soft = r2_score(y_test, prediction_soft)
    r2_soft_improve = r2_score(y_test, prediction_soft_improve)

    mse_regular = mean_squared_error(y_test, prediction_regular, squared=False)
    mse_soft = mean_squared_error(y_test, prediction_soft, squared=False)
    mse_soft_improve = mean_squared_error(y_test, prediction_soft_improve, squared=False)

    mape_regular = mean_absolute_percentage_error(y_test, prediction_regular)
    mape_soft = mean_absolute_percentage_error(y_test, prediction_soft)
    mape_soft_improve = mean_absolute_percentage_error(y_test, prediction_soft_improve)

    return r2_regular, r2_soft, r2_soft_improve,  mse_regular, mse_soft, mse_soft_improve,mape_regular, mape_soft, mape_soft_improve


def cross_valitaion_clf(soft_model_improve, soft_model, regular_model, X, y):
    roc_auc = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    scoring = {"accuracy": "accuracy", "roc_auc": roc_auc}

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=1)
    scores_regular = cross_validate(regular_model, X, y, scoring=scoring, cv=cv, n_jobs=1)
    scores_soft = cross_validate(soft_model, X, y, scoring=scoring, cv=cv, n_jobs=1)
    scores_soft_improve = cross_validate(soft_model_improve, X, y, scoring=scoring, cv=cv, n_jobs=1)

    return scores_regular, scores_soft, scores_soft_improve


def cross_valitaion_reg_2(soft_tree_improve, soft_model, regular_model, X, y):
    scoring = {'r2': 'r2', "mse": "neg_mean_squared_error",
               'mape': 'neg_mean_absolute_percentage_error'}

    folds = RepeatedKFold(n_splits=2, n_repeats=2, random_state=100)

    scores_regular_r2 = cross_val_score(regular_model, X, y, scoring='r2', cv=folds, n_jobs=1)
    scores_regular_mse = cross_val_score(regular_model, X, y, scoring='neg_mean_squared_error', cv=folds, n_jobs=1)
    scores_regular_mape = cross_val_score(regular_model, X, y, scoring='neg_mean_absolute_percentage_error', cv=folds, n_jobs=1)


    scores_soft_r2 = cross_val_score(soft_model, X, y, scoring='r2', cv=folds, n_jobs=1)
    scores_soft_mse = cross_val_score(soft_model, X, y, scoring='neg_mean_squared_error', cv=folds, n_jobs=1)
    scores_soft_mape = cross_val_score(soft_model, X, y, scoring='neg_mean_absolute_percentage_error', cv=folds, n_jobs=1)


    scores_soft_improve_r2 = cross_val_score(soft_tree_improve, X, y, scoring='r2', cv=folds, n_jobs=1)
    scores_soft_improve_mse = cross_val_score(soft_tree_improve, X, y, scoring='neg_mean_squared_error', cv=folds, n_jobs=1)
    scores_soft_improve_mape = cross_val_score(soft_tree_improve, X, y, scoring='neg_mean_squared_error', cv=folds, n_jobs=1)

    return scores_regular_r2, scores_regular_mse,scores_regular_mape,  scores_soft_r2, scores_soft_mse, scores_soft_mape, \
            scores_soft_improve_r2, scores_soft_improve_mse, scores_soft_improve_mape


def cross_valitaion_reg(soft_tree_improve, soft_model, regular_model, X, y):

    folds = RepeatedKFold(n_splits=5, n_repeats=2, random_state=100)

    model_list = {regular_model: {}, soft_model: {}, soft_tree_improve:{}}

    for model in model_list.keys():
        scorer = MultiScorer({
            'r2': (r2_score, {}),
            'mse': (mean_squared_error, {}),
            'mape': (mean_absolute_percentage_error, {})
        })

        start_time = time.time()

        cross = cross_val_score(model, X, y, scoring=scorer, cv=folds)
        results = scorer.get_results()

        end_time = time.time() - start_time

        for metric_name in results.keys():
            model_list[model][metric_name+"_avg"] = np.average(results[metric_name])
            model_list[model][metric_name+'_std'] = np.std(results[metric_name])
        model_list[model]['time'] = end_time

    return model_list




