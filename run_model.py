import csv

import numpy as np
from mlxtend.evaluate import paired_ttest_5x2cv

from data_preparation import get_data
from tree_regrassion import *
from tree_classification import *
from soft_our_clf import *
from soft_our_reg import *
from evaluation import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


clf_data = ['churn_modelling', 'fetal_health', 'mobile_price_classification', 'performance_prediction', 'winequality_red']
reg_data = ['avocado_prices', 'bike_sharing_demand', 'life_expectancy_data', 'song_data', 'test_scores']

def write_to_csv(row):
    path = "/sise/home/efrco/result_700_life.csv"

    with open(path, 'a', newline='') as f:
        write = csv.writer(f)
        write.writerow(row)


def run_reg_model(X, y, a, n, name):

    # regular tree
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regular_tree = DecisionTreeRegressor()
    regular_tree.fit(X_train, y_train)
    regular_prediction = regular_tree.predict(X_test)

    # soft tree
    soft_tree = treeRegrassion(a, n)
    soft_tree.fit(X_train, y_train)
    soft_predictions = soft_tree.predict(X_test)

    # soft tree improve
    soft_tree_improve = treeSoftRegImprove(a, n)
    soft_tree_improve.fit(X_train, y_train)
    soft_predictions_improve = soft_tree_improve.predict(X_test)

    # mse and rmse
    r2_regular, r2_soft, r2_soft_improve, mse_regular, mse_soft, mse_soft_improve, \
     mape_regular, mape_soft, mape_soft_improve = model_evaluation_reg(
        soft_predictions_improve, soft_predictions, regular_prediction, y_test)
    # sp = pd.DataFrame(soft_predictions, columns=['soft'])
    # rp = pd.DataFrame(regular_prediction, columns=['regular'])
    # spImprove = pd.DataFrame(soft_predictions_improve, columns=['soft_improve'])
    #
    # sp['regular'] = rp
    # sp['true_y'] = y_test.tolist()
    # sp['soft_improve'] = spImprove
    #
    # # fig soft vs. true
    # s = sns.lmplot(x='true_y', y='soft', data=sp)
    # s.set_axis_labels('y_true', 'y_pred')
    # plt.savefig(f'/sise/home/efrco/modelim-compute/result2/regression/{name}_soft.png')
    #
    # # fig regular vs. true
    # r = sns.lmplot(x='true_y', y='regular', data=sp)
    # r.set_axis_labels('y_true', 'y_pred')
    # plt.savefig(f'/sise/home/efrco/modelim-compute/result2/regression/{name}_regular.png')
    #
    # # fig soft_improve vs. true
    # r = sns.lmplot(x='true_y', y='soft_improve', data=sp)
    # r.set_axis_labels('y_true', 'y_pred')
    # plt.savefig(f'/sise/home/efrco/modelim-compute/result2/regression/{name}_soft_improve.png')

    # cross validation
    model_list = cross_valitaion_reg(soft_tree_improve, soft_tree, regular_tree, X, y)

    write_to_csv(['r2_regular', r2_regular])
    write_to_csv(['r2_soft', r2_soft])
    write_to_csv(['r2_soft_improve', r2_soft_improve])

    write_to_csv(['mse_regular', mse_regular])
    write_to_csv(['mse_soft', mse_soft])
    write_to_csv(['mse_soft_improve', mse_soft_improve])

    write_to_csv(['mape_regular', mape_regular])
    write_to_csv(['mape_soft', mape_soft])
    write_to_csv(['mape_soft_improve', mape_soft_improve])

    for model_name in model_list.keys():

        name = model_name.__class__.__name__

        write_to_csv(['CV_r2_avg_' + name, model_list[model_name]['r2_avg']])
        write_to_csv(['CV_r2_std_' + name, model_list[model_name]['r2_std']])

        write_to_csv(['CV_mse_avg_' + name, model_list[model_name]['mse_avg']])
        write_to_csv(['CV_mse_std_' + name, model_list[model_name]['mse_std']])

        write_to_csv(['CV_mape_avg_' + name, model_list[model_name]['mape_avg']])
        write_to_csv(['CV_mape_std_' + name, model_list[model_name]['mape_std']])
        write_to_csv(['CV_time_' + name, model_list[model_name]['time']])


def run_clf_model(X, y, a, n, name):

    # regular tree
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    regular_tree = DecisionTreeClassifier()
    regular_tree.fit(X_train, y_train)
    regular_proba = regular_tree.predict_proba(X_test)
    regular_prediction = regular_tree.predict(X_test)

    # soft tree
    soft_tree = treeSoft(a, n)
    soft_tree.fit(X_train, y_train)
    soft_proba = soft_tree.predict_proba(X_test)
    soft_predictions = soft_tree.predict(X_test)

    # soft tree improve
    soft_tree_improve = treeSoftImprove(a, n)
    soft_tree_improve.fit(X_train, y_train)
    soft_proba_improve = soft_tree_improve.predict_proba(X_test)
    soft_predictions_improve = soft_tree_improve.predict(X_test)

    # accuracy and auc
    accuracy_regular, accuracy_soft, accuracy_soft_improve = model_accuracy_clf(soft_predictions_improve,
                                                                                soft_predictions, regular_prediction,
                                                                                y_test)
    auc_regular, auc_soft, auc_soft_improve = model_auc_clf(soft_proba_improve, soft_proba, regular_proba, y_test)

    # plots - cm & roc
    cm_regular = confusion_matrix(y_test, regular_prediction)
    p_regular = ConfusionMatrixDisplay(confusion_matrix=cm_regular,  display_labels=regular_tree.classes_)
    p_regular.plot()
    plt.savefig(f'/sise/home/efrco/modelim-compute/result/confusion_matrix/{name}_regular.png')

    cm_soft = confusion_matrix(y_test, soft_predictions)
    p_soft = ConfusionMatrixDisplay(confusion_matrix=cm_soft, display_labels=soft_tree.classes_)
    p_soft.plot()
    plt.savefig(f'/sise/home/efrco/modelim-compute/result/confusion_matrix/{name}_soft.png')

    cm_soft_improve = confusion_matrix(y_test, soft_predictions_improve)
    p_soft_improve = ConfusionMatrixDisplay(confusion_matrix=cm_soft_improve, display_labels=soft_tree_improve.classes_)
    p_soft_improve.plot()
    plt.savefig(f'/sise/home/efrco/modelim-compute/result/confusion_matrix/{name}_soft_improve.png')


    # cross validation
    cv_regular, cv_soft, cv_soft_improve = cross_valitaion_clf(soft_tree_improve, soft_tree, regular_tree, X, y)

    # metric without cross
    write_to_csv(['accuracy_regular', accuracy_regular])
    write_to_csv(['accuracy_soft', accuracy_soft])
    write_to_csv(['accuracy_soft_improve', accuracy_soft_improve])
    write_to_csv(['auc_regular', auc_regular])
    write_to_csv(['auc_soft', auc_soft])
    write_to_csv(['auc_soft_improve', auc_soft_improve])

    # metric with cross accuracy
    write_to_csv(['CV_regular_accuracy_avg', np.average(cv_regular['test_accuracy'])])
    write_to_csv(['CV_regular_accuracy_std', np.std(cv_regular['test_accuracy'])])

    write_to_csv(['CV_soft_accuracy_avg',  np.average(cv_soft['test_accuracy'])])
    write_to_csv(['CV_soft_accuracy_std', np.std(cv_soft['test_accuracy'])])

    write_to_csv(['CV_soft_improve_accuracy_avg',  np.average(cv_soft_improve['test_accuracy'])])
    write_to_csv(['CV_soft_improve_accuracy_std',  np.std(cv_soft_improve['test_accuracy'])])

    # metric with cross auc
    write_to_csv(['CV_regular_auc_avg', np.average(cv_regular['test_roc_auc'])])
    write_to_csv(['CV_regular_auc_std', np.std(cv_regular['test_roc_auc'])])

    write_to_csv(['CV_soft_auc_avg', np.average(cv_soft['test_roc_auc'])])
    write_to_csv(['CV_soft_auc_std', np.std(cv_soft['test_roc_auc'])])

    write_to_csv(['CV_soft_improve_auc_avg', np.average(cv_soft_improve['test_roc_auc'])])
    write_to_csv(['CV_soft_improve_auc_std', np.std(cv_soft_improve['test_roc_auc'])])

    # metric with cross time
    write_to_csv(['CV_regular_time_avg', np.average(cv_regular['score_time'])])
    write_to_csv(['CV_soft_time_avg', np.average(cv_soft['score_time'])])
    write_to_csv(['CV_soft_improve_time_avg', np.average(cv_soft_improve['score_time'])])






def run_reg_model_sens(X, y, a, n, name, alpha_left_wight,tresh_epsilon_up, tresh_epsilon_down):

    # regular tree
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # soft tree improve
    soft_tree_improve = treeSoftRegImprove(a, n, alpha_left_wight, tresh_epsilon_up, tresh_epsilon_down)
    soft_tree_improve.fit(X_train, y_train)
    soft_predictions_improve = soft_tree_improve.predict(X_test)

    # mse and rmse
    r2_regular, r2_soft, r2_soft_improve, mse_regular, mse_soft, mse_soft_improve, \
     mape_regular, mape_soft, mape_soft_improve = model_evaluation_reg(
        soft_predictions_improve, soft_predictions_improve, soft_predictions_improve, y_test)


    write_to_csv(['r2_soft_improve', r2_soft_improve])
    write_to_csv(['mse_soft_improve', mse_soft_improve])
    write_to_csv(['mape_soft_improve', mape_soft_improve])



def run_clf_model_sene(X, y, a, n, name, alpha_left_wight,tresh_epsilon_up, tresh_epsilon_down):

    # regular tree
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # soft tree improve
    soft_tree_improve = treeSoftImprove(a, n)
    soft_tree_improve.fit(X_train, y_train)
    soft_proba_improve = soft_tree_improve.predict_proba(X_test)
    soft_predictions_improve = soft_tree_improve.predict(X_test)

    # accuracy and auc
    accuracy_regular, accuracy_soft, accuracy_soft_improve = model_accuracy_clf(soft_predictions_improve,
                                                                                soft_predictions_improve, soft_predictions_improve,
                                                                                y_test)
    auc_regular, auc_soft, auc_soft_improve = model_auc_clf(soft_proba_improve, soft_proba_improve, soft_proba_improve, y_test)


    write_to_csv(['accuracy_soft_improve', accuracy_soft_improve])
    write_to_csv(['auc_soft_improve', auc_soft_improve])



def sensetivity_analysis():
    # alpha_lst = [0.05, 0.1, 0.2, 0.3, 0.4]
    # n_lst = [100, 150, 200, 350, 500, 700, 1000]
    a = 0.1
    n = 80
    alpha_left_wight_list = [0.7, 0.8, 0.9]
    tresh_epsilon_up_list = [0.35, 0.4, 0.45, 0.5]
    tresh_epsilon_down_list = [0.05, 0.1, 0.2, 0.3]

    for alpha_left_wight in alpha_left_wight_list:
        write_to_csv([alpha_left_wight, "alpha_left_wight!!!!!!!!!!!!!!!"])
        tresh_epsilon_up = 0.4
        tresh_epsilon_down = 0.05
        for data in clf_data:
            write_to_csv([data])
            X, y = get_data('clf', data)
            run_clf_model_sene(X, y, a, n, data, alpha_left_wight, tresh_epsilon_up, tresh_epsilon_down)

        for data in reg_data:
            write_to_csv([data])
            X, y = get_data('reg', data)
            run_reg_model_sens(X, y, a, n, data, alpha_left_wight, tresh_epsilon_up, tresh_epsilon_down)

    for tresh_epsilon_up in tresh_epsilon_up_list:
        write_to_csv([tresh_epsilon_up, "tresh_epsilon_up!!!!!!!!!!!!!!!"])
        alpha_left_wight = 0.8
        tresh_epsilon_down = 0.05
        for data in clf_data:
            write_to_csv([data])
            X, y = get_data('clf', data)
            run_clf_model_sene(X, y, a, n, data, alpha_left_wight, tresh_epsilon_up, tresh_epsilon_down)

        for data in reg_data:
            write_to_csv([data])
            X, y = get_data('reg', data)
            run_reg_model_sens(X, y, a, n, data, alpha_left_wight, tresh_epsilon_up, tresh_epsilon_down)


    for tresh_epsilon_down in tresh_epsilon_down_list:
        write_to_csv([tresh_epsilon_down, "tresh_epsilon_down!!!!!!!!!!!!!!!"])
        alpha_left_wight = 0.8
        tresh_epsilon_up = 0.4
        for data in clf_data:
            write_to_csv([data])
            X, y = get_data('clf', data)
            run_clf_model_sene(X, y, a, n, data, alpha_left_wight, tresh_epsilon_up, tresh_epsilon_down)

        for data in reg_data:
            write_to_csv([data])
            X, y = get_data('reg', data)
            run_reg_model_sens(X, y, a, n, data, alpha_left_wight, tresh_epsilon_up, tresh_epsilon_down)




# reg_data = ['song_data', 'test_scores']

# for data in clf_data:
#     a = 0.1
#     n = 350
#     write_to_csv([data])
#     X, y = get_data('clf', data)
#     run_clf_model(X, y, a, n, data)
# reg_data = ['life_expectancy_data']

# for data in reg_data:
#     print("for 300")
#     a = 0.05
#     n = 300
#     write_to_csv([data])
#     X, y = get_data('reg', data)
#     run_reg_model(X, y, a, n, data)
#
# for data in reg_data:
#     print("for 700")
#     a = 0.05
#     n = 700
#     write_to_csv([data])
#     X, y = get_data('reg', data)
#     run_reg_model(X, y, a, n, data)
#


def run_clf_model_test(X, y, a, n, data):

    X = X.to_numpy()
    y = y.to_numpy()
    regular_tree = DecisionTreeClassifier()

    # soft tree
    soft_tree = treeSoft(a, n)
    soft_tree_improve = treeSoftImprove(a, n)


    # t, z = paired_ttest_5x2cv(estimator1=soft_tree,
    #                           estimator2=soft_tree_improve, scoring='accuracy',
    #                           X=X, y=y, random_seed=1)
    # roc_auc = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    # print("Accuracy soft_tree vs improve ")
    # print('t statistic: %.10f' % t)
    # print('p value: %.10f' % z)



    # t, z = paired_ttest_5x2cv(estimator1=soft_tree_improve, estimator2=soft_tree, scoring=roc_auc,
    #                              X=X, y=y,  random_seed=1)
    # t, z = paired_ttest_5x2cv(estimator1=regular_tree,
    #                           estimator2=soft_tree_improve, scoring=roc_auc,
    #                           X=X, y=y, random_seed=1)
    roc_auc = make_scorer(roc_auc_score, multi_class='ovr', needs_proba=True)
    # print("roc_auc regular vs improve ")
    # print('t statistic: %.10f' % t)
    # print('p value: %.10f' % z)
    #
    t, z = paired_ttest_5x2cv(estimator1=soft_tree,
                              estimator2=soft_tree_improve, scoring=roc_auc,
                              X=X, y=y, random_seed=1)
    print("roc_auc soft_tree vs improve ")
    print('t statistic: %.10f' % t)
    print('p value: %.10f' % z)


def run_reg_model_test(X, y, a, n, data):
    from mlxtend.data import iris_data
    X = X.to_numpy()
    y = y.to_numpy()
    regular_tree = DecisionTreeRegressor()

    # soft tree
    soft_tree = treeRegrassion(a, n)
    soft_tree_improve = treeSoftRegImprove(a, n)

    t, z = paired_ttest_5x2cv(estimator1=regular_tree,
                                 estimator2=soft_tree_improve, scoring='neg_root_mean_squared_error',
                                 X=X, y=y, random_seed=1)
    print("result for regular and improve")
    print('t statistic: %.10f' % t)
    print('p value: %.10f' % z)

    t, z = paired_ttest_5x2cv(estimator1=soft_tree,
                                 estimator2=soft_tree_improve, scoring='neg_root_mean_squared_error',
                                 X=X, y=y, random_seed=1)
    print("result for soft and improve")
    print('t statistic: %.10f' % t)
    print('p value: %.10f' % z)




clf_data = ['churn_modelling', 'fetal_health', 'mobile_price_classification', 'performance_prediction', 'winequality_red']

clf_data = ['churn_modelling']


def t_Test():
    print("CLF WITG N=churn_modelling 10, AND ROC")


    for data in clf_data:
        print("#################################################")
        print(data)

        a = 0.1
        n = 160
        X, y = get_data('clf', data)
        run_clf_model_test(X, y, a, n, data)
    # print("REG WITH N=50, AND RMSE")

    # for data in reg_data:
    #     print("#################################################")
    #     print(data)
    #     a = 0.1
    #     n = 1
    #     X, y = get_data('reg', data)
    #     run_reg_model_test(X, y, a, n, data)


t_Test()
# sensetivity_analysis()
# #
#
