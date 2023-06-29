from sklearn.model_selection import train_test_split
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import full_suite
from tree_regrassion import *
from tree_classification import *
from data_preparation import get_data
from soft_our_clf import *
from soft_our_reg import *

CLF_PATH = '.\\data\\clf\\'
REG_PATH = '.\\data\\reg\\'
clf_data = ['churn_modelling', 'fetal_health', 'mobile_price_classification', 'performance_prediction', 'winequality_red']

#reg_data = ['test_scores', 'life_expectancy_data', 'song_data','bike_sharing_demand','avocado_prices']
reg_data = ['song_data','bike_sharing_demand','avocado_prices']

def read_data(path):
    return pd.read_csv(path)


def deepchecker(df, X,y, target, task, a, n, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    #df_train, df_test = train_test_split(df, stratify=df[target].values, random_state=0)
    df_train = pd.concat([X_train, y_train], axis=1)
    df_train.to_csv("X_test.csv")
    df_test = pd.concat([X_test, y_test], axis=1)
    ds_train = Dataset(df_train, label=target, cat_features=[])
    ds_test = Dataset(df_test, label=target, cat_features=[])
    suite = full_suite()
    model = None
    if task == 'clf':
        model = treeSoftImprove(a, n)
    else:
        model = treeSoftRegImprove(a, n)
    model.fit(df_train.drop(target, axis=1), df_train[target])
    suite.run(train_dataset=ds_train, test_dataset=ds_test, model=model).save_as_html(f'./deepchecks/outputs/{name}.html')


def run_checker(task, data, a, n):
    X, y = get_data(task, data)
    target = str(y.name)
    df = pd.concat([X, y], axis=1)
    deepchecker(df, X,y, target, task, a, n, data)
#
# for data in clf_data:
#     run_checker('clf', data, 0.1, 100)

for data in reg_data:
    run_checker('reg', data, 0.05, 100)
