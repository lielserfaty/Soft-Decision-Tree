import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from data_preparation import get_data
from pandas.plotting import table
import pandas_profiling
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



CLF_PATH = '.\\data\\clf\\'
REG_PATH = '.\\data\\reg\\'

clf_data = ['churn_modelling', 'fetal_health', 'mobile_price_classification', 'performance_prediction', 'winequality_red']
#reg_data = ['avocado_prices', 'bike_sharing_demand', 'life_expectancy_data', 'song_data', 'test_scores']
reg_data = ['test_scores']


def run_data_analysis(task, data_name):
    if task == 'clf':
        print("-"*40 + data_name + "-"*40)
        X, y = get_data('clf', data_name)
        data_analysis(X,y, data_name)
    else:
        print("-" * 40 + data_name + "-" * 40)
        X, y = get_data('reg', data_name)
        data_analysis(X,y, data_name)


def check_null(X, y):
    if np.isnan(y).any() or X.isnull().values.any():
        print("There are null in the data")
    else:
        print("There are NOT null in the data")
    print("-"*10)


def head_data(X, y, name):
    df = pd.concat([X, y], axis=1)
    report = pandas_profiling.ProfileReport(df, explorative=True)
    report.to_file(f"./plots/reports/{name}.html")
    df.head(7).to_csv(f'./plots/head/{name}.csv')


def describe_df(df, name):
    df.describe().to_csv(f'./plots/describe/{name}.csv')


def target_count_plot(y, name):
    ax1 = sns.countplot(x=y)
    total = len(y)
    for p in ax1.patches:
        percentage = '{:.1f}%'.format(100 * p.get_height() / total)
        ax1.annotate(percentage, (p.get_x() + 0.25, p.get_height() + 0.01))
    plt.savefig(f'./plots/target_count/{name}.png')


def corr(X, name):
    plt.figure(figsize=(22, 12))
    corr = X.corr()
    mask = np.triu(np.ones_like(corr, dtype='bool'))
    sns.heatmap(corr, mask=mask, annot=True, center=0, fmt='.2f', square=True, cmap="Blues").set(
        title="correlation matrix")
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns, cmap="Blues")
    plt.savefig(f'./plots/corr/{name}.png')


def variables_distribution(df, target, name):
    x = 4
    y = (len(df.columns)//x) + 1
    fig, ax = plt.subplots(y, x, figsize=(20, 10))
    i, j = 0, 0
    for col in df.columns:
        try:
            sns.kdeplot(df[col], hue=target, ax=ax[i,j], shade=True)
        except:
            continue
        j += 1
        if j == x:
            j = 0
            i += 1
    fig.tight_layout()
    plt.savefig(f'./plots/feature_plot/{name}.png')


def data_analysis(X, y, name):
    # check null
    check_null(X, y)
    print(X.info())

    # head
    head_data(X, y, name)

    # describe
    describe_df(X, name)

    # count-plot of target
    target_count_plot(y, name)

    # correlation
    corr(X, name)

    # feature distribution
    variables_distribution(X, y, name)

# for data in clf_data:
#     run_data_analysis("clf", data)

for data in reg_data:
    run_data_analysis("reg", data)


