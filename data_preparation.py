import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn import tree
from imblearn.over_sampling import RandomOverSampler
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime
from sklearn.metrics import mean_squared_error
from scipy import stats


CLF_PATH = "/sise/home/efrco/modelim-compute/data/clf/"
REG_PATH = "/sise/home/efrco/modelim-compute/data/reg/"
le = LabelEncoder()
clf_data = ['churn_modelling', 'fetal_health', 'mobile_price_classification', 'performance_prediction', 'winequality_red']
reg_data = ['avocado_prices', 'bike_sharing_demand', 'life_expectancy_data', 'song_data', 'test_scores']


def read_data(path):
    return pd.read_csv(path)


def get_data(task, data_name):
    if task == 'clf':
        df = read_data(CLF_PATH + data_name + ".csv")
        X, y = eval(f"pp_{data_name}" + "(df)")

    else:
        df = read_data(REG_PATH + data_name + ".csv")
        X, y = eval(f"pp_{data_name}" + "(df)")
        if data_name == 'song_data':
            X = X.truncate(before=0, after=5000)
            y = y.truncate(before=0, after=5000)
    return X, y


def pp_avocado_prices(df):
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df.drop(['Date', 'Unnamed: 0', 'Total Bags'], axis=1, inplace=True)
    df['type'] = le.fit_transform(df['type'])
    df['region'] = le.fit_transform(df['region'])
    y = df['AveragePrice']
    X = df.drop(['AveragePrice'], axis=1)
    return X, y


def pp_bike_sharing_demand(df):
    df.drop_duplicates(inplace=True)
    df["hour"] = [t.hour for t in pd.DatetimeIndex(df.datetime)]
    df["day"] = [t.dayofweek for t in pd.DatetimeIndex(df.datetime)]
    df["month"] = [t.month for t in pd.DatetimeIndex(df.datetime)]
    df['year'] = [t.year for t in pd.DatetimeIndex(df.datetime)]
    df['year'] = df['year'].map({2011: 0, 2012: 1})
    df.drop('datetime', axis=1, inplace=True)
    df.drop(['casual', 'registered', 'atemp'], axis=1, inplace=True)
    X = df.drop('count', axis=1)
    y = df['count']
    return X, y


def pp_life_expectancy_data(data):
    # data preparation

    data.fillna(data.mean(), inplace=True)
    le = LabelEncoder()
    data['Country'] = le.fit_transform(data['Country'])
    le = LabelEncoder()
    data['Status'] = le.fit_transform(data['Status'])

    X = data.drop('Life expectancy ', axis=1)
    y = data['Life expectancy ']

    X.drop([" thinness 5-9 years", 'under-five deaths '], axis=1, inplace=True)
    return X, y


def pp_song_data(df):
    df.drop_duplicates(inplace=True)
    df = df.drop(["song_name", 'instrumentalness'], axis=1)
    df = df.astype("float")

    X = df.drop('song_popularity', axis=1)
    y = df['song_popularity']
    return X, y


def pp_test_scores(df):
    df.drop('student_id', axis=1, inplace=True)
    df['school'] = le.fit_transform(df['school'])
    df['school_setting'] = le.fit_transform(df['school_setting'])
    df['school_type'] = le.fit_transform(df['school_type'])
    df['classroom'] = le.fit_transform(df['classroom'])
    df['teaching_method'] = le.fit_transform(df['teaching_method'])
    df['gender'] = le.fit_transform(df['gender'])
    df['lunch'] = le.fit_transform(df['lunch'])
    X = df.drop('posttest', axis=1)
    y = df['posttest']
    return X, y


def pp_churn_modelling(df):
    df.drop_duplicates(inplace=True)
    df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)
    df['Geography'] = le.fit_transform(df['Geography'])
    df['Gender'] = le.fit_transform(df['Gender'])
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    return X, y


def pp_fetal_health(df):
    df.drop_duplicates(inplace=True)
    df.drop(['histogram_mode', 'histogram_median', 'baseline value', 'histogram_width', 'light_decelerations', 'histogram_min', 'histogram_max'], axis=1, inplace=True)
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health']
    return X, y


def pp_mobile_price_classification(df):
    X = df.drop('price_range', axis=1)
    y = df['price_range']
    return X, y


def pp_winequality_red(df):
    df.drop_duplicates(inplace=True)
    df.drop(['total sulfur dioxide'], axis=1, inplace=True)
    X = df.drop('quality', axis=1)
    y = df['quality']
    return X, y


def pp_performance_prediction(df):
    df.drop_duplicates(inplace=True)
    df.drop(['Name', '3PointPercent'], axis=1, inplace=True)
    X = df.drop('Target', axis=1)
    y = df['Target']
    return X, y
