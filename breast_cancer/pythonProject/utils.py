import pandas as pd
from sklearn.model_selection import train_test_split
import joblib


def loading_data():
    df = pd.read_csv(r'breast-cancer.csv')
    df.drop(columns=['id'], axis=1, inplace=True)
    return df


def checking_nulls_duplicates(df):
    nulls = df.isna().sum().sum()
    duplicates = df.duplicated().sum()
    return nulls, duplicates


def splitting_y_x(df):
    x = df.drop(columns=['diagnosis'], axis=1)
    y = df['diagnosis']
    return x, y


def checking_possible_classes(y):
    values = []
    for v in y:
        if not (v in values):
            values.append(v)
    print('possible classes:', values)


def converting_classes_into_0_1(y):
    y = y.replace({'M': 1, 'B': 0})
    return y


def split_data_into_train_test(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
    return x_train, x_test, y_train, y_test


def fitting_scaler(scaler,x_train):
    x_train = scaler.fit_transform(x_train)
    return x_train


def transform_using_scaler(scaler, x_test):
    x_test = scaler.transform(x_test)
    return x_test


def saving_model(model, scaler):
    joblib.dump(model, 'svc_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')


def loading_model():
    model = joblib.load('svc_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler


def model_prediction(model, x):
    return model.predict(x)


def prediction_mapping(pred):
    if pred == 0:
        return 'B'
    else:
        return 'M'
