from sklearn.model_selection import train_test_split
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score


def loading_data(path):
    df = pd.read_csv(path)
    return df


def splitting_data(df):
    y = df['Survived']
    x = df.drop(columns=['Survived'], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42, shuffle=True)
    return x_train, x_test, y_train, y_test


def log_model(model, params, metrics, model_name):
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, model_name)


def evaluate_model(y_train_pred, y_test_pred, y_train, y_test):
    return {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "train_precision": precision_score(y_train, y_train_pred),
        "train_recall": recall_score(y_train, y_train_pred),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_precision": precision_score(y_test, y_test_pred),
        "test_recall": recall_score(y_test, y_test_pred)
    }
