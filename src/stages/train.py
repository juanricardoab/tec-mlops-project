import argparse

import pandas as pd
import numpy as np
from typing import Text
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import joblib

        
def evaluate_model(config_path) -> None:
    
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    with open(config['train']['model_path'], "rb") as f:
        model = pickle.load(f)

    y_test = pd.read_csv(config['data_split']['y_test'])
    
    y_train = pd.read_csv(config['data_split']['y_train'])
    X_train = pd.read_csv(config['data_split']['x_train'])
    X_test = pd.read_csv(config['data_split']['x_test'])

    y_pred = model.predict(X_train)


   

    y_t = np.square(y_test)
    y_p = np.square(y_pred)
    y_train2 = np.square(y_train)
    y_train_pred = np.square(model.predict(X_train))

    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)
    r2_train = r2_score(y_train2, y_train_pred)
    r2 = r2_score(y_t, y_p)
    r2_adjusted = 1 - (1 - r2) * (
        (len(X_test) - 1) / (len(X_test) - X_test.shape[1] - 1)
    )

    print("MSE :", mse)
    print("RMSE :", rmse)
    print("MAE :", mae)
    print("Train R2 :", r2_train)
    print("Test R2 :", r2)
    print("Adjusted R2 : ", r2_adjusted)

    model_score = [mse, rmse, mae, r2_train, r2, r2_adjusted]
    return model_score



def get_regresion_model():
    return LinearRegression(fit_intercept=True)


def train_model(config_path: Text) -> None:
    """Train model.
    Args:
        config_path {Text}: path to config
    """

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    X_train = pd.read_csv(config['data_split']['x_train'])
    y_train = pd.read_csv(config['data_split']['y_train'])
    X_test = pd.read_csv(config['data_split']['x_test'])
    y_test = pd.read_csv(config['data_split']['y_test'])

    model = get_regresion_model()
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    

    with open(config['train']['model_path'], "wb") as f:
        pickle.dump(model, f)

    y_t = np.square(y_test)
    y_p = np.square(predict)
    y_train2 = np.square(y_train)
    y_train_pred = np.square(model.predict(X_train))

    mse = mean_squared_error(y_t, y_p)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_t, y_p)
    r2_train = r2_score(y_train2, y_train_pred)
    r2 = r2_score(y_t, y_p)
    r2_adjusted = 1 - (1 - r2) * (
        (len(X_test) - 1) / (len(X_test) - X_test.shape[1] - 1)
    )

    print("MSE :", mse)
    print("RMSE :", rmse)
    print("MAE :", mae)
    print("Train R2 :", r2_train)
    print("Test R2 :", r2)
    print("Adjusted R2 : ", r2_adjusted)




    # models_path = config['train']['model_path']
    # joblib.dump(model, models_path)



if __name__ == '__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    train_model(config_path=args.config)