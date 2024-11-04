import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor


def load_x_y_data(pathX, pathY):
    X = pd.read_csv("./data/processed/X.csv")
    y = pd.read_csv("./data/processed/y.csv")
    return X, y


def scale_x_y_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = np.sqrt(y)
    return X_scaled, y_scaled


def split_data(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


# src/utils/utils.py

def get_regresion_model(model_type="linear"):
    """
    Returns a regression model based on the specified model type.

    Parameters:
        model_type (str): The type of regression model to return. Options include:
                          "linear", "ridge", "lasso", "elasticnet", "decision_tree",
                          "random_forest", "svr", and "knn".

    Returns:
        model: The regression model instance.
    """
    if model_type == "linear":
        return LinearRegression()
    elif model_type == "ridge":
        return Ridge()
    elif model_type == "lasso":
        return Lasso()
    elif model_type == "elasticnet":
        return ElasticNet()
    elif model_type == "decision_tree":
        return DecisionTreeRegressor()
    elif model_type == "random_forest":
        return RandomForestRegressor()
    elif model_type == "svr":
        return SVR()
    elif model_type == "knn":
        return KNeighborsRegressor()
    else:
        raise ValueError(f"Unsupported model type '{model_type}'. Supported types are: "
                         "'linear', 'ridge', 'lasso', 'elasticnet', 'decision_tree', "
                         "'random_forest', 'svr', and 'knn'.")


def get_svr_model(params={"kernel": "linear", "C": 1.0}):
    return SVR(**params)


def get_knn_model(params={"n_neighbors": 5}):
    return KNeighborsRegressor(**params)


def get_random_forest_model(params={"n_estimators": 100, "max_depth": None}):
    return RandomForestRegressor(**params)


def get_decision_tree_model(params={"max_depth": None}):
    return DecisionTreeRegressor(**params)


def evaluate_model(model, X_train, X_test, y_train, y_test, y_pred):
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
