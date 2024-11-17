from src.utils.utils import (
    evaluate_model,
    get_regresion_model,
    load_x_y_data,
    scale_x_y_data,
    split_data,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score
from src.stages.preprocess import PreprocessData
from src.utils.dataExplorer import DataExplorer
import pickle
import sys
import os
import mlflow
from mlflow.models import infer_signature

# Agregar la raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))


class BikeSharingModel:
    def __init__(self, fileNumber, model_type="linear"):
        self.fileNumber = fileNumber
        self.model_type = model_type  # Added model_type to select the regression model
        # defining continuous, categorical and dependent variable
        self.continuous_variables = [
            "temp",
            "atemp",
            "hum",
            "windspeed",
            "casual",
            "registered",
        ]
        self.categorical_variables = [
            "season",
            "yr",
            "mnth",
            "hr",
            "holiday",
            "weekday",
            "workingday",
            "weathersit",
        ]
        self.dependent_variable = ["cnt"]
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self, image_path='./app/data/processed/'):
        bike_sharing = fetch_ucirepo(id=self.fileNumber)
        self.bike_sharing_df = bike_sharing.data.original
        DataExplorer.explore_data(self.bike_sharing_df)
        self.data_cleaned = DataExplorer.changes_format_data(
            self.bike_sharing_df, self.categorical_variables
        )
        DataExplorer.explore_data(self.data_cleaned)
#        DataExplorer.plot_histograms(self.data_cleaned, image_path)
        # DataExplorer.plot_distribution_graphs(self.data_cleaned, image_path)
        # DataExplorer.plot_correlation_matrix(self.data_cleaned, image_path)
        return self

    def preprocess_data(self):
        self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            self.data_cleaned, "season"
        )
        # Repeat one-hot encoding for other categorical variables
        for column in ["mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]:
            self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
                self.data_cleaned_oneHot, column
            )

        PreprocessData.min_max_scaler(self.data_cleaned_oneHot)
        self.X = self.data_cleaned_oneHot.drop(columns=["cnt", "dteday"])
        self.y = self.data_cleaned_oneHot["cnt"]
        # save X and y to csv

        self.X.to_csv("./app/data/processed/X.csv", index=False)
        self.y.to_csv("./app/data/processed/y.csv", index=False)
        return self

    def train_model(self):
        self.X, self.y = load_x_y_data(
            "./app/data/processed/X.csv", "./app/data/processed/y.csv"
        )
        self.X, self.y = scale_x_y_data(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            self.X, self.y
        )
        self.model = get_regresion_model(self.model_type)  # Pass model_type to get_regresion_model
        self.model.fit(self.X_train, self.y_train)
        self.predict = self.model.predict(self.X_test)
        return self

    def evaluate_model(self):
        self.model_score = evaluate_model(
            self.model,
            self.X_train,
            self.X_test,
            self.y_train,
            self.y_test,
            self.predict,
        )
        return self

    def cross_validate_model(self):
        scores = cross_val_score(self.model, self.X, self.y, cv=5, scoring="r2")
        self.cv_scores = scores
        self.cv_mean_score = scores.mean()
        self.cv_std_score = scores.std()

        print(f"Cross-Validation Scores: {scores}")
        print(f"Mean R2 Score: {self.cv_mean_score}")
        print(f"Standard Deviation of R2 Score: {self.cv_std_score}")

        return self

    def save_model(self, model_path):
        with open(model_path, "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, model_path):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        return self

    def train_and_log_model(self):
        model_name = self.model_type.capitalize() + "Regression"
        self.X, self.y = load_x_y_data(
            "./app/data/processed/X.csv", "./app/data/processed/y.csv"
        )
        self.X, self.y = scale_x_y_data(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            self.X, self.y
        )

        mlflow.set_tracking_uri("http://localhost:5020")
        mlflow.set_experiment(f"BikeSharingModel_{model_name}")

        with mlflow.start_run(run_name=model_name):
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)
            mse = mean_squared_error(self.y_test, y_pred)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            mlflow.log_metrics({"MSE": mse, "MAE": mae, "r2": r2})
            signature = infer_signature(self.X_test, self.model.predict(self.X_test))
            mlflow.sklearn.log_model(self.model, artifact_path="models", signature=signature)
