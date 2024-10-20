import pickle

import mlflow

from src.utils.dataExplorer import DataExplorer
from src.stages.preprocess import PreprocessData
from sklearn.model_selection import cross_val_score
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.utils.utils import (
    evaluate_model,
    get_regresion_model,
    load_x_y_data,
    scale_x_y_data,
    split_data,
)


class BikeSharingModel:
    def __init__(self, fileNumber):
        self.fileNumber = fileNumber
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
        ##self.model_pipeline = Pipeline([
        ##    ('scaler', StandardScaler()),
        ##    ('classifier', LogisticRegression(max_iter=1000))
        ##])
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self, image_path='./data/processed/'):
        bike_sharing = fetch_ucirepo(id=self.fileNumber)
        self.bike_sharing_df = bike_sharing.data.original
        DataExplorer.explore_data(self.bike_sharing_df)
        self.data_cleaned = DataExplorer.changes_format_data(
            self.bike_sharing_df, self.categorical_variables
        )
        DataExplorer.explore_data(self.data_cleaned)
        DataExplorer.plot_histograms(self.data_cleaned, image_path)
        DataExplorer.plot_distribution_graphs(self.data_cleaned, image_path)
        DataExplorer.plot_correlation_matrix(self.data_cleaned, image_path)
        # DataExplorer.plot_correlation_graphs(
        #     self.data_cleaned,
        #     self.continuous_variables,
        #     self.dependent_variable,
        #     self.categorical_variables,
        # )
        # DataExplorer.plot_average_rent_over_time(self.data_cleaned)
        return self

    def preprocess_data(self):
        self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            self.data_cleaned, "season"
        )
        self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            self.data_cleaned_oneHot, "mnth"
        )
        self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            self.data_cleaned_oneHot, "hr"
        )
        self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            self.data_cleaned_oneHot, "holiday"
        )
        self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            self.data_cleaned_oneHot, "weekday"
        )
        self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            self.data_cleaned_oneHot, "workingday"
        )
        self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            self.data_cleaned_oneHot, "weathersit"
        )

        PreprocessData.min_max_scaler(self.data_cleaned_oneHot)
        self.X = self.data_cleaned_oneHot.drop(columns=["cnt", "dteday"])
        self.y = self.data_cleaned_oneHot["cnt"]
        # save X and y to csv
        self.X.to_csv("./data/processed/X.csv", index=False)
        self.y.to_csv("./data/processed/y.csv", index=False)
        return self

    def train_model(self):
        self.X, self.y = load_x_y_data(
            "./data/processed/X.csv", "./data/processed/y.csv"
        )
        self.X, self.y = scale_x_y_data(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            self.X, self.y
        )
        self.model = get_regresion_model()
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
        with open("./data/models/lin_reg_model.pkl", "wb") as f:
            pickle.dump(self.model, f)

    def load_model(self, model_path):
        with open("./data/models/lin_reg_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        return self
    
    def train_and_log_model(self):
        params_lr = {"C": 1.0, "solver": "liblinear", "random_state": 42}
        model_lr = get_regresion_model(params=params_lr)
        model_name = "LinearRegression"
        self.X, self.y = load_x_y_data(
            "./data/processed/X.csv", "./data/processed/y.csv"
        )
        self.X, self.y = scale_x_y_data(self.X, self.y)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            self.X, self.y
        )
        
        mlflow.set_tracking_uri("http://localhost:5020")
        mlflow.set_experiment(f"BikeSharingModel_{model_name}")

        with mlflow.start_run(run_name=model_name):
            model_lr.fit(self.X_train, self.y_train)
            y_pred = model_lr.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            prec = precision_score(self.y_test, y_pred, average='weighted')
            rec = recall_score(self.y_test, y_pred, average='weighted')
            mlflow.log_params(params_lr)
            mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})
            mlflow.sklearn.log_model(model_lr, artifact_path="models")
