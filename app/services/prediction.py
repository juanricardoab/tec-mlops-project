import joblib
import os

from app.src.report.explore import DataExplorer
from app.src.stages.preprocess import PreprocessData
from sklearn.preprocessing import StandardScaler


class PredictionService:
    def __init__(self):
        self.model = None

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = joblib.load(model_path)
        return self.model

    def preprocess_data(self, data):
        bike_sharing_df = data
        DataExplorer.explore_data(bike_sharing_df)
        data_cleaned = DataExplorer.changes_format_data(
            bike_sharing_df, [
                "season",
                "yr",
                "mnth",
                "hr",
                "holiday",
                "weekday",
                "workingday",
                "weathersit",
            ]
        )
        DataExplorer.explore_data(data_cleaned)
        data_cleaned_oneHot = PreprocessData.one_hot_encoding(
            data_cleaned, "season"
        )
        # Repeat one-hot encoding for other categorical variables
        for column in ["mnth", "hr", "holiday", "weekday", "workingday", "weathersit"]:
            data_cleaned_oneHot = PreprocessData.one_hot_encoding(
                data_cleaned_oneHot, column
            )

        PreprocessData.min_max_scaler(data_cleaned_oneHot)
        X = data_cleaned_oneHot.drop(columns=["dteday"], errors='ignore')

        expected_columns = [
            "yr", "temp", "atemp", "hum", "windspeed", "casual", "registered",
            "season_1", "season_2", "season_3", "season_4", "mnth_1", "mnth_2",
            "mnth_3", "mnth_4", "mnth_5", "mnth_6", "mnth_7", "mnth_8", "mnth_9",
            "mnth_10", "mnth_11", "mnth_12", "hr_0", "hr_1", "hr_2", "hr_3", "hr_4",
            "hr_5", "hr_6", "hr_7", "hr_8", "hr_9", "hr_10", "hr_11", "hr_12",
            "hr_13", "hr_14", "hr_15", "hr_16", "hr_17", "hr_18", "hr_19", "hr_20",
            "hr_21", "hr_22", "hr_23", "holiday_0", "holiday_1", "weekday_0", "weekday_1",
            "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6", "workingday_0",
            "workingday_1", "weathersit_1", "weathersit_2", "weathersit_3", "weathersit_4"
        ]

        for col in expected_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[expected_columns]

        return X

    def predict(self, data):
        predictions = self.model.predict(data)
        return predictions
