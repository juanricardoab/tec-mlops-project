import math
import pytest
import numpy as np
from bikeSharingModel import BikeSharingModel
from unittest.mock import patch, MagicMock
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TestBikeSharingModel:
    def setup_method(self):
        self.model = BikeSharingModel(275)
        self.model.model = MagicMock()
        self.model.X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        self.model.y = [0, 1, 0, 1, 0]
        
    def test_check_create_object(self):
        assert self.model.fileNumber == 275
        
    def test_train_model(self):
        mock_X = np.array(self.model.X)
        mock_y = np.array(self.model.y)
        mock_X_train = mock_X[:3]
        mock_X_test = mock_X[3:]
        mock_y_train = mock_y[:3]
        mock_y_test = mock_y[3:]
        mock_predictions = np.array([1, 0])

        mock_model = self.model.model
        mock_model.predict.return_value = mock_predictions

        with patch("bikeSharingModel.load_x_y_data", return_value=(mock_X, mock_y)) as mock_load_x_y_data, \
            patch("bikeSharingModel.scale_x_y_data", return_value=(mock_X, mock_y)) as mock_scale_x_y_data, \
            patch("bikeSharingModel.split_data", return_value=(mock_X_train, mock_X_test, mock_y_train, mock_y_test)) as mock_split_data, \
            patch("bikeSharingModel.get_regresion_model", return_value=mock_model) as mock_get_regresion_model:
            
            result = self.model.train_model()
            mock_load_x_y_data.assert_called_once_with("./data/processed/X.csv", "./data/processed/y.csv")
            mock_scale_x_y_data.assert_called_once_with(mock_X, mock_y)
            mock_split_data.assert_called_once_with(mock_X, mock_y)
            mock_get_regresion_model.assert_called_once()
            mock_model.fit.assert_called_once_with(mock_X_train, mock_y_train)
            mock_model.predict.assert_called_once_with(mock_X_test)

            assert np.array_equal(self.model.X_train, mock_X_train)
            assert np.array_equal(self.model.X_test, mock_X_test)
            assert np.array_equal(self.model.y_train, mock_y_train)
            assert np.array_equal(self.model.y_test, mock_y_test)
            assert self.model.model == mock_model
            assert np.array_equal(self.model.predict, mock_predictions)
            assert result is self.model
        
    def test_cross_validate_model(self):
        mock_scores = np.array([0.8, 0.75, 0.78, 0.82, 0.77])

        with patch("bikeSharingModel.cross_val_score", return_value=mock_scores) as mock_cross_val_score:
            result = self.model.cross_validate_model()

            mock_cross_val_score.assert_called_once_with(self.model.model, self.model.X, self.model.y, cv=5, scoring="r2")

            assert np.array_equal(self.model.cv_scores, mock_scores)
            assert self.model.cv_mean_score == pytest.approx(mock_scores.mean())
            assert self.model.cv_std_score == pytest.approx(mock_scores.std())
            assert result is self.model
        
    def test_save_model(self):
        with patch("builtins.open", new_callable=MagicMock) as mock_open, \
            patch("pickle.dump") as mock_pickle_dump:

            self.model.save_model()

            mock_open.assert_called_once_with("./data/models/lin_reg_model.pkl", "wb")
            mock_pickle_dump.assert_called_once_with(self.model.model, mock_open().__enter__())

    def test_load_model(self):
        with patch("builtins.open", new_callable=MagicMock) as mock_open, \
            patch("pickle.load", return_value=self.model.model) as mock_pickle_load:

            loaded_instance = self.model.load_model()

            mock_open.assert_called_once_with("./data/models/lin_reg_model.pkl", "rb")
            mock_pickle_load.assert_called_once_with(mock_open().__enter__())
            
            assert loaded_instance is self.model
            assert loaded_instance.model == self.model.model
        
    def test_train_and_log_model(self):
        X = [[1, 2], [3, 4]]
        y = [0, 1]
        X_train, X_test, y_train, y_test = [[1, 2]], [[3, 4]], [0], [1]
        
        mock_model = MagicMock(spec=LinearRegression)
        mock_model.predict.return_value = y_test
        mock_mse = mean_squared_error(y_test, y_test)
        mock_mae = mean_absolute_error(y_test, y_test)
        mock_r2 = r2_score(y_test, y_test)

        with patch("bikeSharingModel.get_regresion_model", return_value=mock_model), \
            patch("bikeSharingModel.load_x_y_data", return_value=(X, y)), \
            patch("bikeSharingModel.scale_x_y_data", return_value=(X, y)), \
            patch("bikeSharingModel.split_data", return_value=(X_train, X_test, y_train, y_test)), \
            patch("mlflow.set_tracking_uri"), \
            patch("mlflow.set_experiment"), \
            patch("mlflow.start_run"), \
            patch("mlflow.log_metrics") as mock_log_metrics, \
            patch("mlflow.sklearn.log_model") as mock_log_model:

            self.model.train_and_log_model()
            mock_model.fit.assert_called_once_with(X_train, y_train)
            metrics = mock_log_metrics.call_args[0][0]

            assert metrics["MSE"] == pytest.approx(mock_mse)
            assert metrics["MAE"] == pytest.approx(mock_mae)
            assert math.isnan(metrics["r2"]) == math.isnan(mock_r2)

            mock_log_model.assert_called_once_with(mock_model, artifact_path="models")