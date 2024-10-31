import pytest
import pandas as pd
from unittest.mock import patch
from sklearn.linear_model import LinearRegression

from src.utils.utils import (
    load_x_y_data,
    get_regresion_model
    )

def test_load_x_y_data():
    mock_df_X = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    mock_df_y = pd.DataFrame({"target": [0, 1, 0]})

    with patch("pandas.read_csv", side_effect=[mock_df_X, mock_df_y]) as mock_read_csv:
        X, y = load_x_y_data("fake_path_X.csv", "fake_path_Y.csv")

        mock_read_csv.assert_any_call("fake_path_X.csv")
        mock_read_csv.assert_any_call("fake_path_Y.csv")

        pd.testing.assert_frame_equal(X, mock_df_X)
        pd.testing.assert_frame_equal(y, mock_df_y)
        
def test_get_regresion_model_default():
    model = get_regresion_model()
    assert isinstance(model, LinearRegression)
    assert model.fit_intercept is True

def test_get_regresion_model_custom_params():
    custom_params = {"fit_intercept": False}
    model = get_regresion_model(custom_params)
    assert isinstance(model, LinearRegression)
    assert model.fit_intercept is False