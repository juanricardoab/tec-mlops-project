import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


from src.utils.utils import (
    scale_x_y_data
    )

def test_scale_x_y_data_sizing():
    mock_df_X = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    mock_df_y = pd.DataFrame({"target": [0, 1, 0]})
    X_scaled, y_scaled = scale_x_y_data(mock_df_X, mock_df_y)
    #Revisar que el procesamiento del escalamiento no elimine o agregue registros
    assert X_scaled.size == mock_df_X.size
    assert y_scaled.size == mock_df_y.size

def test_scale_x_y_data_targetProcessed():
    mock_df_X = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    mock_df_y = pd.DataFrame({"target": [0, 1, 0]})
    X_scaled, y_scaled = scale_x_y_data(mock_df_X, mock_df_y)
    #Revisar que la variable de salida si regrese valores sqtr
    assert y_scaled.equals(np.sqrt(mock_df_y))