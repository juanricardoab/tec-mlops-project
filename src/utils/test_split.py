import pytest
from unittest.mock import patch
from sklearn.model_selection import train_test_split
from unittest.mock import MagicMock
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

import os
import sys

script_dir = os.path.dirname( __file__ )
mymodule_dir = os.path.join( script_dir, '..', 'src', 'utils' )
sys.path.append( mymodule_dir )


from utils import split_data, evaluate_model

def test_split_data():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    test_size = 0.2
    random_state = 42

    with patch('utils.train_test_split') as mock_train_test_split:
        mock_train_test_split.return_value = (X[:4], X[4:], y[:4], y[4:])
        
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
        
        s
        mock_train_test_split.assert_called_once_with(X, y, test_size=test_size, random_state=random_state)
        
       
        assert (X_train == X[:4]).all()
        assert (X_test == X[4:]).all()
        assert (y_train == y[:4]).all()
        assert (y_test == y[4:]).all()

def test_evaluate_model():
    
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    X_test = np.array([[7, 8], [9, 10]])
    y_train = np.array([1, 2, 3])
    y_test = np.array([4, 5])
    y_pred = np.array([4.1, 4.9])
    
    
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1.1, 2.1, 3.1])

    model_score = evaluate_model(mock_model, X_train, X_test, y_train, y_test, y_pred)


    y_t = np.square(y_test)
    y_p = np.square(y_pred)
    y_train2 = np.square(y_train)
    y_train_pred = np.square(mock_model.predict(X_train))

    expected_mse = mean_squared_error(y_t, y_p)
    expected_rmse = np.sqrt(expected_mse)
    expected_mae = mean_absolute_error(y_t, y_p)
    expected_r2_train = r2_score(y_train2, y_train_pred)
    expected_r2 = r2_score(y_t, y_p)
    expected_r2_adjusted = 1 - (1 - expected_r2) * (
        (len(X_test) - 1) / (len(X_test) - X_test.shape[1] - 1)
    )


    assert model_score[0] == expected_mse
    assert model_score[1] == expected_rmse
    assert model_score[2] == expected_mae
    assert model_score[3] == expected_r2_train
    assert model_score[4] == expected_r2
    assert model_score[5] == expected_r2_adjusted

if __name__ == '__main__':
    pytest.main()