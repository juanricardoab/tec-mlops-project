import yaml 
import argparse
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_x_y_data(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    y_scaled = np.sqrt(y)
    return X_scaled, y_scaled


def split_data(X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def data_split(config_path) -> None:
    config = yaml.safe_load(open(config_path))

    X = pd.read_csv(config['preprocess']['x'])
    y = pd.read_csv(config['preprocess']['y'])
    test_size = config['data_split']['test_size']
    random_state = config['base']['seed']

    
    X, y = scale_x_y_data(X, y)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    
    X_train.to_csv(config['data_split']['x_train'])
    y_train.to_csv(config['data_split']['y_train'])
    
    X_test.to_csv(config['data_split']['x_test'])
    y_test.to_csv(config['data_split']['y_test'])

    print('Data split completed')




if __name__=='__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)