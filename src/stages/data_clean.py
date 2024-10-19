import yaml 
import argparse

import pandas as pd
from dataExplorer import DataExplorer




"""
Load the data form the params.yaml file 
"""
def changes_format_data(data, categorical_variables):
        #Converting the 'dteday' column to datetime format
        data['dteday'] = pd.to_datetime(data['dteday'])
        #Dropping the 'instant' column
        data_cleaned = data.drop(columns=['instant'])
        #Converting the variables to categorical
        data_cleaned[categorical_variables] = data[categorical_variables].astype('category')
        return data_cleaned

def data_clean(config_path) -> None:
    config = yaml.safe_load(open(config_path))

    bike_sharing_df = pd.read_csv(config['data_load']['bike_csv'])
    categorical_variables = config['data_load']['categorical_variable']

    
    data_cleaned = changes_format_data(bike_sharing_df,
                                                    categorical_variables)
    data_cleaned.drop('dteday',inplace=True, axis=1)
    

    data_cleaned.to_csv(config['data_clean']['clean_csv'])

    print('The Data Frame is cleaned')
    

    return None
    

if __name__=='__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_clean(config_path=args.config)
