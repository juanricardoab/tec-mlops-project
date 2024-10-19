import pandas as pd
from sklearn import preprocessing
from datetime import datetime as dt
import yaml 
import argparse

class PreprocessData:
    
    @staticmethod
    def one_hot_encoding(df_original, columna_encoding):
        dummies = pd.get_dummies(df_original[[columna_encoding]])
        res_df = pd.concat([df_original, dummies], axis=1)
        res_df.drop([columna_encoding], axis='columns', inplace = True)
        return(res_df)
    
    @staticmethod
    def min_max_scaler(data):
        preproc= preprocessing.MinMaxScaler()
        preproc.fit(data[['temp','hum','windspeed', 'casual', 'registered']])
        minmax_df=pd.DataFrame(preproc.transform(data[['temp','hum','windspeed', 'casual', 'registered']]), columns=['temp','hum','windspeed', 'casual', 'registered'])
        print(minmax_df.describe())


def preprocess_data(config_path) -> None:
    config = yaml.safe_load(open(config_path))
    data_cleaned = pd.read_csv(config['data_clean']['clean_csv'])

    data_cleaned_oneHot = PreprocessData.one_hot_encoding(data_cleaned, 'season')
    data_cleaned_oneHot = PreprocessData.one_hot_encoding(data_cleaned_oneHot, 'mnth')
    data_cleaned_oneHot = PreprocessData.one_hot_encoding(data_cleaned_oneHot, 'hr')
    data_cleaned_oneHot = PreprocessData.one_hot_encoding(data_cleaned_oneHot, 'holiday')
    data_cleaned_oneHot = PreprocessData.one_hot_encoding(data_cleaned_oneHot, 'weekday')
    data_cleaned_oneHot = PreprocessData.one_hot_encoding(data_cleaned_oneHot, 'workingday')
    data_cleaned_oneHot = PreprocessData.one_hot_encoding(data_cleaned_oneHot, 'weathersit')
    
    PreprocessData.min_max_scaler(data_cleaned_oneHot)
    X = data_cleaned_oneHot.drop(columns=['cnt'])
    y = data_cleaned_oneHot['cnt']
    #save X and y to csv
    X.to_csv(config['preprocess']['x'], index=False)
    y.to_csv(config['preprocess']['y'], index=False)

    print('The X and y variables are saved in a csv file ')
    return None

if __name__=='__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    preprocess_data(config_path=args.config)