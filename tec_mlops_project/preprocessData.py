import pandas as pd
from sklearn import preprocessing
from datetime import datetime as dt

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