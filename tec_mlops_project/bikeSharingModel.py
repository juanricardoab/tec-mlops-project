import pandas as pd
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
from datetime import datetime as dt

from dataExplorer import DataExplorer
from preprocessData import PreprocessData

class BikeSharingModel:
    def __init__(self, fileNumber):
        self.fileNumber = fileNumber
        # defining continuous, categorical and dependent variable
        self.continuous_variables = ['temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered']
        self.categorical_variables = ['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
        self.dependent_variable = ['cnt']
        ##self.model_pipeline = Pipeline([
        ##    ('scaler', StandardScaler()),
        ##    ('classifier', LogisticRegression(max_iter=1000))
        ##])
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self):
        bike_sharing = fetch_ucirepo(id=self.fileNumber) 
        self.bike_sharing_df = bike_sharing.data.original
        DataExplorer.explore_data(self.bike_sharing_df)
        self.data_cleaned = DataExplorer.changes_format_data(self.bike_sharing_df, self.categorical_variables)
        DataExplorer.explore_data(self.data_cleaned)
        DataExplorer.plot_histograms(self.data_cleaned)
        DataExplorer.plot_distribution_graphs(self.data_cleaned)
        DataExplorer.plot_correlation_matrix(self.data_cleaned)
        DataExplorer.plot_correlation_graphs(self.data_cleaned, self.continuous_variables, self.dependent_variable, self.categorical_variables)
        DataExplorer.plot_average_rent_over_time(self.data_cleaned)
        return self

    def preprocess_data(self):
        columns_to_encode = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday', 'weathersit']
        
        # Aplicar one-hot encoding a las columnas en una sola iteración
        for column in columns_to_encode:
            self.data_cleaned_oneHot = PreprocessData.one_hot_encoding(self.data_cleaned_oneHot, column)
        
        # Aplicar el escalado
        PreprocessData.min_max_scaler(self.data_cleaned_oneHot)
        
        # Separar las características (X) y el objetivo (y)
        self.X = self.data_cleaned_oneHot.drop(columns=['cnt'])
        self.y = self.data_cleaned_oneHot['cnt']
        
        # Guardar en CSV (esto podría ser modularizado si deseas reutilizar la lógica)
        self._save_to_csv(self.X, self.y)
        
        return self

    def _save_to_csv(self, X, y):
        """Guarda las variables X e y en archivos CSV."""
        X.to_csv('../data/processed/X.csv', index=False)
        y.to_csv('../data/processed/y.csv', index=False)
    
    def train_model(self):

        return self
    
    def evaluate_model(self):

        return self
    
    def cross_validate_model(self):

        return self
