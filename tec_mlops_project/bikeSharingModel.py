import pandas as pd
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
from datetime import datetime as dt

from dataExplorer import DataExplorer

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
        
        return self
    
    def train_model(self):

        return self
    
    def evaluate_model(self):

        return self
    
    def cross_validate_model(self):

        return self
