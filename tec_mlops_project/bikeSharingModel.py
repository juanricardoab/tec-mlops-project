import pandas as pd
from ucimlrepo import fetch_ucirepo 
import seaborn as sns
from datetime import datetime as dt

from dataExplorer import DataExplorer

class BikeSharingModel:
    def __init__(self, fileNumber):
        self.fileNumber = fileNumber
        ##self.model_pipeline = Pipeline([
        ##    ('scaler', StandardScaler()),
        ##    ('classifier', LogisticRegression(max_iter=1000))
        ##])
        self.X_train, self.X_test, self.y_train, self.y_test = [None] * 4

    def load_data(self):
        bike_sharing = fetch_ucirepo(id=self.fileNumber) 
        self.bike_sharing_df = bike_sharing.data.original
        DataExplorer.explore_data(self.bike_sharing_df)
        return self

    def preprocess_data(self):
        
        return self
    
    def train_model(self):

        return self
    
    def evaluate_model(self):

        return self
    
    def cross_validate_model(self):

        return self
