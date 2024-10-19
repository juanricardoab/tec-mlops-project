import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DataExplorer:
    @staticmethod
    def explore_data(data):
        print("Head: \n", data.head().T, "\n")
        print("Describe: \n", data.describe(include= 'all').round(2), "\n")
        print("Info: \n", data.info(), "\n")
        print("Data duplicated: \n", data.duplicated().sum(), "\n")
        print("Columns: \n", data.columns, "\n")
        for i in data.columns.tolist():
            print(f"Número de valores unicos en la columna {i}: {data[i].nunique()}")
    
    @staticmethod
    def changes_format_data(data, categorical_variables):
        #Converting the 'dteday' column to datetime format
        data['dteday'] = pd.to_datetime(data['dteday'])
        #Dropping the 'instant' column
        data_cleaned = data.drop(columns=['instant'])
        #Converting the variables to categorical
        data_cleaned[categorical_variables] = data[categorical_variables].astype('category')
        return data_cleaned
    
    @staticmethod
    def plot_histograms(data):
        #Histogramas de las variables cuantitativas
        plt.figure(figsize=(12,6))
        plt.subplot(3, 2, 1)
        data['temp'].hist(bins=30) 
        plt.title("temp")
        plt.subplot(3, 2, 2)
        data['atemp'].hist(bins=30) 
        plt.title("atemp")
        plt.subplot(3, 2, 3)
        data['hum'].hist(bins=30) 
        plt.title("hum")
        plt.subplot(3, 2, 4)
        data['windspeed'].hist(bins=30) 
        plt.title("winspeed")
        plt.subplot(3, 2, 5)
        data['casual'].hist(bins=30) 
        plt.title("casual")
        plt.subplot(3, 2, 6)
        data['registered'].hist(bins=30) 
        plt.title("registered")
        plt.tight_layout()
        plt.show()
        
        #Categorical variables
        plt.figure(figsize=(12,8))
        plt.subplot(4, 2, 1)
        sns.countplot(x='season',data=data)
        plt.title("season")
        plt.subplot(4, 2, 2)
        sns.countplot(x='yr',data=data)
        plt.title("year")
        plt.subplot(4, 2, 3)
        sns.countplot(x='mnth',data=data)
        plt.title("month")
        plt.subplot(4, 2, 4)
        sns.countplot(x='hr',data=data)
        plt.title("hour")
        plt.subplot(4, 2, 5)
        sns.countplot(x='holiday',data=data)
        plt.title("holiday")
        plt.subplot(4, 2, 6)
        sns.countplot(x='weekday',data=data)
        plt.title("weekday")
        plt.subplot(4, 2, 7)
        sns.countplot(x='workingday',data=data)
        plt.title("workingday")
        plt.subplot(4, 2, 8)
        sns.countplot(x='weathersit',data=data)
        plt.title("weathersit")
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_distribution_graphs(data):
        # Visualization code for distribution of target variable
        plt.figure(figsize=(8,6))
        sns.distplot(data['cnt'])
        plt.xlabel("Rented Bike Count")
        plt.title('Distribution Plot of Rented Bike Count')
        plt.show()
        
        for col in data.describe().columns:
            fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(13,4))
            sns.histplot(data[col], ax = axes[0],kde = True)
            sns.boxplot(data[col], ax = axes[1],orient='h',showmeans=True,color='pink')
            fig.suptitle("Distribution plot of "+ col, fontsize = 12)
            plt.show()

    @staticmethod
    def plot_correlation_matrix(data):
        plt.figure(figsize=(12,8))
        dfCorrelation = data.corr(method='pearson')
        sns.heatmap(round(dfCorrelation,2), annot=True)
        
    @staticmethod
    def plot_correlation_graphs(data, continuous_variables, dependent_variable, categorical_variables):
        # Analyzing the relationship between the dependent variable and the continuous variables
        for i in continuous_variables:
            plt.figure(figsize=(8,6))
            sns.regplot(x=i,y=dependent_variable[0],data=data)
            plt.ylabel("Rented Bike Count")
            plt.xlabel(i)
            plt.title(i+' vs '+ dependent_variable[0])
            plt.show()
            
        # Analyzing the relationship between the dependent variable and the categorical variables
        for i in categorical_variables:
            plt.figure(figsize=(8,6))
            sns.barplot(x=i,y=dependent_variable[0],data=data)
            plt.ylabel("Rented Bike Count")
            plt.xlabel(i)
            plt.title(i+' vs '+ dependent_variable[0])
            plt.show()
            
    @staticmethod
    def plot_average_rent_over_time(data):
        # Rented bike per hour
        avg_rent_hrs = data.groupby('hr')['cnt'].mean()

        # plot average rent over time(hrs)
        plt.figure(figsize=(10,5))
        sns.lineplot(data=avg_rent_hrs, marker='o')
        plt.ylabel("Rented Bike Count")
        plt.xlabel("Hour")
        plt.title('Average bike rented per hour')
        plt.show()

    @staticmethod
    def final_changes_format_data(data):
        #Eliminando la variable atemp
        data = data.drop(columns=['atemp'])
        data = data.drop(columns=['yr'])