# Main function for running the pipeline
from bikeSharingModel import BikeSharingModel
import yaml
import mlflow
import glob


## Execute BikeSharingModel
#  @Param fileNumber int
## ------------------------

def log_model_scores(model):
    score_list = ['mse', 'rmse', 'mae', 'r2_train', 'r2', 'r2_adjusted']
    for index, score in enumerate(model.model_score):
        mlflow.log_metric(score_list[index], score)


def log_model_cv_scores(model):
    mlflow.log_metric("cv_mean_score", model.cv_mean_score)
    mlflow.log_metric("cv_std_score", model.cv_std_score)

def load_graphs():
    for file in glob.glob("./data/processed/*.png"):
        mlflow.log_artifact(file)

def main(fileNumber):
    #log model and experiments step
    model = BikeSharingModel(fileNumber)
    model.load_data()
    model.preprocess_data()
    model.train_and_log_model()   
    load_graphs()


if __name__ == "__main__":
    with open("./params.yaml") as conf_file:
        config = yaml.safe_load(conf_file)
    main(config['base']['fileNumber'])
