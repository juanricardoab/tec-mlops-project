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
    mlflow.set_tracking_uri("http://localhost:5020")
    mlflow.set_experiment("BikeSharingModel")
    images_path = "./data/processed/"

    with mlflow.start_run() as run:
        mlflow.log_param("fileNumber", fileNumber)
        mlflow.log_param("images_path", images_path)
        model = BikeSharingModel(fileNumber)
        model.load_data()
        model.preprocess_data()
        model.train_model()
        model.evaluate_model()
        log_model_scores(model)
        model.cross_validate_model()
        log_model_cv_scores(model)

        mlflow.sklearn.log_model(model.model, "model")
        load_graphs()
        # model.save_model()
        # model.load_model()


if __name__ == "__main__":
    with open("./params.yaml") as conf_file:
        config = yaml.safe_load(conf_file)
    main(config['base']['fileNumber'])
