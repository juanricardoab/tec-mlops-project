# Main function for running the pipeline
from bikeSharingModel import BikeSharingModel
import yaml
import mlflow


## Execute BikeSharingModel
#  @Param fileNumber int
## ------------------------

def log_model_scores(model, mlflow):
    score_list = ['mse', 'rmse', 'mae', 'r2_train', 'r2', 'r2_adjusted']
    for index, score in enumerate(model.model_score):
        mlflow.log_metric(score_list[index], score)


def log_model_cv_scores(model, mlflow):
    mlflow.log_metric("cv_mean_score", model.cv_mean_score)
    mlflow.log_metric("cv_std_score", model.cv_std_score)


def main(fileNumber):
    mlflow.set_tracking_uri("http://localhost:5020")
    mlflow.set_experiment("BikeSharingModel")

    with mlflow.start_run() as run:
        mlflow.log_param("fileNumber", fileNumber)
        model = BikeSharingModel(fileNumber)
        model.load_data()
        model.preprocess_data()
        model.train_model()
        model.evaluate_model()
        log_model_scores(model, mlflow)
        model.cross_validate_model()
        log_model_cv_scores(model, mlflow)

        mlflow.sklearn.log_model(model.model, "model")
        # model.save_model()
        # model.load_model()


if __name__ == "__main__":
    with open("./params.yaml") as conf_file:
        config = yaml.safe_load(conf_file)
    main(config['base']['fileNumber'])
