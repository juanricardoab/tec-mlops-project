import pickle
from bikeSharingModel import BikeSharingModel
import yaml
import mlflow
import glob
import argparse

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


def main(fileNumber, model_type="linear"):
    mlflow.set_tracking_uri("http://localhost:5020")
    mlflow.set_experiment(f"BikeSharingModel_{model_type.capitalize()}")
    images_path = "./data/processed/"

    with mlflow.start_run() as run:
        mlflow.log_param("fileNumber", fileNumber)
        mlflow.log_param("images_path", images_path)
        model = BikeSharingModel(fileNumber, model_type=model_type)
        model.load_data()
        model.preprocess_data()
        model.train_model()
        model.evaluate_model()
        log_model_scores(model)
        model.cross_validate_model()
        log_model_cv_scores(model)
        with open("./app/models/model_final.pkl.joblib", "wb") as f:
            pickle.dump(model, f)
        mlflow.sklearn.log_model(model.model, "model")
        load_graphs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BikeSharingModel with specified config file.")
    parser.add_argument(
        "--config",
        type=str,
        default="./params.yaml",
        help="Path to the configuration file (params.yaml)"
    )
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, "r") as conf_file:
        config = yaml.safe_load(conf_file)

    # Extract parameters from config
    file_number = config['base']['fileNumber']
    model_type = config['base']['model_type']

    # Run main function with extracted parameters
    main(file_number, model_type)
