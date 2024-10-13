# Main function for running the pipeline
from bikeSharingModel import BikeSharingModel
import yaml


## Execute BikeSharingModel
#  @Param fileNumber int
## ------------------------
def main(fileNumber):
    model = BikeSharingModel(fileNumber)
    model.load_data()
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.cross_validate_model()
    # model.save_model()
    # model.load_model()


if __name__ == "__main__":
    with open("./params.yaml") as conf_file:
        config = yaml.safe_load(conf_file)
    main(config['data_load']['fileNumber'])
