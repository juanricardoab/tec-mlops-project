
# Main function for running the pipeline
from bikeSharingModel import BikeSharingModel

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
    
if __name__ == "__main__":
    main(275)