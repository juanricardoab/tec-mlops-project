
# Main function for running the pipeline
from tec_mlops_project.bikeSharingModel import BikeSharingModel

def main(filepath):
    model = BikeSharingModel(filepath)
    model.load_data()
    model.preprocess_data()
    model.train_model()
    model.evaluate_model()
    model.cross_validate_model()