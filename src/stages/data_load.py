import yaml 
import argparse
from ucimlrepo import fetch_ucirepo

"""
Load the data form the params.yaml file 
"""
def data_load(config_path) -> None:
    config = yaml.safe_load(open(config_path))
    fileNumber = config['base']['fileNumber']
    bike_sharing = fetch_ucirepo(id = fileNumber)
    bike_sharing_df = bike_sharing.data.original
    print('Bike sharing file downloaded')

    bike_sharing_df.to_csv(config['data_load']['bike_csv'])
    
    print('Bike.csv file saved')
    return None
    

if __name__=='__main__':

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)



