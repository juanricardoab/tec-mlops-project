import yaml 

"""
Load the data form the params.yaml file 
"""


with open("./params.yaml") as conf_file:
    config = yaml.safe_load(conf_file)

print(config['data_load']['fileNumber'])