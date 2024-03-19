import yaml

def load_config(config_path = 'pipeline/configs/config.yaml'):
    with open(config_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data