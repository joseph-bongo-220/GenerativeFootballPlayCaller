import yaml

def get_config():
    with open("config.yml") as file:
        config = yaml.load(file)
    return config