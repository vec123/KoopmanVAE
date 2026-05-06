import yaml

class Config:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            if isinstance(v, dict):
                v = Config(v)
            self.__dict__[k] = v

    @staticmethod
    def from_yaml(path):
        with open(path, 'r') as f:
            return Config(yaml.safe_load(f))