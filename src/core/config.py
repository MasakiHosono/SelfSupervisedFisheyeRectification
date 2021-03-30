import os
import yaml

class Config:
    def __init__(self, config_file):
        self.config_file = config_file

        if not os.path.exists(self.config_file):
            print('ERROR: No such file or directory {}'.format(self.config_file))
            exit(1)

        with open(self.config_file, 'r') as f:
            self.config = yaml.load(f)

    def getDict(self):
        return self.config
