from os.path import dirname, abspath, join

import yaml


def load_config():
    config_dir = join(dirname(dirname(abspath(__file__))), 'config')
    with open(join(config_dir, 'config.yaml')) as f:
        base_config = yaml.load(f.read(), Loader=yaml.FullLoader)
        with open(join(config_dir, base_config['arg_file'])) as f1:
            configs = yaml.load(f1.read(), Loader=yaml.FullLoader)
    return configs
