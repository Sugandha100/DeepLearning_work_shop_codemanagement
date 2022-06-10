import os

import yaml

from torch import cuda

config = [
    {
        'model_params': {
            'lit_resnet': {
                'name': 'LitResnet',
                'max_epochs': 30,
                'accelerator': 'auto',
                'pretrained': False,
                'num_classes': 10,
                'lr': 0.05
            },
            'swa_resnet':{
                'name': 'SWAResnet',
                'lr': 0.01
            }
        }
    },
    {
        'exp_params': {
            'PATH_DATASETS': os.environ.get("PATH_DATASETS", "data"),
            'BATCH_SIZE': 256 if cuda.is_available() else 64,
            'NUM_WORKERS': int(os.cpu_count()),
            'METRICS_DIR': 'metrics'
        }
    },
    {
        'log_params':{
            'save_dir': 'logs/'
        }
    }
]

# Write to yaml file
def write_config():
    with open('config/config.yaml', 'w') as yaml_file:
        yaml.dump(config, yaml_file)

def load_config():
     # Load configs
    config_file = open('./config/config.yaml', mode='r')
    return yaml.load(config_file, Loader=yaml.FullLoader)


if __name__ == '__main__':
    write_config()