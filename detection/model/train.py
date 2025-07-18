from ultralytics import YOLO
import argparse
import wandb
import os
import sys
import torch

# aggiungo al path la cartella padre per importare i moduli
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# torch.backends.cudnn.benchmark = False

from utils.params import load_config
from utils import timer


# USAGE: train --config path/to/config.yaml
def train():
    # parser per i paramerticl
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file", default='config/config.yaml')
    args = parser.parse_args()

    # prendi il dizionario dei parametri
    config = load_config(args.config)

    if config['wandb']:
        print('Questo train verrà loggato su wandb')
        # prendi la chiave per wandb
        secrets = load_config("config/secrets.yaml")

        # Wandb configuration
        wandb.login(key=secrets['wandb_api_key'])
        wandb.init(
            project='VMART-ObjectDetection',
            config=config,
            name=config['experiment'],
            entity=config['entity'],
            save_code=True
        )
    else:
        print('Questo train NON verrà loggato su wandb')

    config = load_config(args.config)

    config['train']['name'] = config['experiment']

    # inizializza il modello di yolo
    model = YOLO(config['train']['model'])

    # parametri del train
    train_conf = config['train']

    # prendi tutti i parametri tranne il model
    train_params = {k: v for k, v in train_conf.items() if k != 'model'}

    with timer.Timer('Train YOLO'):
        # fai partire il train
        model.train(**train_params)

if __name__ == '__main__':
    train()
