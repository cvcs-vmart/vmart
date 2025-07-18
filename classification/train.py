import argparse
import json
import os

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.models import EfficientNet_V2_L_Weights

from Dataset import WikiArtDataset
from VMARTResnet import VMARTResnet
from WandbManager import WandbManager

from collections import Counter
from torch.utils.data import WeightedRandomSampler

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VMART ResNet")
    parser.add_argument('--config', type=str, default='config/config.yaml')

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    torch.backends.cudnn.enabled = config['useCudnn']

    isWandEnabled = config['wandb']['enabled']
    if isWandEnabled:
        wandb_ = WandbManager(config['wandb']['project'], config['wandb']['entity'], config['wandb']['name'],
                              config['train'])
    else:
        wandb_ = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATASET_PATH = config['dataset']['path']
    CSV_PATH = config['dataset']['csv']

    preprocess = EfficientNet_V2_L_Weights.DEFAULT.transforms()

    dataset = WikiArtDataset(csv_file=CSV_PATH, root_dir=DATASET_PATH, transform=preprocess)

    if config['train']['resume']['checkpoint'] and config['train']['resume']['split_indices']:
        print(f"Loading dataset split from {config['train']['resume']['split_indices']}")
        with open(config['train']['resume']['split_indices'], 'r') as f:
            indices = json.load(f)
        train = torch.utils.data.Subset(dataset, indices['train'])
        test = torch.utils.data.Subset(dataset, indices['test'])
    else:
        if config['train']['resume']['checkpoint']:
            print("Warning: Resuming training without a specified dataset split. A new random split will be used.")

        print("Creating new dataset split.")
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train, test = torch.utils.data.random_split(dataset, [train_size, test_size])

        os.makedirs("splits", exist_ok=True)
        split_indices_path = "splits/split_indices.json"
        with open(split_indices_path, 'w') as f:
            json.dump({'train': train.indices, 'test': test.indices}, f)

        if wandb_ is not None:
            wandb_.save("splits/split_indices.json")

        print(f"Dataset split indices saved to '{split_indices_path}'.")
        print(
            "To resume training with this exact split, set 'train.resume.split_indices' in your config file to this path.")

    BATCH_SIZE = config['train']['batch_size']
    NUM_WORKERS = config['train']['n_workers']

    sampler = None
    # Se l'oversampling è abilitato, crea un sampler pesato per il training set
    if config['train'].get('oversampling', False):
        print("Applicazione dell'oversampling per il training set basato sullo stile.")
        # Ottieni le etichette di stile per il set di addestramento
        s=train.indices
        train_styles = [dataset.style2idx[dataset.annotations.iloc[i, 3]] for i in train.indices]

        # Conta le frequenze delle classi
        class_counts = Counter(train_styles)

        # Calcola il peso per ogni classe (inverso della frequenza)
        class_weights = {c: 1.0 / count for c, count in class_counts.items()}

        # Assegna un peso a ogni campione nel set di addestramento
        sample_weights = [class_weights[style] for style in train_styles]

        sampler = WeightedRandomSampler(torch.DoubleTensor(sample_weights), len(sample_weights))

    # Il test_loader non dovrebbe essere mescolato per una valutazione coerente
    test_loader = DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=NUM_WORKERS)
    # L'opzione shuffle e il sampler sono mutuamente esclusivi
    train_loader = DataLoader(train, batch_size=BATCH_SIZE, shuffle=(sampler is None), sampler=sampler, pin_memory=True,
                              num_workers=NUM_WORKERS)
    model = VMARTResnet(config["train"], device, wandb_)
    model = model.to(device)

    if config['train']['resume']['checkpoint']:
        print('Resuming training from checkpoint...')
        model.load_weights(
            config['train']['resume']['style'],
            config['train']['resume']['genre'],
            config['train']['resume']['model']
        )
    else:
        print("Starting training from scratch.")

    model.trainResNet(train_loader, test_loader)
