import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import os

class WikiArtDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)

        # Rimuovi i campioni con genere 'unknown'
        original_count = len(self.annotations)
        self.annotations = self.annotations[self.annotations['genre'] != 139].reset_index(drop=True)
        print(f"Dataset: rimossi {original_count - len(self.annotations)} campioni con genere 'unknown'.")

        self.root_dir = root_dir
        self.transform = transform

        self.styles = sorted(self.annotations['style'].unique())
        self.genres = sorted(self.annotations['genre'].unique())

        self.style2idx = {v: i for i, v in enumerate(self.styles)}
        self.genre2idx = {v: i for i, v in enumerate(self.genres)}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")

        style_label = self.style2idx[self.annotations.iloc[idx, 3]]
        genre_label = self.genre2idx[self.annotations.iloc[idx, 2]]

        if self.transform:
            image = self.transform(image)

        return image, style_label, genre_label
