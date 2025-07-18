from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import numpy as np
import torch
import faiss
import os
import argparse

def add_vector_to_index(embedding, index):
    # Convert embedding to numpy
    vector = embedding.detach().cpu().numpy()
    # Convert to float32 numpy
    vector = np.float32(vector)
    # Normalize vector: important to avoid wrong results when searching
    faiss.normalize_L2(vector)
    # Add to index
    index.add(vector)

def index_creator(path):

    images = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.jpg'):
                images.append(root + '/' + file)

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small',
                                                    use_fast=True)  # carica automaticamente i parametri del preprocessore associato al modello scelto dalla libreria Hugging face; questo preprocessore sa come trasformare un'immagine grezza nei tensori che il modello DINOv2 si aspetta
    model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)  # carica il modello DINOv2 versione small

    index = faiss.IndexFlatL2(384)
    dic = {}

    for i, image_path in enumerate(images):
        dic[str(i)] = image_path
        img = Image.open(image_path).convert('RGB')
        with torch.no_grad():
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Sposta sul device
            outputs = model(**inputs)
        features = outputs.last_hidden_state
        add_vector_to_index(features.mean(dim=1), index)
        print(f"Ho inserito la {i}-th immagine nell'indice.")
        print(f"Al momento l'indice ha {index.ntotal} elementi.")

    with open('all_paintings.json', 'w') as f:
        json.dump(dictionary, f) # type: ignore
    faiss.write_index(index, "all_paintings.index")
    print(f"Fatto, costruito e salvato l'indice con {index.ntotal} elementi.")

if __name__ == '__main__':
    # parser per i parametri
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirpath", type=str, help="Path to YAML config file", default='config/config.yaml')
    args = parser.parse_args()

    index_creator(args.dirpath)

