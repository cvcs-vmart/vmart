import os
from transformers import AutoImageProcessor, AutoModel
import torch
import faiss
import cv2
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=True)
model = AutoModel.from_pretrained('facebook/dinov2-small').to(device)


# vado a calcolarmi la distanza minima in ogni classe di immagine (cartelle in wikiart)

images = []
distances = []


for root, dirs, files in os.walk("/work/cvcs2025/bilardello_melis_prato/wikiart"):
    for file in files:
        if file.endswith(".jpg"):
            images.append(cv2.imread(root + '/' + file))

    if len(images) > 1:
        with torch.no_grad():
            inputs = processor(images=images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(**inputs)

        features = output.last_hidden_state
        features = features.mean(dim=1)
        vectors= features.detach().cpu().numpy()
        vectors = np.float32(vectors)
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatL2(384)
        index.add(vectors) # type: ignore

        minn = 5.0
        for i, img in enumerate(images):
            vector = index.reconstruct(i).reshape(1,-1) # type: ignore
            d, j = index.search(vector, 2) # type: ignore
            if d[0][1] < minn:
                minn = d[0][1]

        print(f'Nella cartella {root} la distanza minima è {minn}')
        distances.append(minn)
        del index

    images = []

distances_arr = np.array(distances)
distances_mean = distances_arr.mean()
print(f'La media delle distanze minime è: {distances_mean}')




# Vado a calcolarmi la distanza massima che posso avere tra vettori appartenenti allo stesso dipinto visto da prospettive diverse