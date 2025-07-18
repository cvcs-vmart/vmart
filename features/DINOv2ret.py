import argparse
import os

import cv2

# ⚠️ Soluzione temporanea per ignorare il conflitto tra runtime OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np
from flask import Flask, request
import json
import requests
from threading import Thread

app = Flask(__name__)
device = None
processor = None
model = None
index = None
vr_ip = ""
vr_port = ""
vr_protocol = ""


def send_data_vr(images_to_send, id):
    """
    Codifica le immagini in formato JPEG e le invia al server Unity
    come richiesta multipart/form-data.
    """
    files = []
    for idx, img_data in enumerate(images_to_send):
        # Codifica l'immagine (array numpy) in un formato di file (es. JPEG) in memoria
        success, encoded_image = cv2.imencode('.jpg', img_data)
        if success:
            # imencode restituisce un array, lo convertiamo in bytes
            image_bytes = encoded_image.tobytes()
            files.append(('images', (f'image{idx}.jpg', image_bytes, 'image/jpeg')))

    files.append(("id", (None, id)))

    if not files:
        print("[ERROR in send_data_vr]: Nessuna immagine da inviare.")
        return

    try:
        # L'indirizzo IP deve essere quello del visore Meta Quest sulla stessa rete locale
        response = requests.post(
            f'{vr_protocol}://{vr_ip}:{vr_port}/post_retrieval',
            files=files,
            verify=False,
            timeout=10  # Aggiungi un timeout per evitare che la richiesta rimanga appesa
        )
        print(f"[send_data_vr] Response from Unity: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR in send_data_vr]: Impossibile connettersi al server Unity. {e}")
    except Exception as e:
        print(f"[ERROR in send_data_vr]: Errore imprevisto. {e}")


def emb_faiss(img):
    try:
        with torch.no_grad():
            input_ = processor(images=img, return_tensors="pt")
            input_ = {k: v.to(device) for k, v in input_.items()}  # Sposta sul device
            output = model(**input_)

        features = output.last_hidden_state
        features = features.mean(dim=1)
        vector = features.detach().cpu().numpy()
        vector = np.float32(vector)
        faiss.normalize_L2(vector)
        return vector

    except Exception as e:
        print(f"[ERROR in emb_faiss]: {e}")
        return None


@app.route('/dino_ret', methods=['POST'])
def dino_retrieval():
    try:
        file = request.form.get('embedding')
        id = request.form.get('id')

        emb_img = json.loads(file)
        emb_img = np.array(emb_img, dtype=np.float32)
        emb_img = emb_img.reshape(1, -1)

        # Search the first 3 paintings more similar to img
        d, i = index.search(emb_img, 3)

        # TODO
        # cambiare il valore della soglia, calcolando la media distanza intra-classe e inter-classe
        # if d[0][0] < 0.8:

        # for j, idx in enumerate(i[0]):
        #   if d[0][j] < threshold:

        # TODO
        # Manda immagine al visore
        with open("all_paintings.json", "r") as dictionary:
            dic = json.load(dictionary)

        images_to_send = []
        print("Immagini simili trovate:")
        for j in i[0]:
            image_path = dic.get(str(j))
            if image_path:
                print(f"  - Index: {j}, Path: {image_path}")
                # Leggi il file dell'immagine dal disco
                img = cv2.imread(image_path)
                if img is not None:
                    images_to_send.append(img)
                else:
                    print(f"  - ATTENZIONE: Impossibile leggere l'immagine dal percorso: {image_path}")
            else:
                print(f"  - ATTENZIONE: Indice {j} non trovato in all_paintings.json")

        # TODO
        # send images to vr
        Thread(target=send_data_vr, args=(images_to_send, id)).start()

        return "OK", 200


    except Exception as e:
        print(f"[ERROR in dino_retrieval]: {e}")
        return "ERROR", 400


def load_model():
    device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor_ = AutoImageProcessor.from_pretrained('facebook/dinov2-small', use_fast=True)
    model_ = AutoModel.from_pretrained('facebook/dinov2-small').to(device_)
    return device_, processor_, model_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DinoFex")
    parser.add_argument('--VRIp', type=str, default="192.168.186.246")
    parser.add_argument('--index_path', type=str, default="/homes/dbilardello/repos/VMART-DINOv2/all_paintings.index")
    parser.add_argument('--VRport', type=str, default="8080")
    parser.add_argument('--protocol', type=str, default="http")
    parser.add_argument("--useCUDNN", type=str, help="Use cuDNN", default='True')

    args = parser.parse_args()

    vr_ip = args.VRIp
    vr_port = args.VRport
    vr_protocol = args.protocol

    if args.useCUDNN != "True":
        torch.backends.cudnn.enabled = False

    device, processor, model = load_model()
    index = faiss.read_index(args.index_path)

    app.run(host='0.0.0.0', port=6666)
