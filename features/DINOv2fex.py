import argparse
import json
import os

# ⚠️ Soluzione temporanea per ignorare il conflitto tra runtime OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from transformers import AutoImageProcessor, AutoModel
import faiss
import numpy as np
from flask import Flask, request
import requests
import cv2
from threading import Thread

app = Flask(__name__)
cnt = 0
model = None
processor = None
device = None
DEBUG = False
image_detected = []

index = faiss.IndexFlatL2(384)
vr_ip = ""
vr_port = ""
vr_protocol = ""


def send_data_to_dinoret(emb_img, id):
    try:
        emb_img = json.dumps(emb_img.tolist())
        res = requests.post(
            'http://127.0.0.1:6666/dino_ret',
            files={
                "embedding": (None, emb_img),
                "id": (None, id)
            },  # Use the 'json' parameter to send data as JSON in the body
        )

    except Exception as e:
        return "ERROR", 400


def send_data_to_resnet(img, id):
    try:
        requests.post(
            'http://127.0.0.1:5001/predict',
            files={
                'file': ('image.jpg', img, 'img/jpeg'),
                'id': (None, id),
            }
        )

    except Exception as e:
        return "ERROR", 400


def send_data_vr(camera_pose, img_data, img_id):
    try:
        img_data = json.loads(img_data)
        sf = json.loads(camera_pose)

        s = {"centerX": img_data[0], "centerY": img_data[1], "nWidth": img_data[2], "nHeight": img_data[3]}

        r = [s]

        # Prepare the data to be sent in the request body
        payload = {
            "camera_pose": sf,
            "detected_quadri": r,
            # the is must be per painting
            "id": img_id,
        }

        res = requests.post(
            f'{vr_protocol}://{vr_ip}:{vr_port}/post_detections',
            json=payload,  # Use the 'json' parameter to send data as JSON in the body
            verify=False
        )

        # It's good practice to check the response status
        res.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)

        print(camera_pose)
        print(r)
        print(f"Server response: {res.status_code}")
        print(f"Response body: {res.text}")

    except requests.exceptions.RequestException as e:
        print(f'Network or HTTP Error: {e}')
    except json.JSONDecodeError as e:
        print(f'JSON Decoding Error: {e}')
    except Exception as e:
        print(f'An unexpected error occurred: {e}')


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


@app.route('/for_resnet_and_dino_ret', methods=['POST'])
def start_resnet_ret():
    global index
    try:

        id_img = request.json.get("id")
        print("è arrivato ", request.json)
        id_img = int(id_img)

        # DINO retrieval
        vector = index.reconstruct(id_img)  # type: ignore
        Thread(target=send_data_to_dinoret, args=(vector, id_img)).start()

        # ResNet
        Thread(target=send_data_to_resnet, args=(image_detected[id_img], id_img)).start()

        return "ok", 200
    except Exception as e:
        print(e)
        return "ERROR", 400


@app.route('/dino_fex', methods=['POST'])
def start_dino():
    try:
        file = request.files['image']
        camera_pose = request.form.get("camera_pose")
        img_data = request.form.get("img_data")
        file_bytes = file.read()
        numpy_array = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)
        # create with DINO the embedding of img and transform it as a vector for faiss
        img_v = emb_faiss(img)

        global index
        global cnt

        if index.ntotal != 0:
            d, i = index.search(img_v, 1)  # type: ignore
            if d[0][0] > 0.6:
                # if True:
                # comunica al visore che deve costruire l'oggetto quadro
                global cnt
                Thread(target=send_data_vr, args=(camera_pose, img_data, cnt)).start()

                index.add(img_v)  # type: ignore
                cnt += 1
                image_detected.append(file_bytes)

                # Update the faiss index file
                # faiss.write_index(index, "detected_images.index")
                print("Detectetata nuova immagine")
                return "ok", 200
            else:
                print("Il quadro è già stato detectato")
                return "Il quadro è già stato detectato", 200
        else:

            Thread(target=send_data_vr, args=(camera_pose, img_data, cnt)).start()

            index.add(img_v)  # type: ignore
            cnt += 1
            image_detected.append(file_bytes)

            # Update the faiss index file
            # faiss.write_index(index, "detected_images.index")
            print("Detectetata nuova immagine")
            return "ok", 200

    except Exception as e:
        print(f"[ERROR in dino_fex]: {e}")
        return "ERROR", 400


def load_model():
    # Load the model and processor
    device_ = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    processor_ = AutoImageProcessor.from_pretrained('facebook/dinov2-small',
                                                    use_fast=True)  # carica automaticamente i parametri del preprocessore associato al modello scelto dalla libreria Hugging face; questo preprocessore sa come trasformare un'immagine grezza nei tensori che il modello DINOv2 si aspetta
    model_ = AutoModel.from_pretrained('facebook/dinov2-small').to(device_)  # carica il modello DINOv2 versione small

    return device_, processor_, model_


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DinoFex")
    parser.add_argument('--VRIp', type=str, default="192.168.186.246")
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

    app.run(host='0.0.0.0', port=3333)
