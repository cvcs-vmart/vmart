import argparse
import json
import os
import sys
import time
from threading import Thread

import cv2
import numpy as np
import requests
import torch
from flask import Flask, request, jsonify
from ultralytics import YOLO
from waitress import serve

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.params import load_config
from utils import timer

app = Flask(__name__)
model = None
params = None

# global
file_bytes = None
bbox = None

pid = -1


def send_to_transform_service(file_bytes, bboxs, camera_pose):
    try:
        with timer.Timer('POST to transform: '):
            #print(f'Invio dati al servizio di trasformazione... {time.time()}')
            requests.post(
                'http://127.0.0.1:2222/transform',
                files={
                    'image': ('image.jpg', file_bytes, 'image/jpeg'),
                    'bboxs': (None, bboxs),  # campo di testo, senza filename
                    'camera_pose': (None, camera_pose),
                }

            )
    except Exception as e:
        print(f"[OBJECT DETECTION]: Errore nella POST a transform: {e}")


@app.route('/object_detection', methods=['POST'])
def predict():
    global pid
    # leggi l'immagine e porta a bytes
    file = request.files['image']
    file_bytes = file.read()

    # da byte a numpy array
    numpy_array = np.frombuffer(file_bytes, np.uint8)
    # da numpy array a immagine OpenCV
    img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

    position = request.form.get('camera_pose')

    if pid == -1:
        if os.path.exists('logs'):
            # prendi l'ultimo pid dalla cartella logs
            existing_pids = [d for d in os.listdir('logs') if os.path.isdir(os.path.join('logs', d))]
            if existing_pids:
                pid = max(existing_pids, key=int)

        pid = int(pid) + 1
        os.mkdir(f'logs/{pid}')
        print(f'[OBJECT DETECTION]: Creata cartella logs/{pid} per il processo {pid}')

    result = model.predict(img, **params)

    existing = os.listdir(f'logs/{pid}')
    count = sum(1 for f in existing if f.endswith('_original.jpg'))

    original_path = f'logs/{pid}/{count + 1}_original.jpg'
    detection_path = f'logs/{pid}/{count + 1}_detection.jpg'
    extraction_path = f'logs/{pid}/{count + 1}_extracted.jpg'

    cv2.imwrite(original_path, img)
    result[0].save(detection_path)

    bboxs = result[0].boxes.xyxy.tolist()

    # filtra le bboxs in base al rapporto tra larghezza e altezza
    bboxs_res = []
    bboxs_hw = result[0].boxes.xywh.tolist()
    for i, elem in enumerate(bboxs_hw):
        x, y, w, h = elem
        if not (w / h < 0.2 or h / w < 0.2):
            # filtra le bboxs in base alla distanza dal bordo dell'immagine
            pad = 5
            h, w = img.shape[:2]
            x1, y1, x2, y2 = bboxs[i]
            if (x1 > pad and y1 > pad and x2 < w - pad and y2 < h - pad):
                # aggiungi la bbox alla lista dei risultati
                bboxs_res.append(bboxs[i])

    # draw contours and save
    img_c = img.copy()
    for bbox in bboxs_res:
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(img_c, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imwrite(extraction_path, img_c)

    bboxs_res_j = json.dumps(bboxs_res)

    # manda i dati al servizio di trasformazione
    Thread(target=send_to_transform_service, args=(file_bytes, bboxs_res_j, position)).start()

    return jsonify({"status": "OK"})


def load_model():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to YAML config file", default='../config/config.yaml')
    parser.add_argument("--useCUDNN", type=str, help="Use cuDNN", default='True')
    parser.add_argument("--conf", type=float, help="YOLO confidence", default=0.9)
    args = parser.parse_args()

    if args.useCUDNN != "True":
        torch.backends.cudnn.enabled = False

    config = load_config(args.config)

    # carica il modello
    model = None

    # carica i pesi
    if os.path.exists(config['inference']['weights']):
        model = YOLO(config['inference']['weights'])
    else:
        print(f'[OBJECT DETECTION]: La cartellozza con i pesi non esiste: {config["inference"]["weights"]}')

    inference_params = {k: v for k, v in config['inference'].items() if k != 'model' and k != 'weights'}

    # overwriting confidence
    inference_params['conf'] = args.conf

    return model, inference_params


if __name__ == '__main__':
    # carica il modello e prend i parametri
    model, params = load_model()

    startup_img = torch.zeros((1, 3, 640, 640))  # immagine di avvio per il modello
    result = model.predict(startup_img, **params)

    #result[0].save('startup.jpg')  # salva l'immagine di avvio per verificare che il modello funzioni

    print('[OBJECT DETECTION]: Model loaded successfully!')

    serve(app, host='0.0.0.0', port=1111, threads=4)
    # app.run(host='0.0.0.0', port=1111)
