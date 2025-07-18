import json
import os
import sys
import time
from threading import Thread

import cv2
import numpy as np
import requests
from flask import Flask, request
from waitress import serve

from contour import find_contours
from crop import crop, resize_img
from utils import timer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

app = Flask(__name__)


def send_data_DINO(imgs, camera_pose):
    for img, img_data in imgs:
        try:
            print('Camera pose:', camera_pose)
            print('img_data:', img_data)
            requests.post(
                'http://127.0.0.1:3333/dino_fex',
                files={'image': img,
                       'camera_pose': (None, camera_pose),
                       'img_data': (None, json.dumps(img_data))
                       }
            )
            print('Sent image to DINO service')
        except Exception as e:
            print(f'Errore: {e}')


@app.route('/transform', methods=['POST'])
def transform():
    print(f'Ricezione dati servizio di trasformazione... {time.time()}')
    with timer.Timer('Transformation time'):

        file = request.files['image']
        bboxs = request.form.get('bboxs')

        bboxs_list = json.loads(bboxs)

        position = request.form.get('camera_pose')

        # leggi i byte del file immagine
        file_bytes = file.read()

        # da byte a numpy array
        numpy_array = np.frombuffer(file_bytes, np.uint8)

        # da numpy array a immagine OpenCV
        img = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

        # per ogni bbox, ritaglia l'immagine
        cropped_imgs = []
        response_visor = []
        print(f"Received {len(bboxs_list)} bounding boxes")
        for bbox in bboxs_list:
            x1, y1, x2, y2 = map(int, bbox)

            # preparazione risposta per il visore
            nx = (x1 + x2) // 2
            ny = (y1 + y2) // 2

            nheight = abs(y2 - y1)
            nwidth = abs(x2 - x1)

            response_visor.append([nx, ny, nwidth, nheight])
            #print('Cropping image')
            cropped_imgs.append(resize_img(crop(img, x1, y1, x2, y2, pad=10)))

        transformed_image = []
        for i, cropped_img in enumerate(cropped_imgs):
            #print('Processing cropped image...')
            # rileva i contorni
            transformed_image.append(find_contours(cropped_img, i+200))

        imgs_data = []
        for i, img in enumerate(transformed_image):
            #print('Converting image to bytes...')
            # converti l'immagine in byte
            _, img_bytes = cv2.imencode('.jpg', img)
            imgs_data.append((('image.jpg', img_bytes.tobytes(), 'image/jpeg'), response_visor[i]))

        Thread(target=send_data_DINO, args=(imgs_data, position)).start()

    return "OK"


if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=2222, threads=4)
     # app.run(host='0.0.0.0', port=2222)
