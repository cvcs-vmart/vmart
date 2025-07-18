import argparse
import json
import time
from threading import Thread

import requests
import torch
import yaml
from PIL import Image
from flask import Flask, jsonify, request
from torchvision.models import EfficientNet_V2_L_Weights
from waitress import serve

from VMARTResnet import VMARTResnet

vr_ip = ""
vr_port = ""
vr_protocol = ""


def predict(path, preprocess, model):
    image = Image.open(path).convert("RGB")

    image = preprocess(image)
    image = image.to(device)
    image = image.unsqueeze(0)

    return model.predict(image)


def send_data_vr(id, style, genre):
    data = {'id': id, 'style': style, 'genre': genre}
    print(data)
    try:
        # L'indirizzo IP deve essere quello del visore Meta Quest sulla stessa rete locale
        response = requests.post(
            f'{vr_protocol}://{vr_ip}:{vr_port}/post_resnet',
            verify=False,
            json=data
        )
        print(f"[send_data_vr] Response from Unity: {response.status_code} - {response.text}")

    except requests.exceptions.RequestException as e:
        print(f"[ERROR in send_data_vr]: Impossibile connettersi al server Unity. {e}")
    except Exception as e:
        print(f"[ERROR in send_data_vr]: Errore imprevisto. {e}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="VMART ResNet")
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--mode', type=str, default='server', choices=['server', 'test'])
    parser.add_argument('--VRport', type=str, default="8080")
    parser.add_argument('--protocol', type=str, default="http")
    parser.add_argument('--VRIp', type=str, default="192.168.186.246")
    parser.add_argument("--useCUDNN", type=str, help="Use cuDNN", default='True')

    args = parser.parse_args()

    vr_ip = args.VRIp
    vr_port = args.VRport
    vr_protocol = args.protocol

    if args.useCUDNN != "True":
        torch.backends.cudnn.enabled = False

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    torch.backends.cudnn.enabled = config['useCudnn']

    wandb_ = None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = EfficientNet_V2_L_Weights.DEFAULT.transforms()

    model = VMARTResnet(config["train"], device, wandb_)
    model = model.to(device)

    model.load_weights(config["inference"]["weight_path_style"],
                       config["inference"]["weight_path_genre"], config["inference"]["weight_path_model"], )

    try:
        with open('LabelMapping.json', 'r') as file:
            mappings = json.load(file)

        style_map = mappings["style"]
        genre_map = mappings["genre"]

    except FileNotFoundError:
        print(f"Error: The file mapping json file was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the mapping json file. Check for syntax errors in the JSON file.")
    except Exception as e:
        print(f"An unexpected error occurred: mapping json file rading error")

    if args.mode == 'server':
        app = Flask(__name__)


        @app.route('/predict', methods=['POST'])
        def handle_predict():
            if 'file' not in request.files:
                return jsonify({'error': 'no file provided'}), 400

            req_id = request.form.get('id')
            if not req_id:
                return jsonify({'error': 'no id provided'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'no file selected'}), 400
            if file:
                s, g = predict(file, preprocess, model)

                style = style_map[f"{s}"]
                genre = genre_map[f"{g}"]
                Thread(target=send_data_vr, args=(req_id, style, genre)).start()

                return jsonify({'style': style, 'genre': genre, 'id': req_id})

            return 400


        server_config = config.get('server', {})
        host = server_config.get('host', '0.0.0.0')
        port = server_config.get('port', 5000)
        is_production = server_config.get('production', False)

        if is_production:
            print(f"Production server is running on http://{host}:{port}")
            serve(app, host=host, port=port)
        else:
            print(f"Development server is running on http://{host}:{port}")
            app.run(host=host, port=port)

    elif args.mode == 'test':
        images = [("test/baroque_portrait.jpg", "Baroque", "Portrait"),
                  ("test/expressionism_landscape.jpg", "Expressionism", "landscape"),
                  ("test/cubism_painting.jpg", "cubism", "genre painting"),
                  ("test/Actionpainting_unknown.jpg", "Actionpainting", "unknown/painting"),
                  ("test/Actionpainting_unknown2.jpg", "Actionpainting", "unknown/painting"),
                  ("test/contemporary_realism_portrit.jpg", "contemporary_realism", "Portrait"),
                  ("test/pop_art.jpg", "pop_art", "unk"),
                  ("test/a.y.-jackson_algoma-in-november-1935.jpg", "art noveu modern", "unk"),
                  ("test/anne-appleby_mulberry-2008.jpg", "color field", "unk"),
                  ("test/andre-derain_the-port-of-collioure-1905-1.jpg", "pointilillism", "unk")]

        for img, gt_s, gt_g in images:
            print(f"Image {img}")
            start_time = time.time()
            s, g = predict(img, preprocess, model)
            elapsed_time = time.time() - start_time

            print("predict:", style_map[f"{s}"], genre_map[f"{g}"])
            print("gt:", gt_s, gt_g)
            print(f"Tempo di esecuzione predict: {elapsed_time:.4f} secondi")
            print("==================================")
