import argparse
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import requests
import websockets

from preprocessing import select_best_frame

# Configurazione
show_stream = False
verbose = False
buffer = []
processing_url = "http://127.0.0.1:1111/object_detection"


# Funzione per inviare la richiesta HTTP in un thread separato
def send_image_for_processing(processing_url, files):
    try:
        response = requests.post(processing_url, files=files)
        response.raise_for_status()
        if verbose:
            print(f"Risposta dal server di processing: {response.json()}")
    except requests.exceptions.RequestException as e:
        if verbose:
            print(f"Errore richiesta HTTP: {e}")


async def handler(websocket):
    """
    Gestisce la connessione WebSocket in entrata.
    """
    if verbose:
        print(f"Client connesso da {websocket.remote_address}")
    try:
        async for message in websocket:
            # Estrai la lunghezza del JSON (4 byte little-endian)
            json_length = int.from_bytes(message[:4], byteorder='little')
            json_bytes = message[4:4 + json_length]
            jpg_bytes = message[4 + json_length:]

            # Decodifica il JSON
            data = json.loads(json_bytes.decode('utf-8'))

            # Estrai i dati della camera
            camera_pose = data.get('camera_pose', {})
            position = camera_pose.get('position')
            rotation = camera_pose.get('rotation')

            # Decodifica l'immagine JPEG
            img_np = np.frombuffer(jpg_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

            if frame is None:
                if verbose:
                    print("Errore: frame non valido ricevuto.")
                continue

            if verbose:
                print(f"Frame ricevuto. Pose: Position={position}, Rotation={rotation}")

            buffer.append((frame, camera_pose))
            if len(buffer) > 10:

                best_frame, best_pose, _ = select_best_frame(buffer)

                buffer.clear()

                is_success, im_buf_arr = cv2.imencode(".jpg", best_frame)
                if not is_success:
                    if verbose:
                        print("Errore codifica JPEG.")
                    continue

                files = {"image": ('image.jpg', im_buf_arr.tobytes(), 'image/jpeg'),
                         'camera_pose': (None, json.dumps(best_pose))}

                if verbose:
                    print("sto inviando")

                # Invio su un altro thread
                loop = asyncio.get_running_loop()
                executor = getattr(handler, "_executor", None)
                if executor is None:
                    executor = ThreadPoolExecutor()
                    handler._executor = executor
                loop.run_in_executor(executor, send_image_for_processing, processing_url, files)

            if show_stream:
                cv2.imshow('XR Camera Stream', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except websockets.exceptions.ConnectionClosed as e:
        if verbose:
            print(f"Client disconnesso: {e.reason} (codice: {e.code})")
    except Exception as e:
        if verbose:
            print(f"Errore: {e}")
    finally:
        if verbose:
            print(f"Connessione con {websocket.remote_address} chiusa.")


async def main():
    """
    Avvia il server WebSocket.
    """
    host = "0.0.0.0"
    port = 12345
    print(f"Server WebSocket in ascolto su ws://{host}:{port}")

    async with websockets.serve(handler, host, port):
        await asyncio.Future()  # Esegui per sempre


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SocketConnection")
    parser.add_argument('--show_stream', type=int, default=0)
    parser.add_argument('--verbose', type=int, default=1)

    args = parser.parse_args()

    if args.show_stream == 0:
        show_stream = False
    else:
        show_stream = True

    if args.verbose == 0:
        verbose = False
    else:
        verbose = True

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        if verbose:
            print("\nServer fermato.")
    finally:
        cv2.destroyAllWindows()
