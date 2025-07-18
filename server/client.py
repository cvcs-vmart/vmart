import socket
import argparse
import time


def send_image(image_path, host, port, chunk_size=1024):
    # Leggi l'immagine in binario
    with open("C:\\Users\\bilar\\Pictures\\Screenshots\\Screenshot 2025-01-27 161536.png", "rb") as f:
        image_data = f.read()

    # Crea il socket UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Invia l'immagine a chunk
    total_chunks = (len(image_data) // chunk_size) + 1
    for i in range(total_chunks):
        start = i * chunk_size
        end = start + chunk_size
        chunk = image_data[start:end]

        # Aggiungi un header semplice (numero chunk, totale)
        header = f"{i:05d}{total_chunks:05d}".encode()
        sock.sendto(header + chunk, (host, port))
        time.sleep(0.001)  # Evita congestione

    print(f"Inviato {total_chunks} chunk a {host}:{port}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invia immagine via UDP")
    parser.add_argument("image", help="Percorso dell'immagine (es: test.py.jpg)")
    parser.add_argument("--host", default="192.168.210.21", help="IP del server")
    parser.add_argument("--port", type=int, default=12345, help="Porta del server")
    args = parser.parse_args()

    send_image(args.image, args.host, args.port)