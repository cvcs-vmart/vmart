from flask import Flask, request, Response
import requests

vr_ip = "192.168.186.246"
vr_port = 8080  # Modifica se necessario

app = Flask(__name__)

@app.route('/', defaults={'path': ''}, methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'])
def proxy(path):
    url = f"http://{vr_ip}:{vr_port}/{path}"
    resp = requests.request(
        method=request.method,
        url=url,
        headers={key: value for key, value in request.headers if key != 'Host'},
        params=request.args,
        data=request.get_data(),
        cookies=request.cookies,
        allow_redirects=False
    )
    excluded_headers = ['content-encoding', 'content-length', 'transfer-encoding', 'connection']
    headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
    response = Response(resp.content, resp.status_code, headers)
    return response

app.run(host='0.0.0.0', port=8080)
