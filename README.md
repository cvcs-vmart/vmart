# VMART (Visual Museum Augmented Reality Tour)

## Usage

You can use VMART if you have a Meta Quest 3, and the services MUST be running on AImagelab SRV.

The unity app can be found here: [Unity App](https://github.com/cvcs-vmart/Unity-app)

## Installation
1. Clone the repositoty

2. Create a virtual environment and install the requirements

3. Run all the servers
   - /server/server.py 
   - /detection/model/inference.py
   - /transformation/transform.py
   - /features/DINOv2fex.py
   - /features/DINOv2ret.py
   - /classification/server.py

NB: there are some config files that contains some paths that you need to change according to your local setup.

- detection/config/config.yaml
- classification/config/config.yaml


### Other Requirements
You will need to download the weigths for the classification and the index file for retrieval.

- Classification weights: [Download Link](https://escanortargaryen.dev/vmart/efficientnet-weights/)

- Index file for retrieval: [Download Link](https://escanortargaryen.dev/vmart/index/)


NB: In order to perform correctly the retrieval, you need to have access to "/work/cvcs2025/bilardello_melis_prato/wikiart" directory inside the cluster.
