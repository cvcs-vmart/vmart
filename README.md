# VMART (Visual Museum Augmented Reality Tour)
by D. Bilardello, F. Melis, E. Prato for Computer vision and Cognitive Systems' project.
## The project
<img width="3224" height="2290" alt="pipeline" src="https://github.com/user-attachments/assets/430049eb-4349-475b-80a8-e18a760bd992" />
A novel way to explore art galleries, leveraging on various computer vision and AI-based techniques. We show how a structured elaboration pipeline combined with a well-integrated mixed reality scene can achieve very good results and provide a new way of exploring museums. We employed plenty different techniques and strategies, exploring their advantages and their drawback. Our Visual Museum Augmented Reality Tour, VMART, uses YOLO object detection to localize paintings and adjust perceptive distortion with classical computer vision methods. We then use DINOv2 to create embeddings used to help the painting construction and consent the retrieval. Finally, we use a modified EfficientNet finetuned for style and genre classification. We designed a new way of exploring a museum, by consenting a direct interaction between the user and the paintings, with information and similar artwork displayed on-demand.
<br><br>

Paper link: https://escanortargaryen.dev/vmart/paper.pdf
<br>
Presentation slides: https://escanortargaryen.dev/vmart/VMART-presentation.pdf

## Installation
1. Download [Wikiart dataset](https://archive.org/details/wikiart-dataset).
2. Clone this repository
3. Create a virtual environment and install the requirements
4. Download the weigths for the [classification](https://escanortargaryen.dev/vmart/efficientnet-weights/) and the [index](https://escanortargaryen.dev/vmart/index/) file for retrieval.
5. Run all the servers
   - `/server/server.py`
   - `/detection/model/inference.py`
   - `/transformation/transform.py`
   - `/features/DINOv2fex.py`
   - `/features/DINOv2ret.py`
   - `/classification/server.py`
6. Install the [Unity App](https://github.com/cvcs-vmart/Unity-app) on your Meta quest 3/3s headset.
7. Run the app.

## Configuration
- In order to perform correctly the retrieval, you need to have the Wikiart dataset and retrival index in the same device.
- There are some config files that contains some paths that you need to change according to your local setup.
   - `detection/config/config.yaml`
   - `classification/config/config.yaml`
- Edit the `all_paintings.json` with your absolute path of the dataset.
- For easy connection between headset and server, we recommend using the same network on both devices.
- Depending on the environment, you may have to change the communication IP addresses of the services and within the Unity app.

## Screenshots and video
### Video
Long video as we test it in the Galleria Estense: https://youtu.be/lxSlbR4vHUM
### Screenshots

![com oculus vrshell-20250721-101623](https://github.com/user-attachments/assets/0bb97e15-c6e1-42ff-8041-42596487557f)

![com oculus vrshell-20250721-120941](https://github.com/user-attachments/assets/bc087a05-cc13-4ded-a7bf-ef40f43d8d2a)

## Special thanks
Special thanks to:
- our professors R. Cucchiara and L. Baraldi.
- E. Turri Phd. at UNIMORE
- Galleria Estense
