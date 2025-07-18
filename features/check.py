import json
import cv2

with open("all_paintings.json", "r") as file:
    dictionary = json.load(file)

#img = cv2.imread(dictionary[])

print(dictionary)
