import cv2
import numpy as np
import time
import sys
import os

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

config_path = "/home/anirban/obj_det/cfg/yolo3.cfg"

weights_path = "/home/anirban/obj_det/yolov3-coco/yolov3.weights"

labels = open("data/coco.names").read().strip().split("\n")

colors = np.random.randint(0, 255, size=(len(labels),3), dtype="uint8")
# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

path_name = "images/street.jpg"
image = cv2.image(path_name)
file_name = os.path.basename(path_name)
filename, ext = file_name.split(".")

h, w = image.shpae[:2]
blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

print("image.shape:", image.shape)
print("blob.shape:", blob.shape)