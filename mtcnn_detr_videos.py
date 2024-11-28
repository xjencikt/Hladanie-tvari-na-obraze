import numpy as np
import cv2
import os
import dlib
import cv2
import numpy as np
import torch
from mtcnn import MTCNN
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from retinaface import RetinaFace
from transformers import AutoImageProcessor, AutoModelForObjectDetection, pipeline
import mediapipe as mp

def detr(image):
    im = Image.fromarray(image)

    processor = AutoImageProcessor.from_pretrained("aditmohan96/detr-finetuned-face")
    model = AutoModelForObjectDetection.from_pretrained("aditmohan96/detr-finetuned-face")

    inputs = processor(images=im, return_tensors="pt")

    outputs = model(**inputs)

    target_sizes = torch.tensor([im.size[::-1]])
    score_threshold = 0.5
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=score_threshold)[0]

    for box in results["boxes"]:
        x, y, w, h = box

        x, y, w, h = int(x), int(y), int(w), int(h)

        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

def mt_cnn_face_detector(image):
    mtcnn = MTCNN()

    faces = mtcnn.detect_faces(image)

    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('MTCNN', image)
    cv2.waitKey(0)

path = 'videos/'

file = np.random.choice(os.listdir(path))

data = np.load(path + file)

print(data.keys())

video = data['colorImages']

print("Rozmery videa:", video.shape)

face_detection = data['boundingBox']
landmarks_2d = data['landmarks2D']

print("Rozmery detekcie tvari:", face_detection.shape)
print("Rozmery tvarovych bodov:", landmarks_2d.shape)

frame_index = 0
frame_rate = 1

cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

while True:
    frame = video[:, :, :, frame_index]
    black = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)

    face = face_detection[:, :, frame_index]
    landmarks = landmarks_2d[:, :, frame_index]

    face = face.astype(int)
    landmarks = landmarks.astype(int)

    black = cv2.rectangle(black, (face[0, 0], face[0, 1]), (face[-1, 0], face[-1, 1]), (0, 255, 0), 2)

    for j in range(landmarks.shape[0]):
        black = cv2.circle(black, (landmarks[j, 0], landmarks[j, 1]), 2, (0, 0, 255), -1)

    frame = np.hstack((frame, black))

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += frame_rate

    if frame_index >= video.shape[-1]:
        frame_index = 0

cv2.destroyAllWindows()





