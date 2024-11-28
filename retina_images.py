import os
import cv2
import csv
from retinaface import RetinaFace

def retinaface_wrapper(image):
    faces = RetinaFace.detect_faces(image)
    return faces

def retinaface(image):
    faces = retinaface_wrapper(image)

    if faces:
        first_face = next(iter(faces.values()))
        x1, y1, x2, y2 = first_face['facial_area']
    else:
        x1, y1, x2, y2 = 0, 0, 0, 0

    return x1, y1, x2, y2

folder_path = "images_augmented/Karin_Viard_1"
output_csv = "retina_coordinates.csv"

image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

with open(output_csv, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["Folder Name", "File Name", "X1", "Y1", "X2", "Y2"])

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        x1, y1, x2, y2 = retinaface(image)

        folder_name = os.path.basename(folder_path)
        csv_writer.writerow([folder_name, image_file, x1, y1, x2, y2])
