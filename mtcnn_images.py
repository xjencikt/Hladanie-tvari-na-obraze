import cv2
import os
import csv
from mtcnn import MTCNN

def apply_mtcnn_on_folder(folder_path, output_csv):
    mtcnn = MTCNN()

    with open(output_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Folder', 'Image', 'X', 'Y', 'X2', 'Y2'])

        for root, dirs, files in sorted(os.walk(folder_path)):
            for filename in sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0])):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    image_path = os.path.join(root, filename)
                    image = cv2.imread(image_path)

                    faces = mtcnn.detect_faces(image)

                    if len(faces) > 0:
                        x, y, w, h = faces[0]['box']
                        x2 = x + w
                        y2 = y + h
                        csv_writer.writerow([os.path.basename(root), filename, x, y, x2, y2])
                    else:
                        csv_writer.writerow([os.path.basename(root), filename, 0, 0, 0, 0])


folder_path = 'sharpening_brightness'
output_csv = 'txt_files/mtcnn_sharpening.csv'

apply_mtcnn_on_folder(folder_path, output_csv)
