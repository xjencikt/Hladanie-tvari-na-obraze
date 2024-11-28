import os
import cv2
import csv


def process_images(directory, output_csv):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Folder', 'Image', 'x1', 'y1', 'x2', 'y2'])

        for root, dirs, files in os.walk(directory):
            files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

            for filename in files:
                if filename.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(root, filename)
                    img = cv2.imread(img_path)

                    if img is not None:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                        else:
                            x, y, w, h = 0, 0, 0, 0

                        csvwriter.writerow([os.path.basename(root), filename, x, y, x+w, y+h])

images_dir = 'sharpening_brightness'

output_csv = 'txt_files/haar_sharpening_brightness.csv'

process_images(images_dir, output_csv)
