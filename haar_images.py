import os
import cv2

def process_images(directory, output_folder):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    for (x, y, w, h) in faces:
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                    output_path = os.path.join(output_folder, filename)
                    cv2.imwrite(output_path, img)

images_dir = 'images_original/Katharine_Hepburn_3'

output_folder = 'FORGIF'

process_images(images_dir, output_folder)
