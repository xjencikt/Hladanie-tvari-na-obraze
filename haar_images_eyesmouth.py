import os
import cv2
import numpy as np

def calculate_mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def process_images(directory):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

    mse_values = []
    total_images = 0

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
                        roi_gray = gray[y:y+h, x:x+w]
                        roi_color = img[y:y+h, x:x+w]

                        eyes = eye_cascade.detectMultiScale(roi_gray)
                        for (ex, ey, ew, eh) in eyes:
                            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

                        mouths = mouth_cascade.detectMultiScale(roi_gray)
                        for (nx, ny, nw, nh) in mouths:
                            cv2.rectangle(roi_color, (nx, ny), (nx+nw, ny+nh), (0, 0, 255), 2)

                    mse = calculate_mse(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                    mse_values.append(mse)
                    total_images += 1

    avg_mse = np.mean(mse_values)
    min_mse = np.min(mse_values)
    max_mse = np.max(mse_values)

    print("Summary for folder:", directory)
    print("Total images processed:", total_images)
    print("Average MSE:", avg_mse)
    print("Minimum MSE:", min_mse)
    print("Maximum MSE:", max_mse)

images_dir = 'sharpening_brightness'

process_images(images_dir)
