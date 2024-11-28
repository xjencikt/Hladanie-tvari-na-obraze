import os
import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def detect_eyes(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) >= 2:
            return eyes[0][:2], eyes[1][:2]

    return None, None

def calculate_average_eye_position(image_dir):
    left_eye_sum = np.array([0, 0])
    right_eye_sum = np.array([0, 0])
    num_images = 0

    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                left_eye, right_eye = detect_eyes(image_path)
                if left_eye is not None and right_eye is not None:
                    left_eye_sum += np.array(left_eye)
                    right_eye_sum += np.array(right_eye)
                    num_images += 1

    if num_images > 0:
        average_left_eye = left_eye_sum // num_images
        average_right_eye = right_eye_sum // num_images
        return average_left_eye, average_right_eye
    else:
        return None, None

def resize_image(image, target_size):
    resized_image = cv2.resize(image, target_size)
    return resized_image

def normalize_face(image_path, output_dir, avg_left_eye, avg_right_eye, target_size):
    image = cv2.imread(image_path)
    resized_image = resize_image(image, target_size)
    left_eye, right_eye = detect_eyes(image_path)

    if left_eye is not None and right_eye is not None:
        dx_left = avg_left_eye[0] - left_eye[0]
        dy_left = avg_left_eye[1] - left_eye[1]

        dx_right = avg_right_eye[0] - right_eye[0]
        dy_right = avg_right_eye[1] - right_eye[1]

        rows, cols, _ = resized_image.shape
        M_left = np.float32([[1, 0, dx_left], [0, 1, dy_left]])
        M_right = np.float32([[1, 0, dx_right], [0, 1, dy_right]])

        normalized_image_left = cv2.warpAffine(resized_image, M_left, (cols, rows))
        normalized_image_right = cv2.warpAffine(normalized_image_left, M_right, (cols, rows))

        relative_path = os.path.relpath(image_path, input_dir)
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        output_path = os.path.join(output_subdir, os.path.basename(image_path))
        os.makedirs(output_subdir, exist_ok=True)
        cv2.imwrite(output_path, normalized_image_right)

input_dir = "images_original"
output_dir = "normalized_faces_haar"
target_size = (250, 250)

avg_left_eye, avg_right_eye = calculate_average_eye_position(input_dir)

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            normalize_face(image_path, output_dir, avg_left_eye, avg_right_eye, target_size)

