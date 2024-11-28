import os
import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()

def detect_eyes(image_path):
    image = cv2.imread(image_path)
    result = detector.detect_faces(image)

    if result:
        face = result[0]['box']
        keypoints = result[0]['keypoints']

        left_eye = (keypoints['left_eye'])
        right_eye = (keypoints['right_eye'])

        return left_eye, right_eye

def calculate_average_eye_position(image_dir, target_size):
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
    result = detector.detect_faces(resized_image)

    if result:
        face = result[0]['box']
        keypoints = result[0]['keypoints']

        left_eye = (keypoints['left_eye'])
        right_eye = (keypoints['right_eye'])


        dx = avg_left_eye[0] - left_eye[0]
        dy = avg_left_eye[1] - left_eye[1]

        rows, cols, _ = resized_image.shape
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        normalized_image = cv2.warpAffine(resized_image, M, (cols, rows))

        relative_path = os.path.relpath(image_path, input_dir)
        output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
        output_path = os.path.join(output_subdir, os.path.basename(image_path))
        os.makedirs(output_subdir, exist_ok=True)
        cv2.imwrite(output_path, normalized_image)


input_dir = "images_original"

output_dir = "normalized_faces_mtcnn"

target_size = (250, 250)


avg_left_eye, avg_right_eye = calculate_average_eye_position(input_dir, target_size)

for root, dirs, files in os.walk(input_dir):
    for filename in files:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            normalize_face(image_path, output_dir, avg_left_eye, avg_right_eye, target_size)
