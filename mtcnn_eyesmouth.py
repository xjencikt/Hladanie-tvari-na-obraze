import os
import cv2
import numpy as np
from mtcnn import MTCNN

def calculate_mse(image1, image2):
    err = np.sum((image1.astype("float") - image2.astype("float")) ** 2)
    err /= float(image1.shape[0] * image1.shape[1])
    return err

def process_images(directory):
    detector = MTCNN()

    mse_values = []
    total_images = 0

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)

                if img is not None:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                    detections = detector.detect_faces(rgb_img)

                    for detection in detections:
                        x, y, w, h = detection['box']
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

                        mse = calculate_mse(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        mse_values.append(mse)
                        total_images += 1

                        left_eye = detection['keypoints']['left_eye']
                        right_eye = detection['keypoints']['right_eye']
                        cv2.rectangle(img, (left_eye[0] - 10, left_eye[1] - 10), (left_eye[0] + 10, left_eye[1] + 10), (255, 0, 0), 2)
                        cv2.rectangle(img, (right_eye[0] - 10, right_eye[1] - 10), (right_eye[0] + 10, right_eye[1] + 10), (255, 0, 0), 2)

                        mouth_left = detection['keypoints']['mouth_left']
                        mouth_right = detection['keypoints']['mouth_right']
                        mouth_top = (int((mouth_left[0] + mouth_right[0]) / 2),
                                     int((mouth_left[1] + mouth_right[1]) / 2) - int(
                                         2 * (mouth_right[1] - mouth_left[1])))
                        mouth_bottom = (int((mouth_left[0] + mouth_right[0]) / 2),
                                        int((mouth_left[1] + mouth_right[1]) / 2) + int(
                                            3 * (mouth_right[1] - mouth_left[1])))

                        mouth_outline = [mouth_left, mouth_bottom, mouth_right, mouth_top]
                        for i in range(len(mouth_outline) - 1):
                            cv2.line(img, mouth_outline[i], mouth_outline[i + 1], (0, 0, 255), 2)
                        cv2.line(img, mouth_outline[-1], mouth_outline[0], (0, 0, 255), 2)



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
