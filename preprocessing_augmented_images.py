import numpy as np
import cv2
import os

path = 'videos/'

video_files = os.listdir(path)

for file in video_files:
    data = np.load(path + file)

    video = data['colorImages']

    save_path = 'sharpening_brightness/' + os.path.splitext(file)[0] + '/'
    os.makedirs(save_path, exist_ok=True)

    frame_rate = 1

    import cv2
    import numpy as np


    def flip_horizontal(image):
        return cv2.flip(image, 1)

    def brightness(image):
        brightness_factor = 0.8
        return np.clip(image * brightness_factor, 0, 1)

    def sharpen(image):
        kernel = np.array([[0, -1, 0],
                           [-1, 7, -1],
                           [0, -1, 0]])
        sharpened_image = cv2.filter2D(image, -1, kernel)
        return sharpened_image

    def blur(image):
        kernel_size = (3, 3)
        blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
        return blurred_image


    def apply_augmentation(frame):
        frame = brightness(frame)
        frame = sharpen(frame)
        frame = blur(frame)
        return frame


    for frame_index in range(0, video.shape[-1], frame_rate * 5):
        frame = video[:, :, :, frame_index]

        frame_normalized = frame.astype(float) / 255.0

        augmented_frame = apply_augmentation(frame_normalized)

        # new_width = 400
        # new_height = 400
        # resized_frame = cv2.resize(augmented_frame, (new_width, new_height))

        cv2.imwrite(save_path + f"frame_{frame_index}.jpg", augmented_frame * 255)


    face_detection = data['boundingBox']
    landmarks_2d = data['landmarks2D']

    frame_index = 0
    frame_rate = 10

    # cv2.namedWindow('Video', cv2.WINDOW_NORMAL)

    # new_width = 400
    # new_height = 400

    while True:
        frame = video[:, :, :, frame_index]

        frame_normalized = frame.astype(float) / 255.0

        augmented_frame = apply_augmentation(frame_normalized)

        # resized_frame = cv2.resize(augmented_frame, (new_width, new_height))  # Specify new_width and new_height


        black = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
        face = face_detection[:, :, frame_index]
        landmarks = landmarks_2d[:, :, frame_index]

        face = face.astype(int)
        landmarks = landmarks.astype(int)

        black = cv2.rectangle(black, (face[0, 0], face[0, 1]), (face[-1, 0], face[-1, 1]), (0, 255, 0), 2)

        for j in range(landmarks.shape[0]):
            black = cv2.circle(black, (landmarks[j, 0], landmarks[j, 1]), 2, (0, 0, 255), -1)

        frame_with_bb = np.hstack((augmented_frame, black))

        # cv2.imshow('Video', frame_with_bb)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        frame_index += frame_rate

        if frame_index >= video.shape[-1]:
            break

    cv2.destroyAllWindows()

print("Frames saved successfully.")
