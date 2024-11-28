import numpy as np
import cv2
import os

path = 'videos/'

file = np.random.choice(os.listdir(path))

data = np.load("videos/Kim_Cattrall_4.npz")

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

    eye_landmarks = landmarks[36:48]

    for j in range(eye_landmarks.shape[0]):
        x, y = eye_landmarks[j, 0], eye_landmarks[j, 1]
        print(f"Eye {j+1}: x={x}, y={y}")
        black = cv2.circle(black, (x, y), 2, (255, 0, 0), -1)

    frame = np.hstack((frame, black))

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += frame_rate

    if frame_index >= video.shape[-1]:
        frame_index = 0

cv2.destroyAllWindows()

