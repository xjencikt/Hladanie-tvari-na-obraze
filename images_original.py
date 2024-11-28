import numpy as np
import cv2
import os

path = 'videos/'

video_files = os.listdir(path)

for file in video_files:
    output_path = f'output_frames/{file[:-4]}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    data = np.load(path + file)

    video = data['colorImages']

    frame_index = 0
    frame_rate = 5

    while True:
        frame = video[:, :, :, frame_index]

        if frame_index % frame_rate == 0:
            cv2.imwrite(output_path + f'frame_{frame_index}.jpg', frame)

        frame_index += 1

        if frame_index >= video.shape[-1]:
            break


print("All videos played and frames saved successfully.")
