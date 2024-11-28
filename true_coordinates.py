import numpy as np
import cv2
import os
import csv

def print_rectangle_coordinates(file_path, frame_index, face, csv_writer):
    video_name = os.path.splitext(os.path.basename(file_path))[0]
    frame_name = f"frame_{frame_index}.jpg"
    csv_writer.writerow([video_name, frame_name, face[0, 0], face[0, 1], face[-1, 0], face[-1, 1]])

video_dir = 'videos/'
output_csv = 'true_original_rectangles.csv'

with open(output_csv, mode='w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    for file in os.listdir(video_dir):
        if file.endswith(".npz"):
            data = np.load(os.path.join(video_dir, file))

            video = data['colorImages']

            face_detection = data['boundingBox']

            frame_rate = 1

            for frame_index in range(0, video.shape[-1], frame_rate):
                face = face_detection[:, :, frame_index]

                face = face.astype(int)

                if frame_index % 5 == 0:
                    print_rectangle_coordinates(os.path.join(video_dir, file), frame_index, face, csv_writer)
