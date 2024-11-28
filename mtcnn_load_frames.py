import cv2
import os
from mtcnn import MTCNN

def apply_mtcnn_and_draw_rectangles(folder_path):
    mtcnn = MTCNN()

    for root, dirs, files in sorted(os.walk(folder_path)):
        for filename in sorted(files, key=lambda x: int(x.split('_')[1].split('.')[0])):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(root, filename)
                image = cv2.imread(image_path)

                faces = mtcnn.detect_faces(image)

                for face in faces:
                    x, y, w, h = face['box']
                    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

                cv2.imshow('Image with Rectangles', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

folder_path = 'images_original/Kyle_Shewfelt_0'

apply_mtcnn_and_draw_rectangles(folder_path)
