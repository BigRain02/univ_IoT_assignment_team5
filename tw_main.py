import cv2
from ultralytics import YOLO
import numpy as np
from tw_algo import use_result


def main():
    model = YOLO("best1.pt")
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)

        use_result(results, frame)

        cv2.imshow("Img", frame)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'ESC' to quit
            break


if __name__ == '__main__':
    main()
