import cv2
from ultralytics import YOLO
import time
from eye_and_pupil_detect import use_result, log_summary


def main():
    model = YOLO("best.pt")
    cap = cv2.VideoCapture(0)

    start_time = time.time()

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

    end_time = time.time()
    duration = end_time - start_time

    log_summary(duration)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
