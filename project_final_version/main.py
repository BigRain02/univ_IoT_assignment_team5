import cv2
import time
from ultralytics import YOLO
from gaze_tracking import GazeTracking
from gaze_tracking.detect_and_feedback import use_result


model = YOLO("runs/detect/train/weights/best.pt")

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

while True:
	_, frame = webcam.read()

	yolo_result = model(frame)
	if yolo_result:
		use_result(yolo_result, frame)
	else:
		continue

	if cv2.waitKey(1) == 27:
		break

webcam.release()
cv2.destroyAllWindows()
