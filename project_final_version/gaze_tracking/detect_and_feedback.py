import cv2
import time
import numpy as np
from gaze_tracking import GazeTracking


def use_result(yolo_result, frame):
	# eyes boxes
	if yolo_result:
		bboxes = np.array(yolo_result[0].boxes.xyxy.cpu(), dtype="int")
		classes = np.array(yolo_result[0].boxes.cls.cpu(), dtype="int")
		names = yolo_result[0].names
		predict_box = zip(classes, bboxes)
		for cls, bbox in predict_box:
			(x, y, x2, y2) = bbox
			print("bounding box (", x, y, x2, y2, ") has class ", cls, " which is ", names[cls])
			if cls == 0:
				cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
				cv2.putText(frame, str(names[cls]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

			scale_percent = 60
			width = int(frame.shape[1] * scale_percent / 100)
			height = int(frame.shape[0] * scale_percent / 100)
			dim = (width, height)

			frame_s = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
			cv2.imshow("Img", frame_s)

	# detect pulil and feedback
	blinking_count = 0

	gaze = GazeTracking()
	gaze.refresh(frame)

	frame = gaze.annotated_frame()
	text = ""

	if gaze.is_blinking():
		blinking_count += 1
		text = "Blinking too often! Blink your eyes a little more relaxed."
	elif gaze.is_right():
		text = "You're looking to the right. Look at the center."
	elif gaze.is_left():
		text = "You're looking to the left. Look at the center."
	elif gaze.is_center():
		text = "You're looking to the center. That's good!"

	cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 30, 30), 2)

	left_pupil = gaze.pupil_left_coords()
	right_pupil = gaze.pupil_right_coords()
	cv2.putText(frame, "Left pupil:  " + str(left_pupil), (20, 130), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)
	cv2.putText(frame, "Right pupil: " + str(right_pupil), (20, 165), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)

	cv2.imshow("Demo", frame)
