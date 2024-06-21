import cv2
import time
from gaze_tracking import GazeTracking


def use_result():
	gaze = GazeTracking()
	webcam = cv2.VideoCapture(0)

	blinking_count = 0

	while True:
		# We get a new frame from the webcam
		_, frame = webcam.read()

		# We send this frame to GazeTracking to analyze it
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

		# if blinking_count >= 4:
		# 	cv2.putText(frame, "Blinking too often!", (200, 20), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
		# blinking_count = 0

		if cv2.waitKey(1) == 27:
			break

	webcam.release()
	cv2.destroyAllWindows()
