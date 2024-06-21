import cv2
from gaze_tracking import GazeTracking
import time

gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

blink_counter = 0
# blink_msg_shown = False
start_time = time.time()

while True:
	# We get a new frame from the webcam
	_, frame = webcam.read()

	# We send this frame to GazeTracking to analyze it
	gaze.refresh(frame)

	frame = gaze.annotated_frame()
	text = ""

	blink_cnt_start_time = time.time()
	while time.time() - blink_cnt_start_time <= 5:
		if gaze.is_blinking():
			text = "Blinking"
			blink_counter += 1
		# cv2.putText(frame, f"blinking_count={blink_counter}", (180, 20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)

	if blink_counter >= 5:
		text_start_time = time.time()
		while time.time() - text_start_time <= 2:
			cv2.putText(frame, "Blink too often!", (90, 180), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
		#blink_message_shown = True
		#start_time = time.time()
	blink_counter = 0

	if gaze.is_right():
		text = "Looking right"
	elif gaze.is_left():
		text = "Looking left"
	elif gaze.is_center():
		text = "Looking center"



	cv2.putText(frame, text, (90, 50), cv2.FONT_HERSHEY_DUPLEX, 1.4, (147, 58, 31), 2)

	left_pupil = gaze.pupil_left_coords()
	right_pupil = gaze.pupil_right_coords()
	cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 110), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)
	cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 145), cv2.FONT_HERSHEY_DUPLEX, 0.7, (147, 58, 31), 1)

	cv2.imshow("Demo", frame)

	if cv2.waitKey(1) == 27:
		break

webcam.release()
cv2.destroyAllWindows()