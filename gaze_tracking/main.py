import os
import time
import cv2
from gaze_tracking import GazeTracking


gaze = GazeTracking()
webcam = cv2.VideoCapture(0)

blink_counter = 0
start_time = time.time()
message_start_time = None
message_duration = 2  # 메시지가 표시될 시간 (초)

frame_id = 0

while True:
	# 비디오에서 새로운 프레임을 가져옴
	ret, frame = webcam.read()
	if not ret:
		break

	# 프레임을 GazeTracking에 전달하여 분석
	gaze.refresh(frame)

	frame = gaze.annotated_frame()
	text = ""

	if gaze.is_blinking():
		blink_counter += 1

	current_time = time.time()

	# 5초 내에 5번 깜빡이면 메시지 표시
	if current_time - start_time <= 5:
		if blink_counter >= 5 and message_start_time is None:
			message_start_time = current_time  # 메시지 표시 시작 시간 기록
	else:
		start_time = current_time
		blink_counter = 0
		message_start_time = None

	# 메시지 표시 로직
	if message_start_time and current_time - message_start_time <= message_duration:
		cv2.putText(frame, "Blink too often!", (90, 180), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
	elif message_start_time and current_time - message_start_time > message_duration:
		message_start_time = None  # 메시지 표시 시간 초기화

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
