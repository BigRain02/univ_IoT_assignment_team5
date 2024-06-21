import os
import torch
import numpy as np
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

cur_dir = os.getcwd()
model_path = os.path.join(cur_dir, '../stage-1', 'best.pt')
img_path = os.path.join(cur_dir, '', 'test_img1.jpg')


def use_result(results, frame):
	if (results and results[0]):
		bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
		classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
		names = results[0].names
		pred_box = zip(classes, bboxes)
		for cls, bbox in pred_box:
			(x, y, x2, y2) = bbox
			if cls == 0:
				cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
				cv2.putText(frame, str(names[cls]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
			scale_percent = 40
			width = int(frame.shape[1] * scale_percent / 100)
			height = int(frame.shape[0] * scale_percent / 100)
			dim = (width, height)
			frame_s = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
			cv2.imshow("Img", frame_s)
	return


model = YOLO(model_path)
eye_rst = model(img_path)
frame = cv2.imread(img_path)


def detect_pupil_position(eye_xyxy, frame):
	x1, y1 = eye_xyxy[0], eye_xyxy[1]
	x2, y2 = eye_xyxy[2], eye_xyxy[3]
	eye_roi = frame[y1:y2, x1:x2]
	eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
	_, binary_image = cv2.threshold(eye_gray, 70, 255, cv2.THRESH_BINARY)

	contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	max_contour = max(contours, key=cv2.contourArea)

	moments = cv2.moments(max_contour)
	pupil_center_x = int(moments['m10'] / moments['m00'])
	pupil_center_y = int(moments['m01'] / moments['m00'])

	result_image = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)

	# 눈동자 중심 좌표 표시
	cv2.circle(result_image, (pupil_center_x, pupil_center_y), 5, (0, 0, 255), -1)

	# 눈동자의 좌/우측 양 끝 좌표 찾기
	leftmost = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
	rightmost = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
	cv2.circle(result_image, leftmost, 5, (255, 0, 0), -1)
	cv2.circle(result_image, rightmost, 5, (255, 0, 0), -1)

	# 흰자의 양 끝 좌표 찾기
	topmost = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
	bottommost = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
	cv2.circle(result_image, topmost, 5, (0, 255, 0), -1)
	cv2.circle(result_image, bottommost, 5, (0, 255, 0), -1)

	plt.imshow(result_image)
	plt.axis('off')
	plt.show()

	return {
		"pupil_center": (pupil_center_x + x1, pupil_center_y + y1),
		"leftmost": (leftmost[0] + x1, leftmost[1] + y1),
		"rightmost": (rightmost[0] + x1, rightmost[1] + y1),
		"topmost": (topmost[0] + x1, topmost[1] + y1),
		"bottommost": (bottommost[0] + x1, bottommost[1] + y1)
	}


def use_rst(eye_rst, frame):
	eyes_xyxy = eye_rst[0].boxes.xyxy.numpy()
	left_xyxy = np.round(eyes_xyxy[0]).astype(int)
	right_xyxy = np.round(eyes_xyxy[1]).astype(int)
	left_eye_data = detect_pupil_position(left_xyxy, frame)
	right_eye_data = detect_pupil_position(right_xyxy, frame)

	print("Left Eye Data:", left_eye_data)
	print("Right Eye Data:", right_eye_data)


use_rst(eye_rst, frame)
