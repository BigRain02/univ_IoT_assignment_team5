import os
import cv2
import torch
import dlib
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt

cur_dir = os.getcwd()
model_path = os.path.join(cur_dir, '../stage-1', 'best.pt')
image_path = os.path.join(cur_dir, '', 'test_img1.jpg')
face_landmarks_face = os.path.join(cur_dir, '../gaze_tracking/shape_predictor_68_face_landmarks.dat')

model = YOLO(model_path)
# eye_rst = model(img_path)
# frame = cv2.imread(img_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmarks_face)


# 눈 영역에서 눈동자 위치를 추출하는 함수
def get_pupil_positions(image, eye_region):
    x, y, w, h = eye_region[0]
    eye = image[y:y + h, x:x + w]
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    eye_landmarks = predictor(gray_eye, dlib.rectangle(0, 0, w, h))

    # 왼쪽 눈의 랜드마크는 36-41, 오른쪽 눈의 랜드마크는 42-47
    left_eye_points = [eye_landmarks.part(i) for i in range(36, 42)]
    right_eye_points = [eye_landmarks.part(i) for i in range(42, 48)]

    left_pupil = (int(sum([p.x for p in left_eye_points]) / 6), int(sum([p.y for p in left_eye_points]) / 6))
    right_pupil = (int(sum([p.x for p in right_eye_points]) / 6), int(sum([p.y for p in right_eye_points]) / 6))

    return left_pupil, right_pupil


# 이미지에서 눈을 감지하는 함수
def detect_eyes(image):
    eyes_rst = model(image)
    eyes_xyxy = eyes_rst[0].boxes.xyxy.numpy()
    eyes = []


    for xyxy in eyes_xyxy:
        x1, y1, x2, y2 = xyxy
        eyes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

    return eyes


# 눈동자 위치 시각화
def visualize_pupils(image, eye_posigtions, pupil_positions):
    for eye_pos, pupil_pos in zip(eye_positions, pupil_positions):
        eye_x, eye_y, eye_w, eye_h = eye_pos
        pupil_x, pupil_y = pupil_pos
        cv2.circle(image, (eye_x + pupil_x, eye_y + pupil_y), 3, (0, 0, 255), -1)  # 눈동자 위치에 빨간색 점 표시
        cv2.rectangle(image, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 255, 0), 2)  # 눈 영역에 초록색 사각형 표시

    # 시각화된 이미지를 화면에 출력
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


image = cv2.imread(image_path)

eye_positions = detect_eyes(image)  # 눈 감지

# 눈동자 추출
pupil_positions = []
for eye_pos in eye_positions:
    left_pupil, right_pupil = get_pupil_positions(image, eye_pos)
    pupil_positions.append((left_pupil, right_pupil))

# 눈동자 위치 시각화
visualize_pupils(image, eye_positions, pupil_positions)
