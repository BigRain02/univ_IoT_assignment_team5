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
        print("YOLOv8 Result = ", results[0].__dict__)
        #         # print("YOLOv8 Result.boxes = ", results[0].boxes)
        #         # print("YOLOv8 Result.boxes.xyxy = ", results[0].boxes.xyxy)
        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        print("YOLOv8 Result.boxes.xyxy.cpu() = ", bboxes)
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        print("YOLOv8 Result.boxes.cls.cpu() = ", classes)
        names = results[0].names
        pred_box = zip(classes, bboxes)
        for cls, bbox in pred_box:
            (x, y, x2, y2) = bbox
            print("bounding box (", x, y, x2, y2, ") has class ", cls, " which is ", names[cls])
            #             # Draw bounding box for dog
            if (cls == 0):
                cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
                #                 # Display class of bounding box
                #                 # cv2.putText(frame, str(cls), (x, y-5), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 2)
                cv2.putText(frame, str(names[cls]), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

            scale_percent = 40  # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            frame_s = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("Img", frame_s)

    return


model = YOLO(model_path)
eye_rst = model(img_path)
frame = cv2.imread(img_path)



def detect_pupil_position(eye_xyxy):
    x1, y1 = eye_xyxy[0], eye_xyxy[1]
    x2, y2 = eye_xyxy[2], eye_xyxy[3]

    eye_roi = frame[y1:y2, x1:x2]
    eye_gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(eye_gray, 70, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    opened_image_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(opened_image_rgb, contours, -1, (0, 255, 0), 2)
    max_contour = max(contours, key=cv2.contourArea)  # 필요할까?
    # opened_image_rgb = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
    # cv2.drawContours(opened_image_rgb, [max_contour], -1, (0, 255, 0), 2)

    plt.imshow(opened_image_rgb)
    plt.axis('off')
    plt.show()

    moments = cv2.moments(max_contour)
    pupil_center_x = int(moments['m10'] / moments['m00'])
    pupil_center_y = int(moments['m01'] / moments['m00'])

    w = eye_xyxy[2] - eye_xyxy[0]
    h = eye_xyxy[3] + eye_xyxy[1]
    center_x = (eye_xyxy[0] + eye_xyxy[2]) / 2
    center_y = (eye_xyxy[1] + eye_xyxy[3]) / 2

    distance_threshold = 0.2 * min(w, h)  # 임의로 설정한 거리 임계값
    distance = np.sqrt((pupil_center_x - center_x) ** 2 + (pupil_center_y - center_y) ** 2)

    result_image = cv2.cvtColor(eye_gray, cv2.COLOR_GRAY2BGR)

    # cv2.circle(result_image, (int(center_x), int(center_y)), 1, (255, 0, 255), -1)  # 눈 중심
    cv2.circle(result_image, (pupil_center_x, pupil_center_y), 1, (0, 0, 255), -1)  # 눈동자 중심

    # ###################################################
    leftmost = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
    rightmost = tuple(max_contour[max_contour[:, :, 0].argmax()][0])

    cv2.circle(result_image, leftmost, 1, (255, 0, 0), -1)
    cv2.circle(result_image, rightmost, 1, (255, 0, 0), -1)
    # ###################################################

    plt.imshow(result_image)
    plt.axis('off')
    plt.show()

    if distance < distance_threshold:
        return "눈의 중앙에 위치"
    else:
        return "눈의 중앙에 위치하지 않음"
    # plt.imshow(opened_image_rgb, cmap='gray')
    # plt.axis('off')
    # plt.show()


def use_rst(eye_rst):
    eyes_xyxy = eye_rst[0].boxes.xyxy.numpy()
    left_xyxy = np.round(eyes_xyxy[0]).astype(int)
    right_xyxy = np.round(eyes_xyxy[1]).astype(int)
    left_eye_roi = detect_pupil_position(left_xyxy)
    right_eye_roi = detect_pupil_position(right_xyxy)

    print(left_eye_roi)
    print(right_eye_roi)


use_rst(eye_rst)
