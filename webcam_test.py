import torch
import numpy as np

from ultralytics import YOLO
import cv2


def use_result(results, frame):
    returnTuple = ()
    if (results and results[0]):
        #         # print("YOLOv8 Result.boxes = ", results[0].boxes)
        #         # print("YOLOv8 Result.boxes.xyxy = ", results[0].boxes.xyxy)
        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        names = results[0].names
        pred_box = zip(classes, bboxes)
        for cls, bbox in pred_box:
            (x, y, x2, y2) = bbox
            print("bounding box (", x, y, x2, y2, ") has class ", cls, " which is ", names[cls])
            #             # Draw bounding box for dog
            returnTuple = (x,y,x2,y2)
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

<<<<<<< HEAD:webcam_test.py
    return
model = YOLO("best1.pt")
=======
    return returnTuple

model = YOLO("best.pt")
>>>>>>> b6c9a33ad11beb513ea69635dd4c1fdcafa4e392:stage-1/webcamTest.py
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)

<<<<<<< HEAD:webcam_test.py
    use_result(results, frame)
=======
    tup = use_result(results, frame)
    with open('log.txt','a') as file:
        for xy in tup:
            file.write(f"{xy} ")
        file.write("\n")
    ### Use result!!
>>>>>>> b6c9a33ad11beb513ea69635dd4c1fdcafa4e392:stage-1/webcamTest.py

    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
