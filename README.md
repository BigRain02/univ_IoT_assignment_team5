#  [Embedded systems and IoT Team Project] Eye Tracking
Team 5 Members: Moon seunghyeon, Kim Taewoo, Maximilian Bongers, Liu Yuyang
<br/><br/>

![Python](https://img.shields.io/badge/Code-Python3.9-blue)
![Windows](https://img.shields.io/badge/Platform-Windows-purple)

<br/>
<table>
    <tr>
        <td>This project is part of the mobile systems engineering class _Embedded systems and IoT_ at Dankook university (단국대학교). We are team 5 and are working on an algorithm/AI-based program to detect the direction a person is looking to, also known as eye tracking. Our motivation is to help people improve their presentation skills by collecting data about where they look, i.e. the audience, the screen or their notes, but we also see many more uses like helping drivers stay focused on the street or companies testing their advertisement and whether it "catches" the public's eye.</td>
    </tr>
</table>

<br/><br/><br/>

## Overview
we train yolov8 on customized datasets to recognize human eyes. In addition, it recognizes human pupils and identifies the movement of the gaze, displaying feedback on gaze processing in real time on the webcam screen. The overall process is shown in the picture below.  
![Concept poster of the steps involved in detecting the person's eyes and where they look](/team5-poster.png)

<br/><br/><br/>

## Directory Structure
- directory 'project_attempt': containing the codes you tried early on in the project  
- directory 'project_final_version': with improved recognition performance and final completion code
- In this readme file, only the description of the final version (directory 'project_final_version') was written, and the demo video was also taken using the code of the final version. If you want to read the description of the previous attempt code, please refer to project_attempt/README.md .
<details>
  <summary>Directory Structure</summary>
IoT_team5/  <br/>
└─ project_attempt/  <br/>
   └─ customed_dataset/  <br/>
      └─ images/  <br/>
         └─ train/  <br/>
         └─ val/  <br/>
      └─ labels/  <br/>
      └─ dataset.yaml  <br/>
   └─ best.pt  <br/>
   └─ eye_and_pupil_detect.py  <br/>
   └─ main.py  <br/>
   └─ summary.log  <br/>
   └─ train.ipynb  <br/>
└─ project_final_version/  <br/>
   └─ dataset/  <br/>
      └─ images/  <br/>
      └─ labels/  <br/>
      └─ dataset.yaml  <br/>
   └─ gaze_tracking/  <br/>
      └─ __init__.py  <br/>
      └─ calibration.py  <br/>
      └─ detect_and_feedback.py  <br/>
      └─ eye.py  <br/>
      └─ gaze_tracking.py  <br/>
      └─ pupil.py  <br/>
      └─ shape_predictor_68_face_landmarks.dat  <br/>
   └─ run/detect/train  <br/>
   └─ README.md  <br/>
   └─ main.py  <br/>
   └─ requirements.txt  <br/>
   └─ train.py  <br/>
   └─ yolov8s.pt  <br/>
</details>

<br/><br/><br/>

## Customed Dataset
- The dataset used is a sub-selection of the dataset Labeled Faces in the Wild, short [LFW](https://www.kaggle.com/datasets/atulanandjha/lfwpeople).
- We labeled 532 images and validated with another 354 ourselves. 

<br/><br/><br/>

## Training
- We uses [YOLOv8 by ultralytics](https://github.com/ultralytics/ultralytics) to find any eyes in the frame.
- We trained model in 10 epochs.
- Weights and last.pt and best.pt for each epoch can be found in project_final_version/runs/detect.

<br/><br/><br/>

## Our Algorithm
- Our assignment was to use yolov8 to recognize objects and use the results to write our own algorithms (open source available). Therefore, we used yolov8 to recognize eyes, and if it was well recognized, we used the results to do the following.
    - Hit the red bounding box in your eyes to mark it.
    - Find the pupil within the recognized eye area and display the position of the pupil of both eyes, the position of the gaze, and whether the eye is blinking, and the feedback on it on the webcam screen in real time.
<br/>
- Below is the algorithm content written by our team. You can find it in the use_result function in project_final_version/gaze_tracking/detect_and_feedback.py. And you can use this function in the main.py.
<br/>

```
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

    # detect pupil and feedback
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
```
<br/><br/><br/>

# 5. Youtube Link for Demo
- In the part where I turn on the webcam in the video to show my face, due to a slight error, the part that says 'center' is missing and only the voice that says left, right blinking is saved. However, if you look at the screen, you can see that the feedback about the center appears normally when I'm staring at the center.
- https://youtu.be/XA6bB0BemBo
  
