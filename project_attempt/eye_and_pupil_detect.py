import cv2
import numpy as np

# 이전 프레임의 좌표를 저장할 변수 (좌, 우 눈과 눈동자 각각)
previous_eye_coords = [None, None]
previous_pupil_coords = [None, None]
previous_eye_state = [None, None]  # 좌, 우 눈의 이전 상태 ('open' 또는 'closed')

blink_count = 0
movement_count = 0

def detect_blink(current_pupil_coords, previous_pupil_coords):
    global blink_count
    for i in range(2):
        if previous_pupil_coords[i] and not current_pupil_coords[i]:
            blink_count += 1
            print(f"Eye {i+1} blinked")

def use_result(results, frame):
    global previous_eye_coords, previous_pupil_coords, previous_eye_state, movement_count

    height, width, _ = frame.shape
    mid_x = width // 2

    if results and results[0]:
        bboxes = np.array(results[0].boxes.xyxy.cpu(), dtype="int")
        classes = np.array(results[0].boxes.cls.cpu(), dtype="int")
        confs = np.array(results[0].boxes.conf.cpu())
        names = results[0].names

        # 눈과 눈동자 박스와 confidence 값을 저장할 리스트 초기화
        left_eyes = []
        right_eyes = []
        left_pupils = []
        right_pupils = []

        for cls, bbox, conf in zip(classes, bboxes, confs):
            (x, y, x2, y2) = bbox
            if x2 <= mid_x:  # 좌측
                if names[cls] == "eye":
                    left_eyes.append((conf, (x, y, x2, y2)))
                elif names[cls] == "pupil":
                    left_pupils.append((conf, (x, y, x2, y2)))
            else:  # 우측
                if names[cls] == "eye":
                    right_eyes.append((conf, (x, y, x2, y2)))
                elif names[cls] == "pupil":
                    right_pupils.append((conf, (x, y, x2, y2)))

        # confidence 값으로 내림차순 정렬
        left_eyes.sort(reverse=True, key=lambda x: x[0])
        right_eyes.sort(reverse=True, key=lambda x: x[0])
        left_pupils.sort(reverse=True, key=lambda x: x[0])
        right_pupils.sort(reverse=True, key=lambda x: x[0])

        # confidence가 가장 높은 한 쌍 선택
        selected_left_eye = left_eyes[0] if left_eyes else None
        selected_right_eye = right_eyes[0] if right_eyes else None
        selected_left_pupil = left_pupils[0] if left_pupils else None
        selected_right_pupil = right_pupils[0] if right_pupils else None

        # 현재 프레임의 좌표를 저장할 변수 초기화
        current_eye_coords = [None, None]
        current_pupil_coords = [None, None]

        if selected_left_eye:
            conf, bbox = selected_left_eye
            (x, y, x2, y2) = bbox
            current_eye_coords[0] = (x, y, x2, y2)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, "eye", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        if selected_right_eye:
            conf, bbox = selected_right_eye
            (x, y, x2, y2) = bbox
            current_eye_coords[1] = (x, y, x2, y2)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, "eye", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        if selected_left_pupil:
            conf, bbox = selected_left_pupil
            (x, y, x2, y2) = bbox
            current_pupil_coords[0] = (x, y, x2, y2)
            color = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, "pupil", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        if selected_right_pupil:
            conf, bbox = selected_right_pupil
            (x, y, x2, y2) = bbox
            current_pupil_coords[1] = (x, y, x2, y2)
            color = (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
            cv2.putText(frame, "pupil", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)

        # 눈 깜빡임 감지
        detect_blink(current_pupil_coords, previous_pupil_coords)

        # 눈과 눈동자 좌표가 모두 있는 경우 움직임을 계산
        for i in range(2):
            if current_eye_coords[i] and current_pupil_coords[i]:
                if previous_eye_coords[i] and previous_pupil_coords[i]:
                    eye_center_x = (current_eye_coords[i][0] + current_eye_coords[i][2]) / 2
                    eye_center_y = (current_eye_coords[i][1] + current_eye_coords[i][3]) / 2
                    pupil_center_x = (current_pupil_coords[i][0] + current_pupil_coords[i][2]) / 2
                    pupil_center_y = (current_pupil_coords[i][1] + current_pupil_coords[i][3]) / 2

                    prev_eye_center_x = (previous_eye_coords[i][0] + previous_eye_coords[i][2]) / 2
                    prev_eye_center_y = (previous_eye_coords[i][1] + previous_eye_coords[i][3]) / 2
                    prev_pupil_center_x = (previous_pupil_coords[i][0] + previous_pupil_coords[i][2]) / 2
                    prev_pupil_center_y = (previous_pupil_coords[i][1] + previous_pupil_coords[i][3]) / 2

                    # 눈동자의 상대적 움직임 계산
                    dx = pupil_center_x - eye_center_x - (prev_pupil_center_x - prev_eye_center_x)
                    dy = pupil_center_y - eye_center_y - (prev_pupil_center_y - prev_eye_center_y)

                    # 움직임을 기반으로 특정 동작 감지 (예: 특정 임계값 초과 시)
                    if abs(dx) > 5 or abs(dy) > 5:
                        movement_count += 1
                        print(f"Detected significant movement in eye {i+1}: dx={dx}, dy={dy}")

                # 이전 좌표 업데이트
                previous_eye_coords[i] = current_eye_coords[i]
                previous_pupil_coords[i] = current_pupil_coords[i]

        scale_percent = 40  # 원래 크기의 백분율
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        dim = (width, height)

        # 이미지 리사이즈
        frame_s = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Img", frame_s)

    return

def log_summary(duration):
    global blink_count, movement_count
    with open("summary.log", "a") as log_file_summary:
        log_file_summary.write(f"Duration: {duration} seconds\n")
        log_file_summary.write(f"Total blinks: {blink_count}\n")
        log_file_summary.write(f"Total movements: {movement_count}\n")
