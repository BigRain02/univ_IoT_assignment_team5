from ultralytics import YOLO

model = YOLO("yolov8s.pt")
model.train(data='C:/Users/82108/Desktop/MSH202309-/2024-1/IOT/IoT_team5/IoT_team5/dataset/dataset.yaml', epochs=10)