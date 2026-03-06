
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # yolov5 & 8  الاكثر استخدام
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    result = model(frame)

    a = result[0].plot()
    cv2.imshow('Yolotest', a)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
















