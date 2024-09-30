import torch
import cv2
from pathlib import Path

import numpy as np



model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    
    results = model(frame)

    # Render results on the frame
    results.render()  # Draw boxes and labels

    # Display the frame with detections
    cv2.imshow("YOLOv5 Live Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()



