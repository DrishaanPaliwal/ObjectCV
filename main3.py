import torch
import cv2

# Load YOLOv5 large model (yolov5l)
model = torch.hub.load('ultralytics/yolov5', 'yolov5l', pretrained=True)

# Capture video from webcam (0 is the default webcam index)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform detection
    results = model(frame)

    # Render the results (draw bounding boxes and labels on the frame)
    results.render()

    # Display the frame with detections
    cv2.imshow("YOLOv5l Live Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
