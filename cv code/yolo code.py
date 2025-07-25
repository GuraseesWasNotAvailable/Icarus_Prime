from ultralytics import YOLO
import cv2
import time
import torch

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name}")

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # You can also use yolov8s.pt, yolov8m.pt, etc.
model.to('cuda')

# Open webcam (use 0) or replace with a video file path
cap = cv2.VideoCapture(1)

# Cell phone class index in COCO dataset
CELL_PHONE_CLASS_ID = 67

# Initialize FPS timer
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Start timer
    start_time = time.time()

    # Run YOLOv8 inference
    results = model(frame, device='cuda',classes=[CELL_PHONE_CLASS_ID], verbose=False)[0]
    end_time = time.time()
    # print(end_time-start_time)
    # Loop through detections
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == CELL_PHONE_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw bounding box and centroid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Cell Phone ({cx}, {cy})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            print(f'Cell Phone Centroid: ({cx}, {cy})')
        #print(end_time-start_time)
    # Calculate and display FPS
    end_time = time.time()
    fps = 1 / (end_time - prev_time) if prev_time != 0 else 0
    prev_time = end_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("YOLO Cell Phone Detection with FPS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
