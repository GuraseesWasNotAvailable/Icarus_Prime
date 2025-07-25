import cv2
import time
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO("best.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    stime = time.time()
    # Run inference
    results = model.predict(source=frame, conf=0.8, save=False, verbose=False)
    etime = time.time()

    # Loop through detections
    boxes = results[0].boxes
    for box in boxes:
        class_id = int(box.cls[0])  # Get class index
        if class_id != 0:
            continue  # Skip if not "white ball" (assuming class 0)

        # Get box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        # Get confidence score
        conf = float(box.conf[0])
        conf_percent = int(conf * 100)

        # Draw box and info
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        label = f"Ball: {conf_percent}% | Center: ({cx}, {cy})"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    print(f"FPS: {1/(etime - stime):.2f}")
    # Show the frame
    cv2.imshow("YOLOv8 White Ball Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

