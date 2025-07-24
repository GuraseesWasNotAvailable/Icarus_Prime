from ultralytics import YOLO
import cv2
import time
import torch
import serial

# === Serial Configuration ===
SERIAL_PORT = 'COM3'  # Change this to your actual serial port
BAUD_RATE = 9600

# Try to connect to the serial port
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Give microcontroller time to reset
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException:
    print(f"Failed to connect to {SERIAL_PORT}. Check your device.")
    ser = None

# CUDA check
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Load YOLOv8 model to GPU
model = YOLO('yolov8n.pt')
model.to('cuda')

# Open webcam
cap = cv2.VideoCapture(1)

CELL_PHONE_CLASS_ID = 67
prev_time = 0

# Grab a frame to determine center
ret, frame = cap.read()
if not ret:
    print("Failed to read from camera.")
    cap.release()
    if ser:
        ser.close()
    exit()

frame_height, frame_width = frame.shape[:2]
center_x, center_y = frame_width // 2, frame_height // 2

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # Inference
    results = model(frame, device='cuda', classes=[CELL_PHONE_CLASS_ID], verbose=False)[0]

    # Draw origin marker
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.putText(frame, '(0,0)', (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id == CELL_PHONE_CLASS_ID:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Convert to screen-centered coordinates
            rel_cx = cx - center_x
            rel_cy = cy - center_y

            # Draw bounding box and centroid
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Cell Phone ({rel_cx}, {rel_cy})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            print(f'Cell Phone Centroid (relative): ({rel_cx}, {rel_cy})')

            # Send over serial
            if ser:
                try:
                    ser.write(f"{rel_cx},{rel_cy}\n".encode())
                except serial.SerialException:
                    print("Serial write failed.")

    # FPS
    end_time = time.time()
    fps = 1 / (end_time - prev_time) if prev_time != 0 else 0
    prev_time = end_time
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO Cell Phone Detection with FPS", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if ser:
    ser.close()
