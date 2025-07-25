from ultralytics import YOLO
import cv2
import time
import serial  # Import pyserial

# === Serial Port Configuration ===
SERIAL_PORT = 'COM3'  # Change to your actual serial port
BAUD_RATE = 9600

# Try to open the serial port
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)  # Give time for Arduino to reset
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except serial.SerialException:
    print(f"Failed to connect to {SERIAL_PORT}. Check your connection.")
    ser = None

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open webcam
cap = cv2.VideoCapture(0)

CELL_PHONE_CLASS_ID = 67
prev_time = 0

# Get frame size
ret, frame = cap.read()
if not ret:
    print("Failed to access camera.")
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

    results = model(frame, classes=[CELL_PHONE_CLASS_ID], verbose=False)[0]

    # Draw screen center (0,0)
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
    cv2.putText(frame, '(0,0)', (center_x + 10, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            rel_cx = cx - center_x
            rel_cy = cy - center_y

            # Draw box and info
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f'Cell Phone ({rel_cx}, {rel_cy})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Send data to Serial
            if ser:
                try:
                    ser.write(f"{rel_cx},{rel_cy}\n".encode())
                except serial.SerialException:
                    print("Serial write failed.")

            print(f'Relative Centroid: ({rel_cx}, {rel_cy})')

    # FPS Calculation
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
