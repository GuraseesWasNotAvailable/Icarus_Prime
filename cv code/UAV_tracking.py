import cv2 as cv
from ultralytics import YOLO #You only look once algorithm
import numpy as np
import time

class ObjectTracker:
    def __init__(self, model_path="yolov8n.pt", target_class_id = 0):
        self.model = YOLO(model_path)
        self.target_class_id = target_class_id
        self.tracker = None
        self.kalman = None
        self.tracking_active = False
        self.frame_width = 0
        self.frame_height = 0

    def initialize_tracker(self, frame, bbox):
        self.tracker = cv.TrackerKCF_create()
        self.tracker.init(frame, bbox)
        self.tracking_active = True

        self.kalman = cv.KalmanFilter(4,2)

        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                [0, 1, 0, 1],
                                                [0, 0, 1, 0],
                                                [0, 0, 0, 1]], np.float32)
        
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-3
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kalman.statePost = np.array([[bbox[0]], [bbox[1]], [0], [0]], np.float32)

    def update_kalman(self, measurement):
        self.kalman.predict()
        corrected_state = self.kalman.correct(measurement)
        return corrected_state

    def detect_and_track(self, frame):
        if self.frame_width == 0:
            self.frame_height, self.frame_width = frame.shape[:2]

        if not self.tracking_active:
            results = self.model(frame, verbose=False) # Run YOLO inference
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == self.target_class_id:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        bbox = (x1, y1, x2 - x1, y2 - y1) # OpenCV bbox format (x, y, w, h)
                        self.initialize_tracker(frame, bbox)
                        return bbox, True, "YOLO Detection" # Object detected
            return None, False, "No object detected" # No object found by YOLO
        else:
            success, bbox = self.tracker.update(frame)
            if success:
                # Update Kalman Filter with tracker's bbox center
                measurement = np.array([[bbox[0] + bbox[2]/2], [bbox[1] + bbox[3]/2]], np.float32)
                kalman_state = self.update_kalman(measurement)
                # Use Kalman's predicted position for robust tracking
                kalman_x = int(kalman_state[0, 0] - bbox[2]/2)
                kalman_y = int(kalman_state[1, 0] - bbox[3]/2)
                robust_bbox = (kalman_x, kalman_y, bbox[2], bbox[3])
                return robust_bbox, True, "KCF Tracking with Kalman"
            else:
                self.tracking_active = False # Tracker lost the object
                return None, False, "KCF Tracking Lost"

if __name__ == "__main__":
    # using webcam for simulation, change 0 to a diff integer for the UAV camera
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        exit()

    tracker_system = ObjectTracker(model_path="yolov8n.pt", target_class_id=0) # replace tracker_class_id depending on object to be tracked using coco dataset, using 0 for person for the purposes of the simulation

    # PID Controller gains (to be tuned for the actual UAV, taking random values for now)
    Kp_lat, Ki_lat, Kd_lat = 0.01, 0.0001, 0.005 # Lateral control
    Kp_vert, Ki_vert, Kd_vert = 0.01, 0.0001, 0.005 # Vertical control

    integral_x = 0
    integral_y = 0
    prev_error_x = 0
    prev_error_y = 0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        center_x = frame_width / 2
        center_y = frame_height / 2

        bbox, success, status = tracker_system.detect_and_track(frame)

        if success and bbox:
            x, y, w, h = map(int, bbox)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, status, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            obj_center_x = x + w / 2
            obj_center_y = y + h / 2

            # Calculate errors
            error_x = obj_center_x - center_x
            error_y = obj_center_y - center_y

            # Update integral and derivative terms
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time

            integral_x += error_x * dt
            integral_y += error_y * dt

            derivative_x = (error_x - prev_error_x) / dt
            derivative_y = (error_y - prev_error_y) / dt

            prev_error_x = error_x
            prev_error_y = error_y

            # PID Control Outputs (these would be commands for UAV)
            lateral_command = Kp_lat * error_x + Ki_lat * integral_x + Kd_lat * derivative_x
            vertical_command = Kp_vert * error_y + Ki_vert * integral_y + Kd_vert * derivative_y

            cv.putText(frame, f"Lat Cmd: {lateral_command:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv.putText(frame, f"Vert Cmd: {vertical_command:.2f}", (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Draw center lines and object center
            cv.line(frame, (int(center_x), 0), (int(center_x), frame_height), (255, 255, 255), 1)
            cv.line(frame, (0, int(center_y)), (frame_width, int(center_y)), (255, 255, 255), 1)
            cv.circle(frame, (int(obj_center_x), int(obj_center_y)), 5, (0, 0, 255), -1)

        else:
            cv.putText(frame, status, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Reset integral terms if object is lost
            integral_x = 0
            integral_y = 0

        cv.imshow("Object Tracking for UAV Guidance", frame)

        if cv.waitKey(20) & 0xFF == ord('q') :
            break

    cap.release()
    cv.destroyAllWindows()