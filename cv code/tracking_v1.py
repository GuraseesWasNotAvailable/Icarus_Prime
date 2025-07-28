import numpy as np
import time

# --- 1. Kalman Filter Implementation ---
# This class remains IDENTICAL as its purpose is purely state estimation.
class KalmanFilter:
    """
    A 2D Kalman Filter for tracking an object's position and velocity.
    State vector: [x, y, vx, vy] (position and velocity in x and y directions)
    Measurement vector: [x, y] (observed position from YOLO)
    """
    def __init__(self, dt, process_noise_cov, measurement_noise_cov, initial_state, initial_covariance):
        self.dt = dt
        self.A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self.Q = np.eye(4) * process_noise_cov
        self.R = np.eye(2) * measurement_noise_cov
        self.x_hat = initial_state
        self.P = initial_covariance

    def predict(self):
        self.x_hat = np.dot(self.A, self.x_hat)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.x_hat)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x_hat = self.x_hat + np.dot(K, y)
        self.P = np.dot((np.eye(self.A.shape[0]) - np.dot(K, self.H)), self.P)

    def get_estimated_position(self):
        return self.x_hat[0], self.x_hat[1]

# --- 2. PID Controller Implementation ---
# This class also remains IDENTICAL. The interpretation of its output changes.
class PIDController:
    """
    A generic PID controller for generating control signals.
    """
    def __init__(self, kp, ki, kd, output_min, output_max, integral_max=None):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_max = integral_max

        self._previous_error = 0.0
        self._integral = 0.0
        self._last_time = None

    def calculate(self, setpoint, process_variable):
        current_time = time.time()
        if self._last_time is None:
            self._last_time = current_time
            return 0.0

        dt = current_time - self._last_time
        if dt == 0:
            return self.output_min if self._previous_error < 0 else self.output_max

        error = setpoint - process_variable

        p_term = self.kp * error

        self._integral += error * dt
        if self.integral_max is not None:
            self._integral = max(min(self._integral, self.integral_max), -self.integral_max)
        i_term = self.ki * self._integral

        d_term = self.kd * ((error - self._previous_error) / dt)

        output = p_term + i_term + d_term
        output = max(self.output_min, min(output, self.output_max))

        self._previous_error = error
        self._last_time = current_time

        return output

    def reset(self):
        self._previous_error = 0.0
        self._integral = 0.0
        self._last_time = None

# --- 3. ESP32 Servo Control (Conceptual MicroPython) ---
# This section remains conceptually the same but explicitly targets a gimbal.

# IMPORTANT: This part requires MicroPython firmware on your ESP32
# and the 'machine' module. It cannot be run directly as standard Python.


from machine import Pin, PWM
import time

class ESP32GimbalControl:
    def __init__(self, pitch_pin_num, yaw_pin_num, freq=50):
        self.freq = freq

        self.pitch_pin = Pin(pitch_pin_num)
        self.pitch_pwm = PWM(self.pitch_pin, freq=self.freq)

        self.yaw_pin = Pin(yaw_pin_num)
        self.yaw_pwm = PWM(self.yaw_pin, freq=self.freq)

        # Duty cycle values for 0.5ms (0 deg) to 2.5ms (180 deg) pulse for 50Hz PWM
        self.min_duty = self.ms_to_duty_cycle(0.5)
        self.max_duty = self.ms_to_duty_cycle(2.5)
        self.range_duty = self.max_duty - self.min_duty

    def ms_to_duty_cycle(self, ms_pulse):
        return int((ms_pulse / (1000 / self.freq)) * 1023) # ESP32 duty cycle resolution

    def set_servo_angle(self, servo_pwm, angle_degrees):
        angle_degrees = max(0, min(angle_degrees, 180)) # Clamp to 0-180 for standard servos
        duty = self.min_duty + (angle_degrees / 180.0) * self.range_duty
        servo_pwm.duty(int(duty))

    def move_gimbal(self, pitch_angle, yaw_angle):
        # Set pitch and yaw servo angles
        # Note: You might need to adjust which servo receives which angle
        # and the direction (e.g., pitch_angle vs 180-pitch_angle) based on your gimbal.
        self.set_servo_angle(self.pitch_pwm, pitch_angle)
        self.set_servo_angle(self.yaw_pwm, yaw_angle)

    def deinit(self):
        self.pitch_pwm.deinit()
        self.yaw_pwm.deinit()


# --- 4. Main Tracking Loop Simulation (for demonstration) ---
# The logic here is also largely the same, but the 'target' is now the gimbal.

if __name__ == "__main__":
    # --- Configuration Parameters ---
    DT = 0.1 # seconds (e.g., 10 frames per second)

    # Kalman Filter Parameters
    PROCESS_NOISE_COV = 0.1
    MEASUREMENT_NOISE_COV = 10.0
    INITIAL_STATE = np.array([0.0, 0.0, 0.0, 0.0])
    INITIAL_COVARIANCE = np.eye(4) * 100.0

    # PID Controller Parameters
    # These gains need to be tuned for your specific GIMBAL and its servos.
    # The output range now represents degrees of *correction* for the gimbal.
    # For instance, an output of 10 means rotate the gimbal 10 degrees.
    # Assuming your gimbal can move +/- 30 degrees from its center.
    PID_OUTPUT_MAX_DEGREE = 30 # Max degrees the PID asks the gimbal to move from its current position
    PID_OUTPUT_MIN_DEGREE = -30

    KP_PITCH = 0.1 # Tune for camera's vertical movement
    KI_PITCH = 0.01
    KD_PITCH = 0.005
    INTEGRAL_MAX_PITCH = 10

    KP_YAW = 0.1 # Tune for camera's horizontal movement
    KI_YAW = 0.01
    KD_YAW = 0.005
    INTEGRAL_MAX_YAW = 10

    # Camera Frame Center (Target for PID)
    CAMERA_CENTER_X = 320 # Example: For a 640x480 resolution
    CAMERA_CENTER_Y = 240 # Example: For a 640x480 resolution

    # Servo Angle Mapping for Gimbal
    # This assumes the gimbal servos are commanded in absolute angles (0-180 degrees).
    # You'll need to know the *current* angle of your gimbal.
    # For a simple control, you might adjust the current angle by the PID output.
    # Let's assume an initial gimbal position and current angles
    # In a real system, you might read these from encoders or assume a home position.
    current_gimbal_pitch_angle = 90.0 # Start at 90 degrees (center)
    current_gimbal_yaw_angle = 90.0   # Start at 90 degrees (center)

    def calculate_new_gimbal_angle(current_angle, pid_output, servo_min_limit=0, servo_max_limit=180):
        """
        Calculates the new target angle for a gimbal servo based on PID output.
        PID output is a *correction* in degrees.
        Positive PID output on YAW means move camera to the right (increase yaw angle).
        Positive PID output on PITCH means move camera up (decrease pitch angle, assuming image Y increases downwards).
        You'll need to confirm these directions with your physical setup.
        """
        # For X-axis (Yaw), if PID output is positive, we want to move the camera right to center the object.
        # This typically means increasing the yaw servo angle.
        # For Y-axis (Pitch), if PID output is positive, we want to move the camera up to center the object.
        # If your image Y increases downwards, moving 'up' means decreasing the pitch servo angle.
        # THIS IS A CRITICAL MAPPING that depends on your gimbal's orientation!
        # Example mapping (adjust signs based on your gimbal):
        # new_angle = current_angle + pid_output # For yaw where increasing angle moves right
        # new_angle = current_angle - pid_output # For pitch where decreasing angle moves up

        # Let's assume:
        # Yaw PID output: positive means rotate camera to the right (increase servo angle)
        # Pitch PID output: positive means rotate camera down (increase servo angle, if Y increases down)
        # IF YOU WANT TO MOVE CAMERA UP FOR POSITIVE PID, you need to subtract.
        # This example assumes positive PID output increases servo angle.
        # Adjust for YOUR gimbal's actual behavior:
        
        # Example: pid_output > 0 means object is right/down of center, need to move gimbal right/down
        # So, increase yaw angle to look right, increase pitch angle to look down.
        new_angle = current_angle + pid_output
        return max(servo_min_limit, min(servo_max_limit, new_angle))


    # --- Initialize Components ---
    kalman_filter = KalmanFilter(DT, PROCESS_NOISE_COV, MEASUREMENT_NOISE_COV,
                                 INITIAL_STATE, INITIAL_COVARIANCE)

    # PID for Yaw control (horizontal error, X-axis -> controls gimbal yaw)
    pid_yaw = PIDController(KP_YAW, KI_YAW, KD_YAW, PID_OUTPUT_MIN_DEGREE, PID_OUTPUT_MAX_DEGREE, INTEGRAL_MAX_YAW)
    # PID for Pitch control (vertical error, Y-axis -> controls gimbal pitch)
    pid_pitch = PIDController(KP_PITCH, KI_PITCH, KD_PITCH, PID_OUTPUT_MIN_DEGREE, PID_OUTPUT_MAX_DEGREE, INTEGRAL_MAX_PITCH)

    # ESP32 Gimbal Control (Conceptual - would run on ESP32)
    # gimbal_controller = ESP32GimbalControl(pitch_pin_num=2, yaw_pin_num=4) # Example pins

    print("Starting camera gimbal tracking simulation...")
    print("Press Ctrl+C to stop.")

    # --- Simulation Loop ---
    simulated_true_x = 0
    simulated_true_y = 0
    simulated_vx = 5 # pixels/sec
    simulated_vy = 3 # pixels/sec
    frame_count = 0

    try:
        while True:
            start_time = time.time()

            # --- 1. Simulate YOLO Measurement (replace with actual YOLO output) ---
            simulated_true_x += simulated_vx * DT
            simulated_true_y += simulated_vy * DT

            yolo_measured_x = simulated_true_x + np.random.normal(0, np.sqrt(MEASUREMENT_NOISE_COV))
            yolo_measured_y = simulated_true_y + np.random.normal(0, np.sqrt(MEASUREMENT_NOISE_COV))
            current_measurement = np.array([yolo_measured_x, yolo_measured_y])

            # --- 2. Kalman Filter Prediction and Update ---
            kalman_filter.predict()
            kalman_filter.update(current_measurement)
            estimated_x, estimated_y = kalman_filter.get_estimated_position()

            print(f"\n--- Frame {frame_count} ---")
            print(f"Simulated True Pos: ({simulated_true_x:.2f}, {simulated_true_y:.2f})")
            print(f"YOLO Measured Pos:  ({yolo_measured_x:.2f}, {yolo_measured_y:.2f})")
            print(f"Kalman Estimated Pos: ({estimated_x:.2f}, {estimated_y:.2f})")

            # --- 3. PID Control Calculation ---
            # PID calculates the *correction* needed in degrees for the gimbal.
            # Error: Camera_Center - Estimated_Object_Position
            yaw_pid_output = pid_yaw.calculate(CAMERA_CENTER_X, estimated_x)
            pitch_pid_output = pid_pitch.calculate(CAMERA_CENTER_Y, estimated_y)

            print(f"Yaw PID Correction: {yaw_pid_output:.2f} degrees (Target X: {CAMERA_CENTER_X})")
            print(f"Pitch PID Correction: {pitch_pid_output:.2f} degrees (Target Y: {CAMERA_CENTER_Y})")

            # --- 4. Gimbal Servo Control ---
            # Update current gimbal angles based on PID output
            # This is where the core change happens: PID output directly influences the gimbal's current angle.
            target_gimbal_yaw_angle = calculate_new_gimbal_angle(current_gimbal_yaw_angle, yaw_pid_output)
            target_gimbal_pitch_angle = calculate_new_gimbal_angle(current_gimbal_pitch_angle, pitch_pid_output)

            # For simulation, update the 'current' angle for the next iteration
            current_gimbal_yaw_angle = target_gimbal_yaw_angle
            current_gimbal_pitch_angle = target_gimbal_pitch_angle

            print(f"Target Gimbal Yaw Angle: {target_gimbal_yaw_angle:.2f} degrees")
            print(f"Target Gimbal Pitch Angle: {target_gimbal_pitch_angle:.2f} degrees")

            # In a real ESP32 setup, you would call:
            # gimbal_controller.move_gimbal(target_gimbal_pitch_angle, target_gimbal_yaw_angle)

            # --- Maintain Loop Timing ---
            end_time = time.time()
            elapsed_time = end_time - start_time
            sleep_time = DT - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                print(f"Warning: Loop took longer than DT ({elapsed_time:.4f}s vs {DT}s)")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
        # In a real ESP32 setup, you would de-initialize servos:
        # gimbal_controller.deinit()

