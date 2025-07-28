#include <Arduino.h>
#include <ESP32Servo.h> // Include the ESP32 Servo library

// --- Configuration Parameters ---
const float DT = 0.1f; // seconds (e.g., 10 frames per second)

// Kalman Filter Parameters
const float PROCESS_NOISE_COV = 0.1f;
const float MEASUREMENT_NOISE_COV = 10.0f;

// PID Controller Parameters
const float PID_OUTPUT_MAX_DEGREE = 30.0f;
const float PID_OUTPUT_MIN_DEGREE = -30.0f;

const float KP_PITCH = 0.1f;
const float KI_PITCH = 0.01f;
const float KD_PITCH = 0.005f;
const float INTEGRAL_MAX_PITCH = 10.0f;

const float KP_YAW = 0.1f;
const float KI_YAW = 0.01f;
const float KD_YAW = 0.005f;
const float INTEGRAL_MAX_YAW = 10.0f;

// Camera Frame Center (Target for PID)
const float CAMERA_CENTER_X = 320.0f; // Example: For a 640x480 resolution
const float CAMERA_CENTER_Y = 240.0f; // Example: For a 640x480 resolution

// Servo Pin Definitions (Adjust these to your ESP32 GPIOs)
// Use GPIOs that support PWM. Common choices: 0, 2, 4, 5, 12-19, 21-23, 25-27.
const int PITCH_SERVO_GPIO = 13;
const int YAW_SERVO_GPIO = 12;

// Servo Objects
Servo pitchServo;
Servo yawServo;

// --- 1. Kalman Filter Implementation (C++ Class) ---

// Helper functions for basic matrix operations (simplified for 2D Kalman)
// These are not full-fledged matrix libraries, but sufficient for this specific Kalman filter.

// C = A * B (Matrix * Vector)
void matMul_4x4_4x1(float A[4][4], float B[4], float C[4]) {
    for (int i = 0; i < 4; i++) {
        C[i] = 0;
        for (int k = 0; k < 4; k++) {
            C[i] += A[i][k] * B[k];
        }
    }
}

// C = A * B (Matrix * Matrix)
void matMul_4x4_4x4(float A[4][4], float B[4][4], float C[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// C = A * B (Matrix * Matrix)
void matMul_2x4_4x1(float A[2][4], float B[4], float C[2]) {
    for (int i = 0; i < 2; i++) {
        C[i] = 0;
        for (int k = 0; k < 4; k++) {
            C[i] += A[i][k] * B[k];
        }
    }
}

// C = A * B (Matrix * Matrix)
void matMul_2x4_4x2(float A[2][4], float B[4][2], float C[2][2]) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// C = A * B (Matrix * Matrix)
void matMul_4x2_2x2(float A[4][2], float B[2][2], float C[4][2]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 2; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// C = A + B
void matAdd(float A[], float B[], float C[], int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] + B[i];
    }
}

// C = A - B
void matSub(float A[], float B[], float C[], int size) {
    for (int i = 0; i < size; i++) {
        C[i] = A[i] - B[i];
    }
}

// C = A + B (Matrix)
void matAdd_MxN(float A[], float B[], float C[], int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        C[i] = A[i] + B[i];
    }
}

// C = A - B (Matrix)
void matSub_MxN(float A[], float B[], float C[], int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        C[i] = A[i] - B[i];
    }
}

// B = A_T
void matTranspose_4x4(float A[4][4], float B[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            B[j][i] = A[i][j];
        }
    }
}

void matTranspose_2x4(float A[2][4], float B[4][2]) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            B[j][i] = A[i][j];
        }
    }
}

// 2x2 Matrix Inverse
int matInv_2x2(float A[2][2], float B[2][2]) {
    float det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
    if (fabs(det) < 1e-6) { // Check for singularity
        Serial.println("Matrix is singular, cannot invert.");
        return -1;
    }
    float inv_det = 1.0f / det;
    B[0][0] = A[1][1] * inv_det;
    B[0][1] = -A[0][1] * inv_det;
    B[1][0] = -A[1][0] * inv_det;
    B[1][1] = A[0][0] * inv_det;
    return 0;
}

// C = A * B (4x4 * 4x2 = 4x2)
void matMul_4x4_4x2(float A[4][4], float B[4][2], float C[4][2]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            C[i][j] = 0;
            for (int k = 0; k < 4; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

class KalmanFilter {
public:
    float dt;
    float A[4][4]; // State transition matrix
    float H[2][4]; // Measurement matrix
    float Q[4][4]; // Process noise covariance
    float R[2][2]; // Measurement noise covariance
    float x_hat[4]; // State estimate [x, y, vx, vy]
    float P[4][4]; // Estimate covariance

    KalmanFilter(float dt_val, float process_noise_cov_val, float measurement_noise_cov_val,
                 float initial_x, float initial_y, float initial_vx, float initial_vy) {
        dt = dt_val;

        // Initialize A
        float A_init[4][4] = {
            {1, 0, dt, 0},
            {0, 1, 0, dt},
            {0, 0, 1, 0},
            {0, 0, 0, 1}
        };
        memcpy(A, A_init, sizeof(A_init));

        // Initialize H
        float H_init[2][4] = {
            {1, 0, 0, 0},
            {0, 1, 0, 0}
        };
        memcpy(H, H_init, sizeof(H_init));

        // Initialize Q
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                Q[i][j] = (i == j) ? process_noise_cov_val : 0;
            }
        }

        // Initialize R
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                R[i][j] = (i == j) ? measurement_noise_cov_val : 0;
            }
        }

        // Initial state estimate
        x_hat[0] = initial_x;
        x_hat[1] = initial_y;
        x_hat[2] = initial_vx;
        x_hat[3] = initial_vy;

        // Initial estimate covariance (P)
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                P[i][j] = (i == j) ? 100.0f : 0; // High initial uncertainty
            }
        }
    }

    void predict() {
        float temp_x_hat[4];
        matMul_4x4_4x1(A, x_hat, temp_x_hat); // x_hat = A * x_hat
        memcpy(x_hat, temp_x_hat, sizeof(temp_x_hat));

        float A_T[4][4];
        matTranspose_4x4(A, A_T);
        float temp_P[4][4];
        matMul_4x4_4x4(A, P, temp_P); // temp_P = A * P
        float new_P[4][4];
        matMul_4x4_4x4(temp_P, A_T, new_P); // new_P = temp_P * A_T
        matAdd_MxN((float*)new_P, (float*)Q, (float*)P, 4, 4); // P = new_P + Q
    }

    void update(float measurement_x, float measurement_y) {
        float measurement[2] = {measurement_x, measurement_y};

        float H_x_hat[2];
        matMul_2x4_4x1(H, x_hat, H_x_hat); // H * x_hat

        float y[2];
        matSub(measurement, H_x_hat, y, 2); // y = z - H * x_hat

        float H_T[4][2];
        matTranspose_2x4(H, H_T);

        float P_H_T[4][2];
        matMul_4x4_4x2(P, H_T, P_H_T); // P * H_T

        float H_P_H_T[2][2];
        matMul_2x4_4x2(H, P_H_T, H_P_H_T); // H * P * H_T

        float S[2][2];
        matAdd_MxN((float*)H_P_H_T, (float*)R, (float*)S, 2, 2); // S = H * P * H_T + R

        float S_inv[2][2];
        if (matInv_2x2(S, S_inv) != 0) {
            // Error handling for singular matrix
            return;
        }

        float K[4][2];
        matMul_4x2_2x2(P_H_T, S_inv, K); // K = P * H_T * S_inv

        float K_y[4];
        matMul_4x2_2x1(K, y, K_y); // K * y (Need a 4x2 * 2x1 -> 4x1 matmul helper)
        matAdd(x_hat, K_y, x_hat, 4); // x_hat = x_hat + K * y

        float I_minus_KH[4][4];
        float K_H[4][4];
        matMul_4x2_2x4(K, H, K_H); // K * H (Need a 4x2 * 2x4 -> 4x4 matmul helper)

        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                I_minus_KH[i][j] = (i == j) ? (1.0f - K_H[i][j]) : (-K_H[i][j]);
            }
        }
        float new_P_updated[4][4];
        matMul_4x4_4x4(I_minus_KH, P, new_P_updated); // P = (I - K * H) * P
        memcpy(P, new_P_updated, sizeof(new_P_updated));
    }

    // Additional matMul helper for 4x2 * 2x1 -> 4x1
    void matMul_4x2_2x1(float A[4][2], float B[2], float C[4]) {
        for (int i = 0; i < 4; i++) {
            C[i] = 0;
            for (int k = 0; k < 2; k++) {
                C[i] += A[i][k] * B[k];
            }
        }
    }

    // Additional matMul helper for 4x2 * 2x4 -> 4x4
    void matMul_4x2_2x4(float A[4][2], float B[2][4], float C[4][4]) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                C[i][j] = 0;
                for (int k = 0; k < 2; k++) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
    }
};

// --- 2. PID Controller Implementation (C++ Class) ---

class PIDController {
public:
    float kp;
    float ki;
    float kd;
    float output_min;
    float output_max;
    float integral_max;

    float previous_error;
    float integral;
    unsigned long last_time_micros; // Use unsigned long for micros()

    PIDController(float kp_val, float ki_val, float kd_val,
                  float output_min_val, float output_max_val, float integral_max_val = 0.0f) {
        kp = kp_val;
        ki = ki_val;
        kd = kd_val;
        output_min = output_min_val;
        output_max = output_max_val;
        integral_max = integral_max_val;

        previous_error = 0.0f;
        integral = 0.0f;
        last_time_micros = 0;
    }

    float calculate(float setpoint, float process_variable) {
        unsigned long current_time_micros = micros();
        if (last_time_micros == 0) {
            last_time_micros = current_time_micros;
            return 0.0f;
        }

        float dt = (float)(current_time_micros - last_time_micros) / 1000000.0f; // Convert uS to seconds
        last_time_micros = current_time_micros;

        if (dt == 0) {
            return (previous_error < 0) ? output_min : output_max;
        }

        float error = setpoint - process_variable;

        // Proportional term
        float p_term = kp * error;

        // Integral term
        integral += error * dt;
        if (integral_max > 0) { // Only clamp if integral_max is set
            integral = fmaxf(fminf(integral, integral_max), -integral_max);
        }
        float i_term = ki * integral;

        // Derivative term
        float d_term = kd * ((error - previous_error) / dt);

        // Total output
        float output = p_term + i_term + d_term;

        // Clamp output to min/max values
        output = fmaxf(output_min, fminf(output, output_max));

        // Store current error for the next iteration
        previous_error = error;

        return output;
    }

    void reset() {
        previous_error = 0.0f;
        integral = 0.0f;
        last_time_micros = 0;
    }
};

// --- Global Instances ---
KalmanFilter kf(DT, PROCESS_NOISE_COV, MEASUREMENT_NOISE_COV, 0.0f, 0.0f, 0.0f, 0.0f);
PIDController pidYaw(KP_YAW, KI_YAW, KD_YAW, PID_OUTPUT_MIN_DEGREE, PID_OUTPUT_MAX_DEGREE, INTEGRAL_MAX_YAW);
PIDController pidPitch(KP_PITCH, KI_PITCH, KD_PITCH, PID_OUTPUT_MIN_DEGREE, PID_OUTPUT_MAX_DEGREE, INTEGRAL_MAX_PITCH);

// Global variables for current gimbal angles
volatile float currentGimbalPitchAngle = 90.0f; // Start at 90 degrees (center)
volatile float currentGimbalYawAngle = 90.0f;   // Start at 90 degrees (center)

// --- Helper for Gimbal Angle Calculation ---
float calculateNewGimbalAngle(float current_angle, float pid_output, float servo_min_limit, float servo_max_limit) {
    // PID output is a *correction* in degrees.
    // Positive PID output on YAW means object is right of center, need to move gimbal right (increase yaw angle).
    // Positive PID output on PITCH means object is below center, need to move gimbal down (increase pitch angle).
    // Adjust signs based on your physical gimbal mounting and servo direction.
    float new_angle = current_angle + pid_output;
    return fmaxf(servo_min_limit, fminf(servo_max_limit, new_angle));
}

// --- Arduino Setup Function ---
void setup() {
    Serial.begin(115200); // Initialize serial communication for debugging
    Serial.println("Starting ESP32 Camera Gimbal Tracking System (Arduino)...");

    // Allow allocation of all timers to servos
    ESP32PWM::allocateTimer(0);
    ESP32PWM::allocateTimer(1);
    ESP32PWM::allocateTimer(2);
    ESP32PWM::allocateTimer(3);

    // Attach servos to pins
    pitchServo.setPeriodHertz(50); // Standard 50 Hz servo
    pitchServo.attach(PITCH_SERVO_GPIO, 500, 2500); // GPIO, min/max pulse width in us
    yawServo.setPeriodHertz(50);   // Standard 50 Hz servo
    yawServo.attach(YAW_SERVO_GPIO, 500, 2500);     // GPIO, min/max pulse width in us

    // Set initial servo positions
    pitchServo.write(currentGimbalPitchAngle);
    yawServo.write(currentGimbalYawAngle);
    delay(1000); // Give servos time to move to initial position

    Serial.println("Servos initialized.");
}

// --- Arduino Loop Function ---
void loop() {
    static unsigned long lastLoopTimeMicros = 0;
    unsigned long currentLoopTimeMicros = micros();
    float loop_dt_seconds = (float)(currentLoopTimeMicros - lastLoopTimeMicros) / 1000000.0f;
    lastLoopTimeMicros = currentLoopTimeMicros;

    // --- Simulation Variables (Replace with actual YOLO input) ---
    // In a real scenario, you'd get these from your YOLO system.
    // This part simulates a noisy measurement of a moving object.
    static float simulated_true_x = 0.0f;
    static float simulated_true_y = 0.0f;
    static float simulated_vx = 5.0f; // pixels/sec
    static float simulated_vy = 3.0f; // pixels/sec
    static unsigned long frame_count = 0;

    simulated_true_x += simulated_vx * DT;
    simulated_true_y += simulated_vy * DT;

    // Add random noise to simulate YOLO detection inaccuracies
    float yolo_measured_x = simulated_true_x + ((float)random(0, 1000) / 1000.0f - 0.5f) * 2.0f * sqrtf(MEASUREMENT_NOISE_COV);
    float yolo_measured_y = simulated_true_y + ((float)random(0, 1000) / 1000.0f - 0.5f) * 2.0f * sqrtf(MEASUREMENT_NOISE_COV);
    // -----------------------------------------------------------

    // Kalman Filter Prediction and Update
    kf.predict();
    kf.update(yolo_measured_x, yolo_measured_y);
    float estimated_x = kf.x_hat[0];
    float estimated_y = kf.x_hat[1];

    // PID Control Calculation
    float yaw_pid_output = pidYaw.calculate(CAMERA_CENTER_X, estimated_x);
    float pitch_pid_output = pidPitch.calculate(CAMERA_CENTER_Y, estimated_y);

    // Calculate new gimbal angles
    currentGimbalYawAngle = calculateNewGimbalAngle(currentGimbalYawAngle, yaw_pid_output, 0.0f, 180.0f);
    currentGimbalPitchAngle = calculateNewGimbalAngle(currentGimbalPitchAngle, pitch_pid_output, 0.0f, 180.0f);

    // Move Gimbal Servos
    pitchServo.write(currentGimbalPitchAngle);
    yawServo.write(currentGimbalYawAngle);

    // --- Debug Logging ---
    Serial.printf("Frame %lu: YOLO(%.2f,%.2f) Est(%.2f,%.2f) PID Y/P(%.2f,%.2f) Gimbal Y/P(%.2f,%.2f)\n",
                  frame_count, yolo_measured_x, yolo_measured_y, estimated_x, estimated_y,
                  yaw_pid_output, pitch_pid_output, currentGimbalYawAngle, currentGimbalYawAngle);
    // ---------------------

    // Maintain Loop Timing
    unsigned long targetLoopDurationMicros = (unsigned long)(DT * 1000000);
    unsigned long elapsedLoopTimeMicros = micros() - currentLoopTimeMicros;
    if (elapsedLoopTimeMicros < targetLoopDurationMicros) {
        delayMicroseconds(targetLoopDurationMicros - elapsedLoopTimeMicros);
    } else {
        Serial.printf("WARNING: Loop took longer than DT (%.2fms vs %.2fms)\n",
                      (float)elapsedLoopTimeMicros / 1000.0f, DT * 1000.0f);
    }

    frame_count++;
}

