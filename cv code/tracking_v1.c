#include <stdio.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "driver/ledc.h" // For PWM control of servos
#include "esp_timer.h"   // For high-resolution timing

// --- Configuration Parameters ---
#define DT 0.1f // seconds (e.g., 10 frames per second)

// Kalman Filter Parameters
#define PROCESS_NOISE_COV 0.1f
#define MEASUREMENT_NOISE_COV 10.0f

// PID Controller Parameters
#define PID_OUTPUT_MAX_DEGREE 30.0f
#define PID_OUTPUT_MIN_DEGREE -30.0f

#define KP_PITCH 0.1f
#define KI_PITCH 0.01f
#define KD_PITCH 0.005f
#define INTEGRAL_MAX_PITCH 10.0f

#define KP_YAW 0.1f
#define KI_YAW 0.01f
#define KD_YAW 0.005f
#define INTEGRAL_MAX_YAW 10.0f

// Camera Frame Center (Target for PID)
#define CAMERA_CENTER_X 320.0f // Example: For a 640x480 resolution
#define CAMERA_CENTER_Y 240.0f // Example: For a 640x480 resolution

// Servo Pin Definitions (Adjust these to your ESP32 GPIOs)
// Use GPIOs that support LEDC (PWM). Common choices: 0, 2, 4, 5, 12-19, 21-23, 25-27.
#define PITCH_SERVO_GPIO 13
#define YAW_SERVO_GPIO 12

// Servo PWM Configuration
#define SERVO_PWM_FREQ 50      // 50 Hz for standard servos
#define LEDC_TIMER LEDC_TIMER_0
#define LEDC_MODE LEDC_LOW_SPEED_MODE
#define LEDC_DUTY_RES LEDC_TIMER_10_BIT // 10-bit resolution (0-1023)

// Servo Pulse Widths in microseconds (uS)
// These values need to be calibrated for your specific servos.
// Standard servos: 0 deg ~500uS, 90 deg ~1500uS, 180 deg ~2500uS
#define SERVO_MIN_PULSE_US 500
#define SERVO_MAX_PULSE_US 2500

static const char *TAG = "GIMBAL_TRACKING";

// --- 1. Kalman Filter Implementation ---

// Matrix multiplication helper (C = A * B)
void mat_mul(float *A, float *B, float *C, int r1, int c1, int r2, int c2) {
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            C[i * c2 + j] = 0;
            for (int k = 0; k < c1; k++) {
                C[i * c2 + j] += A[i * c1 + k] * B[k * c2 + j];
            }
        }
    }
}

// Matrix addition helper (C = A + B)
void mat_add(float *A, float *B, float *C, int r, int c) {
    for (int i = 0; i < r * c; i++) {
        C[i] = A[i] + B[i];
    }
}

// Matrix subtraction helper (C = A - B)
void mat_sub(float *A, float *B, float *C, int r, int c) {
    for (int i = 0; i < r * c; i++) {
        C[i] = A[i] - B[i];
    }
}

// Matrix transpose helper (B = A_T)
void mat_transpose(float *A, float *B, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            B[j * r + i] = A[i * c + j];
        }
    }
}

// 2x2 Matrix Inverse (only for 2x2, as needed for S in Kalman)
int mat_inv_2x2(float *A, float *B) {
    float det = A[0] * A[3] - A[1] * A[2];
    if (fabs(det) < 1e-6) { // Check for singularity
        ESP_LOGE(TAG, "Matrix is singular, cannot invert.");
        return -1;
    }
    float inv_det = 1.0f / det;
    B[0] = A[3] * inv_det;
    B[1] = -A[1] * inv_det;
    B[2] = -A[2] * inv_det;
    B[3] = A[0] * inv_det;
    return 0;
}

typedef struct {
    float dt;
    float A[4*4]; // State transition matrix
    float H[2*4]; // Measurement matrix
    float Q[4*4]; // Process noise covariance
    float R[2*2]; // Measurement noise covariance
    float x_hat[4]; // State estimate [x, y, vx, vy]
    float P[4*4]; // Estimate covariance
} KalmanFilter;

void kalman_filter_init(KalmanFilter *kf, float dt, float process_noise_cov, float measurement_noise_cov,
                        float initial_x, float initial_y, float initial_vx, float initial_vy) {
    kf->dt = dt;

    // Initialize A (State transition matrix)
    float A_init[4*4] = {
        1, 0, dt, 0,
        0, 1, 0, dt,
        0, 0, 1, 0,
        0, 0, 0, 1
    };
    memcpy(kf->A, A_init, sizeof(A_init));

    // Initialize H (Measurement matrix)
    float H_init[2*4] = {
        1, 0, 0, 0,
        0, 1, 0, 0
    };
    memcpy(kf->H, H_init, sizeof(H_init));

    // Initialize Q (Process noise covariance)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            kf->Q[i*4 + j] = (i == j) ? process_noise_cov : 0;
        }
    }

    // Initialize R (Measurement noise covariance)
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            kf->R[i*2 + j] = (i == j) ? measurement_noise_cov : 0;
        }
    }

    // Initial state estimate
    kf->x_hat[0] = initial_x;
    kf->x_hat[1] = initial_y;
    kf->x_hat[2] = initial_vx;
    kf->x_hat[3] = initial_vy;

    // Initial estimate covariance (P)
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            kf->P[i*4 + j] = (i == j) ? 100.0f : 0; // High initial uncertainty
        }
    }
}

void kalman_filter_predict(KalmanFilter *kf) {
    float temp_x_hat[4];
    mat_mul(kf->A, kf->x_hat, temp_x_hat, 4, 4, 4, 1); // x_hat = A * x_hat
    memcpy(kf->x_hat, temp_x_hat, sizeof(temp_x_hat));

    float A_T[4*4];
    mat_transpose(kf->A, A_T, 4, 4);
    float temp_P[4*4];
    mat_mul(kf->A, kf->P, temp_P, 4, 4, 4, 4); // temp_P = A * P
    float new_P[4*4];
    mat_mul(temp_P, A_T, new_P, 4, 4, 4, 4); // new_P = temp_P * A_T
    mat_add(new_P, kf->Q, kf->P, 4, 4); // P = new_P + Q
}

void kalman_filter_update(KalmanFilter *kf, float measurement_x, float measurement_y) {
    float measurement[2] = {measurement_x, measurement_y};

    float H_x_hat[2];
    mat_mul(kf->H, kf->x_hat, H_x_hat, 2, 4, 4, 1); // H * x_hat

    float y[2];
    mat_sub(measurement, H_x_hat, y, 2, 1); // y = z - H * x_hat

    float H_T[4*2];
    mat_transpose(kf->H, H_T, 2, 4);

    float P_H_T[4*2];
    mat_mul(kf->P, H_T, P_H_T, 4, 4, 4, 2); // P * H_T

    float H_P_H_T[2*2];
    mat_mul(kf->H, P_H_T, H_P_H_T, 2, 4, 4, 2); // H * P * H_T

    float S[2*2];
    mat_add(H_P_H_T, kf->R, S, 2, 2); // S = H * P * H_T + R

    float S_inv[2*2];
    if (mat_inv_2x2(S, S_inv) != 0) {
        // Handle error, e.g., return or log
        return;
    }

    float K[4*2];
    mat_mul(P_H_T, S_inv, K, 4, 2, 2, 2); // K = P * H_T * S_inv

    float K_y[4];
    mat_mul(K, y, K_y, 4, 2, 2, 1); // K * y
    mat_add(kf->x_hat, K_y, kf->x_hat, 4, 1); // x_hat = x_hat + K * y

    float I_minus_KH[4*4];
    float K_H[4*4];
    mat_mul(K, kf->H, K_H, 4, 2, 2, 4); // K * H
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            I_minus_KH[i*4 + j] = (i == j) ? (1.0f - K_H[i*4 + j]) : (-K_H[i*4 + j]);
        }
    }
    float new_P_updated[4*4];
    mat_mul(I_minus_KH, kf->P, new_P_updated, 4, 4, 4, 4); // P = (I - K * H) * P
    memcpy(kf->P, new_P_updated, sizeof(new_P_updated));
}

// --- 2. PID Controller Implementation ---

typedef struct {
    float kp;
    float ki;
    float kd;
    float output_min;
    float output_max;
    float integral_max;

    float previous_error;
    float integral;
    int64_t last_time_us; // Use microseconds for precision
} PIDController;

void pid_controller_init(PIDController *pid, float kp, float ki, float kd,
                         float output_min, float output_max, float integral_max) {
    pid->kp = kp;
    pid->ki = ki;
    pid->kd = kd;
    pid->output_min = output_min;
    pid->output_max = output_max;
    pid->integral_max = integral_max;

    pid->previous_error = 0.0f;
    pid->integral = 0.0f;
    pid->last_time_us = 0; // Will be set on first calculate call
}

float pid_controller_calculate(PIDController *pid, float setpoint, float process_variable) {
    int64_t current_time_us = esp_timer_get_time();
    if (pid->last_time_us == 0) {
        pid->last_time_us = current_time_us;
        return 0.0f; // Return 0 on first call as dt is not defined
    }

    float dt = (float)(current_time_us - pid->last_time_us) / 1000000.0f; // Convert uS to seconds
    pid->last_time_us = current_time_us;

    if (dt == 0) {
        return (pid->previous_error < 0) ? pid->output_min : pid->output_max;
    }

    float error = setpoint - process_variable;

    // Proportional term
    float p_term = pid->kp * error;

    // Integral term
    pid->integral += error * dt;
    if (pid->integral_max > 0) { // Only clamp if integral_max is set
        pid->integral = fmaxf(fminf(pid->integral, pid->integral_max), -pid->integral_max);
    }
    float i_term = pid->ki * pid->integral;

    // Derivative term
    float d_term = pid->kd * ((error - pid->previous_error) / dt);

    // Total output
    float output = p_term + i_term + d_term;

    // Clamp output to min/max values
    output = fmaxf(pid->output_min, fminf(output, pid->output_max));

    // Store current error for the next iteration
    pid->previous_error = error;

    return output;
}

void pid_controller_reset(PIDController *pid) {
    pid->previous_error = 0.0f;
    pid->integral = 0.0f;
    pid->last_time_us = 0;
}

// --- 3. ESP32 Gimbal Servo Control ---

// Function to convert angle (0-180) to PWM duty cycle
// Duty cycle is calculated based on pulse width (uS) for a given frequency and resolution.
uint32_t angle_to_duty(float angle_degrees) {
    // Clamp angle to valid range
    angle_degrees = fmaxf(0.0f, fminf(180.0f, angle_degrees));

    // Calculate pulse width in microseconds
    // Map 0-180 degrees to SERVO_MIN_PULSE_US to SERVO_MAX_PULSE_US
    float pulse_us = SERVO_MIN_PULSE_US + (angle_degrees / 180.0f) * (SERVO_MAX_PULSE_US - SERVO_MIN_PULSE_US);

    // Calculate duty cycle from pulse width
    // Duty cycle = (pulse_us / (1,000,000 uS / SERVO_PWM_FREQ)) * (2^LEDC_DUTY_RES - 1)
    // For 50Hz, period = 20,000 uS
    uint32_t duty = (uint32_t)((pulse_us / (1000000.0f / SERVO_PWM_FREQ)) * (1 << LEDC_DUTY_RES));
    return duty;
}

// Initialize LEDC peripheral for a specific channel
void servo_init(ledc_channel_t channel, int gpio_num) {
    ledc_timer_config_t ledc_timer = {
        .speed_mode = LEDC_MODE,
        .timer_num = LEDC_TIMER,
        .duty_resolution = LEDC_DUTY_RES,
        .freq_hz = SERVO_PWM_FREQ,
        .clk_cfg = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&ledc_timer));

    ledc_channel_config_t ledc_channel = {
        .speed_mode = LEDC_MODE,
        .channel = channel,
        .timer_sel = LEDC_TIMER,
        .intr_type = LEDC_INTR_DISABLE,
        .gpio_num = gpio_num,
        .duty = 0, // Set duty to 0 initially
        .hpoint = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ledc_channel));

    ESP_LOGI(TAG, "Servo on GPIO %d initialized for LEDC channel %d", gpio_num, channel);
}

// Set servo angle for a specific channel
void servo_set_angle(ledc_channel_t channel, float angle_degrees) {
    uint32_t duty = angle_to_duty(angle_degrees);
    ESP_ERROR_CHECK(ledc_set_duty(LEDC_MODE, channel, duty));
    ESP_ERROR_CHECK(ledc_update_duty(LEDC_MODE, channel));
}

// --- Main Application ---

// Global variables for current gimbal angles (can be updated by PID)
volatile float current_gimbal_pitch_angle = 90.0f; // Start at 90 degrees (center)
volatile float current_gimbal_yaw_angle = 90.0f;   // Start at 90 degrees (center)

// Function to calculate new gimbal angle based on PID output
float calculate_new_gimbal_angle(float current_angle, float pid_output, float servo_min_limit, float servo_max_limit) {
    // PID output is a *correction* in degrees.
    // Positive PID output on YAW means object is right of center, need to move gimbal right (increase yaw angle).
    // Positive PID output on PITCH means object is below center, need to move gimbal down (increase pitch angle).
    // Adjust signs based on your physical gimbal mounting and servo direction.
    float new_angle = current_angle + pid_output;
    return fmaxf(servo_min_limit, fminf(servo_max_limit, new_angle));
}

void app_main(void) {
    ESP_LOGI(TAG, "Starting ESP32 Camera Gimbal Tracking System...");

    // Initialize Kalman Filter
    KalmanFilter kf;
    kalman_filter_init(&kf, DT, PROCESS_NOISE_COV, MEASUREMENT_NOISE_COV,
                       0.0f, 0.0f, 0.0f, 0.0f); // Initial state [x,y,vx,vy]

    // Initialize PID Controllers
    PIDController pid_yaw, pid_pitch;
    pid_controller_init(&pid_yaw, KP_YAW, KI_YAW, KD_YAW,
                        PID_OUTPUT_MIN_DEGREE, PID_OUTPUT_MAX_DEGREE, INTEGRAL_MAX_YAW);
    pid_controller_init(&pid_pitch, KP_PITCH, KI_PITCH, KD_PITCH,
                         PID_OUTPUT_MIN_DEGREE, PID_OUTPUT_MAX_DEGREE, INTEGRAL_MAX_PITCH);

    // Initialize Servos
    servo_init(LEDC_CHANNEL_0, PITCH_SERVO_GPIO); // Pitch servo on LEDC Channel 0
    servo_init(LEDC_CHANNEL_1, YAW_SERVO_GPIO);   // Yaw servo on LEDC Channel 1

    // Set initial servo positions
    servo_set_angle(LEDC_CHANNEL_0, current_gimbal_pitch_angle);
    servo_set_angle(LEDC_CHANNEL_1, current_gimbal_yaw_angle);
    vTaskDelay(pdMS_TO_TICKS(1000)); // Give servos time to move to initial position

    // --- Simulation Variables (Replace with actual YOLO input) ---
    float simulated_true_x = 0.0f;
    float simulated_true_y = 0.0f;
    float simulated_vx = 5.0f; // pixels/sec
    float simulated_vy = 3.0f; // pixels/sec
    uint32_t frame_count = 0;
    int64_t loop_start_time_us;
    int64_t loop_end_time_us;
    int64_t elapsed_time_us;
    int64_t sleep_time_us;

    // Main Tracking Loop
    while (1) {
        loop_start_time_us = esp_timer_get_time();

        // --- REPLACE THIS SECTION WITH YOUR ACTUAL YOLO DATA INPUT ---
        // This is where you would receive (x, y) coordinates from your YOLO system.
        // For example, reading from UART, Wi-Fi, or a camera module.
        // For now, we simulate a noisy measurement of a moving object.
        simulated_true_x += simulated_vx * DT;
        simulated_true_y += simulated_vy * DT;

        float yolo_measured_x = simulated_true_x + ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrtf(MEASUREMENT_NOISE_COV);
        float yolo_measured_y = simulated_true_y + ((float)rand() / RAND_MAX - 0.5f) * 2.0f * sqrtf(MEASUREMENT_NOISE_COV);
        // -----------------------------------------------------------

        // Kalman Filter Prediction and Update
        kalman_filter_predict(&kf);
        kalman_filter_update(&kf, yolo_measured_x, yolo_measured_y);
        float estimated_x = kf.x_hat[0];
        float estimated_y = kf.x_hat[1];

        // PID Control Calculation
        float yaw_pid_output = pid_yaw.calculate(&pid_yaw, CAMERA_CENTER_X, estimated_x);
        float pitch_pid_output = pid_pitch.calculate(&pid_pitch, CAMERA_CENTER_Y, estimated_y);

        // Calculate new gimbal angles
        current_gimbal_yaw_angle = calculate_new_gimbal_angle(current_gimbal_yaw_angle, yaw_pid_output, 0.0f, 180.0f);
        current_gimbal_pitch_angle = calculate_new_gimbal_angle(current_gimbal_pitch_angle, pitch_pid_output, 0.0f, 180.0f);

        // Move Gimbal Servos
        servo_set_angle(LEDC_CHANNEL_0, current_gimbal_pitch_angle);
        servo_set_angle(LEDC_CHANNEL_1, current_gimbal_yaw_angle);

        // --- Debug Logging (can be disabled for performance) ---
        ESP_LOGI(TAG, "Frame %lu: YOLO(%.2f,%.2f) Est(%.2f,%.2f) PID Y/P(%.2f,%.2f) Gimbal Y/P(%.2f,%.2f)",
                 frame_count, yolo_measured_x, yolo_measured_y, estimated_x, estimated_y,
                 yaw_pid_output, pitch_pid_output, current_gimbal_yaw_angle, current_gimbal_pitch_angle);
        // -------------------------------------------------------

        // Maintain Loop Timing
        loop_end_time_us = esp_timer_get_time();
        elapsed_time_us = loop_end_time_us - loop_start_time_us;
        sleep_time_us = (int64_t)(DT * 1000000) - elapsed_time_us;

        if (sleep_time_us > 0) {
            vTaskDelay(pdUS_TO_TICKS(sleep_time_us)); // Delay in microseconds
        } else {
            ESP_LOGW(TAG, "Loop took longer than DT (%.2fms vs %.2fms)",
                     (float)elapsed_time_us / 1000.0f, DT * 1000.0f);
        }

        frame_count++;
    }
}
