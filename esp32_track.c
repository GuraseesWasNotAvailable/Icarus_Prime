#include <ESP32Servo.h>
#include <BasicLinearAlgebra.h>

using namespace BLA;

#define SERVO_YAW_PIN    17
#define SERVO_PITCH_PIN  18

#define PROCESS_NOISE_COV     1.0f
#define MEASUREMENT_NOISE_COV 5.0f
#define DT 0.1f

Servo yawServo, pitchServo;

float currentGimbalYawAngle = 90;
float currentGimbalPitchAngle = 90;

class KalmanFilter {
public:
    KalmanFilter() {
        A = {1, DT, 0, 0,
             0, 1,  0, 0,
             0, 0,  1, DT,
             0, 0,  0, 1};

        H = {1, 0, 0, 0,
             0, 0, 1, 0};

        P.Fill(0);
        for (int i = 0; i < 4; i++) P(i, i) = 1.0f;

        Q.Fill(0);
        for (int i = 0; i < 4; i++) Q(i, i) = PROCESS_NOISE_COV;

        R.Fill(0);
        for (int i = 0; i < 2; i++) R(i, i) = MEASUREMENT_NOISE_COV;

        state = {120, 0, 90, 0};  // Assume starting at center
    }

    void update(float x_meas, float y_meas) {
        // Predict
        state = A * state;
        P = A * P * ~A + Q;

        BLA::Matrix<2> z = {x_meas, y_meas};
        auto S = H * P * ~H + R;
        auto S_inv = S;
        Invert(S_inv);
        auto K = P * ~H * S_inv;

        state = state + K * (z - H * state);

        BLA::Matrix<4, 4> I;
        I.Fill(0);
        for (int i = 0; i < 4; i++) I(i, i) = 1.0f;

        P = (I - K * H) * P;
    }

    float getX() { return state(0); }
    float getY() { return state(2); }

private:
    BLA::Matrix<4> state;
    BLA::Matrix<4, 4> A, P, Q;
    BLA::Matrix<2, 4> H;
    BLA::Matrix<2, 2> R;
};

KalmanFilter kf;

float Kp = 0.08, Ki = 0.01, Kd = 0.075;
float integral_x = 0, prev_error_x = 0;
float integral_y = 0, prev_error_y = 0;

float clamp(float value, float min_val, float max_val) {
    if (value > max_val) return max_val;
    if (value < min_val) return min_val;
    return value;
}

float calculatePID(float error, float& integral, float& prev_error) {
    integral += error * DT;
    float derivative = (error - prev_error) / DT;
    prev_error = error;
    float output = Kp * error + Ki * integral + Kd * derivative;
    return clamp(output, -10, 10); // Clamp PID output to ±10 degrees
}

float calculateNewGimbalAngle(float currentAngle, float output) {
    return clamp(currentAngle + output, 0, 180);
}

void setup() {
    Serial.begin(115200);
    yawServo.attach(SERVO_YAW_PIN);
    pitchServo.attach(SERVO_PITCH_PIN);
    yawServo.write(currentGimbalYawAngle);
    pitchServo.write(currentGimbalPitchAngle);
}

void loop() {
    static String inputString = "";

    while (Serial.available()) {
        char inChar = (char)Serial.read();
        if (inChar == '\n') {
            int commaIndex = inputString.indexOf(',');
            if (commaIndex > 0) {
                float measured_x = inputString.substring(0, commaIndex).toFloat();
                float measured_y = inputString.substring(commaIndex + 1).toFloat();

                if (true) {
                    kf.update(measured_x, measured_y);

                    float est_x = kf.getX();
                    float est_y = kf.getY();

                    float error_x = est_x ;  // Target center
                    float error_y = est_y ;

                    float yaw_output = calculatePID(error_x, integral_x, prev_error_x);
                    float pitch_output = calculatePID(error_y, integral_y, prev_error_y);

                    currentGimbalYawAngle = calculateNewGimbalAngle(currentGimbalYawAngle, -yaw_output);
                    currentGimbalPitchAngle = calculateNewGimbalAngle(currentGimbalPitchAngle, -pitch_output);

                    yawServo.write((int)currentGimbalYawAngle);
                    pitchServo.write((int)currentGimbalPitchAngle);

                    Serial.printf("Input: (%.1f, %.1f) | Est: (%.1f, %.1f) | Yaw: %.1f | Pitch: %.1f\n",
                                  measured_x, measured_y, est_x, est_y,
                                  currentGimbalYawAngle, currentGimbalPitchAngle);
                }
            }
            inputString = "";
        } else {
            inputString += inChar;
        }
    }

    delay(10);
}
