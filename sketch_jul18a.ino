#include <ESP32Servo.h>

// Create servo objects
Servo servoTheta;
Servo servoPhi;

// Define pins for the servos
const int thetaPin = 18;  // Base rotation
const int phiPin = 17;    // Arm elevation

// Angle variables
int thetaAngle = 90;
int phiAngle = 90;

// Servo movement settings
const int angleStep = 5;      // Step per input
const int minAngle = 0;       // Min servo angle
const int maxAngle = 180;     // Max servo angle

void setup() {
  Serial.begin(115200);

  // Allow all PWM channels
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  // Attach the servos
  servoTheta.setPeriodHertz(50);  // Standard servo frequency
  servoPhi.setPeriodHertz(50);

  servoTheta.attach(thetaPin, 500, 2400);  // 0° = 500us, 180° = 2400us
  servoPhi.attach(phiPin, 500, 2400);

  // Move to initial positions
  servoTheta.write(thetaAngle);
  servoPhi.write(phiAngle);

  Serial.println("ESP32 Servo Arm Ready. Use W/S for Theta, A/D for Phi.");
}

void loop() {
  if (Serial.available()) {
    char input = Serial.read();

    switch (input) {
      case 'w':
      case 'W':
        thetaAngle = constrain(thetaAngle + angleStep, minAngle, maxAngle);
        Serial.print("Theta: ");
        Serial.println(thetaAngle);
        break;
      case 's':
      case 'S':
        thetaAngle = constrain(thetaAngle - angleStep, minAngle, maxAngle);
        Serial.print("Theta: ");
        Serial.println(thetaAngle);
        break;
      case 'a':
      case 'A':
        phiAngle = constrain(phiAngle + angleStep, minAngle, maxAngle);
        Serial.print("Phi: ");
        Serial.println(phiAngle);
        break;
      case 'd':
      case 'D':
        phiAngle = constrain(phiAngle - angleStep, minAngle, maxAngle);
        Serial.print("Phi: ");
        Serial.println(phiAngle);
        break;
      default:
        // Ignore other input
        break;
    }

    // Move servos to updated positions
    servoTheta.write(thetaAngle);
    servoPhi.write(phiAngle);
  }
}
