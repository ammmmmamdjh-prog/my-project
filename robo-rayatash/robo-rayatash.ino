
/*
  =========================================
              RAYATASH ROBOT
       7 Sensor Line Follower Robot
  =========================================

  Controller: Arduino UNO
  Motor Driver: L298N
  Sensors: 7 Digital IR Sensors

  Sensor states:
  1 = Line detected
  0 = No line detected
*/

#define ENA 5
#define IN1 8
#define IN2 9

#define ENB 6
#define IN3 10
#define IN4 11

const byte sensorPins[7] = {
  2, 3, 4, 7, A0, A1, A2
};

const bool SENSOR_ACTIVE_LOW = true;

int baseSpeed = 130;
const int maxSpeed = 220;

float Kp = 28.0;
float Kd = 12.0;

int sensorState[7] = {0, 0, 0, 0, 0, 0, 0};

const int weights[7] = {
  -3, -2, -1, 0, 1, 2, 3
};

float error = 0;
float lastError = 0;
float derivative = 0;

void setup() {

  Serial.begin(9600);

  for (byte i = 0; i < 7; i++) {
    pinMode(sensorPins[i], INPUT);
  }

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  pinMode(ENB, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  stopMotors();

  Serial.println("================================");
  Serial.println("          RAYATASH ROBOT");
  Serial.println("      7 SENSOR LINE FOLLOWER");
  Serial.println("================================");

  delay(1000);
}

void readSensors() {

  for (byte i = 0; i < 7; i++) {

    int rawValue = digitalRead(sensorPins[i]);

    sensorState[i] =
      SENSOR_ACTIVE_LOW ? !rawValue : rawValue;
  }
}

bool calculateError() {

  int weightedSum = 0;
  int activeSensors = 0;

  for (byte i = 0; i < 7; i++) {

    if (sensorState[i] == 1) {

      weightedSum += weights[i];
      activeSensors++;
    }
  }

  if (activeSensors == 0) {
    return false;
  }

  error = (float)weightedSum / activeSensors;

  return true;
}

void setMotor(int leftSpeed, int rightSpeed) {

  leftSpeed = constrain(leftSpeed, -255, 255);
  rightSpeed = constrain(rightSpeed, -255, 255);

  if (leftSpeed >= 0) {

    digitalWrite(IN1, HIGH);
    digitalWrite(IN2, LOW);

  } else {

    digitalWrite(IN1, LOW);
    digitalWrite(IN2, HIGH);
  }

  if (rightSpeed >= 0) {

    digitalWrite(IN3, HIGH);
    digitalWrite(IN4, LOW);

  } else {

    digitalWrite(IN3, LOW);
    digitalWrite(IN4, HIGH);
  }

  analogWrite(ENA, abs(leftSpeed));
  analogWrite(ENB, abs(rightSpeed));
}

void stopMotors() {

  analogWrite(ENA, 0);
  analogWrite(ENB, 0);

  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW);
  digitalWrite(IN4, LOW);
}

void searchLine() {

  if (lastError >= 0) {

    setMotor(110, -110);

  } else {

    setMotor(-110, 110);
  }
}

void printSensorData(bool lineDetected) {

  Serial.print("Sensors: ");

  for (byte i = 0; i < 7; i++) {

    Serial.print("S");
    Serial.print(i + 1);
    Serial.print(":");
    Serial.print(sensorState[i]);

    if (i < 6) {
      Serial.print(" | ");
    }
  }

  if (lineDetected) {

    Serial.print("   Error: ");
    Serial.println(error, 2);

  } else {

    Serial.println("   Line: LOST");
  }
}

void loop() {

  readSensors();

  bool lineDetected = calculateError();

  printSensorData(lineDetected);

  if (!lineDetected) {

    Serial.println("Line Lost! Searching...");

    searchLine();

    delay(10);
    return;
  }

  derivative = error - lastError;

  float correction =
    (Kp * error) + (Kd * derivative);

  int leftSpeed =
    baseSpeed + (int)correction;

  int rightSpeed =
    baseSpeed - (int)correction;

  leftSpeed = constrain(
    leftSpeed, -maxSpeed, maxSpeed
  );

  rightSpeed = constrain(
    rightSpeed, -maxSpeed, maxSpeed
  );

  setMotor(leftSpeed, rightSpeed);

  lastError = error;

  delay(10);
}
