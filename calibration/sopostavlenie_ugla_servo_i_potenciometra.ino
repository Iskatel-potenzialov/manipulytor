#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// Параметры серво
#define SERVO_MIN 102      // 0°
#define SERVO_MAX 512      // 180°
#define SERVO_CHANNEL 0    // Канал PCA9685

// Пин потенциометра
const int potPin = A1;

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// Текущее значение импульса (для плавного движения)
int currentPulse;

void setup() {
  Serial.begin(9600);
  Serial.println("Введите угол (0-180) для серво, будет выполнено 5 замеров потенциометра.");

  pwm.begin();
  pwm.setPWMFreq(50);      // 50 Гц для серво
  delay(10);

  // Начальное положение 90°
  currentPulse = map(90, 0, 180, SERVO_MIN, SERVO_MAX);
  pwm.setPWM(SERVO_CHANNEL, 0, currentPulse);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    int targetAngle = input.toInt();

    if (targetAngle >= 0 && targetAngle <= 180) {
      int targetPulse = map(targetAngle, 0, 180, SERVO_MIN, SERVO_MAX);

      // ----- ПЛАВНОЕ ДВИЖЕНИЕ -----
      if (targetPulse > currentPulse) {
        for (int p = currentPulse; p <= targetPulse; p++) {
          pwm.setPWM(SERVO_CHANNEL, 0, p);
          delay(20);                     // скорость движения
        }
      } else {
        for (int p = currentPulse; p >= targetPulse; p--) {
          pwm.setPWM(SERVO_CHANNEL, 0, p);
          delay(20);
        }
      }
      currentPulse = targetPulse;        // запоминаем новое положение

      Serial.print("Серво плавно переведено в угол ");
      Serial.print(targetAngle);
      Serial.println("°");

      // ----- 5 ЗАМЕРОВ ПОТЕНЦИОМЕТРА -----
      const int numReadings = 5;
      int readings[numReadings];
      long sum = 0;

      Serial.println("Измерения потенциометра:");
      for (int i = 0; i < numReadings; i++) {
        readings[i] = analogRead(potPin);
        float voltage = readings[i] * (5.0 / 1023.0);
        sum += readings[i];

        Serial.print("  ");
        Serial.print(i + 1);
        Serial.print(": ADC = ");
        Serial.print(readings[i]);
        Serial.print(" (");
        Serial.print(voltage);
        Serial.println(" V)");

        delay(200);  // пауза между измерениями
      }

      // Среднее значение
      int average = sum / numReadings;
      float avgVoltage = average * (5.0 / 1023.0);
      Serial.print("Среднее (ADC) = ");
      Serial.print(average);
      Serial.print(", среднее напряжение = ");
      Serial.print(avgVoltage);
      Serial.println(" V");
      Serial.println();
    } else {
      Serial.println("Ошибка: угол должен быть от 0 до 180");
    }
  }
}