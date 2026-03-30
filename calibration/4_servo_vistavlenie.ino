#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ============================================================================
// 1. ПАРАМЕТРЫ СЕРВО
// ============================================================================
#define SERVO_MIN 102      // 0°
#define SERVO_MAX 512      // 180°

// Каналы PCA9685 для 4 сервоприводов
#define SERVO_CHANNEL_0  0
#define SERVO_CHANNEL_1  2
#define SERVO_CHANNEL_2  4
#define SERVO_CHANNEL_3  6

#define NUM_SERVOS 4

// ============================================================================
// 2. ПИНЫ ПОТЕНЦИОМЕТРОВ (4 шт)
// ============================================================================
const int potPins[NUM_SERVOS] = {A0, A1, A2, A3};

// ============================================================================
// 3. ОБЪЕКТ PCA9685
// ============================================================================
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// ============================================================================
// 4. ТЕКУЩИЕ ИМПУЛЬСЫ (для плавного движения)
// ============================================================================
int currentPulses[NUM_SERVOS];

// ============================================================================
// 5. ФУНКЦИЯ: Получить номер канала по индексу серво (0-3)
// ============================================================================
int getServoChannel(int idx) {
  switch(idx) {
    case 0: return SERVO_CHANNEL_0;
    case 1: return SERVO_CHANNEL_1;
    case 2: return SERVO_CHANNEL_2;
    case 3: return SERVO_CHANNEL_3;
    default: return 0;
  }
}

// ============================================================================
// 6. SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);
  Serial.println("Введите 4 угла через запятую (0-180) для 4 серво.");
  Serial.println("Пример: 67,120,15,27");
  Serial.println("Будет выполнено 5 замеров каждого потенциометра.");

  pwm.begin();
  pwm.setPWMFreq(50);      // 50 Гц для серво
  delay(10);

  // Начальное положение (можно задать свои начальные углы)
  int initAngles[NUM_SERVOS] = {64, 116, 92, 110};  // например, из калибровки
  for (int i = 0; i < NUM_SERVOS; i++) {
    currentPulses[i] = map(initAngles[i], 0, 180, SERVO_MIN, SERVO_MAX);
    pwm.setPWM(getServoChannel(i), 0, currentPulses[i]);
  }

  Serial.println("Серво в начальном положении.");
}

// ============================================================================
// 7. ПАРСИНГ ВХОДНОЙ СТРОКИ (4 целых числа через запятую)
// ============================================================================
bool parseAngles(String input, int angles[NUM_SERVOS]) {
  int idx = 0;
  char *ptr;
  char str[50];
  input.toCharArray(str, sizeof(str));
  char *token = strtok(str, ",");
  while (token != NULL && idx < NUM_SERVOS) {
    angles[idx++] = atoi(token);
    token = strtok(NULL, ",");
  }
  return (idx == NUM_SERVOS);
}

// ============================================================================
// 8. ПЛАВНОЕ ДВИЖЕНИЕ ВСЕХ СЕРВО ОДНОВРЕМЕННО
// ============================================================================
void smoothMove(int targetPulses[NUM_SERVOS]) {
  // Определяем максимальное количество шагов (максимальная разница импульсов)
  int maxSteps = 0;
  for (int i = 0; i < NUM_SERVOS; i++) {
    int steps = abs(targetPulses[i] - currentPulses[i]);
    if (steps > maxSteps) maxSteps = steps;
  }

  // Пошаговое движение всех серво
  for (int step = 0; step <= maxSteps; step++) {
    for (int i = 0; i < NUM_SERVOS; i++) {
      int newPulse;
      if (targetPulses[i] > currentPulses[i]) {
        newPulse = currentPulses[i] + step;
        if (newPulse > targetPulses[i]) newPulse = targetPulses[i];
      } else if (targetPulses[i] < currentPulses[i]) {
        newPulse = currentPulses[i] - step;
        if (newPulse < targetPulses[i]) newPulse = targetPulses[i];
      } else {
        newPulse = currentPulses[i];
      }
      pwm.setPWM(getServoChannel(i), 0, newPulse);
    }
    delay(15);  // скорость движения (можно регулировать)
  }

  // Обновляем текущие импульсы
  for (int i = 0; i < NUM_SERVOS; i++) {
    currentPulses[i] = targetPulses[i];
  }
}

// ============================================================================
// 9. ИЗМЕРЕНИЕ ПОТЕНЦИОМЕТРОВ (5 замеров для каждого, вывод среднего)
// ============================================================================
void readAllPotentiometers() {
  const int numReadings = 5;
  Serial.println("\nИзмерения потенциометров (5 замеров каждого):");

  for (int servo = 0; servo < NUM_SERVOS; servo++) {
    long sum = 0;
    Serial.print("Серво ");
    Serial.print(servo + 1);
    Serial.print(" (пин A");
    Serial.print(servo);
    Serial.println("):");

    for (int i = 0; i < numReadings; i++) {
      int value = analogRead(potPins[servo]);
      float voltage = value * (5.0 / 1023.0);
      sum += value;

      Serial.print("  ");
      Serial.print(i + 1);
      Serial.print(": ADC = ");
      Serial.print(value);
      Serial.print(" (");
      Serial.print(voltage);
      Serial.println(" V)");

      delay(200); // пауза между замерами
    }

    int average = sum / numReadings;
    float avgVoltage = average * (5.0 / 1023.0);
    Serial.print("  Среднее: ADC = ");
    Serial.print(average);
    Serial.print(", напряжение = ");
    Serial.print(avgVoltage);
    Serial.println(" V\n");
  }
}

// ============================================================================
// 10. LOOP
// ============================================================================
void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();

    if (input.length() == 0) return;

    // Специальная команда "read" – только чтение потенциометров
    if (input.equalsIgnoreCase("read")) {
      readAllPotentiometers();
      return;
    }

    int angles[NUM_SERVOS];
    if (!parseAngles(input, angles)) {
      Serial.println("Ошибка: введите 4 угла через запятую (пример: 67,120,15,27)");
      return;
    }

    // Проверка диапазона
    bool valid = true;
    for (int i = 0; i < NUM_SERVOS; i++) {
      if (angles[i] < 0 || angles[i] > 180) {
        Serial.print("Ошибка: угол ");
        Serial.print(angles[i]);
        Serial.println(" должен быть от 0 до 180");
        valid = false;
      }
    }
    if (!valid) return;

    // Преобразуем углы в импульсы
    int targetPulses[NUM_SERVOS];
    for (int i = 0; i < NUM_SERVOS; i++) {
      targetPulses[i] = map(angles[i], 0, 180, SERVO_MIN, SERVO_MAX);
    }

    // Плавное движение
    smoothMove(targetPulses);

    // Вывод информации
    Serial.print("Серво плавно переведены в углы: ");
    for (int i = 0; i < NUM_SERVOS; i++) {
      Serial.print(angles[i]);
      if (i < NUM_SERVOS - 1) Serial.print(", ");
    }
    Serial.println("°");

    // Измерение потенциометров
    readAllPotentiometers();
  }
}