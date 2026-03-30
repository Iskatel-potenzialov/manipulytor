#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x40);

// ============================================================================
// 1. НАСТРОЙКИ PWM
// ============================================================================
#define SERVOMIN 102   // Минимальный импульс (0°)
#define SERVOMAX 512   // Максимальный импульс (180°)

// ============================================================================
// 2. КАНАЛЫ PCA9685 (4 сервопривода)
// ============================================================================
// Нумерация каналов: 0-15 (физические пины на плате)
#define SERVO_CHANNEL_0  0   // Серво 1 (база)
#define SERVO_CHANNEL_1  2   // Серво 2 (плечо)
#define SERVO_CHANNEL_2  4   // Серво 3 (предплечье)
#define SERVO_CHANNEL_3  6   // Серво 4 (запястье)

#define NUM_SERVOS 4

// ============================================================================
// 3. ПЕРЕМЕННЫЕ 
// ============================================================================
const int shutdownPin = 7;
bool systemStopped = false;  
float currentAngles[NUM_SERVOS] = {65, 115, 92, 110}; // Базовая

// ============================================================================
// 4. ОГРАНИЧЕНИЯ УГЛОВ (механические пределы серво)
// ============================================================================
const float SERVO_MIN_ANGLES[NUM_SERVOS] = {0, 40, 0, 0};      // Мин углы
const float SERVO_MAX_ANGLES[NUM_SERVOS] = {130, 170, 150, 170}; // Макс углы

// ============================================================================
// 5. ФУНКЦИЯ: Получить номер канала
// ============================================================================
int getServoChannel(int index) {
  switch(index) {
    case 0: return SERVO_CHANNEL_0;
    case 1: return SERVO_CHANNEL_1;
    case 2: return SERVO_CHANNEL_2;
    case 3: return SERVO_CHANNEL_3;
    default: return 0;
  }
}

// ============================================================================
// 6. ФУНКЦИЯ: Конвертация угол → PWM импульс (с ограничениями)
// ============================================================================
int angleToPulse(float angle, int servoIndex) {
  // Ограничиваем угол для конкретного серво
  angle = constrain(angle, SERVO_MIN_ANGLES[servoIndex], SERVO_MAX_ANGLES[servoIndex]);
  return map(angle, 0, 180, SERVOMIN, SERVOMAX);
}

// ============================================================================
// 7. ФУНКЦИЯ: Установка углов (с плавностью для Home)
// ============================================================================
void setServoAngles(float angles[NUM_SERVOS], bool smooth = false) {
  if (smooth) {
    // ⭐ ПЛАВНОЕ движение (для HOME) - 50 шагов × 20мс = 1 секунда
    int steps = 50;
    float stepAngles[NUM_SERVOS];
    
    // Вычисляем шаг для каждого серво
    for (int i = 0; i < NUM_SERVOS; i++) {
      stepAngles[i] = (angles[i] - currentAngles[i]) / steps;
    }
    
    // Пошаговое движение
    for (int s = 0; s < steps; s++) {
      for (int i = 0; i < NUM_SERVOS; i++) {
        float angle = currentAngles[i] + stepAngles[i];
        int pulse = angleToPulse(angle, i);
        pwm.setPWM(getServoChannel(i), 0, pulse);
      }
      delay(20);  // Задержка между шагами
    }
    
    // Сохраняем финальные углы
    for (int i = 0; i < NUM_SERVOS; i++) {
      currentAngles[i] = angles[i];
    }
  } else {
    // ⭐ МГНОВЕННОЕ движение (для LSTM траектории)
    for (int i = 0; i < NUM_SERVOS; i++) {
      int pulse = angleToPulse(angles[i], i);
      pwm.setPWM(getServoChannel(i), 0, pulse);
      currentAngles[i] = angles[i];
    }
  }
  

}

// ============================================================================
// 8. ФУНКЦИЯ: Парсинг углов из команды SET_ANG
// ============================================================================
bool parseAngles(String cmd, float* angles) {
  int index = 0;
  String remaining = cmd.substring(8);  // После "SET_ANG,"
  
  while (remaining.length() > 0 && index < NUM_SERVOS) {
    int comma = remaining.indexOf(',');
    if (comma == -1) {
      angles[index] = remaining.toFloat();
      index++;
      break;
    }
    angles[index] = remaining.substring(0, comma).toFloat();
    remaining = remaining.substring(comma + 1);
    index++;
  }
  
  return (index == NUM_SERVOS);
}

// ============================================================================
// 9. ФУНКЦИЯ: Безопасная остановка
// ============================================================================
void safeShutdown() {
  Serial.println("🛑 SAFE SHUTDOWN");
  
  // Безопасные углы (серво не перегружены)
  float safeAngles[NUM_SERVOS] = {0, 90, 45, 90};
  setServoAngles(safeAngles, false);
  
  digitalWrite(shutdownPin, HIGH);  // Сигнал на отключение питания
  systemStopped = true;
  
  Serial.println("STOPPED");
}

// ============================================================================
// 10. ФУНКЦИЯ: Домашняя позиция (с плавностью!)
// ============================================================================
void goToHome() {
  float homeAngles[NUM_SERVOS] =  {65, 115, 92, 110}; //база
  
  // Проверка что home в пределах ограничений
  for (int i = 0; i < NUM_SERVOS; i++) {
    if (homeAngles[i] < SERVO_MIN_ANGLES[i] || homeAngles[i] > SERVO_MAX_ANGLES[i]) {
      Serial.print("⚠️ WARNING: Home angle ");
      Serial.print(homeAngles[i]);
      Serial.print(" for servo ");
      Serial.println(i);
    }
  }
  
  setServoAngles(homeAngles, true);  // ⭐ ПЛАВНО!
  Serial.println("HOME");
}

// ============================================================================
// 11. SETUP
// ============================================================================
void setup() {
  Serial.begin(115200);  
  while (!Serial) {
    ;  // Ждём подключения Serial
  }
  
  Serial.println("🤖 Arduino манипулятор готов (4 серво, open-loop)");
  
  // Инициализация PCA9685
  if (!pwm.begin()) {
    Serial.println("❌ Ошибка PCA9685!");
    while (1);
  }
  Serial.println("✅ PCA9685 инициализирован");
  
  pwm.setPWMFreq(50);  // 50 Гц для серво
  Serial.println("✅ PWM частота: 50 Гц");
  
  // Настройка пина отключения
  pinMode(shutdownPin, OUTPUT);
  digitalWrite(shutdownPin, LOW);
  Serial.println("✅ Shutdown pin настроен");
  
  // Переход в домашнюю позицию при старте
  Serial.println("🏠 Переход в домашнюю позицию...");
  goToHome();
  
  Serial.println("\n========================================");
  Serial.println("Команды:");
  Serial.println("  SET_ANG,a1,a2,a3,a4  - Установка углов");
  Serial.println("  GET_ANG              - Получить текущие углы");
  Serial.println("  HOME                 - Домашняя позиция");
  Serial.println("  STOP                 - Безопасная остановка");
  Serial.println("  START                - Выход из остановки");
  Serial.println("  PING                 - Проверка связи");
  Serial.println("========================================\n");
}

// ============================================================================
// 12. LOOP
// ============================================================================
void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command.length() == 0) return;
    
    // ========================================================================
    // Команда: SET_ANG,a1,a2,a3,a4 (LSTM траектория)
    // ========================================================================
    if (command.startsWith("SET_ANG")) {
      if (systemStopped) {
        Serial.println("ERROR: Система остановлена. Отправьте START");
        return;
      }
      
      float angles[NUM_SERVOS];
      if (parseAngles(command, angles)) {
        setServoAngles(angles, false);  // ⭐ МГНОВЕННО для LSTM!
        Serial.println("OK");
      } else {
        Serial.println("ERROR: Неверный формат. Пример: SET_ANG,66.2,116.4,92.9,116.3");
      }
    }
    
    // ========================================================================
    // Команда: GET_ANG (вернуть текущие углы из памяти)
    // ========================================================================
    else if (command == "GET_ANG") {
      Serial.print(currentAngles[0]);
      Serial.print(",");
      Serial.print(currentAngles[1]);
      Serial.print(",");
      Serial.print(currentAngles[2]);
      Serial.print(",");
      Serial.println(currentAngles[3]);
    }
    
    // ========================================================================
    // Команда: HOME (домашняя позиция, плавно)
    // ========================================================================
    else if (command == "HOME") {
      if (systemStopped) {
        Serial.println("ERROR: Система остановлена. Отправьте START");
        return;
      }
      goToHome();
    }
    
    // ========================================================================
    // Команда: STOP (безопасная остановка)
    // ========================================================================
    else if (command == "STOP") {
      safeShutdown();
    }
    
    // ========================================================================
    // Команда: START (выход из остановки)
    // ========================================================================
    else if (command == "START") {
      systemStopped = false;
      digitalWrite(shutdownPin, LOW);
      goToHome();
      Serial.println("STARTED");
    }
    
    // ========================================================================
    // Команда: PING (проверка связи)
    // ========================================================================
    else if (command == "PING") {
      Serial.println("PONG");
    }
    
    // ========================================================================
    // Неизвестная команда
    // ========================================================================
    else {
      Serial.print("ERROR: Неизвестная команда: ");
      Serial.println(command);
    }
  }
  
  delay(5);  // Минимальная задержка для стабильности
}