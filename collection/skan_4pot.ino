const int pots[4] = {A0, A1, A2, A3};
int smooth[4] = {0, 0, 0, 0};
unsigned long lastTime = 0;
const int interval = 10; // 10 мс = 100 Гц

void setup() {
  Serial.begin(115200);
  // Инициализация начальных значений
  for(int i=0; i<4; i++) smooth[i] = analogRead(pots[i]);
}

void loop() {
  if (millis() - lastTime >= interval) {
    lastTime = millis();
    
    Serial.print(millis()); // Время
    
    for (int i = 0; i < 4; i++) {
      int raw = analogRead(pots[i]);
      // Экспоненциальное сглаживание (коэфф. 5)
      smooth[i] = (raw + smooth[i] * 4) / 5; 
      
      Serial.print(",");
      Serial.print(smooth[i]);
    }
    Serial.println();
  }
}
