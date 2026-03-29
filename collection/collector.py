# Импорт библиотек
import serial      # Для работы с USB-портом (Arduino)
import csv         # Для записи файлов формата CSV
import time        # Для пауз и работы со временем

# --- НАСТРОЙКИ ---
PORT = 'COM3'          # ТВОЙ порт (проверь в Диспетчере устройств)
BAUDRATE = 115200      # Скорость связи (как в коде Arduino)
FILENAME = 'data_17_25.csv'  # Имя файла для записи

# Подключение к Arduino
try:
    arduino = serial.Serial(PORT, BAUDRATE, timeout=1)
    time.sleep(2)      # Ждем перезагрузки платы
    print(f"Подключено к {PORT}")
except:
    print("Ошибка: не удалось подключиться к порту. Проверь номер COM.")
    exit()

# Открытие файла для записи
with open(FILENAME, mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Записываем заголовок таблицы
    writer.writerow(['time_ms', 'pot1', 'pot2', 'pot3', 'pot4'])
    print("Запись началась. Нажми Ctrl+C для остановки.")

    try:
        while True:
            # Проверяем, есть ли данные в порту
            if arduino.in_waiting > 0:
                # Читаем строку до символа переноса
                line = arduino.readline().decode('utf-8').strip()
                
                # Разбиваем строку "123,456..." на список
                data = line.split(',')
                
                # Проверка: должно быть 5 чисел (время + 4 потенциометра)
                if len(data) == 5:
                    writer.writerow(data)  # Сохраняем в CSV
                    # print(data)          # Можно раскомментировать для вывода в консоль
                else:
                    print(f"Пропуск битой строки: {line}")
                    
    except KeyboardInterrupt:
        # Безопасное завершение по Ctrl+C
        print("\nЗапись завершена пользователем.")
    finally:
        arduino.close()  # Закрываем порт
        print(f"Файл сохранен: {FILENAME}")