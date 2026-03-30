# ============================================================================
# Воспроизведение заранее записанной траектории
# Режимы: SIMULATION (PyBullet) или REAL_ROBOT (Arduino)
# С добавленным переходом в фиксированную точку с шага 195
# ============================================================================

import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import json
import serial

# ============================================================================
# 1. НАСТРОЙКИ РЕЖИМА РАБОТЫ И ПАРАМЕТРЫ
# ============================================================================
MODE = 'SIMULATION'           # 'SIMULATION' или 'REAL_ROBOT'
TRAJECTORY_INDEX = 50          # номер траектории в датасете (начиная с 0)

# Файлы с данными
TRAJECTORY_NPY = 'manual_trajectories_combined.npy'
TRAJECTORY_JSON = 'trajectory_info_combined.json'

REAL_ROBOT_CONFIG = {
    'port': 'COM3',
    'baudrate': 115200,
    'timeout': 1.0,
}

TRAJECTORY_CONFIG = {
    'num_steps': 200,          # длина траектории (должна совпадать с датасетом)
    'sim_steps': 1,             # шагов симуляции на один шаг траектории
    'min_delay': 0.08,          # минимальная задержка между командами (сек)
    'max_speed': 300,           # макс скорость сервы (град/сек) для расчёта задержки
}

# ============================================================================
# НОВЫЙ ПАРАМЕТР: точка, в которую переходим с шага 195
# ============================================================================
TARGET_FINAL_REAL = [116, 94, 87, 95]   # откалиброванная точка 26 (углы в градусах)

print("=" * 80)
print("🤖 ВОСПРОИЗВЕДЕНИЕ ТРАЕКТОРИИ ИЗ ДАТАСЕТА")
print("=" * 80)
print(f"📍 Режим работы: {MODE}")
print(f"📍 Индекс траектории: {TRAJECTORY_INDEX}")
print(f"🎯 Финальная точка (реальные углы): {TARGET_FINAL_REAL}")
print("=" * 80)

# ============================================================================
# 2. ЗАГРУЗКА ДАТАСЕТА И МЕТАДАННЫХ
# ============================================================================
print("\n📦 ЗАГРУЗКА ДАННЫХ")
print("=" * 80)

try:
    trajectories = np.load(TRAJECTORY_NPY)
    print(f"✅ Датасет загружен: {TRAJECTORY_NPY}")
    print(f"   Форма: {trajectories.shape} (траекторий x {trajectories.shape[1]} шагов x 4 сустава)")
except Exception as e:
    print(f"❌ Ошибка загрузки {TRAJECTORY_NPY}: {e}")
    exit(1)

try:
    with open(TRAJECTORY_JSON, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    print(f"✅ Метаданные загружены: {TRAJECTORY_JSON}")
except Exception as e:
    print(f"⚠️ Не удалось загрузить метаданные (будут использоваться только траектории): {e}")
    meta = None

if TRAJECTORY_INDEX < 0 or TRAJECTORY_INDEX >= trajectories.shape[0]:
    print(f"❌ Индекс траектории {TRAJECTORY_INDEX} вне диапазона (0-{trajectories.shape[0]-1})")
    exit(1)

# Извлекаем нужную траекторию (форма: (200, 4))
traj_norm = trajectories[TRAJECTORY_INDEX]   # уже нормализованные [0,1]
print(f"\n📌 Выбрана траектория #{TRAJECTORY_INDEX}")
if meta and TRAJECTORY_INDEX < len(meta):
    print(f"   Метаданные: {meta[TRAJECTORY_INDEX]}")
else:
    print("   Метаданные отсутствуют")

# ============================================================================
# 3. КАЛИБРОВОЧНЫЕ ПАРАМЕТРЫ 
# ============================================================================
print("\n🔧 КАЛИБРОВОЧНЫЕ ПАРАМЕТРЫ")
print("=" * 80)


CALIB = {
    1: {'offset': 67,  'dir': 1,  'A': 523,  'scale': 3.81},
    2: {'offset': 126, 'dir': -1, 'A': 869,  'scale': 3.65}, 
    3: {'offset': 9,  'dir': 1,  'A': 740,  'scale': 3.93},  
    4: {'offset': 24,  'dir': 1,  'A': 963,  'scale': 3.89}, 

}

def urdf_to_real(urdf_angle, servo_num):
    cfg = CALIB[servo_num]
    return (urdf_angle * cfg['dir']) + cfg['offset']

def real_to_urdf(real_angle, servo_num):
    cfg = CALIB[servo_num]
    return (real_angle - cfg['offset']) * cfg['dir']

def real_to_pot(real_angle, servo_num):
    cfg = CALIB[servo_num]
    return cfg['A'] - (real_angle * cfg['scale'])

def pot_to_real(pot, servo_num):
    cfg = CALIB[servo_num]
    return (cfg['A'] - pot) / cfg['scale']

def real_to_norm(real_angle, servo_num):
    pot = real_to_pot(real_angle, servo_num)
    return pot / 1023.0

def norm_to_real(norm_value, servo_num):
    pot = norm_value * 1023.0
    return pot_to_real(pot, servo_num)

print("   ✅ Функции конвертации готовы")

# ============================================================================
# 4. ОПРЕДЕЛЕНИЕ ДОМАШНЕЙ ПОЗИЦИИ (HOME) И ЦЕЛЕВОЙ ТОЧКИ
# ============================================================================
HOME_REAL_DEG = [66.19, 116.39, 92.91, 116.26]      # Базовая точка неоткалиброванная
HOME_URDF_DEG = [real_to_urdf(HOME_REAL_DEG[i], i+1) for i in range(4)]
HOME_URDF_RAD = [math.radians(a) for a in HOME_URDF_DEG]
HOME_NORM = [real_to_norm(HOME_REAL_DEG[i], i+1) for i in range(4)]

# Преобразуем финальную целевую точку в нормализованные значения
TARGET_FINAL_NORM = [real_to_norm(TARGET_FINAL_REAL[i], i+1) for i in range(4)]

print(f"\n📍 БАЗОВАЯ ПОЗИЦИЯ (HOME):")
print(f"   Реальные углы: {HOME_REAL_DEG}")
print(f"   Нормализованные: {[round(n,3) for n in HOME_NORM]}")
print(f"\n🎯 ФИНАЛЬНАЯ ЦЕЛЕВАЯ ПОЗИЦИЯ:")
print(f"   Реальные углы: {TARGET_FINAL_REAL}")
print(f"   Нормализованные: {[round(n,4) for n in TARGET_FINAL_NORM]}")

# ============================================================================
# 5. КЛАСС УПРАВЛЕНИЯ РЕАЛЬНЫМ РОБОТОМ
# ============================================================================
class RealRobotController:
    def __init__(self, port='COM3', baudrate=115200, timeout=1.0):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(5)
            self.ser.reset_input_buffer() 
            self.current_angles = HOME_REAL_DEG.copy()
            self.is_connected = True
            print(f"   ✅ Подключено к {port}")
        except Exception as e:
            print(f"   ❌ Ошибка подключения: {e}")
            self.is_connected = False
    
    def close(self):
        if self.is_connected:
            self.ser.close()
            print("   ✅ Serial порт закрыт")
    
    def send_command(self, cmd):
        if not self.is_connected:
            return "ERROR"
        try:
            self.ser.write((cmd + '\n').encode())
            response = self.ser.readline().decode().strip()
            return response
        except Exception as e:
            print(f"   ⚠️ Ошибка Serial: {e}")
            return "ERROR"
    
    def set_servo_angles(self, real_angles_deg):
        if not self.is_connected:
            return False
        
        cmd = f'SET_ANG,{real_angles_deg[0]:.2f},{real_angles_deg[1]:.2f},{real_angles_deg[2]:.2f},{real_angles_deg[3]:.2f}'
        response = self.send_command(cmd)
        
        if response == 'OK':
            self.current_angles = real_angles_deg.copy()
            return True
        else:
            print(f"   ⚠️ Arduino ответ: {response}")
            return False
    
    def get_current_angles(self):
        return self.current_angles.copy()
    
    def go_home(self):
        response = self.send_command('HOME')
        if response == 'HOME':
            self.current_angles = HOME_REAL_DEG.copy()
            return True
        return False
    
    def stop(self):
        response = self.send_command('STOP')
        return response == 'STOPPED'
    
    def start(self):
        response = self.send_command('START')
        if response == 'STARTED':
            self.current_angles = HOME_REAL_DEG.copy()
            return True
        return False
    
    def ping(self):
        response = self.send_command('PING')
        return response == 'PONG'
    
    def calculate_delay(self, old_angles, new_angles):
        max_diff = max(abs(new_angles[i] - old_angles[i]) for i in range(4))
        delay_time = max(max_diff / TRAJECTORY_CONFIG['max_speed'], TRAJECTORY_CONFIG['min_delay'])
        return delay_time

# ============================================================================
# 6. ПОДГОТОВКА СИМУЛЯЦИИ ИЛИ РЕАЛЬНОГО РОБОТА
# ============================================================================
robot_id = None
physicsClient = None
robot = None

if MODE == 'SIMULATION':
    print("\n🚀 ЗАПУСК PYBULLET СИМУЛЯЦИИ")
    print("=" * 80)
    
    physicsClient = p.connect(p.GUI, options='--renderDevice=egl --width=1024 --height=768')
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(
        cameraDistance=0.7,
        cameraYaw=90,
        cameraPitch=-10,
        cameraTargetPosition=[0.075, 0, 0.15]
    )
    
    robot_id = p.loadURDF("5dof_manipulator_120326.urdf", useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    print(f"   ✅ Загружено суставов: {num_joints}")
    
    # ============================================================================
    # РУЧНОЕ СОЗДАНИЕ ШАХМАТНОЙ ДОСКИ (32×32 мм клетки)
    # ============================================================================
    print("\n🏁 Создание шахматной доски...")
    square_size = 0.032
    board_size = 8
    board_center = [0.168, 0, -0.04]
    colors = [(0.9, 0.9, 0.9, 1), (0.3, 0.3, 0.3, 1)]
    cells_created = 0
    for row in range(board_size):
        for col in range(board_size):
            x = board_center[0] + (col - board_size/2 + 0.5) * square_size
            y = board_center[1] + (row - board_size/2 + 0.5) * square_size
            z = board_center[2]
            color_idx = (row + col) % 2
            visual = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[square_size/2, square_size/2, 0.001],
                rgbaColor=colors[color_idx]
            )
            collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=[square_size/2, square_size/2, 0.001]
            )
            p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=collision,
                baseVisualShapeIndex=visual,
                basePosition=[x, y, z]
            )
            cells_created += 1
    print(f"   ✅ Создана шахматная доска {board_size}x{board_size} клеток")
    
    # Установка робота в начальную позицию траектории (первый шаг)
    start_norm = traj_norm[0]
    start_real = [norm_to_real(start_norm[i], i+1) for i in range(4)]
    start_urdf = [real_to_urdf(start_real[i], i+1) for i in range(4)]
    start_rad = [math.radians(a) for a in start_urdf]
    for i in range(4):
        p.resetJointState(robot_id, i, start_rad[i])
    p.stepSimulation()
    time.sleep(0.5)
    print(f"   ✅ Робот установлен в начальное положение траектории")

elif MODE == 'REAL_ROBOT':
    print("\n🔌 ПОДКЛЮЧЕНИЕ К РЕАЛЬНОМУ РОБОТУ")
    print("=" * 80)
    
    robot = RealRobotController(
        port=REAL_ROBOT_CONFIG['port'],
        baudrate=REAL_ROBOT_CONFIG['baudrate'],
        timeout=REAL_ROBOT_CONFIG['timeout']
    )
    
    if not robot.is_connected:
        print("   ❌ Не удалось подключиться к роботу!")
        exit()
    
    if not robot.ping():
        print("   ❌ Нет ответа от Arduino!")
        exit()
    print("   ✅ Связь с Arduino установлена")
    
    # Перемещаем робота в начальную позицию траектории
    start_norm = traj_norm[0]
    start_real = [norm_to_real(start_norm[i], i+1) for i in range(4)]
    print(f"   🏠 Переход в начальную позицию траектории: {[round(a,1) for a in start_real]}")
    robot.set_servo_angles(start_real)
    time.sleep(2)
    print("   ✅ Начальная позиция установлена")

# ============================================================================
# 7. ВОСПРОИЗВЕДЕНИЕ ТРАЕКТОРИИ С ПЕРЕХОДОМ В ФИНАЛЬНУЮ ТОЧКУ С ШАГА 195
# ============================================================================
print(f"\n🎬 ВОСПРОИЗВЕДЕНИЕ ТРАЕКТОРИИ #{TRAJECTORY_INDEX}")
print(f"   Шагов: {traj_norm.shape[0]}")
print(f"   Переход в финальную точку начинается с шага 195")
print("   " + "=" * 50)

prev_real_angles = None
if MODE == 'REAL_ROBOT':
    prev_real_angles = [norm_to_real(traj_norm[0][i], i+1) for i in range(4)]

# Переменная для запоминания состояния на 194 шаге
state_at_194 = None

for step in range(traj_norm.shape[0]):
    # Определяем, какой набор углов использовать
    if step == 194:
        # Запоминаем состояние перед началом перехода
        state_at_194 = traj_norm[step].copy()
        angles_norm = traj_norm[step]
    elif step >= 195 and state_at_194 is not None:
        # Линейная интерполяция от состояния на 194 шаге к целевой точке
        alpha = (step - 194) / (traj_norm.shape[0] - 1 - 194)  # всего 5 шагов
        angles_norm = (1 - alpha) * np.array(state_at_194) + alpha * np.array(TARGET_FINAL_NORM)
    else:
        angles_norm = traj_norm[step]
    
    # Конвертация в реальные углы
    angles_real = [norm_to_real(angles_norm[i], i+1) for i in range(4)]
    
    # Вывод для каждого 10-го шага (и последнего)
    if step % 10 == 0 or step == traj_norm.shape[0] - 1:
        pot_values = [v * 1023.0 for v in angles_norm]
        print(f"\n📊 Шаг {step:3d}:")
        print(f"   Потенциометры (ADC): {[int(round(p)) for p in pot_values]}")
        print(f"   Нормализованные:     {[round(v, 4) for v in angles_norm]}")
        print(f"   Углы (град):         {[round(a, 2) for a in angles_real]}")
    
    if MODE == 'SIMULATION':
        # Преобразование в URDF и радианы
        angles_urdf = [real_to_urdf(angles_real[i], i+1) for i in range(4)]
        angles_rad = [math.radians(a) for a in angles_urdf]
        
        for i in range(4):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                                   targetPosition=angles_rad[i], force=2000)
        
        for _ in range(TRAJECTORY_CONFIG['sim_steps']):
            p.stepSimulation()
            time.sleep(1/240)
        
        # Визуализация точки концевого эффектора (жёлтая)
        state = p.getLinkState(robot_id, num_joints-1)
        p.addUserDebugPoints([state[4]], [[1, 1, 0]], 3)
        
    elif MODE == 'REAL_ROBOT':
        robot.set_servo_angles(angles_real)
        delay = robot.calculate_delay(prev_real_angles, angles_real)
        time.sleep(delay)
        prev_real_angles = angles_real.copy()
    
    # Вывод прогресса
    if step % 20 == 0 or step == traj_norm.shape[0]-1:
        if MODE == 'SIMULATION':
            pos = p.getLinkState(robot_id, num_joints-1)[4]
            print(f"   Шаг {step:3d}/199: позиция концевика {pos}")
        else:
            print(f"   Шаг {step:3d}/199: углы {[round(a,1) for a in angles_real]}")

print("   " + "=" * 50)
print("✅ Траектория воспроизведена")

# ============================================================================
# 8. ЗАВЕРШЕНИЕ РАБОТЫ
# ============================================================================
print("\n" + "=" * 80)
print("✅ ЗАВЕРШЕНИЕ РАБОТЫ")
print("=" * 80)

if MODE == 'SIMULATION':
    input("\n👉 Нажмите Enter для завершения симуляции...")
    p.disconnect()
    print("👋 PyBullet отключён")

elif MODE == 'REAL_ROBOT':
    print("\n🏠 Возврат в домашнюю позицию...")
    robot.go_home()
    time.sleep(2)
    robot.close()
    print("👋 Реальный робот отключён")
    input("\n👉 Нажмите Enter для завершения...")

print("\n✅ Программа завершена успешно!")