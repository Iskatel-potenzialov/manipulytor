import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import torch
import torch.nn as nn
import json
import serial
import cv2
import os

# ============================================================================
# 1. НАСТРОЙКИ РЕЖИМА РАБОТЫ
# ============================================================================
MODE = 'SIMULATION'  # 'SIMULATION' или 'REAL_ROBOT'

REAL_ROBOT_CONFIG = {
    'port': 'COM3',
    'baudrate': 115200,
    'timeout': 1.0,
}

TRAJECTORY_CONFIG = {
    'num_steps': 200,
    'sim_steps': 1,
    'min_delay': 0.08,
    'max_speed': 300,
}

# ⭐ НАСТРОЙКИ BLENDING (наведение на цель)
BLEND_CONFIG = {
    'enabled': True,           # Включить blending
    'start_step': 100,         # Начинать наведение за 50 шагов до конца 120
}

# ============================================================================
# НАСТРОЙКИ СОХРАНЕНИЯ/ЗАГРУЗКИ ТРАЕКТОРИЙ
# ============================================================================
SAVE_TRAJECTORIES = False          # Сохранять сгенерированные траектории в файл
LOAD_TRAJECTORIES = False          # Загружать траектории из файла вместо генерации
TRAJECTORY_FILE = "trajectories_2modeli.npy"  # Имя файла для сохранения/загрузки trajectories.npy

# ============================================================================
# НАСТРОЙКИ LSTM МОДЕЛЕЙ ДЛЯ РАЗНЫХ НАПРАВЛЕНИЙ
# ============================================================================
# Модель для движения БАЗА → ТОЧКА (point)
LSTM_MODEL_POINT = 'lstm_trajectory_best_weighted_point.pth'
NORM_PARAMS_POINT = 'lstm_normalization_params_weighted_512_300e.json'
# Модель для движения ТОЧКА → БАЗА (base)
LSTM_MODEL_BASE = 'lstm_trajectory_best_weighted_baza.pth'
NORM_PARAMS_BASE = 'lstm_normalization_params_weighted_512_300e.json'

print("=" * 80)
print("🤖 LSTM ТРАЕКТОРИЯ: ГИБРИДНЫЙ ПОДХОД (Open-Loop + Blending)")
print("=" * 80)
print(f"📍 Режим работы: {MODE}")
if MODE == 'REAL_ROBOT':
    print(f"📍 Порт: {REAL_ROBOT_CONFIG['port']}")
    print(f"📍 Baudrate: {REAL_ROBOT_CONFIG['baudrate']}")
print(f"📍 Шагов траектории: {TRAJECTORY_CONFIG['num_steps']}")
if BLEND_CONFIG['enabled']:
    print(f"📍 Blending: шаги {BLEND_CONFIG['start_step']}-{TRAJECTORY_CONFIG['num_steps']}")
print(f"📍 Сохранять траектории: {SAVE_TRAJECTORIES}")
print(f"📍 Загружать траектории: {LOAD_TRAJECTORIES}")
if LOAD_TRAJECTORIES or SAVE_TRAJECTORIES:
    print(f"📍 Файл траекторий: {TRAJECTORY_FILE}")
print("=" * 80)

# ============================================================================
# 2. ЗАГРУЗКА LSTM МОДЕЛЕЙ
# ============================================================================
print("\n" + "=" * 80)
print("📦 ЗАГРУЗКА LSTM МОДЕЛЕЙ")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"📍 Устройство: {device}")

class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size=9, hidden_size=512, num_layers=2, output_size=4, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        lstm_out = self.dropout(lstm_out)
        return self.fc(lstm_out)

# Переменные для моделей
lstm_model_point = None
lstm_model_base = None
norm_params_point = None
norm_params_base = None
LSTM_AVAILABLE_POINT = False
LSTM_AVAILABLE_BASE = False

# Загрузка модели для движения к точке (point)
try:
    lstm_model_point = TrajectoryLSTM(input_size=9, hidden_size=512, num_layers=2, output_size=4, dropout=0.2)
    lstm_model_point.load_state_dict(torch.load(LSTM_MODEL_POINT, map_location=device))
    lstm_model_point.to(device)
    lstm_model_point.eval()
    print(f"   ✅ Модель POINT загружена: {LSTM_MODEL_POINT}")
    
    with open(NORM_PARAMS_POINT, 'r') as f:
        norm_params_point = json.load(f)
    print(f"   ✅ Параметры нормализации POINT загружены")
    LSTM_AVAILABLE_POINT = True
except FileNotFoundError as e:
    print(f"   ⚠️ Файл модели POINT не найден: {e}")
    LSTM_AVAILABLE_POINT = False
except Exception as e:
    print(f"   ❌ Ошибка загрузки модели POINT: {e}")
    LSTM_AVAILABLE_POINT = False

# Загрузка модели для движения к базе (base)
try:
    lstm_model_base = TrajectoryLSTM(input_size=9, hidden_size=512, num_layers=2, output_size=4, dropout=0.2)
    lstm_model_base.load_state_dict(torch.load(LSTM_MODEL_BASE, map_location=device))
    lstm_model_base.to(device)
    lstm_model_base.eval()
    print(f"   ✅ Модель BASE загружена: {LSTM_MODEL_BASE}")
    
    with open(NORM_PARAMS_BASE, 'r') as f:
        norm_params_base = json.load(f)
    print(f"   ✅ Параметры нормализации BASE загружены")
    LSTM_AVAILABLE_BASE = True
except FileNotFoundError as e:
    print(f"   ⚠️ Файл модели BASE не найден: {e}")
    LSTM_AVAILABLE_BASE = False
except Exception as e:
    print(f"   ❌ Ошибка загрузки модели BASE: {e}")
    LSTM_AVAILABLE_BASE = False

# Для обратной совместимости (если нужна проверка наличия хотя бы одной модели)
LSTM_AVAILABLE = LSTM_AVAILABLE_POINT or LSTM_AVAILABLE_BASE
if not LSTM_AVAILABLE:
    print("   ⚠️ Нет ни одной загруженной LSTM модели. Генерация траекторий невозможна.")

# ============================================================================
# 3. КАЛИБРОВОЧНЫЕ ПАРАМЕТРЫ (теперь для 5 сервоприводов)
# ============================================================================
print("\n" + "=" * 80)
print("🔧 КАЛИБРОВОЧНЫЕ ПАРАМЕТРЫ")
print("=" * 80)

# Для серво 5 (клешня) используются приблизительные значения,
# т.к. точная калибровка неизвестна. offset=0, dir=1, A=512, scale=1.
CALIB = {
    1: {'offset': 67,  'dir': 1,  'A': 523,  'scale': 3.81},
    2: {'offset': 126, 'dir': -1, 'A': 869,  'scale': 3.65}, 
    3: {'offset': 9,   'dir': 1,  'A': 740,  'scale': 3.93}, 
    4: {'offset': 24,  'dir': 1,  'A': 963,  'scale': 3.89}, 
    5: {'offset': 0,   'dir': 1,  'A': 512,  'scale': 1.0},  
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

print("   ✅ Функции конвертации готовы (включая серво 5)")

# ============================================================================
# 4. ПОЗИЦИИ РОБОТА (теперь 5 углов)
# ============================================================================

HOME_REAL_DEG = [65, 115, 92, 110, 90]      # Базовая точка, клешня 90°
POINT1_REAL_DEG = [108, 95, 87, 95, 90]     # Клетка h6, до захвата клешня 90°
POINT2_REAL_DEG = [79, 85, 68, 102, 128]    # Клетка f4, после захвата клешня 128°


# Преобразование в URDF координаты (градусы) для всех 5 серво
HOME_URDF_DEG = [real_to_urdf(HOME_REAL_DEG[i], i+1) for i in range(5)]
POINT1_URDF_DEG = [real_to_urdf(POINT1_REAL_DEG[i], i+1) for i in range(5)]
POINT2_URDF_DEG = [real_to_urdf(POINT2_REAL_DEG[i], i+1) for i in range(5)]

# Радианы для URDF (первые 4 используются в симуляции)
HOME_URDF_RAD = [math.radians(a) for a in HOME_URDF_DEG[:4]]
POINT1_URDF_RAD = [math.radians(a) for a in POINT1_URDF_DEG[:4]]
POINT2_URDF_RAD = [math.radians(a) for a in POINT2_URDF_DEG[:4]]

# Нормализованные значения (для LSTM, используются только первые 4)
HOME_NORM = [real_to_norm(HOME_REAL_DEG[i], i+1) for i in range(4)]
POINT1_NORM = [real_to_norm(POINT1_REAL_DEG[i], i+1) for i in range(4)]
POINT2_NORM = [real_to_norm(POINT2_REAL_DEG[i], i+1) for i in range(4)]

print(f"\n📍 ПОЗИЦИИ РОБОТА (реальные углы):")
print(f"   Home:     {HOME_REAL_DEG}")
print(f"   Point1:   {POINT1_REAL_DEG}")
print(f"   Point2:   {POINT2_REAL_DEG}")

# ============================================================================
# 5. КЛАСС УПРАВЛЕНИЯ РЕАЛЬНЫМ РОБОТОМ (адаптирован для 5 серво)
# ============================================================================
class RealRobotController:
    def __init__(self, port='COM3', baudrate=115200, timeout=1.0):
        try:
            self.ser = serial.Serial(port, baudrate, timeout=timeout)
            time.sleep(5)
            self.ser.reset_input_buffer() 
            self.current_angles = HOME_REAL_DEG.copy()  # теперь 5 элементов
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
        """Принимает список из 5 углов, отправляет команду SET_ANG с 5 значениями"""
        if not self.is_connected:
            return False
        
        cmd = f'SET_ANG,{real_angles_deg[0]:.2f},{real_angles_deg[1]:.2f},{real_angles_deg[2]:.2f},{real_angles_deg[3]:.2f},{real_angles_deg[4]:.2f}'
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
        """Учитывает все 5 сервоприводов"""
        max_diff = max(abs(new_angles[i] - old_angles[i]) for i in range(5))
        delay_time = max(max_diff / TRAJECTORY_CONFIG['max_speed'], TRAJECTORY_CONFIG['min_delay'])
        return delay_time

# ============================================================================
# 6. ФУНКЦИИ ПРЕДСКАЗАНИЯ LSTM (с выбором модели)
# ============================================================================
def predict_next_step(current_norm, target_norm, progress, model):
    """
    Предсказывает следующий шаг с использованием переданной модели LSTM.
    Если model is None, возвращает текущие углы (без изменений).
    """
    if model is None:
        return current_norm.copy()
    
    cur = np.ravel(current_norm)
    tgt = np.ravel(target_norm)
    input_vec = np.concatenate([cur, tgt, [progress]]).astype(np.float32)
    input_tensor = torch.tensor(input_vec).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(input_tensor)
        next_norm = prediction.detach().cpu().numpy()[0]
    
    return next_norm

# ⭐ ФУНКЦИЯ: Blending (наведение на цель) - без изменений
def apply_blending(lstm_pred, target_norm, step, total_steps, start_step, power=3.0):
    """
    Смешивает предсказание LSTM с целевой точкой.
    Параметр power > 1 обеспечивает замедление в конце:
    - power=1: линейное (равномерное)
    - power=2: квадратичное (быстрый старт, плавный конец)
    - power=3: ещё более плавный конец
    """
    if step < start_step:
        return lstm_pred
    
    # линейная доля от начала blending до конца (0..1)
    t = (step - start_step) / (total_steps - start_step)
    
    # нелинейное преобразование: быстрое начало, медленный конец
    blend = 1 - (1 - t) ** power
    
    blended = lstm_pred * (1 - blend) + target_norm * blend
    return blended

# ============================================================================
# 7. ЗАПУСК PYBULLET (без изменений, кроме использования только первых 4 углов)
# ============================================================================
robot_id = None
physicsClient = None

if MODE == 'SIMULATION':
    print("\n" + "=" * 80)
    print("🚀 ЗАПУСК PYBULLET СИМУЛЯЦИИ")
    print("=" * 80)
    
    physicsClient = p.connect(p.GUI, options='--renderDevice=egl --width=1024 --height=768')
    p.setGravity(0, 0, -9.81)
    p.setPhysicsEngineParameter(numSolverIterations=100)
    
    p.resetDebugVisualizerCamera(
        cameraDistance=0.7,
        cameraYaw=90,
        cameraPitch=-10,
        cameraTargetPosition=[0.075, 0, 0.15]
    )
    
    p.addUserDebugText("X", [0.1, 0, 0], [1,0,0])
    p.addUserDebugText("Y", [0, 0.1, 0], [0,1,0])
    p.addUserDebugText("Z", [0, 0, 0.1], [0,0,1])
    
    robot_id = p.loadURDF("5dof_manipulator_120326.urdf", useFixedBase=True)
    num_joints = p.getNumJoints(robot_id)
    print(f"   ✅ Найдено суставов: {num_joints}")
    
    # ... (создание доски, маркеров и т.д. без изменений, используем первые 4 угла)
    print("\n🏁 Создание шахматной доски...")
    square_size = 0.032
    board_size = 8
    board_center = [0.168, 0, -0.02]
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
    
    # Маркеры для точек (используем только первые 4 угла)
    print("\n🔵 Размещение маркеров...")
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    
    # POINT1 (красный)
    for i in range(4):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                               targetPosition=POINT1_URDF_RAD[i], force=500)
    for _ in range(300):
        p.stepSimulation()
        time.sleep(1/240)
    target_pos = p.getLinkState(robot_id, num_joints-1)[4]
    visual_sphere_red = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.0075,
        rgbaColor=[1, 0, 0, 1]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_sphere_red, basePosition=target_pos)
    
    # POINT2 (жёлтый)
    for i in range(4):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                               targetPosition=POINT2_URDF_RAD[i], force=500)
    for _ in range(300):
        p.stepSimulation()
        time.sleep(1/240)
    target_pos2 = p.getLinkState(robot_id, num_joints-1)[4]
    visual_sphere_yellow = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.0075,
        rgbaColor=[1, 1, 0, 1]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_sphere_yellow, basePosition=target_pos2)
    
    # HOME (синий)
    for i in range(4):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                               targetPosition=HOME_URDF_RAD[i], force=500)
    for _ in range(300):
        p.stepSimulation()
        time.sleep(1/240)
    home_pos = p.getLinkState(robot_id, num_joints-1)[4]
    visual_sphere_blue = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=0.0075,
        rgbaColor=[0, 0, 1, 1]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual_sphere_blue, basePosition=home_pos)
    
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    print("   ✅ Маркеры размещены")

# ============================================================================
# 8. ПОДКЛЮЧЕНИЕ К РЕАЛЬНОМУ РОБОТУ
# ============================================================================
robot = None

if MODE == 'REAL_ROBOT':
    print("\n" + "=" * 80)
    print("🔌 ПОДКЛЮЧЕНИЕ К РЕАЛЬНОМУ РОБОТУ")
    print("=" * 80)
    
    robot = RealRobotController(
        port=REAL_ROBOT_CONFIG['port'],
        baudrate=REAL_ROBOT_CONFIG['baudrate'],
        timeout=REAL_ROBOT_CONFIG['timeout']
    )
    
    if not robot.is_connected:
        print("   ❌ Не удалось подключиться к роботу!")
        input("\n👉 Нажмите Enter для выхода...")
        exit()
    
    if not robot.ping():
        print("   ❌ Нет ответа от Arduino!")
        input("\n👉 Нажмите Enter для выхода...")
        exit()
    print("   ✅ Связь с Arduino установлена")
    
    print("   🏠 Переход в домашнюю позицию...")
    robot.go_home()
    time.sleep(2)
    print("   ✅ Домашняя позиция установлена")

# ============================================================================
# 9. ПОЗИЦИЯ 1: URDF DEFAULT (только симуляция)
# ============================================================================
if MODE == 'SIMULATION':
    print(f"\n{'='*60}")
    print("📍 ПОЗИЦИЯ 1: URDF DEFAULT [0°, 0°, 0°, 0°]")
    print(f"{'='*60}")
    
    URDF_DEFAULT_RAD = [0.0, 0.0, 0.0, 0.0]
    
    for i in range(4):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                               targetPosition=URDF_DEFAULT_RAD[i], force=500)
    
    for _ in range(500):
        p.stepSimulation()
        time.sleep(1/240)
    
    state = p.getLinkState(robot_id, num_joints-1)
    print(f"   ✅ Позиция установлена: {state[4]}")
    
    time.sleep(2)

# ============================================================================
# 10. ПОЗИЦИЯ 2: HOME POSITION
# ============================================================================
print(f"\n{'='*60}")
print("📍 ПОЗИЦИЯ 2: HOME POSITION (Базовая точка)")
print(f"{'='*60}")
print(f"   Реальные углы: {HOME_REAL_DEG}")

if MODE == 'SIMULATION':
    for i in range(4):
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                               targetPosition=HOME_URDF_RAD[i], force=500)
    for _ in range(500):
        p.stepSimulation()
        time.sleep(1/240)
    
    state = p.getLinkState(robot_id, num_joints-1)
    print(f"   ✅ Позиция установлена: {state[4]}")

elif MODE == 'REAL_ROBOT':
    print("   ✅ Робот уже в домашней позиции")

time.sleep(2)

# ============================================================================
# 11. ПОСЛЕДОВАТЕЛЬНОСТЬ ПЕРЕМЕЩЕНИЙ С ПАУЗАМИ (LSTM + BLENDING)
# ============================================================================

NUM_STEPS = TRAJECTORY_CONFIG['num_steps']
SIM_STEPS = TRAJECTORY_CONFIG['sim_steps']
BLEND_START = BLEND_CONFIG['start_step'] if BLEND_CONFIG['enabled'] else NUM_STEPS

# --- Общие переменные для обоих режимов ---
early_stop = False
stable_counter = 0
trajectory_angles = []   # для реального робота (будет переопределено позже)
frame_count = 0          # для симуляции

# --- Параметры досрочного завершения (только для реального робота) ---
if MODE == 'REAL_ROBOT':
    TOLERANCE_DEG = 0.1
    STABLE_STEPS_REQUIRED = 1
    stable_counter = 0
    early_stop = False
    trajectory_angles = []

# --- ЗАПИСЬ ВИДЕО (только для симуляции) ---
# (закомментировано)

# ----------------------------------------------------------------------------
# Функция выполнения одного перемещения (адаптирована для 5 серво, с поддержкой загруженной траектории и выбором модели)
# ----------------------------------------------------------------------------
def run_trajectory(start_real, target_real, description, precomputed_trajectory=None, model=None):
    """
    start_real, target_real — списки из 5 реальных углов (градусы).
    Если precomputed_trajectory передан (список списков [шаг][5 углов]),
    то он используется вместо генерации (LSTM не вызывается).
    model — модель LSTM для генерации (если не передана и precomputed_trajectory is None, генерация невозможна)
    """
    global early_stop, stable_counter, trajectory_angles, frame_count

    print(f"\n{'='*60}")
    print(f"📍 {description}")
    print(f"{'='*60}")
    print(f"   Стартовые углы (реальные): {[round(a,1) for a in start_real]}")
    print(f"   Целевые углы (реальные):   {[round(a,1) for a in target_real]}")

    # Если передана готовая траектория, используем её
    if precomputed_trajectory is not None:
        print(f"   Используется загруженная траектория ({len(precomputed_trajectory)} шагов)")
        trajectory_angles = precomputed_trajectory.copy()

        if MODE == 'SIMULATION':
            # Проигрываем загруженную траекторию в симуляции
            for step_idx, ang in enumerate(trajectory_angles):
                # Преобразуем реальные углы в радианы для первых 4 серво
                angles_urdf = [real_to_urdf(ang[i], i+1) for i in range(4)]
                angles_rad = [math.radians(a) for a in angles_urdf]
                
                # Отправляем команды в PyBullet
                for i in range(4):
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                                        targetPosition=angles_rad[i], force=2000)
                for _ in range(SIM_STEPS):
                    p.stepSimulation()
                    time.sleep(1/200)
                
                # Добавляем точку траектории (жёлтую)
                state = p.getLinkState(robot_id, num_joints-1)[4]
                p.addUserDebugPoints([state], [[1, 1, 0]], 3)

                # Вывод прогресса (каждые 10 шагов)
                if step_idx % 10 == 0 or step_idx == len(trajectory_angles)-1:
                    pos = p.getLinkState(robot_id, num_joints-1)[4]
                    print(f"   Шаг {step_idx:3d}/{len(trajectory_angles)-1}: позиция {pos}")

            # После завершения цикла выводим статистику
            final_real = trajectory_angles[-1]
            print(f"\n📊 СРАВНЕНИЕ (реальные углы):")
            print(f"   Финальные: {[round(v,1) for v in final_real]}")
            print(f"   Целевые:   {[round(v,1) for v in target_real]}")
            
            return len(trajectory_angles)  # Завершаем функцию, так как траектория уже проиграна
        
    else:
        # Генерация траектории через LSTM
        if model is None:
            print("   ❌ Ошибка: не передана модель LSTM для генерации траектории.")
            return 0
        
        # Нормализованные значения для первых 4 серво
        current_norm = [real_to_norm(start_real[i], i+1) for i in range(4)]
        target_norm = [real_to_norm(target_real[i], i+1) for i in range(4)]

        # Сброс счётчиков для реального робота
        if MODE == 'REAL_ROBOT':
            stable_counter = 0
            early_stop = False
            trajectory_angles = []

        # Пятый угол (клешня) остаётся неизменным на протяжении всей траектории
        gripper_angle = start_real[4]

        for step in range(NUM_STEPS):
            progress = step / (NUM_STEPS - 1)
            lstm_pred = predict_next_step(current_norm, target_norm, progress, model)
            angles_norm_4 = apply_blending(lstm_pred, np.array(target_norm), step, NUM_STEPS, BLEND_START)

            # Преобразуем первые 4 в реальные углы
            angles_real_4 = [norm_to_real(angles_norm_4[i], i+1) for i in range(4)]
            # Полный вектор углов (5 элементов) с неизменной клешнёй
            angles_real = angles_real_4 + [gripper_angle]

            # Для симуляции нужны только первые 4 в радианах
            angles_urdf = [real_to_urdf(angles_real[i], i+1) for i in range(4)]
            angles_rad = [math.radians(a) for a in angles_urdf]

            # ----- РЕАЛЬНЫЙ РОБОТ: накопление и проверка цели -----
            if MODE == 'REAL_ROBOT':
                trajectory_angles.append(angles_real.copy())
                if not early_stop:
                    diff = [abs(angles_real[i] - target_real[i]) for i in range(5)]
                    if all(d <= TOLERANCE_DEG for d in diff):
                        stable_counter += 1
                    else:
                        stable_counter = 0
                    if stable_counter >= STABLE_STEPS_REQUIRED:
                        print(f"\n✅ Цель достигнута на шаге {step}")
                        early_stop = True

            # ----- СИМУЛЯЦИЯ: отправка команд и запись видео -----
            if MODE == 'SIMULATION':
                for i in range(4):
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL,
                                           targetPosition=angles_rad[i], force=2000)
                for _ in range(SIM_STEPS):
                    p.stepSimulation()
                    time.sleep(1/240)
                state = p.getLinkState(robot_id, num_joints-1)[4]
                p.addUserDebugPoints([state], [[1, 1, 0]], 3)

            # ----- Вывод прогресса -----
            if step % 10 == 0 or step >= BLEND_START:
                blend_pct = 0 if step < BLEND_START else round((step - BLEND_START) / (NUM_STEPS - BLEND_START) * 100)
                if MODE == 'SIMULATION':
                    pos = p.getLinkState(robot_id, num_joints-1)[4]
                    print(f"   Шаг {step:3d}/{NUM_STEPS}: blend={blend_pct:3d}% | позиция {pos}")
                else:
                    print(f"   Шаг {step:3d}/{NUM_STEPS}: blend={blend_pct:3d}% | углы {[round(a,2) for a in angles_real]}")

            current_norm = angles_norm_4.copy()
            if early_stop:
                break

    # ----- После генерации (или загрузки): отправка на реального робота -----
    if MODE == 'REAL_ROBOT' and len(trajectory_angles) > 0:
        print(f"\n🔄 Отправка траектории ({len(trajectory_angles)} шагов)...")
        prev = trajectory_angles[0]
        for idx in range(1, len(trajectory_angles)):
            ang = trajectory_angles[idx]
            robot.set_servo_angles(ang)
            delay = robot.calculate_delay(prev, ang)
            time.sleep(delay)
            prev = ang.copy()
            if idx % 10 == 0 or idx == len(trajectory_angles)-1:
                print(f"   Отправлен шаг {idx:3d}/{len(trajectory_angles)-1}: углы {[round(a,1) for a in ang]}")
        print("   ✅ Перемещение завершено")

    # ----- Финальная статистика -----
    final_real = trajectory_angles[-1] if MODE == 'REAL_ROBOT' else (angles_real if 'angles_real' in locals() else start_real)
    print(f"\n📊 СРАВНЕНИЕ (реальные углы):")
    print(f"   Финальные: {[round(v,1) for v in final_real]}")
    print(f"   Целевые:   {[round(v,1) for v in target_real]}")

    if MODE == 'SIMULATION' and 'state' in locals():
        print(f"   Финальная позиция концевика: {state}")

    return len(trajectory_angles) if MODE == 'REAL_ROBOT' else (step+1 if 'step' in locals() else 0)

# ----------------------------------------------------------------------------
# Выполнение всех четырёх перемещений с паузами и управлением клешнёй
# ----------------------------------------------------------------------------

if LOAD_TRAJECTORIES:
    try:
        traj_array = np.load(TRAJECTORY_FILE, allow_pickle=True)
        loaded_trajectories = traj_array.tolist()  # обратно в список списков
        print(f"\n📂 Загружено {len(loaded_trajectories)} траекторий из {TRAJECTORY_FILE}")
        if len(loaded_trajectories) != 4:
            print(f"   ⚠️ Ожидалось 4 траектории, загружено {len(loaded_trajectories)}. Продолжаем с имеющимися.")
    except Exception as e:
        print(f"   ❌ Ошибка загрузки файла: {e}")
        exit()        
        


    # Исполняем траектории по порядку
    # 1. HOME → POINT1
    run_trajectory(HOME_REAL_DEG, POINT1_REAL_DEG, "HOME → POINT1 (загружено)", precomputed_trajectory=loaded_trajectories[0])

    # Захват
    if MODE == 'REAL_ROBOT':
        print("\n⏸️ Пауза 2 секунды перед захватом...")
        time.sleep(2)
        new_angles = robot.current_angles.copy()
        new_angles[4] = 128.0
        prev_angles = robot.current_angles.copy()
        robot.set_servo_angles(new_angles)
        delay = robot.calculate_delay(prev_angles, new_angles)
        time.sleep(delay)
        print("✅ Захват выполнен, угол клешни 128°")
    else:
        print("\n⏸️ (Симуляция) Захват клешни не моделируется")
        time.sleep(0)

    print("\n⏸️ Пауза 2 секунд...")
    time.sleep(0)

    # 2. POINT1 → HOME (с изменённым углом клешни)
    point1_after_grasp = POINT1_REAL_DEG.copy()
    point1_after_grasp[4] = 128.0
    run_trajectory(point1_after_grasp, HOME_REAL_DEG, "POINT1 → HOME (загружено)", precomputed_trajectory=loaded_trajectories[1])

    print("\n⏸️ Пауза 2 секунд...")
    time.sleep(0)

    # 3. HOME → POINT2 (стартовые углы берём из текущего состояния робота)
    if MODE == 'REAL_ROBOT':
        run_trajectory(robot.current_angles, POINT2_REAL_DEG, "HOME → POINT2 (загружено)", precomputed_trajectory=loaded_trajectories[2])
    else:
        run_trajectory(HOME_REAL_DEG, POINT2_REAL_DEG, "HOME → POINT2 (загружено)", precomputed_trajectory=loaded_trajectories[2])

    # Отпускание
    if MODE == 'REAL_ROBOT':
        print("\n⏸️ Пауза 2 секунды перед отпусканием...")
        time.sleep(2)
        new_angles = robot.current_angles.copy()
        new_angles[4] = 90.0
        prev_angles = robot.current_angles.copy()
        robot.set_servo_angles(new_angles)
        delay = robot.calculate_delay(prev_angles, new_angles)
        time.sleep(delay)
        print("✅ Отпускание выполнено, угол клешни 90°")
    else:
        print("\n⏸️ (Симуляция) Отпускание клешни не моделируется")
        time.sleep(0)

    print("\n⏸️ Пауза 2 секунд...")
    time.sleep(0)

    # 4. POINT2 → HOME
    point2_after_release = POINT2_REAL_DEG.copy()
    point2_after_release[4] = 90.0
    if MODE == 'REAL_ROBOT':
        run_trajectory(robot.current_angles, HOME_REAL_DEG, "POINT2 → HOME (загружено)", precomputed_trajectory=loaded_trajectories[3])
    else:
        run_trajectory(point2_after_release, HOME_REAL_DEG, "POINT2 → HOME (загружено)", precomputed_trajectory=loaded_trajectories[3])

else:
    # Режим генерации
    if not LSTM_AVAILABLE:
        print("   ⚠️ LSTM недоступна, генерация траекторий невозможна.")
        exit()

    all_trajectories = []  # список для сохранения всех четырёх траекторий

    # 1. HOME → POINT1 (используем модель point)
    run_trajectory(HOME_REAL_DEG, POINT1_REAL_DEG, "HOME → POINT1", model=lstm_model_point)
    all_trajectories.append(trajectory_angles.copy())

    # Захват
    if MODE == 'REAL_ROBOT':
        print("\n⏸️ Пауза 2 секунды перед захватом...")
        time.sleep(2)
        new_angles = robot.current_angles.copy()
        new_angles[4] = 128.0
        prev_angles = robot.current_angles.copy()
        robot.set_servo_angles(new_angles)
        delay = robot.calculate_delay(prev_angles, new_angles)
        time.sleep(delay)
        print("✅ Захват выполнен, угол клешни 128°")
    else:
        print("\n⏸️ (Симуляция) Захват клешни не моделируется")
        time.sleep(2)

    print("\n⏸️ Пауза 10 секунд...")
    time.sleep(2)

    # 2. POINT1 → HOME (используем модель base)
    point1_after_grasp = POINT1_REAL_DEG.copy()
    point1_after_grasp[4] = 128.0
    run_trajectory(point1_after_grasp, HOME_REAL_DEG, "POINT1 → HOME", model=lstm_model_base)
    all_trajectories.append(trajectory_angles.copy())

    print("\n⏸️ Пауза 10 секунд...")
    time.sleep(2)

    # 3. HOME → POINT2 (используем модель point)
    if MODE == 'REAL_ROBOT':
        run_trajectory(robot.current_angles, POINT2_REAL_DEG, "HOME → POINT2", model=lstm_model_point)
    else:
        run_trajectory(HOME_REAL_DEG, POINT2_REAL_DEG, "HOME → POINT2", model=lstm_model_point)
    all_trajectories.append(trajectory_angles.copy())

    # Отпускание
    if MODE == 'REAL_ROBOT':
        print("\n⏸️ Пауза 2 секунды перед отпусканием...")
        time.sleep(2)
        new_angles = robot.current_angles.copy()
        new_angles[4] = 90.0
        prev_angles = robot.current_angles.copy()
        robot.set_servo_angles(new_angles)
        delay = robot.calculate_delay(prev_angles, new_angles)
        time.sleep(delay)
        print("✅ Отпускание выполнено, угол клешни 90°")
    else:
        print("\n⏸️ (Симуляция) Отпускание клешни не моделируется")
        time.sleep(2)

    print("\n⏸️ Пауза 10 секунд...")
    time.sleep(2)

    # 4. POINT2 → HOME (используем модель base)
    point2_after_release = POINT2_REAL_DEG.copy()
    point2_after_release[4] = 90.0
    if MODE == 'REAL_ROBOT':
        run_trajectory(robot.current_angles, HOME_REAL_DEG, "POINT2 → HOME", model=lstm_model_base)
    else:
        run_trajectory(point2_after_release, HOME_REAL_DEG, "POINT2 → HOME", model=lstm_model_base)
    all_trajectories.append(trajectory_angles.copy())

    # Сохраняем все траектории, если нужно
    if SAVE_TRAJECTORIES:
        try:
            # Сохраняем как массив объектов (поддерживает разную длину)
            np.save(TRAJECTORY_FILE, np.array(all_trajectories, dtype=object), allow_pickle=True)
            print(f"\n💾 Все 4 траектории сохранены в {TRAJECTORY_FILE} (с переменной длиной)")
        except Exception as e:
            print(f"   ❌ Ошибка сохранения файла: {e}")
            
# ============================================================================
# 12. ЗАВЕРШЕНИЕ РАБОТЫ
# ============================================================================
print("\n" + "=" * 80)
print("✅ СИМУЛЯЦИЯ ЗАВЕРШЕНА")
print("=" * 80)

if MODE == 'SIMULATION':
    if LSTM_AVAILABLE:
        print(f"📊 LSTM модели:")
        if lstm_model_point:
            print(f"   POINT: {LSTM_MODEL_POINT}")
        if lstm_model_base:
            print(f"   BASE:  {LSTM_MODEL_BASE}")
        print(f"   Устройство: {device}")
    else:
        print(f"⚠️ LSTM не загружена")
    
    input("\n👉 Нажмите Enter для завершения...")
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