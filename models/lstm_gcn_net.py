# %%
# ==============================================================================
# 1. ИМПОРТЫ (исправленный datetime)
# ==============================================================================
import os, numpy as np, torch, torch.nn as nn, random
from datetime import datetime  # ✅ Класс datetime, а не модуль
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
import random

CLASSES = [
    'drink water', 'eat meal', 'brushing teeth', 'brushing hair', 'drop',
    'pickup', 'throw', 'sitting down', 'standing up', 'clapping', 'reading',
    'writing', 'tear up paper', 'wear jacket', 'taking off jacket',
    'wear a shoe', 'taking off a shoe', 'wear socks', 'taking off socks',
    'stretching arm', 'kicking', 'punching', 'kicking 2', 'punching 2',
    'falling', 'hammering', 'kicking something', 'punching 3', 'dancing',
    'kicking 3', 'writing 2', 'taking a selfie', 'checking time',
    'rub two hands together', 'walking zigzag', 'walking with irregular speed',
    'walking with heavy steps', 'arm circles', 'arm swings', 'lunge',
    'squats', 'banded squats', 'arm curls', 'prior box squats', 'pushups',
    'bench press', 'deadlift', 'jump jacks', 'rowing', 'running on treadmill',
    'situps', 'lunges', 'jump rope', 'pushup jacks', 'high knees',
    'heels down', 'side kick', 'round house kick', 'fore kick', 'side kick 2',
    'side lunge'
]
DECIMATION = 3
# %%
# ==============================================================================
# 2. ФУНКЦИИ ОБРАБОТКИ ДАННЫХ (без изменений)
# ==============================================================================
def parse_skeleton(filepath):
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    if len(lines) < 4: return None
    try:
        num_frames, num_bodies = int(lines[0]), int(lines[1])
    except: return None
    data, idx = [], 2
    for frame in range(num_frames):
        frame_data = []
        for body in range(min(num_bodies, 2)):
            if idx >= len(lines): break
            idx += 1  # служебная строка
            try:
                num_joints = int(lines[idx]); idx += 1
            except: break
            joints = []
            for _ in range(min(num_joints, 25)):
                if idx >= len(lines): break
                parts = lines[idx].split()
                if len(parts) >= 3:
                    x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                    joints.append([x, y, z] if abs(x) < 100 else [0,0,0])
                else: joints.append([0,0,0])
                idx += 1
            while len(joints) < 25: joints.append([0,0,0])
            frame_data.append(joints)
        while len(frame_data) < 2: frame_data.append([[0,0,0]]*25)
        data.append(frame_data)
    return np.array(data, dtype=np.float32)

def normalize_skeleton(data):
    if np.max(np.abs(data)) < 1e-6: return data
    hip = data[:, :, 0, :].copy()
    data = data - hip[:, :, np.newaxis, :]
    scale = np.max(np.abs(data))
    if scale > 1e-6: data = data / scale
    return data

def interpolate_frames(data, target=30):
    T, M, V, C = data.shape
    if T == target: return data
    old_t, new_t = np.linspace(0, 1, T), np.linspace(0, 1, target)
    result = np.zeros((target, M, V, C), dtype=np.float32)
    for m in range(M):
        for v in range(V):
            for c in range(C):
                f = interp1d(old_t, data[:, m, v, c], kind='linear',
                             bounds_error=False, fill_value=(data[0,m,v,c], data[-1,m,v,c]))
                result[:, m, v, c] = f(new_t)
    return result

def extract_label(filename):
    parts = filename.replace('.skeleton', '').split('A')
    if len(parts) >= 2:
        return int(parts[-1]) - 1
    return 0

# %%
# ==============================================================================
# 3. DATASET И МОДЕЛЬ
# ==============================================================================
class SkeletonDataset(Dataset):
    def __init__(self, files, skeleton_dir):
        self.files = files
        self.skeleton_dir = skeleton_dir
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        filepath = os.path.join(self.skeleton_dir, self.files[idx])
        data = parse_skeleton(filepath)
        if data is None: return self[0]
        data = normalize_skeleton(data)
        #если кадров в файле больше 30 * DECIMATION, то берем случайно 60 последовательных
        if len(data)>30 * DECIMATION:
            start = random.randint(0,len(data)-30 * DECIMATION + 1)
            data = data[start:start + 30 * DECIMATION]
        data = interpolate_frames(data, target=30 * DECIMATION)
        # Прореживаем кадры до 30
        data = data[::DECIMATION]
        # Усредняем по телам → (30, 25, 3), есть над чем поработать, добавит ошибок, если несколько тел
        data = data.mean(axis=1)  # или data[:, 0, ...] для первого тела
        tensor = torch.FloatTensor(data).permute(0, 1, 2)  # (T, C, V)
        label = extract_label(self.files[idx])
        return tensor, label

class LSTMSkeletonNet(nn.Module):
    def __init__(self, num_classes=60, input_size=75, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM принимает (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,     # 25 суставов × 3 координаты = 75
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        # Вход: (N, T, C, V) → хотим (N, T, C*V)
        N, T, C, V = x.shape
        x = x.view(N, T, -1)  # (N, 30, 75)

        # LSTM
        lstm_out, (hidden, _) = self.lstm(x)  # hidden: (num_layers, N, hidden_size)

        # Берём последнее состояние
        out = hidden[-1]  # (N, hidden_size)

        # Классификация
        return self.classifier(out)

    def train_model(self, config_path: str):
        """
        Обучает модель с параметрами из YAML-файла.

        Args:
            config_path (str): Путь к файлу конфигурации lstm_params.yml
        """
        # Загружаем параметры из YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Извлекаем параметры
        skeleton_dir = config['data']['skeleton_dir']
        batch_size = config['training']['batch_size']
        epochs = config['training']['epochs']
        lr = config['training']['learning_rate']
        weight_decay = config['training'].get('weight_decay', 0.01)
        num_workers = config['training'].get('num_workers', 8)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"🔧 Устройство: {device}")
        print(f"📊 Конфигурация: {config_path}")

        # TensorBoard
        run_name = f"lstm_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=f'runs/{run_name}')
        print(f"📈 Логи: runs/{run_name}")

        # Данные
        all_files = [f for f in os.listdir(skeleton_dir) if f.endswith('.skeleton')]
        print(f"📁 Всего файлов: {len(all_files)}")

        train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
        print(f"📊 Train: {len(train_files)}, Val: {len(val_files)}")

        train_dataset = SkeletonDataset(train_files, skeleton_dir)
        val_dataset = SkeletonDataset(val_files, skeleton_dir)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers)

        # Перемещаем модель на устройство
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"🔧 Параметры: {sum(p.numel() for p in self.parameters()):,}")
        print(f"\n🚀 Начало тренировки ({epochs} эпох)...\n")

        best_acc = 0.0
        os.makedirs('models', exist_ok=True)

        for epoch in range(epochs):
            # Тренировка
            self.train()
            train_loss, train_correct, train_total = 0, 0, 0
            for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                data, labels = data.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()
                train_total += labels.size(0)

            # Валидация
            self.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = self(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += labels.size(0)

            # Метрики
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            scheduler.step()

            # Сохранение лучшей модели
            if val_acc > best_acc:
                best_acc = val_acc
                model_path = 'models/best_lstm_model.pth'
                torch.save(self.state_dict(), model_path)
                print(f"✅ Сохранена лучшая модель: {model_path} | Val Acc: {val_acc:.2f}%")

            # Логирование в TensorBoard
            writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch+1)
            writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch+1)
            writer.add_scalar('Accuracy/train', train_acc, epoch+1)
            writer.add_scalar('Accuracy/val', val_acc, epoch+1)
            writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch+1)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {train_loss/len(train_loader):.3f} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        writer.close()
        print(f"\n✅ Тренировка завершена! Лучшая точность: {best_acc:.2f}%")
        print(f"📊 TensorBoard: %tensorboard --logdir runs")

    def predict(self, data):
        """
        Выполняет предсказание на входных данных.

        Args:
            data: Входные данные в одном из форматов:
                - Путь к .skeleton файлу
                - NumPy массив формы (T, M, V, C)
                - Torch тензор формы (1, T, C, V)
                - Объект Pose или список объектов Pose из pose_format.py

        Returns:
            dict: Словарь с результатами:
                - 'logits': сырые выходы модели
                - 'probabilities': вероятности классов
                - 'predicted_class': предсказанный класс
                - 'confidence': уверенность (максимальная вероятность)
        """
        self.eval()  # Переводим модель в режим inference
        device = next(self.parameters()).device  # Получаем устройство модели

        # Подготовка данных
        if isinstance(data, str):
            # Если строка — считаем, что это путь к файлу
            data = parse_skeleton(data)
            if data is None:
                raise ValueError("Не удалось загрузить данные из файла")
        elif isinstance(data, np.ndarray):
            # Если NumPy массив — используем как есть
            pass
        elif torch.is_tensor(data):
            # Если тензор — преобразуем в NumPy
            data = data.cpu().numpy()
        elif hasattr(data, '__class__') and data.__class__.__name__ == 'Pose':
            # Если объект Pose из pose_format.py
            from models.pose_format import JOINTS

            # Извлекаем все кадры для данного ID
            frames = {}
            current_pose = data

            # Собираем все ключевые точки по кадрам
            while True:
                frame_idx = getattr(current_pose, 'frame_idx', 0)
                person_id = getattr(current_pose, 'id', 0)

                if frame_idx not in frames:
                    frames[frame_idx] = []

                # Извлекаем координаты суставов
                keypoints_3d = []
                for joint_idx in sorted(JOINTS.keys()):
                    kpt = current_pose.keypoints[joint_idx]
                    # Предполагаем Z=0 для 2D поз
                    keypoints_3d.append([kpt[0], kpt[1], 0.0])

                frames[frame_idx].append(keypoints_3d)

                # Это упрощённый пример - в реальности нужно получить последовательность
                # из всех поз с одинаковым id через внешний источник
                break

            # Преобразуем в массив
            sorted_frames = sorted(frames.items())
            data = np.array([frame_data[0] for _, frame_data in sorted_frames], dtype=np.float32)
            data = data.reshape(-1, 1, 25, 3)  # (T, M, V, C), M=1 для одного человека
        elif isinstance(data, list) and len(data) > 0 and hasattr(data[0], '__class__') and data[0].__class__.__name__ == 'Pose':
            # Если список объектов Pose (последовательность поз одного человека)
            from models.pose_format import JOINTS

            # Сортируем по номеру кадра
            sorted_poses = sorted(data, key=lambda x: getattr(x, 'frame_idx', 0))

            sequence = []
            for pose in sorted_poses:
                keypoints_3d = []
                for joint_idx in range(25): #25 первых записей
                    kpt = pose.keypoints[joint_idx]
                    # Предполагаем Z=0 для 2D поз
                    keypoints_3d.append([kpt[0], kpt[1], 0.0])
                sequence.append(keypoints_3d)

            data = np.array(sequence, dtype=np.float32)
            data = data.reshape(-1, 1, 25, 3)  # (T, M, V, C)
        else:
            raise TypeError(f"Неподдерживаемый тип данных: {type(data)}")

        # Обработка данных
        #если кадров в файле больше 60, то берем случайно 60 последовательных
        if len(data)>30 * DECIMATION:
            start = random.randint(0,len(data)-30 * DECIMATION + 1)
            data = data[start:start + 30 * DECIMATION]
        data = normalize_skeleton(data)
        data = interpolate_frames(data, target=30 * DECIMATION)
        # Прореживаем данные
        data = data[::DECIMATION]
        data = data.mean(axis=1)  # Усредняем по телам
        tensor = torch.FloatTensor(data).unsqueeze(0)  # Добавляем batch dimension
        tensor = tensor.permute(0, 1, 2, 3)  # (1, T, C, V)

        # Предсказание
        with torch.no_grad():
            tensor = tensor.to(device)
            output = self(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

        return {
            'logits': output.cpu().numpy()[0],
            'probabilities': probabilities.cpu().numpy()[0],
            'predicted_class': predicted.item(),
            'confidence': confidence.item()
        }

    def load_weights(self, weights_path: str):
        """
        Загружает веса модели из файла.

        Args:
            weights_path (str): Путь к файлу с весами (.pth)
        """
        try:
            self.load_state_dict(torch.load(weights_path, map_location=next(self.parameters()).device))
            print(f"✅ Веса загружены из {weights_path}")
        except Exception as e:
            print(f"❌ Ошибка при загрузке весов: {e}")
            raise