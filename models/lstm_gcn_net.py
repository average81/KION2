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
DECIMATION = 1
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
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    data = data - 0.5
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
    def __init__(self, files, skeleton_dir, bodies):
        self.files = files
        self.skeleton_dir = skeleton_dir
        self.bodies = bodies
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        filepath = os.path.join(self.skeleton_dir, self.files[idx])
        data = parse_skeleton(filepath)
        if data is None: return self[0]
        data = normalize_skeleton(data)
        #если кадров в файле больше 60 * DECIMATION, то берем случайно 60 * DECIMATION последовательных
        if len(data)>60 * DECIMATION:
            start = random.randint(0,len(data)-60 * DECIMATION + 1)
            data = data[start:start + 60 * DECIMATION]
        data = interpolate_frames(data, target=60 * DECIMATION)
        # Прореживаем кадры до 60
        data = data[::DECIMATION]

        data = data[:,:self.bodies,...]
        # Дублируем, если тел меньше, чем self.bodies
        if data.ndim == 3:
            data = np.expand_dims(data, axis=1)  # Убедимся, что ось M есть
        if data.shape[1] < self.bodies:
            body_to_duplicate = data[:, 0:1, :, :]  # (T, 1, V, C)
            num_to_duplicate = self.bodies - data.shape[1]
            duplicates = np.tile(body_to_duplicate, (1, num_to_duplicate, 1, 1))  # (T, num_to_duplicate, V, C)
            data = np.concatenate([data, duplicates], axis=1)
        
        # Перемещаем ось тел внутрь признаков: (T, 2, C, V) → (T, C*2, V)
        #data = data.transpose(0, 2, 1, 3)  # (T, C, 2, V)
        #data = data.reshape(data.shape[0], data.shape[1] * data.shape[2], data.shape[3])  # (T, C*2, V)
        tensor = torch.FloatTensor(data)  # (T, C*2, V)
        label = extract_label(self.files[idx])
        return tensor, label

class LSTMSkeletonNet(nn.Module):
    def __init__(self, num_classes=60, input_size=75,bodies = 2, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bodies = bodies
        self.mask_threshold = 1e-3

        # Добавляем 1D-свёртку на вход
        self.conv1d = nn.Conv1d(
            in_channels=input_size * bodies,  # (C*V) — признаки по всем суставам
            out_channels=input_size * bodies,  # Сохраняем размерность
            kernel_size=3,
            padding=1
        )
        self.bn = nn.BatchNorm1d(input_size * bodies)
        self.relu = nn.ReLU()
        # LSTM принимает (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size * bodies,     # 25 суставов × 3 координаты
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
        # Вход: (N, T, M, C, V) → хотим (N, T, C*V)
        N, T, M, C, V = x.shape

        # Перегруппируем признаки: (N, T, M, C, V) -> (N, T, M, C*V)
        x = x.contiguous().view(N, T, M, -1)  # (N, T, M, 75)

        # Транспонируем для применения Conv1d: (N, T, M, 75) -> (N*M, 75, T)
        xm = x.view(N, T, -1).transpose(1, 2)  # (N*M, 75, T)

        # Применяем 1D-свёртку + BN + ReLU
        xm = self.conv1d(xm)  # (N, 150, T)
        xm = self.bn(xm)      # (N, 150, T)
        xm = self.relu(xm)

        # Возвращаем в форму (N, T, 150)
        xm = xm.transpose(1, 2)

        # LSTM: вход (T, N*M, input_size), выход (T, N*M, hidden_size)
        lstm_out, (hidden, _) = self.lstm(xm)  # hidden: (num_layers, N*M, hidden_size)

        # Берём последний слой скрытого состояния: (N*M, hidden_size)
        h_last = hidden[-1]  # (N*M, hidden_size)

        return self.classifier(h_last)

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

        train_dataset = SkeletonDataset(train_files, skeleton_dir, self.bodies)
        val_dataset = SkeletonDataset(val_files, skeleton_dir, self.bodies)

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

            # Логирование дополнительных метрик при валидации
            if (epoch + 1) % 5 == 0:  # Каждые 5 эпох для экономии времени
                self.eval()
                all_labels = []
                all_preds = []
                with torch.no_grad():
                    for data, labels in val_loader:
                        data, labels = data.to(device), labels.to(device)
                        outputs = self(data)
                        _, predicted = outputs.max(1)
                        all_labels.extend(labels.cpu().numpy())
                        all_preds.extend(predicted.cpu().numpy())
                
                from sklearn.metrics import precision_recall_fscore_support
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
                
                writer.add_scalar('Precision/val', precision, epoch+1)
                writer.add_scalar('Recall/val', recall, epoch+1)
                writer.add_scalar('F1-Score/val', f1, epoch+1)

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Loss: {train_loss/len(train_loader):.3f} | "
                  f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")

        writer.close()
        print(f"\n✅ Тренировка завершена! Лучшая точность: {best_acc:.2f}%")
        print(f"📊 TensorBoard: %tensorboard --logdir runs")

        # Вычисление дополнительных метрик на валидационной выборке
        self.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = self(data)
                _, predicted = outputs.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_preds, target_names=CLASSES, digits=4)
        print("\n📋 Полный отчёт по классификации на валидационной выборке:")
        print(report)
        
        # Дополнительно выведем средние метрики
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        print(f"\n📊 Средние метрики (macro avg):\n   Precision: {precision:.4f}\n   Recall:    {recall:.4f}\n   F1-Score:  {f1:.4f}")

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

            frames = {}
            for pose in sorted_poses:
                frame_idx = getattr(pose, 'frame_idx', 0)
                person_id = getattr(pose, 'id', 0)

                if frame_idx not in frames:
                    frames[frame_idx] = {}

                keypoints_3d = []
                for joint_idx in range(25):
                    kpt = pose.keypoints[joint_idx]
                    keypoints_3d.append([kpt[0], kpt[1], 0.0])

                frames[frame_idx][person_id] = keypoints_3d

            # Преобразуем в массив (T, M, V, C)
            sorted_frames = sorted(frames.items())
            max_persons = max(len(frame_data) for _, frame_data in sorted_frames)
            sequence = []
            for _, frame_data in sorted_frames:
                frame_joints = [frame_data[i] for i in sorted(frame_data.keys())]
                while len(frame_joints) < max_persons:
                    frame_joints.append([[0,0,0]] * 25)
                sequence.append(frame_joints[:self.bodies])

            data = np.array(sequence, dtype=np.float32)
        else:
            raise TypeError(f"Неподдерживаемый тип данных: {type(data)}")

        # Обработка данных
        #если кадров в файле больше 60* DECIMATION, то берем случайно 60* DECIMATION последовательных
        #if len(data)>60 * DECIMATION:
        #    start = random.randint(0,len(data)-60 * DECIMATION + 1)
        #    data = data[start:start + 60 * DECIMATION]
        data = normalize_skeleton(data)
        data = interpolate_frames(data, target=60 * DECIMATION)
        # Прореживаем данные
        #data = data[::DECIMATION]
        #data = data.mean(axis=1)  # Усредняем по телам
        data = data[:,:self.bodies,...]
        if data.shape[1] < self.bodies:
            body_to_duplicate = data[:, 0:1, :, :]
            num_to_duplicate = self.bodies - data.shape[1]
            duplicates = np.tile(body_to_duplicate, (1, num_to_duplicate, 1, 1))
            data = np.concatenate([data, duplicates], axis=1)

        tensor = torch.FloatTensor(data).unsqueeze(0)  # Добавляем batch dimension


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