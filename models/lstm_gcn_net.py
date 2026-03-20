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
import json

CLASSES = [
    "drink water",
    "eat meal/snack",
    "brushing teeth",
    "brushing hair",
    "drop",
    "pickup",
    "throw",
    "sitting down",
    "standing up (from sitting position)",
    "clapping",
    "reading",
    "writing",
    "tear up paper",
    "wear jacket",
    "take off jacket",
    "wear a shoe",
    "take off a shoe",
    "wear on glasses",
    "take off glasses",
    "put on a hat/cap",
    "take off a hat/cap",
    "cheer up",
    "hand waving",
    "kicking something",
    "reach into pocket",
    "hopping (one foot jumping)",
    "jump up",
    "make a phone call/answer phone",
    "playing with phone/tablet",
    "typing on a keyboard",
    "pointing to something with finger",
    "taking a selfie",
    "check time (from watch)",
    "rub two hands together",
    "nod head/bow",
    "shake head",
    "wipe face",
    "salute",
    "put the palms together",
    "cross hands in front (say stop)",
    "sneeze/cough",
    "staggering",
    "falling",
    "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)",
    "touch neck (neckache)",
    "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person",
    "kicking other person",
    "pushing other person",
    "pat on back of other person",
    "point finger at the other person",
    "hugging other person",
    "giving something to other person",
    "touch other person's pocket",
    "handshaking",
    "walking towards each other",
    "walking apart from each other"
]
DECIMATION = 2
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
                if len(parts) >= 7:
                    x, y = float(parts[5]), float(parts[6])
                    joints.append([x, y])
                else: joints.append([0,0])
                idx += 1
            while len(joints) < 25: joints.append([0,0])
            frame_data.append(joints)
        while len(frame_data) < 2: frame_data.append([[0,0]]*25)
        data.append(frame_data)
    return np.array(data, dtype=np.float32)

def augment_skeleton(data, noise_std=0.05):
    flip_prob = 0.2
    noise = np.random.normal(0, noise_std, data.shape).astype(np.float32)
    scale = random.uniform(0.9, 1)
    data = data * scale
    # Случайное отражение (по оси X) — только если действия симметричны
    if random.random() < flip_prob:
        data[..., 0] = -data[..., 0]  # инвертируем X координату
    t = np.random.randint(1, 10)
    data = data[:-t]
    noise = noise[:-t]
    return data + noise

def normalize_skeleton(data):
    if np.max(np.abs(data)) < 1e-6: return data
    # Заменяем NaN на нули
    data = np.nan_to_num(data, nan=0.0)
    hip = data[:, :, 0, :].copy()
    data = data - hip[:, :, np.newaxis, :]
    scale = np.max(data) - np.min(data)
    if scale > 1e-6: data = (data - data.min()) / scale - 0.5
    #print(data.shape)
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
        #Аугментация
        data = augment_skeleton(data)
        #print(data.min(),data.max())
        #если кадров в файле больше 60 * DECIMATION, то берем случайно 60 * DECIMATION последовательных
        if len(data) > 60 * DECIMATION:
            id = np.linspace(0, len(data) - 1, 60 * DECIMATION).astype(int)
            data = data[id]
        else:
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

        # Меняем местами C и V: (T, M, V, C) → (T, M, C, V)
        data = data.transpose(0, 1, 3, 2)
        tensor = torch.FloatTensor(data)  # (T, C*2, V)
        label = extract_label(self.files[idx])
        return tensor, label

class LSTMSkeletonNet(nn.Module):
    def __init__(self, num_classes=60,bodies = 2, hidden_size=256, num_layers=2, dropout=0.3, fusion='attention'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bodies = bodies
        self.fusion = fusion  # 'sum', 'mean', 'max'

        # Вход: (N, T, M, C, V) → хотим (N, C, T, V)
        # Применяем Conv2d по (T, V) для извлечения временных паттернов
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * bodies,  # C (x,y координаты)
                out_channels=128,
                kernel_size=(9, 3),  # (временные окна, пространственные соседи)
                padding=(4, 1)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,  # C (x,y координаты)
                out_channels=128,
                kernel_size=(7, 1),  # (временные окна, пространственные соседи)
                padding=(3, 0)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,  # C (x,y координаты)
                out_channels=128,
                kernel_size=(3, 5),  # (временные окна, пространственные соседи)
                padding=(1, 2)
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))
        )

        # Финальная обработка и классификация
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(
                in_channels =128,
                out_channels =64,
                kernel_size=(3,1),
                padding=(1, 0)),
            nn.BatchNorm2d(64)
        )

        self.joint_weights = nn.Parameter(torch.ones(25))
        self.pre_lstm_dropout = nn.Dropout(dropout)
        self.pre_lstm_norm = nn.LayerNorm(64)  # нормализация по признакам

        # LSTM для последовательной обработки
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.LayerNorm(1)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        # Вход: (N, T, M, C, V)
        N, T, M, C, V = x.shape
        assert C == 2, f"Ожидалось 2 координаты (x,y), получено {C}"
        assert V == 25, f"Ожидалось 25 суставов, получено {V}"
        # свертка по суставами времени раздельно по телам
        x = x.permute(0, 3, 1, 4, 2).contiguous()  # → (N, C, T, V, M)
        x = x.view(N, C * M, T, V)  # (N, C*M, T, V)
        # Применяем Conv2d
        x = self.conv1(x)  # (N, 128, T, V)
        x = self.conv2(x)  # (N, 128, T, V)
        x = self.conv3(x)  # (N, 128, T, V)

        #x = x.view(N,M, 128, T//4, V)
        #x = x.permute(0, 2, 3, 1, 4).contiguous()  # (N, 64, T, M, V)
        #x = x.view(N, 128, T//4, -1)
        x = self.fusion_conv(x)  # (N, 64, T, V)

        # Создаем веса для каждого тела и сустава
        #x = x.permute(0, 2, 3, 1, 4).contiguous()  # (N, 64, T, M, V)
        # = x.view(N, 64, T, -1) # (N, 64, T, M * V)
        joint_weights_flat = self.joint_weights.view(1, 1, 1, -1).expand(-1, 64, -1, -1)  # (1, 64, 1, V)
        x = x * joint_weights_flat  # (N, 64, T, M*V)
        if self.fusion == 'sum':
            x = (x * joint_weights_flat).sum(dim=-1)  # (N, 64, T)
        elif self.fusion == 'mean':
            x = (x * joint_weights_flat).mean(dim=-1)
        elif self.fusion == 'max':
            x = (x * joint_weights_flat).max(dim=-1)[0]
        elif self.fusion == 'attention':
            # Soft attention over joints
            weights = torch.softmax(self.joint_weights.view(1, 1, 1, -1), dim=-1)
            x = (x * weights).sum(dim=-1)  # (N, 64, T)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion}")

        # Подготовка для LSTM: (N, 64, T) → (N, T, 64)
        x = x.transpose(1, 2)  # (N, T, 64)
        x = self.pre_lstm_norm(x)  # LayerNorm по признакам
        x = self.pre_lstm_dropout(x)
        # LSTM для последовательной обработки
        lstm_out, (hidden, _) = self.lstm(x)  # (N, T, hidden_size*2)
        attn_weights = torch.softmax(self.temporal_attention(lstm_out), dim=1)  # (N, T, 1)
        t_pooled = (lstm_out * attn_weights).sum(dim=1)  # (N, 2*hidden_size)
        # Используем последнее скрытое состояние
        h_last = hidden[-2:]  # последние два слоя: forward и backward
        h_last = torch.cat([h_last[0], h_last[1]], dim=1)  # конкатенируем
        x = torch.cat([h_last, t_pooled], dim=1)  # (N, 4*hidden_size)
        return self.classifier(x)

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
        pretrained_weights = config['training'].get('pretrained_weights', None)  # ✅ Новый параметр

        print(f"🔧 Устройство: {device}")
        print(f"📊 Конфигурация: {config_path}")

        # Путь для сохранения/загрузки списков файлов
        split_file = os.path.join(skeleton_dir, 'train_val_split.json')

        # Проверяем, существует ли уже разбиение
        if os.path.exists(split_file):
            print(f"🔄 Загружаем существующее разбиение из {split_file}")
            with open(split_file, 'r') as f:
                split_data = json.load(f)
                train_files = split_data['train_files']
                val_files = split_data['val_files']
            # Фильтруем только существующие файлы (на случай удаления)
            all_available = set(os.listdir(skeleton_dir))
            train_files = [f for f in train_files if f in all_available]
            val_files = [f for f in val_files if f in all_available]
            print(f"✅ Загружено: Train={len(train_files)}, Val={len(val_files)}")
        else:
            print(f"🆕 Создаём новое разбиение данных...")
            all_files = [f for f in os.listdir(skeleton_dir) if f.endswith('.skeleton')]
            print(f"📁 Всего файлов: {len(all_files)}")

            # Извлекаем метки для всех файлов
            labels = [extract_label(f) for f in all_files]

            # Стратифицированное разбиение
            train_files, val_files = train_test_split(
                all_files,
                test_size=0.2,
                random_state=42,
                stratify=labels
            )
            # Сохраняем разбиение
            split_data = {
                'train_files': train_files,
                'val_files': val_files,
                'split_date': datetime.now().isoformat(),
                'total_files': len(all_files),
                'test_size': 0.2,
                'random_state': 42
            }
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"✅ Разбиение сохранено в {split_file}")

        # Загрузка предобученных весов ДО оптимизатора
        if pretrained_weights and os.path.exists(pretrained_weights):
            print(f"🔄 Загружаем предобученные веса из: {pretrained_weights}")
            try:
                self.load_state_dict(torch.load(pretrained_weights, map_location=device))
                print("✅ Веса успешно загружены")
            except Exception as e:
                print(f"❌ Ошибка при загрузке весов: {e}")
                raise
        elif pretrained_weights:
            raise FileNotFoundError(f"Файл с весами не найден: {pretrained_weights}")
        self.to(device)
        scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
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

        # Оптимизация DataLoader для CUDA
        pin_memory = device.type == 'cuda'
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory,
                                persistent_workers=(num_workers > 0))


        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
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
                # Используем autocast для автоматического смешанного precision
                data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
                    outputs = self(data)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                
                # Используем scaler для градиентов в mixed precision
                if device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
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
                with torch.cuda.amp.autocast(enabled=device.type=='cuda'):
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
                keypoints_2d = []
                for joint_idx in sorted(JOINTS.keys()):
                    kpt = current_pose.keypoints[joint_idx]
                    # Предполагаем Z=0 для 2D поз
                    keypoints_2d.append([kpt[5], kpt[6]])

                frames[frame_idx].append(keypoints_2d)

                # Это упрощённый пример - в реальности нужно получить последовательность
                # из всех поз с одинаковым id через внешний источник
                break

            # Преобразуем в массив
            sorted_frames = sorted(frames.items())
            data = np.array([frame_data[0] for _, frame_data in sorted_frames], dtype=np.float32)
            data = data.reshape(-1, 1, 25, 2)  # (T, M, V, C), M=1 для одного человека
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

                keypoints_2d = []
                for joint_idx in range(25):
                    kpt = pose.keypoints[joint_idx]
                    keypoints_2d.append([kpt[0], kpt[1]])

                frames[frame_idx][person_id] = keypoints_2d

            # Преобразуем в массив (T, M, V, C)
            sorted_frames = sorted(frames.items())
            max_persons = max(len(frame_data) for _, frame_data in sorted_frames)
            sequence = []
            for _, frame_data in sorted_frames:
                frame_joints = [frame_data[i] for i in sorted(frame_data.keys())]
                while len(frame_joints) < max_persons:
                    frame_joints.append([[0,0]] * 25)
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
        data = data[::DECIMATION]  # Прореживаем до 60 кадров
        data = data[:, :self.bodies, ...]

        if data.shape[1] < self.bodies:
            body_to_duplicate = data[:, 0:1, :, :]
            num_to_duplicate = self.bodies - data.shape[1]
            duplicates = np.tile(body_to_duplicate, (1, num_to_duplicate, 1, 1))
            data = np.concatenate([data, duplicates], axis=1)
        # Транспонируем: (T, M, V, C) -> (T, M, C, V)
        data = data.transpose(0, 1, 3, 2)
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