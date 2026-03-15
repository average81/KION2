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
    return data + noise

def normalize_skeleton(data):
    if np.max(np.abs(data)) < 1e-6: return data
    # Заменяем NaN на нули
    data = np.nan_to_num(data, nan=0.0)
    hip = data[:, :, 0, :].copy()
    data = data - hip[:, :, np.newaxis, :]
    scale = np.max(np.abs(data)) - np.min(np.abs(data))
    if scale > 1e-6: data = (data - np.min(np.abs(data))) / scale - 0.5
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

class GCN(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix):
        super(GCN, self).__init__()
        self.adj = adj_matrix
        self.W = nn.Linear(in_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (N, T, C, V) → GCN применяется по V (суставам)
        N, T, C, V = x.size()
        x = x.contiguous().view(N * T, C, V)  # → (N*T, C, V)

        # Нормализация adjacency matrix (симметричная)
        adj = self.adj.to(x.device)
        D = torch.sum(adj, dim=1)  # Степени узлов
        D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.
        adj_norm = torch.mm(torch.mm(D_inv_sqrt, adj), D_inv_sqrt)
        #Residual connection
        x_res = x  # (N*T, C, V)
        # Применяем GCN: X' = ReLU( A @ X^T @ W )
        x = x.transpose(1, 2)  # → (N*T, V, C)
        x = torch.matmul(adj_norm, x)  # → (N*T, V, C)
        x = self.W(x)  # → (N*T, V, out_channels)
        x = x.transpose(1, 2)  # → (N*T, out_channels, V)
        x = x + x_res  # Residual
        x = x.view(N, T, -1, V)  # → (N, T, out_channels, V)
        return self.relu(x)
class GCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adj_matrix, use_temporal_conv=True):
        super().__init__()
        self.gcn = GCN(in_channels, out_channels, adj_matrix)
        if use_temporal_conv:
            self.tcn = nn.Conv2d(out_channels, out_channels, kernel_size=(3,1), padding=(1,0))
            self.relu = nn.ReLU()
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # x: (N, T, C, V)
        x = self.gcn(x)  # (N, T, C, V)
        x = x.permute(0, 2, 1, 3)  # (N, C, T, V)
        x = self.tcn(x)  # (N, C, T, V)
        x = self.relu(self.bn(x))
        x = x.permute(0, 2, 1, 3)  # (N, T, C, V)
        return x
class LSTMSkeletonNet(nn.Module):
    def __init__(self, num_classes=60, input_size=50, bodies = 2, hidden_size=256, num_layers=2, dropout=0.3, fusion='sum'):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bodies = bodies
        self.fusion = fusion  # 'sum', 'mean', 'max'

        # Вход: (N, T, M, C, V) → хотим (N, C, T, V)
        # Применяем Conv2d по (T, V) для извлечения временных паттернов
        self.temporal_conv = nn.Conv2d(
            in_channels=2,  # C (x,y координаты)
            out_channels=64,
            kernel_size=(9, 3),  # (временные окна, пространственные соседи)
            padding=(4, 1)
        )
        self.temporal_bn = nn.BatchNorm2d(64)
        self.temporal_relu = nn.ReLU()
        
        # GCN для пространственных отношений
        self.adj_matrix = self.get_skeleton_adjacency()  # Создаём матрицу
        self.gcn_stack = nn.Sequential(
            GCNBlock(64, 64, self.adj_matrix),
            GCNBlock(64, 64, self.adj_matrix),
            GCNBlock(64, 64, self.adj_matrix)
        )
        
        # Финальная обработка и классификация
        self.fusion_conv = nn.Conv2d(64, 64, kernel_size=1)
        self.fusion_bn = nn.BatchNorm2d(64)
        self.joint_weights = nn.Parameter(torch.ones(25))
        
        # LSTM для последовательной обработки
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def get_skeleton_adjacency(self):
        #Сопоставление костей и суставов для NTU-RGBD
        num_joints = 25
        adj = np.zeros((num_joints, num_joints), dtype=np.float32)

        # Основные связи (основываясь на человеческой анатомии)
        edges = [
            (0, 1), (1, 20), (20, 2), (2, 3),
            (20, 4), (4, 5), (5, 6), (6, 7),
            (20, 8), (8, 9), (9, 10), (10, 11),
            (20, 12), (12, 13), (13, 14), (14, 15),
            (20, 16), (16, 17), (17, 18), (18, 19),
            # Дополнительные (голова, шея и т.д.)
            (0, 21), (21, 22), (22, 23), (23, 24)
        ]
        for i, j in edges:
            adj[i, j] = 1
            adj[j, i] = 1
        
        # Для нескольких тел: создаем блочно-диагональную матрицу
        if self.bodies > 1:
            # Создаем матрицу смежности для каждого тела
            single_adj = adj.copy()
            # Создаем блочно-диагональную матрицу для всех тел
            adj = np.kron(np.eye(self.bodies), single_adj)
        
        return torch.tensor(adj, dtype=torch.float32)
    def forward(self, x):
        # Вход: (N, T, M, C, V)
        N, T, M, C, V = x.shape
        assert C == 2, f"Ожидалось 2 координаты (x,y), получено {C}"
        assert V == 25, f"Ожидалось 25 суставов, получено {V}"
        
        # Объединяем размерности тел и признаков: (N, T, M, C, V) → (N, T, C, V, M)
        x = x.permute(0, 1, 3, 4, 2).contiguous()  # (N, T, C, V, M)
        
        # Сохраняем информацию о взаимодействии между телами, не усредняя по M
        # Формируем признаки, объединяя данные всех тел: (N, T, C, V, M) → (N, T, C, V*M)
        x = x.view(N, T, C, -1)  # (N, T, C, V*M)
        
        # Транспонируем для Conv2d: (N, T, C, V*M) → (N, C, T, V*M)
        x = x.permute(0, 2, 1, 3).contiguous()  # (N, C, T, V*M)
        
        # Обновляем размерность V для учета объединенных данных тел
        V_effective = V * M  # Эффективное количество суставов после объединения тел
        
        # Применяем Conv2d по (T, V*M) для извлечения временных паттернов
        x = self.temporal_conv(x)  # (N, 64, T, V*M)
        x = self.temporal_bn(x)
        x = self.temporal_relu(x)
        
        # Транспонируем для GCN: (N, 64, T, V*M) → (N, T, 64, V*M)
        x = x.permute(0, 2, 1, 3).contiguous()  # (N, T, 64, V*M)
        
        # Применяем GCN для пространственных отношений
        x = self.gcn_stack(x)  # (N, T, 64, V*M)
        
        # Вернём обратно в (N, 64, T, V*M)
        x = x.permute(0, 2, 1, 3).contiguous()  # (N, 64, T, V*M)
        
        # Финальная обработка
        x = self.fusion_bn(self.fusion_conv(x))  # (N, 64, T, V*M)
        # Создаем веса для каждого тела и сустава
        joint_weights_expanded = self.joint_weights.view(1, 1, 1, V, 1).expand(-1, -1, -1, -1, M)
        joint_weights_flat = joint_weights_expanded.reshape(1, 1, 1, V * M)  # (1, 1, 1, V*M)
        x = x * joint_weights_flat  # Взвешиваем суставы
        x = x.sum(dim=-1)  # Суммируем по суставам и телам: (N, 64, T)
        
        # Подготовка для LSTM: (N, 64, T) → (N, T, 64)
        x = x.transpose(1, 2)  # (N, T, 64)
        
        # LSTM для последовательной обработки
        lstm_out, (hidden, _) = self.lstm(x)  # (N, T, hidden_size*2)
        
        # Используем последнее скрытое состояние
        h_last = hidden[-2:]  # последние два слоя: forward и backward
        h_last = torch.cat([h_last[0], h_last[1]], dim=1)  # конкатенируем
        
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
        pretrained_weights = config['training'].get('pretrained_weights', None)  # ✅ Новый параметр

        print(f"🔧 Устройство: {device}")
        print(f"📊 Конфигурация: {config_path}")
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