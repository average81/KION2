# stgcn_wrapper.py
import os
import json
import torch
# Та же реализация, что и при обучении (config: model: net.st_gcn.Model)
from models.stgcn.net.st_gcn import Model
import numpy as np
from torch.utils.data import Dataset

DECIMATION = 1

class SkeletonDataset(Dataset):
    def __init__(self, files, skeleton_dir, num_joints, num_person):
        self.files = files
        self.skeleton_dir = skeleton_dir
        self.num_joints = num_joints
        self.num_person = num_person

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filepath = os.path.join(self.skeleton_dir, self.files[idx])
        data = self.parse_skeleton(filepath)
        if data is None:
            return self[0]
        data = self.normalize_skeleton(data)
        data = self.augment_skeleton(data)

        if len(data) > 120 * DECIMATION:
            id = np.linspace(0, len(data) - 1, 120 * DECIMATION).astype(int)
            data = data[id]
        else:
            data = self.interpolate_frames(data, target=120 * DECIMATION)
        # Прореживаем
        data = data[::DECIMATION]

        # Обрезаем или дополняем количество суставов и тел
        data = data[:, :self.num_person, :self.num_joints, :]
        if data.shape[1] < self.num_person:
            # Дублируем первого человека
            body_to_duplicate = data[:, 0:1, :, :]
            num_to_duplicate = self.num_person - data.shape[1]
            duplicates = np.tile(body_to_duplicate, (1, num_to_duplicate, 1, 1))
            data = np.concatenate([data, duplicates], axis=1)

        # Транспонируем: (T, M, V, C) -> (C, T, V, M)
        data = data.transpose(3, 0, 2, 1)
        tensor = torch.FloatTensor(data)
        label = self.extract_label(self.files[idx])
        return tensor, label

    def parse_skeleton(self, filepath):
        with open(filepath, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]
        if len(lines) < 4:
            return None
        try:
            num_frames, num_bodies = int(lines[0]), int(lines[1])
        except:
            return None
        data, idx = [], 2
        for frame in range(num_frames):
            frame_data = []
            for body in range(min(num_bodies, 2)):
                if idx >= len(lines):
                    break
                idx += 1  # служебная строка
                try:
                    num_joints = int(lines[idx]); idx += 1
                except:
                    break
                joints = []
                for _ in range(min(num_joints, 25)):
                    if idx >= len(lines):
                        break
                    parts = lines[idx].split()
                    if len(parts) >= 7:
                        x, y = float(parts[5]), float(parts[6])
                        joints.append([x, y])
                    else:
                        joints.append([0, 0])
                    idx += 1
                while len(joints) < 25:
                    joints.append([0, 0])
                frame_data.append(joints)
            while len(frame_data) < 2:
                frame_data.append([[0, 0]] * 25)
            data.append(frame_data)
        
        # Конвертация из NTU-RGB+D (25 суставов) в OpenPose (18 суставов)
        ntu_to_openpose = [
            3,   # 0: nose        <- head
            2,   # 1: neck        <- neck
            8,   # 2: r_shoulder  <- right_shoulder
            9,   # 3: r_elbow     <- right_elbow
            10,  # 4: r_wrist     <- right_wrist
            4,   # 5: l_shoulder  <- left_shoulder
            5,   # 6: l_elbow     <- left_elbow
            6,   # 7: l_wrist     <- left_wrist
            16,  # 8: r_hip       <- right_hip
            17,  # 9: r_knee      <- right_knee
            18,  # 10: r_ankle    <- right_ankle
            12,  # 11: l_hip      <- left_hip
            13,  # 12: l_knee     <- left_knee
            14,  # 13: l_ankle    <- left_ankle
            -1,  # 14: right_eye  (нет аналога)
            -1,  # 15: left_eye
            -1,  # 16: right_ear
            -1,  # 17: left_ear
        ]
        
        # Создаем массив с 3 координатами (X, Y, Z), где Z=0
        converted_data = np.zeros((len(data), 2, 18, 3), dtype=np.float32)
        for t, frame in enumerate(data):
            for m, body in enumerate(frame):
                for v_out, v_ntu in enumerate(ntu_to_openpose):
                    if v_ntu != -1:
                        # Заполняем X и Y координаты, Z остается 0
                        converted_data[t, m, v_out, :2] = body[v_ntu]
        return converted_data

    def augment_skeleton(self, data, noise_std=0.05):
        flip_prob = 0.2
        noise = np.random.normal(0, noise_std, data.shape).astype(np.float32)
        scale = np.random.uniform(0.7, 1.0)
        data = data * scale
        # Случайное отражение (по оси X)
        if np.random.rand() < flip_prob:
            data[..., 0] = -data[..., 0]
        # Случайное обнуление координат одной из ключевых точек
        T, M, V, C = data.shape
        for frame_idx in range(T):
            if np.random.rand() < 0.1:  # 10% вероятность аугментации
                person_idx = np.random.randint(M)
                joint_idx = np.random.randint(V)
                data[frame_idx, person_idx, joint_idx, :] = 0  # Обнуляем X и Y координаты
        return data + noise

    def normalize_skeleton(self, data):
        if np.max(np.abs(data)) < 1e-6:
            return data
        data = np.nan_to_num(data, nan=0.0)
        # Нормализация по центру таза (первый сустав)
        hip_indices = [8, 11]
        hip_centers = np.mean(data[:, :, hip_indices, :], axis=2, keepdims=True)
        
        # Создаем маску для ненулевых точек (где хотя бы одна из координат X, Y, Z не равна нулю)
        non_zero_mask = np.any(data != 0, axis=-1, keepdims=True)
        
        # Вычитаем центр таза только из ненулевых точек
        data = np.where(non_zero_mask, data - hip_centers, data)
        # Масштабирование: вычисляем максимальное расстояние от центра таза до любой точки для каждого тела отдельно
        for m in range(data.shape[1]):
            person_data = data[:, m, :, :]

            scale = np.max(person_data)
            if scale > 1e-6:
                data[:, m, :, :] = person_data / scale
        return data

    def interpolate_frames(self, data, target=300):
        T, M, V, C = data.shape
        if T == target:
            return data
        old_t, new_t = np.linspace(0, 1, T), np.linspace(0, 1, target)
        result = np.zeros((target, M, V, C), dtype=np.float32)
        for m in range(M):
            for v in range(V):
                for c in range(C):
                    f = np.interp(new_t, old_t, data[:, m, v, c])
                    result[:, m, v, c] = f
        return result

    def extract_label(self, filename):
        parts = filename.replace('.skeleton', '').split('A')
        if len(parts) >= 2:
            return int(parts[-1]) - 1
        return 0

class STGCNWrapper:
    def __init__(
        self,
        weights_path='./models/st_gcn.kinetics.pt',
        label_map_path='kinetics400-id2label.txt',
        device='cpu',
        num_class: int = 400,
    ):
        self.weights_path = weights_path
        self.label_map_path = label_map_path
        self.num_class = num_class
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # модель как в config/st_gcn/kinetics-skeleton/test.yaml
        self.model = Model(
            in_channels=3,
            num_class=num_class,
            edge_importance_weighting=True,
            graph_args={'layout': 'openpose', 'strategy': 'spatial'}
        ).to(self.device)
        
        # Установка атрибутов num_person и num_node для совместимости с датасетом
        self.model.graph.num_person = 2

        ckpt = torch.load(weights_path, map_location=self.device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        self.model.load_state_dict(ckpt)
        self.model.eval()

        # загрузка словаря id->label (можно отключить, если не нужен)
        self.id2label = None
        if label_map_path is not None and os.path.exists(label_map_path):
            with open(label_map_path, 'r') as f:
                self.id2label = json.load(f)

    def train_model(self, config_path: str):
        """
        Обучает модель с параметрами из YAML-файла.

        Args:
            config_path (str): Путь к файлу конфигурации stgcn_params.yml
        """
        import yaml
        import os
        import json
        from datetime import datetime
        from torch.utils.data import Dataset, DataLoader
        from torch.utils.tensorboard import SummaryWriter
        from sklearn.model_selection import train_test_split
        from tqdm import tqdm
        import torch.nn as nn

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
        pretrained_weights = config['training'].get('pretrained_weights', None)

        print(f"🔧 Устройство: {self.device}")
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
            labels = [self.extract_label(f) for f in all_files]

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
                self.model.load_state_dict(torch.load(pretrained_weights, map_location=self.device))
                print("✅ Веса успешно загружены")
            except Exception as e:
                print(f"❌ Ошибка при загрузке весов: {e}")
                raise
        elif pretrained_weights:
            raise FileNotFoundError(f"Файл с весами не найден: {pretrained_weights}")

        self.model.to(self.device)
        self.model.train()

        scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        # TensorBoard
        run_name = f"stgcn_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        writer = SummaryWriter(log_dir=f'runs/{run_name}')
        print(f"📈 Логи: runs/{run_name}")

        # Создаем датасеты
        train_dataset = SkeletonDataset(train_files, skeleton_dir, self.model.graph.num_node, self.model.graph.num_person)
        val_dataset = SkeletonDataset(val_files, skeleton_dir, self.model.graph.num_node, self.model.graph.num_person)

        # Оптимизация DataLoader для CUDA
        pin_memory = self.device.type == 'cuda'
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory,
                                  persistent_workers=(num_workers > 0))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=pin_memory,
                                persistent_workers=(num_workers > 0))

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        print(f"🔧 Параметры: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"\n🚀 Начало тренировки ({epochs} эпох)...\n")

        best_acc = 0.0
        os.makedirs('models', exist_ok=True)

        for epoch in range(epochs):
            # Тренировка
            train_loss, train_correct, train_total = 0, 0, 0
            for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                # Используем autocast для автоматического смешанного precision
                data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)
                
                optimizer.zero_grad()
                
                # Используем scaler для градиентов в mixed precision
                if self.device.type == 'cuda':
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
            self.model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.device.type=='cuda'):
                    for data, labels in val_loader:
                        data, labels = data.to(self.device), labels.to(self.device)
                        outputs = self.model(data)
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
                model_path = 'models/best_stgcn_model.pth'
                torch.save(self.model.state_dict(), model_path)
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

        # Вычисление дополнительных метрик на валидационной выборке
        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        
        from sklearn.metrics import classification_report
        report = classification_report(all_labels, all_preds, digits=4)
        print("\n📋 Полный отчёт по классификации на валидационной выборке:")
        print(report)
        
        # Дополнительно выведем средние метрики
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
        print(f"\n📊 Средние метрики (macro avg):\n   Precision: {precision:.4f}\n   Recall:    {recall:.4f}\n   F1-Score:  {f1:.4f}")

    @torch.no_grad()
    def predict_logits(self, data_numpy):
        """
        data_numpy: np.ndarray формы (1, 3, T, V, M), V=18, M=1
        возвращает torch.Tensor формы (1, num_class)
        """
        data_tensor = torch.from_numpy(data_numpy).float().to(self.device)
        out = self.model(data_tensor)
        return out

    @torch.no_grad()
    def predict_topk(self, data_numpy, k=5):
        """
        Возвращает список top-k (cls_id, prob, label_str)
        """
        logits = self.predict_logits(data_numpy)  # (1,num_class)
        prob = torch.softmax(logits, dim=1)[0]
        topk = torch.topk(prob, k=k)

        results = []
        for cls_id, p in zip(topk.indices.tolist(), topk.values.tolist()):
            label = None
            if self.id2label is not None:
                label = self.id2label.get(str(cls_id), None)
            results.append((cls_id, float(p), label))
        return results

    def extract_label(self, filename):
        parts = filename.replace('.skeleton', '').split('A')
        if len(parts) >= 2:
            return int(parts[-1]) - 1
        return 0