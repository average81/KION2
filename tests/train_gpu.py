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
        data = interpolate_frames(data, target=30)
        # Усредняем по телам → (30, 25, 3), есть над чем поработать, добавит ошибок, если несколько тел
        data = data.mean(axis=1)  # или data[:, 0, ...] для первого тела
        tensor = torch.FloatTensor(data).permute(0, 2, 1)  # (T, C, V)
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

# %%
# ==============================================================================
# 4. ТРЕНИРОВКА (с исправлениями для GPU)
# ==============================================================================
def train():
    skeleton_dir = 'C:/Users/above/IdeaProjects/NTU-RGB+D 120/nturgb+d_skeletons'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Устройство: {device}")
    
    # TensorBoard
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=f'runs/{run_name}')
    print(f"📊 Логи: runs/{run_name}")
    
    # Данные
    all_files = [f for f in os.listdir(skeleton_dir) if f.endswith('.skeleton')]
    print(f"📁 Всего файлов: {len(all_files)}")
    
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    print(f"📊 Train: {len(train_files)}, Val: {len(val_files)}")
    
    train_dataset = SkeletonDataset(train_files, skeleton_dir)
    val_dataset = SkeletonDataset(val_files, skeleton_dir)
    
    # 🔥 Уменьшаем batch_size для стабильности на новых GPU
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    
    # Модель
    model = LSTMSkeletonNet(num_classes=60).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    
    print(f"🔧 Параметры: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\n🚀 Тренировка (50 эпох)...\n")
    
    best_acc = 0
    for epoch in range(50):  # 50 эпох 
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                outputs = model(data)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        scheduler.step()
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
        
        print(f"Epoch {epoch+1}/50: Loss={train_loss/len(train_loader):.3f}, "
              f"Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%")
        
        # 🔥 TensorBoard: ТОЛЬКО скаляры (без histogram!)
        writer.add_scalar('Loss/train', train_loss/len(train_loader), epoch+1)
        writer.add_scalar('Loss/val', val_loss/len(val_loader), epoch+1)
        writer.add_scalar('Accuracy/train', train_acc, epoch+1)
        writer.add_scalar('Accuracy/val', val_acc, epoch+1)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch+1)
    
    writer.close()
    print(f"\n✅ Готово! Лучшая точность: {best_acc:.2f}%")
    print(f"📊 TensorBoard: %tensorboard --logdir runs")

# %%
# ==============================================================================
# 5. ЗАПУСК
# ==============================================================================
if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)

    train()