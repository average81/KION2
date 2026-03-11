"""
Вспомогательный скрипт для «быстрого осмотра» датасета Kinetics-skeleton
или совместимых с ним .npy/.pkl:

- показывает баланс классов (самые частые и редкие метки);
- печатает форму массива скелетов (N, 3, T, 18, M);
- выводит статистику по координатам (x, y, score) для первого клипа;
- рисует несколько кадров первого клипа и сохраняет их в PNG.

По умолчанию ожидает файлы из `video_samples`:
    - labels: `video_samples/val_label.pkl`
    - data:   `video_samples/val_data.npy`

При необходимости изменить `label_path` и `data_path` под свой датасет
и запустить:

Запуск из корня проекта:
    python tests/tools/inspect_labels.py
"""
import sys
from pathlib import Path

# Корень проекта: пути к данным считаем от него, чтобы скрипт работал из любой папки
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# Пути к файлам меток и данных (относительно корня проекта)
label_path = ROOT / "video_samples" / "val_label.pkl"
data_path = ROOT / "video_samples" / "val_data.npy"

with open(label_path, "rb") as f:
    sample_name, sample_label = pickle.load(f)

print("Всего сэмплов:", len(sample_name))

# считаем частоты классов
counter = Counter(sample_label)

print("\n5 самых частых классов:")
for cls, cnt in counter.most_common(5):
    print(f"class {cls}: {cnt} раз")

print("\n5 самых редких классов:")
# сортируем по возрастанию частоты
for cls, cnt in sorted(counter.items(), key=lambda x: x[1])[:5]:
    print(f"class {cls}: {cnt} раз")

# проверка формы данных
data = np.load(data_path, mmap_mode="r")
print("\nФорма data:", data.shape)  # (N, C, T, V, M)

N, C, T, V, M = data.shape
sample_idx = 0
person_idx = 0
person = data[sample_idx, :, :, :, person_idx]  # (3, T, V)

# ------------------------------------------------------------------
# Величины координат: первый ролик, первый человек (x, y, score)
# ------------------------------------------------------------------
x_all = person[0]   # (T, V)
y_all = person[1]   # (T, V)
s_all = person[2]   # (T, V)

print("\n--- Координаты суставов (sample 0, person 0) ---")
print("Канал x:  min = {:.4f}, max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
    float(x_all.min()), float(x_all.max()), float(x_all.mean()), float(x_all.std())))
print("Канал y:  min = {:.4f}, max = {:.4f}, mean = {:.4f}, std = {:.4f}".format(
    float(y_all.min()), float(y_all.max()), float(y_all.mean()), float(y_all.std())))
print("Канал score: min = {:.4f}, max = {:.4f}, mean = {:.4f}".format(
    float(s_all.min()), float(s_all.max()), float(s_all.mean())))
print("(После нормализации Kinetics: x,y обычно порядка [-0.5, 0.5] вокруг 0; score в [0, 1].)")

# ------------------------------------------------------------------
# Визуализация: 25 кадров первого ролика, первый человек (M=0)
# ------------------------------------------------------------------

num_frames_to_show = min(25, T)
cols = 5
rows = int(np.ceil(num_frames_to_show / cols))

plt.figure(figsize=(cols * 3, rows * 3))
for i in range(num_frames_to_show):
    t = int(i * (T - 1) / max(num_frames_to_show - 1, 1))  # равномерно по всей длине
    x = person[0, t]  # (V,)
    y = person[1, t]  # (V,)

    ax = plt.subplot(rows, cols, i + 1)
    ax.scatter(x, y, c="red", s=10)
    for j, (xx, yy) in enumerate(zip(x, y)):
        ax.text(xx, yy, str(j), fontsize=6)
    ax.set_title(f"frame {t}")
    ax.invert_yaxis()
    ax.axis("off")

plt.tight_layout()
# В среде без интерактивного окна сохраняем картинку в файл
out_path = "sample0_person0_frames.png"
plt.savefig(out_path, dpi=150)
print(f"\nКадры первого ролика первого человека сохранены в {out_path}")