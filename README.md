# KION2

## О проекте

**Задача:** по видео с людьми определить **класс действия** (ходьба, мах рукой, вставание и т.п.) — в духе датасетов вроде NTU RGB+D / Kinetics-Skeleton, без разметки «вручную» на каждом кадре, опираясь на **скелет** (ключевые точки тела).

**Как это устроено:**

1. **Оценка позы** — из кадров извлекаются координаты суставов (в проекте — пайплайн на **YOLO-Pose** / `VideoProcessor`, см. `config.yml` и тесты).
2. **Распознавание действия** — последовательность поз подаётся в **ST-GCN** (графовая сеть по скелету во времени): на выходе — вероятности по классам действий (например, **60 классов NTU-60** или **400 Kinetics**, в зависимости от весов и конфига).
3. **Обучение** — по желанию модель ST-GCN дообучается на своих данных в формате скелетов (конфиги в `models/stgcn/config/`, данные и сборка `.npy` — в `models/stgcn/data/` и скриптах сборки).

**Что лежит в репозитории:** код пайплайна «видео → позы → действие», обвязка ST-GCN, утилиты для данных и отладки; тяжёлые веса и датасеты — локально или по ссылкам (готовые веса ST-GCN для NTU и Kinetics — в разделе «Демо» ниже).

Подбор и обучение моделей ведётся под **pose estimation** актёров и последующую классификацию действий на выбранном датасете.

### Откуда взята идея (исходный ST-GCN)

Распознавание действий по **скелету** в виде **ST-GCN** (Spatial Temporal Graph Convolutional Network) опирается на оригинальную работу и открытый код:

- **Статья:** S. Yan, Y. Xiong, D. Lin — *Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition* (AAAI 2018), [arXiv:1801.07455](https://arxiv.org/abs/1801.07455).
- **Репозиторий с кодом обучения и конфигами:** [github.com/yysijie/st-gcn](https://github.com/yysijie/st-gcn).

В **KION2** используется **адаптированная** часть этого пайплайна (`models/stgcn/`: `main.py`, сеть, фидеры, конфиги под NTU/Kinetics). Снимки оригинальных конфигов и описания — в каталоге `models/stgcn/config/st_gcn/upstream/`.

**Важно:** в [yysijie/st-gcn](https://github.com/yysijie/st-gcn) основной фокус — модель ST‑GCN и обучение по уже подготовленным скелетным данным (NTU 3D‑скелеты, Kinetics‑skeleton от OpenPose), плюс скрипты упаковки этих скелетов в `.npy`/`.pkl` для тренинга. В репозитории есть демо, которое умеет прогонять OpenPose по видео, но универсальный конвейер «произвольные свои видео → собственный скелетный датасет для обучения» там не завершён и остаётся на стороне пользователя.

В **KION2** сейчас реализован свой блок оценки поз (YOLO‑Pose, `VideoProcessor`, см. `config.yml` и `tests/`), который позволяет одним запуском получить JSON с 2D‑ключевыми точками из видео. Цепочка «видео → позы → действие» у нас собирается из этого блока оценки поз и адаптированного ST‑GCN; автоматической упаковки таких поз в `.npy`/`.pkl` для обучения пока нет и планируется как следующий шаг пайплайна.

### Наша разработка (поверх ST-GCN)

Отдельно от upstream заложено следующее — **это разработка KION2**:

- **Работа с 2D-скелетами** — весь целевой сценарий завязан на **плоские** координаты суставов (OpenPose-18, формат как у Kinetics-skeleton), а не на 3D-ветку оригинального пайплайна.
- **Подготовка датасета для 2D** — скрипты и пайплайн сборки массивов из NTU RGB+D (в т.ч. из `.skeleton` в `*_data18.npy` / метки `.pkl` под конфиги `ntu-xsub-kinetics-2d`), пути и нормализация под нашу обвязку: см. `models/stgcn/tools/build_ntu_xsub_2d_25_18.py` и разделы ниже про данные.

## Установка зависимостей

**PyTorch** — подберите сборку под свою версию CUDA:

```bash
# Проверьте версию CUDA:
nvidia-smi

# Для CUDA 13.0:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Для CUDA 12.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Для CUDA 12.6:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Для CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Для CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Для CPU (если нет GPU):
pip install torch torchvision
```

**Важно:**

- Сначала установите PyTorch, затем остальные зависимости.
- Актуальные сборки: [pytorch.org — Get Started](https://pytorch.org/get-started/locally/)

Остальные пакеты:

```bash
pip install -r requirements.txt
```

## ffmpeg

Нужен для части сценариев работы с видео.

**Ubuntu / Linux:**

```bash
sudo apt update
sudo apt install ffmpeg
```

**Windows 10/11:**

```bash
winget install ffmpeg
```

## Заметки: сторонние варианты pose estimation

Справочный обзор (MediaPipe, OpenPose, YOLO-pose и др.) вынесен в отдельный файл — для KION2 он **необязателен**; в проекте используется свой пайплайн оценки позы (см. раздел «О проекте» выше).

- **[notes/pose_estimation_overview.md](notes/pose_estimation_overview.md)** — обзор вариантов  
- **[notes/README.md](notes/README.md)** — список заметок в каталоге `notes/`

## Справочник по данным (формат скелета)

**Зачем:** понять, в каком виде хранятся суставы, как нормализуются координаты и как устроены `.npy`/`.pkl` — это нужно и для обучения, и для отладки инференса.

- **[Формат поз и нормализация](models/stgcn/docs/pose_and_normalization.md)** — OpenPose-18, нормализация,  
  связь с NTU (`models/stgcn/data/`).

---

## Подготовка данных NTU (если обучаете ST-GCN сами)

**Зачем:** из сырых скелетов NTU собрать массивы `*.npy` и метки `*.pkl`, которые читает `main.py` по конфигам.

- Корень данных: **`models/stgcn/data/`** (подробнее: `models/stgcn/data/README.md`).
- Исходники: `*.skeleton` в `NTU-RGB-D/nturgb+d_skeletons/`
- Списки сплита: `xsub_train_label.pkl`, `xsub_val_label.pkl` в `NTU-RGB-D-2D-from-color/`
- После сборки: `xsub_*_data25.npy`, `xsub_*_data18.npy`, `*_label.pkl` в том же каталоге  
- Большие файлы и логи в репо не коммитятся; чекпоинты обучения — в `models/stgcn/work_dir/`

Скрипт сборки:

```bash
python models/stgcn/tools/build_ntu_xsub_2d_25_18.py
```

---

## Пакетная обработка видео → JSON с позами (`tests/video_to_pose.py`)

**Зачем:** из **многих роликов** (несколько папок с `.avi`/`.mp4` и т.д.) одним запуском получить **JSON с 2D‑ключевыми точками** — тот же пайплайн, что и для одного файла: **YOLO‑Pose** + `VideoProcessor` (см. корневой `config.yml`, поле `pose_ext_model`). Упаковка таких JSON в `.npy`/`.pkl` для обучения ST‑GCN в разделе **«О проекте»** выше отмечена как следующий шаг пайплайна.

Запуск из **корня репозитория** (чтобы находился `config.yml`, либо укажите `--config_path`).

| Параметр | Описание |
|----------|----------|
| `video` | один файл (без `--batch_root`) |
| `--batch_root DIR [DIR ...]` | одна или несколько корневых папок с видео |
| `--recursive` | искать файлы **во вложенных** каталогах (для **NTU RGB+D** `nturgb+d_rgb` ролики лежат глубже — флаг **обязателен**) |
| `--output_dir` | базовый каталог для результатов; сохраняется структура подпапок |
| `--no_vis` | только JSON, без рендера видео со скелетом (быстрее) |

Расширения поиска: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`, `.m4v`.

Если передано **несколько** `--batch_root` и папки называются одинаково (`…/nturgbd_rgb_s00X/nturgb+d_rgb`), в `output_dir` добавляется метка вида **`nturgbd_rgb_s001`**, чтобы не смешивать выходы.

Примеры:

```bash
# одна папка, все видео рекурсивно (типично NTU)
python tests/video_to_pose.py --batch_root /path/to/nturgb+d_rgb --recursive \
  --output_dir /path/to/out_poses --no_vis

# несколько корневых каталогов
python tests/video_to_pose.py --batch_root /path/a/nturgb+d_rgb /path/b/nturgb+d_rgb \
  --recursive --output_dir /path/to/out_poses --no_vis
```

**Многострочная команда в bash:** не ставьте обратный слэш `\` в **конце последней** строки — иначе оболочка будет ждать продолжения (приглашение `>`).

---

## Обучение классификатора действий (ST-GCN)

**Зачем:** обучить или дообучить сеть, которая по **последовательности скелетов** выдаёт класс действия (не путать с «видео → действие» в следующем разделе — там уже готовый демо-скрипт).

- Описание конфигов: **[models/stgcn/config/st_gcn/README.md](models/stgcn/config/st_gcn/README.md)**  
  (`ntu-xsub-kinetics-2d` — NTU, `kinetics-skeleton` — 400 классов; прочее — `upstream/`).
- Запуск из каталога **`models/stgcn`**:

```bash
cd models/stgcn
python main.py recognition -c config/st_gcn/ntu-xsub-kinetics-2d/train.yaml
```

---

## Демо: одно видео → позы → действие (инференс)

**Зачем:** без своего датасета прогнать **ролик**: из кадров извлекаются позы, затем ST-GCN выводит Top‑k классов действий.

### Веса

Готовые чекпоинты ST-GCN лежат в одной папке на облаке:

- **[KION2_weights — Облако Mail](https://cloud.mail.ru/public/bDAx/TJPWohriL)** — оттуда же берите и **NTU-60**, и **Kinetics** (400 классов).

| Файл в `models/` | Назначение |
|------------------|------------|
| **`st_gcn.ntu60.pt`** | 60 классов NTU (пресет `--dataset ntu60` в демо-скрипте) |
| **`st_gcn.kinetics.pt`** | 400 классов Kinetics-Skeleton (`--dataset kinetics400`) |

Положить нужные `.pt` в каталог **`models/`** под этими именами или указать путь: `--stgcn_weights путь.pt`. Свой чекпоинт после обучения можно положить рядом и передать тем же флагом.

### Скрипт и команда

Скрипт: `tests/action_detection_video-yolo-stgcn.py` (YOLO / VideoProcessor → ST-GCN).

Из **корня репозитория**:

```bash
python tests/action_detection_video-yolo-stgcn.py path/to/video.mp4
```

---

## Утилиты для отладки и проверки данных

**Зачем:** смотреть баланс классов, форму тензоров, прогонять ST-GCN по отдельным JSON/`.npy` без полного видео-пайплайна.

| Скрипт | Что делает |
|--------|------------|
| `tests/video_to_pose.py` | видео → JSON с позами; **пакетно** — отдельный раздел **«Пакетная обработка видео → JSON с позами»** выше (`--batch_root`, `--recursive`) |
| `tests/tools/inspect_labels.py` | обзор `.npy` + `.pkl`: классы, форма `(N, 3, T, 18, M)`, статистика каналов |
| `tests/tools/show_inference_input.py` | что уходит в ST-GCN при инференсе, визуализация скелетов |
| `tests/tools/test_stgcn_on_openpose_data.py` | ST-GCN по папке `*_keypoints.json`, Top‑5 |
| `tests/tools/validate_stgcn_npy_pkl.py` | оффлайн Top‑1/Top‑5 или предсказания по `.npy`/`.pkl` |
