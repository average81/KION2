# Формат поз и нормализация (ST-GCN / Kinetics)

Проект **KION2** решает задачу **распознавания действия по видео** через скелет: сначала из кадров получают ключевые точки тела, затем последовательность поз обрабатывает **ST-GCN**. Этот документ — технический справочник: какие суставы в каком порядке, как нормализуются координаты и в каком виде данные лежат в `.npy`/`.pkl` для обучения и проверки модели.

**NTU RGB+D:** каталог с данными на диске — **`models/stgcn/data/`** (см. раздел 5 ниже).

---

## 1. Нумерация суставов (18, OpenPose layout)

Модель ST-GCN ожидает **18 суставов** в порядке OpenPose (подмножество BODY_25). Левая/правая сторона — со стороны человека в кадре.


| Индекс | Часть тела (en) | По-русски       |
| ------ | --------------- | --------------- |
| 0      | nose            | нос             |
| 1      | neck            | шея             |
| 2      | right_shoulder  | правое плечо    |
| 3      | right_elbow     | правый локоть   |
| 4      | right_wrist     | правое запястье |
| 5      | left_shoulder   | левое плечо     |
| 6      | left_elbow      | левый локоть    |
| 7      | left_wrist      | левое запястье  |
| 8      | right_hip       | правое бедро    |
| 9      | right_knee      | правое колено   |
| 10     | right_ankle     | правая лодыжка  |
| 11     | left_hip        | левое бедро     |
| 12     | left_knee       | левое колено    |
| 13     | left_ankle      | левая лодыжка   |
| 14     | right_eye       | правый глаз     |
| 15     | left_eye        | левый глаз      |
| 16     | right_ear       | правое ухо      |
| 17     | left_ear        | левое ухо       |


В проекте этот порядок используется в адаптере `models/stgcn/json_to_stgcn_adapter.py` (маппинг нашей 30-точечной разметки → 18 через `OUR30_TO_OPENPOSE18`) и соответствует BODY25→18 в `models/stgcn/openpose_to_stgcn_adapter.py`.

---

## 2. Нормализация координат (Kinetics pipeline)

Источник: описание нормализации в оригинальном Kinetics-skeleton пайплайне ST-GCN (upstream yysijie / mmskeleton; центрация по кадру). В этом репозитории обучение Kinetics идёт через `feeder.feeder.Feeder` и массивы `.npy`, отдельный исторический фидер с каталогом JSON-файлов не используется.

- **Исходные координаты**: нормализованы к отрезку **[0, 1]** по ширине и высоте кадра (x — по ширине, y — по высоте).
- **Преобразование**: только **центрация**:
  - `x_new = x - 0.5`, `y_new = y - 0.5`
  - центр кадра становится (0, 0); типичный диапазон после центрации около **[-0.5, 0.5]** (в сохранённых .npy могут быть иные пределы в зависимости от обрезки/размера кадра).
- **Третий канал**: confidence/score в **[0, 1]**; где score == 0, координаты x и y обнуляются.
- **Масштабирования по торсу / mid_hip в пайплайне Kinetics нет.** В проекте и OpenPose, и наш JSON нормализуются так же: пиксели → [0,1] по width×height (по умолчанию 340×256) → центрация −0.5 (`openpose_to_stgcn_adapter.py`, `json_to_stgcn_adapter.py`).

---

## 3. Размер изображения и видео

Для **Kinetics-skeleton** (по описанию датасета и README):

- Разрешение видео: **340×256** пикселей.
- Частота кадров: **30 fps**.
- Координаты в [0, 1] заданы относительно этого размера кадра.

---

## 4. Формат данных

### Массив скелетов (.npy)

- Форма: **(N, 3, T, 18, M)**
  - **N** — число сэмплов (клипов);
  - **3** — каналы (x, y, score);
  - **T** — число кадров (например 300);
  - **18** — суставов;
  - **M** — человек (часто 2; второй слот может быть padding при одном человеке в кадре).

### Метки (.pkl)

- **Вариант 1**: кортеж `(sample_name, sample_label)` — список имён сэмплов и список индексов классов (0..399 для Kinetics).
- **Вариант 2**: словарь с ключом `label`, `y`, `labels` или `val_label` — массив меток длины N.

Загрузка обоих форматов реализована в `validate_stgcn_npy_pkl.py`.

### Использование в проекте

- Адаптер «наш JSON → тензор для ST-GCN»: `models/stgcn/json_to_stgcn_adapter.py` (нормализация как в Kinetics: width/height → [0,1] → −0.5; на выходе форма (1, 3, T, 18, 1)). Аналогично загрузка OpenPose JSON в `models/stgcn/openpose_to_stgcn_adapter.py`.
- Валидация на .npy/.pkl: `validate_stgcn_npy_pkl.py` (Top-1/Top-5, опция `--all-persons`).
- Просмотр статистики по координатам и визуализация кадров: `tests/tools/inspect_labels.py`.

---

## 5. Пути к данным NTU RGB+D (этот репозиторий)

**Корень данных — `models/stgcn/data/`** (не `models/stgcn/dataset/data/`). Подробнее: `models/stgcn/data/README.md`.

| Этап | Путь |
|------|------|
| Исходные 2D color скелеты NTU | `models/stgcn/data/NTU-RGB-D/nturgb+d_skeletons/*.skeleton` |
| Списки xsub (имена + метки), вход сборки | `models/stgcn/data/NTU-RGB-D-2D-from-color/xsub_train_label.pkl`, `xsub_val_label.pkl` |
| Выход сборки (формат Kinetics/OpenPose-18) | `models/stgcn/data/NTU-RGB-D-2D-from-color/xsub_train_data18.npy`, `xsub_val_data18.npy`, соответствующие `*_label.pkl` |

Сборка массивов: из **корня репозитория** `KION2_exp`:

```bash
python models/stgcn/tools/build_ntu_xsub_2d_25_18.py
```

Скрипт использует `STGCN_ROOT = models/stgcn` и читает/пишет под `models/stgcn/data/` (см. `tools/build_ntu_xsub_2d_25_18.py`).

**Обучение ST-GCN** (`models/stgcn/main.py`): рабочий каталог — **`models/stgcn`**. В `config/st_gcn/ntu-xsub-kinetics-2d/train.yaml` пути к `.npy`/`.pkl` заданы относительно него, например `./data/NTU-RGB-D-2D-from-color/xsub_train_data18.npy`. Логи и чекпоинты — в `./work_dir/...` (см. поле `work_dir` в том же yaml).

Цепочка в общем виде: **`.skeleton` → `build_ntu_xsub_2d_25_18.py` → `*_data18.npy` + `*_label.pkl` → `feeder.feeder.Feeder` → `net.st_gcn.Model` (60 классов NTU, layout openpose).**

