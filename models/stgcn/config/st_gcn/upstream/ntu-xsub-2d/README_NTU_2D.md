# NTU RGB+D 2D — пайплайн для оценки 2D-модели (upstream)

> **В этом репозитории** основной сценарий — `config/st_gcn/ntu-xsub-kinetics-2d/` (формат Kinetics, сборка через `tools/build_ntu_xsub_2d_25_18.py`). Ниже — оригинальное описание из upstream ST-GCN для конфигов в `upstream/ntu-xsub-2d/`.

Скелеты NTU приводятся к 2D: каналы **(x, y, mask)** вместо (x, y, z). Так можно оценить верхнюю границу качества чисто 2D-модели на том же датасете.

## 1. Скачать NTU RGB+D

- Скачай **3D skeletons** (≈5.8 GB) с [официального сайта](http://rose1.ntu.edu.sg/datasets/actionrecognition.asp).
- Распакуй в каталог, например: `data/NTU-RGB-D/nturgb+d_skeletons/` (внутри — файлы `*.skeleton`).

## 2. Сгенерировать 2D-датасет

Из **корня репозитория** `st-gcn`:

```bash
cd tools
python ntu_gendata_2d.py \
  --data_path ../data/NTU-RGB-D/nturgb+d_skeletons \
  --out_folder ../data/NTU-RGB-D-2D \
  --ignored_sample_path ../resource/NTU-RGB-D/samples_with_missing_skeletons.txt
cd ..
```

Появятся:

- `data/NTU-RGB-D-2D/xsub/train_data.npy`, `train_label.pkl`, `val_data.npy`, `val_label.pkl`
- `data/NTU-RGB-D-2D/xview/` — то же для cross-view.

## 3. Обучить модель (Cross-Subject)

```bash
python main.py recognition -c config/st_gcn/upstream/ntu-xsub-2d/train.yaml
```

Чекпоинты сохраняются в `work_dir/recognition/ntu-xsub-2d/ST_GCN/` (например, `epoch80_model.pt`).

## 4. Оценить на валидации

```bash
python main.py recognition -c config/st_gcn/upstream/ntu-xsub-2d/test.yaml \
  --weights ./work_dir/recognition/ntu-xsub-2d/ST_GCN/epoch80_model.pt
```

## 5. Cross-View (опционально)

- Обучение: `config/st_gcn/upstream/ntu-xview-2d/train.yaml`
- Тест: `config/st_gcn/upstream/ntu-xview-2d/test.yaml` и `--weights .../ntu-xview-2d/ST_GCN/epoch80_model.pt`

## Ориентир по метрикам

- **3D NTU (оригинал статьи):** Cross-Subject ~81.6%, Cross-View ~88.8%.
- **2D (x, y, mask):** точность обычно ниже (нет глубины). Полученные на 2D цифры дают верхнюю границу того, чего можно ждать от 2D-скелетов на NTU.
