"""
Тестирование работы ST-GCN в данном проекте на выборке из .npy (скелеты) и .pkl (метки, опционально).

Ожидаемый формат:
  - .npy: массив скелетов формы (N, 3, T, 18, M) — N сэмплов, 3 канала, T кадров, 18 суставов, M человек.
    По умолчанию используется первый человек. С флагом --all-persons предсказания делаются по каждому человеку (например, для армрестлинга).
  - .pkl: список меток длины N (int, класс на сэмпл) или словарь с ключом 'label' / 'y' / 'labels' / 'val_label' (массив меток).
    Метки 0..399 — классы Kinetics; тогда считается accuracy. Иначе выводятся предсказания и метки для ручной проверки.

Запуск из корня проекта:
  python tests/tools/validate_stgcn_npy_pkl.py path/to/data.npy path/to/labels.pkl
  python tests/tools/validate_stgcn_npy_pkl.py path/to/data.npy  # без pkl — только предсказания
  python tests/tools/validate_stgcn_npy_pkl.py path/to/data.npy path/to/labels.pkl --max-samples 100  # отладка на 100 сэмплах
  python tests/tools/validate_stgcn_npy_pkl.py ... --all-persons  # предсказание по каждому человеку в кадре
"""
import sys
from pathlib import Path

# Корень проекта в sys.path для импорта models.stgcn
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import pickle

import numpy as np

from models.stgcn.stgcn_wrapper import STGCNWrapper


def load_validation_data(npy_path: str, pkl_path: str | None):
    data = np.load(npy_path)
    labels = None
    if pkl_path is not None:
        with open(pkl_path, "rb") as f:
            raw = pickle.load(f)
        # варианты формата:
        # 1) просто список/массив меток
        if isinstance(raw, (list, np.ndarray)):
            labels = np.asarray(raw)
        # 2) кортеж (sample_name, sample_label) как в Kinetics
        elif isinstance(raw, tuple) and len(raw) == 2:
            _, sample_label = raw
            labels = np.asarray(sample_label)
        # 3) словарь с ключом label / y / labels / val_label
        elif isinstance(raw, dict):
            labels = raw.get("label", raw.get("y", raw.get("labels", raw.get("val_label", None))))
            if labels is not None:
                labels = np.asarray(labels)
        if labels is not None and len(labels) != len(data):
            raise ValueError(
                f"Длина меток ({len(labels)}) не совпадает с числом сэмплов в .npy ({len(data)})"
            )
    return data, labels


def ensure_stgcn_shape(x: np.ndarray, keep_all_persons: bool = False) -> np.ndarray:
    """
    Приводит батч к форме (N, 3, T, 18, 1) или (N, 3, T, 18, M).
    - keep_all_persons=False: при любом M оставляем только первого человека → (N, 3, T, 18, 1).
    - keep_all_persons=True: не трогаем M → (N, 3, T, 18, M). Для 4D (N,T,V,C) добавляем ось M=1.
    """
    if x.ndim == 5:
        N, c, T, V, M = x.shape
        if (c, V) == (3, 18):
            if keep_all_persons:
                return x
            return x[:, :, :, :, 0:1].copy()
    if x.ndim == 4:
        N, T, V, C = x.shape
        if V == 18 and C == 3:
            x = np.transpose(x, (0, 3, 1, 2))  # (N,T,18,3) -> (N,3,T,18)
            x = x[:, :, :, :, np.newaxis]      # (N,3,T,18,1)
            return x
    raise ValueError(
        f"Неожиданная форма данных: {x.shape}. Ожидается (N, 3, T, 18, M) или (N, T, 18, 3)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Проверка работы ST-GCN на валидационной выборке (.npy + .pkl)."
    )
    parser.add_argument(
        "npy_path",
        type=str,
        help="Путь к .npy файлу с данными (форма (N, 3, T, 18, 1) или (N, T, 18, 3)).",
    )
    parser.add_argument(
        "pkl_path",
        type=str,
        nargs="?",
        default=None,
        help="Путь к .pkl с метками (список или dict с ключом label/y). Опционально.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="models/st_gcn.kinetics.pt",
        help="Путь к весам ST-GCN.",
    )
    parser.add_argument(
        "--labels-file",
        type=str,
        default="models/stgcn/kinetics400-id2label.txt",
        help="Путь к id2label (Kinetics).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Устройство (cpu/cuda).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Максимум сэмплов для проверки (для отладки).",
    )
    parser.add_argument(
        "--all-persons",
        action="store_true",
        help="Делать предсказание по каждому человеку в кадре (например армрестлинг). Accuracy: засчитывается, если хотя бы один человек угадал класс.",
    )
    args = parser.parse_args()

    data, labels = load_validation_data(args.npy_path, args.pkl_path)
    orig_shape = data.shape
    data = ensure_stgcn_shape(data, keep_all_persons=args.all_persons)
    n_persons = data.shape[-1]
    if n_persons > 1:
        print(f"Форма данных: было {orig_shape} → для модели {data.shape} (предсказания по {n_persons} людям).")
    else:
        print(f"Форма данных: было {orig_shape} → для модели {data.shape}")

    if args.max_samples is not None:
        data = data[: args.max_samples]
        if labels is not None:
            labels = labels[: args.max_samples]

    model = STGCNWrapper(
        weights_path=args.weights,
        label_map_path=args.labels_file,
        device=args.device,
    )

    n = len(data)
    if args.all_persons and data.shape[-1] > 1:
        # Предсказание по каждому человеку: predictions[i] = [pred_p0, pred_p1, ...]
        predictions_per_person = []
        for i in range(n):
            preds_this = []
            for p in range(data.shape[-1]):
                sample = data[i : i + 1, :, :, :, p : p + 1]  # (1, 3, T, 18, 1)
                logits = model.predict_logits(sample)
                pred_cls = int(logits[0].argmax().item())
                preds_this.append(pred_cls)
            predictions_per_person.append(preds_this)

        print(f"Обработано сэмплов: {n}, людей в кадре: {data.shape[-1]}")
        if labels is not None:
            labels = np.asarray(labels)
            kinetics_range = (labels >= 0) & (labels < 400)
            if np.all(kinetics_range):
                # Правильно, если хотя бы один человек угадал класс
                correct = sum(
                    1 for i in range(n) if labels[i] in predictions_per_person[i]
                )
                acc = correct / n
                print(f"Accuracy (хотя бы один человек угадал): {acc:.4f} ({correct}/{n})")
            else:
                print("Метки не в диапазоне Kinetics 0..399 — accuracy не считается.")
            print("\nПервые 10: [человек0, человек1, ...] | true | labels")
            for i in range(min(10, n)):
                preds = predictions_per_person[i]
                lbls = [model.id2label.get(str(pid), "") for pid in preds]
                gt = labels[i]
                print(f"  {i}: {preds} | {gt} | {lbls}")
        else:
            print("Метки не заданы. Предсказания по людям (первые 10 сэмплов):")
            for i in range(min(10, n)):
                preds = predictions_per_person[i]
                lbls = [model.id2label.get(str(pid), "") for pid in preds]
                print(f"  {i}: {preds}  {lbls}")
    else:
        # Один человек (первый): собираем логиты для Top-1 и Top-5 как в оригинале
        result_frag = []
        for i in range(n):
            sample = data[i : i + 1]
            logits = model.predict_logits(sample)
            result_frag.append(logits.cpu().numpy())
        result = np.concatenate(result_frag, axis=0)  # (N, 400)

        print(f"Обработано сэмплов: {n}")
        if labels is not None:
            labels = np.asarray(labels)
            kinetics_range = (labels >= 0) & (labels < 400)
            if np.all(kinetics_range):
                # Top-K как в yysijie: rank = argsort (по возрастанию), top-k = последние k индексов
                rank = result.argsort(axis=1)
                hit_top1 = np.array([labels[i] in rank[i, -1:] for i in range(n)])
                hit_top5 = np.array([labels[i] in rank[i, -5:] for i in range(n)])
                acc1 = hit_top1.mean()
                acc5 = hit_top5.mean()
                print(f"Top1: {100 * acc1:.2f}% ({int(hit_top1.sum())}/{n})")
                print(f"Top5: {100 * acc5:.2f}% ({int(hit_top5.sum())}/{n})")
            else:
                print("Метки не в диапазоне Kinetics 0..399 — accuracy не считается.")
            predictions = result.argmax(axis=1)
            print("\nПервые 10: pred | true | label_pred")
            for i in range(min(10, n)):
                pid = int(predictions[i])
                lab = model.id2label.get(str(pid), "") if model.id2label else ""
                gt = labels[i]
                print(f"  {i}: {pid:3d} | {gt} | {lab}")
        else:
            predictions = result.argmax(axis=1)
            print("Метки не заданы (.pkl не передан). Top-1 предсказания:")
            for i in range(min(10, n)):
                pid = int(predictions[i])
                lab = model.id2label.get(str(pid), "") if model.id2label else ""
                print(f"  {i}: {pid:3d}  {lab}")


if __name__ == "__main__":
    main()
