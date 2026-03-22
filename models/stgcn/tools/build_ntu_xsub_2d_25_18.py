"""
Сборка xsub train/val .npy/.pkl из NTU .skeleton (2D color).

Пути к данным: app/stgcn/data/NTU-RGB-D/...
Запуск из корня репозитория:
  python app/stgcn/tools/build_ntu_xsub_2d_25_18.py
"""
import pickle
from pathlib import Path
import sys

# Каталог app/stgcn — в sys.path, чтобы работал пакет `tools`
STGCN_ROOT = Path(__file__).resolve().parents[1]
if str(STGCN_ROOT) not in sys.path:
    sys.path.insert(0, str(STGCN_ROOT))

from tools.gen_ntu_2d_color_to_25_18 import build_ntu_2d_25_18

# Данные NTU лежат под app/stgcn/data/ (не от cwd)
DATA_ROOT = STGCN_ROOT / "data"

def load_names_labels(pkl_path):
    with open(pkl_path, "rb") as f:
        sample_name, sample_label = pickle.load(f)
    # В NTU pkls имена часто с суффиксом «.skeleton», у Path(...).stem — без него,
    # как в name_to_path и в build_ntu_2d_25_18 (skel_path.stem).
    names = [Path(n).stem for n in sample_name]
    label_dict = {Path(n).stem: int(l) for n, l in zip(sample_name, sample_label)}
    return names, label_dict

def main():
    root_skel = DATA_ROOT / "NTU-RGB-D" / "nturgb+d_skeletons"

    out_root = DATA_ROOT / "NTU-RGB-D-2D-from-color"

    # 1) сплиты xsub: берём имена/лейблы из уже существующих pkl,
    # а не из NTU train_label.pkl/val_label.pkl (их может не быть в репозитории).
    train_names, train_label_dict = load_names_labels(
        out_root / "xsub_train_label.pkl"
    )
    val_names, val_label_dict = load_names_labels(
        out_root / "xsub_val_label.pkl"
    )

    # общий label_dict (train+val)
    label_dict = {}
    label_dict.update(train_label_dict)
    label_dict.update(val_label_dict)

    # 2) собираем пути к skeleton-файлам по именам
    name_to_path = {p.stem: p for p in root_skel.glob("*.skeleton")}

    train_paths = [name_to_path[n] for n in train_names if n in name_to_path]
    val_paths = [name_to_path[n] for n in val_names if n in name_to_path]

    print(
        f"Найдено скелетонов: train {len(train_paths)}/{len(train_names)}, "
        f"val {len(val_paths)}/{len(val_names)}"
    )

    out_root.mkdir(parents=True, exist_ok=True)

    # train
    build_ntu_2d_25_18(
        train_paths,
        label_dict,
        out_data25_path=out_root / "xsub_train_data25.npy",
        out_data18_path=out_root / "xsub_train_data18.npy",
        out_label_path=out_root / "xsub_train_label.pkl",
    )

    # val
    build_ntu_2d_25_18(
        val_paths,
        label_dict,
        out_data25_path=out_root / "xsub_val_data25.npy",
        out_data18_path=out_root / "xsub_val_data18.npy",
        out_label_path=out_root / "xsub_val_label.pkl",
    )

if __name__ == "__main__":
    main()