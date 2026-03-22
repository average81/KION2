# Данные для ST-GCN (локально)

В репозитории **не хранятся** большие файлы: `*.skeleton`, `*.npy` в этой ветке игнорируются через корневой `.gitignore`.

Ожидаемая структура (после распаковки NTU и сборки скриптом `tools/build_ntu_xsub_2d_25_18.py`):

| Путь | Содержимое |
|------|------------|
| `NTU-RGB-D/nturgb+d_skeletons/*.skeleton` | исходные 2D color скелеты NTU RGB+D |
| `NTU-RGB-D-2D-from-color/xsub_*_data18.npy`, `*_data25.npy`, `*_label.pkl` | выход сборки для обучения (пути в `config/st_gcn/ntu-xsub-kinetics-2d/*.yaml`) |

Корень данных в скрипте сборки: `app/stgcn/data/` (см. `tools/build_ntu_xsub_2d_25_18.py`).

Чекпоинты и логи обучения пишутся в `app/stgcn/work_dir/` — тоже в `.gitignore`.
