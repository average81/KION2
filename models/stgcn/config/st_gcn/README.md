# Конфигурации ST-GCN

Исходная идея и структура конфигов — из проекта **[yysijie/st-gcn](https://github.com/yysijie/st-gcn)** (см. корневой `README.md` репозитория KION2).

Здесь — yaml для **обучения и теста** модели **ST-GCN**: по последовательности скелетов (ключевых точек) предсказывается **класс действия**. Это та же модель, что используется в конце пайплайна «видео → позы → действие» из корневого README; конфиги задают датасет (NTU / Kinetics), число классов, пути к `.npy`/`.pkl` и каталог с чекпоинтами.

## Актуальные (редактируйте их под свой пайплайн)

| Каталог | Назначение |
|---------|------------|
| **`ntu-xsub-kinetics-2d/`** | NTU RGB+D 2D (color), формат как Kinetics `(N,3,T,18,2)` — основной сценарий проекта. |
| **`kinetics-skeleton/`** | Kinetics-Skeleton, 400 классов (предобученные веса Kinetics). |

Запуск из каталога `models/stgcn`:

```bash
python main.py recognition -c config/st_gcn/ntu-xsub-kinetics-2d/train.yaml
python main.py recognition -c config/st_gcn/ntu-xsub-kinetics-2d/test.yaml --weights path/to/epochNN_model.pt
```

---

## `upstream/` — конфиги из исходного ST-GCN (резерв)

Не используются в текущем пайплайне; оставлены для справки (другие сплиты NTU, 2D-варианты, twostream).

| Подпапка | Описание |
|----------|-----------|
| `ntu-xsub/`, `ntu-xview/` | Классические сплиты NTU (3D / иной формат фидера в upstream). |
| `ntu-xsub-2d/`, `ntu-xview-2d/` | Варианты 2D до появления kinetics-формата; см. README внутри. |
| `st_gcn.twostream/` | Двухпоточная модель `net.st_gcn_twostream.Model`. |

Пути в командах: `config/st_gcn/upstream/<подпапка>/...`.
