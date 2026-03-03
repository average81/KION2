# KION2

Подбор и обучение моделей, которые по позам (pose estimation) актёров в видео определяют их действия на предложенном датасете.

Установите зависимости:
   
   **Установите PyTorch с нужной версией CUDA:**
   
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
   - Сначала установите PyTorch, затем остальные зависимости
   - Для актуальной информации о совместимых версиях используйте официальный сайт: https://pytorch.org/get-started/locally/


```bash
pip install -r requirements.txt
```

Для работы необходим ffmpeg
Установка Ubuntu Linux: 

```
sudo apt update
sudo apt install ffmpeg
```
Установка Windows10/11: 

```
winget install ffmpeg
```


Основные варианты для pose estimation

MediaPipe / BlazePose (Google)
Что это: лёгкая модель позы тела (есть варианты full / lite).
Плюсы: очень быстро, хорошо для реального времени, есть готовые пайплайны, простое API.
Минусы: меньше точность, чем тяжёлые SOTA‑модели; закрытая экосистема, но бесплатная.

MoveNet (Google, через TensorFlow / TF Hub)
Что это: очень быстрая и достаточно точная 2D‑поза.
Плюсы: отлично подходит для видео в реальном времени, простая интеграция в Python.
Минусы: по качеству суставов иногда уступает тяжёлым моделям типа HRNet.

OpenPose
Что это: классический фреймворк для позы тела + рук + лица.
Плюсы: много примеров, хорошее качество, много точек.
Минусы: медленный, тяжёлый, запуск сложнее; может быть оверхед для хакатона.

AlphaPose
Что это: высокоточная multi-person pose estimation.
Плюсы: хорошее качество, много туториалов.
Минусы: тяжелее в настройке, медленнее, чем MediaPipe/MoveNet.

MMPose (из MMDetection / OpenMMLab)
Что это: огромный зоопарк моделей (HRNet, ResNet‑based, и т.д.).
Плюсы: можно выбрать баланс скорость/точность, единый API, много готовых конфигов.
Минусы: порог входа выше, нужно разбираться с конфигами и экосистемой.

Detectron2 (Keypoint R-CNN)
Что это: general‑purpose фреймворк от Meta, есть keypoint‑модели.
Плюсы: мощно и гибко.
Минусы: тяжело для быстрых экспериментов, сложность настройки.

YOLO‑pose (YOLOv7‑pose, YOLOv8‑pose)
Что это: модели, совмещающие детекцию человека и суставов.
Плюсы: быстро, удобно, если нужно сразу и человек, и ключевые точки.
Минусы: экосистема не такая устоявшаяся, как у OpenPose/MediaPipe.