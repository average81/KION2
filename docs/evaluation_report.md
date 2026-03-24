# Результаты тестирования моделей распознавания поз и действий по последовательности поз.

Визуализация работы проекта с видеофайлом выглядит примерно так:

![Пример работы](potter.gif)  

Метрики качества разных комбинаций моделей в рамках проекта (суммарные метрики):

| модель распознавания поз | модель распознавания действий | accuracy | f1 | precision | recall | Время обработки фильма 2ч FullHD 
|---|---|---|---|---|---|----------------------------------|
| YOLOv8-Pose-N | STGCN_model_kinetics |  |  |  |  |                                  |
| YOLOv8-Pose-N | STGCN_model_rgbd |  |  |  |  |                                  |
| YOLOv8-Pose-N | Conv3dNet |  |  |  |  |                                  |
| YOLOv8-Pose-N | LSTMSkeletonNet |  |  |  |  |                                  |
| YOLOv8-Pose-S | STGCN_model_kinetics |  |  |  |  |                                  |
| YOLOv8-Pose-S | STGCN_model_rgbd |  |  |  |  |                                  |
| YOLOv8-Pose-S | Conv3dNet |  |  |  |  |                                  |
| YOLOv8-Pose-S | LSTMSkeletonNet |  |  |  |  |                                  |
| YOLOv26-Pose-N | STGCN_model_kinetics |  |  |  |  | 33 минуты                        |
| YOLOv26-Pose-N | STGCN_model_rgbd |  |  |  |  |                                  |
| YOLOv26-Pose-N | Conv3dNet |  |  |  |  |                                  |
| YOLOv26-Pose-N | LSTMSkeletonNet |  |  |  |  |                                  |

## Анализ полученных метрик.

