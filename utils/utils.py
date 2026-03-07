import yaml
import numpy as np

#Открытие файла настроек в  yaml
def open_yaml(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)

#Сохранение yaml
def save_yaml(file, data):
    with open(file, 'w') as f:
        yaml.dump(data, f)

# Конвертер объектов в списки
def numpy_to_builtin(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: numpy_to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(numpy_to_builtin(v) for v in obj)
    return obj