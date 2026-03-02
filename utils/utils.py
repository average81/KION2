import yaml

#Открытие файла настроек в  yaml
def open_yaml(file):
    with open(file, 'r') as f:
        return yaml.safe_load(f)

#Сохранение yaml
def save_yaml(file, data):
    with open(file, 'w') as f:
        yaml.dump(data, f)