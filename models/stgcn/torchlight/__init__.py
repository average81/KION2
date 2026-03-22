# Без этого файла каталог torchlight/ рядом с main.py становится namespace-пакетом
# (PEP 420), и «import torchlight» не видит import_class из вложенного пакета.
from .torchlight import *
