#!/usr/bin/env python
import argparse
import sys

# torchlight
import torchlight
from torchlight import import_class

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')

    # Доступен только recognition (обучение/тест). Демо-процессоры из upstream ST-GCN
    # не подключены — при необходимости добавьте processor/demo_*.py и зарегистрируйте здесь.
    processors = {
        'recognition': import_class('processor.recognition.REC_Processor'),
    }

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[2:])

    p.start()
