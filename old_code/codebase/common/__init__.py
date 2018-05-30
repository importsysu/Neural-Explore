from common.graph_handler import GraphHandler
from common.tensorflow import *

class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)
    def update(self, **entries):
        self.__dict__.update(entries)
