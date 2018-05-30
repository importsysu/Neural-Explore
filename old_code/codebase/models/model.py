import tensorflow as tf
import numpy as np

class Model(object):

    def __init__(self, config, scope = "Graph"):
        raise NotImplementedError

    def _build_forward(self):
        raise NotImplementedError

    def _build_loss(self):
        raise NotImplementedError

    def get_loss(self):
        raise NotImplementedError

    def get_global_step(self):
        raise NotImplementedError

    def get_var_list(self):
        raise NotImplementedError

    def get_feed_dict(self):
        raise NotImplementedError

    def get_summary(self):
        pass
