import tensorflow as tf
import numpy as np

from models import Model
from common import average_gradients

class Trainer(object):
    def __init__(self, config, model):
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.loss = model.get_loss()
        self.summary = model.get_summary()
        self.global_step = model.get_global_step()

    def set_trainer(self, optimizer = 'Momentum', **argv):
        if optimizer == 'Momentum':
            self.learning_rate = tf.train.exponential_decay(
                self.config.init_lr,  # Base learning rate.
                self.global_step * self.config.batch_size,  # Current index into the dataset.
                argv['training_size'],  # Decay step.
                0.95,  # Decay rate.
                staircase=True)
            # Use simple momentum for the optimization.
            self.opt = tf.train.MomentumOptimizer(self.learning_rate, 0.9)
        elif optimizer == 'Adadelta':
            self.opt = tf.train.AdadeltaOptimizer(self.config.init_lr)
        else:
            raise ValueError('Invalid Optimizer')

        self.grads = self.opt.compute_gradients(self.loss, self.model.get_var_list())
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def step(self, sess, batch, get_summary=False):
        #assert isinstance(sess, tf.Session)
        feed_dict = self.model.get_feed_dict(batch, True)
        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.get_summary()
        self.models = models
        losses = []
        grads_list = []
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                loss = model.get_loss()
                grads = self.opt.compute_gradients(loss, var_list=self.var_list)
                losses.append(loss)
                grads_list.append(grads)

        self.loss = tf.add_n(losses)/len(losses)
        self.grads = average_gradients(grads_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def step(self, sess, batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))

        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
