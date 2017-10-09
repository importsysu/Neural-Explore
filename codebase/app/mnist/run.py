import numpy as np
import tensorflow as tf

from common import Config
from dataset import Dataset
from models.tutorial_cnn import TutorialCNN
from train.trainer import Trainer
from .cli import flags

def _eval(sess, model, eval_data, config):
    assert isinstance(eval_data, Dataset)
    sum_eq, sum_loss = 0.0, 0.0
    cnt = 0
    for batch in eval_data.get_batches(config.batch_size):
        cnt += 1
        feed_dict = model.get_feed_dict(batch, True)
        loss, predictions = sess.run([model.loss, model.prediction], feed_dict)
        sum_loss += loss
        sum_eq += np.sum(np.equal(predictions, batch['label']))
    avg_loss = sum_loss / cnt
    print('loss = {}, error rate = {}'.format(avg_loss, 1. - sum_eq / eval_data.size))

def _train(config):
    print('Building Model')
    model = TutorialCNN(config)

    print('Loading dataset')
    train_data = Dataset("mnist", "train")
    validation_data = train_data.split_validation(config.validation_size)

    print('Init trainer')
    trainer = Trainer(config, model)

    with tf.Session() as sess:
        print('Start session')
        tf.global_variables_initializer().run()
        print('Initialized!')

        for batch in train_data.get_batches(config.batch_size, config.num_steps):
            trainer.step(sess, batch)
            global_step = sess.run(trainer.global_step)
            if global_step % config.eval_frequency:
                _eval(sess, model, validation_data, config)

def init_config(config):
    config.update(IMAGE_SIZE=28)
    config.update(NUM_CHANNELS=1)
    config.update(SEED=66478)
    config.update(NUM_LABELS=10)

def main():
    flags.FLAGS._parse_flags()
    config = Config(**flags.FLAGS.__flags)
    init_config(config)
    _train(config)
