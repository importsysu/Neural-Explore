import sys
import time
import numpy
import tensorflow as tf

from dataset import Dataset
from models import TutorialCNN
from train.trainer import Trainer
from app.mnist.cli import flags
from common import Config


def eval_error_in_dataset(dataset, model, sess, config):
    """Get all predictions for a dataset by running it in small batches."""
    diff_cnt = 0
    for batch in dataset.get_batches(config.batch_size, sample_type='once'):
        feed_dict = model.get_feed_dict(batch, is_train=False, supervised=False)
        prediction = sess.run(model.prediction, feed_dict=feed_dict)
        diff_cnt += numpy.sum(prediction != batch['label'])
    return 100.0 * diff_cnt / dataset.size


def _train(config):
    VALIDATION_SIZE = 5000  # Size of the validation set.
    BATCH_SIZE = 64

    train_dataset = Dataset("mnist", 'train')
    test_dataset = Dataset("mnist", 'test')
    validation_dataset = train_dataset.split_validation(VALIDATION_SIZE)

    model = TutorialCNN(config)

    trainer = Trainer(config, model)
    trainer.set_trainer(optimizer = 'Momentum', training_size = train_dataset.size)

    # Create a local session to run the training.
    start_time = time.time()
    with tf.Session() as sess:
        # Run all the initializers to prepare the trainable parameters.
        tf.global_variables_initializer().run()
        print('Initialized!')

        for batch in train_dataset.get_batches(BATCH_SIZE, num_epoches=config.num_epochs):
            step = sess.run(trainer.global_step)
            feed_dict = model.get_feed_dict(batch, is_train=True, supervised=True)
            sess.run(trainer.train_op, feed_dict=feed_dict)
            if step % config.eval_period == 0:
                l, lr, acc = sess.run([model.loss, trainer.learning_rate, model.accuracy], feed_dict=feed_dict)
                elapsed_time = time.time() - start_time
                start_time = time.time()
                print('Step %d, %.1f ms' % (step, 1000 * elapsed_time / config.eval_period))
                print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
                print('Minibatch error: %.1f%%' % (100. - 100. * acc, ))
                print('Validation error: %.1f%%' % eval_error_in_dataset(validation_dataset, model, sess, config))
                sys.stdout.flush()

        test_error = eval_error_in_dataset(test_dataset, model, sess, config)
        print('Test error: %.1f%%' % test_error)


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
