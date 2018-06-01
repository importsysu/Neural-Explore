from os.path import join, exists
import numpy as np
import tensorflow as tf
from nets.alexnet import alexnet_v2, alexnet_v2_arg_scope
from input import download_dataset

tf.app.flags.DEFINE_integer('batch_size', 60, "Batch size")
tf.app.flags.DEFINE_integer('num_iters', 10000, "Iteration count")
tf.app.flags.DEFINE_float('weight_decay', 0.1, 'Weight decay')
tf.app.flags.DEFINE_float('learning_rate', 0.03, 'Learning rate')
tf.app.flags.DEFINE_string('data_dir', './datasets/cifar-100', 'Dataset directory')
tf.app.flags.DEFINE_string('save_path', './log/alexnet_cifar100', 'Model parameter save path')
tf.app.flags.DEFINE_string('stage', 'train', 'Training stage')

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

def load_cifar(split_name):
    image_path = join(FLAGS.data_dir, split_name + '_images.npy')
    label_path = join(FLAGS.data_dir, split_name + '_fine_labels.npy')
    if not exists(image_path) or not exists(label_path):
        download_dataset('cifar-100')
    image_data, label_data = np.load(image_path), np.load(label_path)
    if FLAGS.stage == 'train':
        return tf.train.slice_input_producer([image_data, label_data], shuffle=True)
    else:
        return tf.train.slice_input_producer([image_data, label_data], shuffle=False, num_epochs=1)


def build_train_op(image_tensor, label_tensor, is_training):
    alexnet_argscope = alexnet_v2_arg_scope(weight_decay=FLAGS.weight_decay)
    global_step = tf.get_variable(name="global_step", shape=[], dtype=tf.int32, trainable=False)
    with slim.arg_scope(alexnet_argscope):
        logits, end_points = alexnet_v2(image_tensor, is_training=is_training, num_classes=100)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_tensor))
    accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), label_tensor),tf.int32))
    end_points['loss'], end_points['accuracy'] = loss, accuracy
    if is_training:
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
        return train_op, end_points
    else:
        return None, end_points


def train(sess, train_op, end_points):
    for step in range(FLAGS.num_iters):
        if (step + 1) % 100 == 0:
            _, loss_val = sess.run([train_op, end_points['loss']])
            print('loss:' + str(loss_val))
        else:
            train_op.run()


def test(sess, end_points):
    loss, accuracy = 0, 0
    for _ in range(FLAGS.num_iters):
        loss += sess.run(end_points['loss'])
    loss /= FLAGS.num_iters
    accuracy /= FLAGS.num_iters
    print('loss = {}, accuracy = {}'.format(loss, accuracy))


def main(_):
    image_tensor, label_tensor = load_cifar(FLAGS.stage)
    image_batch, label_batch = tf.train.batch([image_tensor, label_tensor], FLAGS.batch_size)
    image_batch = tf.image.resize_images(image_batch, [224, 224])
    label_batch = tf.cast(label_batch, tf.int32)
    is_training = FLAGS.stage in ['train']
    train_op, end_points = build_train_op(image_batch, label_batch, is_training)
    saver = tf.train.Saver(tf.trainable_variables())
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    if FLAGS.stage == 'train':
        train(sess, train_op, end_points)
        saver.save(sess, FLAGS.save_path)
    else:
        saver.restore(sess, FLAGS.save_path)
        test(sess, end_points)
    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
