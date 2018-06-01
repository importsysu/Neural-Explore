import tensorflow as tf
from nets.lenet import lenet, lenet_arg_scope
from input import load_mnist

tf.app.flags.DEFINE_integer('batch_size', 60, "Batch size")
tf.app.flags.DEFINE_integer('num_iters', 10000, "Iteration count")
tf.app.flags.DEFINE_float('weight_decay', 0.1, 'Weight decay')
tf.app.flags.DEFINE_float('learning_rate', 0.03, 'Learning rate')

FLAGS = tf.app.flags.FLAGS

slim = tf.contrib.slim

def build_train_op(image_tensor, label_tensor):
    lenet_argscope = lenet_arg_scope(weight_decay=FLAGS.weight_decay) 
    with slim.arg_scope(lenet_argscope):
        logits, end_points = lenet(image_tensor, is_training=True)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_tensor))
    end_points['loss'] = loss
    optimizer = tf.train.AdadeltaOptimizer(learning_rate=FLAGS.learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op, end_points

def train(train_op, end_points):
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for step in range(FLAGS.num_iters):
        if (step + 1) % 100 == 0:
            loss_val = sess.run([train_op, end_points['loss']])
            print('loss:' + str(loss_val))
        else:
            train_op.run()
    coord.request_stop()
    coord.join(threads)

def main(_):
    image_tensor, label_tensor = load_mnist('train')
    image_batch, label_batch = tf.train.batch([image_tensor, label_tensor], FLAGS.batch_size)
    image_batch = tf.expand_dims(image_batch, -1)
    label_batch = tf.cast(label_batch, tf.int32)
    train_op, end_points = build_train_op(image_batch, label_batch)
    train(train_op, end_points)

if __name__ == '__main__':
    tf.app.run()
