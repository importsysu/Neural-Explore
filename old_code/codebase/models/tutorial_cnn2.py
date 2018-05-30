import tensorflow as tf

from models.model import Model

from common.layers import conv

class TutorialCNN(Model):
    """Implementation of a simple CNN."""

    def __init__(self, config, scope = "MnistGraph"):
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)
        # Define forward inputs here
        #batch_size remains None here
        if config.use_placeholder:
            self.x = tf.placeholder(tf.float32, shape=(None, config.IMAGE_SIZE, config.IMAGE_SIZE, config.NUM_CHANNELS))
            self.y = tf.placeholder(tf.int64, shape=(None,))
            self.is_train = tf.placeholder('bool', [], 'is_train')
        else:
            raise ValueError('placeholders are required currently.')

        with tf.variable_scope(scope):
            self._build_forward()
            self._build_loss()
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)

    def _build_forward(self):
        self.conv1 = tf.layers.conv2d(self.x, 32, [5, 5], scope='conv1')

        with tf.variable_scope('conv1') as scope:
            conv1_weights = tf.Variable(
                tf.truncated_normal([5, 5, self.config.NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                                    stddev=0.1,
                                    seed=self.config.SEED))
            conv1_biases = tf.Variable(tf.zeros([32]))
            self.debug_var = conv1_weights
            self.conv1 = tf.nn.conv2d(self.x,
                                conv1_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            self.relu1 = tf.nn.relu(tf.nn.bias_add(self.conv1, conv1_biases))
            self.pool1 = tf.nn.max_pool(self.relu1,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
        with tf.variable_scope('conv2') as scope:
            conv2_weights = tf.Variable(
                tf.truncated_normal([5, 5, 32, 64],
                                    stddev=0.1,
                                    seed=self.config.SEED))
            conv2_biases = tf.Variable(tf.constant(0.1, shape=[64]))
            self.conv2 = tf.nn.conv2d(self.pool1,
                                conv2_weights,
                                strides=[1, 1, 1, 1],
                                padding='SAME')
            self.relu2 = tf.nn.relu(tf.nn.bias_add(self.conv2, conv2_biases))
            self.pool2 = tf.nn.max_pool(self.relu2,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME')
        #pool_shape = self.pool2.get_shape().as_list()
        #flatterned = tf.reshape(self.pool2, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
        flat_size = self.pool2.shape[1] * self.pool2.shape[2] * self.pool2.shape[3]
        flatterned = tf.reshape(tensor=self.pool2, shape=(-1, flat_size._value))
        with tf.variable_scope('fc1') as scope:
            fc1_weights = tf.Variable(  # fully connected, depth 512.
                tf.truncated_normal([flatterned.shape[1]._value, 512],
                                    stddev=0.1,
                                    seed=self.config.SEED))
            #                    [self.config.IMAGE_SIZE // 4 * self.config.IMAGE_SIZE // 4 * 64, 512],
            fc1_biases = tf.Variable(tf.constant(0.1, shape=[512]))
            tf.add_to_collection("reg_vars", fc1_weights)
            tf.add_to_collection("reg_vars", fc1_biases)
            _fc1 = tf.nn.relu(tf.matmul(flatterned, fc1_weights) + fc1_biases)
            self.fc1 = tf.cond(self.is_train, lambda: tf.nn.dropout(_fc1, 0.5, seed=self.config.SEED), lambda: _fc1)
        with tf.variable_scope('fc2') as scope:
            fc2_weights = tf.Variable(
                tf.truncated_normal([512, self.config.NUM_LABELS],
                                    stddev=0.1,
                                    seed=self.config.SEED))
            fc2_biases = tf.Variable(tf.constant(0.1, shape=[self.config.NUM_LABELS]))
            tf.add_to_collection("reg_vars", fc2_weights)
            tf.add_to_collection("reg_vars", fc2_biases)
            self.fc2 = tf.matmul(self.fc1, fc2_weights) + fc2_biases
        self.logits = self.fc2
        # Get outputs
        self.prob = tf.nn.softmax(self.fc2)
        self.prediction = tf.argmax(self.prob, axis=1)
        correct_prediction = tf.equal(self.prediction, self.y)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def _build_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        self.loss += self.config.wd * tf.reduce_sum([tf.nn.l2_loss(var) for var in tf.get_collection("reg_vars")])

    def get_summary(self):
        return None

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_var_list(self):
        return self.var_list

    def get_feed_dict(self, batch, is_train, supervised=True):
        feed_dict = {}
        feed_dict[self.x] = batch['data']
        feed_dict[self.is_train] = is_train
        if supervised:
            feed_dict[self.y] = batch['label']
        return feed_dict
