import pickle
import tensorflow as tf
import numpy as np
from sklearn import preprocessing

batch_size = 32
num_batches = 100


def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())


class AlexNet:
    def __init__(self, learning_rate=1e-3, keep_prob=1):
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob

    def build_network(self):
        parameters = []
        images = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="images")
        labels = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="labels")

        # first convolution layer, filter_size: 11x11, stride: 4, filter_num: 64
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 96], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)
            # print_activations(conv1)
            parameters += [kernel, biases]

        # LRN layer
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        # first max pool layer, filter_size: 3x3, stride: 2
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        # print_activations(pool1)

        # second convolution layer, filter_size: 5x5, stride: 1, filter_num: 192
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]
            # print_activations(conv2)

        # LRN layer
        lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
        # second max pool layer, filter_size: 3x3, stride: 2
        pool2 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        # print_activations(pool2)

        # third convolution layer, filter size: 3x3, stride: 1, filter_num: 384
        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]
            # print_activations(conv3)

        # forth convolution layer, filter size: 3x3, stride: 1, filter_num: 384
        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]
            # print_activations(conv4)

        # fifth convolution layer, filter size: 3x3, stride: 1, filter_num: 384
        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope)
            parameters += [kernel, biases]
            # print_activations(conv5)

        # third max pool layer, filter_size: 3x3, stride: 2
        pool3 = tf.nn.max_pool(lrn2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool3')
        # print_activations(pool3)

        # flatten
        shape = pool3.get_shape().as_list()
        fc_input = tf.reshape(pool3, [-1, shape[1] * shape[2] * shape[3]], name='flatten')
        # print_activations(fc_input)

        # first full connection layer
        with tf.name_scope("fc1") as scope:
            weights = tf.Variable(
                tf.truncated_normal([fc_input.get_shape().as_list()[1], 4096], dtype=tf.float32, stddev=1e-1),
                trainable=True, name="weights")
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name="biases")
            fc1 = tf.nn.relu(tf.matmul(fc_input, weights) + biases, name=scope)
            dropout1 = tf.nn.dropout(fc1, rate=1 - self.keep_prob)
            parameters += [weights, biases]
            # print_activations(dropout1)

        # second full connection layer
        with tf.name_scope("fc2") as scope:
            weights = tf.Variable(tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1),
                                  trainable=True, name="weights")
            biases = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name="biases")
            fc2 = tf.nn.relu(tf.matmul(dropout1, weights) + biases, name=scope)
            dropout2 = tf.nn.dropout(fc2, rate=1 - self.keep_prob)
            parameters += [weights, biases]
            # print_activations(dropout2)

        # third full connection layer
        with tf.name_scope("fc3") as scope:
            weights = tf.Variable(tf.truncated_normal([4096, 1000], dtype=tf.float32, stddev=1e-1),
                                  trainable=True, name="weights")
            biases = tf.Variable(tf.constant(0.0, shape=[1000], dtype=tf.float32), trainable=True, name="biases")
            fc3 = tf.nn.bias_add(tf.matmul(dropout2, weights), biases, name=scope)
            dropout3 = tf.nn.dropout(fc3, rate=1 - self.keep_prob)
            parameters += [weights, biases]
            # print_activations(dropout3)

        # output of full connection layer
        with tf.name_scope("output") as scope:
            weights = tf.Variable(tf.truncated_normal([1000, 10], stddev=1e-1), dtype=tf.float32, name="weights")
            biases = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name="biases")
            predict = tf.nn.softmax(tf.matmul(dropout3, weights) + biases, name=scope)
            parameters += [weights, biases]
            # print_activations(predict)

        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=tf.matmul(dropout3, weights) + biases, labels=labels)
        optm = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

        error_rate = tf.reduce_mean(tf.cast(tf.not_equal(tf.argmax(predict, 1), tf.argmax(labels, 1)), tf.float32))

        return predict, error_rate, loss, optm, parameters, images, labels


# with open("cifar-10-batches-py/test_batch", 'rb') as fo:
#     data = pickle.load(fo, encoding='latin1')
#
# print(data.keys())

images = np.random.uniform(0, 255, [batch_size, 224, 224, 3])
labels = np.random.random_integers(0, 9, [batch_size, 1])

encoder = preprocessing.LabelBinarizer()
encoder.fit(np.arange(10).reshape((10, 1)))
oneHotLabels = encoder.transform(labels)

net = AlexNet(1e-8)
predict, error_rate, loss, optm, parameters, images_placeholder, labels_placeholder = net.build_network()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    feeds = {images_placeholder: images, labels_placeholder: oneHotLabels}
    err = 10000000000
    epoch = 0
    while err > 1e-2:
        epoch += 1
        print("The epoch %d:" % epoch)
        err, loss_val, predict_result = sess.run([error_rate, loss, predict], feed_dict=feeds)
        # print("predict_result：", predict_result)
        print("error_rate： %f" % err)
        print("loss：", loss_val)

        if err > 1e-2:
            sess.run(optm, feed_dict=feeds)
    print("train finished")
