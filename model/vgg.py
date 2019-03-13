"""
VggNet有多种版本，下面VggNetX指不同版本的VggNet模型类
"""
import tensorflow as tf


class VggNet:
    def conv_op(self, input_op, name, kw, kh, n_out, dw, dh, p):
        """
        构建VggNet的通用卷积操作, 激活函数统一采用ReLU
        :param input_op: 入参
        :param name: 命名空间
        :param kw: 卷积核宽度
        :param kh: 卷积核高度
        :param n_out: 卷积核数目
        :param dw: 横向步长
        :param dh: 纵向步长
        :param p: 参数列表(用于正则化惩罚)
        :return: tensor
        """
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            kernel = tf.get_variable("kernel", [kw, kh, n_in, n_out], dtype=tf.float32,
                                     initializer=tf.contrib.layers.xavier_initializer_conv2d())
            conv = tf.nn.conv2d(input_op, kernel, [1, dw, dh, 1], padding="SAME")
            bias = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[n_out]), name="bias")
            activation = tf.nn.relu(tf.nn.bias_add(conv, bias), name=scope)
            p += [conv, bias]
        return activation

    def pool_op(self, input_op, name, kw, kh, dw, dh):
        """
        构建VggNet的通用池化操作, 统一采用最大值池化
        :param input_op: 入参
        :param name: 命名空间
        :param kw: 池化宽度
        :param kh: 池化高度
        :param dw: 横向步长
        :param dh: 纵向步长
        :return: tensor
        """
        n_in = input_op.get_shape()[-1].value
        return tf.nn.max_pool(input_op, ksize=[1, kw, kh, 1], strides=[1, dw, dh, 1], padding="SAME", name=name)

    def fc_op(self, input_op, name, n_out, p):
        """
        构建VggNet的通用全连接操作, 激活函数统一采用ReLU
        :param input_op: 入参
        :param name: 命名空间
        :param n_out: 输出维度
        :param p: 参数列表(用于正则化惩罚)
        :return: tensor
        """
        n_in = input_op.get_shape()[-1].value

        with tf.name_scope(name) as scope:
            weight = tf.get_variable("weight", [n_in, n_out], dtype=tf.float32,
                                     initializer=tf.contrib.laters.xavier_initializer())
            bias = tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[n_out]), name="bias")
            activation = tf.nn.relu(tf.matmul(input_op, weight) + weight, name=scope)
            p += [weight, bias]
        return activation


class VggNetA(VggNet):
    def build_network(self, keep_prob=0.75, n_out=1000):
        params = []
        images = tf.placeholder(tf.float32, (None, 224, 244, 3))
        labels = tf.placeholder(tf.float32, (None, 1000))

        conv1_1 = self.conv_op(images, "conv1_1", 3, 3, 64, 1, 1, params)
        pool1 = self.pool_op(conv1_1, "pool1", 2, 2, 2, 2)

        conv2_1 = self.conv_op(pool1, "conv2_1", 3, 3, 128, 1, 1, params)
        pool2 = self.pool_op(conv2_1, "pool2", 2, 2, 2, 2)

        conv3_1 = self.conv_op(pool2, "conv3_1", 3, 3, 256, 1, 1, params)
        conv3_2 = self.conv_op(conv3_1, "conv3_2", 3, 3, 256, 1, 1, params)
        pool3 = self.pool_op(conv3_2, "pool3", 2, 2, 2, 2)

        conv4_1 = self.conv_op(pool3, "conv4_1", 3, 3, 512, 1, 1, params)
        conv4_2 = self.conv_op(conv4_1, "conv4_2", 3, 3, 512, 1, 1, params)
        pool4 = self.pool_op(conv4_2, "pool4", 2, 2, 2, 2)

        conv5_1 = self.conv_op(pool4, "conv5_1", 3, 3, 512, 1, 1, params)
        conv5_2 = self.conv_op(conv5_1, "conv5_2", 3, 3, 512, 1, 1, params)
        pool5 = self.pool_op(conv5_2, "pool5", 2, 2, 2, 2)

        # flatten
        shape = pool5.get_shape().as_list()
        fc_input = tf.reshape(pool5, (shape[1] * shape[2] * shape[3]))

        # full connection layer
        fc6 = self.fc_op(fc_input, "fc6", 4096, params)
        fc6_dropout = tf.nn.dropout(fc6, rate=1 - keep_prob, name="fc6_dropout")
        fc7 = self.fc_op(fc6_dropout, "fc7", 4096, params)
        fc7_dropout = tf.nn.dropout(fc7, rate=1 - keep_prob, name="fc7_dropout")
        fc8 = self.fc_op(fc7_dropout, "fc8", n_out, params)

        predict = tf.argmax(tf.nn.softmax(fc8), 1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8, labels=labels)
        error_rate = tf.reduce_mean(tf.cast(tf.not_equal(predict, tf.argmax(labels, 1)), tf.float32))

        return predict, error_rate, loss, images, labels, params


class VggNetALrn(VggNet):
    def build_network(self, keep_prob=0.75, n_out=1000):
        params = []
        images = tf.placeholder(tf.float32, (None, 224, 244, 3))
        labels = tf.placeholder(tf.float32, (None, 1000))

        conv1_1 = self.conv_op(images, "conv1_1", 3, 3, 64, 1, 1, params)
        lrn1 = tf.nn.lrn(conv1_1, 3, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
        pool1 = self.pool_op(lrn1, "pool1", 2, 2, 2, 2)

        conv2_1 = self.conv_op(pool1, "conv2_1", 3, 3, 128, 1, 1, params)
        pool2 = self.pool_op(conv2_1, "pool2", 2, 2, 2, 2)

        conv3_1 = self.conv_op(pool2, "conv3_1", 3, 3, 256, 1, 1, params)
        conv3_2 = self.conv_op(conv3_1, "conv3_2", 3, 3, 256, 1, 1, params)
        pool3 = self.pool_op(conv3_2, "pool3", 2, 2, 2, 2)

        conv4_1 = self.conv_op(pool3, "conv4_1", 3, 3, 512, 1, 1, params)
        conv4_2 = self.conv_op(conv4_1, "conv4_2", 3, 3, 512, 1, 1, params)
        pool4 = self.pool_op(conv4_2, "pool4", 2, 2, 2, 2)

        conv5_1 = self.conv_op(pool4, "conv5_1", 3, 3, 512, 1, 1, params)
        conv5_2 = self.conv_op(conv5_1, "conv5_2", 3, 3, 512, 1, 1, params)
        pool5 = self.pool_op(conv5_2, "pool5", 2, 2, 2, 2)

        # flatten
        shape = pool5.get_shape().as_list()
        fc_input = tf.reshape(pool5, (shape[1] * shape[2] * shape[3]))

        # full connection layer
        fc6 = self.fc_op(fc_input, "fc6", 4096, params)
        fc6_dropout = tf.nn.dropout(fc6, rate=1 - keep_prob, name="fc6_dropout")
        fc7 = self.fc_op(fc6_dropout, "fc7", 4096, params)
        fc7_dropout = tf.nn.dropout(fc7, rate=1 - keep_prob, name="fc7_dropout")
        fc8 = self.fc_op(fc7_dropout, "fc8", n_out, params)

        predict = tf.argmax(tf.nn.softmax(fc8), 1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8, labels=labels)
        error_rate = tf.reduce_mean(tf.cast(tf.not_equal(predict, tf.argmax(labels, 1)), tf.float32))

        return predict, error_rate, loss, images, labels, params


class VggNetB(VggNet):
    def build_network(self, keep_prob=0.75, n_out=1000):
        params = []
        images = tf.placeholder(tf.float32, (None, 224, 244, 3))
        labels = tf.placeholder(tf.float32, (None, 1000))

        conv1_1 = self.conv_op(images, "conv1_1", 3, 3, 64, 1, 1, params)
        conv1_2 = self.conv_op(conv1_1, "conv1_2", 3, 3, 64, 1, 1, params)
        pool1 = self.pool_op(conv1_2, "pool1", 2, 2, 2, 2)

        conv2_1 = self.conv_op(pool1, "conv2_1", 3, 3, 128, 1, 1, params)
        conv2_2 = self.conv_op(conv2_1, "conv2_2", 3, 3, 128, 1, 1, params)
        pool2 = self.pool_op(conv2_2, "pool2", 2, 2, 2, 2)

        conv3_1 = self.conv_op(pool2, "conv3_1", 3, 3, 256, 1, 1, params)
        conv3_2 = self.conv_op(conv3_1, "conv3_2", 3, 3, 256, 1, 1, params)
        pool3 = self.pool_op(conv3_2, "pool3", 2, 2, 2, 2)

        conv4_1 = self.conv_op(pool3, "conv4_1", 3, 3, 512, 1, 1, params)
        conv4_2 = self.conv_op(conv4_1, "conv4_2", 3, 3, 512, 1, 1, params)
        pool4 = self.pool_op(conv4_2, "pool4", 2, 2, 2, 2)

        conv5_1 = self.conv_op(pool4, "conv5_1", 3, 3, 512, 1, 1, params)
        conv5_2 = self.conv_op(conv5_1, "conv5_2", 3, 3, 512, 1, 1, params)
        pool5 = self.pool_op(conv5_2, "pool5", 2, 2, 2, 2)

        # flatten
        shape = pool5.get_shape().as_list()
        fc_input = tf.reshape(pool5, (shape[1] * shape[2] * shape[3]))

        # full connection layer
        fc6 = self.fc_op(fc_input, "fc6", 4096, params)
        fc6_dropout = tf.nn.dropout(fc6, rate=1 - keep_prob, name="fc6_dropout")
        fc7 = self.fc_op(fc6_dropout, "fc7", 4096, params)
        fc7_dropout = tf.nn.dropout(fc7, rate=1 - keep_prob, name="fc7_dropout")
        fc8 = self.fc_op(fc7_dropout, "fc8", n_out, params)

        predict = tf.argmax(tf.nn.softmax(fc8), 1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8, labels=labels)
        error_rate = tf.reduce_mean(tf.cast(tf.not_equal(predict, tf.argmax(labels, 1)), tf.float32))

        return predict, error_rate, loss, images, labels, params


class VggNetC(VggNet):
    def build_network(self, keep_prob=0.75, n_out=1000):
        params = []
        images = tf.placeholder(tf.float32, (None, 224, 244, 3))
        labels = tf.placeholder(tf.float32, (None, 1000))

        conv1_1 = self.conv_op(images, "conv1_1", 3, 3, 64, 1, 1, params)
        conv1_2 = self.conv_op(conv1_1, "conv1_2", 3, 3, 64, 1, 1, params)
        pool1 = self.pool_op(conv1_2, "pool1", 2, 2, 2, 2)

        conv2_1 = self.conv_op(pool1, "conv2_1", 3, 3, 128, 1, 1, params)
        conv2_2 = self.conv_op(conv2_1, "conv2_2", 3, 3, 128, 1, 1, params)
        pool2 = self.pool_op(conv2_2, "pool2", 2, 2, 2, 2)

        conv3_1 = self.conv_op(pool2, "conv3_1", 3, 3, 256, 1, 1, params)
        conv3_2 = self.conv_op(conv3_1, "conv3_2", 3, 3, 256, 1, 1, params)
        conv3_3 = self.conv_op(conv3_2, "conv3_3", 1, 1, 256, 1, 1, params)
        pool3 = self.pool_op(conv3_3, "pool3", 2, 2, 2, 2)

        conv4_1 = self.conv_op(pool3, "conv4_1", 3, 3, 512, 1, 1, params)
        conv4_2 = self.conv_op(conv4_1, "conv4_2", 3, 3, 512, 1, 1, params)
        conv4_3 = self.conv_op(conv4_2, "conv4_3", 1, 1, 512, 1, 1, params)
        pool4 = self.pool_op(conv4_3, "pool4", 2, 2, 2, 2)

        conv5_1 = self.conv_op(pool4, "conv5_1", 3, 3, 512, 1, 1, params)
        conv5_2 = self.conv_op(conv5_1, "conv5_2", 3, 3, 512, 1, 1, params)
        conv5_3 = self.conv_op(conv5_2, "conv5_3", 1, 1, 512, 1, 1, params)
        pool5 = self.pool_op(conv5_3, "pool5", 2, 2, 2, 2)

        # flatten
        shape = pool5.get_shape().as_list()
        fc_input = tf.reshape(pool5, (shape[1] * shape[2] * shape[3]))

        # full connection layer
        fc6 = self.fc_op(fc_input, "fc6", 4096, params)
        fc6_dropout = tf.nn.dropout(fc6, rate=1 - keep_prob, name="fc6_dropout")
        fc7 = self.fc_op(fc6_dropout, "fc7", 4096, params)
        fc7_dropout = tf.nn.dropout(fc7, rate=1 - keep_prob, name="fc7_dropout")
        fc8 = self.fc_op(fc7_dropout, "fc8", n_out, params)

        predict = tf.argmax(tf.nn.softmax(fc8), 1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8, labels=labels)
        error_rate = tf.reduce_mean(tf.cast(tf.not_equal(predict, tf.argmax(labels, 1)), tf.float32))

        return predict, error_rate, loss, images, labels, params


class VggNetD(VggNet):
    def build_network(self, keep_prob=0.75, n_out=1000):
        params = []
        images = tf.placeholder(tf.float32, (None, 224, 244, 3))
        labels = tf.placeholder(tf.float32, (None, 1000))

        conv1_1 = self.conv_op(images, "conv1_1", 3, 3, 64, 1, 1, params)
        conv1_2 = self.conv_op(conv1_1, "conv1_2", 3, 3, 64, 1, 1, params)
        pool1 = self.pool_op(conv1_2, "pool1", 2, 2, 2, 2)

        conv2_1 = self.conv_op(pool1, "conv2_1", 3, 3, 128, 1, 1, params)
        conv2_2 = self.conv_op(conv2_1, "conv2_2", 3, 3, 128, 1, 1, params)
        pool2 = self.pool_op(conv2_2, "pool2", 2, 2, 2, 2)

        conv3_1 = self.conv_op(pool2, "conv3_1", 3, 3, 256, 1, 1, params)
        conv3_2 = self.conv_op(conv3_1, "conv3_2", 3, 3, 256, 1, 1, params)
        conv3_3 = self.conv_op(conv3_2, "conv3_3", 3, 3, 256, 1, 1, params)
        pool3 = self.pool_op(conv3_3, "pool3", 2, 2, 2, 2)

        conv4_1 = self.conv_op(pool3, "conv4_1", 3, 3, 512, 1, 1, params)
        conv4_2 = self.conv_op(conv4_1, "conv4_2", 3, 3, 512, 1, 1, params)
        conv4_3 = self.conv_op(conv4_2, "conv4_3", 3, 3, 512, 1, 1, params)
        pool4 = self.pool_op(conv4_3, "pool4", 2, 2, 2, 2)

        conv5_1 = self.conv_op(pool4, "conv5_1", 3, 3, 512, 1, 1, params)
        conv5_2 = self.conv_op(conv5_1, "conv5_2", 3, 3, 512, 1, 1, params)
        conv5_3 = self.conv_op(conv5_2, "conv5_3", 3, 3, 512, 1, 1, params)
        pool5 = self.pool_op(conv5_3, "pool5", 2, 2, 2, 2)

        # flatten
        shape = pool5.get_shape().as_list()
        fc_input = tf.reshape(pool5, (shape[1] * shape[2] * shape[3]))

        # full connection layer
        fc6 = self.fc_op(fc_input, "fc6", 4096, params)
        fc6_dropout = tf.nn.dropout(fc6, rate=1 - keep_prob, name="fc6_dropout")
        fc7 = self.fc_op(fc6_dropout, "fc7", 4096, params)
        fc7_dropout = tf.nn.dropout(fc7, rate=1 - keep_prob, name="fc7_dropout")
        fc8 = self.fc_op(fc7_dropout, "fc8", n_out, params)

        predict = tf.argmax(tf.nn.softmax(fc8), 1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8, labels=labels)
        error_rate = tf.reduce_mean(tf.cast(tf.not_equal(predict, tf.argmax(labels, 1)), tf.float32))

        return predict, error_rate, loss, images, labels, params


class VggNetE(VggNet):
    def build_network(self, keep_prob=0.75, n_out=1000):
        params = []
        images = tf.placeholder(tf.float32, (None, 224, 244, 3))
        labels = tf.placeholder(tf.float32, (None, 1000))

        conv1_1 = self.conv_op(images, "conv1_1", 3, 3, 64, 1, 1, params)
        conv1_2 = self.conv_op(conv1_1, "conv1_2", 3, 3, 64, 1, 1, params)
        pool1 = self.pool_op(conv1_2, "pool1", 2, 2, 2, 2)

        conv2_1 = self.conv_op(pool1, "conv2_1", 3, 3, 128, 1, 1, params)
        conv2_2 = self.conv_op(conv2_1, "conv2_2", 3, 3, 128, 1, 1, params)
        pool2 = self.pool_op(conv2_2, "pool2", 2, 2, 2, 2)

        conv3_1 = self.conv_op(pool2, "conv3_1", 3, 3, 256, 1, 1, params)
        conv3_2 = self.conv_op(conv3_1, "conv3_2", 3, 3, 256, 1, 1, params)
        conv3_3 = self.conv_op(conv3_2, "conv3_3", 3, 3, 256, 1, 1, params)
        conv3_4 = self.conv_op(conv3_3, "conv3_4", 3, 3, 256, 1, 1, params)
        pool3 = self.pool_op(conv3_4, "pool3", 2, 2, 2, 2)

        conv4_1 = self.conv_op(pool3, "conv4_1", 3, 3, 512, 1, 1, params)
        conv4_2 = self.conv_op(conv4_1, "conv4_2", 3, 3, 512, 1, 1, params)
        conv4_3 = self.conv_op(conv4_2, "conv4_3", 3, 3, 512, 1, 1, params)
        conv4_4 = self.conv_op(conv4_3, "conv4_4", 3, 3, 512, 1, 1, params)
        pool4 = self.pool_op(conv4_4, "pool4", 2, 2, 2, 2)

        conv5_1 = self.conv_op(pool4, "conv5_1", 3, 3, 512, 1, 1, params)
        conv5_2 = self.conv_op(conv5_1, "conv5_2", 3, 3, 512, 1, 1, params)
        conv5_3 = self.conv_op(conv5_2, "conv5_3", 3, 3, 512, 1, 1, params)
        conv5_4 = self.conv_op(conv5_3, "conv5_4", 3, 3, 512, 1, 1, params)
        pool5 = self.pool_op(conv5_4, "pool5", 2, 2, 2, 2)

        # flatten
        shape = pool5.get_shape().as_list()
        fc_input = tf.reshape(pool5, (shape[1] * shape[2] * shape[3]))

        # full connection layer
        fc6 = self.fc_op(fc_input, "fc6", 4096, params)
        fc6_dropout = tf.nn.dropout(fc6, rate=1 - keep_prob, name="fc6_dropout")
        fc7 = self.fc_op(fc6_dropout, "fc7", 4096, params)
        fc7_dropout = tf.nn.dropout(fc7, rate=1 - keep_prob, name="fc7_dropout")
        fc8 = self.fc_op(fc7_dropout, "fc8", n_out, params)

        predict = tf.argmax(tf.nn.softmax(fc8), 1)
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8, labels=labels)
        error_rate = tf.reduce_mean(tf.cast(tf.not_equal(predict, tf.argmax(labels, 1)), tf.float32))

        return predict, error_rate, loss, images, labels, params
