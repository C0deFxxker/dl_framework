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
    pass


class VggNetALrn(VggNet):
    pass


class VggNetB(VggNet):
    pass


class VggNetC(VggNet):
    pass


class VggNetD(VggNet):
    pass


class VggNetE(VggNet):
    pass
