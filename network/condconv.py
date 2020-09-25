import tensorflow as tf
from config import WEIGHT_DECAY
from tensorflow.keras import layers


def conv2d(kernel_size, stride, filters, kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY), padding="same", use_bias=False,
           kernel_initializer="he_normal", **kwargs):
    return layers.Conv2D(kernel_size=kernel_size, strides=stride, filters=filters, kernel_regularizer=kernel_regularizer, padding=padding,
                         use_bias=use_bias, kernel_initializer=kernel_initializer, **kwargs)


class Routing(layers.Layer):
    def __init__(self, out_channels, dropout_rate, temperature=30, **kwargs):
        super(Routing, self).__init__(**kwargs)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.dropout = layers.Dropout(rate=dropout_rate)
        self.fc = layers.Dense(units=out_channels)
        self.softmax = layers.Softmax()
        self.temperature = temperature

    def call(self, inputs, **kwargs):
        """
        :param inputs: (b, c, h, w)
        :return: (b, out_features)
        """
        out = self.avgpool(inputs)
        out = self.dropout(out)

        # refer to paper: https://arxiv.org/pdf/1912.03458.pdf
        out = self.softmax(self.fc(out) * 1.0 / self.temperature)
        return out


class CondConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, stride=1, use_bias=True, num_experts=3, padding="same", **kwargs):
        super(CondConv2D, self).__init__(**kwargs)

        self.routing = Routing(out_channels=num_experts, dropout_rate=0.2, name="routing_layer")
        self.convs = []
        for _ in range(num_experts):
            self.convs.append(conv2d(filters=filters, stride=stride, kernel_size=kernel_size, use_bias=use_bias, padding=padding))

    def call(self, inputs, **kwargs):
        """
        :param inputs: (b, h, w, c)
        :return: (b, h_out, w_out, filters)
        """
        routing_weights = self.routing(inputs)
        feature = routing_weights[:, 0] * tf.transpose(self.convs[0](inputs), perm=[1, 2, 3, 0])
        for i in range(1, len(self.convs)):
            feature += routing_weights[:, i] * tf.transpose(self.convs[i](inputs), perm=[1, 2, 3, 0])
        feature = tf.transpose(feature, perm=[3, 0, 1, 2])
        return feature
