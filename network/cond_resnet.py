import tensorflow as tf
from config import WEIGHT_DECAY
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from .condconv import CondConv2D


class CondCifarBasicBlock(layers.Layer):
    expansion = 1

    def __init__(self, filters, stride=1, option='A', num_experts=3, **kwargs):
        super(CondCifarBasicBlock, self).__init__(**kwargs)
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.conv1 = CondConv2D(kernel_size=3, filters=filters, stride=stride, use_bias=False, num_experts=num_experts, name="conv1")
        self.bn2 = layers.BatchNormalization(name="bn2")
        self.conv2 = CondConv2D(kernel_size=3, filters=filters, stride=1, use_bias=False, num_experts=num_experts, name="conv2")

        self.shortcut = None
        if stride != 1:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = lambda x: tf.pad(tf.nn.avg_pool2d(x, (2, 2), strides=(1, 2, 2, 1), padding='SAME'),
                                                 [[0, 0], [0, 0], [0, 0], [filters // 4, filters // 4]])
            elif option == 'B':
                self.shortcut = Sequential([
                    CondConv2D(kernel_size=1, filters=self.expansion * filters, stride=stride, use_bias=False, num_experts=num_experts),
                    layers.BatchNormalization()
                ], name="shortcut")

    def call(self, inputs, training=None, **kwargs):

        out = tf.nn.relu(self.bn1(self.conv1(inputs), training=training))
        out = self.bn2(self.conv2(out), training=training)

        if self.shortcut is not None:
            inputs = self.shortcut(inputs)

        out += inputs
        out = tf.nn.relu(out)
        return out


class CondCifarResNet(Model):
    def __init__(self, num_layers, num_classes=10, num_experts=3):
        super(CondCifarResNet, self).__init__()

        block = CondCifarBasicBlock
        num_blocks = int((num_layers - 2) / 6)

        self.conv1 = CondConv2D(kernel_size=3, filters=16, stride=1, use_bias=False, num_experts=num_experts, name="conv1")
        self.bn1 = layers.BatchNormalization(name="bn1")
        self.layer1 = self._make_layer(block, 16, num_blocks, stride=1, num_experts=num_experts, name="layer1")
        self.layer2 = self._make_layer(block, 32, num_blocks, stride=2, num_experts=num_experts, name="layer2")
        self.layer3 = self._make_layer(block, 64, num_blocks, stride=2, num_experts=num_experts, name="layer3")
        self.gavgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(units=num_classes, name="fc")

    def _make_layer(self, block, filters, num_blocks, stride, name, num_experts):
        blocks_list = [block(filters, stride, num_experts=num_experts)]
        for i in range(1, num_blocks):
            blocks_list.append(block(filters, 1, num_experts=num_experts))

        return Sequential(blocks_list, name=name)

    def call(self, inputs, training=None, mask=None):
        out = tf.nn.relu(self.bn1(self.conv1(inputs), training=training))
        out = self.layer1(out, training=training)
        out = self.layer2(out, training=training)
        out = self.layer3(out, training=training)
        out = self.gavgpool(out)
        out = self.fc(out)
        return out
