import numpy as np
from config import *
import tensorflow as tf


def get_cifar_dataset(num_class, train_batch, val_batch):
    if num_class == 10:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    def _parse_image_train(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = (image - MEAN["cifar"]) / STD["cifar"]

        image = tf.image.random_crop(tf.pad(image, [[4, 4], [4, 4], [0, 0]]), size=[32, 32, 3])
        image = tf.image.random_flip_left_right(image)

        return image, label

    def _parse_image_val(image, label):
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = (image - MEAN["cifar"]) / STD["cifar"]
        return image, label

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(_parse_image_train,
                                                                               num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buffer_size=len(y_train)).batch(batch_size=train_batch).prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(_parse_image_val,
                                                                           num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size=val_batch).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, np.ceil(len(y_train) / train_batch), val_dataset, np.ceil(len(y_test) / val_batch)


def get_datasets(name, train_batch, val_batch):
    if name == "cifar10":
        return get_cifar_dataset(num_class=10, train_batch=train_batch, val_batch=val_batch)
    elif name == "cifar100":
        return get_cifar_dataset(num_class=100, train_batch=train_batch, val_batch=val_batch)
    else:
        raise ValueError("Dataset only support cifar10, cifar100 and ILSVRC2012, but get {}!".format(name))


if __name__ == '__main__':
    train_date, train_batch_num, val_data, val_batch_num = get_cifar_dataset(num_class=10, train_batch=70, val_batch=1)

    for b, (d, l) in enumerate(train_date):
        print(d.shape)
        print(l.shape)
        break
