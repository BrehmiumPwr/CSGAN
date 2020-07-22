import tensorflow as tf

import cv2
import numpy as np


def calc_aspect_preserving_image_size_crop(image, target_image_size):
    shape = tf.shape(image)
    scale_factor = tf.cast(target_image_size, dtype=tf.float32) / tf.cast(tf.reduce_min(shape[:2]), dtype=tf.float32)
    new_shape = tf.cast(shape[:2], dtype=tf.float32) * scale_factor
    new_height = tf.cast(new_shape[0], dtype=tf.int32)
    new_width = tf.cast(new_shape[1], dtype=tf.int32)
    return new_height, new_width

def calc_aspect_preserving_image_size_pad(image, target_image_size):
    shape = tf.shape(image)
    scale_factor = tf.cast(target_image_size, dtype=tf.float32) / tf.cast(tf.reduce_max(shape[:2]), dtype=tf.float32)
    new_shape = tf.cast(shape[:2], dtype=tf.float32) * scale_factor
    new_height = tf.cast(new_shape[0], dtype=tf.int32)
    new_width = tf.cast(new_shape[1], dtype=tf.int32)
    return new_height, new_width

def py_resize_rgb(image, new_size):
    image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_AREA)
    return image

def py_resize_grey(image, new_size):
    image = cv2.resize(image, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
    return np.expand_dims(image, axis=-1)

def aspect_preserving_resize(image, target_size, resize_mode="pad"):
    channels_in = image.shape[-1]
    if resize_mode == "crop":
        new_image_height, new_image_width = calc_aspect_preserving_image_size_crop(image, target_size)
    elif resize_mode == "pad":
        new_image_height, new_image_width = calc_aspect_preserving_image_size_pad(image, target_size)
    if channels_in == 3:
        image = tf.py_func(lambda x, y: py_resize_rgb(x, y), [image, (new_image_height, new_image_width)], [image.dtype])[0]
    elif channels_in == 1:
        image = tf.py_func(lambda x, y: py_resize_grey(x, y), [image, (new_image_height, new_image_width)], [image.dtype])[0]

    image.set_shape([None, None, channels_in])
    return image