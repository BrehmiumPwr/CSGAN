import tensorflow as tf
from tensorflow.python.ops import init_ops as init
import numpy as np

def pad(data_in, ks, s=1, algorithm="REFLECT"):
    if ks==1 and s==1:
        return data_in

    p = int((ks - 1) / 2)
    if s > 1:
        #  (n+2*p-f)
        input_spatial_dims = tf.shape(data_in)[1:3]

        output_spatial_dims = tf.cast(tf.math.ceil(input_spatial_dims / s), dtype=tf.int32)
        padding = (output_spatial_dims * s + ks - 1) - input_spatial_dims
        pad_start = padding // 2
        pad_end = padding - pad_start
    else:
        pad_start = [p, p]
        pad_end = [p, p]

    return tf.pad(data_in, [[0, 0], [pad_start[0], pad_end[0]], [pad_start[1], pad_end[1]], [0, 0]], algorithm)


def append_vec(features, conditioning_vec=None, name="append_condition"):
    if conditioning_vec is not None:
        with tf.name_scope(name):
            if len(conditioning_vec.shape) == 2:
                conditioning_vec = tf.expand_dims(conditioning_vec, axis=1)
                conditioning_vec = tf.expand_dims(conditioning_vec, axis=1)
            feature_shape = tf.shape(features)
            expanded_conditioning_vec = tf.tile(conditioning_vec, tf.stack([1, feature_shape[1], feature_shape[2], 1]))
            return tf.concat([features, expanded_conditioning_vec], axis=-1)
    else:
        return features


def scale_gradient_signal(net, factor):
    return (1.0 - factor) * tf.stop_gradient(net) + factor * net


def ascent(start_value, end_value, start_iteration, end_iteration, global_step):
    with tf.name_scope("ascent"):
        global_step = tf.cast(global_step, dtype=tf.float32)
        start_value = tf.cast(start_value, dtype=tf.float32)
        end_value = tf.cast(end_value, dtype=tf.float32)
        val = start_value + (end_value - start_value) * (
                (global_step - start_iteration) / (end_iteration - start_iteration))
        val = tf.maximum(0.0, val)
        val = tf.minimum(val, end_value)
        return val


def descent(start_value, end_value, start_iteration, end_iteration, global_step):
    with tf.name_scope("ascent"):
        global_step = tf.cast(global_step, dtype=tf.float32)
        start_value = tf.cast(start_value, dtype=tf.float32)
        end_value = tf.cast(end_value, dtype=tf.float32)
        decay_step_range = end_iteration - start_iteration
        decay_value_range = start_value - end_value
        current_progess = global_step - start_iteration

        multiplier = current_progess / decay_step_range

        decayed = start_value - decay_value_range * multiplier
        val = tf.maximum(end_value, decayed)
        val = tf.minimum(start_value, val)
        return val


class RandomDepthwiseInitializer(init.Initializer):
    """Initializer that generates tensors with a normal distribution.

    Args:
      mean: a python scalar or a scalar tensor. Mean of the random values
        to generate.
      stddev: a python scalar or a scalar tensor. Standard deviation of the
        random values to generate.
      seed: A Python integer. Used to create random seeds. See
        `tf.set_random_seed`
        for behavior.
      dtype: Default data type, used if no `dtype` argument is provided when
        calling the initializer. Only floating point types are supported.
    """

    def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=tf.dtypes.float32):
        self.mean = mean
        self.stddev = stddev
        self.seed = seed
        self.dtype = dtype  # tf._assert_float_dtype(tf.dtypes.as_dtype(dtype))

    def __call__(self, shape, dtype=None, partition_info=None):
        if dtype is None:
            dtype = self.dtype
        dist = tf.random.normal(shape, self.mean, self.stddev, dtype, seed=self.seed)
        normed = dist - tf.reduce_mean(dist, axis=[0, 1], keepdims=True)
        #normed = normed / tf.reduce_sum(tf.abs(normed), axis=[0,1], keepdims=True)
        return normed

    def get_config(self):
        return {
            "mean": self.mean,
            "stddev": self.stddev,
            "seed": self.seed,
            "dtype": self.dtype.name
        }


class RunningMean(object):
    def __init__(self, input_shape, axis=None, smoothing_factor=0.9999, keepdims=True, clip_min=None, clip_max=None, name="running_mean"):
        self.axis=axis
        self.smoothing_factor = smoothing_factor
        self.keepdims = keepdims
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.name = name
        self.input_shape = input_shape
        if axis is not None:
            if keepdims:
                for x in range(len(self.axis)):
                    ax = self.axis[x]
                    self.input_shape[ax] = 1
            else:
                for x in range(len(self.axis)-1, -1, -1):
                    ax = self.axis[x]
                    del self.input_shape[ax]

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self.mean = tf.get_variable(name="mean", initializer=tf.zeros_initializer,
                                        shape=self.input_shape, dtype=tf.float32, trainable=False)

    def update(self, current_value, stop_step=1500000, training=False):
        def do_update(current_value):
            cur_mean = tf.reduce_mean(current_value, axis=self.axis, keepdims=self.keepdims)
            if self.clip_min is not None and self.clip_max is not None:
                cur_mean = tf.clip_by_value(cur_mean, self.clip_min, self.clip_max)
            running = (self.smoothing_factor * self.mean) + ((1.0 - self.smoothing_factor) * cur_mean)
            with tf.control_dependencies([tf.assign(self.mean, running)]):
                current_value = tf.identity(current_value)
            return current_value

        def train_or_not(training=False):
            return tf.cond(training, lambda: do_update(current_value), lambda: current_value)

        running = tf.greater(tf.cast(stop_step, tf.int64), tf.train.get_or_create_global_step())
        return train_or_not(training=tf.logical_and(running, training))

    def get_mean(self):
        return self.mean
