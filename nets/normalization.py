import tensorflow as tf
from .cnn_ops import instance_norm, group_norm, pixel_norm

normalization = {
    "instancenorm": instance_norm,
    "groupnorm": lambda x, name: group_norm(x, name, channel_per_group=4),
    "layernorm": lambda x, name: tf.contrib.layers.layer_norm(x, scope=name),
    "none": lambda x, name: x,
    "pixelnorm": lambda x, name: pixel_norm(x, name=name)
}