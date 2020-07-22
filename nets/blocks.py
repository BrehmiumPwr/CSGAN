import tensorflow as tf
from .cnn_ops import *
from .cnn_utils import *


def residual(x, dim, act, norm, sn, ks=3, s=1):
    y = act(x)
    y = conv2d(y, dim, ks=ks, s=s, padding='SAME', use_spectral_norm=sn, norm=norm, act=act, name='conv1')
    y = conv2d(y, dim, ks=ks, s=1, padding='SAME', use_spectral_norm=sn, norm=norm, act=None, name='conv2')
    return y


def separable_residual(x, dim, act, norm, sn, ks=3, s=1):
    y = act(x)
    y = depthwise_conv2d(y, ks=ks, s=s, padding='SAME', use_spectral_norm=False, norm=norm, act=None, name='conv1_depthwise')
    y = conv2d(y, output_dim=dim, ks=1, s=1, padding='SAME', use_spectral_norm=sn, norm=norm, act=act, name='conv1_spatial')
    y = depthwise_conv2d(y, ks=ks, s=1, padding='SAME', use_spectral_norm=False, norm=norm, act=None, name='conv2_depthwise')
    y = conv2d(y, output_dim=dim, ks=1, s=1, padding='SAME', use_spectral_norm=sn, norm=norm, act=None, name='conv2_spatial')
    return y


def bottleneck(x, dim, act, norm, sn, ks=3, s=1):
    dim_in = x.shape[-1]
    intermediate_dim = dim_in // 2
    y = act(x)
    y = conv2d(y, intermediate_dim, ks=1, s=1, padding='SAME', use_spectral_norm=sn, norm=norm, act=act, name='conv1')
    y = conv2d(y, intermediate_dim, ks=ks, s=s, padding='SAME', use_spectral_norm=sn, norm=norm, act=act, name='conv2')
    y = conv2d(y, dim, ks=1, s=1, padding='SAME', use_spectral_norm=sn, norm=norm, act=None, name='conv3')
    return y


def inverse_bottleneck(x, dim, act, norm, sn, ks=3, s=1):
    dim_in = x.shape[-1]
    expansion = 6
    intermediate_dim = dim_in * expansion
    y = conv2d(x, intermediate_dim, ks=1, s=1, padding='SAME', use_spectral_norm=sn, norm=norm, act=act, name='conv1')
    y = depthwise_conv2d(y, ks=ks, s=s, padding='SAME', use_spectral_norm=False, norm=norm, act=act, name='conv2')
    y = conv2d(y, dim, ks=1, s=1, padding='SAME', use_spectral_norm=sn, norm=norm, name='conv3')
    return y


def residual_block(x, dim, act, norm, sn, ks=3, s=1, type=residual, name='res'):
    input_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        # y = append_conditioning_vec(x, conditioning)
        # y = spatial_attention(y, size=2)
        x = norm(x, "norm1")
        y = type(x, dim=dim, act=act, norm=None, sn=sn, ks=ks, s=s)

        if s > 1 or input_channels != dim:
            x = conv2d(x, output_dim=dim, ks=1, s=s, use_spectral_norm=sn, norm=norm, act=None, padding="SAME")
        return y + x


def dual_path_block(x, dim_main_path, dim_second_path, act, norm, sn, ks=3, type=residual, name="dp_block"):
    with tf.variable_scope(name):
        is_first_block = not isinstance(x, tuple)
        if not is_first_block:
            residual_path = x[0]
            dense_path = x[1]
            fused = tf.concat([residual_path, dense_path], axis=-1)
        else:
            fused = x
            residual_path = x
            dense_path = None

        fused = norm(fused, "norm1")
        y = type(fused, dim=dim_main_path + dim_second_path, act=act, norm=None, sn=sn, ks=ks)
        residual_out = y[:, :, :, :dim_main_path]
        dense_out = y[:, :, :, dim_main_path:]

        residual_path = residual_out + residual_path
        if not is_first_block:
            dense_path = tf.concat([dense_path, dense_out], axis=-1)
        else:
            dense_path = dense_out

        return residual_path, dense_path
