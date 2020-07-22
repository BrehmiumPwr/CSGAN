import numpy as np
import tensorflow as tf
#import tensorflow.contrib.slim as slim
from nets.cnn_utils import *
from tensorflow.python.ops import math_ops

def layer_lifting(activation):
    max = tf.maximum(0.0, activation)
    min = tf.minimum(0.0, activation)
    return tf.concat([min, max], axis=-1)


def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)


def global_features(features, channels=None, use_spectral_norm=False):
    spatial_dims = tf.shape(features)[1:3]
    if channels is None:
        channels = features.get_shape().as_list()[-1]
    global_features = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
    global_features = tf.tile(global_features, tf.stack([1, spatial_dims[0], spatial_dims[1], 1]))
    # global_features = tf.image.resize_nearest_neighbor(global_features, spatial_dims)
    new_features = tf.concat([features, global_features], axis=-1)
    return conv2d(new_features, channels, ks=3, s=1, use_spectral_norm=use_spectral_norm,
                  name="global_feature_infusion")


def instance_norm(input, name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale", [depth], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [depth], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (input - mean) * inv
        return scale * normalized + offset


def group_norm(x, name="group_norm", channel_per_group=16, eps=1e-5, ):
    with tf.variable_scope(name):
        # x: input features with shape [N,H,W, C]
        # gamma, beta: scale and offset, with shape [C]
        # G: number of groups for GN
        shape = tf.shape(x)
        H = shape[1]
        W = shape[2]
        C = x.get_shape().as_list()[-1]
        G = C // channel_per_group
        scale = tf.get_variable("scale", [C], initializer=tf.random_normal_initializer(1.0, 0.02, dtype=tf.float32))
        offset = tf.get_variable("offset", [C], initializer=tf.constant_initializer(0.0))

        N = tf.shape(x)[0]
        x = tf.reshape(x, tf.stack([N, H, W, G, C // G], axis=0))
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        x = tf.reshape(x, tf.stack([N, H, W, C], axis=0))
        return x * scale + offset


def spectral_norm(w, istraining=True, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    if istraining:
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)
    else:
        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def spectral_norm_weight3(w, iteration=1):
    return tf.contrib.gan.features.spectral_normalize(w, power_iteration_rounds=iteration)


def spectral_norm_weight2(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
    # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def power_iteration(i, u_i, v_i):
        v_ip1 = tf.nn.l2_normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = tf.nn.l2_normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1

    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                   u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
    )
    if update_collection is None:
        #warnings.warn(
        #    'Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
        #    '. Please consider using a update collection instead.')
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        with tf.control_dependencies([u.assign(u_final)]):
            W_bar = tf.reshape(W_bar, W_shape)
    else:
        sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
        # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
        W_bar = W_reshaped / sigma
        W_bar = tf.reshape(W_bar, W_shape)
        # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
        # has already been collected on the first call.
        if update_collection != None:
            tf.add_to_collection(update_collection, u.assign(u_final))
    if with_sigma:
        return W_bar, sigma
    else:
        return W_bar


def hw_flatten(x):
    s = x.get_shape().as_list()
    return tf.reshape(x, shape=[1, -1, s[-1]])


# adapted from https://github.com/taki0112/Self-Attention-GAN-Tensorflow
def attention(x, use_spectral_norm=False, name='attention'):
    channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name):
        f = conv2d(x, channels // 8, ks=1, s=1, use_spectral_norm=use_spectral_norm, use_wscale=False,
                   name='f_conv')  # [bs, h, w, c']
        g = conv2d(x, channels // 8, ks=1, s=1, use_spectral_norm=use_spectral_norm, use_wscale=False,
                   name='g_conv')  # [bs, h, w, c']
        h = conv2d(x, channels, ks=1, s=1, use_spectral_norm=use_spectral_norm, use_wscale=False,
                   name='h_conv')  # [bs, h, w, c]

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=tf.shape(x))  # [bs, h, w, C]
        x = gamma * o + x

    return x


def upsample(feat_in, factor):
    input_shape = tf.shape(feat_in)
    channels = feat_in.shape[-1]
    reshaped_features = tf.reshape(feat_in, shape=tf.stack(
        [input_shape[0], input_shape[1], 1, input_shape[2], 1, channels]))
    replicated_features = tf.tile(reshaped_features, multiples=[1, 1, factor, 1, factor, 1])
    upscaled_features = tf.reshape(replicated_features, shape=tf.stack(
        [input_shape[0], input_shape[1] * factor, input_shape[2] * factor, channels]))
    return upscaled_features


def spatial_attention(features, size):
    attention_map = tf.nn.avg_pool(features, ksize=[1, size, size, 1], strides=[1, size, size, 1], padding="VALID")
    attention_map = conv2d(attention_map, output_dim=1, ks=3, s=1, act=tf.nn.sigmoid)
    attention_map = upsample(attention_map, factor=size)
    return features * attention_map


def aspp(features, output_dim, ks=3, rates=[2, 4, 8], use_spectral_norm=False, activation=tf.nn.relu,
         share_weights=True, feature_fusion="concat", global_features=True, name="aspp"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        outputs = []
        for rate in rates:
            if share_weights:
                layer_name = "conv2d"
            else:
                layer_name = "conv2d_rate{}".format(rate)
            x = depthwise_conv2d(features, ks=ks, rate=rate, use_spectral_norm=False, act=activation,
                                 padding='SAME', name=layer_name, pad_algorithm="CONSTANT")
            outputs.append(x)

        outputs.append(conv2d(features, output_dim, ks=1, s=1, use_spectral_norm=use_spectral_norm,
                              padding='SAME', name="conv2d_ks1"))
        if global_features:
            input_spatial_dims = tf.shape(features)[1:3]
            global_features = tf.reduce_mean(features, axis=[1, 2], keepdims=True)
            global_features = conv2d(global_features, output_dim, ks=1, s=1, use_spectral_norm=use_spectral_norm,
                                     padding='SAME', act=activation, name="global_features")
            global_features = tf.tile(global_features,
                                      multiples=tf.stack([1, input_spatial_dims[0], input_spatial_dims[1], 1]))
            outputs.append(global_features)

        if feature_fusion == "add":
            return tf.add_n(outputs, name="aspp_out")
        elif feature_fusion == "concat":
            stacked = tf.concat(outputs, axis=-1, name="feature_fusion")
            return conv2d(stacked, output_dim, ks=1, s=1, use_spectral_norm=use_spectral_norm,
                                     padding='SAME', act=activation, name="aspp_out")
        else:
            raise Exception("undefined fusion in aspp")


def convolution_aftermath(data, use_bias=True, use_scale=True, norm=None, act=None, name="conv_aftermath"):
    output_dim = data.get_shape().as_list()[-1]
    if use_bias:
        b = tf.get_variable(shape=(output_dim), dtype=tf.float32, initializer=tf.zeros_initializer(),
                            name=name + "_bias")
        data = data + b

    if use_scale:
        s = tf.get_variable(shape=(output_dim), dtype=tf.float32, initializer=tf.ones_initializer(),
                            name=name + "_scale")
        data = data * s
    if norm is not None:
        data = norm(data, name="norm")

    if act is not None:
        data = act(data)

    return data

def conv2d(input_, output_dim, ks=3, s=2, gain=np.sqrt(2), rate=1, weight_decay=0.00001, padding='SAME', pad_algorithm="REFLECT",
           norm=None,
           act=None,
           name="conv2d",
           use_wscale=False,
           use_spectral_norm=True,
           use_bias=True,
           use_scale=True):
    with tf.variable_scope(name):
        input_channels = input_.get_shape().as_list()[-1]
        w = get_weight([ks, ks, input_channels, output_dim], gain=gain, use_wscale=use_wscale)
        if use_spectral_norm:
            w = spectral_norm(w)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w) * weight_decay)
        if padding.lower() == "same":
            virtual_ks = (rate * (ks - 1)) + 1
            input_ = pad(input_, ks=virtual_ks, s=s, algorithm=pad_algorithm)
        if s != 2:
            w = tf.cast(w, input_.dtype)
            y = tf.nn.conv2d(input_, w, strides=[1, s, s, 1], dilations=[1, rate, rate, 1], padding="VALID",
                             data_format='NHWC')
        else:
            y = conv2d_downscale2d(input_, w, rate=rate)

        return convolution_aftermath(y, use_bias=use_bias, use_scale=use_scale, norm=norm, act=act, name=name)


def depthwise_conv2d(input_, ks=3, gain=np.sqrt(2), s=1, rate=1, weight_decay=0.00001, padding='SAME', pad_algorithm="REFLECT",
                     norm=None,
                     act=None,
                     name="conv2d",
                     use_wscale=False,
                     use_spectral_norm=True,
                     use_bias=True,
                     use_scale=True):
    with tf.variable_scope(name):
        input_channels = input_.get_shape().as_list()[-1]
        w = get_weight([ks, ks, input_channels, 1], gain=gain, use_wscale=use_wscale, depthwise=True)
        if use_spectral_norm:
            w = spectral_norm(w)
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w) * weight_decay)
        w = tf.cast(w, input_.dtype)

        if padding.lower() == "same":
            virtual_ks = (rate * (ks - 1)) + 1
            input_ = pad(input_, ks=virtual_ks, s=s, algorithm=pad_algorithm)
        y = tf.nn.depthwise_conv2d(input_, w, strides=[1, s, s, 1], rate=[rate,rate], padding="VALID",
                                   data_format='NHWC')
        return convolution_aftermath(y, use_bias=use_bias, use_scale=use_scale, norm=norm, act=act, name=name)


def conv2d_downscale2d(x, w, rate=1):
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, w, strides=[1, 2, 2, 1], dilations=[1, rate, rate, 1], padding="VALID", data_format='NHWC')


def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None, depthwise=False):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init

    if depthwise:
        fan_in = np.prod(shape[:2])
        std = gain / np.sqrt(fan_in)
        return tf.get_variable('weight', shape=shape, initializer=RandomDepthwiseInitializer(0, std))
    elif use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))


def res_from_depth(feat, output_dims, factor=2, use_spectral_norm=True, name="res_from_depth"):
    with tf.variable_scope(name):
        inflated_features = conv2d(feat, output_dim=output_dims * (factor ** 2), ks=1, s=1,
                                   use_spectral_norm=use_spectral_norm, name="conv2d")
        upscaled_features = tf.nn.depth_to_space(inflated_features, block_size=factor)
        return upscaled_features


def upscale2d_conv2d(x, fmaps, kernel, gain=np.sqrt(2), s=2, use_wscale=False, weight_decay=0.00001,
                     use_spectral_norm=False):
    assert kernel >= 1 and kernel % 2 == 1
    input_channels = x.get_shape().as_list()[-1]
    w = get_weight([kernel, kernel, fmaps, input_channels], gain=gain, use_wscale=use_wscale,
                   fan_in=(kernel ** 2) * input_channels)
    tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.nn.l2_loss(w) * weight_decay)
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    if use_spectral_norm:
        w = spectral_norm(w)
    os = [tf.shape(x)[0], tf.shape(x)[1] * 2, tf.shape(x)[2] * 2, fmaps]
    return tf.nn.conv2d_transpose(x, w, os, strides=[1, s, s, 1], padding='SAME', data_format='NHWC')


def upscale_nearest_neighbour(feat_in, factor=2):
    input_shape = tf.shape(feat_in)
    channels = feat_in.shape[-1]
    reshaped_features = tf.reshape(feat_in, shape=tf.stack(
        [input_shape[0], input_shape[1], 1, input_shape[2], 1, channels]))
    replicated_features = tf.tile(reshaped_features, multiples=[1, 1, factor, 1, factor, 1])
    upscaled_features = tf.reshape(replicated_features, shape=tf.stack(
        [input_shape[0], input_shape[1] * factor, input_shape[2] * factor, channels]))
    return upscaled_features


def abs_act(x):
    x = tf.abs(x) - 1.0
    return tf.minimum(x, x * 0.1) + 1.0


@tf.custom_gradient
def binary_act(features):
    activation = tf.cast(tf.greater(features, 0.0), dtype=tf.float32) - .5
    activation *= 0.9
    activation += 0.5

    def grad(dy):
        return dy  # tf.where(tf.less(dy, 0.0), -tf.ones_like(dy), tf.ones_like(dy))
        # return math_ops.sigmoid_grad(features, dy, "binarygrad"), 0

    return activation, grad


@tf.custom_gradient
def binary_act_with_sigmoid_back(features):
    activation = tf.cast(tf.greater(features, 0.0), dtype=tf.float32) - .5
    activation *= 0.9
    activation += 0.5

    def grad(dy):
        # return dy#tf.where(tf.less(dy, 0.0), -tf.ones_like(dy), tf.ones_like(dy))
        return math_ops.sigmoid_grad(features, dy, "binarygrad")

    return activation, grad


def binary_act_scaled(features):
    scale = 0.8
    features = binary_act(features, scale)
    rescaled_features = ((features - 0.5) / scale) + 0.5
    return rescaled_features


def upscalenn_conv2d(input_, output_dim, factor=2, ks=3, gain=np.sqrt(2), rate=1, weight_decay=0.00001,
                     name="upscale_2d",
                     use_wscale=False,
                     use_spectral_norm=True,
                     norm=None,
                     act=None,
                     use_bias=True,
                     use_scale=True):
    y = upsample(input_, factor=factor)  # upscale_nearest_neighbour(input_, factor=factor)
    return conv2d(y,
                  output_dim=output_dim,
                  ks=ks,
                  s=1,
                  gain=gain,
                  rate=rate,
                  weight_decay=weight_decay,
                  padding="SAME",
                  name=name,
                  use_wscale=use_wscale,
                  use_spectral_norm=use_spectral_norm,
                  use_bias=use_bias,
                  use_scale=use_scale,
                  norm=norm,
                  act=act)


def deconv2d(input_, output_dim, ks=5, s=2, stddev=0.02, wscale=False, weight_decay=0.00001, use_spectral_norm=False,
             norm=None,
             act=None,
             name="deconv2d",
             use_bias=True,
             use_scale=True):
    with tf.variable_scope(name):
        if s != 2:
            raise Exception()
            #y slim.conv2d_transpose(input_, output_dim, ks, s, padding='SAME', activation_fn=None,
            #                             weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
            #                             biases_initializer=None)
        else:
            y =  upscale2d_conv2d(input_, output_dim, s=s, kernel=ks, use_wscale=wscale, weight_decay=weight_decay,
                                    use_spectral_norm=use_spectral_norm)

        return convolution_aftermath(y, use_bias=use_bias, use_scale=use_scale, norm=norm, act=act)


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [input_.get_shape()[-1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias


def convolutional_dropout(features, rate=.1, training=False):
    channels = features.get_shape().as_list()[-1]
    batch_size = tf.shape(features)[0]
    noise_shape = [batch_size, 1, 1, channels]
    return tf.layers.dropout(features, rate=rate, noise_shape=noise_shape, training=training)


def label_to_one_hot(label, depth):
    one_hot = tf.one_hot(label, depth, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)
    one_hot = tf.expand_dims(one_hot, axis=[1])
    return tf.expand_dims(one_hot, axis=[1])


def get_random_one_hot(batch_size, depth):
    idx = tf.random_uniform(shape=[batch_size], minval=0, maxval=depth + 1, dtype=tf.int32)
    return label_to_one_hot(idx, depth)

def style_mod(x, dlatent, name="stylemod"):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        if len(x.shape) == 2:
            x = tf.expand_dims(x, axis=1)
            x = tf.expand_dims(x, axis=1)

        style = conv2d(dlatent, output_dim=x.shape[-1]*2, ks=1, s=1, use_bias=True, use_scale=True)
        style = tf.reshape(style, [-1, 1, 1, x.shape[-1], 2])
        return x * (style[:, :, :, :, 0] + 1) + style[:, :, :, :, 1]

def pixel_norm(x, epsilon=1e-8, name="PixelNorm"):
    with tf.variable_scope(name):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=-1, keepdims=True) + epsilon)