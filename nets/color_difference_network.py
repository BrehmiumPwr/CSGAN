from .Net import Net
from .cnn_ops import *
from .cnn_utils import *
from .blocks import *


class CDN(Net):
    def __init__(self, d_mult=64, norm=None, spectral_norm=True, activation=tf.nn.leaky_relu, istraining=False,
                 num_colors=11,
                 use_dropout=True,
                 name="color_difference_network"):
        super().__init__(name)
        self.d_mult = d_mult
        self.sn = spectral_norm
        self.act = activation
        self.num_colors = num_colors
        self.use_dropout = use_dropout
        self.norm = norm
        self.istraining = istraining
        self.color_vec = None
        self.style_mod = True
        self.preprocess_condition = True
        self.mean_correct = True

    def conditioning_network(self, conditioning):
        with tf.variable_scope("conditioning_network"):
            # add spatial dims if necessary
            if len(conditioning.shape) == 2:
                conditioning = tf.expand_dims(conditioning, axis=1)
                conditioning = tf.expand_dims(conditioning, axis=1)
            conditioning = conv2d(conditioning, output_dim=self.d_mult * 2, ks=1, s=1, padding="SAME", norm=self.norm, act=self.act, name="conv1")
            conditioning = conv2d(conditioning, output_dim=self.d_mult * 2, ks=1, s=1, padding="SAME", norm=self.norm, act=self.act, name="conv2")
            conditioning = conv2d(conditioning, output_dim=self.d_mult * 4, ks=1, s=1, padding="SAME", norm=self.norm, act=self.act, name="conv3")
            conditioning = conv2d(conditioning, output_dim=self.d_mult * 4, ks=1, s=1, padding="SAME", norm=self.norm, act=self.act, name="conv4")
            return conditioning

    def __call__(self, features, conditioning=None):
        if self.style_mod:
            if self.preprocess_condition:
                cond = self.conditioning_network(conditioning=conditioning)
            else:
                cond = append_vec(conditioning, tf.random.normal(shape=(self.num_colors, 1, 1, 64), mean=0.0, stddev=1.0))
            cond_fuse = lambda x, name: style_mod(x, cond, name=name)
        else:
            cond = append_vec(conditioning, tf.random.normal(shape=(self.num_colors, 1, 1, 64), mean=0.0, stddev=1.0))
            cond_fuse = lambda x, name: append_vec(x, cond, name=name)
        with tf.variable_scope(self.name):
            cvec = conv2d(features, self.d_mult, ks=7, s=2, padding='SAME',
                                   name='g_cvec_c1', use_wscale=False, norm=None, act=self.act,
                                   use_spectral_norm=self.sn)
            cvec = residual_block(cvec, self.d_mult * 2, act=self.act, norm=self.norm, sn=self.sn, ks=3, s=1,
                                  type=separable_residual, name="block1")
            cvec = residual_block(cvec, self.d_mult * 4, act=self.act, norm=self.norm, sn=self.sn, ks=3, s=2,
                                  type=separable_residual, name="block2")
            cvec = residual_block(cvec, self.d_mult * 4, act=self.act, norm=self.norm, sn=self.sn, ks=3, s=1,
                                  type=separable_residual, name="block3")
            cvec = residual_block(cvec, self.d_mult * 8, act=self.act, norm=self.norm, sn=self.sn, ks=3, s=2,
                                  type=separable_residual, name="block4")
            cvec = residual_block(cvec, self.d_mult * 8, act=self.act, norm=self.norm, sn=self.sn, ks=3, s=1,
                                  type=separable_residual, name="block5")
            cvec = tf.reduce_mean(cvec, axis=[1, 2], keepdims=True)

            cvec = tf.tile(cvec, multiples=[self.num_colors, 1, 1, 1])

            # make output non-deterministic
            # cvec = append_vec(cvec, tf.random.truncated_normal(shape=(self.num_colors, 1, 1, 64), mean=0.0, stddev=1.0))
            cvec = cond_fuse(cvec, name="fuse1")
            cvec = conv2d(cvec, self.d_mult * 8, ks=1, s=1, padding='VALID',
                          name='g_cvec_fc1', use_wscale=False,
                          use_spectral_norm=self.sn)
            cvec = self.act(cond_fuse(cvec, name="fuse2"))

            cvec = conv2d(cvec, self.d_mult * 8, ks=1, s=1, padding='VALID',
                          name='g_cvec_fc2', use_wscale=False,
                          use_spectral_norm=self.sn)
            cvec = self.act(cond_fuse(cvec, name="fuse3"))
            self.color_vec = conv2d(cvec, 3, ks=1, s=1, padding='VALID',
                                    name='g_cvec_fc3', use_wscale=False,
                                    use_spectral_norm=False)
            if self.mean_correct:
                running_mean = RunningMean(input_shape=self.color_vec.get_shape().as_list(), axis=[1, 2], clip_min=-2.0,
                                           clip_max=2.0)
                per_color_mean_correction = running_mean.get_mean()
                self.color_vec += per_color_mean_correction
                self.color_vec = running_mean.update(self.color_vec, training=self.istraining)
                for x in range(self.num_colors):
                    tf.summary.scalar("color_" + str(x), tensor=tf.reduce_mean(per_color_mean_correction[x]))

            tf.summary.histogram("color_vec_len",
                                 values=tf.sqrt(tf.reduce_sum(tf.square(self.color_vec), axis=[1, 2, 3])))
            tf.summary.histogram("color_vec_mean", values=tf.reduce_mean(self.color_vec))

            return self.color_vec