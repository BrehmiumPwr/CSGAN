from nets.Net import Net
from nets.cnn_ops import *


class Discriminator(Net):
    def __init__(self, d_mult=64, norm=None, spectral_norm=True, activation=tf.nn.leaky_relu, num_outputs=1,
                 istraining=False, name="discriminator"):
        super().__init__(name)
        self.d_mult = d_mult
        self.sn = spectral_norm
        self.act = activation
        self.num_outputs = num_outputs
        self.from_rgb_act = self.act
        self.norm = norm
        self.istraining = istraining

    def __call__(self, image, conditioning=None):
        raise Exception("not implemented")

    def prediction_head(self, features, conditioning_vec, name, num_outputs=1):
        with tf.variable_scope(name):
            features = append_vec(features, conditioning_vec)
            pred = conv2d(features, num_outputs, ks=1, s=1, use_spectral_norm=self.sn, norm=None, act=None,
                          padding="SAME", name="conv")
            return pred

    def from_rgb(self, data_in, scales, lvl):
        x = data_in
        #if scales[lvl] > 1:
        #    size = [1, scales[lvl], scales[lvl], 1]
        #    x = tf.nn.avg_pool2d(x, ksize=size,
        #                         strides=size, padding="SAME", name="avg_pool_lvl_" + str(lvl))
        if lvl < len(scales) - 1:
            input_dim = self.d_mult * scales[lvl + 1]
        else:
            input_dim = self.d_mult * scales[lvl]
        x = conv2d(x, output_dim=input_dim,
                   ks=3,
                   s=scales[lvl],
                   act=self.from_rgb_act,
                   rate=scales[lvl],
                   padding="SAME",
                   use_spectral_norm=self.sn,
                   name="d_from_rgb_lvl" + str(lvl))
        return x
