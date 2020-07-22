from nets.discriminator import Discriminator
from nets.cnn_ops import *


class SpatialDiscriminator(Discriminator):
    def __init__(self, d_mult=64, norm=None, spectral_norm=True, activation=tf.nn.leaky_relu, num_outputs=1, istraining=False, name="discriminator"):
        super().__init__(d_mult=d_mult, norm=norm, spectral_norm=spectral_norm, activation=activation, num_outputs=num_outputs, istraining=istraining, name=name)

    def __call__(self, image, conditioning=None):
        return self.larger_spatial_discriminator(image, conditioning)

    def larger_spatial_discriminator(self, image, conditioning):
        net_identifier = "large_spacial"
        with tf.variable_scope(net_identifier+self.name, reuse=tf.AUTO_REUSE):
            h0 = conv2d(image, self.d_mult, ks=3, s=1, padding='VALID', use_spectral_norm=self.sn, norm=None, act=self.act, name='d_h0_conv1')
            # s=1 3x3
            h0 = conv2d(h0, self.d_mult, ks=3, s=1, padding='VALID', use_spectral_norm=self.sn, norm=self.norm, act=self.act,  name='d_h0_conv2')
            # s=1 5x5

            h1 = conv2d(h0, self.d_mult * 2, ks=3, s=2, padding='VALID', use_spectral_norm=self.sn, norm=self.norm, act=self.act,  name='d_h1_conv1')
            # s=1 7x7
            h1 = conv2d(h1, self.d_mult * 2, ks=3, s=1, padding='VALID', use_spectral_norm=self.sn, norm=self.norm, act=self.act,  name='d_h1_conv2')
            # s=2 11x11

            h2 = conv2d(h1, self.d_mult * 4, ks=3, s=2, padding='VALID', use_spectral_norm=self.sn, norm=self.norm, act=self.act,  name='d_h2_conv1')
            # s=2 15x15
            h2 = conv2d(h2, self.d_mult * 4, ks=3, s=1, padding='VALID', use_spectral_norm=self.sn, norm=self.norm, act=self.act,  name='d_h2_conv2')
            # s=4 23x23

            h2 = convolutional_dropout(h2, training=self.istraining)
            h3 = append_vec(h2, conditioning)
            h3 = conv2d(h3, self.d_mult * 8, ks=3, s=2, padding='VALID', use_spectral_norm=self.sn, norm=self.norm, act=self.act,  name='d_h3_conv1')
            # s=4 31x31

            h3 = convolutional_dropout(h3, training=self.istraining)
            h3 = append_vec(h3, conditioning)
            h3 = conv2d(h3, self.d_mult * 8, ks=3, s=1, padding='VALID', use_spectral_norm=self.sn, norm=self.norm, act=self.act,  name='d_h3_conv2')
            # s=8 47x47

            h3 = convolutional_dropout(h3, training=self.istraining)
            h4 = self.prediction_head(h3, conditioning, "pred", num_outputs=self.num_outputs)
            # h4 is (32 x 32 x 1)
        return h4

