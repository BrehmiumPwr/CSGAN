from nets.discriminator import Discriminator
from nets.cnn_ops import *
from nets.growing_discriminator import GrowingDiscriminator


class GrowingStandardDiscriminator(GrowingDiscriminator):
    def __init__(self, d_mult=64, norm=None, spectral_norm=True, activation=tf.nn.leaky_relu, num_outputs=1,
                 istraining=False, schedule=None, scales=None, len_fading_phase=30000, name="discriminator"):
        super().__init__(d_mult=d_mult, norm=norm, spectral_norm=spectral_norm, activation=activation,
                         num_outputs=num_outputs, istraining=istraining, schedule=schedule, scales=scales,
                         len_fading_phase=len_fading_phase, name=name)

    def block(self, x, output_dim, s=1, name="block"):
        with tf.variable_scope(name):
            x = conv2d(x, output_dim=output_dim, ks=3, s=1, padding="SAME",
                                          use_spectral_norm=self.sn, norm=self.norm, act=self.act,
                                          name="conv1")
            x = conv2d(x, output_dim=output_dim, ks=3, s=s, padding="SAME",
                                          use_spectral_norm=self.sn, norm=self.norm, act=self.act,
                                          name="conv2")
            return x