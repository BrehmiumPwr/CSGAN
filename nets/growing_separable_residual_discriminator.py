from nets.cnn_ops import *
from nets.growing_discriminator import GrowingDiscriminator
from .blocks import residual_block, separable_residual

class GrowingSeparableResidualDiscriminator(GrowingDiscriminator):
    def __init__(self, d_mult=64, norm=None, spectral_norm=True, activation=tf.nn.leaky_relu, num_outputs=1,
                 istraining=False, schedule=None, scales=None, len_fading_phase=30000, name="discriminator"):
        super().__init__(d_mult=d_mult, norm=norm, spectral_norm=spectral_norm, activation=activation,
                         num_outputs=num_outputs, istraining=istraining, schedule=schedule, scales=scales,
                         len_fading_phase=len_fading_phase, name=name)
        # no activation in first layer. We use pre-activated residual blocks.
        self.from_rgb_act = lambda x: x

    def block(self, x, output_dim, s=1, name="block"):
        return residual_block(x, dim=output_dim, act=self.act, norm=self.norm, sn=self.sn, ks=3, s=s, type=separable_residual,
                              name=name)

