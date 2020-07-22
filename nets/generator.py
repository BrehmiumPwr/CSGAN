from .Net import Net
from .color_difference_network import CDN
from .blocks import *
from .normalization import normalization


class Generator(Net):
    def __init__(self, d_mult=64, norm=None, spectral_norm=True, activation=tf.nn.leaky_relu, istraining=False,
                 num_colors=11, separate_location_and_intensity=False, share_location_and_intensity=False,
                 num_semi_supervised_classes=2,
                 use_aspp=True,
                 use_dropout=True,
                 block_type="residual",
                 network="resnet",
                 name="generator"):
        super().__init__(name)
        self.d_mult = d_mult
        self.sn = spectral_norm
        self.act = activation
        self.num_colors = num_colors
        self.separate_location_and_intensity = separate_location_and_intensity
        self.separate_location_and_intensity_with_shared_features = \
            self.separate_location_and_intensity and share_location_and_intensity
        self.use_aspp = use_aspp
        self.use_dropout = use_dropout
        self.norm = norm
        self.style_mod_in_backbone = False

        self.istraining = istraining
        self.color_vec = None
        block_types = {
            "residual": residual,
            "separable_residual": separable_residual,
            "bottleneck": bottleneck,
            "inversebottleneck": inverse_bottleneck
        }
        block_type = block_types[block_type]
        network_types = {
            "resnet": lambda x, name: residual_block(x, self.d_mult * 4, self.act, self.norm, self.sn, ks=3,
                                                     type=block_type, name=name),
            "dualpath": lambda x, name: dual_path_block(x, self.d_mult * 4, 16, self.act, self.norm, self.sn, ks=3,
                                                        type=block_type, name=name)
        }
        self.block = network_types[network]
        self.num_semisupervised_classes = num_semi_supervised_classes

        self.CDN = CDN(d_mult=self.d_mult, norm=normalization['none'], spectral_norm=self.sn, activation=self.act, istraining=self.istraining,
                       num_colors=self.num_colors, use_dropout=use_dropout, name="color_difference_network")

    def style_net(self, batch_size):
        with tf.variable_scope("style_net", reuse=tf.AUTO_REUSE):
            if self.style_mod_in_backbone:
                rand = tf.random.truncated_normal(shape=(batch_size, 1, 1, self.d_mult))
                rand = self.act(conv2d(rand, output_dim=self.d_mult * 2, ks=1, s=1, padding="SAME", name="conv1"))
                rand = self.act(conv2d(rand, output_dim=self.d_mult * 2, ks=1, s=1, padding="SAME", name="conv2"))
                rand = self.act(conv2d(rand, output_dim=self.d_mult * 4, ks=1, s=1, padding="SAME", name="conv3"))
                rand = self.act(conv2d(rand, output_dim=self.d_mult * 4, ks=1, s=1, padding="SAME", name="conv4"))
                rand = self.act(conv2d(rand, output_dim=self.d_mult * 4, ks=1, s=1, padding="SAME", name="conv5"))
                rand = self.act(conv2d(rand, output_dim=self.d_mult * 4, ks=1, s=1, padding="SAME", name="conv6"))
                return rand
            else:
                return None

    def __call__(self, image, conditioning=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            style = self.style_net(tf.shape(image)[0])
            c1 = conv2d(image, self.d_mult, ks=7, s=1, padding='SAME', use_spectral_norm=self.sn, act=self.act,
                        name='g_e1_c')
            # s=1 7x7
            c2 = conv2d(c1, self.d_mult * 2, ks=3, s=2, padding='SAME', use_spectral_norm=self.sn, norm=self.norm,
                        act=self.act, name='g_e2_c')
            # s=1 9x9
            c3 = conv2d(c2, self.d_mult * 4, ks=3, s=2, padding='SAME', use_spectral_norm=self.sn, norm=self.norm,
                        name='g_e3_c')
            # s=2 13x13

            num_blocks = 10
            features = c3
            for x in range(num_blocks):
                features = self.block(features, 'block_' + str(x))
                if self.style_mod_in_backbone:
                    features = tf.cond(self.istraining, lambda: style_mod(features, style, name="style_" + str(x)),
                                       lambda: features)

                if self.use_dropout:
                    if isinstance(features, tuple):
                        features = (convolutional_dropout(features[0], training=self.istraining),
                                    convolutional_dropout(features[1], training=self.istraining))
                    else:
                        features = convolutional_dropout(features, training=self.istraining)

            if isinstance(features, tuple):
                features = tf.concat([*features], axis=-1)

            if self.use_aspp:
                features = aspp(features, output_dim=self.d_mult * 4, ks=3, rates=[6, 12, 18],
                                use_spectral_norm=self.sn, activation=self.act, share_weights=False,
                                feature_fusion="concat", global_features=True, name="aspp")

            self.color_vec = self.CDN(image, conditioning=conditioning)

            # Upsampling layers
            d2 = deconv2d(features, self.d_mult * 2, ks=3, s=2, use_spectral_norm=self.sn, norm=self.norm, act=self.act,
                          name='g_d2_dc')
            d2 = tf.concat([d2, c2], axis=-1)

            d1 = deconv2d(d2, self.d_mult, ks=3, s=2, use_spectral_norm=self.sn, norm=self.norm, act=self.act,
                          name='g_d1_dc')
            d1 = tf.concat([d1, c1], axis=-1)

            ks_head = 7
            # s=1 163x163
            if self.separate_location_and_intensity:
                if self.separate_location_and_intensity_with_shared_features:
                    intensity = self.col_delta_prediction_head(d1, ks=ks_head, out_act=binary_act_with_sigmoid_back, name="intensity")
                    intensity = tf.nn.sigmoid(intensity)
                else:
                    location = self.col_delta_prediction_head(d1, ks=ks_head, out_act= binary_act, name="location")
                    intensity = (abs_act(self.col_delta_prediction_head(d1, ks=ks_head, name="intensity")) + 0.5) * (
                            1. / 1.5)
                colored_mask = (location * intensity) * self.color_vec
            else:
                location = self.col_delta_prediction_head(d1, ks=ks_head, out_act=abs_act, name="location")
                colored_mask = location * self.color_vec

            pixelwise_rescaling = 1.0 - (0.2 * conv2d(d1, output_dim=3, ks=3, s=1, padding='SAME', name="pixelwise_rescaling",
                   use_wscale=False,
                   norm=None,
                   act=tf.nn.sigmoid,
                   use_spectral_norm=self.sn))

            colored_mask *= pixelwise_rescaling

            # colorize image
            rgb_full_size = colored_mask + image

            semi_logits = conv2d(d1, output_dim=self.num_semisupervised_classes, ks=ks_head, s=1,
                                 use_spectral_norm=False, name="semi_head")
            return rgb_full_size, location, semi_logits

    def col_delta_prediction_head(self, features, name, ks=3, out_act=tf.nn.relu, depth=0):
        with tf.variable_scope(name):
            for x in range(depth):
                features = conv2d(features, self.d_mult, ks=ks, s=1, padding='SAME', name="conv{}".format(x + 1),
                                  use_wscale=False,
                                  norm=self.norm,
                                  act=self.act,
                                  use_spectral_norm=self.sn)
            # features = tf.nn.l2_normalize(features, axis=-1)
            mask = conv2d(features, output_dim=1, ks=ks, s=1, padding='SAME', name="pred", use_wscale=False,
                          norm=None,
                          act = out_act,
                          use_spectral_norm=False)
            # mask = tf.nn.softmax(mask, axis=-1)[:,:,:,:1]
            return mask
