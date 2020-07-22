from nets.discriminator import Discriminator
from nets.cnn_ops import *
from tabulate import tabulate


class GrowingDiscriminator(Discriminator):
    def __init__(self, d_mult=64, norm=None, spectral_norm=True, activation=tf.nn.leaky_relu, num_outputs=1,
                 istraining=False, schedule=None, scales=None, len_fading_phase=30000, name="discriminator"):
        super().__init__(d_mult=d_mult, norm=norm, spectral_norm=spectral_norm, activation=activation,
                         num_outputs=num_outputs, istraining=istraining, name=name)
        if schedule is None:
            self.schedule = [0, 60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000, 540000, 600000, 660000]
        else:
            self.schedule = schedule
        self.schedule = np.array(self.schedule)
        self.schedule_fade = self.schedule + len_fading_phase
        self.schedule_fade[0] = 0
        self.fading_weights = ascent(0.0, 1.0, self.schedule, self.schedule_fade, tf.train.get_or_create_global_step())
        self.fading = tf.greater(self.fading_weights, 0.0)
        if scales is None:
            self.scales = [8, 8, 8, 8, 4, 4, 4, 4, 4, 2, 2, 2]
        else:
            self.scales = scales
        self.scales = np.array(self.scales)
        self.lvls = [x for x in range(len(self.scales))]
        assert self.scales.shape[0] == self.schedule.shape[0]
        [tf.summary.scalar("growing_weights/fade_in_lvl_{}".format(x), self.fading_weights[x]) for x in range(len(self.scales))]

        print("Discriminator growing schedule:", flush=True)
        print(tabulate(zip(self.lvls, self.schedule, self.schedule_fade, self.scales), headers=['level', 'fade_start', 'fade_end', 'scale']), flush=True)
        self.active = tf.greater(tf.train.get_or_create_global_step(), self.schedule)
        self.inactive = tf.less_equal(tf.train.get_or_create_global_step(), self.schedule)

    def __call__(self, image, conditioning=None):
        self.conditioning = conditioning
        return self.growing_discriminator(image, conditioning)

    def block(self, input, output_dim, s=1, name="block"):
        pass

    def fade_in(self, main_stream, from_rgb, weight):
        main_stream = main_stream()
        return tf.cond(tf.greater(weight, 1.0), lambda: main_stream, lambda: (weight * main_stream) + ((1.0 - weight)*from_rgb()))

    def growing_discriminator(self, image, conditioning):
        def grow(data_in, lvl):
            output_dim = self.d_mult * self.scales[lvl]
            x = lambda: self.from_rgb(data_in, self.scales, lvl)
            if lvl < (len(self.schedule) - 1):
                x = tf.cond(self.inactive[lvl + 1],
                            x,
                            lambda: self.fade_in(lambda: grow(data_in, lvl + 1), x, self.fading_weights[lvl + 1])
                            )
            else:
                x = x()
            # check if consecutive layers require downsampling
            s = 1
            if lvl > 0:
                cur_scale = self.scales[lvl]
                next_scale = self.scales[lvl - 1]
                if cur_scale != next_scale:
                    s = int(next_scale / cur_scale)
            x = self.block(x, output_dim=output_dim, s=s, name="block_lvl_" + str(lvl))
            return x

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            feat = grow(image, lvl=0)
            h4 = self.prediction_head(feat, conditioning, "pred", num_outputs=self.num_outputs)
        return h4
