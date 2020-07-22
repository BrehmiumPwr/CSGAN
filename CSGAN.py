import model_storage
from gradient_clipping import adaptive_clipping_fn
from tqdm import tqdm
from PIL import Image
from itertools import chain
from losses import *
from img_utils import *
import os
import scipy
import time
from UpdateSteps import *

# nets
from nets import *


def label_to_one_hot(label, depth):
    one_hot = tf.one_hot(label, depth, on_value=1.0, off_value=0.0, axis=-1, dtype=tf.float32)
    one_hot = tf.expand_dims(one_hot, axis=[1])
    return tf.expand_dims(one_hot, axis=[1])


activations = {
    "lrelu": tf.nn.leaky_relu,
    "relu": tf.nn.relu,
    "layerlifting": cnn_ops.layer_lifting
}
optimizers = {
    "adam": lambda lr, name: tf.train.AdamOptimizer(lr, beta1=0.5, name=name),
    "sgd": lambda lr, name: tf.train.GradientDescentOptimizer(lr, name=name),
    "momentum": lambda lr, name: tf.train.MomentumOptimizer(lr, name=name),
}


class CSGAN(object):
    def __init__(self, test_fn, session, options):
        self.loaded = False

        if not isinstance(test_fn, list) and not isinstance(test_fn, tuple):
            test_fn = [test_fn]
        self.test_fn = test_fn
        self.global_step = tf.train.get_or_create_global_step()
        self.batchsize = 1
        self.learning_rate = tf.constant(options["learning_rate"])
        self.decay_lr = options["decay_lr"]

        if self.decay_lr:
            # setup learning rate schedule
            self.learning_rate = descent(start_value=self.learning_rate,
                                         end_value=0.0,
                                         start_iteration=1500000,
                                         end_iteration=2000000,
                                         global_step=self.global_step)
        tf.summary.scalar("learning_rate", self.learning_rate)

        self.optimizer = optimizers[options["optimizer"].lower()](self.learning_rate, "optimizer")
        self.save_path = os.path.join(options["model_dir"], options["model_name"])
        self.colorsupervise = options["colorsupervise"]

        self.phase = options["phase"]
        self.image_size = options["image_size"]
        self.instance_noise = options["instance_noise"]
        self.dropout = options["dropout"]
        self.GAN = options["GAN"].lower()
        self.epochs = options["epochs"]
        self.init_phase = options["init_phase"]
        self.identity_loss = None
        self.use_identity_loss = options["use_identity_loss"]
        self.identity_weight_value = options["identity_weight"]
        self.identity_on_no_object_images = options["identity_on_no_object_images"]
        self.weight_decay = options["weight_decay"]
        self.single_step_update = options["single_step_update"]
        self.weight_decay_active = self.weight_decay > 0.0

        self.session = session
        self.saver = None
        self.best_saver = None

        self.d_norm = normalization[options["discriminator_norm"]]
        self.d_act = activations[options["d_act"].lower()]
        self.d_sn = options["d_sn"]
        self.use_d_reg = options["d_reg"]
        self.g_norm = normalization[options["generator_norm"]]
        self.g_act = activations[options["g_act"].lower()]
        self.g_sn = options["g_sn"]

        self.d_mult = options["d_mult"]
        self.g_mult = options["g_mult"]

        self.istraining = tf.placeholder_with_default(False, shape=(), name="istraining")
        self.identity_weight = tf.placeholder_with_default(.2, shape=(), name="identity_weight")

        self.semisupervised = options["semisupervised"]
        self.semi_weight = options["semi_weight"]
        self.semi_use_all_classes = options["semi_use_all_classes"]
        self.semi_ignore_classes = [int(x) for x in options["semi_ignore_classes"]]

        if self.GAN == "lsgan":
            self.GAN_g_loss = SquaredError()
            self.GAN_d_loss = SquaredError()
        elif self.GAN in ["gan", "relativisticsgan", "relativisticavgsgan"]:
            self.GAN_g_loss = SigmoidXEntropyLoss()
            self.GAN_d_loss = SigmoidXEntropyLoss()
        elif self.GAN in ["wgan", "wgan-gp"]:
            self.GAN_d_loss = WassersteinLoss()
            self.GAN_g_loss = WassersteinLoss()
        elif self.GAN in ["hubergan"]:
            self.GAN_d_loss = HuberLoss()
            self.GAN_g_loss = HuberLoss()
        elif self.GAN in ["absgan"]:
            self.GAN_d_loss = AbsLoss()
            self.GAN_g_loss = AbsLoss()

        if self.phase == "train":
            self.generator_dataset = options["generator_dataset"]
            self.discriminator_dataset = options["discriminator_dataset"]
            self.valid_colors = [x for x in range(self.discriminator_dataset.count_colors())]
        else:
            self.valid_colors = options["valid_colors"]

        self.num_colors = len(self.valid_colors)
        if self.colorsupervise:
            disc_num_outputs = self.num_colors
        else:
            disc_num_outputs = 1

        self.discriminator_schedule = np.array(options["discriminator_schedule"])
        self.discriminator_scales = np.array(options["discriminator_scales"])
        self.len_fading_phase = options["fade_in_phase"]
        self.end_fading_phase = self.discriminator_schedule + self.len_fading_phase
        self.reinit_iterations = np.sort(np.concatenate([self.discriminator_schedule, self.end_fading_phase]))[2:]

        discs = {
            "growing_residual": lambda: GrowingResidualDiscriminator(norm=self.d_norm,
                                                                     spectral_norm=self.d_sn,
                                                                     activation=self.d_act,
                                                                     num_outputs=disc_num_outputs,
                                                                     istraining=self.istraining,
                                                                     d_mult=self.d_mult,
                                                                     schedule=self.discriminator_schedule,
                                                                     scales=self.discriminator_scales,
                                                                     len_fading_phase=self.len_fading_phase),
            "growing_separable_residual": lambda: GrowingSeparableResidualDiscriminator(norm=self.d_norm,
                                                                                        spectral_norm=self.d_sn,
                                                                                        activation=self.d_act,
                                                                                        num_outputs=disc_num_outputs,
                                                                                        istraining=self.istraining,
                                                                                        d_mult=self.d_mult,
                                                                                        schedule=self.discriminator_schedule,
                                                                                        scales=self.discriminator_scales,
                                                                                        len_fading_phase=self.len_fading_phase),
            "growing_bottleneck": lambda: GrowingBottleneckDiscriminator(norm=self.d_norm,
                                                                         spectral_norm=self.d_sn,
                                                                         activation=self.d_act,
                                                                         num_outputs=disc_num_outputs,
                                                                         istraining=self.istraining,
                                                                         d_mult=self.d_mult,
                                                                         schedule=self.discriminator_schedule,
                                                                         scales=self.discriminator_scales,
                                                                         len_fading_phase=self.len_fading_phase),
            "growing_mobilenet": lambda: GrowingMobilenetV2Discriminator(norm=self.d_norm,
                                                                         spectral_norm=self.d_sn,
                                                                         activation=self.d_act,
                                                                         num_outputs=disc_num_outputs,
                                                                         istraining=self.istraining,
                                                                         d_mult=self.d_mult,
                                                                         schedule=self.discriminator_schedule,
                                                                         scales=self.discriminator_scales,
                                                                         len_fading_phase=self.len_fading_phase)
        }
        disc = discs[options["discriminator"]]()
        self.discriminator = [disc]

        self.gp = [GradientPenalty(x) for x in self.discriminator]

        if self.phase == "train":
            if self.semisupervised:
                if self.semi_use_all_classes:
                    relevant_classes = [x for x in range(1, 21)]
                else:
                    relevant_classes = [7]
                [relevant_classes.remove(x) for x in self.semi_ignore_classes]
                supervision_dataset = options["supervision_dataset"]
                semi_images, semi_labels = supervision_dataset.get_iterator().get_next()

                self.semi_images = semi_images["image"]
                self.semi_labels = semi_labels[0]
                self.num_semi_classes = supervision_dataset.num_classes
            else:
                self.num_semi_classes = 1

            self.generator = Generator(norm=self.g_norm,
                                       use_aspp=True,
                                       spectral_norm=self.g_sn,
                                       use_dropout=self.dropout,
                                       activation=self.g_act,
                                       num_colors=self.num_colors,
                                       istraining=self.istraining,
                                       d_mult=self.g_mult,
                                       separate_location_and_intensity=options["separate_location_and_intensity"],
                                       share_location_and_intensity=options["share_location_and_intensity_weights"],
                                       block_type=options["block_type"],
                                       network=options["generator_network"],
                                       num_semi_supervised_classes=self.num_semi_classes)

            if self.use_identity_loss and self.identity_on_no_object_images:
                self.identity_data = options["noobject_dataset"]

            self.set_up_train_model()
        elif self.phase == "test":
            if self.semisupervised:
                if self.semi_use_all_classes:
                    relevant_classes = [x for x in range(1, 21)]
                else:
                    relevant_classes = [7]
                [relevant_classes.remove(x) for x in self.semi_ignore_classes]
                self.num_semi_classes = len([0] + relevant_classes)
            else:
                self.num_semi_classes = 2
            self.generator = Generator(norm=self.g_norm,
                                       use_aspp=True,
                                       use_dropout=self.dropout,
                                       spectral_norm=self.g_sn,
                                       activation=self.g_act,
                                       num_colors=self.num_colors,
                                       istraining=self.istraining,
                                       d_mult=self.g_mult,
                                       separate_location_and_intensity=options["separate_location_and_intensity"],
                                       share_location_and_intensity=options["share_location_and_intensity_weights"],
                                       block_type=options["block_type"],
                                       network=options["generator_network"],
                                       num_semi_supervised_classes=self.num_semi_classes)
            self.set_up_test_model()

        self.optimizer_variables = [x for x in tf.all_variables() if "optimizer" in x.name]
        self.optimizer_initializer = tf.initialize_variables(self.optimizer_variables)

    def __call__(self, image, scale_image=False):
        '''
        get segmentation for a single image
        :param image:
        :param scale_image:
        :return:
        '''
        self.load()
        if scale_image:
            image_spatial_dims = np.array(image.shape)[:2]
            image = aspect_preserving_resize(image, self.image_size, resize_method=cv2.INTER_AREA)
            # # scale smaller side to expected size
            # image_spatial_dims = np.array(image.shape)[:2]
            # smaller_side = np.min(image_spatial_dims)
            # resize_factor = self.FLAGS.image_size / smaller_side
            # target_size = (image_spatial_dims * resize_factor).astype(np.int32)
            #
            # image = Image.fromarray(image)
            # image = image.resize((target_size[1], target_size[0]), resample=Image.ANTIALIAS)
            # image = np.array(image)

        # crop a few pixels such that height and width are multiples of the generators maximum stride
        new_shape = np.array(image.shape)[:2]
        offsets = np.mod(new_shape, 4)
        real_offsets = (new_shape // 4) * 4
        image = image[:real_offsets[0], :real_offsets[1], :]

        # normalize image
        image = image.astype(np.float32) / 127.5 - 1.0

        # add batch dimension
        image = np.expand_dims(image, axis=0)

        # get result
        segmentation, augmented_images, semi_segmentation = self.session.run(
            [self.segmentation_mask, self.g_fake_color_batch, self.semi_logits_test],
            feed_dict={self.g_real_image: image, self.istraining: False})
        segmentation = segmentation[0, :, :, 0]
        semi_segmentation = np.argmax(semi_segmentation[0, :, :, :], axis=-1)
        segmentation = np.pad(segmentation, [[0, offsets[0]], [0, offsets[1]]], mode="constant", constant_values=0)
        semi_segmentation = np.pad(semi_segmentation, [[0, offsets[0]], [0, offsets[1]]], mode="constant",
                                   constant_values=0)
        augmented_images = np.pad(augmented_images, [[0, 0], [0, offsets[0]], [0, offsets[1]], [0, 0]], mode="constant",
                                  constant_values=0)

        # rescale segmentation if we downscaled in the beginning
        if scale_image:
            segmentation = Image.fromarray(segmentation)
            segmentation = segmentation.resize((image_spatial_dims[1], image_spatial_dims[0]), resample=Image.NEAREST)
            segmentation = np.array(segmentation)
            semi_segmentation = Image.fromarray(np.uint8(semi_segmentation))
            semi_segmentation = semi_segmentation.resize((image_spatial_dims[1], image_spatial_dims[0]),
                                                         resample=Image.NEAREST)
            semi_segmentation = np.array(semi_segmentation)

            result_augmentations = []
            for idx in range(augmented_images.shape[0]):
                augmented_image = augmented_images[idx]
                augmented_image = np.clip(augmented_image, a_min=-1.0, a_max=1.0)
                augmented_image = Image.fromarray(np.uint8((augmented_image + 1.0) * 127.5))
                augmented_image = augmented_image.resize((image_spatial_dims[1], image_spatial_dims[0]),
                                                         resample=Image.BICUBIC)
                augmented_image = np.array(augmented_image)
                result_augmentations.append(augmented_image)
        else:
            result_augmentations = []
            for idx in range(augmented_images.shape[0]):
                augmented_image = augmented_images[idx]
                augmented_image = np.clip(augmented_image, a_min=-1.0, a_max=1.0)
                augmented_image = (augmented_image + 1.0) * 127.5
                augmented_image = np.array(augmented_image)
                result_augmentations.append(augmented_image)

        return segmentation, semi_segmentation, result_augmentations

    def set_up_test_model(self):
        '''
        set up the model for inference only
        :return:
        '''
        self.g_real_image = tf.placeholder(dtype=tf.float32, shape=(None, None, None, 3))
        conditioning = tf.reshape(tf.eye(self.num_colors, self.num_colors),
                                  shape=(self.num_colors, 1, 1, self.num_colors))
        self.g_fake_color_batch, self.segmentation_mask, self.semi_logits_test = self.generator(self.g_real_image,
                                                                                                conditioning=conditioning)

    def set_up_train_model(self):
        '''
        set up the model for training
        :return:
        '''
        # get data

        num_colors = self.discriminator_dataset.count_colors()
        self.g_features, _ = self.generator_dataset.get_iterator().get_next()
        self.d_features, self.d_labels = self.discriminator_dataset.get_iterator().get_next()

        # self.g_real_color = self.g_labels[0]
        self.d_real_color = self.d_labels[0]
        self.g_real_image = self.g_features["image"]
        self.d_real_image = self.d_features["image"]

        # one hot encoding of colors
        # self.g_real_color_hot = label_to_one_hot(self.g_real_color, num_colors)
        self.d_real_color_hot = label_to_one_hot(self.d_real_color, num_colors)

        self.g_real_image.set_shape((self.batchsize, None, None, 3))

        # copy the image. One for each target color
        self.g_real_image_batch = tf.concat([self.g_real_image for x in range(num_colors)], axis=0)

        # create one hot encoding of target colors. It's just a square identity matrix
        self.g_fake_color_batch = tf.reshape(tf.eye(num_colors, num_colors),
                                             shape=(num_colors, 1, 1, num_colors))

        # generator takes a single image and converts it to all target colors. The mask is the same for all versions
        self.fake_images, self.segmentation_mask, self.semi_logits_test = self.generator(self.g_real_image,
                                                                                         conditioning=self.g_fake_color_batch)

        if self.semisupervised:
            _, _, self.semi_logits = self.generator(self.semi_images,
                                                    conditioning=tf.zeros_like(self.g_fake_color_batch))

        if self.use_identity_loss:
            if self.identity_on_no_object_images:
                self.identity_image_target = self.identity_data.get_iterator().get_next()[0]
                self.identity_image, _, _ = self.generator(self.identity_image_target,
                                                           conditioning=self.g_fake_color_batch)
            else:
                # the identity image is special in that it is the one for which source and target color match,
                # i.e., no changes are required
                self.identity_image_target = self.g_real_image
                self.identity_image = self.fake_images[self.g_real_color[0]:self.g_real_color[0] + 1]

        # decide whether we need a condition that we feed to our discriminators
        if self.colorsupervise:
            real_condition = None
            fake_condition = None
        else:
            real_condition = self.d_real_color_hot
            fake_condition = self.g_fake_color_batch

        # add instance noise to the input of the discriminator
        if self.instance_noise > 0.0:
            self.d_real_input = self.d_real_image + tf.random_normal(tf.shape(self.d_real_image), mean=0.0,
                                                                     stddev=self.instance_noise)
            self.d_fake_input = self.fake_images + tf.random_normal(tf.shape(self.fake_images), mean=0.0,
                                                                    stddev=self.instance_noise)
        else:
            self.d_real_input = self.d_real_image
            self.d_fake_input = self.fake_images

        # run all discriminators once on the real data.
        self.d_real_scores = [disc(self.d_real_input, conditioning=real_condition) for disc in self.discriminator]

        # run all discriminators once on a batch of images of all colors
        self.d_fake_scores = [disc(self.d_fake_input, conditioning=fake_condition) for disc in self.discriminator]

        # define loss functions
        self.generator_loss()
        self.discriminator_loss()

        # set up optimizer for generator
        self.g_optim = self.optimize_generator(self.global_step)

        # set up optimizer for each discriminator
        self.d_optim = tf.group(
            [self.optimize_discriminator(discriminator, global_step=None) for discriminator in self.discriminator])

        # set up weight decay
        if self.weight_decay_active:
            #l2_reg = [x for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if
            #          ("discriminator" in x.name) or ("generator" in x.name)]
            l2_reg = chain(*[x.variables() for x in self.discriminator])
            l2_reg = tf.add_n([tf.nn.l2_loss(x) * self.weight_decay for x in l2_reg])
            self.l2_optim = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(l2_reg)
            self.d_optim = tf.group([self.d_optim, self.l2_optim])

    def generator_loss(self):
        '''
        calculate the generator loss
        :return:
        '''

        self.g_loss = 0.0
        # when supervising with colors, we need to select the correct output neurons
        # when conditioning on colors, there is only one output neuron, i.e., no selection needed
        if self.colorsupervise:
            real_score_weight = self.d_real_color_hot
            fake_score_weight = self.g_fake_color_batch
        else:
            real_score_weight = 1.0
            fake_score_weight = 1.0

        # calculate generator loss for different gan formulations
        if self.GAN.lower() in ["gan", "lsgan", "hubergan"]:
            self.g_loss = tf.add_n(
                [self.GAN_g_loss(labels=1.0, logits=fake_score, weights=fake_score_weight) for fake_score in
                 self.d_fake_scores])
        elif self.GAN.lower() in ["absgan"]:
            self.g_loss = tf.add_n(
                [self.GAN_g_loss(labels=0.0, logits=fake_score, weights=fake_score_weight) for fake_score in
                 self.d_fake_scores])
        elif self.GAN.lower() in ["wgan", "wgan-gp"]:
            self.g_loss = tf.add_n([self.GAN_g_loss(labels=0.0, logits=fake_score * fake_score_weight) for fake_score in
                                    self.d_fake_scores])
        elif self.GAN.lower() in ["relativisticavglsgan"]:
            r_scores = [tf.losses.compute_weighted_loss(real_score, weights=real_score_weight) for real_score in
                        self.d_real_scores]
            f_scores = [tf.losses.compute_weighted_loss(fake_score, weights=fake_score_weight) for fake_score in
                        self.d_fake_scores]

            self.g_loss = tf.add_n([tf.losses.compute_weighted_loss(
                tf.square(self.d_real_scores[x] - f_scores[x] + 1.0), weights=real_score_weight) for x in
                range(len(self.discriminator))])
            self.g_loss += tf.add_n([tf.losses.compute_weighted_loss(
                tf.square(self.d_fake_scores[x] - r_scores[x] - 1.0), weights=fake_score_weight) for x in
                range(len(self.discriminator))])
        elif self.GAN in ["relativisticavgsgan"]:
            r_scores = [tf.losses.compute_weighted_loss(real_score, weights=real_score_weight) for real_score in
                        self.d_real_scores]
            f_scores = [tf.losses.compute_weighted_loss(fake_score, weights=fake_score_weight) for fake_score in
                        self.d_fake_scores]
            d_real_loss = tf.add_n(
                [self.GAN_g_loss(labels=0.0, logits=self.d_real_scores[x] - f_scores[x], weights=real_score_weight) for
                 x in
                 range(len(self.d_real_scores))])
            d_fake_loss = tf.add_n(
                [self.GAN_g_loss(labels=1.0, logits=self.d_fake_scores[x] - r_scores[x], weights=fake_score_weight) for
                 x in
                 range(len(self.d_fake_scores))])
            self.g_loss = d_real_loss + d_fake_loss
        elif self.GAN.lower() in ["relativisticsgan"]:
            r_scores = [tf.losses.compute_weighted_loss(real_score, weights=real_score_weight) for real_score in
                        self.d_real_scores]
            f_scores = [tf.losses.compute_weighted_loss(fake_score, weights=fake_score_weight) for fake_score in
                        self.d_fake_scores]
            scores = [f_scores[x] - r_scores[x] for x in range(len(r_scores))]
            self.g_loss = tf.add_n(
                [self.GAN_g_loss(labels=1.0, logits=score, weights=1.0) for score in
                 scores])

        if self.use_identity_loss:
            # the output should largely be similar to the input.
            # And there is a special case where the source and target color match.
            # In this case, we want the network to make no changes at all.
            self.identity_loss = tf.reduce_mean(
                tf.losses.absolute_difference(labels=self.identity_image_target, predictions=self.identity_image))
            if not self.identity_on_no_object_images:
                self.identity_loss += tf.reduce_mean(
                    tf.losses.absolute_difference(labels=self.identity_image_target, predictions=self.fake_images))

            # summaries
            tf.summary.scalar(name="loss/identity_loss", tensor=tf.reduce_mean(self.identity_loss))

        # self.g_loss *= self.g_real_weight

        if self.semisupervised:
            self.semi_loss = tf.losses.sparse_softmax_cross_entropy(labels=tf.cast(self.semi_labels, dtype=tf.int32),
                                                                    logits=self.semi_logits)
            tf.summary.scalar(name="loss/semi_loss", tensor=tf.reduce_mean(self.semi_loss))

            self.g_loss += self.semi_weight * self.semi_loss

        # summaries
        tf.summary.scalar(name="loss/generator_loss", tensor=tf.reduce_mean(self.g_loss))

    def discriminator_loss(self):
        '''
        calculate the discriminator/critic loss
        :return:
        '''
        # when supervising with colors, we need to select the correct output neurons
        # when conditioning on colors, there is only one output neuron, i.e., no selection needed

        self.d_loss = 0.0
        if self.colorsupervise:
            real_score_weight = self.d_real_color_hot
            fake_score_weight = self.g_fake_color_batch
        else:
            real_score_weight = 1.0
            fake_score_weight = 1.0

        # calculate discriminator loss for different gan formulations
        if self.GAN in ["gan", "lsgan", "hubergan"]:
            d_real_loss = tf.add_n(
                [self.GAN_d_loss(labels=1.0, logits=real_score, weights=real_score_weight) for real_score in
                 self.d_real_scores])
            d_fake_loss = tf.add_n(
                [self.GAN_d_loss(labels=0.0, logits=fake_score, weights=fake_score_weight) for fake_score in
                 self.d_fake_scores])
            self.d_loss = d_real_loss + d_fake_loss
        elif self.GAN in ["absgan"]:
            d_real_loss = tf.add_n(
                [self.GAN_d_loss(labels=0.0, logits=real_score, weights=real_score_weight) for real_score in
                 self.d_real_scores])
            d_fake_loss = tf.add_n(
                [self.GAN_d_loss(labels=1.0, logits=fake_score, weights=fake_score_weight) for fake_score in
                 self.d_fake_scores])
            self.d_loss = d_real_loss + d_fake_loss
        elif self.GAN in ["wgan"]:
            clip_D = []
            for x in range(len(self.discriminator)):
                clip_D += [p.assign(tf.clip_by_value(p, -0.01, 0.01)) for p in self.discriminator[x].variables()]
            with tf.control_dependencies(clip_D):
                self.d_loss = tf.add_n(
                    [-self.GAN_d_loss(labels=self.d_real_scores[x], logits=self.d_fake_scores[x]) for x in
                     range(len(self.discriminator))])

        elif self.GAN in ["wgan-gp"]:
            self.d_loss = tf.add_n(
                [-self.GAN_d_loss(labels=self.d_real_scores[x], logits=self.d_fake_scores[x]) for x in
                 range(len(self.discriminator))])

            self.d_loss += tf.add_n([10.0 * gp(fake_image=self.fake_images, real_image=self.g_real_image_batch,
                                               fake_condition=self.g_fake_color_batch,
                                               real_condition=self.g_real_color_hot) for gp in self.gp])
        elif self.GAN in ["relativisticavglsgan"]:
            r_scores = [tf.losses.compute_weighted_loss(real_score, weights=real_score_weight) for real_score in
                        self.d_real_scores]
            f_scores = [tf.losses.compute_weighted_loss(fake_score, weights=fake_score_weight) for fake_score in
                        self.d_fake_scores]
            self.d_loss = tf.add_n([tf.losses.compute_weighted_loss(
                tf.square(self.d_real_scores[x] - f_scores[x] - 1.0), weights=real_score_weight) for x in
                range(len(self.discriminator))])
            self.d_loss += tf.add_n([tf.losses.compute_weighted_loss(
                tf.square(self.d_fake_scores[x] - r_scores[x] + 1.0), weights=fake_score_weight) for x in
                range(len(self.discriminator))])
        elif self.GAN in ["relativisticavgsgan"]:
            r_scores = [tf.losses.compute_weighted_loss(real_score, weights=real_score_weight) for real_score in
                        self.d_real_scores]
            f_scores = [tf.losses.compute_weighted_loss(fake_score, weights=fake_score_weight) for fake_score in
                        self.d_fake_scores]
            d_real_loss = tf.add_n(
                [self.GAN_d_loss(labels=1.0, logits=self.d_real_scores[x] - f_scores[x], weights=real_score_weight) for
                 x in range(len(self.d_real_scores))])
            d_fake_loss = tf.add_n(
                [self.GAN_d_loss(labels=0.0, logits=self.d_fake_scores[x] - r_scores[x], weights=fake_score_weight) for
                 x in range(len(self.d_fake_scores))])
            self.d_loss = d_real_loss + d_fake_loss
        elif self.GAN.lower() in ["relativisticsgan"]:
            r_scores = [tf.losses.compute_weighted_loss(real_score, weights=real_score_weight) for real_score in
                        self.d_real_scores]
            f_scores = [tf.losses.compute_weighted_loss(fake_score, weights=fake_score_weight) for fake_score in
                        self.d_fake_scores]
            scores = [r_scores[x] - f_scores[x] for x in range(len(r_scores))]
            self.d_loss = tf.add_n(
                [self.GAN_d_loss(labels=1.0, logits=score, weights=1.0) for score in scores])

        if self.GAN == "gan":
            self.d_reg_act = tf.nn.sigmoid
        elif self.GAN == "lsgan":
            self.d_reg_act = tf.square
        elif self.GAN == "absgan":
            self.d_reg_act = tf.abs

        if self.use_d_reg:
            '''self.d_loss += tf.add_n([(0.1 / 2.0) * Discriminator_Regularizer(self.d_real_scores[x],
                                                                             self.d_real_image,
                                                                             real_score_weight,
                                                                             self.d_fake_scores[x],
                                                                             self.fake_images,
                                                                             fake_score_weight,
                                                                             act=self.d_reg_act) for x in
                                     range(len(self.discriminator))])'''
            self.d_loss += tf.add_n([Discriminator_Regularizer2(score_real=self.d_real_scores[x],
                                                                              x_real=self.d_real_image
                                                                             ) for x in
                                     range(len(self.discriminator))])

        # apply loss weights
        # self.d_loss *= self.d_real_weight

        # summaries
        tf.summary.scalar("loss/discriminator_loss", tf.reduce_mean(self.d_loss))

    def optimize_discriminator(self, discriminator, global_step=None):
        '''
        defines the optimization/gradient descent routine for a single discriminator
        :param discriminator: the output of the discriminator
        :param global_step:
        :return:
        '''
        # calculate gradients of the error with respect to given discriminator
        # grads_and_vars = self.optimizer.compute_gradients(self.d_loss, var_list=discriminator.variables())
        d_vars = discriminator.variables()
        grads_disc = tf.gradients(self.d_loss, d_vars, name="gan_disc_gradients",
                                  unconnected_gradients='zero')
        grads_and_vars = list(zip(grads_disc, d_vars))
        # [tf.summary.histogram("discriminator_gradients_raw/{}".format(var.name), grad) for grad, var in grads_and_vars]
        [tf.summary.scalar("discriminator_gradients_raw/{}".format(var.name), tf.reduce_mean(tf.sqrt(grad ** 2))) for
         grad, var in grads_and_vars]

        # clip gradients
        clip_fn = adaptive_clipping_fn(global_step=global_step, static_max_norm=5.0)
        grads_and_vars = clip_fn(grads_and_vars)
        # [tf.summary.histogram("discriminator_gradients/{}".format(var.name), grad) for grad, var in grads_and_vars]
        [tf.summary.scalar("discriminator_gradients/{}".format(var.name), tf.reduce_mean(tf.sqrt(grad ** 2))) for
         grad, var in grads_and_vars]

        # update weights
        return self.optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    def optimize_generator(self, global_step=None):
        '''
        defines the optimization/gradient descent routine for the generator
        :param global_step:
        :return:
        '''
        # calculate gradients of the gan error with respect to the generator
        g_vars = self.generator.variables()
        grads_gan = tf.gradients(self.g_loss, g_vars, name="gan_gen_gradients",
                                 unconnected_gradients='zero')
        grads_and_vars = list(zip(grads_gan, g_vars))
        [tf.summary.scalar("generator_gradients_gan_raw/{}".format(var.name), tf.reduce_mean(tf.sqrt(grad ** 2))) for
         grad, var in grads_and_vars]

        if self.use_identity_loss:
            # calculate gradients of the identity error with respect to the generator.
            # This time we stop backpropagation into the color prediction network
            grads_identity = tf.gradients(self.identity_loss, g_vars,
                                          gate_gradients=True,
                                          name="identity_gradients",
                                          stop_gradients=[self.generator.color_vec],
                                          unconnected_gradients='zero')
            grads_and_vars_identity = list(zip(grads_identity, g_vars))
            [tf.summary.scalar("generator_gradients_identity_raw/{}".format(var.name),
                               tf.reduce_mean(tf.sqrt(grad ** 2)))
             for grad, var in grads_and_vars_identity]

            # merge gradients from both errors
            new_grads_and_vars = []
            for grad, var in grads_and_vars:
                match = [x for x in grads_and_vars_identity if x[1].name == var.name]
                if len(match) > 0:
                    match_grad, match_var = match[0]
                    match_grad_magnitude = tf.reduce_mean(tf.abs(match_grad))
                    grad_magnitude = tf.reduce_mean(tf.abs(grad))
                    ratio = tf.cond(tf.greater(match_grad_magnitude, 0),
                                    lambda: grad_magnitude / ((1.0 / self.identity_weight) * match_grad_magnitude),
                                    lambda: 1.0)
                    # ratio = tf.Print(ratio, [match_grad_magnitude, grad_magnitude])
                    match_grad = match_grad * ratio
                    new_grad = tf.add(grad, match_grad)  # /2.0
                    new_grads_and_vars.append((new_grad, var))
                else:
                    new_grads_and_vars.append((grad, var))
        else:
            new_grads_and_vars = grads_and_vars

        # clip gradients
        clip_fn = adaptive_clipping_fn(global_step=self.global_step, static_max_norm=5.0)
        new_grads_and_vars = clip_fn(new_grads_and_vars)
        [tf.summary.scalar("generator_gradients/{}".format(var.name), tf.reduce_mean(tf.sqrt(grad ** 2))) for grad, var
         in new_grads_and_vars]

        # update weights
        return self.optimizer.apply_gradients(new_grads_and_vars, global_step=global_step)

    def load(self, subpath=None):
        if not self.loaded:
            if subpath is not None:
                load_path = os.path.join(self.save_path, subpath)
            else:
                load_path = self.save_path
            # restore if checkpoint available
            model_storage.restore(self.session, load_path, "")
            self.loaded = True

    def train(self):
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        self.load()
        # set up summary writer
        summaries = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(self.save_path)  # , graph=self.session.graph)

        # how many steps do we need to touch the sky?
        num_samples = self.generator_dataset.count()
        total_steps = 2000000  # self.epochs * num_samples

        # get current step and start from there
        cur_step = self.session.run(self.global_step)
        pbar = tqdm(range(cur_step, total_steps))

        # in the beginning we do not use the identity loss
        identity_weight = 0.0

        max_iou = 0.0
        steps_per_second = 0.0

        if self.single_step_update:
            update_step = SingleStepUpdate(model=self)
        else:
            update_step = DualStepWithDataCacheUpdate(model=self)
        # kick off training
        for x in pbar:
            start = time.time()
            gen_loss, disc_loss, cur_step, lr = update_step()

            # measure speed and smooth over time
            steps_per_second = (steps_per_second * 0.95) + (0.05 / (time.time() - start))

            # update progress bar
            pbar.set_description(
                "Training Epoch: {}, Loss (d/g): {:.2f}/{:.2f}, LR: {:.7f}".format(cur_step // num_samples,
                                                                                   disc_loss,
                                                                                   gen_loss, lr))

            # write summaries and evaluate
            if x % 5000 == 0 or total_steps - x < 2:
                logs = self.session.run(summaries, feed_dict={self.istraining: True,
                                                              self.identity_weight: identity_weight})
                summary_writer.add_summary(logs, global_step=x)
                # is the init phase over? If so, we start using the identity loss
                if cur_step > self.init_phase:
                    identity_weight = self.identity_weight_value
                else:
                    identity_weight = 0.0

                results = []
                summary_values = []
                if len(self.test_fn) > 0:
                    result = self.test_fn[0](self)
                    score, thresh = result["IoU"], result["threshold"]
                    [summary_values.append(tf.Summary.Value(tag=key, simple_value=result[key])) for key in
                     result.keys()]
                    results.append(result)
                    for test_fn in self.test_fn[1:]:
                        result = test_fn(self, thresh)
                        results.append(result)
                        [summary_values.append(tf.Summary.Value(tag=key, simple_value=result[key])) for key in
                         result.keys()]
                    summary_values.append(tf.Summary.Value(tag="steps_per_second", simple_value=steps_per_second))
                    summary = tf.Summary(value=summary_values)
                    summary_writer.add_summary(summary, x)
                    self.save()
                    self.sample_during_train()
                    if len(results) > 0:
                        if results[0]["IoU"] > max_iou:
                            self.save_best()
                            max_iou = results[0]["IoU"]

    def sample_during_train(self):
        '''
        write an example image as well as all corresponding versions with modified colors and the learned segmentation mask
        :return:
        '''

        # min max norm
        def norm(arr):
            arr -= np.min(arr)
            return arr / (np.max(arr) + 1e-8)

        if not os.path.exists(os.path.join(self.save_path, "samples")):
            os.makedirs(os.path.join(self.save_path, "samples"), exist_ok=True)
        output_path = os.path.join(self.save_path, "samples",
                                   'sample_{}.png'.format(self.session.run(self.global_step)))
        real_img = self.session.run(self.g_real_image)
        fake_images, delta_img, segmentation = self.sample(real_img[0])
        out_img = np.concatenate([norm(real_img[0]),
                                  np.tile(norm(segmentation), [1, 1, 3]),
                                  norm(delta_img)] +
                                 [norm(fake_images[x]) for x in range(fake_images.shape[0])],
                                 axis=1)
        scipy.misc.imsave(output_path, out_img)

    def sample(self, real_img):
        target_colors = self.discriminator_dataset.color_count
        one_hot_color_label = np.reshape(np.identity(target_colors), newshape=(target_colors, 1, 1, target_colors))
        fake_img_batch, segmentation = self.session.run([self.fake_images, self.segmentation_mask],
                                                        feed_dict={self.g_fake_color_batch: one_hot_color_label,
                                                                   self.g_real_image: np.expand_dims(real_img, axis=0)}
                                                        )
        segmentation = segmentation[0]
        delta_img = np.zeros_like(real_img)
        for x in range(fake_img_batch.shape[0]):
            fake_img = fake_img_batch[x]
            delta_img += np.abs(real_img - fake_img)

        return fake_img_batch, delta_img, segmentation

    def save(self):
        '''
        write checkpoint
        :return:
        '''
        if self.saver is None:
            self.saver = tf.train.Saver()

        if self.session is not None:
            os.makedirs(self.save_path, exist_ok=True)
            self.saver.save(self.session, os.path.join(self.save_path, "model.ckpt"), self.global_step)
        else:
            print("Warning: Cannot save model! No active session")

    def save_best(self):
        '''
        write checkpoint
        :return:
        '''
        if self.best_saver is None:
            self.best_saver = tf.train.Saver()
        if self.session is not None:
            path = os.path.join(self.save_path, "best")
            os.makedirs(path, exist_ok=True)
            self.best_saver.save(self.session, os.path.join(path, "model.ckpt"), self.global_step)
        else:
            print("Warning: Cannot save model! No active session")
