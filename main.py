import CSGAN
import tensorflow as tf
import os
from tabulate import tabulate
from subprocess import call
import signal
import sys
import json
from absl import app
from absl import flags
from dataset_definitions import cars, alpaca, handbag

tf.disable_v2_behavior()

FLAGS = flags.FLAGS

schedules = {
    "60k_12_long": [0, 120000, 240000, 360000, 480000, 600000, 720000, 840000, 960000, 1080000, 1200000, 1320000],
    "60k_12": [0, 60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000, 540000, 600000, 660000],
    "60k_16": [0, 60000, 120000, 180000, 240000, 300000, 360000, 420000, 480000, 540000, 600000, 660000, 720000, 780000,
               840000, 900000],
    "120k_12": [0, 120000, 240000, 360000, 480000, 600000, 720000, 840000, 960000, 1080000, 1200000, 1320000],

}
scales = {
    "12": [8, 8, 8, 8, 4, 4, 4, 4, 4, 2, 2, 2],
    "12_small": [8, 8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4],
    "16": [16, 16, 16, 16, 8, 8, 8, 8, 4, 4, 4, 4, 4, 2, 2, 2],
}

datasets = {
    "cars": lambda options: cars(options),
    "alpaca": lambda options: alpaca(options),
    "handbag": lambda options: handbag(options),
}

# general
flags.DEFINE_string('phase', "train", "phase to run. train or test")
flags.DEFINE_string('gpu', "4", "GPU to use. Empty string for CPU mode")
flags.DEFINE_string('model_dir', "models", 'folder in which trained models are stored')
flags.DEFINE_string('model_name', "debug", 'name of the current model')

# model details
flags.DEFINE_string('GAN', 'gan', 'gan loss. gan, lsgan, hubergan, relativistic, wgan, wgan-gp')

# training
flags.DEFINE_integer('epochs', 500, "number of epochs to train")
flags.DEFINE_integer('image_size', 128, "size of training images")
flags.DEFINE_float('learning_rate', 0.0001, "initial learning rate")
flags.DEFINE_bool('decay_lr', True, "whether to decay the learning rate over time")
flags.DEFINE_bool('dropout', True, "whether to use dropout or not")
flags.DEFINE_string('optimizer', "adam", "adam, sgd, momentum")
flags.DEFINE_float('identity_weight', .2, "multiplier for identity loss")
flags.DEFINE_bool('use_identity_loss', False, "use_identity_loss or not")
flags.DEFINE_bool('identity_on_no_object_images', True,
                  "calculate the identity loss on images without any relevant objects or use the normal dataset")
flags.DEFINE_bool('single_step_update', False,
                  "update discriminator and generator at once")
flags.DEFINE_bool('colorsupervise', True, "supervise with colors?")
flags.DEFINE_integer('init_phase', 0, "number of steps to run without identity loss")
flags.DEFINE_float('instance_noise', 0.0, "stddev of normal distribution used to generate instance noise")
flags.DEFINE_float('weight_decay', 0.0, "weight decay")
flags.DEFINE_string("data", "cars", "path of the generator data")

# generator
flags.DEFINE_string('block_type', "separable_residual", "blocks to use in the generator [residual, bottleneck]")
flags.DEFINE_string('generator_network', "resnet", "type of generator network [resnet, dualpath]")
flags.DEFINE_string('generator_norm', "none",
                    "feature norm used in generator [instancenorm, groupnorm, layernorm, none]")
flags.DEFINE_bool('separate_location_and_intensity', False,
                  'whether location and intensity should be predicted separately')
flags.DEFINE_bool('share_location_and_intensity_weights', False,
                  "if location and intensity are separated, should we use the same weights to predict them")
flags.DEFINE_bool('semisupervised', False, "Use labelled dataset for supervision of generator")
flags.DEFINE_bool('semi_use_all_classes', True, "Use all classes from semi dataset")
flags.DEFINE_list("semi_ignore_classes", [], "classes to ignore when using semi supervision")
flags.DEFINE_float('semi_weight', 1.0, "weight of semi loss")
flags.DEFINE_bool('g_sn', True, "use spectral normalization in generator")
flags.DEFINE_integer('g_mult', 64, "base channel multiplier generator")
flags.DEFINE_string('g_act', 'lrelu', 'activation function used in the generator')

# discriminator
flags.DEFINE_string('discriminator', 'growing_separable_residual', 'the discriminator architecture to use')
flags.DEFINE_string('discriminator_norm', "none",
                    "feature norm used in discriminator [instancenorm, groupnorm, layernorm, none]")
flags.DEFINE_list('discriminator_schedule', schedules["60k_12"], 'discriminator growing schedule')
flags.DEFINE_list('discriminator_scales', scales["12"], 'discriminator scales')
flags.DEFINE_integer('fade_in_phase', 30000, "number of steps used for fading in new layers")
flags.DEFINE_bool('d_reg', True, 'enable or disable discriminator regularization')
flags.DEFINE_bool('d_sn', True, "use spectral normalization in discriminator")
flags.DEFINE_integer('d_mult', 64, "base channel multiplier discriminator")
flags.DEFINE_string('d_act', 'lrelu', 'activation function used in the discriminator')



# kill switches
def signal_handler(sig, frame, clean_fn=None):
    print('Caught external kill signal')
    if clean_fn is not None:
        print("Cleaning up!")
        clean_fn()
    sys.exit(0)


def register_killswitch():
    signal.signal(signal.SIGINT, signal_handler)
    # signal.signal(signal.SIGKILL, signal_handler)


def build_model():
    register_killswitch()
    if FLAGS.phase.lower() == "train":
        commit_str = "\"Model at experiment {}\"".format(FLAGS.model_name)
        call(["git", "add", "."])
        call(["git", "commit", "-m", commit_str])
        call(["git", "push", "-u", "origin", "dev"])
    elif FLAGS.phase.lower() == "test":
        commit_str = os.popen("git log --grep={} | grep commit".format(FLAGS.model_name)).readlines()
        if len(commit_str) > 0:
            commit_str = commit_str[0].strip("\n").replace("commit ", "")
        else:
            print("Could not find matching commit in history")

    cmdline_options = [(key, FLAGS[key].value) for key in FLAGS.flag_values_dict()]
    print(tabulate(cmdline_options, headers=["key", "value"]), flush=True)

    # convert command line options to options dictionary
    options = dict(cmdline_options)
    path = os.path.join(options["model_dir"], options["model_name"])
    filename = os.path.join(path, "options.json")
    os.makedirs(path, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(options, f)

    config = tf.ConfigProto()
    config.graph_options.place_pruned_graph = True
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF

    with tf.Session(config=config) as sess:

        options, test_fn = datasets[options["data"]](options)

        model = CSGAN.CSGAN(test_fn=test_fn,
                            session=sess,
                            options=options)
        if FLAGS.phase.lower() == "train":
            model.train()
        else:
            print("Unknown phase")


def main(argv):
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
    build_model()


if __name__ == '__main__':
    app.run(main)
