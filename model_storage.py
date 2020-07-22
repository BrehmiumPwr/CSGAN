import tensorflow as tf
import os
from tqdm import tqdm


def init_uninitialized(sess):
    uninitialized = []
    for var in tqdm(tf.global_variables(), desc=" [*] Checking saved variables"):
        try:
            sess.run(var)
        except Exception as ex:
            uninitialized.append(var)
    if len(uninitialized) > 0:
        for var in uninitialized:
            print(var)
        print("Warning: Some variables were not included in checkpoint. See above")
    tf.variables_initializer(uninitialized).run()


def restore(sess, checkpoint_dir, model_dir, ignore=[]):
    '''
    adapted from https://github.com/tensorflow/tensorflow/issues/312
    :param checkpoint_dir:
    :param ignore: ignore variable if variable name contains any of strings in list
    :return:
    '''
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        save_file = os.path.join(checkpoint_dir, ckpt_name)

        reader = tf.train.NewCheckpointReader(save_file)
        saved_shapes = reader.get_variable_to_shape_map()
        var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                            if var.name.split(':')[0] in saved_shapes])
        restore_vars = []
        name2var = dict(zip(map(lambda x: x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
        with tf.variable_scope('', reuse=True):
            print(" [*] Looking for matching variable names:")
            for var_name, saved_var_name in var_names:
                curr_var = name2var[saved_var_name]
                var_shape = curr_var.get_shape().as_list()
                print(curr_var, var_shape, saved_shapes[saved_var_name])
                if var_shape == saved_shapes[saved_var_name]:
                    restore_vars.append(curr_var)
                else:
                    print("shape mismatch for variable: {} | {} vs. {}".format(var_name, var_shape, saved_shapes[saved_var_name]))
        new_restore_vars = []
        if len(ignore) > 0:
            for restore_var in restore_vars:
                load = True
                for s in ignore:
                    if s in restore_var.name:
                        load = False
                if load:
                    new_restore_vars.append(restore_var)
                else:
                    print("Ignoring variable: {}".format(restore_var.name))
            restore_vars = new_restore_vars
        print(" [*] Restoring variables from {}".format(ckpt_name))
        saver = tf.train.Saver(restore_vars)
        print(restore_vars)
        saver.restore(sess, save_file)
        counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
        print(" [*] Success to read {}".format(ckpt_name))
        init_uninitialized(sess)
        return True, counter
    else:
        print(" [*] Failed to find a checkpoint")
        tf.global_variables_initializer().run()
        return False, 0
