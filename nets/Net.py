import tensorflow as tf

class Net(object):
    '''parent network class. Basically just to get list of variables for all networks'''
    def __init__(self, name="net"):
        self.name = name

    def variables(self):
        t_vars = tf.trainable_variables()
        return [var for var in t_vars if self.name in var.name]