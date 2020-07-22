import tensorflow as tf



class LossFunction:
    def __init__(self):
        self.loss = 0

    def reduction(self, reduction):
        if reduction.lower() == "mean":
            return tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS
        elif reduction.lower() == "sum":
            return tf.losses.Reduction.SUM
        else:
            print("invalid reduction")

    def __call__(self, logits, labels, weights=1.0, reduction="mean"):
        pass


class SigmoidXEntropyLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, labels, weights=1.0, reduction="mean"):
        reduce = super().reduction(reduction)
        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(logits) * labels,
                                                    logits=logits,
                                                    weights=weights,
                                                    reduction=reduce)
        return self.loss


class SquaredError(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, labels, weights=1.0, reduction="mean"):
        reduce = super().reduction(reduction)
        self.loss = tf.losses.mean_squared_error(labels=tf.ones_like(logits) * labels,
                                                 predictions=logits,
                                                 weights=weights,
                                                 reduction=reduce)
        return self.loss


class HuberLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, labels, weights=1.0, reduction="mean"):
        reduce = super().reduction(reduction)
        self.loss = tf.losses.huber_loss(labels=tf.ones_like(logits) * labels,
                                         predictions=logits,
                                         weights=weights,
                                         delta=1.0,
                                         reduction=reduce)
        return self.loss


class WassersteinLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, labels, weights=1.0, reduction="mean"):
        # labels = real score
        # logits = fake score
        self.loss = tf.reduce_mean(labels) - tf.reduce_mean(logits)
        return self.loss

class SmoothedAbsLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, labels, weights=1.0, reduction="mean"):
        # labels = real score
        # logits = fake score
        self.loss = tf.abs(tf.nn.tanh(logits-labels))
        if reduction =="mean":
            self.loss = tf.reduce_mean(self.loss)
        elif reduction == "sum":
            self.loss = tf.reduce_sum(self.loss)
        return self.loss

class AbsLoss(LossFunction):
    def __init__(self):
        super().__init__()

    def __call__(self, logits, labels, weights=1.0, reduction="mean"):
        # labels = real score
        # logits = fake score
        self.loss = tf.abs(logits-labels)
        if reduction =="mean":
            self.loss = tf.reduce_mean(self.loss)
        elif reduction == "sum":
            self.loss = tf.reduce_sum(self.loss)
        return self.loss

class GradientPenalty(LossFunction):
    def __init__(self, discriminator):
        super().__init__()
        self.discriminator = discriminator

    def __call__(self, fake_image, real_image, fake_condition, real_condition, weights=1.0, reduction="mean"):
        # Gradient Penalty
        self.epsilon = tf.random_uniform(
            shape=tf.stack([tf.shape(fake_image)[0], 1, 1, 1]),
            minval=0.,
            maxval=1.)
        mixed_image = real_image + self.epsilon * (fake_image - real_image)
        mixed_condition = real_condition + self.epsilon * (fake_condition - real_condition)
        D_X_hat = self.discriminator(mixed_image, mixed_condition)
        grad_D_X_hat = tf.gradients(D_X_hat, [mixed_image])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad_D_X_hat), reduction_indices=[1, 2, 3]))
        self.loss = tf.square(slopes - 1.0)
        return self.loss

def Discriminator_Regularizer2(x_real, score_real, gamma=10.0):
    gradients_x = tf.gradients(score_real, x_real)[0]
    disc_reg = gamma * 0.5 * tf.reduce_mean(tf.square(gradients_x))
    tf.summary.scalar("disc_regularization", disc_reg)
    return disc_reg


def Discriminator_Regularizer(D1_logits, D1_arg, D1_weights, D2_logits, D2_arg, D2_weights, act=tf.nn.sigmoid):
    # https://github.com/rothk/Stabilizing_GANs
    D1_logits = tf.reduce_mean(D1_logits, axis=[1, 2], keepdims=True) * D1_weights
    D2_logits = tf.reduce_mean(D2_logits, axis=[1, 2], keepdims=True) * D2_weights
    D1 = act(D1_logits) * D1_weights
    D2 = act(D2_logits) * D2_weights
    grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
    grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
    batch_size = tf.shape(D1_logits)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, tf.stack([batch_size, 1, 1, -1])), axis=-1, keepdims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, tf.stack([batch_size, 1, 1, -1])), axis=-1, keepdims=True)

    # set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
    # assert grad_D1_logits_norm.shape == D1.shape
    # assert grad_D2_logits_norm.shape == D2.shape

    reg_D1 = tf.multiply(tf.square(1.0 - D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer