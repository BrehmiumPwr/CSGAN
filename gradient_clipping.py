import tensorflow as tf
from tensorflow.python.training import moving_averages


def _adaptive_max_norm(norm, std_factor, decay, global_step, epsilon, name):
    """Find max_norm given norm and previous average."""
    with tf.variable_scope(name, "AdaptiveMaxNorm", [norm]):
        log_norm = tf.log(norm + epsilon)

        def moving_average(name, value, decay):
            moving_average_variable = tf.get_variable(
                name,
                shape=value.get_shape(),
                dtype=value.dtype,
                initializer=tf.zeros_initializer(),
                trainable=False)
            return moving_averages.assign_moving_average(
                moving_average_variable, value, decay, zero_debias=False)

        # quicker adaptation at the beginning
        if global_step is not None:
            n = tf.to_float(global_step)
            decay = tf.minimum(decay, n / (n + 1.))

        # update averages
        mean = moving_average("mean", log_norm, decay)
        sq_mean = moving_average("sq_mean", tf.square(log_norm), decay)

        variance = sq_mean - tf.square(mean)
        std = tf.sqrt(tf.maximum(epsilon, variance))
        max_norms = tf.exp(mean + std_factor * std)


    return max_norms, mean


def adaptive_clipping_fn(std_factor=2.,
                         decay=0.95,
                         static_max_norm=None,
                         global_step=None,
                         report_summary=False,
                         epsilon=1e-8,
                         name=None):
    """Adapt the clipping value using statistics on the norms.
    Implement adaptive gradient as presented in section 3.2.1 of
    https://arxiv.org/abs/1412.1602.
    Keeps a moving average of the mean and std of the log(norm) of the gradient.
    If the norm exceeds `exp(mean + std_factor*std)` then all gradients will be
    rescaled such that the global norm becomes `exp(mean)`.
    Args:
      std_factor: Python scaler (or tensor).
        `max_norm = exp(mean + std_factor*std)`
      decay: The smoothing factor of the moving averages.
      static_max_norm: If provided, will threshold the norm to this value as an
        extra safety.
      global_step: Optional global_step. If provided, `decay = decay*n/(n+1)`.
        This provides a quicker adaptation of the mean for the first steps.
      report_summary: If `True`, will add histogram summaries of the `max_norm`.
      epsilon: Small value chosen to avoid zero variance.
      name: The name for this operation is used to scope operations and summaries.
    Returns:
      A function for applying gradient clipping.
    """

    def gradient_clipping(grads_and_vars):
        """Internal function for adaptive clipping."""
        grads, variables = zip(*grads_and_vars)

        norm = tf.global_norm(grads)

        max_norm, log_mean = _adaptive_max_norm(norm, std_factor, decay,
                                                global_step, epsilon, name)

        # reports the max gradient norm for debugging
        if report_summary:
            tf.summary.scalar("global_norm/adaptive_max_gradient_norm", max_norm)

        # factor will be 1. if norm is smaller than max_norm
        factor = tf.where(norm < max_norm,
                          tf.ones_like(norm),
                          tf.exp(log_mean) / norm)

        if static_max_norm is not None:
            factor = tf.minimum(static_max_norm / norm, factor)

        # apply factor
        clipped_grads = []
        for grad in grads:
            if grad is None:
                clipped_grads.append(None)
            elif isinstance(grad, tf.IndexedSlices):
                clipped_grads.append(
                    tf.IndexedSlices(grad.values * factor, grad.indices,
                                     grad.dense_shape))
            else:
                clipped_grads.append(grad * factor)

        return list(zip(clipped_grads, variables))

    return gradient_clipping
