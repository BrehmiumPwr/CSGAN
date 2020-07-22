import numpy as np
import tensorflow as tf
import cv2

def aspect_preserving_resize(image, size, resize_method=cv2.INTER_AREA):
    # scale larger side to expected size
    image_spatial_dims = np.array(image.shape)[:2]
    larger_side = np.max(image_spatial_dims)
    resize_factor = size / larger_side
    target_size = (image_spatial_dims * resize_factor).astype(np.int32)
    image = cv2.resize(image, (target_size[1], target_size[0]), interpolation=resize_method)
    return image


def random_flip_left_right(image, uniform_random):
    """Randomly (50% chance) flip an image along axis `flip_index`.

    Args:
      image: 4-D Tensor of shape `[batch, height, width, channels]` or
             3-D Tensor of shape `[height, width, channels]`.
      flip_index: Dimension along which to flip image. Vertical: 0, Horizontal: 1
      seed: A Python integer. Used to create a random seed. See
        `tf.set_random_seed`
        for behavior.
      scope_name: Name of the scope in which the ops are added.

    Returns:
      A tensor of the same type and shape as `image`.

    Raises:
      ValueError: if the shape of `image` not supported.
    """
    shape = image.get_shape()
    if shape.ndims == 3 or shape.ndims is None:
        mirror_cond = tf.less(uniform_random, .5)
        result = tf.cond(
            mirror_cond,
            lambda: tf.reverse(image, [1]),
            lambda: image,
        )
        result.set_shape(shape)
        return result
    elif shape.ndims == 4:
        batch_size = tf.shape(image)[0]
        flips = tf.round(
            tf.reshape(uniform_random, [batch_size, 1, 1, 1])
        )
        flips = tf.cast(flips, image.dtype)
        flipped_input = tf.reverse(image, [2])
        return flips * flipped_input + (1 - flips) * image
    else:
        raise ValueError('\'image\' must have either 3 or 4 dimensions.')