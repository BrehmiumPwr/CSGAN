from data.ImageDataset import ImageDataset
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import codecs
import numpy as np
import img_utils

VOCData = ImageDataset(path="../datasets/cars256",
                       #image_set="train",
                       image_size=128,
                       randomize_size=True,
                       repeat_indefinitely=True,
                       crop_to_size_factor=True,
                       size_factor=4,
                       random_flip=False,
                       resize_in_advance=False)

out = VOCData.getImagesIterator().get_next()

with tf.Session() as sess:
    for x in range(100):
        features, filename = sess.run([out[0], out[3]])
        fn = codecs.decode(filename[0], "utf-8")
        img = (features[0] + 1.0) * 127.5
        image_pil = Image.open(fn)
        image_np = np.array(image_pil)
        image_np = img_utils.aspect_preserving_resize(image_np, 128)
        plt.imshow(np.array(image_np))
        plt.show()
        plt.imshow(np.uint8(img))
        plt.show()
