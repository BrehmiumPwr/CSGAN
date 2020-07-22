from data.VOC2012Data import VOC2012Data
import tensorflow as tf
import matplotlib.pyplot as plt

VOCData = VOC2012Data(path="../datasets/VOC2012", image_set="train", image_size=128, repeat_indefinitely=True, crop_to_size_factor=True, size_factor=4, resize_in_advance=False)

tf_image, tf_label = VOCData.get_iterator().get_next()


with tf.Session() as sess:
    for x in range(100):
        features, labels = sess.run([tf_image, tf_label])
        img = features["image"][0]
        label = labels[0][0][:,:,0]
        assert img.shape[:2] == label.shape[:2]
        plt.imshow(img)
        plt.show()
        plt.imshow(label)
        plt.show()
