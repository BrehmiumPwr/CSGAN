import tensorflow as tf
from resize_utils import *
from tqdm import tqdm
import os
from img_utils import random_flip_left_right
from PIL import Image

dtype_mappings = {
    'uint8': tf.uint8,
    "int64": tf.int32
}

class GenericDataset(object):

    def __init__(self, path, batch_size=1, randomize_size=True, random_flip=True, repeat_indefinitely=True,
                 square_pad=False, random_crop=False, random_brightness=False, random_contrast=False,
                 random_saturation=False, crop_to_size_factor=False, image_size=128, size_factor=4,
                 resize_in_advance=True):
        self.path = path
        print("[*] Preparing data from {}".format(path), flush=True)
        self.resize_in_advance = resize_in_advance
        self.batch_size = batch_size
        self.randomize_size = randomize_size
        self.random_flip = random_flip
        self.repeat = repeat_indefinitely
        self.square_pad = square_pad
        self.random_crop = random_crop
        self.crop_to_size_factor = crop_to_size_factor
        self.target_size = image_size
        self.size_factor = size_factor
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_saturation = random_saturation
        self.data = self.get_data()
        self.image_names = [x for x in self.data.keys()]
        self.labels = [self.data[x] for x in self.image_names]

        self.dtypes = [tf.string]
        if len(self.labels) > 0:
            for x in self.labels[0]:
                dtype = type(x)
                if dtype == np.ndarray:
                    self.dtypes.append(dtype_mappings[str(x.dtype)])
                else:
                    self.dtypes.append(dtype_mappings[str(x.dtype)])
            self.is_spatial_label = self.label_structure()

        self.dtypes = tuple(self.dtypes)

        self.adjust_size()

        self.prepared_data = []
        for x in range(len(self.image_names)):
            output = [self.image_names[x]]
            label = self.labels[x]
            for l in label:
                output.append(l)
            self.prepared_data.append(tuple(output))

        print("[**] Found {} examples".format(len(self.prepared_data)), flush=True)

    def read_spatial_gt(self, filename):
        return np.array(Image.open(filename))#np.array(cv2.imread(filename, cv2.IMREAD_UNCHANGED))

    def read_image(self, filename):
        return np.array(cv2.imread(filename, cv2.IMREAD_COLOR))[:,:,::-1]

    def load_and_resize_image(self, infilename, outfilename, larger_side):
        img = Image.open(infilename)
        img = img.convert("RGB")
        img.thumbnail([larger_side, larger_side])
        img.save(outfilename, "PNG", icc_profile=None)

    def label_structure(self):
        spatial_label = []
        if type(self.labels[0]) in [tuple, list]:
            for x in range(len(self.labels[0])):
                spatial_label.append(self.is_spatial_data(self.labels[0][x]))
        return spatial_label

    def is_spatial_data(self, data):
        is_spatial = False
        try:
            data_shape = data.shape
            if len(data_shape) > 1:
                is_spatial = True
        except:
            pass
        return is_spatial

    def adjust_size(self):
        if self.resize_in_advance:
            resize_size = self.target_size * 2
            target_folder = self.path + "_" + str(resize_size)
            new_files = []
            for x in tqdm(range(len(self.image_names)), desc="resizing images to {}".format(resize_size)):
                src_file = self.image_names[x]
                target_file = src_file.replace(self.path, target_folder)
                target_file = target_file.replace(os.path.splitext(target_file)[-1], ".png")
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                if not os.path.exists(target_file):
                    # in case any files are not readable
                    try:
                        self.load_and_resize_image(src_file, target_file, larger_side=resize_size)
                        new_files.append(target_file)
                    except Exception as e:
                        print(e)
                else:
                    new_files.append(target_file)
            self.image_names = new_files

    def get_data(self):
        pass

    def data_generator(self):
        for x in range(len(self.image_names)):
            yield self.prepared_data[x]

    def get_iterator(self):
        dataset = tf.data.Dataset.from_generator(self.data_generator, self.dtypes)

        if not self.repeat:
            dataset = dataset.shuffle(buffer_size=len(self.image_names))
        else:
            dataset = dataset.shuffle(buffer_size=len(self.image_names))
            dataset = dataset.repeat()

        dataset = dataset.map(self._loadImageFunction, num_parallel_calls=20)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.prefetch(25)

        iterator = dataset.make_one_shot_iterator()

        return iterator


    def crop_to_size_fac(self, image, target_size):
        image_decoded = aspect_preserving_resize(image, target_size=target_size, resize_mode="pad")
        resized_size = tf.shape(image_decoded)[:2]
        target_crop_size = (resized_size // self.size_factor) * self.size_factor
        image_decoded = image_decoded[:target_crop_size[0], :target_crop_size[1], :]
        return image_decoded

    def _loadImageFunction(self, filename, *args):
        labels = [x for x in args]
        for x in range(len(labels)):
            if self.is_spatial_label[x]:
                labels[x].set_shape([None, None, 1])
            else:
                labels[x].set_shape(())

        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, 3)
        image_decoded.set_shape([None, None, 3])
        image_decoded = tf.cast(image_decoded, tf.float32)
        image_decoded = image_decoded / 255.0

        if self.randomize_size:
            # max_upper_dev = int((self.target_size * 0.3) // self.size_factor)
            # max_lower_dev = -int((self.target_size * 0.3) // self.size_factor)
            # delta_size = tf.random_uniform(shape=(), minval=max_lower_dev, maxval=max_upper_dev,
            #                               dtype=tf.int32) * self.size_factor
            delta_size = tf.truncated_normal(shape=(), mean=0.0, stddev=self.target_size * 0.15, dtype=tf.float32)
            delta_size = tf.cast(delta_size, dtype=tf.int32)
        else:
            delta_size = 0
        image_size = self.target_size + delta_size
        if self.random_crop:
            target_size = tf.cast(tf.multiply(tf.cast(image_size, dtype=tf.float32), 1.3), dtype=tf.int32)
            image_decoded = aspect_preserving_resize(image_decoded, target_size=target_size, resize_mode="crop")
            image_decoded = tf.random_crop(image_decoded, [image_size, image_size, 3])
        elif self.crop_to_size_factor:
            image_decoded = self.crop_to_size_fac(image_decoded, target_size=image_size)
            for x in range(len(labels)):
                if self.is_spatial_label[x]:
                    labels[x] = self.crop_to_size_fac(labels[x], target_size=image_size)
        else:
            image_decoded = aspect_preserving_resize(image_decoded, target_size=image_size, resize_mode="pad")
            for x in range(len(labels)):
                if self.is_spatial_label[x]:
                    labels[x] = aspect_preserving_resize(labels[x], target_size=image_size, resize_mode="pad")

        if self.random_brightness:
            image_decoded = tf.image.random_brightness(image_decoded, max_delta=0.2)
        if self.random_contrast:
            image_decoded = tf.image.random_contrast(image_decoded, lower=.9, upper=1.1)
        if self.random_saturation:
            image_decoded = tf.image.random_saturation(image_decoded, lower=.9, upper=1.1)
        # image_decoded = tf.py_func(self._py_read_image, [filename], tf.uint8)

        image_shape = tf.shape(image_decoded)
        if self.square_pad:
            new_image_shape = tf.stack([tf.reduce_max(image_shape), tf.reduce_max(image_shape)])
            image_decoded = tf.image.resize_image_with_crop_or_pad(image_decoded, new_image_shape[0],
                                                                   new_image_shape[1])
            for x in range(len(labels)):
                if self.is_spatial_label[x]:
                    labels[x] = tf.image.resize_image_with_crop_or_pad(labels[x], new_image_shape[0],
                                                                   new_image_shape[1])
        elif self.crop_to_size_factor:
            new_image_shape = image_shape
        else:
            new_image_shape = (tf.floor_div(image_shape, self.size_factor) +
                               tf.cast(tf.greater(tf.floormod(image_shape, self.size_factor), 0),
                                       dtype=tf.int32)) * self.size_factor
            image_decoded = tf.image.resize_image_with_crop_or_pad(image_decoded, new_image_shape[0],
                                                                   new_image_shape[1])
            for x in range(len(labels)):
                if self.is_spatial_label[x]:
                    labels[x] = tf.image.resize_image_with_crop_or_pad(labels[x], new_image_shape[0],
                                                                   new_image_shape[1])
        if self.random_flip:
            rand = tf.random_uniform([], 0.0, 1.0)
            image_decoded = random_flip_left_right(image_decoded, rand)
            for x in range(len(labels)):
                if self.is_spatial_label[x]:
                    labels[x] = random_flip_left_right(labels[x], rand)


        image_decoded = (image_decoded * 2.0) - 1.0

        image_decoded.set_shape((None, None, 3))
        features = {
            "image": image_decoded
        }
        gt = {}
        for x in range(len(labels)):
            gt[x] = labels[x]
        return features, gt




