import tensorflow as tf

import glob
from os import path
from os.path import isfile
import json
import collections
import numpy as np
import codecs
#from PIL import Image
from resize_utils import *
import os
from tqdm import tqdm


class ImageDataset(object):
    def __init__(self, path, batch_size=1, randomize_size=True, random_flip=True, repeat_indefinitely=True,
                 square_pad=False, random_crop=False, random_brightness=False, random_contrast=False,
                 random_saturation=False, crop_to_size_factor=False, image_size=128, size_factor=4,
                 resize_in_advance=True, rebalance_colors=True, valid_colors=range(0, 12)):
        print(" [*] preparing data from {}".format(path))
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
        self.rebalance_colors = rebalance_colors
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_saturation = random_saturation
        self.path = path
        self.valid_colors = valid_colors
        self.files = self.getAllImagePathsFrom(path)
        self.labels_available = False
        self._set_up()
        self.filter()
        self.rebalance()

        # needs to be at the end
        self.color_count = np.unique(self.colors).shape[0]
        self.num_files = len(self.files)

    def filter(self):
        if self.labels_available:
            print(" [**] Removing invalid colors. Valid colors are: {}".format(self.valid_colors), flush=True)
            count = len(self.files)
            new_colors = [self.colors[x] for x in range(len(self.colors)) if self.colors[x] in self.valid_colors]
            new_files = [self.files[x] for x in range(len(self.colors)) if self.colors[x] in self.valid_colors]
            new_weights = [self.weights[x] for x in range(len(self.colors)) if self.colors[x] in self.valid_colors]

            self.colors = new_colors
            self.files = new_files
            self.weights = new_weights
            diff = count - len(self.files)
            print(" [**] Removed {} files".format(diff), flush=True)
        else:
            print(" [**] Cannot filter images by label if no labels available", flush=True)

    def rebalance(self):
        if self.rebalance_colors:
            colors = np.array(self.colors)
            files = np.array(self.files)

            color_list = np.unique(colors)
            files_by_color = []
            count = []
            for x in range(color_list.shape[0]):
                cur_col = color_list[x]
                files_by_color.append(files[colors == cur_col])
                count.append(files_by_color[x].shape[0])

            max_count = max(count)
            for x in range(len(files_by_color)):
                files_by_color[x] = np.concatenate(
                    [files_by_color[x] for y in range(max_count // files_by_color[x].shape[0])], axis=0)

            new_files = []
            new_colors = []
            for x in range(color_list.shape[0]):
                cur_col = color_list[x]
                for y in range(files_by_color[x].shape[0]):
                    cur_file = files_by_color[x][y]
                    new_files.append(cur_file)
                    new_colors.append(cur_col)
            self.files = new_files
            self.colors = new_colors
            self.weights = [1.0 for x in range(len(new_files))]

    def _set_up(self):

        if not (len(self.labels) == len(self.files) or len(self.labels) == 0):
            raise Exception("files missing")

        if len(self.labels) > 0:
            self.colors = [int(self.labels[x]["color"]) for x in range(len(self.labels))]
            print(" [**] found {} color labels for {} images".format(len(self.colors), len(self.files)), flush=True)
            counts = collections.Counter(self.colors)
            print(" [**] label frequency: {}".format(counts), flush=True)
            keys, vals = counts.keys(), counts.values()
            max_val, min_val = max(vals), np.min(vals)
            weights_dict = {}
            for key in keys:
                weights_dict[key] = max_val / counts[key]
            self.weights = np.array([weights_dict[col] for col in self.colors])
            mean = np.mean(list(weights_dict.values()))
            self.weights /= mean
            self.labels_available = True
        else:
            self.colors = [-1 for x in range(len(self.files))]
            self.weights = [1 for x in range(len(self.colors))]
            print(" [**] found no labels for {} images".format(len(self.files)), flush=True)

    def count(self):
        return self.num_files

    def count_colors(self):
        return self.color_count

    def isType(file, desiredType):
        filename, extension = path.splitext(file)
        return extension == desiredType

    def load_and_resize_image(self, infilename, outfilename, larger_side):
        img = Image.open(infilename)
        img = img.convert("RGB")
        img.thumbnail([larger_side, larger_side])
        img.save(outfilename, "PNG", icc_profile=None)

    def getAllImagePathsFrom(self, path):
        image_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
        imgpaths = glob.glob(path + '/**/*', recursive=True)# + glob.glob(path + '/*', recursive=True)
        imgpaths = [x for x in imgpaths if os.path.splitext(x)[-1] in image_extensions]
        #imgpaths += [x for x in glob.iglob(path + '/*.png', recursive=True)]
        self.labels = self.read_labels(imgpaths)

        new_files = []
        if self.resize_in_advance:
            resize_size = self.target_size * 2
            target_folder = path + str(resize_size)
            for x in tqdm(range(len(imgpaths)), desc="resizing images to {}".format(resize_size)):
                src_file = imgpaths[x]
                target_file = src_file.replace(path, target_folder)
                target_file = target_file.replace(os.path.splitext(target_file)[-1], ".png")
                os.makedirs(os.path.dirname(target_file), exist_ok=True)
                if not os.path.exists(target_file):
                    self.load_and_resize_image(src_file, target_file, larger_side=resize_size)
                new_files.append(target_file)
            imgpaths = new_files
        return imgpaths

    def read_labels(self, paths):
        labels = []
        for x in range(len(paths)):
            filename = paths[x] + ".json"
            if not isfile(filename):
                return []
            with open(filename) as f:
                data = json.load(f)
            labels.append(data["labels"])
        return labels

    def getImagesIterator(self):
        tf_paths = tf.constant(self.files, dtype=tf.string)
        tf_colors = tf.constant(self.colors, dtype=tf.int32)
        tf_weights = tf.constant(self.weights, dtype=tf.float32)
        dataset = tf.data.Dataset.from_tensor_slices((tf_paths, tf_colors, tf_weights))

        if not self.repeat:
            dataset = dataset.shuffle(buffer_size=len(self.files))
        else:
            dataset = dataset.shuffle(buffer_size=len(self.files))
            dataset = dataset.repeat()
            # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=len(paths)))

        #dataset = dataset.apply(
        #    tf.data.experimental.map_and_batch(lambda x, y, z: self._loadImageFunction(x, y, z),
        #                                       self.batch_size,
        #                                       num_parallel_batches=10))
        dataset = dataset.map(lambda x, y, z: self._loadImageFunction(x, y, z), num_parallel_calls=10)
        dataset = dataset.batch(batch_size=self.batch_size)
        # dataset = dataset.map(lambda x: _loadImageFunction(x, image_size=image_size), num_parallel_calls=25)
        # dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(25)

        iterator = dataset.make_one_shot_iterator()

        return iterator

    def _py_read_image(self, filename):
        filename = codecs.decode(filename, 'utf-8')
        img = Image.open(filename)
        img = img.convert('RGB')
        img = np.array(img)
        return img

    def _loadImageFunction(self, filename, color, weight):
        if self.randomize_size:
            max_upper_dev = int((self.target_size * 0.3) // self.size_factor)
            max_lower_dev = -int((self.target_size * 0.3) // self.size_factor)
            delta_size = tf.random_uniform(shape=(), minval=max_lower_dev, maxval=max_upper_dev,
                                           dtype=tf.int32) * self.size_factor
        else:
            delta_size = 0
        image_size = self.target_size + delta_size

        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, 3)
        image_decoded.set_shape([None, None, 3])
        image_decoded = tf.cast(image_decoded, tf.float32)
        image_decoded = image_decoded / 255.0

        if self.random_crop:
            target_size = tf.cast(tf.multiply(tf.cast(image_size, dtype=tf.float32), 1.3), dtype=tf.int32)
            image_decoded = aspect_preserving_resize(image_decoded, target_size=target_size, resize_mode="crop")
            image_decoded = tf.random_crop(image_decoded, [image_size, image_size, 3])
        else:
            image_decoded = aspect_preserving_resize(image_decoded, target_size=image_size, resize_mode="pad")

        if self.crop_to_size_factor:
            resized_size = tf.shape(image_decoded)[:2]
            target_crop_size = (resized_size // self.size_factor) * self.size_factor
            image_decoded = image_decoded[:target_crop_size[0], :target_crop_size[1], :]



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
        elif self.crop_to_size_factor:
            new_image_shape = image_shape
        else:
            new_image_shape = (tf.floor_div(image_shape, self.size_factor) +
                               tf.cast(tf.greater(tf.floormod(image_shape, self.size_factor), 0),
                                       dtype=tf.int32)) * self.size_factor
            image_decoded = tf.image.resize_image_with_crop_or_pad(image_decoded, new_image_shape[0],
                                                                   new_image_shape[1])
        offset_y = (new_image_shape[0] - image_shape[0]) // 2
        offset_x = (new_image_shape[1] - image_shape[1]) // 2

        if self.random_flip:
            image_decoded = tf.image.random_flip_left_right(image_decoded)

        image_decoded = (image_decoded * 2.0) - 1.0

        return image_decoded, color, weight, filename, [offset_y, offset_x], image_shape
