import glob
from os.path import isfile
import json
import collections
from resize_utils import *
import os
from data.GenericDataset import GenericDataset


class ColorDataset(GenericDataset):
    def __init__(self, path, batch_size=1, randomize_size=True, random_flip=True, repeat_indefinitely=True,
                 square_pad=False, random_crop=False, random_brightness=False, random_contrast=False,
                 random_saturation=False, crop_to_size_factor=False, image_size=128, size_factor=4,
                 resize_in_advance=True, rebalance_colors=True, valid_colors=range(0, 12)):
        print(" [*] preparing data from {}".format(path))
        self.valid_colors = valid_colors
        self.rebalance_colors = rebalance_colors
        self.path = path
        self.colordata = self.prepare()
        self.num_classes = len(self.valid_colors)
        super().__init__(path=path, batch_size=batch_size, randomize_size=randomize_size, random_flip=random_flip,
                         repeat_indefinitely=repeat_indefinitely, square_pad=square_pad, random_crop=random_crop,
                         random_brightness=random_brightness, random_contrast=random_contrast,
                         random_saturation=random_saturation, crop_to_size_factor=crop_to_size_factor,
                         image_size=image_size, size_factor=size_factor, resize_in_advance=resize_in_advance)

        # needs to be at the end
        self.color_count = np.unique(self.colors).shape[0]
        self.num_files = len(self.files)
        print("[**] Initialized color labelled Dataset {}".format(path), flush=True)

    def get_data(self):
        return self.colordata

    def filter(self):
        if self.labels_available:
            print(" [**] Removing invalid colors. Valid colors are: {}".format(self.valid_colors), flush=True)
            count = len(self.files)
            new_colors = [self.colors[x] for x in range(len(self.colors)) if self.colors[x] in self.valid_colors]
            new_files = [self.files[x] for x in range(len(self.colors)) if self.colors[x] in self.valid_colors]

            self.colors = new_colors
            self.files = new_files
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
            #self.weights = [1.0 for x in range(len(new_files))]

    def prepare(self):
        self.files = self.getAllImagePathsFrom(self.path)
        self.labels_available = False
        self._set_up()
        self.filter()
        self.rebalance()
        self.colors = [[x] for x in self.colors]
        data = list(zip(self.files, self.colors))

        return dict(data)

    def _set_up(self):

        if not (len(self.labels) == len(self.files) or len(self.labels) == 0):
            raise Exception("files missing")

        if len(self.labels) > 0:
            self.colors = [int(self.labels[x]["color"]) for x in range(len(self.labels))]
            print(" [**] found {} color labels for {} images".format(len(self.colors), len(self.files)), flush=True)
            counts = collections.Counter(self.colors)
            print(" [**] label frequency: {}".format(counts), flush=True)
            # keys, vals = counts.keys(), counts.values()
            # max_val, min_val = max(vals), np.min(vals)
            # weights_dict = {}
            # for key in keys:
            #     weights_dict[key] = max_val / counts[key]
            # self.weights = np.array([weights_dict[col] for col in self.colors])
            # mean = np.mean(list(weights_dict.values()))
            # self.weights /= mean
            self.labels_available = True
        else:
            self.colors = [-1 for x in range(len(self.files))]
            #self.weights = [1 for x in range(len(self.colors))]
            print(" [**] found no labels for {} images".format(len(self.files)), flush=True)

    def count(self):
        return self.num_files

    def count_colors(self):
        return self.color_count

    def getAllImagePathsFrom(self, path):
        image_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
        imgpaths = glob.glob(path + '/**/*', recursive=True)  # + glob.glob(path + '/*', recursive=True)
        imgpaths = [x for x in imgpaths if os.path.splitext(x)[-1] in image_extensions]
        # imgpaths += [x for x in glob.iglob(path + '/*.png', recursive=True)]
        self.labels = self.read_labels(imgpaths)
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
