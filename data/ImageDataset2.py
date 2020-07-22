import glob
from resize_utils import *
import os
from data.GenericDataset import GenericDataset


class ImageDataset(GenericDataset):
    def __init__(self, path, batch_size=1, randomize_size=True, random_flip=True, repeat_indefinitely=True,
                 square_pad=False, random_crop=False, random_brightness=False, random_contrast=False,
                 random_saturation=False, crop_to_size_factor=False, image_size=128, size_factor=4,
                 resize_in_advance=True):
        print(" [*] preparing data from {}".format(path))
        self.path = path
        self.data = self.prepare()
        super().__init__(path=path, batch_size=batch_size, randomize_size=randomize_size, random_flip=random_flip,
                         repeat_indefinitely=repeat_indefinitely, square_pad=square_pad, random_crop=random_crop,
                         random_brightness=random_brightness, random_contrast=random_contrast,
                         random_saturation=random_saturation, crop_to_size_factor=crop_to_size_factor,
                         image_size=image_size, size_factor=size_factor, resize_in_advance=resize_in_advance)

        # needs to be at the end
        self.num_files = len(self.files)
        print("[**] Initialized image dataset Dataset {}".format(path), flush=True)

    def get_data(self):
        return self.data

    def prepare(self):
        self.files = self.getAllImagePathsFrom(self.path)
        self.labels = [[np.int64(0)] for x in range(len(self.files))]
        data = list(zip(self.files, self.labels))

        return dict(data)

    def count(self):
        return self.num_files

    def getAllImagePathsFrom(self, path):
        image_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"]
        imgpaths = glob.glob(path + '/**/*', recursive=True)  # + glob.glob(path + '/*', recursive=True)
        imgpaths = [x for x in imgpaths if os.path.splitext(x)[-1] in image_extensions]
        return imgpaths

