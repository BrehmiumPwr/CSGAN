import os
import numpy as np

from img_utils import aspect_preserving_resize
from data.GenericDataset import GenericDataset
from metrics import iou
import cv2


class VOC2012Data(GenericDataset):
    def __init__(self, path, batch_size=1, randomize_size=True, random_flip=True, repeat_indefinitely=True,
                 square_pad=False, random_crop=False, random_brightness=False, random_contrast=False,
                 random_saturation=False, crop_to_size_factor=False, image_size=128, size_factor=4,
                 resize_in_advance=True, relevant_classes=[7], image_set="train"):
        self.imagesets_path = os.path.join(path, "ImageSets", "Segmentation")
        self.image_path = os.path.join(path, "JPEGImages")
        self.segmentation_path = os.path.join(path, "SegmentationClass")
        self.image_set = image_set
        self.relevant_classes = relevant_classes
        self.vocdata = self.prepare()
        self.num_classes = len(self.relevant_classes)
        super().__init__(path=path, batch_size=batch_size, randomize_size=randomize_size, random_flip=random_flip,
                         repeat_indefinitely=repeat_indefinitely,square_pad=square_pad,random_crop=random_crop,
                         random_brightness=random_brightness, random_contrast=random_contrast,
                         random_saturation=random_saturation, crop_to_size_factor=crop_to_size_factor, image_size=image_size,
                         size_factor=size_factor, resize_in_advance=resize_in_advance)

        print("[**] Initialized VOC Dataset {}".format(image_set), flush=True)

    def get_data(self):
        return self.vocdata

    def prepare(self):
        class_dict = self.examples_by_class()
        data = []
        if self.relevant_classes == None:
            self.relevant_classes = [x for x in class_dict.keys()][1:]

        for x in self.relevant_classes:
            data += class_dict[x]

        self.relevant_classes = [0] + self.relevant_classes
        #create image, label dict and implicitly remove duplicates
        for x in range(len(data)):
            filename, segmentation = data[x]
            segmentation = segmentation[0]
            # reorder classes

            for y in range(len(self.relevant_classes)):
                segmentation[segmentation == self.relevant_classes[y]] = y
            data[x][1][0] = segmentation

        return dict(data)

    def examples_by_class(self):
        filelist = os.path.join(self.imagesets_path, self.image_set + ".txt")
        with open(filelist) as f:
            files = f.readlines()
            files = [file.strip("\n") for file in files]

        class_dict = {}
        for file in files:
            segmentation_image = self.read_groundtruth(file)
            classes = np.unique(segmentation_image)
            classes = classes[np.where(classes != 255)]
            filename = os.path.join(self.image_path, file+".jpg")
            for cls in classes:
                if cls in class_dict.keys():
                    class_dict[cls].append((filename, [np.expand_dims(segmentation_image, axis=-1)]))
                else:
                    class_dict[cls] = [(filename, [np.expand_dims(segmentation_image, axis=-1)])]
        return class_dict

    def read_groundtruth(self, id):
        segmentation_image = os.path.join(self.segmentation_path, id +".png")
        segmentation_image = self.read_spatial_gt(segmentation_image)
        if self.relevant_classes is None:
            segmentation_image[segmentation_image == 255] = 0
            return segmentation_image
        else:
            classes = np.unique(segmentation_image)
            relevant_classes = np.array(self.relevant_classes)
            things_to_consider = np.intersect1d(classes, relevant_classes)
            #all_things = np.union1d(classes, relevant_classes)
            irrelevant_things = classes[np.invert(np.isin(classes, things_to_consider))]
            segmentation_image[np.isin(segmentation_image, irrelevant_things)] = 0
            return segmentation_image

    def test_on_data(self, model, thresh=None):
        print(" [*] Starting Evaluation", flush=True)
        print(" [**] Reading Data", flush=True)
        data_dict = self.get_data()
        predictions = []
        gts = []
        for x in data_dict.keys():
            img_fn = x
            image = self.read_image(img_fn)
            gt = data_dict[x][0][:, :, 0]

            gt = aspect_preserving_resize(gt, model.image_size, resize_method=cv2.INTER_NEAREST)
            image = aspect_preserving_resize(image, model.image_size, resize_method=cv2.INTER_AREA)

            color_model_pred, voc_model_pred, _ = model(image, scale_image=False)
            predictions.append(voc_model_pred)
            gts.append(gt)
            # im = np.expand_dims(np.concatenate([gt, prediction], axis=1), axis=-1)
            # im = np.tile(im, [1,1,3])
            # im = np.concatenate([image, im], axis=1).astype(np.uint8)
            # Image.fromarray(im).show()
        return iou(predictions, gts)
