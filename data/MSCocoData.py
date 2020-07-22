import os
import numpy as np

from img_utils import aspect_preserving_resize
from data.GenericDataset import GenericDataset
from PIL import Image
from metrics import iou
import cv2
from pycocotools.coco import COCO
from tqdm import tqdm


class MSCOCOData(GenericDataset):
    def __init__(self, path, batch_size=1, randomize_size=True, random_flip=True, repeat_indefinitely=True,
                 square_pad=False, random_crop=False, random_brightness=False, random_contrast=False,
                 random_saturation=False, crop_to_size_factor=False, image_size=128, size_factor=4,
                 resize_in_advance=True, image_set="train2017"):
        self.image_set = image_set
        self.annotations = os.path.join(path, "annotations/instances_{}.json".format(self.image_set))
        self.image_path = os.path.join(path, "images", self.image_set)
        self.mscocodata = self.prepare()
        super().__init__(path=path, batch_size=batch_size, randomize_size=randomize_size, random_flip=random_flip,
                         repeat_indefinitely=repeat_indefinitely,square_pad=square_pad,random_crop=random_crop,
                         random_brightness=random_brightness, random_contrast=random_contrast,
                         random_saturation=random_saturation, crop_to_size_factor=crop_to_size_factor, image_size=image_size,
                         size_factor=size_factor, resize_in_advance=resize_in_advance)

        print("[**] Initialized MSCOCO Dataset {}".format(image_set), flush=True)

    def get_data(self):
        return self.mscocodata

    def prepare(self):
        target_dir = "datasets/coco_cropped_{}".format(self.image_set)
        os.makedirs(target_dir, exist_ok=True)
        coco = COCO(self.annotations)
        catIds = coco.getCatIds(catNms=['car'])
        imgIds = coco.getImgIds(catIds=catIds)
        annIds = coco.getAnnIds(imgIds=imgIds, catIds=catIds, iscrowd=False)
        anns = coco.loadAnns(annIds)
        #imgs = coco.loadImgs(imgIds)

        data = {}
        for ann in tqdm(anns):
            image_id = ann["image_id"]
            img = coco.loadImgs([image_id])[0]
            img_filename = os.path.join(self.image_path, img["file_name"])
            target_filename = os.path.join(target_dir, str(ann["id"])+".png")

            x1, y1, width, height = ann["bbox"]
            x2, y2 = x1 + width, y1 + height
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)



            if ann["area"] < 3000:
                continue
            aspect_ratio = width / height
            if aspect_ratio > 4 or aspect_ratio < 0.4:
                continue

            mask = coco.annToMask(ann)

            # expand bbox
            expansion_x = int(width*0.2)
            expansion_y = int(height*0.2)

            x1 = max(0, x1 - expansion_x)
            x2 = min(mask.shape[1], x2 + expansion_x)
            y1 = max(0, y1 - expansion_y)
            y2 = min(mask.shape[0], y2 + expansion_y)

            mask = mask[y1:y2, x1:x2]

            if not os.path.isfile(target_filename):
                img_new = Image.open(img_filename)
                img_new = img_new.crop((x1,y1,x2,y2))
                img_new.save(target_filename, "PNG", icc_profile=None)

            data[target_filename] = [mask]
        return data

    def test_on_data(self, model, threshold=0.5):
        print(" [*] Starting Evaluation", flush=True)
        print(" [**] Reading Data", flush=True)
        data_dict = self.get_data()
        predictions = []
        gts = []
        for x in data_dict.keys():
            img_fn = x
            image = self.read_image(img_fn)
            gt = data_dict[x][0]

            gt = aspect_preserving_resize(gt, model.image_size, resize_method=cv2.INTER_NEAREST)
            image = aspect_preserving_resize(image, model.image_size, resize_method=cv2.INTER_AREA)

            color_model_pred, voc_model_pred, _ = model(image, scale_image=False)
            predictions.append(np.int32(color_model_pred > threshold))
            gts.append(gt)
            # im = np.expand_dims(np.concatenate([gt, prediction], axis=1), axis=-1)
            # im = np.tile(im, [1,1,3])
            # im = np.concatenate([image, im], axis=1).astype(np.uint8)
            # Image.fromarray(im).show()
        return iou(predictions, gts)
