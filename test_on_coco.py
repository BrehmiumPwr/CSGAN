from CSGAN import CSGAN
import tensorflow as tf
import json
import os
from data import MSCOCOData
from data import CarsWithMasksData
from img_utils import aspect_preserving_resize
import cv2
from tqdm import tqdm
import numpy as np
from metrics import iou
from PIL import Image
from pycocotools.coco import COCO


model_dir = "models"
model_name = "sgan_128_weak_sn_concat_resCDN_newsampling_separable_newreg_gamma10_fromrgbnew_pixelnorm_downsidecorrect_mobilenet"

model_base_dir = os.path.join(model_dir, model_name)
with open(os.path.join(model_base_dir, "options.json")) as f:
    options = json.load(f)

options["phase"] = "test"
options["valid_colors"] = range(0, 11)
#options, test_fn = dataset_definitions.cars(options)
#coco = MSCOCOData(path="datasets/coco", image_set="train2017")

#data = coco.get_data()
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
session = tf.Session()

with session.as_default():
    model = CSGAN([], session=session, options=options)
    model.load(subpath="best")
    res = CarsWithMasksData.test_on_mask_data(model, "datasets/cars_masks")
    _, thresh = res["IoU"], res["threshold"]
    #print(thresh)
    os.makedirs("coco_results", exist_ok=True)
    counter = 0
    predictions = []
    gts = []
    image_set = "train2017"
    path = "datasets/coco"
    annotations = os.path.join(path, "annotations/instances_{}.json".format(image_set))
    image_path = os.path.join(path, "images", image_set)
    coco = COCO(annotations)
    catIds = coco.getCatIds(catNms=['car'])
    imgIds = coco.getImgIds(catIds=catIds)

    for key in tqdm(imgIds):
        annIds = coco.getAnnIds(imgIds=[key], catIds=catIds, iscrowd=False)
        anns = coco.loadAnns(annIds)
        imgs = coco.loadImgs(key)
        image = np.array(cv2.imread(os.path.join(image_path, imgs[0]["file_name"]), cv2.IMREAD_COLOR))[:, :, ::-1]
        results = np.zeros(image.shape[:2], dtype=np.bool)
        gt_masks = np.zeros_like(results)
        for ann in anns:
            x1, y1, width, height = ann["bbox"]
            x2, y2 = x1 + width, y1 + height
            x1,y1,x2,y2 = int(x1), int(y1), int(x2), int(y2)
            aspect_ratio = width / height
            if aspect_ratio > 6 or aspect_ratio < 0.3:
                continue
            gt = coco.annToMask(ann)
            # expand bbox
            expansion_x = int(width*0.2)
            expansion_y = int(height*0.2)

            x1 = max(0, x1 - expansion_x)
            x2 = min(gt.shape[1], x2 + expansion_x)
            y1 = max(0, y1 - expansion_y)
            y2 = min(gt.shape[0], y2 + expansion_y)

            gt = gt[y1:y2, x1:x2]
            crop = image[y1:y2, x1:x2, :]

            assert gt.shape == crop.shape[:2]
            #gt = aspect_preserving_resize(gt, model.image_size, resize_method=cv2.INTER_NEAREST)

            #crop = aspect_preserving_resize(crop, model.image_size, resize_method=cv2.INTER_AREA)

            color_model_pred, _, _ = model(crop, scale_image=True)

            predicted_mask = (color_model_pred > thresh).astype(np.uint8)
            contours, hierarchy = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                vertices = cv2.approxPolyDP(contour, epsilon, True)
                vertices = cv2.convexHull(vertices, clockwise=True)
                cv2.fillPoly(predicted_mask, [vertices], 1)

            predictions.append(predicted_mask)
            gts.append(gt)
            results[y1:y2, x1:x2] = np.logical_or(results[y1:y2, x1:x2], predicted_mask.astype(np.bool))
            gt_masks[y1:y2, x1:x2] = np.logical_or(gt_masks[y1:y2, x1:x2], gt.astype(np.bool))

        c = np.array([220.0, 50.0, 50.0])
        results = results.astype(np.float)
        colored_mask = (np.expand_dims(results, axis=-1) * c).astype(np.uint8)
        masked_image = cv2.addWeighted(image, 1.0, colored_mask, .8, gamma=0.0)
        masked_image = Image.fromarray(masked_image)
        masked_image.save(os.path.join("coco_results", str(counter) + "mask.png"))

        gt_masks = gt_masks.astype(np.float)
        colored_mask = (np.expand_dims(gt_masks, axis=-1) * c).astype(np.uint8)
        masked_image = cv2.addWeighted(image, 1.0, colored_mask, .8, gamma=0.0)
        masked_image = Image.fromarray(masked_image)
        masked_image.save(os.path.join("coco_results", str(counter) + "gt.png"))
        counter += 1

    print(iou(predictions, gts))


