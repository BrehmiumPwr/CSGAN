import os
from glob import glob
from img_utils import *
from metrics import iou_soft as iou
import cv2


def examples(devkit_path):
    images = glob(os.path.join(devkit_path, "**", "*.jpg"))
    gts = [img.replace(".jpg", "_watershed_mask.png") for img in images]
    return images, gts


def read_groundtruth(filename):
    return np.array(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))

def read_image(filename):
    return np.array(cv2.imread(filename, cv2.IMREAD_COLOR))[:,:,::-1]

def test_on_mask_data(model, devkit_path):
    print(" [*] Starting Evaluation", flush=True)
    print(" [**] Reading Data", flush=True)
    image_fns, gt_fns = examples(devkit_path)
    predictions = []
    gts = []
    for x in range(len(image_fns)):
        img_fn = image_fns[x]
        gt_fn = gt_fns[x]
        image = read_image(img_fn)
        gt = read_groundtruth(gt_fn)
        gt = (gt == 26).astype(np.float32)

        gt = aspect_preserving_resize(gt, model.image_size, resize_method=cv2.INTER_NEAREST)
        image = aspect_preserving_resize(image, model.image_size, resize_method=cv2.INTER_AREA)

        prediction, _, _ = model(image, scale_image=False)
        predictions.append(prediction)
        gts.append(gt)
    return iou(predictions, gts)
