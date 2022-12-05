import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO

class COCOMaksProvider(COCO):
    def __init__(self, ann_file, imgs_dir):
        super().__init__(ann_file)
        self.imgs_dir = imgs_dir
        

    def get_images_and_masks(self, image_id):
        img = self.imgs[image_id]
        image = np.array(Image.open(os.path.join(self.imgs_dir, img['file_name'])))
        cat_ids = self.getCatIds()
        anns_ids = self.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = self.loadAnns(anns_ids)
        masks = []
        for ann in anns:
            masks.append(self.annToMask(ann))
        masks = np.stack(masks).astype(bool)
        labels = np.array([x["category_id"] for x in anns])
        return image, masks, labels

