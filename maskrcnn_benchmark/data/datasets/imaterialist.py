# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
# import torchvision
import numpy as np
import cv2
import pycocotools.mask as mask_utils

from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask
from maskrcnn_benchmark.structures.keypoint import PersonKeypoints
from maskrcnn_benchmark.data.transforms import transforms as T

class iMaterialistDataset(COCODataset):
    def __init__(
        self, ann_file, root, remove_images_without_annotations,
        transforms=None, min_size=800, max_size=1333
    ):
        super(iMaterialistDataset, self).__init__(ann_file, root, 
                            remove_images_without_annotations, transforms)
        self.resize = T.Resize(min_size, max_size)

    def __getitem__(self, idx):
        img, anno = super(COCODataset, self).__getitem__(idx)

        # filter crowd annotations
        # TODO might be better to add an extra field
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes).reshape(-1, 4)  # guard against no boxes
        target = BoxList(boxes, img.size, mode="xywh").convert("xyxy")

        classes = [obj["category_id"] for obj in anno]
        classes = [self.json_category_id_to_contiguous_id[c] for c in classes]
        classes = torch.tensor(classes)
        target.add_field("labels", classes)

        if anno and "segmentation" in anno[0]:
            masks = [obj["segmentation"] for obj in anno]
            # decode RLEs to masks
            masks = mask_utils.decode(masks) # hxwxn
            # the training size for mask
            mask_size = self.resize.get_size(img.size) # (h, w)
            mask_size = (mask_size[1],mask_size[0]) # (w, h)
            # resize mask for saving memory
            mask_list = []
            for i in range(masks.shape[-1]):
                mask = cv2.resize(masks[:,:,i], mask_size,
                    interpolation=cv2.INTER_NEAREST)
                mask_list.append(torch.from_numpy(mask))
            masks = torch.stack(mask_list, dim=0).clone() # nxhxw

            masks = SegmentationMask(masks, mask_size, mode='mask')
            target.add_field("masks", masks)

        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = PersonKeypoints(keypoints, img.size)
            target.add_field("keypoints", keypoints)

        target = target.clip_to_image(remove_empty=True)

        if self._transforms is not None:
            img, target = self._transforms(img, target)

        return img, target, idx