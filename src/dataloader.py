import json
import os
import cv2
import numpy as np
import torch
from src.utils import iou
from torch.utils.data import Dataset

class TrashCanDataset(Dataset):
    def __init__(self, image_root, ann_path, img_size=(256, 256)):
        self.image_root = image_root
        self.img_size = img_size

        with open(ann_path, "r") as f:
            data = json.load(f)

        self.images = {img["id"]: img for img in data["images"]}
        self.annotations = data["annotations"]
        self.categories = [cat["name"] for cat in data["categories"]]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_info = self.images[ann["image_id"]]

        img_path = os.path.join(self.image_root, img_info["file_name"])
        img = cv2.imread(img_path)
        h_orig, w_orig = img.shape[:2]

        # Rescale image
        img = cv2.resize(img, self.img_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Rescale bbox to match resized image
        bbox = np.array(ann["bbox"], dtype=np.float32)  # [x, y, w, h]
        scale_x = self.img_size[0] / w_orig
        scale_y = self.img_size[1] / h_orig
        bbox[0] *= scale_x  # x
        bbox[1] *= scale_y  # y
        bbox[2] *= scale_x  # w
        bbox[3] *= scale_y  # h

        label = ann["category_id"]

        return img, bbox, label