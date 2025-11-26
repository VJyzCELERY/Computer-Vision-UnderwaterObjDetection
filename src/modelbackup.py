import time
import os
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch.nn as nn
import torchvision
from skimage.feature import hog,local_binary_pattern
import torch
from sklearn.cluster import DBSCAN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import random
from src.utils import nms,iou,format_time,CHWtoHWC,pyramid_sliding_window,image_pyramid,sliding_window



class SIFTBoVWExtractor:
    def __init__(self, n_clusters=128):
        self.sift = cv2.SIFT_create()
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.is_fitted = False
        self.n_clusters = n_clusters

    def build_vocab(self, images):
        descriptors_list = []

        for img in images:
            kp, des = self.sift.detectAndCompute(img, None)
            if des is not None:
                descriptors_list.append(des)

        descriptors = np.vstack(descriptors_list)
        self.kmeans.fit(descriptors)
        self.is_fitted = True

    def extract(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        kp, des = self.sift.detectAndCompute(img, None)

        if des is None:
            return np.zeros(self.n_clusters, dtype=np.float32)

        words = self.kmeans.predict(des)
        hist, _ = np.histogram(words, bins=self.n_clusters, range=(0, self.n_clusters))
        hist = hist.astype(np.float32)

        return hist / (np.linalg.norm(hist) + 1e-7)

class AKAZEBoVWExtractor:
    def __init__(self, n_clusters=128):
        # Using AKAZE instead of SIFT
        self.akaze = cv2.AKAZE_create()  # You can customize parameters
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.is_fitted = False
        self.n_clusters = n_clusters
        
    def detectAndCompute(self, img):
        """AKAZE returns descriptors in a different format"""
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return self.akaze.detectAndCompute(img, None)
    
    def build_vocab(self, images):
        descriptors_list = []
        for img in images:
            # AKAZE detects keypoints and computes descriptors
            kp, des = self.detectAndCompute(img)
            if des is not None:
                descriptors_list.append(des)
        descriptors = np.vstack(descriptors_list)
        self.kmeans.fit(descriptors)
        self.is_fitted = True
    
    def extract(self, img):
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Using AKAZE's detectAndCompute
        kp, des = self.detectAndCompute(img)
        if des is None:
            return np.zeros(self.n_clusters, dtype=np.float32)
        
        words = self.kmeans.predict(des)
        hist, _ = np.histogram(words, bins=self.n_clusters, range=(0, self.n_clusters))
        hist = hist.astype(np.float32)
        return hist / (np.linalg.norm(hist) + 1e-7)

class SlidingWindowClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

class SlidingWindowRegressor(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(input_dim, 256)
        self.cls_head = nn.Linear(256, num_classes) 
        self.reg_head = nn.Linear(256, 4)           

    def forward(self, x):
        x = F.relu(self.fc(x))
        cls_logits = self.cls_head(x)
        bbox_deltas = self.reg_head(x)
        return cls_logits, bbox_deltas

class ORBBoVWExtractor:
    def __init__(self, n_clusters=128):
        self.n_clusters = n_clusters
        self.orb = cv2.ORB_create()
        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.is_fitted = False

    def build_vocab(self, patches):
        descriptors_list = []
        for patch in patches:
            kp, des = self.orb.detectAndCompute(patch, None)
            if des is not None:
                descriptors_list.append(des.astype(np.float32)) 
        if len(descriptors_list) == 0:
            raise RuntimeError("No descriptors found for ORB vocabulary.")
        descriptors_all = np.vstack(descriptors_list)
        self.kmeans.fit(descriptors_all)
        self.is_fitted = True

    def extract(self, patch):
        kp, des = self.orb.detectAndCompute(patch, None)
        if des is None:
            return np.zeros(self.n_clusters, dtype=np.float32)
        des = des.astype(np.float32)
        labels = self.kmeans.predict(des)
        hist = np.bincount(labels, minlength=self.n_clusters)
        hist = hist / (hist.sum() + 1e-6)  # normalize
        return hist

class HOGBasedExtractor:
    def __init__(self,orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',channel_axis=-1):
        self.orientations=orientations
        self.pixels_per_cell=pixels_per_cell
        self.cells_per_block=cells_per_block
        self.block_norm=block_norm
        self.channel_axis=channel_axis
        self.channels = 3 if channel_axis == -1 else 1
    def _ensure_format(self, patch):
        """Ensure patch is in the correct format for HOG"""
        if self.channels == 3:
            if patch.ndim == 2:
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2RGB)
            elif patch.shape[2] == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        else:
            if patch.ndim == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        return patch
    def extract(self,patch,**kwargs):
        patch = self._ensure_format(patch)
        return hog(
            patch,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            channel_axis=self.channel_axis,
            **kwargs
        )
    
    def get_dim(self,window_size=(64,64)):
        if self.channels == 3:
            dummy = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        else:
            dummy = np.zeros((window_size[1], window_size[0]), dtype=np.uint8)
        feat = self.extract(dummy)
        return feat.shape[0]

class FastExtractor:
    def __init__(self,**kwargs):
        self.hog = HOGBasedExtractor(**kwargs)
        
    def extract(self, patch):
        feat = self.hog.extract(patch)
        feat = feat.astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-6)
    
    def get_dim(self,window_size=(64,64)):
        return self.hog.get_dim(window_size)

class ObjDetector:
    def __init__(self, 
                classes,image_size=(256,256), iou_thresh=0.5,
                window_step_size=16,window_size=(64,64),min_size=(64,64),
                orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),block_norm='L2-Hys',channel_axis=-1,
                default_conf_thresh=0.7, device="cuda",lr=1e-3):
        self.device = device
        self.classes = classes
        self.num_classes = len(classes)
        self.iou_thresh = iou_thresh
        self.window_step_size=window_step_size
        self.window_size=window_size
        self.hog_cache = {}
        self.min_size=min_size
        self.image_size = image_size
        self.default_conf_thresh = default_conf_thresh
        self.extractor = FastExtractor(orientations=orientations,pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,block_norm=block_norm,channel_axis=channel_axis)
        self.model = SlidingWindowRegressor(input_dim=self.extractor.get_dim((window_size[1],window_size[0])),num_classes=len(classes)).to(device)
        self.optimizer = Adam(
            list(self.model.parameters()),
            lr=lr
        )
    def generate_proposals(self, image):
        win_w, win_h = self.window_size

        for i, scaled in enumerate(image_pyramid(image)):
            if i > 3:   # hard cap pyramid depth
                break

            scale_x = image.shape[1] / scaled.shape[1]
            scale_y = image.shape[0] / scaled.shape[0]

            for (x, y, patch) in sliding_window(
                scaled,
                self.window_step_size,   # e.g. 32
                self.window_size
            ):
                feat = self.extractor.extract(patch)

                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)
                orig_w = int(win_w * scale_x)
                orig_h = int(win_h * scale_y)

                yield orig_x, orig_y, orig_w, orig_h, feat

    def build_vocabulary_from_dataset(self, dataloader, max_patches=5000):
        patches = []
        collected = 0

        # Outer tqdm for dataset
        for imgs, bboxes, labels in tqdm(dataloader, desc="Building vocabulary (images)"):
            for i in range(len(imgs)):
                img_rgb = CHWtoHWC(imgs[i])
                if isinstance(img_rgb, torch.Tensor):
                    img_rgb = img_rgb.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                else:
                    img_rgb = img_rgb.astype(np.uint8)

                gt_box = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]

                # Inner tqdm for proposals in this image
                for x, y, w, h, patch in tqdm(self.generate_keypoint_proposals(img_rgb), 
                                            desc=f"Image {i} proposals", leave=False):
                    if collected >= max_patches:
                        break

                    if iou((x, y, w, h), tuple(gt_box)) >= self.iou_thresh:
                        # Convert patch to gray
                        if patch.ndim == 3 and patch.shape[2] == 3:
                            patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                        else:
                            patch_gray = patch
                        patches.append(patch_gray)
                        collected += 1

                if collected >= max_patches:
                    break

            if collected >= max_patches:
                break

        if len(patches) == 0:
            raise RuntimeError("No positive patches found for vocabulary building.")

        self.extractor.build_vocab(patches)
        print(f"Collected {len(patches)} positive patches for BoVW vocabulary.")
    
    def train_one_epoch(self, dataloader, curr_epoch=None, total_epoch=None, val_loader=None):
        self.model.train()
        total_loss = 0
        total_batches = 0
        val_loss = None

        pbar = tqdm(
            total=len(dataloader) + len(val_loader) if val_loader is not None else len(dataloader),
            desc=f"Epoch [{curr_epoch}/{total_epoch}]" if curr_epoch is not None and total_epoch is not None else "Training",
            unit="batch",
            leave=True
        )

        for imgs, bboxes, labels in dataloader:
            feats = []
            targets_label = []
            targets_bbox = []

            for i in range(imgs.shape[0]):
                img_np = CHWtoHWC(imgs[i])
                gt_boxes = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
                gt_label = int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])

                if gt_boxes.ndim == 1:
                    gt_boxes = np.expand_dims(gt_boxes, axis=0)

                for (x, y, w, h, features) in self.generate_proposals(img_np):
                    for gt_box in gt_boxes:
                        if iou((x, y, w, h), tuple(gt_box)) >= self.iou_thresh:
                            feat = features
                            feats.append(feat)
                            targets_label.append(gt_label)

                            # compute bbox regression targets
                            x_a, y_a, w_a, h_a = x, y, w, h
                            x_gt, y_gt, w_gt, h_gt = gt_box
                            tx = (x_gt - x_a) / w_a
                            ty = (y_gt - y_a) / h_a
                            tw = np.log(w_gt / w_a)
                            th = np.log(h_gt / h_a)
                            targets_bbox.append([tx, ty, tw, th])
                            break

            pbar.update(1)
            if len(feats) == 0:
                continue

            feats_t = torch.tensor(np.array(feats), dtype=torch.float32).to(self.device)
            targets_label_t = torch.tensor(np.array(targets_label), dtype=torch.long).to(self.device)
            targets_bbox_t = torch.tensor(np.array(targets_bbox), dtype=torch.float32).to(self.device)

            cls_logits, bbox_preds = self.model(feats_t)
            cls_loss = F.cross_entropy(cls_logits, targets_label_t)
            reg_loss = F.smooth_l1_loss(bbox_preds, targets_bbox_t)
            loss = cls_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            pbar.set_postfix({"Training loss": total_loss / total_batches})

        if val_loader is not None:
            val_loss = self.validate(val_loader, pbar)
            pbar.set_postfix({"Training loss": total_loss / total_batches, "Validation Loss": val_loss})
            pbar.refresh()

        return 0.0 if total_batches == 0 else total_loss / total_batches, val_loss if val_loss is not None else 0.0
    
    @torch.no_grad()
    def validate(self, dataloader, pbar: tqdm = None):
        self.model.eval()
        total_loss = 0
        total_batches = 0

        for imgs, bboxes, labels in dataloader:
            feats = []
            targets_label = []
            targets_bbox = []

            for i in range(imgs.shape[0]):
                img_np = CHWtoHWC(imgs[i])
                gt_boxes = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
                gt_label = int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])

                if gt_boxes.ndim == 1:
                    gt_boxes = np.expand_dims(gt_boxes, axis=0)

                for (x, y, w, h, features) in self.generate_proposals(img_np):
                    for gt_box in gt_boxes:
                        if iou((x, y, w, h), tuple(gt_box)) >= self.iou_thresh:
                            feat = features
                            feats.append(feat)
                            targets_label.append(gt_label)

                            # compute bbox regression targets
                            x_a, y_a, w_a, h_a = x, y, w, h
                            x_gt, y_gt, w_gt, h_gt = gt_box
                            tx = (x_gt - x_a) / w_a
                            ty = (y_gt - y_a) / h_a
                            tw = np.log(w_gt / w_a)
                            th = np.log(h_gt / h_a)
                            targets_bbox.append([tx, ty, tw, th])
                            break

            if len(feats) == 0:
                if pbar is not None:
                    pbar.update(1)
                continue

            feats_t = torch.tensor(np.array(feats), dtype=torch.float32).to(self.device)
            targets_label_t = torch.tensor(np.array(targets_label), dtype=torch.long).to(self.device)
            targets_bbox_t = torch.tensor(np.array(targets_bbox), dtype=torch.float32).to(self.device)

            cls_logits, bbox_preds = self.model(feats_t)
            cls_loss = F.cross_entropy(cls_logits, targets_label_t)
            reg_loss = F.smooth_l1_loss(bbox_preds, targets_bbox_t)
            loss = cls_loss + reg_loss

            total_loss += loss.item()
            total_batches += 1

            if pbar is not None:
                pbar.update(1)

        return 0.0 if total_batches == 0 else total_loss / total_batches

    def train(self, train_loader, val_loader, val_dataset, epochs=10, vis_every=2,checkpoint_dir="checkpoints", checkpoint_every=20):
        os.makedirs(checkpoint_dir, exist_ok=True)
        total_loss = []
        total_val_loss = []
        t0 = time.time()
        for epoch in range(epochs):
            train_loss,val_loss = self.train_one_epoch(train_loader,curr_epoch=epoch+1,total_epoch=epochs,val_loader=val_loader)

            total_loss.append(train_loss)
            total_val_loss.append(val_loss)
            # Show validation predictions every few epochs
            if (epoch + 1) % vis_every == 0:
                print("Showing validation sample predictions...")
                self.visualize_random_val_samples(val_dataset, num_samples=5)
            if (epoch + 1) % checkpoint_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_checkpoint.pth")
                save_detector(self, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        elapsed = (time.time() - t0)
        print(f'Total Training Time : {format_time(elapsed)}')
        return total_loss,total_val_loss

    def visualize_detections(self, image_rgb, detections, class_names=None):
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        ax = plt.gca()

        for cls_id, score, (x, y, w, h) in detections:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            label = class_names[cls_id] if class_names is not None else self.classes[cls_id]
            ax.text(x, y - 5, f"{label}:{score:.2f}", color="red", fontsize=10, backgroundcolor='white')

        plt.axis("off")
        plt.show()


    @torch.no_grad()
    def predict(self, image_rgb, conf_thresh=None):
        self.model.eval()
        detections = []
        if conf_thresh == None:
            conf_thresh=self.default_conf_thresh
        # if not self.extractor.is_fitted:
        #     raise RuntimeError("BoVW vocabulary not built. Call build_vocabulary_from_dataset() first.")

        proposals = self.generate_proposals(image_rgb)

        for (x, y, w, h, features) in proposals:

            feat = features
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

            cls_logits, bbox_pred = self.model(feat_t)
            probs = torch.softmax(cls_logits, dim=1)
            score, cls_id = torch.max(probs, dim=1)

            if score.item() >= conf_thresh:
                # apply regression deltas to the original proposal
                tx, ty, tw, th = bbox_pred.squeeze().cpu().numpy()
                x_pred = x + tx * w
                y_pred = y + ty * h
                w_pred = w * np.exp(tw)
                h_pred = h * np.exp(th)

                detections.append((int(cls_id.item()), float(score.item()), (x_pred, y_pred, w_pred, h_pred)))

        # Apply NMS
        detections = nms(detections, iou_threshold=self.iou_thresh)
        return detections
    @torch.no_grad()
    def predict_with_visualization(self, image_rgb, conf_thresh=None, show_patches=True):
        self.model.eval()
        if conf_thresh == None:
            conf_thresh=self.default_conf_thresh
        # 1️⃣ Show AKAZE keypoints
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        kps, des = self.extractor.detectAndCompute(gray)
        img_kps = cv2.drawKeypoints(image_rgb, kps, None, color=(0, 255, 0))
        plt.figure(figsize=(8, 6))
        plt.imshow(img_kps)
        plt.title("AKAZE Keypoints")
        plt.axis("off")
        plt.show()

        # 2️⃣ Generate proposals
        proposals = self.generate_proposals(image_rgb)
        print(f"Generated {len(proposals)} RPN proposals.")

        if len(proposals) == 0:
            print("No proposals generated.")
            return []

        detections = []

        # 3️⃣ Visualize sliding windows (optional)
        if show_patches:
            plt.figure(figsize=(12, 6))
            for i, (x, y, w, h, patch) in enumerate(proposals[:5]):
                plt.subplot(1, 5, i + 1)
                plt.imshow(patch)
                plt.title(f"Patch {i}")
                plt.axis("off")
            plt.show()

        # 4️⃣ Classify proposals and apply regression
        for (x, y, w, h, feat) in proposals:
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            cls_logits, bbox_pred = self.model(feat_t)
            probs = torch.softmax(cls_logits, dim=1)
            score, cls_id = torch.max(probs, dim=1)

            if score.item() >= conf_thresh:
                tx, ty, tw, th = bbox_pred.squeeze().cpu().numpy()
                x_pred = x + tx * w
                y_pred = y + ty * h
                w_pred = w * np.exp(tw)
                h_pred = h * np.exp(th)
                detections.append((int(cls_id.item()), float(score.item()), (x_pred, y_pred, w_pred, h_pred)))

        # 5️⃣ Apply NMS
        detections = nms(detections, iou_threshold=self.iou_thresh)

        # 6️⃣ Visualize final detections
        img_out = image_rgb.copy()
        plt.figure(figsize=(8, 6))
        plt.imshow(img_out)
        ax = plt.gca()
        for cls_id, score, (x, y, w, h) in detections:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y - 5, f"{self.classes[cls_id]}:{score:.2f}", color="red")
        plt.title("Final Detections")
        plt.axis("off")
        plt.show()

        return detections
    @torch.no_grad()
    def visualize_random_val_samples(self, val_dataset, num_samples=5):
        self.model.eval()
        indices = random.sample(range(len(val_dataset)), num_samples)

        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(indices):
            img, _, _ = val_dataset[idx]

            # Convert to numpy
            img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img.copy()

            # CHW → HWC
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = img_np.transpose(1, 2, 0)

            # Ensure RGB
            img_np = img_np.astype(np.uint8)
            if img_np.shape[2] != 3:
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img_np

            # Run detector (with bbox regression)
            detections = self.predict(img_rgb)

            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img_rgb)
            if len(detections) == 0:
                plt.title("No detections")
            else:
                for cls_id, score, (x, y, w, h) in detections:
                    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(x, y - 5, f"{self.classes[cls_id]}:{score:.2f}", color="red")
            plt.axis("off")

        plt.tight_layout()
        plt.show()
        # for img in detected:
        #     self.predict_with_visualization(img)

class ObjDetectorRPN:
    def __init__(self, classes,image_size=(256,256), n_clusters=128, iou_thresh=0.5,
                 window_step_size=16,window_size=(64,64),min_size=(64,64),pyramid_scale=1.5,
                 default_conf_thresh=0.7, device="cuda",lr=1e-3):
        self.device = device
        self.classes = classes
        self.num_classes = len(classes)
        self.iou_thresh = iou_thresh
        self.pyramid_scale = pyramid_scale
        self.window_step_size=window_step_size
        self.window_size=window_size
        self.min_size=min_size
        self.image_size = image_size
        self.default_conf_thresh = default_conf_thresh

        # SIFT + BoVW extractor
        self.extractor = AKAZEBoVWExtractor(n_clusters)

        # Classifier head
        self.model = SlidingWindowRegressor(input_dim=n_clusters,num_classes=len(classes)).to(device)
        self.rpn_model = fasterrcnn_resnet50_fpn(weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights).to(device)
        self.rpn_model.eval()  # we only use it for proposals
        for param in self.rpn_model.parameters():
            param.requires_grad = False  # freeze pretrained RPN
        self.optimizer = Adam(
            list(self.model.parameters()),
            lr=lr
        )

    @torch.no_grad()
    def generate_proposals(self, image_rgb, topk=50, score_thresh=0.3):
        """
        Generate candidate boxes using pretrained RPN
        Returns: list of [x, y, w, h]
        """
        self.rpn_model.eval()
        # Convert image to tensor [0,1]
        img_t = torch.tensor(image_rgb / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

        outputs = self.rpn_model(img_t)  # FasterRCNN output
        boxes = outputs[0]['boxes']
        scores = outputs[0]['scores']

        # Filter top proposals by score
        keep = scores > score_thresh
        boxes = boxes[keep]

        if len(boxes) == 0:
            return []

        # Limit to top-k
        if len(boxes) > topk:
            boxes = boxes[:topk]

        # Convert to (x, y, w, h)
        proposals = []
        for b in boxes:
            x1, y1, x2, y2 = b.cpu().numpy()
            proposals.append([x1, y1, x2 - x1, y2 - y1,self.get_rpn_patch(image_rgb, x1, y1, x2 - x1, y2 - y1)])
        return proposals
    
    def get_rpn_patch(self,img_rgb,x,y,w,h):
        x, y, w, h = int(x), int(y), int(w), int(h)

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_rgb.shape[1], x + w), min(img_rgb.shape[0], y + h)
        patch = img_rgb[y1:y2, x1:x2]
        return patch.astype(np.uint8)
    
    def generate_keypoint_proposals(self, image_rgb, eps=30, min_samples=3, max_patches=None,use_sliding_window=True):
        proposals = []
        if use_sliding_window:
            for scaled in image_pyramid(
                image_rgb,
                scale=self.pyramid_scale,
                min_size=self.min_size
            ):
                scale_x = image_rgb.shape[1] / scaled.shape[1]
                scale_y = image_rgb.shape[0] / scaled.shape[0]

                # Slide fixed-size window
                for (x, y, patch) in sliding_window(
                    scaled,
                    self.window_step_size,
                    self.window_size
                ):
                    # --- Skip almost-empty patches (speed boost) ---
                    if patch.mean() < 10:
                        continue

                    # --- Extract features directly from the patch ---
                    features = self.extractor.extract(patch)

                    if features is None:
                        continue

                    # Ensure flat feature vector
                    if isinstance(features, list) or len(features.shape) > 1:
                        features = features.flatten()

                    # Map patch coords back to original image
                    orig_x = int(x * scale_x)
                    orig_y = int(y * scale_y)
                    orig_w = int(self.window_size[0] * scale_x)
                    orig_h = int(self.window_size[1] * scale_y)

                    proposals.append([orig_x, orig_y, orig_w, orig_h, features])

                    if max_patches is not None and len(proposals) >= max_patches:
                        return proposals

            return proposals
        kps, des = self.extractor.detectAndCompute(image_rgb)
        if kps is None or len(kps) == 0:
            return []

        kp_coords = np.array([kp.pt for kp in kps], dtype=np.float32)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(kp_coords)
        labels = clustering.labels_
        
        for lbl in set(labels):
            if lbl == -1:  # noise
                continue
            cluster_points = kp_coords[labels == lbl]
            x_min, y_min = cluster_points.min(axis=0)
            x_max, y_max = cluster_points.max(axis=0)

            # Expand box a bit
            pad_x = (x_max - x_min) * 0.2
            pad_y = (y_max - y_min) * 0.2
            x1 = max(0, int(x_min - pad_x))
            y1 = max(0, int(y_min - pad_y))
            x2 = min(image_rgb.shape[1], int(x_max + pad_x))
            y2 = min(image_rgb.shape[0], int(y_max + pad_y))

            w = x2 - x1
            h = y2 - y1
            patch = image_rgb[y1:y2, x1:x2].copy()
            proposals.append([x1, y1, w, h, patch])

            if max_patches is not None and len(proposals) >= max_patches:
                break

        return proposals

    def build_vocabulary_from_dataset(self, dataloader, max_patches=5000):
        patches = []
        collected = 0

        # Outer tqdm for dataset
        for imgs, bboxes, labels in tqdm(dataloader, desc="Building vocabulary (images)"):
            for i in range(len(imgs)):
                img_rgb = CHWtoHWC(imgs[i])
                if isinstance(img_rgb, torch.Tensor):
                    img_rgb = img_rgb.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                else:
                    img_rgb = img_rgb.astype(np.uint8)

                gt_box = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
                # Inner tqdm for proposals in this image
                for x, y, w, h, patch in tqdm(pyramid_sliding_window(img_rgb,step_size=self.window_step_size,window_size=self.window_size,scale_factor=self.pyramid_scale,min_size=self.min_size), 
                                            desc=f"Image {i} proposals", leave=False):
                    if collected >= max_patches:
                        break

                    if iou((x, y, w, h), tuple(gt_box)) >= self.iou_thresh:
                        # Convert patch to gray
                        if patch.ndim == 3 and patch.shape[2] == 3:
                            patch_gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
                        else:
                            patch_gray = patch
                        patches.append(patch_gray)
                        collected += 1

                if collected >= max_patches:
                    break

            if collected >= max_patches:
                break

        if len(patches) == 0:
            raise RuntimeError("No positive patches found for vocabulary building.")

        self.extractor.build_vocab(patches)
        print(f"Collected {len(patches)} positive patches for BoVW vocabulary.")
    
    def train_one_epoch(self, dataloader, curr_epoch=None, total_epoch=None, val_loader=None):
        self.model.train()
        total_loss = 0
        total_batches = 0
        val_loss = None

        pbar = tqdm(
            total=len(dataloader) + len(val_loader) if val_loader is not None else len(dataloader),
            desc=f"Epoch [{curr_epoch}/{total_epoch}]" if curr_epoch is not None and total_epoch is not None else "Training",
            unit="batch",
            leave=True
        )

        for imgs, bboxes, labels in dataloader:
            feats = []
            targets_label = []
            targets_bbox = []

            for i in range(imgs.shape[0]):
                img_np = CHWtoHWC(imgs[i])
                gt_boxes = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
                gt_label = int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])

                if gt_boxes.ndim == 1:
                    gt_boxes = np.expand_dims(gt_boxes, axis=0)

                for (x, y, w, h, patch) in self.generate_keypoint_proposals(img_np):
                    for gt_box in gt_boxes:
                        if iou((x, y, w, h), tuple(gt_box)) >= self.iou_thresh:
                            feat = self.extractor.extract(patch)
                            feats.append(feat)
                            targets_label.append(gt_label)

                            # compute bbox regression targets
                            x_a, y_a, w_a, h_a = x, y, w, h
                            x_gt, y_gt, w_gt, h_gt = gt_box
                            tx = (x_gt - x_a) / w_a
                            ty = (y_gt - y_a) / h_a
                            tw = np.log(w_gt / w_a)
                            th = np.log(h_gt / h_a)
                            targets_bbox.append([tx, ty, tw, th])
                            break

            pbar.update(1)
            if len(feats) == 0:
                continue

            feats_t = torch.tensor(np.array(feats), dtype=torch.float32).to(self.device)
            targets_label_t = torch.tensor(np.array(targets_label), dtype=torch.long).to(self.device)
            targets_bbox_t = torch.tensor(np.array(targets_bbox), dtype=torch.float32).to(self.device)

            cls_logits, bbox_preds = self.model(feats_t)
            cls_loss = F.cross_entropy(cls_logits, targets_label_t)
            reg_loss = F.smooth_l1_loss(bbox_preds, targets_bbox_t)
            loss = cls_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            pbar.set_postfix({"Training loss": total_loss / total_batches})

        if val_loader is not None:
            val_loss = self.validate(val_loader, pbar)
            pbar.set_postfix({"Training loss": total_loss / total_batches, "Validation Loss": val_loss})
            pbar.refresh()

        return 0.0 if total_batches == 0 else total_loss / total_batches, val_loss if val_loss is not None else 0.0
    
    @torch.no_grad()
    def validate(self, dataloader, pbar: tqdm = None):
        self.model.eval()
        total_loss = 0
        total_batches = 0

        for imgs, bboxes, labels in dataloader:
            feats = []
            targets_label = []
            targets_bbox = []

            for i in range(imgs.shape[0]):
                img_np = CHWtoHWC(imgs[i])
                gt_boxes = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
                gt_label = int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])

                if gt_boxes.ndim == 1:
                    gt_boxes = np.expand_dims(gt_boxes, axis=0)

                for (x, y, w, h, patch) in self.generate_keypoint_proposals(img_np):
                    for gt_box in gt_boxes:
                        if iou((x, y, w, h), tuple(gt_box)) >= self.iou_thresh:
                            feat = self.extractor.extract(patch)
                            feats.append(feat)
                            targets_label.append(gt_label)

                            # compute bbox regression targets
                            x_a, y_a, w_a, h_a = x, y, w, h
                            x_gt, y_gt, w_gt, h_gt = gt_box
                            tx = (x_gt - x_a) / w_a
                            ty = (y_gt - y_a) / h_a
                            tw = np.log(w_gt / w_a)
                            th = np.log(h_gt / h_a)
                            targets_bbox.append([tx, ty, tw, th])
                            break

            if len(feats) == 0:
                if pbar is not None:
                    pbar.update(1)
                continue

            feats_t = torch.tensor(np.array(feats), dtype=torch.float32).to(self.device)
            targets_label_t = torch.tensor(np.array(targets_label), dtype=torch.long).to(self.device)
            targets_bbox_t = torch.tensor(np.array(targets_bbox), dtype=torch.float32).to(self.device)

            cls_logits, bbox_preds = self.model(feats_t)
            cls_loss = F.cross_entropy(cls_logits, targets_label_t)
            reg_loss = F.smooth_l1_loss(bbox_preds, targets_bbox_t)
            loss = cls_loss + reg_loss

            total_loss += loss.item()
            total_batches += 1

            if pbar is not None:
                pbar.update(1)

        return 0.0 if total_batches == 0 else total_loss / total_batches

    def train(self, train_loader, val_loader, val_dataset, epochs=10, vis_every=2,checkpoint_dir="checkpoints", checkpoint_every=20):
        os.makedirs(checkpoint_dir, exist_ok=True)
        total_loss = []
        total_val_loss = []
        t0 = time.time()
        for epoch in range(epochs):
            train_loss,val_loss = self.train_one_epoch(train_loader,curr_epoch=epoch+1,total_epoch=epochs,val_loader=val_loader)

            total_loss.append(train_loss)
            total_val_loss.append(val_loss)
            # Show validation predictions every few epochs
            if (epoch + 1) % vis_every == 0:
                print("Showing validation sample predictions...")
                self.visualize_random_val_samples(val_dataset, num_samples=5)
            if (epoch + 1) % checkpoint_every == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_checkpoint.pth")
                save_detector(self, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        elapsed = (time.time() - t0)
        print(f'Total Training Time : {format_time(elapsed)}')
        return total_loss,total_val_loss

    def visualize_detections(self, image_rgb, detections, class_names=None):
        plt.figure(figsize=(10, 8))
        plt.imshow(image_rgb)
        ax = plt.gca()

        for cls_id, score, (x, y, w, h) in detections:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            label = class_names[cls_id] if class_names is not None else self.classes[cls_id]
            ax.text(x, y - 5, f"{label}:{score:.2f}", color="red", fontsize=10, backgroundcolor='white')

        plt.axis("off")
        plt.show()


    @torch.no_grad()
    def predict(self, image_rgb, conf_thresh=None):
        self.model.eval()
        detections = []
        if conf_thresh == None:
            conf_thresh=self.default_conf_thresh
        if not self.extractor.is_fitted:
            raise RuntimeError("BoVW vocabulary not built. Call build_vocabulary_from_dataset() first.")

        proposals = self.generate_keypoint_proposals(image_rgb)

        for (x, y, w, h, patch) in proposals:
            min_size = 8
            if patch.shape[0] < min_size or patch.shape[1] < min_size:
                continue

            feat = self.extractor.extract(patch)
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

            cls_logits, bbox_pred = self.model(feat_t)
            probs = torch.softmax(cls_logits, dim=1)
            score, cls_id = torch.max(probs, dim=1)

            if score.item() >= conf_thresh:
                # apply regression deltas to the original proposal
                tx, ty, tw, th = bbox_pred.squeeze().cpu().numpy()
                x_pred = x + tx * w
                y_pred = y + ty * h
                w_pred = w * np.exp(tw)
                h_pred = h * np.exp(th)

                detections.append((int(cls_id.item()), float(score.item()), (x_pred, y_pred, w_pred, h_pred)))

        # Apply NMS
        detections = nms(detections, iou_threshold=self.iou_thresh)
        return detections
    @torch.no_grad()
    def predict_with_visualization(self, image_rgb, conf_thresh=None, show_patches=True):
        self.model.eval()
        if conf_thresh == None:
            conf_thresh=self.default_conf_thresh
        # 1️⃣ Show AKAZE keypoints
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        kps, des = self.extractor.detectAndCompute(gray)
        img_kps = cv2.drawKeypoints(image_rgb, kps, None, color=(0, 255, 0))
        plt.figure(figsize=(8, 6))
        plt.imshow(img_kps)
        plt.title("AKAZE Keypoints")
        plt.axis("off")
        plt.show()

        # 2️⃣ Generate proposals
        proposals = self.generate_proposals(image_rgb)
        print(f"Generated {len(proposals)} RPN proposals.")

        if len(proposals) == 0:
            print("No proposals generated.")
            return []

        detections = []

        # 3️⃣ Visualize sliding windows (optional)
        if show_patches:
            plt.figure(figsize=(12, 6))
            for i, (x, y, w, h, patch) in enumerate(proposals[:5]):
                plt.subplot(1, 5, i + 1)
                plt.imshow(patch)
                plt.title(f"Patch {i}")
                plt.axis("off")
            plt.show()

        # 4️⃣ Classify proposals and apply regression
        for (x, y, w, h, patch) in proposals:
            feat = self.extractor.extract(patch)
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)
            cls_logits, bbox_pred = self.model(feat_t)
            probs = torch.softmax(cls_logits, dim=1)
            score, cls_id = torch.max(probs, dim=1)

            if score.item() >= conf_thresh:
                tx, ty, tw, th = bbox_pred.squeeze().cpu().numpy()
                x_pred = x + tx * w
                y_pred = y + ty * h
                w_pred = w * np.exp(tw)
                h_pred = h * np.exp(th)
                detections.append((int(cls_id.item()), float(score.item()), (x_pred, y_pred, w_pred, h_pred)))

        # 5️⃣ Apply NMS
        detections = nms(detections, iou_threshold=self.iou_thresh)

        # 6️⃣ Visualize final detections
        img_out = image_rgb.copy()
        plt.figure(figsize=(8, 6))
        plt.imshow(img_out)
        ax = plt.gca()
        for cls_id, score, (x, y, w, h) in detections:
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x, y - 5, f"{self.classes[cls_id]}:{score:.2f}", color="red")
        plt.title("Final Detections")
        plt.axis("off")
        plt.show()

        return detections
    @torch.no_grad()
    def visualize_random_val_samples(self, val_dataset, num_samples=5):
        self.model.eval()
        indices = random.sample(range(len(val_dataset)), num_samples)

        plt.figure(figsize=(15, 6))
        for i, idx in enumerate(indices):
            img, _, _ = val_dataset[idx]

            # Convert to numpy
            img_np = img.cpu().numpy() if isinstance(img, torch.Tensor) else img.copy()

            # CHW → HWC
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = img_np.transpose(1, 2, 0)

            # Ensure RGB
            img_np = img_np.astype(np.uint8)
            if img_np.shape[2] != 3:
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
            else:
                img_rgb = img_np

            # Run detector (with bbox regression)
            detections = self.predict(img_rgb)

            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img_rgb)
            if len(detections) == 0:
                plt.title("No detections")
            else:
                for cls_id, score, (x, y, w, h) in detections:
                    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(x, y - 5, f"{self.classes[cls_id]}:{score:.2f}", color="red")
            plt.axis("off")

        plt.tight_layout()
        plt.show()
        # for img in detected:
        #     self.predict_with_visualization(img)


def save_detector(detector, path):
    """
    Save ObjDetectorRPN to disk.
    
    Args:
        detector: ObjDetectorRPN instance
        path: file path to save the model (e.g., "detector_rpn.pth")
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        "model_state": detector.model.state_dict(),
        "optimizer_state": detector.optimizer.state_dict(),
        "classes": detector.classes,
        "num_classes": detector.num_classes,
        "n_clusters": detector.extractor.n_clusters,
        "iou_thresh": detector.iou_thresh,
        "window_step_size": detector.window_step_size,
        "window_sizes": detector.window_sizes,
        "image_size": detector.image_size,
        "extractor_kmeans": detector.extractor.kmeans if detector.extractor.is_fitted else None
    }, path)
    
    print(f"Saved ObjDetectorRPN to {path}")


def load_detector(path, device="cuda"):
    """
    Load ObjDetectorRPN from disk.
    
    Args:
        path: file path of saved detector
        device: "cuda" or "cpu"
    
    Returns:
        detector: ObjDetectorRPN instance
    """
    checkpoint = torch.load(path, map_location=device)
    
    detector = ObjDetectorRPN(
        classes=checkpoint["classes"],
        n_clusters=checkpoint["n_clusters"],
        iou_thresh=checkpoint["iou_thresh"],
        window_step_size=checkpoint["window_step_size"],
        window_sizes=checkpoint["window_sizes"],
        image_size=checkpoint.get("image_size", (256, 256)),
        device=device
    )
    
    # Load classifier weights and optimizer
    detector.model.load_state_dict(checkpoint["model_state"])
    detector.optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    # Restore BoVW vocabulary
    if checkpoint["extractor_kmeans"] is not None:
        detector.extractor.kmeans = checkpoint["extractor_kmeans"]
        detector.extractor.is_fitted = True
    
    print(f"Loaded ObjDetectorRPN from {path}")
    return detector
