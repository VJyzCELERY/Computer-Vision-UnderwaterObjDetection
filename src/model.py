import time
import os
import cv2
from tqdm import tqdm
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import torch.nn as nn
import torchvision
import torch
from sklearn.cluster import DBSCAN
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
import random
from src.utils import sliding_window,image_pyramid,nms,iou,format_time,CHWtoHWC,pyramid_sliding_window,merge_overlapping_boxes,aggregate_pyramid_features



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
        self.cls_head = nn.Linear(256, num_classes)  # classification
        self.reg_head = nn.Linear(256, 4)           # bbox regression

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

class MultiFeatureBoVWExtractor:
    def __init__(self, n_clusters=128, use_sift=False):
        self.n_clusters = n_clusters
        
        # Feature extractors
        self.akaze = cv2.AKAZE_create()
        self.orb = cv2.ORB_create()
        self.brisk = cv2.BRISK_create()
        self.use_sift = use_sift
        
        if use_sift:
            self.sift = cv2.SIFT_create()
        else:
            self.sift = None

        # Individual vocabularies
        self.kmeans_akaze = MiniBatchKMeans(n_clusters=n_clusters)
        self.kmeans_orb   = MiniBatchKMeans(n_clusters=n_clusters)
        self.kmeans_brisk = MiniBatchKMeans(n_clusters=n_clusters)
        self.kmeans_sift  = MiniBatchKMeans(n_clusters=n_clusters) if use_sift else None

        self.is_fitted = False

    def _to_gray(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    def _extract_desc(self, extractor, img):
        kp, des = extractor.detectAndCompute(img, None)
        return des

    def build_vocab(self, images):
        akaze_desc, orb_desc, brisk_desc, sift_desc = [], [], [], []

        for img in images:
            img = self._to_gray(img)

            d1 = self._extract_desc(self.akaze, img)
            d2 = self._extract_desc(self.orb, img)
            d3 = self._extract_desc(self.brisk, img)
            d4 = self._extract_desc(self.sift, img) if self.use_sift else None

            if d1 is not None: akaze_desc.append(d1)
            if d2 is not None: orb_desc.append(d2)
            if d3 is not None: brisk_desc.append(d3)
            if self.use_sift and d4 is not None: sift_desc.append(d4)

        # Stack
        akaze_desc = np.vstack(akaze_desc)
        orb_desc   = np.vstack(orb_desc)
        brisk_desc = np.vstack(brisk_desc)
        
        self.kmeans_akaze.fit(akaze_desc)
        self.kmeans_orb.fit(orb_desc)
        self.kmeans_brisk.fit(brisk_desc)

        if self.use_sift:
            sift_desc = np.vstack(sift_desc)
            self.kmeans_sift.fit(sift_desc)

        self.is_fitted = True

    def _compute_hist(self, desc, kmeans):
        if desc is None:
            return np.zeros(self.n_clusters, dtype=np.float32)
        words = kmeans.predict(desc)
        hist, _ = np.histogram(words, bins=self.n_clusters, range=(0, self.n_clusters))
        hist = hist.astype(np.float32)
        return hist / (np.linalg.norm(hist) + 1e-7)

    def extract(self, img):
        assert self.is_fitted, "Call build_vocab() before extract()"
        img = self._to_gray(img)

        d1 = self._extract_desc(self.akaze, img)
        d2 = self._extract_desc(self.orb, img)
        d3 = self._extract_desc(self.brisk, img)
        d4 = self._extract_desc(self.sift, img) if self.use_sift else None

        h1 = self._compute_hist(d1, self.kmeans_akaze)
        h2 = self._compute_hist(d2, self.kmeans_orb)
        h3 = self._compute_hist(d3, self.kmeans_brisk)

        feats = [h1, h2, h3]

        if self.use_sift:
            h4 = self._compute_hist(d4, self.kmeans_sift)
            feats.append(h4)

        # 🔥 Concatenate all feature vectors
        return np.concatenate(feats)

class ObjDetector:
    def __init__(self, classes, n_clusters=128,iou_thresh=0.5,window_step_size=32,window_sizes=[16,32,64,128], device="cuda",lr=1e-3):
        num_classes = len(classes)
        self.device = device
        self.extractor = SIFTBoVWExtractor(n_clusters)
        self.model = SlidingWindowClassifier(n_clusters, num_classes).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.classes =classes
        self.num_classes = num_classes
        self.iou_thresh = iou_thresh
        self.window_step_size = window_step_size
        self.window_sizes = window_sizes
    
    def build_vocabulary_from_dataset(self, dataloader, max_patches=5000):
        patches = []
        collected = 0

        for imgs, bboxes, labels in dataloader:
            for i in range(len(imgs)):
                img_rgb = CHWtoHWC(imgs[i])
                if isinstance(img_rgb, torch.Tensor):
                    img_rgb = img_rgb.cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                else:
                    img_rgb = img_rgb.astype(np.uint8)

                gt_box = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
                
                for (x, y, w, h), patch in sliding_window(img_rgb,step_size=self.window_step_size,window_sizes=self.window_sizes):
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

        if len(patches) == 0:
            raise RuntimeError("No positive patches found for vocabulary building.")

        self.extractor.build_vocab(patches)
        print(f"Collected {len(patches)} positive patches for BoVW vocabulary.")


    def train_one_epoch(self, dataloader,curr_epoch=None,total_epoch=None,val_loader = None):
        self.model.train()
        total_loss = 0
        total_batches = 0
        val_loss=None
        pbar = tqdm(total=len(dataloader)+len(val_loader) if val_loader is not None else 0, desc=f"Epoch [{curr_epoch+1}/{total_epoch}]" if curr_epoch is not None and total_epoch is not None else "Training", unit="batch",leave=True)
        for imgs, bboxes, labels in dataloader:
            feats = []
            targets = []

            for i in range(imgs.shape[0]):
                img_np = CHWtoHWC(imgs[i])
                gt_boxes = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
                gt_label = int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])

                if gt_boxes.ndim == 1:
                    gt_boxes = np.expand_dims(gt_boxes, axis=0)
                for (x,y,w,h), patch in sliding_window(img_np,step_size=self.window_step_size,window_sizes=self.window_sizes):
                    for gt_box in gt_boxes:
                        if iou((x,y,w,h), tuple(gt_box)) >= self.iou_thresh:
                            feat = self.extractor.extract(patch)
                            feats.append(feat)
                            targets.append(gt_label)
                            break

            if len(feats) == 0:
                continue
            feats_t = torch.tensor(np.array(feats), dtype=torch.float32).to(self.device)
            targets_t = torch.tensor(np.array(targets), dtype=torch.long).to(self.device)

            preds = self.model(feats_t)
            loss = F.cross_entropy(preds, targets_t)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_batches += 1
            pbar.update(1)
            pbar.set_postfix({"Training loss": total_loss / total_batches})
        if val_loader is not None:
            val_loss = self.validate(val_loader,pbar)
            pbar.set_postfix({"Training loss": total_loss / total_batches,"Validation Loss":val_loss})
            pbar.refresh()
        
        return 0.0 if total_batches == 0 else total_loss / total_batches, val_loss if val_loss is not None else 0.0
    
    @torch.no_grad()
    def validate(self, dataloader,pbar :tqdm =None):
        self.model.eval()
        total_loss = 0
        total_batches = 0

        for imgs, bboxes, labels in dataloader:
            feats = []
            targets = []

            for i in range(imgs.shape[0]):
                img_np = CHWtoHWC(imgs[i])
                gt_boxes = bboxes[i].cpu().numpy() if isinstance(bboxes[i], torch.Tensor) else bboxes[i]
                gt_label = int(labels[i].item()) if isinstance(labels[i], torch.Tensor) else int(labels[i])

                if gt_boxes.ndim == 1:
                    gt_boxes = np.expand_dims(gt_boxes, axis=0)

                for (x,y,w,h), patch in sliding_window(img_np,step_size=self.window_step_size,window_sizes=self.window_sizes):
                    for gt_box in gt_boxes:
                        if iou((x,y,w,h), tuple(gt_box)) >= self.iou_thresh:
                            feat = self.extractor.extract(patch)
                            feats.append(feat)
                            targets.append(gt_label)
                            break

            if len(feats) == 0:
                continue
            feats_t = torch.tensor(np.array(feats), dtype=torch.float32).to(self.device)
            targets_t = torch.tensor(np.array(targets), dtype=torch.long).to(self.device)

            preds = self.model(feats_t)
            loss = F.cross_entropy(preds, targets_t)

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
            train_loss,val_loss = self.train_one_epoch(train_loader,curr_epoch=epoch,total_epoch=epochs,val_loader=val_loader)

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
        
        plt.figure(figsize=(10,8))
        plt.imshow(image_rgb)
        ax = plt.gca()
        for cls_id, score, (x,y,w,h) in detections:
            rect = plt.Rectangle((x,y), w, h, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            label = class_names[cls_id] if class_names is not None else str(cls_id)
            ax.text(x, y-5, f"{label}:{score:.2f}", color="red", fontsize=10, backgroundcolor='white')
        plt.axis("off")
        plt.show()

    @torch.no_grad()
    def predict(self, image_rgb, conf_thresh=0.5):
        """
        image_rgb: numpy array (H, W, 3) in RGB format
        Returns:
            detections: List of (class_id, score, (x, y, w, h))
        """
        self.model.eval()

        detections = []

        # Make sure extractor is fitted
        if not self.extractor.is_fitted:
            raise RuntimeError("BoVW vocabulary has not been built. Call build_vocabulary_from_dataset() first.")

        for (x, y, w, h), patch in sliding_window(image_rgb,step_size=self.window_step_size,window_sizes=self.window_sizes):

            # Ensure uint8
            patch = patch.astype(np.uint8)

            # Extract BoVW feature
            feat = self.extractor.extract(patch)

            # Torch tensor
            feat_t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(self.device)

            # Forward pass
            logits = self.model(feat_t)
            probs = torch.softmax(logits, dim=1)

            score, cls_id = torch.max(probs, dim=1)

            if score.item() >= conf_thresh:
                detections.append((int(cls_id.item()), float(score.item()), (x, y, w, h)))

        # Apply NMS
        detections = nms(detections, iou_threshold=self.iou_thresh)

        return detections
    @torch.no_grad()
    def visualize_random_val_samples(self, val_dataset, num_samples=5):
        self.model.eval()
        indices = random.sample(range(len(val_dataset)), num_samples)

        plt.figure(figsize=(15, 6))

        for i, idx in enumerate(indices):
            img, _, _ = val_dataset[idx]

            # To numpy
            if isinstance(img, torch.Tensor):
                img_np = img.cpu().numpy()
            else:
                img_np = img.copy()

            # Handle CHW → HWC
            if img_np.ndim == 3 and img_np.shape[0] in [1, 3]:
                img_np = img_np.transpose(1, 2, 0)

            # Ensure uint8
            img_np = img_np.astype(np.uint8)

            # RGB
            if img_np.shape[2] == 3:
                img_rgb = img_np
            else:
                img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

            # Run detector
            detections = self.predict(img_rgb)

            # Plot
            plt.subplot(1, num_samples, i + 1)
            plt.imshow(img_rgb)

            # Draw detections safely
            if len(detections) == 0:
                plt.title("No detections")
            else:
                for cls_id, score, (x, y, w, h) in detections:
                    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
                    plt.gca().add_patch(rect)
                    plt.text(x, y - 5, f"{self.classes[cls_id]}:{score:.2f}", color="red", backgroundcolor="white")

            plt.axis("off")

        plt.tight_layout()
        plt.show()


class ObjDetectorRPN:
    def __init__(self, classes,image_size=(256,256), n_clusters=128, iou_thresh=0.5,window_step_size=32,window_size=64,min_size=(64,64),default_conf_thresh=0.7, device="cuda",lr=1e-3,use_sliding_window=True):
        self.device = device
        self.classes = classes
        self.use_sliding_window=use_sliding_window
        self.num_classes = len(classes)
        self.iou_thresh = iou_thresh
        self.window_step_size=window_step_size
        self.window_size=window_size
        self.min_size = min_size
        self.image_size = image_size
        self.default_conf_thresh = default_conf_thresh
        self.lr=lr
        # SIFT + BoVW extractor
        self.extractor = AKAZEBoVWExtractor(n_clusters)

        # Classifier head (now auto-sized)
        # self.model = SlidingWindowRegressor(
        #     input_dim=feature_dim,
        #     num_classes=len(classes)
        # ).to(device)
        # self.optimizer = Adam(
        #     list(self.model.parameters()),
        #     lr=lr
        # )
        self.model = None
        self.rpn_model = fasterrcnn_resnet50_fpn(weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights).to(device)
        self.rpn_model.eval()  # we only use it for proposals
        for param in self.rpn_model.parameters():
            param.requires_grad = False  # freeze pretrained RPN
        self.optimizer = None
        

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
    
    def generate_keypoint_proposals(self, image_rgb, eps=30, min_samples=3, max_patches=None,use_sliding_window=True,return_patch=False):
        if use_sliding_window and (not self.extractor.is_fitted or return_patch==True):
            return pyramid_sliding_window(image_rgb,step_size=self.window_step_size,window_sizes=self.window_size,min_size=self.min_size)
        elif use_sliding_window and self.extractor.is_fitted:
            return self.generate_aggregated_proposals(image_rgb,self.iou_thresh)
        kps, des = self.extractor.detectAndCompute(image_rgb)
        if kps is None or len(kps) == 0:
            return []

        kp_coords = np.array([kp.pt for kp in kps], dtype=np.float32)
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(kp_coords)
        labels = clustering.labels_

        proposals = []
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
            min_size=8
            if self.extractor.is_fitted and return_patch==False:
                if patch.shape[0] < min_size or patch.shape[1] < min_size:
                    continue
                proposals.append([x1,y1,w,h,self.extractor.extract(patch)])
            else:
                proposals.append([x1, y1, w, h, patch])

            if max_patches is not None and len(proposals) >= max_patches:
                break

        return proposals
    def generate_aggregated_proposals(self, image_rgb, iou_thresh=0.3):
        proposals = []
        for x, y, w, h, patch in pyramid_sliding_window(
                image_rgb,
                step_size=self.window_step_size,
                window_sizes=[32, 64, 128]):
            feat = self.extractor.extract(patch)
            proposals.append((x, y, w, h, feat))

        aggregated = aggregate_pyramid_features(proposals, iou_thresh=iou_thresh)
        return aggregated
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
                for x, y, w, h, patch in tqdm(self.generate_keypoint_proposals(img_rgb,use_sliding_window=self.use_sliding_window), 
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

        feature = self.extractor.extract(patches[0])
        self.model = SlidingWindowRegressor(
            input_dim=feature.shape[0],
            num_classes=len(self.classes)
        ).to(self.device)
        self.optimizer = Adam(
            list(self.model.parameters()),
            lr=self.lr
        )
    
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

                for (x, y, w, h, patch) in self.generate_keypoint_proposals(img_np,use_sliding_window=self.use_sliding_window,return_patch=True):
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

                for (x, y, w, h, patch) in self.generate_keypoint_proposals(img_np,use_sliding_window=self.use_sliding_window,return_patch=True):
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

        proposals = self.generate_keypoint_proposals(image_rgb,use_sliding_window=self.use_sliding_window)

        for (x, y, w, h, feat) in proposals:
            min_size = 8
            # if patch.shape[0] < min_size or patch.shape[1] < min_size:
            #     continue

            # feat = self.extractor.extract(patch)
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
        # Merge overlapping same-class boxes
        detections = merge_overlapping_boxes(detections, iou_thresh=self.iou_thresh)

        # Then NMS
        detections = nms(detections, iou_threshold=self.iou_thresh)

        return detections
    @torch.no_grad()
    def predict_with_visualization(self, image_rgb, conf_thresh=None, show_patches=True):
        self.model.eval()
        if conf_thresh == None:
            conf_thresh=self.default_conf_thresh
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        kps, des = self.extractor.detectAndCompute(gray)
        img_kps = cv2.drawKeypoints(image_rgb, kps, None, color=(0, 255, 0))
        plt.figure(figsize=(8, 6))
        plt.imshow(img_kps)
        plt.title("AKAZE Keypoints")
        plt.axis("off")
        plt.show()

        proposals = self.generate_keypoint_proposals(image_rgb,use_sliding_window=self.use_sliding_window,return_patch=True)
        print(f"Generated {len(proposals)} RPN proposals.")

        if len(proposals) == 0:
            print("No proposals generated.")
            return []

        detections = []

        if show_patches:
            plt.figure(figsize=(12, 6))
            for i, (x, y, w, h, patch) in enumerate(proposals[:5]):
                plt.subplot(1, 5, i + 1)
                plt.imshow(patch)
                plt.title(f"Patch {i}")
                plt.axis("off")
            plt.show()

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
        detections = merge_overlapping_boxes(detections, iou_thresh=self.iou_thresh)
        detections = nms(detections, iou_threshold=self.iou_thresh)

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

class EdgePAN(nn.Module):
    def __init__(self, in_channels=1, feat_channels=16):
        super().__init__()
        
        # 1x1 convs to make channel dimensions consistent
        self.lateral_conv1 = nn.Conv2d(in_channels*4, feat_channels, 1)
        self.lateral_conv2 = nn.Conv2d(in_channels*4, feat_channels, 1)
        self.lateral_conv3 = nn.Conv2d(in_channels*4, feat_channels, 1)

        # PAN bottom-up downsampling layers
        self.down_conv1 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1)
        self.down_conv2 = nn.Conv2d(feat_channels, feat_channels, kernel_size=3, stride=2, padding=1)

    def sobel_filters(self, x):
        # Simple Sobel kernels
        sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=x.device)
        sobel_y = sobel_x.t()

        sobel_x = sobel_x.view(1,1,3,3)
        sobel_y = sobel_y.view(1,1,3,3)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        return gx, gy

    def laplacian_filter(self, x):
        kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=x.device)
        kernel = kernel.view(1,1,3,3)
        return F.conv2d(x, kernel, padding=1)

    def canny_like(self, x):
        # Simple edge magnitude (not full Canny but differentiable)
        gx, gy = self.sobel_filters(x)
        return torch.sqrt(gx**2 + gy**2 + 1e-6)

    def extract_features(self, x):
        """
        Create handcrafted feature maps:
        channels = [original, sobel_x, sobel_y, laplacian]
        """
        gx, gy = self.sobel_filters(x)
        lap = self.laplacian_filter(x)
        edge = self.canny_like(x)

        return torch.cat([x, gx, gy, lap], dim=1)

    def build_pyramid(self, x):
        """Build multi-scale feature pyramid"""
        f1 = self.extract_features(x)                       # H, W
        f2 = self.extract_features(F.avg_pool2d(x, 2))      # H/2, W/2
        f3 = self.extract_features(F.avg_pool2d(x, 4))      # H/4, W/4
        return f1, f2, f3

    def forward(self, x):
        """
        x shape: [B, 1, H, W]
        """

        # 1. Build pyramid
        p1, p2, p3 = self.build_pyramid(x)

        # 2. Align channels
        p1 = self.lateral_conv1(p1)
        p2 = self.lateral_conv2(p2)
        p3 = self.lateral_conv3(p3)

        # 3. PAN bottom-up fusion
        d2 = self.down_conv1(p1) + p2
        d3 = self.down_conv2(d2) + p3

        return {
            "P1": p1,  # high-res features
            "P2": d2,  # mid features
            "P3": d3,  # low-res, strong semantics
        }

class PANToRegressorAdapter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, pan_feats):
        # pan_feats: dict with P1, P2, P3
        f1 = self.pool(pan_feats["P1"])
        f2 = self.pool(pan_feats["P2"])
        f3 = self.pool(pan_feats["P3"])

        # flatten
        f1 = f1.view(f1.size(0), -1)
        f2 = f2.view(f2.size(0), -1)
        f3 = f3.view(f3.size(0), -1)

        # concatenate multi-scale features
        return torch.cat([f1, f2, f3], dim=1)


def save_detector(detector, path):
    """
    Save ObjDetectorRPN to disk.
    
    Args:
        detector: ObjDetectorRPN instance
        path: file path to save the model (e.g., "detector_rpn.pth")
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
        "model_state": detector.model.state_dict() if detector.model is not None else None,
        "optimizer_state": detector.optimizer.state_dict() if detector.optimizer is not None else None,
        "classes": detector.classes,
        "num_classes": detector.num_classes,
        "n_clusters": detector.extractor.n_clusters,
        "iou_thresh": detector.iou_thresh,
        "window_step_size": detector.window_step_size,
        "window_sizes": detector.window_size,
        "min_size":detector.min_size,
        "default_conf_thresh":detector.default_conf_thresh,
        'lr':detector.lr,
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
        min_size=checkpoint['min_size'],
        default_conf_thresh=checkpoint['default_conf_thresh'],
        lr=checkpoint['lr'],
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
