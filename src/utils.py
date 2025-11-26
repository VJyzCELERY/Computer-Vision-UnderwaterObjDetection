import torch
import cv2
import numpy as np

def sliding_window(image, step_size=32, window_sizes=[(32,32),(64,64)]):
    for window_size in window_sizes:
        for y in range(0, image.shape[0] - window_size[0], step_size):
            for x in range(0, image.shape[1] - window_size[1], step_size):
                patch = image[y:y+window_size[0], x:x+window_size[1]]
                if patch.shape[0] != window_size[0] or patch.shape[1] != window_size[1]:
                    continue
                yield (x, y,window_size[0],window_size[1], patch)

def image_pyramid(image, scale=1.5, min_size=(64, 64)):
    current = image.copy()
    yield current

    while True:
        w = int(current.shape[1] / scale)
        h = int(current.shape[0] / scale)

        if w < min_size[0] or h < min_size[1]:
            break

        current = cv2.resize(current, (w, h))
        yield current

def pyramid_sliding_window(image, step_size=32, window_sizes=[32,64,128], scale_factor=1.5, min_size=(32,32)):
    """
    Image pyramid + multi-scale sliding window.
    Yields windows with bounding boxes mapped to ORIGINAL image size.
    """
    orig_h, orig_w = image.shape[:2]
    scaled_image = image.copy()
    scale = 1.0

    while scaled_image.shape[0] >= min_size[1] and scaled_image.shape[1] >= min_size[0]:
        for window_size in window_sizes:
            for y in range(0, scaled_image.shape[0] - window_size + 1, step_size):
                for x in range(0, scaled_image.shape[1] - window_size + 1, step_size):
                    # Map bbox back to original image coordinates
                    x_orig = int(x / scale)
                    y_orig = int(y / scale)
                    w_orig = int(window_size / scale)
                    h_orig = int(window_size / scale)

                    window = scaled_image[y:y+window_size, x:x+window_size]
                    yield x_orig, y_orig, w_orig, h_orig, window

        # Scale down image for next pyramid level
        scale *= scale_factor
        new_w = int(orig_w / scale)
        new_h = int(orig_h / scale)
        if new_w < min_size[0] or new_h < min_size[1]:
            break
        scaled_image = cv2.resize(image, (new_w, new_h))


def aggregate_pyramid_features(proposals, iou_thresh=0.3):
    """
    proposals: list of (x, y, w, h, feature)
    Aggregate features of overlapping windows across scales.
    Returns: list of (x, y, w, h, aggregated_feature)
    """
    aggregated = []
    used = set()
    
    for i, (x1, y1, w1, h1, feat1) in enumerate(proposals):
        if i in used:
            continue
        agg_feat = [feat1]
        x_min, y_min = x1, y1
        x_max, y_max = x1 + w1, y1 + h1
        used.add(i)
        
        for j, (x2, y2, w2, h2, feat2) in enumerate(proposals):
            if j in used:
                continue
            # compute IoU
            xi1 = max(x_min, x2)
            yi1 = max(y_min, y2)
            xi2 = min(x_max, x2 + w2)
            yi2 = min(y_max, y2 + h2)
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            union_area = (x_max - x_min) * (y_max - y_min) + w2 * h2 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou >= iou_thresh:
                agg_feat.append(feat2)
                # update bbox to union
                x_min = min(x_min, x2)
                y_min = min(y_min, y2)
                x_max = max(x_max, x2 + w2)
                y_max = max(y_max, y2 + h2)
                used.add(j)
        
        # Concatenate all overlapping features
        aggregated_feat = np.mean(agg_feat, axis=0)
        aggregated.append((x_min, y_min, x_max - x_min, y_max - y_min, aggregated_feat))
    
    return aggregated


# def pyramid_sliding_window(image, step_size=32, window_size=64, scale_factor=1.5, min_size=(64,64)):
#     """
#     Image pyramid + sliding window.
#     """
#     h, w = image.shape[:2]
#     while w >= min_size[0] and h >= min_size[1]:
#         # Apply sliding window on current scale
#         for y in range(0, h - window_size + 1, step_size):
#             for x in range(0, w - window_size + 1, step_size):
#                 yield x, y, window_size, window_size, image[y:y+window_size, x:x+window_size]

#         # Resize image
#         w = int(w / scale_factor)
#         h = int(h / scale_factor)
#         image = cv2.resize(image, (w, h))


def CHWtoHWC(img):
    if isinstance(img, torch.Tensor):
        img_np = img.cpu().numpy()
        if img_np.ndim == 3 and img_np.shape[0] in [1,3]:  # CHW -> HWC
            img_rgb = img_np.transpose(1,2,0)
        else:
            img_rgb = img_np
    else:
        img_rgb = img
        if img_rgb.ndim == 2: 
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    return img_rgb

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def iou(boxA, boxB):
    # boxes: (x, y, w, h)
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax + aw, bx + bw)
    inter_y2 = min(ay + ah, by + bh)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    areaA = aw * ah
    areaB = bw * bh
    return inter_area / (areaA + areaB - inter_area)
def merge_overlapping_boxes(detections, iou_thresh=0.5):
    """
    Merge boxes of the same class if they strongly overlap or one contains another.
    detections: [(cls_id, score, (x,y,w,h)), ...]
    """
    merged = []
    detections = sorted(detections, key=lambda x: -x[1])  # high score first

    def box_contains(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return (x1 <= x2 and y1 <= y2 and
                x1 + w1 >= x2 + w2 and
                y1 + h1 >= y2 + h2)

    while detections:
        cls_id, score, box = detections.pop(0)
        x, y, w, h = box
        keep = True

        for i, (m_cls, m_score, m_box) in enumerate(merged):
            if cls_id != m_cls:
                continue

            iou_val = iou(box, m_box)
            contains = box_contains(m_box, box) or box_contains(box, m_box)

            if iou_val > iou_thresh or contains:
                # merge by expanding
                mx, my, mw, mh = m_box
                nx = min(x, mx)
                ny = min(y, my)
                nxe = max(x + w, mx + mw)
                nye = max(y + h, my + mh)

                new_box = (nx, ny, nxe - nx, nye - ny)
                new_score = max(score, m_score)

                merged[i] = (cls_id, new_score, new_box)
                keep = False
                break

        if keep:
            merged.append((cls_id, score, box))

    return merged

def nms(detections, iou_threshold=0.5):
    # detections: list of (cls_id, score, (x,y,w,h))
    detections_sorted = sorted(detections, key=lambda x: x[1], reverse=True)
    final_boxes = []
    
    while detections_sorted:
        best = detections_sorted.pop(0)
        final_boxes.append(best)
        to_remove = []
        for i, det in enumerate(detections_sorted):
            _, _, (x,y,w,h) = det
            _, _, (bx,by,bw,bh) = best
            iou_val = iou((x,y,w,h), (bx,by,bw,bh))
            # adapt threshold based on size
            area = w*h
            threshold = iou_threshold  
            if iou_val > threshold:
                to_remove.append(i)
        for idx in reversed(to_remove):
            detections_sorted.pop(idx)
    return final_boxes
