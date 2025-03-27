import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc
from models import SmallObjectDetector
from data_loader import stratified_split
from torch.utils.data import DataLoader
from train import collate_fn
from collections import defaultdict

S, C = 7, 2
IMG_SIZE = 112

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def postprocess(pred, threshold=0.5):
    pred = pred.squeeze(0).detach().cpu().numpy().reshape((S, S, 5 + C))
    detections = []
    for i in range(S):
        for j in range(S):
            cell = pred[i, j]
            objectness = cell[4]
            if objectness > threshold:
                x_center, y_center, w, h = cell[0:4]
                cls_scores = cell[5:]
                predicted_class = int(np.argmax(cls_scores))

                grid_size = IMG_SIZE / S
                cx = (j + x_center) * grid_size
                cy = (i + y_center) * grid_size
                bw = w * IMG_SIZE
                bh = h * IMG_SIZE

                xmin = cx - bw / 2
                ymin = cy - bh / 2
                xmax = cx + bw / 2
                ymax = cy + bh / 2

                detections.append({
                    "class": predicted_class,
                    "bbox": [xmin, ymin, xmax, ymax],
                    "score": objectness
                })
    return detections

def compute_ap(class_id, predictions, ground_truths, iou_threshold=0.5):
    tp = []
    scores = []
    gt = ground_truths[class_id]
    n_gt = len(gt)
    matched = [False] * n_gt

    for pred_box, score in sorted(predictions[class_id], key=lambda x: -x[1]):
        scores.append(score)
        ious = [compute_iou(pred_box, gt_box) for gt_box in gt]
        best_idx = np.argmax(ious) if ious else -1

        if ious and ious[best_idx] >= iou_threshold and not matched[best_idx]:
            tp.append(1)
            matched[best_idx] = True
        else:
            tp.append(0)

    if not tp:
        return 0.0

    tp = np.array(tp)
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / (n_gt + 1e-6)
    precisions = tp_cum / (tp_cum + fp_cum + 1e-6)
    return auc(recalls, precisions)

if __name__ == '__main__':
    _, val_dataset = stratified_split()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = SmallObjectDetector()
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    thresholds = np.linspace(0.1, 0.9, 9)
    best_map = 0.0
    best_threshold = 0.0
    mAP_scores = []

    for threshold in thresholds:
        gt_boxes = defaultdict(list)
        pred_boxes = defaultdict(list)

        with torch.no_grad():
            for images, bboxes, labels, _ in val_loader:
                images = images.to(next(model.parameters()).device)
                outputs = model(images)
                detections = postprocess(outputs, threshold=threshold)

                for box, label in zip(bboxes[0], labels[0]):
                    if label != -1:
                        gt_boxes[label.item()].append(box.tolist())

                for det in detections:
                    pred_boxes[det["class"]].append((det["bbox"], det["score"]))

        ap_per_class = []
        for class_id in range(C):
            ap = compute_ap(class_id, pred_boxes, gt_boxes)
            ap_per_class.append(ap)

        mean_ap = np.mean(ap_per_class)
        mAP_scores.append(mean_ap)
        print(f"Threshold={threshold:.2f} -> mAP={mean_ap:.4f}")

        if mean_ap > best_map:
            best_map = mean_ap
            best_threshold = threshold

    # Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, mAP_scores, marker='o')
    plt.xlabel("Objectness Threshold")
    plt.ylabel("mAP (IoU-based)")
    plt.title("True mAP vs Objectness Threshold")
    plt.grid(True)
    plt.show()

    print(f"✅ Best threshold: {best_threshold:.2f} | Best mAP: {best_map:.4f}")
