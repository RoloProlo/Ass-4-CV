import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from models import SmallObjectDetector
from data_loader import stratified_split
from torch.utils.data import DataLoader
from train import collate_fn

# Dummy IoU function — replace with real version
def iou(box1, box2):
    """Computes IoU between two boxes (x, y, w, h)"""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1_min, x1_max = x1 - w1/2, x1 + w1/2
    y1_min, y1_max = y1 - h1/2, y1 + h1/2
    x2_min, x2_max = x2 - w2/2, x2 + w2/2
    y2_min, y2_max = y2 - h2/2, y2 + h2/2

    inter_xmin = max(x1_min, x2_min)
    inter_ymin = max(y1_min, y2_min)
    inter_xmax = min(x1_max, x2_max)
    inter_ymax = min(y1_max, y2_max)

    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0

def postprocess(pred, threshold=0.5, S=7, B=1, C=2):
    """
    Reshape and parse raw model output into bounding boxes and predicted classes.
    Expects pred shape [343] → reshaped to [7, 7, 7] = [S, S, C + 5]
    """
    # assert pred.numel() == S * S * (C + 5 * B), f"Unexpected pred shape: {pred.shape}"

    pred = pred.squeeze(0).detach().cpu().numpy()  # assuming pred is shape [1, 7, 7, 7]
    detections = []

    for i in range(S):
        for j in range(S):
            cell = pred[i, j]
            x, y, w, h = cell[:4]
            objectness = cell[4]
            cls_scores = cell[5:]

            if objectness > threshold:
                predicted_class = int(np.argmax(cls_scores))
                detections.append((predicted_class, objectness, [x, y, w, h]))

    # print(f"Detected {len(detections)} objects at threshold {threshold}")
    return detections


def evaluate_model(model, val_loader, thresholds=np.linspace(0.0, 1.0, 11), iou_thresh=0.1):
    all_maps = []
    best_conf_matrix = None
    best_threshold = 0
    best_map = -1

    for t in thresholds:
        y_true = []
        y_pred = []
        tp, fp, fn = 0, 0, 0

        for images, bboxes, labels, _ in val_loader:
            images = images.to(device)
            preds = model(images)

            for i in range(len(images)):
                detections = postprocess(preds[i].unsqueeze(0), threshold=t)

                gt_boxes = bboxes[i]
                gt_labels = labels[i]

                matched = set()
                for det_cls, _, det_box in detections:
                    found_match = False
                    for j, gt_box in enumerate(gt_boxes):
                        if j in matched or gt_labels[j].item() == -1:
                            continue

                        # POTENTIAL FIX?
                        x1, y1, x2, y2 = gt_box.cpu().numpy()
                        gt_cx = (x1 + x2) / 2
                        gt_cy = (y1 + y2) / 2
                        gt_w = x2 - x1
                        gt_h = y2 - y1
                        gt_box_converted = [gt_cx, gt_cy, gt_w, gt_h]


                        if iou(det_box, gt_box_converted) > iou_thresh:
                            y_pred.append(det_cls)
                            y_true.append(gt_labels[j].item())
                            matched.add(j)
                            found_match = True
                            break
                    if not found_match:
                        fp += 1
                fn += len(gt_boxes) - len(matched)

        if len(y_pred) > 0:
            conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(2)))
            precision = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=0) + 1e-6)
            recall = np.diag(conf_matrix) / (np.sum(conf_matrix, axis=1) + 1e-6)
            ap = np.mean(precision)  # crude approximation
        else:
            conf_matrix = None
            ap = 0.0

        all_maps.append(ap)
        if ap > best_map:
            best_map = ap
            best_conf_matrix = conf_matrix
            best_threshold = t

    return all_maps, thresholds, best_threshold, best_map, best_conf_matrix

if __name__ == '__main__':
    # Load data and model
    _, val_dataset = stratified_split()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = SmallObjectDetector()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("🔍 Evaluating model...")
    all_maps, thresholds, best_threshold, best_map, best_conf_matrix = evaluate_model(model, val_loader)

    # Plot mAP vs threshold
    plt.plot(thresholds, all_maps, marker='o')
    plt.title("mAP vs Objectness Threshold")
    plt.xlabel("Objectness Threshold")
    plt.ylabel("mAP")
    plt.grid(True)
    plt.show()

    print(f"✅ Best Threshold: {best_threshold} → mAP: {best_map:.4f}")

    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=best_conf_matrix, display_labels=["Class 0", "Class 1"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix @ Threshold {best_threshold}")
    plt.show()
