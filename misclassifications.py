import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, ConfusionMatrixDisplay

from evaluate import postprocess
from models import SmallObjectDetector, CHOICE1
from data_loader import stratified_split
from torch.utils.data import DataLoader
from train import collate_fn
from collections import defaultdict

S, C = 7, 2
IMG_SIZE = 112
MISCLASSIFIED_DIR = "misclassified_images"

if not os.path.exists(MISCLASSIFIED_DIR):
    os.makedirs(MISCLASSIFIED_DIR)


def save_misclassified_images(image, true_box, true_label, pred_box, pred_label, index):
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())

    # Draw ground truth box
    gt_rect = Rectangle((true_box[0], true_box[1]), true_box[2] - true_box[0], true_box[3] - true_box[1],
                        linewidth=2, edgecolor='g', facecolor='none', label=f"GT: {true_label}")
    ax.add_patch(gt_rect)

    # Draw predicted box
    pred_rect = Rectangle((pred_box[0], pred_box[1]), pred_box[2] - pred_box[0], pred_box[3] - pred_box[1],
                          linewidth=2, edgecolor='r', facecolor='none', label=f"Pred: {pred_label}")
    ax.add_patch(pred_rect)

    plt.legend()
    plt.axis("off")
    filepath = os.path.join(MISCLASSIFIED_DIR, f"misclassified_{index}.png")
    plt.savefig(filepath)
    plt.close()


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


# Main script
if __name__ == '__main__':
    _, val_dataset = stratified_split()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = SmallObjectDetector()
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    misclassified_images = []

    with torch.no_grad():
        for idx, (images, bboxes, labels, _) in enumerate(val_loader):
            images = images.to(next(model.parameters()).device)
            outputs = model(images)
            detections = postprocess(outputs, threshold=0.5)

            gt_labels_img = [label.item() for label in labels[0] if label != -1]
            gt_boxes_img = [box.tolist() for box in bboxes[0] if box is not None]

            if detections and gt_labels_img:
                pred_box = detections[0]["bbox"]
                pred_label = detections[0]["class"]
                true_box = gt_boxes_img[0]
                true_label = gt_labels_img[0]

                if true_label != pred_label:
                    save_misclassified_images(images[0], true_box, true_label, pred_box, pred_label, idx)
                    misclassified_images.append(images[0])

    # Display misclassified images
    for img in misclassified_images:
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.show()
