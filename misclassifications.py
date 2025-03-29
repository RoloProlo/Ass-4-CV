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

# Directories for error types
ERROR_DIRS = {
    "misclassified_cat_to_dog": "misclassified_cat_to_dog",
    "misclassified_dog_to_cat": "misclassified_dog_to_cat",
    "false_positives": "false_positives",
    "false_negatives": "false_negatives",
}

# Create directories if they don't exist
for directory in ERROR_DIRS.values():
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_image(image, true_box, true_label, pred_box, pred_label, index, error_type):
    """Save images with bounding boxes in respective error folders."""
    fig, ax = plt.subplots(1)
    ax.imshow(image.permute(1, 2, 0).cpu().numpy())

    # Draw ground truth box (if available)
    if true_box is not None:
        gt_rect = Rectangle(
            (true_box[0], true_box[1]),
            true_box[2] - true_box[0],
            true_box[3] - true_box[1],
            linewidth=2,
            edgecolor='g',
            facecolor='none',
            label=f"GT: {true_label}",
        )
        ax.add_patch(gt_rect)

    # Draw predicted box (if available)
    if pred_box is not None:
        pred_rect = Rectangle(
            (pred_box[0], pred_box[1]),
            pred_box[2] - pred_box[0],
            pred_box[3] - pred_box[1],
            linewidth=2,
            edgecolor='r',
            facecolor='none',
            label=f"Pred: {pred_label}",
        )
        ax.add_patch(pred_rect)

    plt.legend()
    plt.axis("off")

    # Save image in respective error directory
    filepath = os.path.join(ERROR_DIRS[error_type], f"{error_type}_{index}.png")
    plt.savefig(filepath)
    plt.close()


# Main script
if __name__ == "__main__":
    _, val_dataset = stratified_split()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = SmallObjectDetector()
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device("cpu")))
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    misclassified_images = []

    with torch.no_grad():
        for idx, (images, bboxes, labels, _) in enumerate(val_loader):
            images = images.to(next(model.parameters()).device)
            outputs = model(images)
            detections = postprocess(outputs, threshold=0.5)

            # Ground truth
            gt_labels_img = [label.item() for label in labels[0] if label != -1]
            gt_boxes_img = [box.tolist() for box in bboxes[0] if box is not None]

            # Predicted
            if detections:
                pred_box = detections[0]["bbox"]
                pred_label = detections[0]["class"]
            else:
                pred_box, pred_label = None, None

            # Case 1: False Negative (GT exists but no detection)
            if gt_labels_img and not detections:
                save_image(images[0], gt_boxes_img[0], gt_labels_img[0], None, None, idx, "false_negatives")
                continue  # Move to the next image

            # Case 2: False Positive (Detection exists but no GT)
            if not gt_labels_img and detections:
                save_image(images[0], None, None, pred_box, pred_label, idx, "false_positives")
                continue  # Move to the next image

            # Case 3: Misclassification (GT exists, Prediction exists, but wrong label)
            if gt_labels_img and detections:
                true_box, true_label = gt_boxes_img[0], gt_labels_img[0]

                # Misclassified cat as dog
                if true_label == 0 and pred_label == 1:
                    save_image(images[0], true_box, "Cat", pred_box, "Dog", idx, "misclassified_cat_to_dog")
                    misclassified_images.append(images[0])

                # Misclassified dog as cat
                elif true_label == 1 and pred_label == 0:
                    save_image(images[0], true_box, "Dog", pred_box, "Cat", idx, "misclassified_dog_to_cat")
                    misclassified_images.append(images[0])

    # Display misclassified images
    for img in misclassified_images:
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())
        plt.axis("off")
        plt.show()
