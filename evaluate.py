import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, average_precision_score
from models import SmallObjectDetector
from data_loader import stratified_split
from torch.utils.data import DataLoader
from train import collate_fn

S, C = 7, 2  # Grid size & number of classes


# Intersection over Union (IoU) Calculation
def iou(box1, box2):
    """ Compute IoU between two bounding boxes: [x, y, w, h] """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to corner coordinates
    x1_min, x1_max = x1 - w1 / 2, x1 + w1 / 2
    y1_min, y1_max = y1 - h1 / 2, y1 + h1 / 2
    x2_min, x2_max = x2 - w2 / 2, x2 + w2 / 2
    y2_min, y2_max = y2 - h2 / 2, y2 + h2 / 2

    # Compute intersection
    xi_min = max(x1_min, x2_min)
    yi_min = max(y1_min, y2_min)
    xi_max = min(x1_max, x2_max)
    yi_max = min(y1_max, y2_max)

    inter_area = max(0, xi_max - xi_min) * max(0, yi_max - yi_min)
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Compute IoU
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


# Postprocess predictions
def postprocess(pred, threshold=0.5):
    pred = pred.squeeze(0).detach().cpu().numpy().reshape((S, S, 5 + C))
    detections = []
    for i in range(S):
        for j in range(S):
            cell = pred[i, j]
            objectness = cell[4]
            if objectness > threshold:
                cls_scores = cell[5:]
                predicted_class = int(np.argmax(cls_scores))
                confidence = cls_scores[predicted_class] * objectness
                detections.append((predicted_class, confidence))  # Store class & confidence
    return detections


if __name__ == '__main__':
    # Load data and model
    _, val_dataset = stratified_split()
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    model = SmallObjectDetector()
    model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    thresholds = np.linspace(0, 1, 11)
    best_map = 0.0
    best_threshold = 0.0
    best_conf_matrix = None
    mAP_scores = []

    print("\n========== Evaluating mAP across thresholds ==========")
    for threshold in thresholds:
        y_true, y_scores = [], []

        with torch.no_grad():
            for images, bboxes, labels, _ in val_loader:
                images = images.to(next(model.parameters()).device)
                outputs = model(images)  # shape [1, 7, 7, 7]
                preds = postprocess(outputs, threshold=threshold)

                for k in range(len(labels[0])):
                    if labels[0][k] != -1:
                        y_true.append(labels[0][k].item())

                        # If a prediction exists, get the highest-confidence one
                        if preds:
                            best_pred = max(preds, key=lambda x: x[1])  # Highest confidence
                            y_scores.append(best_pred[1])  # Store confidence score
                        else:
                            y_scores.append(0)  # No prediction, assign 0 confidence

        # Compute mAP using sklearn's precision-recall function
        if len(y_true) > 0 and len(y_scores) > 0:
            try:
                ap = average_precision_score(y_true, y_scores)
                mAP_scores.append(ap)
                if ap > best_map:
                    best_map = ap
                    best_threshold = threshold
                    best_conf_matrix = confusion_matrix(y_true, [1 if score > 0.5 else 0 for score in y_scores],
                                                        labels=[0, 1])
            except:
                mAP_scores.append(0)
        else:
            mAP_scores.append(0)

    # Plot mAP curve
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, mAP_scores, marker='o')
    plt.xlabel("Objectness Threshold")
    plt.ylabel("mAP (mean Average Precision)")
    plt.title("mAP vs Objectness Threshold")
    plt.grid(True)
    plt.show()

    print(f"\n✅ Best threshold = {best_threshold:.2f} with mAP = {best_map:.4f}")

    if best_conf_matrix is not None:
        disp = ConfusionMatrixDisplay(best_conf_matrix, display_labels=["cat", "dog"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Confusion Matrix @ Best Threshold {best_threshold:.2f}")
        plt.show()
    else:
        print("No predictions above threshold for any setting.")

    print("\n✅ Evaluation complete")
