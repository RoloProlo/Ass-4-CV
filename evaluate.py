import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score
from models import SmallObjectDetector
from data_loader import stratified_split
from torch.utils.data import DataLoader
from train import collate_fn

S, C = 7, 2

# Postprocess predictions

def postprocess(pred, threshold=0.5):
    pred = pred.squeeze(0).detach().cpu().numpy()  # shape: [7, 7, 7]
    detections = []
    for i in range(S):
        for j in range(S):
            cell = pred[i, j]
            objectness = cell[4]
            if objectness > threshold:
                cls_scores = cell[5:]
                predicted_class = int(np.argmax(cls_scores))
                detections.append(predicted_class)
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
        y_true, y_pred = [], []

        with torch.no_grad():
            for images, bboxes, labels, _ in val_loader:
                images = images.to(next(model.parameters()).device)
                outputs = model(images)  # shape [1, 7, 7, 7]
                preds = postprocess(outputs, threshold=threshold)

                for k in range(len(labels[0])):
                    if labels[0][k] != -1:
                        y_true.extend([labels[0][k].item()] * len(preds))
                        y_pred.extend(preds)

        if len(y_true) > 0:
            try:
                prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
                mAP_scores.append(prec)
                if prec > best_map:
                    best_map = prec
                    best_threshold = threshold
                    best_conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
            except:
                mAP_scores.append(0)
        else:
            mAP_scores.append(0)

    # Plot mAP curve
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, mAP_scores, marker='o')
    plt.xlabel("Objectness Threshold")
    plt.ylabel("mAP (macro avg precision)")
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
