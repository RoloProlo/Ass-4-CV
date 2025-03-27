import torch
from torch.utils.data import DataLoader
from data_loader import stratified_split
from train import collate_fn
from models import SmallObjectDetector
from train import yolo_loss, build_targets
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
epochs = 30
batch_size = 32
lr = 0.001
patience = 5
S, C = 7, 2

# Setup
train_dataset, val_dataset = stratified_split()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SmallObjectDetector().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
best_val_loss = float("inf")

print("\n========== Training ==========")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    for images, bboxes, labels, _ in train_loader:
        images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)
        targets = build_targets(bboxes, labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = yolo_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}")

print("\n========== Evaluation ==========")
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for images, bboxes, labels, _ in val_loader:
        images = images.to(device)
        outputs = model(images)[0]  # shape [7,7,7]
        outputs = outputs.detach().cpu().numpy()

        for i in range(S):
            for j in range(S):
                cell = outputs[i, j]
                obj = cell[4]
                if obj > 0.5:
                    pred_cls = np.argmax(cell[5:])
                    for k in range(len(labels[0])):
                        if labels[0][k] != -1:
                            y_true.append(labels[0][k].item())
                            y_pred.append(pred_cls)

if len(y_true) > 0:
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=["cat", "dog"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - After Training")
    plt.show()
else:
    print("No confident predictions in validation set.")

print("\n✅ Training and evaluation complete")
