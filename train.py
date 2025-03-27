import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import json

def collate_fn(batch):
    """
    Custom collate function to handle variable-length bounding boxes and labels in the DataLoader batch.
    Args:
        batch (list): List of tuples, where each tuple is (image, bboxes, labels, path)
    Returns:
        (images, bboxes, labels, paths): Tensors of images, padded bboxes, and labels
    """
    images, bboxes, labels, paths = zip(*batch)

    # Find the max number of bounding boxes in the batch
    max_bboxes = max([bbox.size(0) for bbox in bboxes])

    # Pad bounding boxes and labels to the maximum number of boxes in the batch
    padded_bboxes = []
    padded_labels = []
    for bbox, label in zip(bboxes, labels):
        pad_size = max_bboxes - bbox.size(0)

        # Pad bounding boxes with zeros
        padded_bbox = torch.cat([bbox, torch.zeros((pad_size, 4))], dim=0)
        padded_bboxes.append(padded_bbox)

        # Pad labels with -1 (or any ignored class index) to match shape
        padded_label = torch.cat([label, torch.full((pad_size,), -1, dtype=torch.int64)], dim=0)
        padded_labels.append(padded_label)

    # Stack images, labels, and the padded bounding boxes
    images = torch.stack(images, 0)
    bboxes = torch.stack(padded_bboxes, 0)
    labels = torch.stack(padded_labels, 0)

    return images, bboxes, labels, paths


# Define the loss components from YOLOv1
import torch.nn.functional as F

def yolo_loss(predictions, targets, S=7, B=1, C=2, lambda_coord=5, lambda_noobj=0.5):
    """
    YOLO-style loss function for model output of shape [batch, 343]
    targets: should be of shape [batch, S, S, 5 + C]
    """

    # ✅ Reshape predictions to [B, S, S, 5 + C]
    predictions = predictions.view(-1, S, S, 5 + C)


    # Split components
    pred_box = predictions[..., 0:4]  # x, y, w, h
    pred_obj = predictions[..., 4]    # objectness
    pred_cls = predictions[..., 5:]   # class scores

    # Targets
    target_box = targets[..., 0:4]
    target_obj = targets[..., 4]
    target_cls = targets[..., 5:]

    # Object mask: where there is an object in the cell
    obj_mask = target_obj > 0

    # Coordinate loss (only where there's an object)
    coord_loss = F.mse_loss(pred_box[obj_mask], target_box[obj_mask], reduction='sum')

    # Objectness loss
    obj_loss = F.mse_loss(pred_obj[obj_mask], target_obj[obj_mask], reduction='sum')
    noobj_loss = F.mse_loss(pred_obj[~obj_mask], target_obj[~obj_mask], reduction='sum')

    # Classification loss (only where there's an object)
    class_loss = F.binary_cross_entropy_with_logits(pred_cls[obj_mask], target_cls[obj_mask], reduction='sum')

    total_loss = lambda_coord * coord_loss + obj_loss + lambda_noobj * noobj_loss + class_loss
    return total_loss

def build_targets(bboxes, labels, S=7, C=2, image_size=112):
    """
    Converts bounding boxes and labels to YOLO target tensor of shape [batch, S, S, 5 + C]
    """
    batch_size = bboxes.shape[0]
    targets = torch.zeros((batch_size, S, S, 5 + C), dtype=torch.float32)

    for i in range(batch_size):
        for j in range(len(bboxes[i])):
            box = bboxes[i][j]
            label = labels[i][j]
            if label == -1:
                continue  # skip padding

            x_center = (box[0] + box[2]) / 2 / 112
            y_center = (box[1] + box[3]) / 2 / 112
            w = (box[2] - box[0]) / 112
            h = (box[3] - box[1]) / 112

            grid_x = int(x_center * S)
            grid_y = int(y_center * S)

            if grid_x >= S: grid_x = S - 1
            if grid_y >= S: grid_y = S - 1

            targets[i, grid_y, grid_x, 0:4] = torch.tensor([x_center, y_center, w, h])
            targets[i, grid_y, grid_x, 4] = 1.0  # objectness
            targets[i, grid_y, grid_x, 5 + label] = 1.0  # one-hot class

    return targets


def train_object_detector(model, train_dataset, val_dataset, num_epochs=30, batch_size=32, learning_rate=0.001, patience=5, save_path="best_model"):
    """
    Train an object detection model and save the best-performing model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float("inf")
    no_improve_epochs = 0

    # create variables to save data
    training_loss = []
    validation_loss = []

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for images, bboxes, labels, _ in train_loader:
            images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            targets = build_targets(bboxes, labels)
            targets = targets.to(device)
            loss = yolo_loss(predictions, targets)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)

        # Validation
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for images, bboxes, labels, _ in val_loader:
                images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)
                predictions = model(images)
                targets = build_targets(bboxes, labels)
                targets = targets.to(device)
                val_loss = yolo_loss(predictions, targets)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        training_loss.append(avg_train_loss)
        validation_loss.append(avg_val_loss)

        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), f'models/'+save_path+'.pth')  # Save model
            print(f"✅ Model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"⏹ Early stopping triggered at epoch {epoch + 1}")
            break

    with open(f'training_results/train_loss{save_path}.json', 'w') as f:
        json.dump(training_loss, f)

    with open(f'training_results/val_loss{save_path}.json', 'w') as f:
        json.dump(validation_loss, f)

    print("🎉 Training complete.")