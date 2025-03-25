import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os

# Dit is een comment voor saus

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
def yolo_loss(predictions, bboxes, labels, lambda_coord=5, lambda_noobj=0.5):
    """
    YOLOv1 Loss Function: calculates the total loss for object detection.

    Args:
        predictions (tensor): Predicted values from the model, shape [batch_size, S, S, B * 5 + C]
        bboxes (tensor): Ground truth bounding boxes, shape [batch_size, num_boxes, 4] (x, y, w, h)
        labels (tensor): Ground truth class labels, shape [batch_size, num_boxes]
        lambda_coord (float): Weight for localization loss.
        lambda_noobj (float): Weight for confidence loss (for no-object cells).

    Returns:
        total_loss (tensor): Total loss computed for the batch.
    """
    # Initialize loss components
    coord_loss = 0.0
    conf_loss = 0.0
    class_loss = 0.0

    # Extract the number of grid cells (S) and number of boxes per cell (B)
    B = 2  # Number of bounding boxes per grid cell, this could be 2, 5, etc. depending on your configuration
    S = 7  # The number of grid cells in both height and width (usually 7 for YOLOv1)

    # Reshape predictions to match the expected format: [batch_size, S, S, B * 5 + C]
    pred_boxes = predictions[..., :4]  # Bounding box predictions (x, y, w, h)
    pred_conf = predictions[..., 4:5]  # Confidence scores
    pred_class = predictions[..., 5:]  # Class probabilities

    # Iterate through each sample in the batch
    for i in range(predictions.size(0)):  # Loop through batch size
        # For each sample, compare each ground truth bounding box with predicted boxes
        for j in range(bboxes.size(1)):  # Loop through ground truth boxes per sample
            # Extract true bounding box and class label for the current box
            true_bbox = bboxes[i, j]  # [4] (x, y, w, h)
            true_label = labels[i, j]  # Class label for the current bounding box

            # Localization Loss (x, y, w, h)
            coord_loss += lambda_coord * torch.sum(torch.abs(pred_boxes[i] - true_bbox))

            # Confidence Loss (objectness loss)
            conf_loss += torch.sum(torch.abs(pred_conf[i] - true_label))

            # Classification Loss
            class_loss += torch.sum(torch.abs(pred_class[i] - true_label))

    # Total loss (sum of all components)
    total_loss = coord_loss + conf_loss + class_loss
    return total_loss


def train_object_detector(model, train_dataset, val_dataset, num_epochs=30, batch_size=32, learning_rate=0.001, patience=5, save_path="best_model.pth"):
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

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for images, bboxes, labels, _ in train_loader:
            images, bboxes, labels = images.to(device), bboxes.to(device), labels.to(device)

            optimizer.zero_grad()
            predictions = model(images)
            loss = yolo_loss(predictions, bboxes, labels)
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
                val_loss = yolo_loss(predictions, bboxes, labels)
                running_val_loss += val_loss.item()

        avg_val_loss = running_val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save the model if validation loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve_epochs = 0
            torch.save(model.state_dict(), save_path)  # Save model
            print(f"✅ Model saved to {save_path} (Val Loss: {best_val_loss:.4f})")
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= patience:
            print(f"⏹ Early stopping triggered at epoch {epoch + 1}")
            break

    print("🎉 Training complete.")