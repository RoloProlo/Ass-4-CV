import os
import glob
import torch
import xml.etree.ElementTree as ET
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dataset paths
IMG_DIR = "data/images"
ANNOTATION_DIR = "data/annotations"
IMG_SIZE = 112  # Target image size (112x112x3)


# Dataset class for loading images and annotations
class CatDogDataset(Dataset):
    def __init__(self, img_dir, ann_dir, transform=None):
        self.img_dir = img_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.img_files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
        self.label_map = {"cat": 0, "dog": 1}  # Label mapping

    def parse_annotation(self, ann_path):
        tree = ET.parse(ann_path)
        root = tree.getroot()
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)
        objects = []

        for obj in root.findall("object"):
            name = obj.find("name").text
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            label = self.label_map.get(name, -1)
            objects.append({"label": label, "bbox": [xmin, ymin, xmax, ymax]})

        return width, height, objects

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        ann_path = self.ann_files[idx]

        image = Image.open(img_path).convert("RGB")
        width, height, objects = self.parse_annotation(ann_path)

        # Calculate scaling factors
        scaler_x = width / IMG_SIZE
        scaler_y = height / IMG_SIZE

        # Transform bounding boxes
        bboxes = []
        for obj in objects:
            xmin = obj['bbox'][0] / scaler_x
            ymin = obj['bbox'][1] / scaler_y
            xmax = obj['bbox'][2] / scaler_x
            ymax = obj['bbox'][3] / scaler_y
            bboxes.append([xmin, ymin, xmax, ymax])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        labels = torch.tensor([obj["label"] for obj in objects], dtype=torch.int64)

        if self.transform:
            image = self.transform(image)

        return image, bboxes, labels, img_path


# Define transformation pipeline
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])


# Function to split dataset into stratified train/val sets
def split_dataset():
    print("Loading dataset...")
    dataset = CatDogDataset(img_dir=IMG_DIR, ann_dir=ANNOTATION_DIR, transform=transform)

    labels = [img_path.split("/")[-1][0] for img_path in dataset.img_files]  # Extract first letter (e.g., 'c' for cat)
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)), test_size=0.2, stratify=labels, random_state=42
    )

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Dataset split complete: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    return train_dataset, val_dataset


# Function to visualize sample images with bounding boxes
def visualize_samples(dataset, num_samples=4):
    dataloader = DataLoader(dataset, batch_size=num_samples, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    images, bboxes, labels, img_paths = next(iter(dataloader))

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    if len(images) == 1:
        axes = [axes]

    for i, (img, bbox, label, path) in enumerate(zip(images, bboxes, labels, img_paths)):
        img = img.permute(1, 2, 0).numpy()
        axes[i].imshow(img)

        for box, lbl in zip(bbox, label):
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='r', facecolor='none')
            axes[i].add_patch(rect)
            axes[i].text(xmin, ymin - 5, f'Label: {lbl.item()}', color='red', fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.5))
        axes[i].axis('off')
        axes[i].set_title(f"File: {os.path.basename(path)}")

    plt.show()


# Main execution
if __name__ == "__main__":
    train_dataset, val_dataset = split_dataset()

    print("Visualizing training samples...")
    visualize_samples(train_dataset)

    print("Visualizing validation samples...")
    visualize_samples(val_dataset)

    print("Data preparation complete. Ready for model training.")
