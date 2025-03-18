import os
import glob
import torch
import xml.etree.ElementTree as ET
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Dataset paths
IMG_DIR = "data/images"
ANNOTATION_DIR = "data/annotations"
IMG_SIZE = 112  # Target image size (112x112x3)


# Dataset class for loading images and annotations
class CatDogDataset:
    def __init__(self, img_files, ann_files, transform=None):
        self.img_files = img_files
        self.ann_files = ann_files
        self.transform = transform
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


# Function to manually split dataset with stratification
def stratified_split():
    print("Loading dataset...")

    # Load all image and annotation file paths
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, "*.png")))
    ann_files = sorted(glob.glob(os.path.join(ANNOTATION_DIR, "*.xml")))

    # Separate images into categories
    cat_files = [(img, ann) for img, ann in zip(img_files, ann_files) if "cat" in img.lower()]
    dog_files = [(img, ann) for img, ann in zip(img_files, ann_files) if "dog" in img.lower()]

    # Shuffle each category
    np.random.seed(42)  # Set seed for reproducibility
    np.random.shuffle(cat_files)
    np.random.shuffle(dog_files)

    # Compute 80/20 split index
    cat_split = int(len(cat_files) * 0.8)
    dog_split = int(len(dog_files) * 0.8)

    # Split into train and validation sets
    train_data = cat_files[:cat_split] + dog_files[:dog_split]
    val_data = cat_files[cat_split:] + dog_files[dog_split:]

    # Shuffle final datasets again (to mix cats and dogs)
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)

    # Extract separate lists for images and annotations
    train_img_files, train_ann_files = zip(*train_data)
    val_img_files, val_ann_files = zip(*val_data)

    # Create dataset instances
    train_dataset = CatDogDataset(list(train_img_files), list(train_ann_files), transform=transform)
    val_dataset = CatDogDataset(list(val_img_files), list(val_ann_files), transform=transform)

    print(f"Stratified split complete: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    return train_dataset, val_dataset


# Function to visualize sample images with bounding boxes
def visualize_samples(dataset, num_samples=4):
    images, bboxes, labels, img_paths = [], [], [], []

    for i in range(num_samples):
        img, bbox, label, path = dataset[i]
        images.append(img)
        bboxes.append(bbox)
        labels.append(label)
        img_paths.append(path)

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
    train_dataset, val_dataset = stratified_split()

    print("Visualizing training samples...")
    visualize_samples(train_dataset)

    print("Visualizing validation samples...")
    visualize_samples(val_dataset)

    print("Data preparation complete. Ready for model training.")
