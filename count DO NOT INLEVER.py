import os

# Directories for error types
ERROR_DIRS = {
    "misclassified_cat_to_dog": "misclassified_cat_to_dog",
    "misclassified_dog_to_cat": "misclassified_dog_to_cat",
    "false_positives": "false_positives",
    "false_negatives": "false_negatives",
    "poor_localization": "poor_localization",
}

def count_images_in_folders():
    """Counts the number of images in each error folder."""
    counts = {}
    for error_type, directory in ERROR_DIRS.items():
        if os.path.exists(directory):
            counts[error_type] = len([f for f in os.listdir(directory) if f.endswith(".png")])
        else:
            counts[error_type] = 0
    return counts

if __name__ == "__main__":
    error_counts = count_images_in_folders()
    for error_type, count in error_counts.items():
        print(f"{error_type}: {count} images")