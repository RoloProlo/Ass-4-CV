import torch
from data_loader import stratified_split
from torch.utils.data import DataLoader
from train import collate_fn


def count_classes():
    """Counts the number of dog and cat images in the dataset efficiently."""

    # Load dataset (only labels are needed)
    train_dataset, val_dataset = stratified_split()

    # Define DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

    def get_counts(dataloader):
        cat_count, dog_count = 0, 0
        for _, _, labels, _ in dataloader:  # Skip images, only use labels
            print(dog_count, cat_count)
            for label_batch in labels:
                for label in label_batch:
                    if label == 0:
                        cat_count += 1
                    elif label == 1:
                        dog_count += 1
        return cat_count, dog_count

    train_cats, train_dogs = get_counts(train_loader)
    val_cats, val_dogs = get_counts(val_loader)

    print(f"Training Set: Cats = {train_cats}, Dogs = {train_dogs}")
    print(f"Validation Set: Cats = {val_cats}, Dogs = {val_dogs}")


if __name__ == "__main__":
    count_classes()
