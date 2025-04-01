import json
import matplotlib.pyplot as plt


def plot_loss_graph(train_json, val_json, loss_name):
    """
    Plots train and validation loss per epoch for a given loss component.

    Parameters:
    - train_json (str): Path to the JSON file containing train loss data.
    - val_json (str): Path to the JSON file containing validation loss data.
    - loss_name (str): Name of the loss component (for labeling the graph).
    """
    # Load the JSON data
    with open(train_json, 'r') as f:
        train_data = json.load(f)

    with open(val_json, 'r') as f:
        val_data = json.load(f)

    epochs = list(range(1, len(train_data) + 1))  # Assuming losses are stored as lists
    train_loss = train_data
    val_loss = val_data

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Train Loss', marker='o')
    plt.plot(epochs, val_loss, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for {loss_name}')
    plt.legend()
    plt.grid(True)
    plt.show()

# Example usage:
plot_loss_graph("training_results/train_class_lossbest_model.json", "training_results/val_class_lossbest_model.json", "Classification loss")
plot_loss_graph("training_results/train_coord_lossbest_model.json", "training_results/val_coord_lossbest_model.json", "Localization loss")
plot_loss_graph("training_results/train_noobj_lossbest_model.json", "training_results/val_noobj_lossbest_model.json", "No-Object loss")
plot_loss_graph("training_results/train_obj_lossbest_model.json", "training_results/val_obj_lossbest_model.json", "Objectness loss")
plot_loss_graph("training_results/train_total_lossbest_model.json", "training_results/val_total_lossbest_model.json", "Total loss")