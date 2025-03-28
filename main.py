from data_loader import stratified_split
from train import train_object_detector
from models import SmallObjectDetector, CHOICE1


# Example usage:
# Define your model, dataset, and other parameters




def train_model(model, dataset, epochs, batchsize, learnrate, patience):
    train_dataset, val_dataset = dataset

    train_object_detector(model, train_dataset, val_dataset, num_epochs=epochs, batch_size=batchsize, learning_rate=learnrate, patience=patience)





if __name__ == '__main__':
    dataset = stratified_split()
    #model = SmallObjectDetector()
    model = CHOICE1
    epochs = 30
    batchsize = 32
    learnrate = 0.001
    patience = 5

    train_model(model, dataset, epochs, batchsize, learnrate, patience)




