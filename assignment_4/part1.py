import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader

# you may not need the below functions, but just in case
from torchvision.transforms import (
    Compose,
    RandomHorizontalFlip,
    Grayscale,
    Resize,
    RandomCrop,
    ToTensor,
)

# device = torch.device('cpu')
fp16 = False
image_size = 28
torch.manual_seed(0)

# cant use os
def get_file_size(file_path):
    with open(file_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
    return size


# could have used torchvision.datasets.FashionMNIST but lets use the provided dataset to be sure
def load_images(file_path):
    num_images = (get_file_size(file_path) - 16) // (image_size * image_size)
    data = None
    with open(file_path, "rb") as f:
        f.read(16)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float16 if fp16 else np.float32)
        data = data.reshape(num_images, image_size, image_size, 1)
    if data is None:
        raise Exception(f"Error loading data from {file_path}")
    return data


def load_labels(file_path):
    with open(file_path, "rb") as f:
        f.read(8)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    if data is None:
        raise Exception(f"Error loading data from {file_path}")
    return data


# files are extracted using gunzip -d command first
train_images = load_images("train_images")
test_images = load_images("test_images")
train_labels = load_labels("train_labels")
test_labels = load_labels("test_labels")

def image_norm(image):
    return 

# normalize image (just to be sure because they look normalized already)
np.apply_along_axis(lambda image: (image - np.min(image)) / (np.max(image) - np.min(image)), 0, train_images)
np.apply_along_axis(lambda image: (image - np.min(image)) / (np.max(image) - np.min(image)), 0, test_images)

nb_classes = len(np.unique(train_labels))

from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.1, stratify=train_labels
)


# hyperparameters
batch_size = 16
epochs = 15
learning_rate = 1e-3

fc1_out_size = 256
fc2_out_size = 64
sgd_momentum = 0.9

# data augmentation
training_transform = Compose(
    [
        ToTensor(),
        RandomHorizontalFlip(),
    ]
)

test_transform = Compose([ToTensor()])


# transform our numpy array to a torch dataset to apply transforms easily
class FMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


training_dataset = FMNISTDataset(train_images, train_labels, training_transform)
test_dataset = FMNISTDataset(test_images, test_labels, test_transform)
val_dataset = FMNISTDataset(val_images, val_labels, test_transform)

# reshuffle data at each epoch
training_dataloader = torch.utils.data.DataLoader(
    training_dataset, batch_size=batch_size, shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=1, shuffle=True
)

# from table provided in the dataset repo
id2label = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

def visualize_images(samples):
    num_cols = 3
    num_rows = (len(samples) // 3) + 1
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))
    axes = axes.flatten()

    for i, (img, label) in enumerate(samples):
        axes[i].imshow(img.squeeze(), cmap="gray")
        axes[i].set_title(label)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

# visualize 9 random samples to verify label alignement
sample_indices = torch.randint(len(training_dataset), size=(9,))
samples = [(training_dataset[i][0], id2label[training_dataset[i][1]]) for i in sample_indices]
# visualize_images(samples)


# define our model
class FMNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # nb of units = (I-F+2P)/S+1
        # (28 - F + 2P)/S + 1
        # 1 equation, 3 variables: we will have to select 2 of them
        # taking F = 3 (required) and S = 1 (dont want to reduce resolution)
        # gives us: (28 - 3 + 2P) + 1
        # we want our spatial dimensions to remain the same i.e. 28 = 26 + 2P
        # P = 1
        # we also do this for the other 2 conv layers
        self.conv_10 = nn.Conv2d(
            in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1
        )

        # torch could infere the in_channels from previous layers but i added it for clarity
        self.conv_5 = nn.Conv2d(
            in_channels=10, out_channels=5, kernel_size=3, stride=1, padding=1
        )

        self.conv_16 = nn.Conv2d(
            in_channels=5, out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.pool = nn.MaxPool2d(2, 2)
        # after the 1st pooling layer the shape of the image is (5,14,14) 
        # because the (2,2) pooling "divide" the (28,28) image by 2
        # after the 2nd pooling layer the shape of the image is (16,7,7) 
        # because the (2,2) pooling "divide" the (14,14) image by 2
        # we flatten this input into a fully connected layer containing 16*7*7 units
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=fc1_out_size)
        self.fc2 = nn.Linear(in_features=fc1_out_size, out_features=fc2_out_size)

        # the last layer should output to the number of classes for multiclass classification
        self.fc3 = nn.Linear(in_features=fc2_out_size, out_features=nb_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv_10(x)))
        x = F.relu(self.conv_5(x))
        x = self.pool(F.relu(self.conv_16(x)))

        # start_dim=1 (see this as an index) to skip batch size (even if 1 in our context using SGD)
        # (batch_size, image_height, image_width, channels), flatten every dimension starting from image_height (included)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(epochs, learning_rate, sgd_momentum, training_dataloader, val_dataloader, net):
    if fp16:
        net.half()
    x_data = list(range(epochs))
    training_losses = [0]*epochs
    val_losses = [0]*epochs

    # momentum helps converging faster
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=sgd_momentum)
    loss = nn.CrossEntropyLoss()

    net.train()
    for epoch in range(epochs):
        epoch_training_loss = 0
        epoch_val_loss = 0
        for inputs, labels in training_dataloader:
            optimizer.zero_grad()
            output = net.forward(inputs)
            training_loss = loss(output, labels)
            training_loss.backward()
            optimizer.step()
            epoch_training_loss += training_loss.item()
            
        # same as training loss but does not perform backpropagation
        net.eval()
        with torch.no_grad():
            for inputs, labels in val_dataloader:
                output = net.forward(inputs)
                val_loss = loss(output, labels)
                epoch_val_loss += val_loss.item()

        epoch_training_loss /= len(training_dataloader)
        epoch_val_loss /= len(val_dataloader)
        print(f'{epoch + 1}: training loss: {epoch_training_loss:.3f}, validation loss: {epoch_val_loss:.3f}')
        training_losses[epoch] = epoch_training_loss
        val_losses[epoch] = epoch_val_loss
    
    plt.plot(x_data, training_losses, color='r', label='training loss')
    plt.plot(x_data, val_losses, color='g', label='validation loss')
    ax = plt.gca()
    ax.set_xlim([0, epoch])
    ax.set_ylim([0, 1])
    plt.xlabel("Epochs")
    plt.ylabel("CE Loss")
    plt.title("Training and validation loss during training process")
    plt.legend(loc='upper left')
    plt.show()
    
    return net

print('Start training...')
net = train_model(epochs, learning_rate, sgd_momentum, training_dataloader, val_dataloader, FMNISTCNN())
print('training completed...')


# test 1 instance
dataiter = iter(test_dataloader)
images, labels = next(dataiter)

# predictions
_, y_pred = torch.max(net(images), 1)

visualize_images(
    [
        (images[i], f"true: {id2label[labels[i].item()]}\npred: {id2label[y_pred[i].item()]}")
        for i in range(len(images))
    ]
)

def compute_accuracy(dataset, model, dataset_name=""):
    num_correct = 0
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for images, y_true in dataset:
            _, y_pred = torch.max(net(images), 1)
            num_correct += (y_true == y_pred).sum().item()
            num_samples += images.size(0)
    accuracy = 100 * num_correct / num_samples
    print(f'{dataset_name} Accuracy = {accuracy:.2f}%')

compute_accuracy(test_dataloader, net, "Test")