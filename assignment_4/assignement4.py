import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sklearn
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.utils.data import RandomSampler, DataLoader

compas_df = pd.read_csv("compas-scores.csv")
print(compas_df.tail(5))

print(compas_df.describe().round(2))

compas_df.dropna(how="any", inplace=True)

# select only columns containing textual data
car_categorial_df = compas_df.select_dtypes(exclude=["int64", "float64"])

# car_df[['DriveTrain','MSRP']].groupby('DriveTrain').describe().sort_values(by=('MSRP', 'mean'), ascending=False)
# car_df['Model'].value_counts()

from sklearn.preprocessing import MinMaxScaler, StandardScaler


def scale_dataset(df, scaler, col=None):
    col = col if col is not None else df.columns
    numeric_df = df[col].select_dtypes(include=["int64", "float64"])
    scaled_numeric_df = scaler.fit_transform(numeric_df)
    df[scaled_numeric_df.columns] = scaled_numeric_df
    return df


def standardize_dataset(df, col=None):
    st_scaler = StandardScaler().set_output(transform="pandas")
    return scale_dataset(df, st_scaler, col)


def normalize_dataset(df, col=None):
    mm_scaler = MinMaxScaler().set_output(transform="pandas")
    return scale_dataset(df, mm_scaler, col)


from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


def apply_ordinal(df, col, ranking):
    if col in df.columns:
        df[col] = df[col].map(ranking)
    return df


def one_hot_encoding(df, col=None):
    col = col if col is not None else df.columns
    categorial_df = df[col].select_dtypes(exclude=["int64", "float64"])
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).set_output(
        transform="pandas"
    )
    encoded_categorial_df = ohe.fit_transform(categorial_df)
    return pd.concat(
        [df.drop(columns=categorial_df.columns, axis=1), encoded_categorial_df], axis=1
    )


from sklearn.model_selection import train_test_split

hm = {"Low": 0, "Medium": 0, "High": 1}
compas_df["score"] = compas_df["score_text"].map(hm.get)

X_train, X_test, y_train, y_test = train_test_split(
    compas_df, test_size=0.2, stratify=compas_df["score"]
)

learning_rate = 1e-3
epochs = 5

input_dim = compas_df.shape[1]
output_dim = 1

device = torch.device('cpu')

class BinaryLogisticRegression(nn.Module):
    activation_function = nn.ReLU

    def __init__(self):
        super().__init__()
        self.layer_widths = [input_dim, 64, 32, 16, 8, output_dim]

        for i in range(len(self.layer_widths) - 1):
            self.layers.append(nn.Linear(i, i + 1))
            self.layers.append(nn.ReLU())

        self.layers = self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)

net = BinaryLogisticRegression().to(device)

loss = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

def train(epochs, net, training_dataset, loss, optimizer):
    for epoch in range(epochs):
        epoch_training_loss = 0
        epoch_val_loss = 0
        for inputs, labels in training_dataset:
            optimizer.zero_grad()
            output = net.forward(inputs)
            training_loss = loss(output, labels)
            training_loss.backward()
            optimizer.step()
            epoch_training_loss += training_loss.item()

train(epochs, net, training_dataset, loss, optimizer)

def compute_accuracy(dataset, model, dataset_name=""):
    num_correct = 0
    num_samples = 0
    
    model.eval()
    with torch.no_grad():
        for samples, y_true in dataset:
            _, y_pred = torch.max(net(samples), 1)
            num_correct += (y_true == y_pred).sum().item()
            num_samples += samples.size(0)
    accuracy = 100 * num_correct / num_samples
    print(f'{dataset_name} Accuracy = {accuracy:.2f}%')

compute_accuracy(test_dataloader, net, "Test")