import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from timeit import default_timer as timer

def get_training_data():
    train_data_raw = pd.read_csv("./mnist_train/mnist_train.csv")
    normalize_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = (train_data_raw.drop('5', axis=1)).to_numpy()
    train_data = train_data.reshape(59999, 28, 28)
    train_data = np.expand_dims(train_data, axis=1)
    print(train_data.shape)
    train_tensor_raw = torch.tensor(train_data, dtype=torch.float32)
    print(train_tensor_raw.shape)
    return normalize_transform(train_tensor_raw)

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

train_tensor = get_training_data()
dataset = TensorDataset(train_tensor)
dataloader = DataLoader(dataset, batch_size=5000, shuffle=True)


num_epochs = 1
for epoch in range(num_epochs):
    print("Entering Epoch " + str(epoch))
    starttime = timer()
    for i, real_images in enumerate(dataloader):
        if i % 10 == 0:
            print(f"Working {i}th image")
            print("Wow that took ", timer() - starttime, " seconds!")
            starttime = timer()
        batch_size = real_images[0].size(0)
        print(real_images[0].size())
        print(real_images[0][0].size())
        print(real_images[0][0][0].size())
        show_image(real_images[0][0][0])
        show_image(real_images[0][1][0])
