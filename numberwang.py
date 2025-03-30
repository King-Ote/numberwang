import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from timeit import default_timer as timer

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

def set_device():
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

# SET UP DISCRIMINATOR --------------------------------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, outch, feats, kern, stri):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(outch, feats, kern),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feats, feats, kern, stride=stri),
            nn.BatchNorm2d(feats),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feats, outch, kern, stride=stri),
            nn.AvgPool2d(2), # Simpler than a linear layer to tidy weird size mismatches?
            nn.Sigmoid()
        )

    def forward(self, thingy):
        return self.net(thingy)

# SET UP GENERATOR --------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self, inch, outch, feats, kern, stri):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(inch, feats * 2, kern),
            nn.BatchNorm2d(feats * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feats * 2, outch, kern, stride=stri, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

def size_input(generator, target_dim=28):
    # Figure out input dimensions required to get target_dim sized output from generator
    generator_dim = target_dim
    for layer in reversed([l for l in generator.net]):
        if hasattr(layer, 'kernel_size'):
            generator_dim = 1 + (generator_dim + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) -
                           layer.output_padding[0] - 1) / layer.stride[0]

    print(f"Our input noise will be {generator_dim} by {generator_dim} pixels")
    if generator_dim == 0:
        print("NET TOO BIG! CANNOT MAKE SMALL ENOUGH IMAGE FROM EVEN SINGLE PIXEL INPUT TENSOR")
    return int(generator_dim)

# RESHAPE TRAINING DATA FROM CSV TO ??? ---------------------------------------------------------------
def get_training_data():
    train_data_raw = pd.read_csv("./mnist_train/mnist_train.csv")
    normalize_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = (train_data_raw.drop('5', axis=1)).to_numpy()
    train_data = train_data.reshape(59999, 28, 28)
    train_data = np.expand_dims(train_data, axis=1)
    print("Training data shape: ", train_data.shape)
    train_tensor_raw = torch.tensor(train_data, dtype=torch.float32)
    return normalize_transform(train_tensor_raw)

def show_image(image):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()


def run_train(dataloader, discriminator, generator, opt_d, opt_g, criterion, num_epochs=5):
    for epoch in range(num_epochs):
        print("Entering Epoch " + str(epoch))
        start_time = timer()
        for i, real_images in enumerate(dataloader):
            if i % 10 == 0:
                print(f"Working {i}th image")
                print("Wow that took ", timer() - start_time, " seconds!")
                start_time = timer()
            batch_size = real_images[0].size(0)
            print("Real images size: ", real_images[0].size())

            real_labels = torch.ones(batch_size, 1, 1, 1)
            fake_labels = torch.zeros(batch_size, 1, 1, 1)

            real_outputs = discriminator(real_images[0])
            opt_d.zero_grad()

            d_loss_real = criterion(real_outputs, real_labels)

            z = torch.randn(batch_size, in_channels, gen_dim, gen_dim)
            fake_images = generator(z)
            fake_outputs = discriminator(fake_images.detach())
            d_loss_fake = criterion(fake_outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()

            opt_g.zero_grad()

            fake_outputs = discriminator(fake_images)
            g_loss = criterion(fake_outputs, real_labels)

            g_loss.backward()
            opt_g.step()


if __name__ == "__main__":
    set_device() # Which GPU to use...

    # Parameters to seed network layers
    in_channels = 64
    out_channels = 1
    feature_maps = 14
    kernel_size = 6
    stride_size = 2

    gen = Generator(in_channels, out_channels, feature_maps, kernel_size, stride_size)
    disc = Discriminator(out_channels, feature_maps, kernel_size, stride_size)
    gen_dim = size_input(gen, 28) # Calc required input size to produce outputs of target_dim aka 28x28

    dataset = TensorDataset(get_training_data()) # Gets dataset - note this is NOT a dataloader! Hell it's a tuple

    gentest_noise_input_tensor = torch.rand(10, in_channels, gen_dim, gen_dim)
    gen_output_tensor = gen(gentest_noise_input_tensor)
    #print("Input size: ", torch.stack(dataset[0]).size())
    print("Gen output tensor shape:  ", gen_output_tensor.shape, " -- ([?, 1, 28, 28] expected)")
    print("Disc output tensor shape: ", disc(torch.stack(dataset[0])).size(), " -- ([1, 1, 1, 1] expected)")
    #show_image(gen_output_tensor[0][0].detach().numpy())


    # LOSS FUNCTION PARAMS
    lr = 0.0002
    beta1 = 0.5

    run_train(DataLoader(dataset, batch_size=5000, shuffle=True),
              disc, gen,
              optim.Adam(disc.parameters(), lr=lr, betas=(beta1, 0.999)),
              optim.Adam(gen.parameters(), lr=lr, betas=(beta1, 0.999)),
              nn.BCELoss(), 5)


    # TEST TRAINED GENERATOR --------------------------------------------------------------------
    # First, using same input as initial set-up white noise input
    output_tensor2 = gen(gentest_noise_input_tensor)
    output_image2 = output_tensor2[3][0].detach().numpy()
    print("output image shape: ", output_image2.shape)
    show_image(output_image2)

    input_tensor2 = torch.rand(10, in_channels, gen_dim, gen_dim)
    output_tensor3 = gen(input_tensor2)
    output_image3 = output_tensor3[3][0].detach().numpy()
    print("output image shape: ", output_image3.shape)
    show_image(output_image3)