import matplotlib.pyplot as plt
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

        # Decoder layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.relu = nn.ReLU()


    def forward(self, x):
        # Encoder
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))

        latent_space_representation = x
        # Decoder
        x = self.relu(self.deconv1(x))
        x = self.relu(self.deconv2(x))
        x = self.relu(self.deconv3(x))
        x = self.relu(self.deconv4(x))
        x = self.deconv5(x)
                
        return x, latent_space_representation

# create a model
model = ConvAutoencoder().to(device)
# load model checkpoint
model.load_state_dict(torch.load('python/model_big_10ep.pt')["model_state_dict"])

def calc_latent(img, vis=False):
    """
    Input: numpy array of image, R*C, unsigned char type (0-255)
    Output: numpy array of latent vector, float type, flatten to 1D
    """
    # get sample outputs
    # resize image to [1, 1, 256, 256]
    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
    # img = img.numpy()[0]
    img = img[np.newaxis, np.newaxis, :]
    # to tensor
    img = torch.from_numpy(img).type(torch.FloatTensor)
    
    output, latent = model(img.to(device))
    output = output.cpu().detach().numpy()
    if vis:
        fig = plt.figure()
        fig.add_subplot(2, 1, 1)  
        # # showing image
        plt.imshow(img.transpose((1,2,0)))
        plt.axis('off')
        plt.title("Original")
        
        # Adds a subplot at the 2nd position
        fig.add_subplot(2, 1, 2)
        
        # showing image
        plt.imshow(output.transpose((1,2,0)))
        plt.axis('off')
        plt.title("Reconstruction")
    return latent.flatten().cpu().detach().numpy()


