# Generic Libraries
import numpy as np
from numpy.lib.function_base import angle
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import GenericUnivariateSelect
from skimage.transform import rescale, resize, downscale_local_mean
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import random

from torch.nn import functional as F
import pickle
import gzip
import numpy as np
import os
from torchvision import transforms


from tqdm import trange

# import torch.utils.data as data
# Removes constant/zero features
from sklearn.feature_selection import VarianceThreshold

# Set GPU if present
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = "cpu"


# ------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------------

def load_zipped_pickle(filename):
    with gzip.open(filename, 'rb') as f:
        loaded_object = pickle.load(f)
        return loaded_object

def save_zipped_pickle(obj, filename):
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, 2)


def evaluate(predictions, targets):
    ious = []
    for p, t in zip(predictions, targets):
        prediction = np.array(p.cpu().detach())
        target = np.array(t.cpu().detach())
        assert target.shape == prediction.shape
        overlap = prediction * target
        union = prediction + target

        ious.append(overlap.sum() / float(union.sum()))

    return torch.median(torch.tensor(ious))

def custom_eval(predictions, targets):
    
    err = torch.square(torch.sub(predictions,targets))
    return torch.mean(err)

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.04):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size()) * self.std + self.mean).to(device)
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

#------------------------------------------------------------------------------------------------------------------

class MyDataset(Dataset):
    def __init__(self, X_images, Y_images, test = False):
        #TODO Push to device when ready
        self.X_images = X_images.to(device)
        self.Y_images = Y_images.to(device)
        self.test = test
        self.add_gaussian = AddGaussianNoise()

    def transform(self, x, y):

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        #print("Size Check 0: ", np.shape(x))
        angle = int(random.random() * 20)
        if random.random() > 0.5:
            angle = 360 - angle

        x = TF.rotate(img=x, angle=angle)
        y = TF.rotate(img=y, angle=angle)

        #print("Size Check 1: ", np.shape(x))
        # Random horizontal flipping
        if random.random() > 0.5:
            x = TF.hflip(x)
            y = TF.hflip(y)
        #print("Size Check 2: ", np.shape(x))
        #Add Gaussian noise to images
        x = self.add_gaussian(x)

        #Convert back as no longer needed
        y = y.squeeze(0)
        return x, y

    def __getitem__(self, index):
        x = self.X_images[index,:,:]
        y = self.Y_images[index,:,:]
        x, y = self.transform(x, y)
        return x, y

    def __len__(self):
        return self.X_images.size(0)

# Defines model
class model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            
            
            )
        self.pool_1 = nn.MaxPool2d(kernel_size=3,stride=2, padding=1, return_indices=True)
        self.conv_2 =  nn.Sequential(
            nn.Conv2d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            
            )
        self.pool_2 = nn.MaxPool2d(kernel_size=3,stride=2, padding=1, return_indices=True)
        self.conv_3 =  nn.Sequential(
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(p=0.1)
            
            )

        self.unpool1 =  nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)


        self.reconv_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
            nn.Dropout(p=0.1)
            
        )
        self.unpool2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

        self.reconv_3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3, padding=1),
        )
       

    def forward(self, x):
        # Down convolution with kernel of size 3 and max pooling
        #x = x.unsqueeze(1)
        #print(np.shape(x))
        #x_conv1 = (self.conv_1(x))
        #x_pool_1, indices_1 = self.pool_1(x_conv1)
        x_pool_1 = x
        #print(np.shape(x_pool_1))
        x_conv2 = (self.conv_2(x_pool_1))
        x_pool_2, indices_2 = self.pool_2(x_conv2)
        #print(np.shape(x_pool_2))
        x_conv3 = (self.conv_3(x_pool_2))
        #print(np.shape(x_conv3))
        x_upconv1 = x_conv3
        x_upconv1 = self.unpool1(x_upconv1, indices_2)

        x_upconv2 = (self.reconv_2(torch.add(x_upconv1,x_conv2)))
        y = x_upconv2
        #print(np.shape(x_upconv2))
        #x_upconv2 = self.unpool2(x_upconv2, indices_1)
        #y = (self.reconv_3(torch.add(x_upconv2,x_conv1)))
        #print(np.shape(y))
        
        y = y.squeeze(1)
        return y


# CONSTANTS
BATCHSIZE =  175
PRINTINTERVAL = 500
NUM_EPOCHS = 2000
LEARNINGRATE = 0.0001#0.0005
IMAGE_SIZE = (64,64)#(128, 128)
# load data
print("Reading in train_data")
train_data = load_zipped_pickle("train.pkl")



train_transform =transforms.Compose([transforms.ToTensor(), transforms.Resize(IMAGE_SIZE)])
sol_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize(IMAGE_SIZE)])

train_X = image_database = torch.empty(195,IMAGE_SIZE[0],IMAGE_SIZE[1])
train_y = image_database = torch.empty(195,IMAGE_SIZE[0],IMAGE_SIZE[1])
indexcounter = 0
for j in train_data:
    for index in j['frames']:
        image = j['video'][:, :, index]
        sol = j['label'][:, :, index]
        image = train_transform(image)
        sol = sol_transform(sol)

        train_X[indexcounter] = image
        train_y[indexcounter] = sol
        indexcounter += 1

# Define the ratio for train-test split
print("Number of Images found: ", indexcounter)
train_ratio = 0.9
num_samples = train_X.size(0)
num_train = int(train_ratio * num_samples)
num_test = num_samples - num_train

# Generate random indices for train and test sets
indices = torch.randperm(num_samples)

# Split the indices into train and test sets
train_indices = indices[:num_train]
test_indices = indices[num_train:]

# Use the indices to get the corresponding data and labels for train and test sets
X_train, y_train = train_X[train_indices], train_y[train_indices]
X_valid, y_valid =train_X[test_indices], train_y[test_indices]

#print(np.shape(X_train))
#print(np.shape(y_train))
#print(np.shape(X_valid))
#print(np.shape(y_valid))

train_dataset = MyDataset(X_train, y_train, test=False)
validation_dataset = MyDataset(X_valid, y_valid, test=True)

#drop last ignores sets that do not fill the batchsize
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=BATCHSIZE, shuffle=True, drop_last=False
)

valid_loader = torch.utils.data.DataLoader(
    dataset=validation_dataset, batch_size=BATCHSIZE, shuffle=False, drop_last=False
)


# Set TRANING MODE AND INITIALIZE MODEL
network = model().to(device)
network.train()

optimizer = torch.optim.Adam(network.parameters(), lr=LEARNINGRATE)
#loss_fn = nn.CrossEntropyLoss()


# TRAINING THE MODEL
progress_bar = trange(NUM_EPOCHS)


def validation(valid_loader):
    # Set to evaluationstate
    network.eval()
    # num_batches = len(valid_loader)
    
    val_loss = 0
    for batch_idx, (batch_x, batch_y) in enumerate(valid_loader):
        with torch.no_grad():
            # Needs to be done for convolution
            
            y_pred = network(batch_x)
            #if batch_idx == 0:
                
                #plt.imshow(y_pred.detach().cpu().numpy()[0])
                #plt.show()
                #plt.imshow(batch_y.detach().cpu().numpy()[0])
                #plt.show()
            #current_val_loss = custom_eval(y_pred, batch_y)
            current_val_loss = custom_eval(y_pred, batch_y)

            val_loss += current_val_loss
    return val_loss

print("Training model...")
# Initializing test loss for early stopping
best_model_state_dict = None
the_last_val_loss = float('inf')
current_val_loss = 100000
for epoch in progress_bar:
    num_batches = len(train_loader)
    training_loss = 0
    for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
        # batch_x are of shape (batch_size, 784), batch_y are of shape (batch_size,)
        network.train()
        # With this Gradients are guaranteed to be None for params that did not receive a gradient
        network.zero_grad()
        #plt.imshow(batch_x.squeeze(1).numpy()[0])
        #plt.show()
        
        # Predict batch
        y_pred = network(batch_x)
        # Compute Loss
        loss = custom_eval(y_pred, batch_y)

        training_loss += loss
        # With this Gradients are guaranteed to be None for params that did not receive a gradient
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # Update progress bar with accuracy occasionally
        if batch_idx % PRINTINTERVAL == 0:
            
            progress_bar.set_postfix(loss=loss.item(), )#acc=current_accuracy.item())
    #print("Current Training Loss for this Epoch:")
    #print(training_loss.cpu().detach().numpy())
    #print(training_loss)
    # Early stopping
    current_val_loss = validation(valid_loader)
    print("Validation loss: ", current_val_loss)
    if current_val_loss < the_last_val_loss:
        best_model_state_dict = network.state_dict()

    if epoch % 250 == 0:
        torch.save(network.state_dict(), f'best_model_checkpoint{epoch}.pth')
    if current_val_loss - the_last_val_loss > 0.2:
        print("Validation loss starts to increase, training should be stopped!")
        break
    else:
        the_last_val_loss = current_val_loss

torch.save(best_model_state_dict, 'best_model.pth')
#best_model = model().to(device)
#best_model.load_state_dict(torch.load('best_model.pth'))
