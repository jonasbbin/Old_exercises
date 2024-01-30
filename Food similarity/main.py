import os
import numpy as np
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from numpy.testing import assert_almost_equal
import pandas as pd
from PIL import Image
from torchvision import transforms
from torchvision import models
from tqdm import tqdm
import random
import pickle

#For outputs in developement
DEV = True

#helper function to get image name from integer number
def padding(x):
    newx = str(x)
    while len(newx) < 5:
        newx = '0' + newx
    newx = newx + '.jpg'
    return newx

    #add GPU as device 
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Assuming that we are on a CUDA machine, this should print a CUDA device:
if DEV:
    print('Device: ' + str(device))
    print('Start reading in data..')

#read in train data 
train_data = np.empty((59515,3), dtype=int)
with open('train_triplets.txt', 'r') as txt:
    count = 0
    for line in txt:
        first, second, third = line.split(' ')
        train_data[count] = [first, second, third]
        count += 1
    

#read in train data 
test_data = np.zeros((59544,3), dtype=int)
with open('test_triplets.txt', 'r') as txt:
    count = 0
    for line in txt:
        first, second, third = line.split(' ')
        test_data[count] = [first, second, third]
        count += 1

if DEV:
    print('Finished data reading..')


#define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#get pretrained resnet152 model 
model = models.resnet152(pretrained=True)    
model.eval()
model.to(device)

if DEV:
    print('Image preprocessing..')

image_database = torch.empty(10000,1,3,224,224)
counter = 0
for image in tqdm(os.listdir('food')):
    input_image = Image.open(os.path.join('food',image))
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)
    image_database[counter] = input_batch
    counter += 1

if DEV:
    print('Finished image preprocessing..')
    print('Starting feature vector calculation..')

image_database = image_database.to(device)
image_features = torch.empty(10000,1000)

counter = 0
for image in tqdm(image_database):
    with torch.no_grad():
        output = model(image)
        image_features [counter] = output[0]
        counter += 1

pickle.dump(image_features, open("image_features.p", "wb"))

image_features = pickle.load(open("image_features.p", "rb"))

if DEV:
    print('Finished feature vector calculation..')
    print('Generate input matrix and output vector..')

X_train = np.zeros((59515,3000))
y_train = np.zeros(59515)
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.FloatTensor(y_train).to(device)

for i in tqdm(range(59515)):
    #randomly decide to swap th images 2 and 3 for different output
    swap = bool(random.getrandbits(1))
    if swap:
        y_train[i] = 0
        X_train[i, 0:1000] = image_features[train_data[i,0]]
        X_train[i, 1000:2000] = image_features[train_data[i,2]]
        X_train[i, 2000:3000] = image_features[train_data[i,1]] 
    else:
        y_train[i] = 1
        X_train[i, 0:1000] = image_features[train_data[i,0]]
        X_train[i, 1000:2000] = image_features[train_data[i,1]]
        X_train[i, 2000:3000] = image_features[train_data[i,2]] 

if DEV:
    print('Finished generating input matrix and ouput vector..')
    print('Start training the NN..')

#define nn model        
class BinaryNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.h1 = nn.Linear(3000, 1000)
        self.h2 = nn.Linear(1000, 100)
        self.h3 = nn.Linear(750, 300)
        self.h4 = nn.Linear(300, 100)
        self.h5 = nn.Linear(100, 1)
        self.h6 = nn.Linear(25, 1)
        #self.relu = nn.LeakyReLU()
        self.relu = nn.ReLU()

    
    def forward(self, x):
        y = self.relu(self.h1(x))
        y = self.relu(self.h2(y))
        #y = self.h3(y)
        #y = self.h4(y)
        y = self.h5(y)
        #y = self.h6(y)
        return y

#send model to GPU
model = BinaryNet()
model = model.to(device)

#define number of epochs
#150 before
epochs = 91
running_loss = 0
#train the network
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
y_pred = torch.empty(59515)
y_train = torch.reshape(y_train,(59515,1))
for t in tqdm(range(epochs)):
    y_pred = model(X_train)
    loss = loss_fn(y_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if DEV:
        running_loss += loss.item()
        if t % 20 == 9:    # print every 1000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (t + 1, i + 1, running_loss / 50))
                running_loss = 0.0

if DEV:
    print('Finished training..')
    print('Preparing test data..')

X_test = np.zeros((59544,3000))
X_test = torch.FloatTensor(X_test).to(device)
for i in tqdm(range(59544)):
    X_test[i, 0:1000] = image_features[test_data[i,0]]
    X_test[i, 1000:2000] = image_features[test_data[i,1]]
    X_test[i, 2000:3000] = image_features[test_data[i,2]]  

if DEV:
    print('Finished preparing test data..') 
    print('Start prediction..')

#predict on test data using trained model 
y_test = model(X_test)
#apply sigmoid function to map values to [0,1]
y_test = torch.sigmoid(y_test)
#round to get only 0s and 1s
y_test = torch.round(y_test)
#get the final results 
y_test = y_test.cpu().detach().numpy()

if DEV:
    print('Finished predicting')
    print("Start writing ouput..")

output = y_test
#3 create beautiful output file 
lines = output
with open('output.txt', 'w') as f:
    for line in tqdm(lines):
        f.write(str(int(line)))
        f.write('\n')

if DEV:
    print('Finished') 