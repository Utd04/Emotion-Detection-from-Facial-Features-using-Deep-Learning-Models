
import torch
import pandas as pd
import numpy as np

from skimage.io import imread, imsave
from skimage.transform import rotate,warp
from skimage.util import random_noise

from skimage import data, io
from skimage import data, exposure

import sys
import csv

data_train = pd.read_csv(sys.argv[1],header= None)
pvt_data = pd.read_csv(sys.argv[2],header= None)
output_dir = sys.argv[3]

X_train = data_train.iloc[:,1:].values
X_train = X_train.astype('float32')
y_train = data_train.iloc[:,0].values.astype('int64')

X_pvt = pvt_data.iloc[:,1:].values
X_pvt= X_pvt.astype('float32')
X_pvt = X_pvt.reshape(X_pvt.shape[0],1,48,48)

#comment this line
# X_train = X_train[0:50]
# y_train = y_train[0:50]
# X_pvt = X_pvt[0:10]

torch_X_pvt = torch.from_numpy(X_pvt).type(torch.FloatTensor)

def training(train_loader, model, num_epochs, optimizer, loss_fn):
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()
    for e in range(num_epochs):
        running_loss = 0
        for x,y in train_loader:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # print('Epoch:',e,'Loss:',running_loss/len(train_loader))
    if torch.cuda.is_available():
        model = model.cpu()
        loss_fn = loss_fn.cpu()


# print(X_train.shape)
# print(y_train.shape)

X_train = X_train.reshape(X_train.shape[0],1,48,48)
final_train_data = []
final_target_train = []
for i in range(X_train.shape[0]):
    final_train_data.append(X_train[i])
    final_train_data.append(rotate(X_train[i], angle=45, mode = 'wrap'))
    final_train_data.append(np.fliplr(X_train[i]))
    final_train_data.append(np.flipud(X_train[i]))
    final_train_data.append(random_noise(X_train[i],var=0.2**2))
    for j in range(5):
        final_target_train.append(y_train[i])

X_train = np.array(final_train_data)
y_train = np.array(final_target_train)

X_train = X_train.reshape(X_train.shape[0],1,48,48)
torch_x_train = torch.from_numpy(X_train).type(torch.FloatTensor)
torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
train_loader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = True)

model = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(32),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=0.25),
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(64),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=0.25),
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(128),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=0.25),
                torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.ReLU(inplace=True),
                torch.nn.BatchNorm2d(128),
                torch.nn.MaxPool2d(kernel_size=2, stride=2),
                torch.nn.Dropout(p=0.25),
                torch.nn.Flatten(start_dim=1),
                torch.nn.Linear(1152, 512),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(),
                torch.nn.Linear(512, 256),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(),
                torch.nn.Linear(256,7),
                
        )

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = torch.nn.CrossEntropyLoss()

training(train_loader, model, 100, optimizer, criterion)

y_pred = None
y1 = None
rows = None
fields = ['Id', 'Prediction'] 
with open(output_dir, 'w') as csvfile:  
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(fields)

i = 0
while i<torch_X_pvt.shape[0] :
    y_pred = model(torch_X_pvt[i:i+1])
    i+=1
    y_pred = torch.argmax(y_pred,dim=1)
    y_pred = y_pred.detach().numpy()
    y_pred = y_pred.reshape((1,1))
    y1 = np.array([i])
    y1 = y1.reshape((1,1))
    rows = np.concatenate((y1,y_pred),axis=0).T
    with open(output_dir, 'at') as csvfile:  
        csvwriter=csv.writer(csvfile)  
        csvwriter.writerows(rows)