
import torch
import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np

import sys
import csv


data_train = pd.read_csv(sys.argv[2],header= None)
data_test = pd.read_csv(sys.argv[3],header= None)

X_train = data_train.iloc[:,1:].values
X_train= X_train.astype('float32')
X_test = data_test.iloc[:,1:].values
X_test= X_test.astype('float32')
y_train = data_train.iloc[:,0].values.astype('int64')

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
        # if e%10==0:
        # print('Epoch:',e,'Loss:',running_loss/len(train_loader))
    if torch.cuda.is_available():
        model.cpu()
        loss_fn = loss_fn.cuda()

question = sys.argv[1]
question = int(question)
output_dir = sys.argv[4]

#Comment this line
# X_train = X_train[0:100]
# y_train = y_train[0:100]
# X_test = X_test[0:25]

#Part A

if question == 1:
    model_nn = torch.nn.Sequential(
                torch.nn.Linear(48*48,100),
                torch.nn.ReLU(),
                torch.nn.Linear(100,7),
    )
    optimizer_nn = torch.optim.SGD(model_nn.parameters(), lr=0.005, weight_decay=0.0001)
    loss_fn_nn = torch.nn.CrossEntropyLoss()

    X_train = X_train.reshape(X_train.shape[0],48*48)
    X_test = X_test.reshape(X_test.shape[0],48*48)

    torch_x_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_x_test = torch.from_numpy(X_test).type(torch.FloatTensor)

    train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = True)

    training(train_loader, model_nn, 200, optimizer_nn, loss_fn_nn)

    fields = ['Id', 'Prediction'] 
    with open(output_dir, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)
        
    i = 0
    while i<torch_x_test.shape[0] :
        y_pred = model_nn(torch_x_test[i:i+1])
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

#Part B

if question==2:
    from skimage.filters import gabor_kernel
    from skimage.filters import gabor
    from skimage import data, io
    from skimage.feature import hog
    from skimage import data, exposure
    
    X_train = (X_train.reshape(X_train.shape[0],48,48))
    X_train_filtered = []
    y_train_filtered = []
    for (x,y) in zip(X_train,y_train):
        filt_real , _ = gabor(x, frequency=0.9, theta = -np.pi/2)
        # _ , filt_real = hog(filt_real, orientations=16, pixels_per_cell=(3, 3),cells_per_block=(2, 2), visualize=True, multichannel=False)
        X_train_filtered.append(filt_real.tolist())
        y_train_filtered.append(y)
        filt_real, _ = gabor(x, frequency=0.9, theta = 0)
        # _ , filt_real = hog(filt_real, orientations=16, pixels_per_cell=(3, 3),cells_per_block=(2, 2), visualize=True, multichannel=False)
        X_train_filtered.append(filt_real.tolist())
        y_train_filtered.append(y)
        filt_real, _ = gabor(x, frequency=0.9, theta = np.pi/2)
        # _ , filt_real = hog(filt_real, orientations=16, pixels_per_cell=(3, 3),cells_per_block=(2, 2), visualize=True, multichannel=False)
        X_train_filtered.append(filt_real.tolist())
        y_train_filtered.append(y)
    X_train = np.array(X_train_filtered)
    y_train = np.array(y_train_filtered)

    model_nn = torch.nn.Sequential(
                torch.nn.Linear(48*48,100),
                torch.nn.ReLU(),
                torch.nn.Linear(100,7),
    )

    optimizer_nn = torch.optim.SGD(model_nn.parameters(), lr=0.005, weight_decay=0.0001)
    loss_fn_nn = torch.nn.CrossEntropyLoss()

    X_train = X_train.reshape(X_train.shape[0],48*48)
    X_test = X_test.reshape(X_test.shape[0],48,48)
    X_test_filtered = []
    for x in X_test:
        filt_real , _ = gabor(x, frequency=0.9, theta = -np.pi/2)
        # _ , filt_real = hog(filt_real, orientations=16, pixels_per_cell=(3, 3),cells_per_block=(2, 2), visualize=True, multichannel=False)
        X_test_filtered.append(filt_real.tolist())
    X_test = np.array(X_test_filtered)
    X_test = np.array(X_test)
    X_test = X_test.reshape(X_test.shape[0],48*48)

    torch_x_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_x_test = torch.from_numpy(X_test).type(torch.FloatTensor)

    train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = True)

    training(train_loader, model_nn, 200, optimizer_nn, loss_fn_nn)

    fields = ['Id', 'Prediction'] 
    with open(output_dir, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)
        
    i = 0
    while i<torch_x_test.shape[0] :
        y_pred = model_nn(torch_x_test[i:i+1])
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

#Part C

if question == 3:
    model_cnn = torch.nn.Sequential(
                    torch.nn.Conv2d(kernel_size=(3,3), stride=3, padding=0, in_channels=1, out_channels=64),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(64),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Conv2d(kernel_size=(2,2), stride=2, padding=0, in_channels=64, out_channels=128),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm2d(128),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                    torch.nn.Flatten(start_dim=1),
                    torch.nn.Linear(512,256),
                    torch.nn.ReLU(),
                    torch.nn.BatchNorm1d(256),
                    torch.nn.Linear(256,7)
            )
    
    optimizer_cnn = torch.optim.SGD(model_cnn.parameters(), lr=0.005, weight_decay=1e-4)
    loss_fn_cnn = torch.nn.CrossEntropyLoss()
    X_train = X_train.reshape(X_train.shape[0],1,48,48)
    X_test = X_test.reshape(X_test.shape[0],1,48,48)

    torch_x_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    torch_y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    torch_x_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    train = torch.utils.data.TensorDataset(torch_x_train,torch_y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size = 32, shuffle = True)

    training(train_loader, model_cnn, 200, optimizer_cnn, loss_fn_cnn)
        
    y1 = []
    for i in range(1,X_test.shape[0]+1):
        y1.append(i)
    y1 = np.array(y1)
    y1 = y1.reshape((1,X_test.shape[0]))
    y_pred = model_cnn(torch_x_test)
    y_pred = torch.argmax(y_pred,dim=1)
    y_pred = y_pred.reshape((1,X_test.shape[0]))

    fields = ['Id', 'Prediction'] 
    with open(output_dir, 'w') as csvfile:  
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)

    rows = np.concatenate((y1,y_pred),axis=0).T
    with open(output_dir, 'at') as csvfile:  
        csvwriter=csv.writer(csvfile)  
        csvwriter.writerows(rows)

