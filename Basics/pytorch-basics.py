import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
from torchsummary import summary

x = torch.tensor(1.,requires_grad=True)
w = torch.tensor(2., requires_grad=True)
b = torch.tensor(3., requires_grad=True)
y = w * x + b
y.backward() #Compute gradients
#grad=differenciation with respect to that variable
print(f'X grad: {x.grad}, W grad: {w.grad}, B grad: {b.grad}')

x = torch.randn(10,3)
y = torch.randn(10,2)
#a fully connected layer
linear = nn.Linear(3,2)
print(f'W: {linear.weight}')
print(f'B: {linear.bias}')
#Build loss func and optimizer
criterion = nn.MSELoss()
#Optimizer requires parameters to compute gradients
optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
#Forward pass
pred = linear(x)
#Compute loss
loss = criterion(pred, y)
print(f'Loss: {loss.item()}')
#Backward pass
loss.backward()
#Gradient values
print(f'dL/dw: {linear.weight.grad}')
print(f'dL/db: {linear.bias.grad}')
#1 step gradient descent
optimizer.step()
#Pred and loss after 1 training epock
pred = linear(x)
loss = criterion(pred, y)
print(f'Loss after 1 step optimization: {loss.item()}')

#numpy to tensor
x = torch.from_numpy(np.array([[1,2], [3,4]]))
print(type(x))
#tensor to numpy
x = torch.tensor([[1,2], [3,4]]).numpy()
print(type(x))

#Load Dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../data',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
image, label = train_dataset[0]
print(image.size())
print(label)
#Data loader provides ques and threads
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=64,shuffle=True)
#turn it into a iter object
data_iter = iter(train_loader)

#Get a mini-batch of img and label
images, labels = data_iter.next()

for images, labels in train_loader:
    #Training code
    pass

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self):
        # 1. Initialize file paths or a list of file names. 
        pass
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pass
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return 0 
custom_dataset = CustomDataset()
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=64, 
                                           shuffle=True)

#Download the architecture and weights of a pretrained model
resnet = torchvision.models.resnet18(pretrained=True)
for param in resnet.parameters():
    param.requires_grad = False #pretrained layers weights wont be updated
#print(summary(resnet,images.size()))
#Replace the top layer**
resnet.fc = nn.Linear(resnet.fc.in_features, 100)
#Only forward pass
images = torch.randn(64, 3, 224, 224)
outputs = resnet(images)
print (outputs.size()) 

# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))