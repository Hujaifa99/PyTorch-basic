import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

input_size = 28 * 28
num_classes = 10
num_epochs = 5
batch_size = 100
lr = 0.001

train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)

model = nn.Linear(in_features=input_size, out_features=num_classes)
criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(model.parameters(), lr=lr)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.reshape(-1, input_size) #Reshape to batch_size X input_size
        #Forward pass
        output = model(images)
        loss = criterion(output, labels)
        #Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        if (i+1)%100==0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{total_step}], Loss {loss.item()}')

#In test no need to compute gradients
with torch.no_grad():
    correct, total = 0, 0
    for images, labels in test_loader:
        images = images.reshape(-1,input_size)
        output = model(images)
        _, pred = torch.max(output.data,1)
        total += labels.size(0)
        correct += (pred == labels).sum()
    print('Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))