import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchsummary import summary
import torchvision.transforms as transforms

#CPU or GPU to train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 28 * 28
num_classes = 10
hidden_size = 500
num_epochs = 5
batch_size = 100
lr = 0.001

train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#build model using class
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

train_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        #Flatten the image excluding batch size
        images = images.reshape(images.size(0), -1).to(device)
        #print(images.size())
        labels = labels.to(device)

        output = model(images)
        loss = criterion(output, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (i+1)%100==0:
            print(f'Epoch [{epoch}/{num_epochs}, Step [{i+1}/{train_step}], Loss: {loss.item()}]')

with torch.no_grad():
    correct, total = 0, 0
    for images, labels in test_loader:
        images = images.reshape(-1, input_size).to(device)
        labels = labels.to(device)
        output = model(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
