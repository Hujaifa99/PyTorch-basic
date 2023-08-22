import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

device = torch.device('cude' if torch.cuda.is_available() else 'cpu')

num_epochs=5
num_classes=10
batch_size=100
lr=0.001

train_dataset = torchvision.datasets.MNIST(root='../../data', train=True, download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.MNIST(root='../../data', train=False, transform=transforms.ToTensor())
train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(test_dataset, batch_size, False)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Linear(7*7*32, num_classes)
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.size())
        #Flattening the image
        out = out.reshape(out.size(0), -1)
        #print(out.size())
        out = self.fc(out)
        return out

model = ConvNet(num_classes).to(device=device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=lr)

train_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = model(images)
        loss = criterion(output, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if (i+1)%100==0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{train_step}], Loss: {loss.item()}')

model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

