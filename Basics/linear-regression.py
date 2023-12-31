import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

input_size = 1
output_size = 1
num_epochs = 60
lr =0.001

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

linear = nn.Linear(input_size, output_size)
criterion = nn.MSELoss()
optim = torch.optim.SGD(linear.parameters(), lr)

for epoch in range(num_epochs):
    input = torch.from_numpy(x_train)
    target = torch.from_numpy(y_train)

    pred = linear(input)
    loss = criterion(pred, target)

    optim.zero_grad()
    loss.backward()
    optim.step()

    if (epoch+1)%5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

prediction = linear(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original Data')
plt.plot(x_train, prediction, label='Fitted Line')
plt.legend()
plt.show()

