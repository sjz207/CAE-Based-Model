import torch as t
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()
# print(net)

# for name, params in net.named_parameters():
#     print(name, ':', params.size())


input = Variable(t.rand(1, 1, 32, 32))
out = net(input)
# print(out.size())

net.zero_grad()
# out.backward(Variable(t.ones(1, 10)))
# print(net.conv1.bias.grad)

target = Variable(t.arange(0, 10))
criterion = nn.MSELoss()
loss = criterion(out, target)

# print(input)
print(out)
# print(target)
print(loss)

# print("before bp:")
# print(net.conv1.bias.grad)
# loss.backward()
#
# print("after bp:")
# print(net.conv1.bias.grad)

optimizer = optim.SGD(net.parameters(), lr=0.01)
optimizer.zero_grad()

loss.backward()
optimizer.step()


print('========================')
out = net(input)
loss = criterion(out, target)
print(out)
print(loss)
