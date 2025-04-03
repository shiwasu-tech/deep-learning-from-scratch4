#%% 
import numpy as np
import torch
import torch.functional as F
import torch.nn as nn

x = torch.tensor(5.0, requires_grad=True)
y = 3 * x ** 2
print(y)

y.backward()
print(x.grad)

# %%
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a, b = torch.tensor(a), torch.tensor(b)
c = torch.matmul(a, b)
print(c)
# %%
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
a, b = torch.tensor(a), torch.tensor(b)
c = torch.matmul(a, b)
print(c)

# %%
def rosenbrock(x0, x1):
    return (x0 - 1) ** 2 + 100 * (x1 - x0 ** 2) ** 2

x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad)
# %%
x0 = torch.tensor(np.array(0.0), requires_grad=True)
x1 = torch.tensor(np.array(2.0), requires_grad=True)

lr = 0.001
iters = 10000

for i in range(iters):
    print(x0, x1)
    y = rosenbrock(x0, x1)
    
    x0.grad = None
    x1.grad = None
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
    
print(x0, x1)

    
# %%
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 * x + np.random.randn(100, 1) * 0.1

import matplotlib.pyplot as plt
plt.scatter(x, y)

# %%
x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

W = torch.zeros(1, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def predict(x):
    return torch.matmul(x, W) + b

def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

lr = 0.1
iters = 1000

for i in range(iters):
    y_pred = predict(x)
    loss = mean_squared_error(y, y_pred)

    W.grad = None
    b.grad = None
    loss.backward()

    W.data -= lr * W.grad
    b.data -= lr * b.grad

    if i % 100 == 0:
        print(f'Iter {i}: W = {W.item()}, b = {b.item()}, loss = {loss.item()}')

plt.scatter(x, y)
plt.plot(x, y_pred.detach().numpy(), color='red')

print(W, b)
# %%
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

I, H, O = 1, 10, 1
W1 = torch.randn(I, H, requires_grad=True)
b1 = torch.zeros(H, requires_grad=True)
W2 = torch.randn(H, O, requires_grad=True)
b2 = torch.zeros(O, requires_grad=True)

def predict(x):
    h = torch.matmul(x, W1) + b1
    h = torch.sigmoid(h)
    y_pred = torch.matmul(h, W2) + b2
    return y_pred

x_tensor = torch.tensor(x, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

lr = 0.1
iters = 10000

for i in range(iters):
    y_pred = predict(x_tensor)
    loss = ((y_tensor - y_pred) ** 2).mean()  # Mean Squared Error

    # 勾配をリセット
    W1.grad = None
    b1.grad = None
    W2.grad = None
    b2.grad = None

    # 逆伝播
    loss.backward()

    # パラメータの更新
    with torch.no_grad():
        W1.data -= lr * W1.grad
        b1.data -= lr * b1.grad
        W2.data -= lr * W2.grad
        b2.data -= lr * b2.grad

    if i % 1000 == 0:
        print(f'Iter {i}: loss = {loss.item()}')

# 結果のプロット
plt.scatter(x, y)
plt.scatter(x, predict(x_tensor).detach().numpy(), color='red')
plt.show()

# %%
linear1 = nn.Linear(5,10)

batch_size, input_size = 100, 5
x = np.random.randn(batch_size, input_size)
x = torch.tensor(x, dtype=torch.float32)
y = linear1(x)

print(y.shape)
print(linear1.weight.shape, linear1.bias.shape)

for param in linear1.named_parameters():
    print(param[0], param[1].shape)
# %%
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lr = 0.2
iters = 10000

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.relu(self.linear1(x))
        y_pred = self.linear2(h)
        return y_pred

model = TwoLayerNet(1, 10, 1)

for i in range(iters):
    y_pred = model.forward(x)
    loss = ((y - y_pred) ** 2).mean()
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
            p.data -= lr * p.grad
    if i % 1000 == 0:
        print(f'Iter {i}: loss = {loss.item()}')
# %%
import torch.optim as optim
np.random.seed(0)
x = np.random.rand(100, 1)
y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

lr = 0.2
iters = 10000

class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = torch.relu(self.linear1(x))
        y_pred = self.linear2(h)
        return y_pred

model = TwoLayerNet(1, 10, 1)
optimizer = optim.Adagrad(model.parameters(), lr=lr)

for i in range(iters):
    y_pred = model.forward(x)
    loss = ((y - y_pred) ** 2).mean()
    model.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 1000 == 0:
        print(f'Iter {i}: loss = {loss.item()}')
# %%
