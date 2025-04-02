#%% 
import numpy as np
import torch
import torch.functional as F

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
