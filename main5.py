# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def one_hot(state):
    HEIGHT, WIDTH = 3, 4
    vec = torch.zeros((HEIGHT * WIDTH))
    y, x = state
    idx = WIDTH * y + x
    vec[idx] = 1.0
    return vec.unsqueeze(0)

state = (2,0)
x = one_hot(state)

print(x.shape)
print(x)

# %%
from collections import defaultdict

Q = defaultdict(lambda: 0)
state = (2,0)
action = 0

print(Q[state, action])
# %%
from collections import defaultdict

Q = defaultdict(lambda: 0)
state = (2,0)
action = 0

print(Q[state, action])

#%%
class QNet(nn.Module):
    def __init__(self):
        super(QNet,self).__init__()
        self.l1 = nn.Linear(1 ,100)
        self.l2 = nn.Linear(100, 4)

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        x = self.l2(x)
        return x
    
# %%
class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr = 0.01
        self.epsilon = 0.1
        self.action_size = 4

        self.qnet = QNet()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=self.lr)
        
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            q_values = self.qnet(state)
            return torch.argmax(q_values).item()
    
    def update(self, state, action, reward, next_state, done):
        q_values = self.qnet(state)
        q_value = q_values[0, action]

        if done:
            next_q = torch.tensor([0.0])
        else:
            next_qs = self.qnet(next_state)
            next_q = torch.max(next_qs, dim=1).detach()
        
        target = reward + self.gamma * next_q

        loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# %%
