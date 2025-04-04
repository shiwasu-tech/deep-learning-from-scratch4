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
        self.l1 = nn.Linear(12 ,100)
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
            next_q = torch.max(next_qs, dim=1)[0].detach()
        
        target = reward + self.gamma * next_q
        qs = self.qnet(state)
        q_value = qs[0, action]
        loss = F.mse_loss(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# %%
from common.gridworld import GridWorld

env = GridWorld()
agent = QLearningAgent()

episodes = 1000
loss_history = []

for episode in range(episodes):
    state = env.reset()
    state = one_hot(state)
    total_loss, cnt = 0, 0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = one_hot(next_state)

        loss = agent.update(state, action, reward, next_state, done)
        total_loss += loss
        cnt += 1
        state = next_state
    
    average_loss = total_loss / cnt
    loss_history.append(average_loss)

# %%
import matplotlib.pyplot as plt
plt.plot(loss_history)
# %%
Q = {}
for state in env.states():
    for action in env.action_space:
        q = agent.qnet(one_hot(state))[:, action]
        Q[state, action] = float(q.data)
env.render_q(Q)
# %%
