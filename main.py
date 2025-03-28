#%%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
rewards = []

for n in range(1,11):
    reward = np.random.rand()
    rewards.append(reward)
    Q = sum(rewards) / n
    print(Q)


#%%

Q = 0 

for n in range(1,11):
    rewardd = np.random.rand()
    Q = Q +(reward - Q) / n
    print(Q)

# %%
class Bandit:
    def __init__(self,arms = 10):
        self.rates = np.random.rand(arms)
    
    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0

bandit = Bandit()

for i in range(3):
    print(bandit.play(0))
    


# %%
bandit = Bandit()
Q = 0

for n in range(1,11):
    reward = bandit.play(0)
    Q += (reward - 0)/n
    print(Q)

# %%

bandit = Bandit()
Qs = np.zeros(10)
ns = np.zeros(10)

for n in range(10):
    action = np.random.randint(0,10)
    reward = bandit.play(action)

    ns[action] += 1
    Qs[action] += (reward - Qs[action]) / ns[action]
    print(Qs)

# %%
class Agent:
    def __init__(self, epsilon, action_size = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0,len(self.Qs))
        return np.argmax(self.Qs)
    
# %%
import matplotlib.pyplot as plt

steps = 1000
epsilon = 0.1

bandit = Bandit()
agent = Agent(epsilon)
total_reward = 0
total_rewards = []

rates = []

for step in range(steps):
    action = agent.get_action()
    reward = bandit.play(action)
    agent.update(action, reward)
    total_reward += reward
    total_rewards.append(total_reward)
    rates.append(total_reward/ (step+1))

print(total_rewards)

plt.xlabel('total rewards')
plt.ylabel('steps')
plt.plot(total_rewards)
plt.show()

plt.xlabel('average rewards')
plt.ylabel('steps')
plt.plot(rates)
plt.show()

# %%
runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    rates = []
    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step+1))
    all_rates[run] = rates

avg_rates = np.mean(all_rates, axis = 0)

plt.xlabel('average rewards')
plt.ylabel('steps')
plt.plot(avg_rates)
plt.show()

# %%
class AlphaAgent:
    def __init__(self, epsilon, alpha, actions = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action, reward):
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha
    
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)
# %%

runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit()
    agent = AlphaAgent(epsilon, 0.8)
    total_reward = 0
    rates = []
    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward
        rates.append(total_reward / (step+1))
    all_rates[run] = rates

avg_rates = np.mean(all_rates, axis = 0)

plt.xlabel('average rewards')
plt.ylabel('steps')
plt.plot(avg_rates)
plt.show()

# %%
V = 1
for i in range(1,100):
    V += -1 * (0.9 ** i)

print(V)

# %%
V = {'L1':0.0, 'L2':0.0}
new_V = V.copy()

for _ in range(100):
    new_V['L1'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])
    new_V['L2'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])
    V = new_V.copy()
    print(V)
# %%

V = {'L1':0.0, 'L2':0.0}
new_V = V.copy()

cnt = 0

while True:
    new_V['L1'] = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    new_V['L2'] = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])

    delta = abs(new_V['L1'] - V['L1'])
    delta = max(delta, abs(new_V['L2'] - V['L2']))

    V = new_V.copy()

    cnt += 1

    if delta < 0.0001:
        print(V)
        print(cnt)
        break

# %%
V = {'L1':0.0, 'L2':0.0}

cnt = 0

while True:
    t = 0.5 * (-1 + 0.9 * V['L1']) + 0.5 * (1 + 0.9 * V['L2'])
    delta = abs(t - V['L1'])
    V['L1'] = t

    t = 0.5 * (0 + 0.9 * V['L1']) + 0.5 * (-1 + 0.9 * V['L2'])
    delta = max(delta, abs(t - V['L2']))
    V['L2'] = t

    cnt += 1
    if delta < 0.0001:
        print(V)
        print(cnt)
        break
# %%
import common.gridworld_render as render_helper

class GridWorld:
    def __init__(self):
        self.action_space = [0,1,2,3]
        self.action_mieaning = {
            0:'UP',
            1:'RIGHT',
            2:'DOWN',
            3:'LEFT'
        }

        self.reward_map = np.array(
            [[0,0,0,1.0],
             [0,None,0,-1.0],
             [0,0,0,1.0]]
        )

        self.goal_state = (0,3)
        self.wall_state = (1,1)
        self.start_state = (2,0)
        self.agent_state = self.start_state
    
    @property
    def height(self):
        return len(self.reward_map)
    
    @property
    def width(self):
        return len(self.reward_map[0])
    
    @property
    def shape(self):
        return self.reward_map.shape
    
    def actions(self):
        return self.action_space
    
    def states(self):
        for h in range(self.height):
            for w in range(self.width):
                yield (h,w)
    
    def next_state(self, state, action):
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        next_state = (state[0] + move[0], state[1] + move[1])
        ny, nx = next_state

        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state
        
        return next_state
    
    def reward(self, state, action, next_state):
        return self.reward_map[next_state]
    
    def render_v(self, v=None, policy=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)

    def render_q(self, q=None, print_value=True):
        renderer = render_helper.Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_q(q, print_value)

#%%

env = GridWorld()

print(env.height)
print(env.width)
print(env.shape)

for action in env.actions():
    print(action)

print('===')

for state in env.states():
    print(state)

# %%
env = GridWorld()
env.render_v()
# %%
env = GridWorld()

V = {}

for state in env.states():
    V[state] = np.random.randn()
env.render_v(V)

# %%
from common.gridworld import GridWorld

env = GridWorld()
V = {}

for state in env.states():
    V[state] = 0

state = (1,2)
print(V[state])
# %%
from collections import defaultdict
from common.gridworld import GridWorld

env = GridWorld()
V = defaultdict(lambda: 0)

state = (1,2)
print(V[state])

pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

state = (0, 1)
print(pi[state])

# %%
pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

state = (0, 1)
print(pi[state])
# %%

def eval_onestep(pi, V, env, gamma=0.9):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue
        
        action_probs = pi[state]
        new_V = 0

        for action,action_prob in action_probs.items():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            new_V += action_prob * (r + gamma * V[next_state])
        
        V[state] = new_V
    return V

def policy_eval(pi, V, env, gamma=0.9, threshold=0.001):
    cnt = 0
    while cnt < 100:
        old_V = V.copy()
        V = eval_onestep(pi, V, env, gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t

        if delta < threshold:
            break

    return V

# %%
from common.gridworld import GridWorld
env = GridWorld()
gamma = 0.9

pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})

V = defaultdict(lambda: 0)
V = policy_eval(pi, V, env, gamma)
env.render_v(V, pi)

# %%
def argmax(d):
    max_value = max(d.values())
    max_key = 0
    for key, value in d.items():
        if value == max_value:
            max_key = key
            break
    return max_key

# %%
action_values = {0: 0.1, 1: -0.3, 2: 9.9, 3: -1.3}
max_action = argmax(action_values)
print(max_action)

# %%
def greedy_policy(V, env, gamma):
    pi = {}
    
    for state in env.states():
        action_values = {}

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values[action] = value
        
        max_action = argmax(action_values)
        action_probs = {0: 0, 1: 0, 2: 0, 3: 0}
        action_probs[max_action] = 1.0
        pi[state] = action_probs
    
    return pi
# %%
def policy_iter(env, gamma, threchold=0.001, is_render=False):
    pi = defaultdict(lambda: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    V = defaultdict(lambda: 0)

    while True:
        V = policy_eval(pi, V, env, gamma, threchold)
        new_pi = greedy_policy(V, env, gamma)
        
        if is_render:
            env.render_v(V, new_pi)

        if new_pi == pi:
            break

        pi = new_pi
    
    return pi

# %%
env = GridWorld()
gamma = 0.9
pi = policy_iter(env, gamma, is_render=True)

# %%
def value_iter_onestep(V, env, gamma):
    for state in env.states():
        if state == env.goal_state:
            V[state] = 0
            continue

        action_values = []

        for action in env.actions():
            next_state = env.next_state(state, action)
            r = env.reward(state, action, next_state)
            value = r + gamma * V[next_state]
            action_values.append(value)

        V[state] = max(action_values)
    
    return V
# %%
def value_iter(V, env, gamma, threshold=0.001, is_render=True):
    while True:
        if is_render:
            env.render_v(V)
        
        old_V = V.copy()
        V = value_iter_onestep(V, env,gamma)

        delta = 0
        for state in V.keys():
            t = abs(V[state] - old_V[state])
            if delta < t:
                delta = t
        
        if delta < threshold:
            break
    
    return V
# %%
from common.gridworld import GridWorld

class CustomGridWorld(GridWorld):
    def __init__(self):
        super().__init__()  # 親クラスの初期化を呼び出す
        # reward_mapをカスタマイズ
        self.reward_map = np.array([
            [0, -5.0, 0, 10.0],  # ゴールの報酬を10.0に変更
            [0, None, 0, -10.0],  # 罰則を-10.0に変更
            [0, 0, 0, -5.0]
        ])

V = defaultdict(lambda: 0)
env = CustomGridWorld()
gamma = 0.9

V = value_iter(V, env, gamma, is_render=True)
pi = greedy_policy(V, env, gamma)
env.render_v(V, pi)

