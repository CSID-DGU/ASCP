import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from CrewPairingEnv import CrewPairingEnv
import pandas as pd


#Policy를 파라미터화 하기 위한 신경망. 하이퍼파라미터 튜닝 작업 필요
#input: current state(all flight-> pairing * max flight 형태로 제공)
#output: probability of each action
class PolicyNetwork(nn.Module): 
    def __init__(self, n_inputs, n_outputs):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(n_inputs, 256),
            nn.ReLU(),
            nn.Linear(256, n_outputs),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.network(x)

class PolicyGradientAgent:
    def __init__(self, env, lr=0.01, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env = env
        self.gamma = gamma

        n_pairings = len(env.initial_pairing_set)
        max_flights = env.max_flights
        n_inputs = 1
        n_outputs = max_flights * n_pairings

        self.policy_net = PolicyNetwork(n_inputs, n_outputs)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) #using Adam for optimizing

        self.saved_log_probs = []
        self.rewards = []
        self.saved_actions = []

    def select_action(self, state):
        print('state :', state ,end=' ')
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # state를 Pytorch 텐서로 변환, 배치 차원 추가 후 GPU로 이동
        action_probs = self.policy_net(state)  # 신경망에서 각 action의 확률 가져옴
        action_probs = action_probs.squeeze().cpu().detach().numpy()  # 배치 제거, 텐서 cpu로 이동, 계산 그래프 분리, numpy로 변환
        
        # 이미 선택한 action을 제외한 확률 계산
        available_actions = [i for i in range(len(action_probs)) if i not in self.saved_actions]
        if sum(action_probs[available_actions]) == 0:
            return state, np.random.choice(available_actions)
        else:
            adjusted_action_probs = action_probs[available_actions] / sum(action_probs[available_actions])
            selected_action = np.random.choice(available_actions, p=adjusted_action_probs)  # 주어진 확률을 바탕으로 무작위 action 선택
        
        
        self.saved_actions.append(selected_action)  # 선택 action 저장
        
        print('\tselected_action :', selected_action, end='\t')
        self.saved_log_probs.append(torch.log(torch.tensor(action_probs[selected_action])))  # 선택 action 로그 확률 저장
        print('saved_log_probs : ',self.saved_log_probs)
        return state, selected_action

    def update(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, torch.tensor(R, requires_grad=True))
        returns = torch.tensor(returns, requires_grad=True)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = sum(policy_loss)
        policy_loss.backward()
        self.optimizer.step()
        del self.rewards[:]
        del self.saved_log_probs[:]

# Usage
initial_pairing_set = pd.read_csv('/home/public/yunairline/dataset/pairingdata/output-data.csv')

cols_to_check = ['1', '2', '3', '4', '5']
initial_pairing_set.dropna(subset=cols_to_check, how='all', inplace=True)
initial_pairing_set.reset_index(drop=True, inplace=True)

two_dim_list = initial_pairing_set.values.tolist()

cleaned_data = [[item for item in row if isinstance(item, str)] for row in two_dim_list]

def extract_integers(lst):
    result = []
    for item in lst:
        integers = [int(element[1:]) for element in item if element.startswith('F')]
        result.append(integers)
    return result

new_data = extract_integers(cleaned_data)

# 결과 출력
state_list = extract_integers(cleaned_data)
# Calculate the number of pairings and maximum flights
n_pairings = len(state_list)
max_flights = max(len(pairing) for pairing in state_list if pairing)

# Initialize the state matrix with -1 (indicating no flight)
state_matrix = np.full((n_pairings, max_flights), -1, dtype=int)

# Fill in the state matrix with flight indices
for i, pairing in enumerate(state_list):
    state_matrix[i, :len(pairing)] = pairing

initial_pairing_set = state_matrix
env = CrewPairingEnv(initial_pairing_set)
agent = PolicyGradientAgent(env)


for i_episode in range(1000):
    state = env.reset()
    for t in range(100):  
        done = False
        while done == False:
            before_idx, after_idx = agent.select_action(state)
            step_pairing_set, reward, done, _ = env.step(before_idx, after_idx)
            print('done : ', done, 'reward : ', reward, end='\n\n')
        agent.rewards.append(reward)
        pairing_set = step_pairing_set
        state = after_idx
    agent.update()