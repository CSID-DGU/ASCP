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
        self.device = torch.device("cuda:0")
        self.env = env
        self.gamma = gamma

        n_pairings = len(env.initial_pairing_set)
        max_flights = env.max_flights
        n_inputs = 1
        n_outputs = max_flights * n_pairings

        self.policy_net = PolicyNetwork(n_inputs, n_outputs).to(self.device)
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
        #print('saved_log_probs : ',self.saved_log_probs)
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

initial_pairing_set = pd.read_excel('/home/public/yunairline/ASCP/ReinforcementLearning/dataset/output.xlsx', header=1)
del initial_pairing_set['pairing index']
max_flight = len(initial_pairing_set.columns)
column = list(range(1, max_flight+1))
column_list = [str(item) for item in column]
initial_pairing_set.columns = column_list
initial_pairing_set = initial_pairing_set.fillna(-1)
pairing_matrix = initial_pairing_set.values
# print(pairing_matrix)

env = CrewPairingEnv(pairing_matrix)
agent = PolicyGradientAgent(env)

pairing_set = []
for i_episode in range(10):
    print('episode : ', i_episode)
    pairing_set, state = env.reset()
    done = False
    trycnt = 0
    while done == False:
        trycnt += 1
        print('trycnt : ', trycnt)
        if trycnt > 50:
            step_pairing_set = pairing_set
            break
        before_idx, after_idx = agent.select_action(state)
        step_pairing_set, reward, done, _ = env.step(before_idx, after_idx)
        print('done : ', done, 'reward : ', reward, end='\n\n')
        
    if trycnt <= 50:
        agent.rewards.append(reward)
        if np.any(pairing_set != step_pairing_set):
            print('!!!!!!!!!!! change pairing set !!!!!!!!!!!')
        pairing_set = step_pairing_set
        state = after_idx
        agent.update()

reshape_list = env.reshape_list(pairing_set.tolist(), num_rows=215, num_columns=4)

hard_score = 0
soft_score = 0
for pair in reshape_list:
    pair_hard_score, pair_soft_score = env.calculateScore(pair)
    hard_score += pair_hard_score
    soft_score += pair_soft_score

#print('hard score : ', hard_score, 'soft score : ', soft_score)

# 'pairing index'를 포함한 데이터프레임 생성
df = pd.DataFrame(reshape_list)

# 'pairing index' 열 추가
df.insert(0, 'pairing index', range(len(df)))

# -1 값을 빈 값으로 변경
df = df.replace(-1.0, '')

col_num = len(df.columns)
score_list = ['soft', soft_score, 'hard', hard_score]
while len(score_list) < col_num:
    score_list.extend([None])

# 맨 위 행에 score값 추가
df.loc[-1] = score_list
df.index = df.index + 1
df = df.sort_index()

# 칼럼명과 첫번째 행 바꾸기
tobe_col = df.iloc[0]
tobe_first = df.columns.tolist()
tobe_first = [np.nan if isinstance(x, (int, float)) else x for x in tobe_first]
print(tobe_first)

df.columns = tobe_col
df.loc[0] = tobe_first

# 엑셀 파일로 저장
df.to_excel('/home/public/yunairline/ASCP/ReinforcementLearning/output/output.xlsx', index=False)