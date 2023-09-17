import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from CrewPairingEnv import CrewPairingEnv
import pandas as pd
import random


# Policy를 파라미터화 하기 위한 신경망. 하이퍼파라미터 튜닝 작업 필요
# input: current state(all flight-> pairing * max flight 형태로 제공)
# output: probability of each action
class PolicyNetwork(nn.Module): # nn.Module은 PyTorch에서 신경망 모델을 정의하고 관리하기 위한 기본 클래스
    # 강화 학습 에이전트의 정책(policy)을 신경망으로 모델링한 클래스
    # 주어진 상태를 입력으로 받아서 각 가능한 행동에 대한 확률을 출력
    def __init__(self, n_inputs, n_outputs): # input: 입력 뉴런 수, 출력 뉴런 수(가능한 모든 행동의 수)
        super(PolicyNetwork, self).__init__() # 상위 클래스인 nn.Module의 생성자를 호출하여 초기화
        self.network = nn.Sequential( # 정책 신경망에 사용할 신경망 구조를 정의, 여기서는 nn.Sequential을 사용하여 여러 개의 레이어를 순차적으로 연결하고 있다.
            nn.Linear(n_inputs, 256), # 입력으로 256개의 뉴런을 가지는 fully connected (dense) 레이어를 생성
            nn.ReLU(), # 활성화 함수 ReLU는 비선형성을 추가하여 신경망이 복잡한 함수를 모델링할 수 있도록 돕는다.
            nn.Linear(256, n_outputs), # 출력으로 256개의 뉴런을 가지는 fully connected (dense) 레이어를 생성 
            nn.Softmax(dim=-1)) # Softmax 함수를 사용하여 출력값을 확률 형태로 변환

    def forward(self, x): # 주어진 입력 x를 정책 신경망의 레이어를 통과시켜서 각 행동에 대한 확률을 출력
        return self.network(x)

# 실제 강화 학습 에이전트를 정의하는 클래스
# 에이전트는 상태를 입력으로 받아서 행동을 선택하고, 선택한 행동의 로그 확률을 저장
class PolicyGradientAgent:
    def __init__(self, env, lr=0.01, gamma=0.99): # 입력: 환경, 학습률, 감가율
        self.device = torch.device("cuda:0") # 학습에 사용할 장치를 GPU로 설정
        self.env = env
        self.gamma = gamma

        n_pairings = len(env.initial_pairing_set) # 초기 페어링 셋의 개수
        max_flights = env.max_flights # 환경에서 최대 비행 횟수
        n_inputs = 1
        n_outputs = max_flights * n_pairings

        self.policy_net = PolicyNetwork(n_inputs, n_outputs).to(self.device) # 정책 신경망(PolicyNetwork)을 생성
        
        # using Adam for optimizing (Adam 옵티마이저를 생성)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 로그 확률, 보상, 선택한 행동을 저장하는 리스트들을 초기화
        self.saved_log_probs = []
        self.rewards = []
        self.saved_actions = []

    def select_action(self, state): # 주어진 상태에서 행동을 선택하고, 선택한 행동의 로그 확률을 저장, state: 현재 상태
        print('state :', state, end=' ')
        state = np.array(state) # 주어진 상태를 numpy 배열로 변환
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)  # state를 Pytorch 텐서로 변환, 배치 차원 추가 후 GPU로 이동
        action_probs = self.policy_net(state)  # 신경망에서 각 action의 확률 가져옴
        # 배치 제거, 텐서 cpu로 이동, 계산 그래프 분리, numpy로 변환
        action_probs = action_probs.squeeze().cpu().detach().numpy()

        # 이미 선택한 action을 제외한 확률 계산
        available_actions = [i for i in range(len(action_probs)) if i not in self.saved_actions]

        # available_actions가 비어있으면 종료
        if not available_actions:
            return state, -1

        # sum(available_actions) = 1이 되도록 확률 조정
        # 만약 available_actions가 비어있으면 모든 action에 대해 1/len(action_probs)로 조정
        if sum(action_probs[available_actions]) == 0:
            adjusted_action_probs = np.ones(len(action_probs[available_actions])) / len(action_probs[available_actions])
        else:
            adjusted_action_probs = action_probs[available_actions] / (sum(action_probs[available_actions]))
        
        selected_action = np.random.choice(available_actions, p=adjusted_action_probs)  # 주어진 확률을 바탕으로 무작위 action 선택

        self.saved_actions.append(selected_action)  # 선택 action 저장

        print('\tselected_action :', selected_action)
        self.saved_log_probs.append(torch.log(torch.tensor(action_probs[selected_action])))  # 선택 action 로그 확률 저장
        # print('saved_log_probs : ',self.saved_log_probs)

        # episode가 끝나면 selected_action 비워주기
        if len(self.saved_actions) == self.env.max_flights:
            del self.saved_actions[:]

        return state, selected_action

    # 에피소드가 끝날 때 호출되며, 저장된 로그 확률과 보상 정보를 사용하여 정책 업데이트를 수행
    def update(self):
        # 누적 보상(R)과 정책 손실(policy_loss), 반환값(returns) 리스트를 초기화 
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]: # 역순으로 저장된 보상을 반복하며 반환값을 계산
            R = r + self.gamma * R
            returns.insert(0, torch.tensor(R, requires_grad=True)) # 현재 반환값을 반환값 리스트의 맨 앞에 추가
        returns = torch.tensor(returns, requires_grad=True) # 반환값 리스트를 텐서로 변환하고, 평균과 표준편차를 사용하여 정규화함
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        for log_prob, R in zip(self.saved_log_probs, returns): # 저장된 로그 확률과 정규화된 반환값을 묶고 반복
            policy_loss.append(-log_prob * R) # 정책 손실에 (-로그 확률 * 정규화된 반환값)을 추가
        self.optimizer.zero_grad() # 옵티마이저의 그래디언트를 초기화
        policy_loss = sum(policy_loss) # 정책 손실을 계산 
        policy_loss.backward() # 역전파를 수행
        self.optimizer.step()
        del self.rewards[:] # 저장된 보상과 로그 확률 리스트를 비움
        del self.saved_log_probs[:]

initial_pairing_set = pd.read_excel(
    '/home/public/yunairline/ASCP/ReinforcementLearning/dataset/output.xlsx', header=1)
del initial_pairing_set['pairing index']

n_pairings = len(initial_pairing_set)
max_flight = len(initial_pairing_set.columns)
column = list(range(1, max_flight+1))
column_list = [str(item) for item in column]
initial_pairing_set.columns = column_list
initial_pairing_set = initial_pairing_set.fillna(-1)
pairing_matrix = initial_pairing_set.values  # numpy array

env = CrewPairingEnv(pairing_matrix)
agent = PolicyGradientAgent(env)

state_history = []
# 리셋하는 부분이 꼭 필요함.!!! -> 여기서 변경된 페어링 샛을 기준으로 새로운 state를 생성할 수 있도록 하고 episode가 끝나면 다시 리셋
pairing_set, state = env.reset(pairing_matrix)
state_history.append(state)
# 에피소드가 돌아갈때 마다 환경을 초기화 시킴 -> 초기에 받은 페어링셋을 1차원으로 만들어서 반환 & 랜덤 라이브러리로 내부에 있는 페어링셋 중 하나를 랜덤으로 선택해서 반환 (즉, 전체 데이터 중 하나의 비행의 인덱스를 반환하고 이를 state로 가짐)

def generate_random_choice(n_pairing, state_history):
    available_numbers = [num for num in range(n_pairing) if num not in state_history]
    
    if not available_numbers:
        print("No available numbers to choose from.")
        return None
    
    random_choice = random.choice(available_numbers)
    return random_choice

for i_episode in range(5):
    """
    하나의 episode에서 하는 일
    1. 랜덤하게 flihgt를 선택해서 state를 받음
    2. step 진행하여 갱신이 있을 때까지 반복
    3. 받은 reward를 저장
    4. step이 끝나면 저장한 reward를 통해 policy를 업데이트
    """
    print('episode : ', i_episode+1)
    
    done = False
    trycnt = 0

    while done == False:
        """
        하나의 step에서 하는 일
        1. state를 받아서 action을 선택
        2. 선택한 action을 통해 다음 state와 reward를 받음
        3. 1~4를 done==True가 될 때 까지 반복
        """
        trycnt += 1
        print('trycnt : ', trycnt)
        before_idx, after_idx = agent.select_action(state)
        # 만약 after_idx가 -1이면 루프 방지를 위해 종료
        if after_idx == -1:
            break
        # idx를 통해서 flight 정보 출력하기
        before_idx = int(before_idx)
        after_idx = int(after_idx)
        print('before_flight : ', pairing_set[before_idx], 'after_flight : ', pairing_set[after_idx])
        # 루프 방지 필요
        step_pairing_set, reward, done, _ = env.step(before_idx, after_idx, pairing_set)
        # pairingset은 done이 True가 아닌 경우 갱신하지 않음
        print('done : ', done, 'reward : ', reward, end='\n\n')

    agent.rewards.append(reward)
    pairing_set = step_pairing_set
    # 0~n_pairings의 array에서 state_history에 있는 값들을 제외한 값들 중 하나를 랜덤하게 선택
    random_choice = generate_random_choice(n_pairings, state_history)
    print('state : ', state)
    agent.update()

    reshape_list = env.reshape_list(
        pairing_set.tolist(), num_rows=215, num_columns=4)

    hard_score = 0
    soft_score = 0
    for pair in reshape_list:
        pair_hard_score, pair_soft_score = env.calculateScore(pair)
        hard_score += pair_hard_score
        soft_score += pair_soft_score

    # print('hard score : ', hard_score, 'soft score : ', soft_score)

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

    import openpyxl
    # 엑셀 파일로 저장
    file_path = "/home/public/yunairline/ASCP/ReinforcementLearning/output/output.xlsx"
    workbook = openpyxl.load_workbook(file_path)
    
    # 새로운 시트 생성
    new_sheet_name = 'episode'+str(i_episode+1)
    new_sheet = workbook.create_sheet(title=new_sheet_name)
    
    df.to_excel('/home/public/yunairline/ASCP/ReinforcementLearning/output/output'+str(new_sheet_name)+'.xlsx', sheet_name='episode'+str(i_episode+1), index=False)
