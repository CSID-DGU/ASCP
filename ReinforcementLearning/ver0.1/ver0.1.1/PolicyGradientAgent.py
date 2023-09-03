import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Policy를 파라미터화 하기 위한 신경망. 하이퍼파라미터 튜닝 작업 필요
# input: current state(all flight-> pairing * max flight 형태로 제공)
# output: probability of each action


class PolicyNetwork(nn.Module):  # nn.Module은 PyTorch에서 신경망 모델을 정의하고 관리하기 위한 기본 클래스
    # 강화 학습 에이전트의 정책(policy)을 신경망으로 모델링한 클래스
    # 주어진 상태를 입력으로 받아서 각 가능한 행동에 대한 확률을 출력
    def __init__(self, n_inputs, n_outputs):  # input: 입력 뉴런 수, 출력 뉴런 수(가능한 모든 행동의 수)
        # 상위 클래스인 nn.Module의 생성자를 호출하여 초기화
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(  # 정책 신경망에 사용할 신경망 구조를 정의, 여기서는 nn.Sequential을 사용하여 여러 개의 레이어를 순차적으로 연결하고 있다.
            # 입력으로 256개의 뉴런을 가지는 fully connected (dense) 레이어를 생성
            nn.Linear(n_inputs, 256),
            nn.ReLU(),  # 활성화 함수 ReLU는 비선형성을 추가하여 신경망이 복잡한 함수를 모델링할 수 있도록 돕는다.
            # 출력으로 256개의 뉴런을 가지는 fully connected (dense) 레이어를 생성
            nn.Linear(256, n_outputs),
            nn.Softmax(dim=-1))  # Softmax 함수를 사용하여 출력값을 확률 형태로 변환

    def forward(self, x):  # 주어진 입력 x를 정책 신경망의 레이어를 통과시켜서 각 행동에 대한 확률을 출력
        return self.network(x)

# 실제 강화 학습 에이전트를 정의하는 클래스
# 에이전트는 상태를 입력으로 받아서 행동을 선택하고, 선택한 행동의 로그 확률을 저장


class PolicyGradientAgent:
    def __init__(self, env, lr=0.01, gamma=0.99):  # 입력: 환경, 학습률, 감가율
        self.device = torch.device("cuda:0")  # 학습에 사용할 장치를 GPU로 설정
        self.env = env
        self.gamma = gamma

        n_pairings = len(env.initial_pairing_set)  # 초기 페어링 셋의 개수
        max_flights = env.max_flights  # 환경에서 최대 비행 횟수
        n_inputs = 1
        n_outputs = max_flights * n_pairings

        self.policy_net = PolicyNetwork(n_inputs, n_outputs).to(
            self.device)  # 정책 신경망(PolicyNetwork)을 생성

        # using Adam for optimizing (Adam 옵티마이저를 생성)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # 로그 확률, 보상, 선택한 행동을 저장하는 리스트들을 초기화
        self.rewards = []
        self.saved_actions = []
        # saved_log_probs: 선택한 행동의 로그 확률 -> action_probs와 차이점????
        self.saved_log_probs = []
        # action_probs: 각 action의 확률 -> 이게 self로 선언되어야 하는 이유는? -> 나중에 갱신될 수 있기 때문?
        self.action_probs = []

    # 주어진 상태에서 행동을 선택하고, 선택한 행동의 로그 확률을 저장, state: 현재 상태(즉, 내 위치)
    def select_action(self, state):
        # gpu 연산을 위해 numpy array를 torch tensor로 변환
        state = np.array(state)  # 주어진 상태를 numpy 배열로 변환
        state = torch.from_numpy(state).float().unsqueeze(0).to(
            self.device)  # state를 Pytorch 텐서로 변환, 배치 차원 추가 후 GPU로 이동
        action_probs = self.policy_net(state)  # 신경망에서 각 action의 확률 가져옴

        # 배치 제거, 텐서 cpu로 이동, 계산 그래프 분리, numpy로 변환
        self.action_probs = action_probs.squeeze().cpu().detach().numpy()

        # 이미 선택한 action을 제외한 확률 계산
        available_actions = [i for i in range(
            len(self.action_probs)) if i not in self.saved_actions]

        # available_actions가 비어있으면 종료
        if not available_actions:
            return state, None
        # available_actions의 action_probs가 모두 0이면 종료
        if sum(self.action_probs[available_actions]) == 0:
            return state, None
        # print('self.action_probs : ', self.action_probs[available_actions])
        adjusted_action_probs = self.action_probs[available_actions] / (sum(
            self.action_probs[available_actions]))  # sum(available_actions) = 1이 되도록 확률 조정
        # print('adjusted_action_probs : ', adjusted_action_probs, 'division : ', sum(self.action_probs[available_actions]))
        selected_action = np.random.choice(
            available_actions, p=adjusted_action_probs)  # 주어진 확률을 바탕으로 무작위 action 선택
        self.saved_actions.append(selected_action)  # 선택 action 저장
        # 선택한 action의 확률을 계산 그래프로 만들기 위해 텐서로 변환
        action_probs = torch.tensor(self.action_probs, requires_grad=True)
        self.saved_log_probs.append(
            torch.log(action_probs[selected_action]))  # 선택 action의 로그 확률 저장

        return state, selected_action

    # 에피소드가 끝날 때 호출되며, 저장된 로그 확률과 보상 정보를 사용하여 정책 업데이트를 수행
    def update(self):
        # 누적 보상(R)과 정책 손실(policy_loss), 반환값(returns) 리스트를 초기화
        R = 0
        policy_loss = []
        returns = []
        for r in self.rewards[::-1]:  # 역순으로 저장된 보상을 반복하며 반환값을 계산
            R = r + self.gamma * R
            # 현재 반환값을 반환값 리스트의 맨 앞에 추가
            returns.insert(0, torch.tensor(R, requires_grad=True))

        # 반환값 리스트를 텐서로 변환하고, 평균과 표준편차를 사용하여 정규화함
        returns = torch.tensor(returns, requires_grad=True)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        # 저장된 로그 확률과 정규화된 반환값을 묶고 반복
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)  # 정책 손실에 (-로그 확률 * 정규화된 반환값)을 추가
        # print('policy_loss : ', policy_loss, 'self.saved_log_probs : ', self.saved_log_probs)

        self.optimizer.zero_grad()  # 옵티마이저의 그래디언트를 초기화
        policy_loss = torch.stack(policy_loss).sum()
        # print('policy_loss : ', policy_loss)
        policy_loss.backward()  # 역전파를 수행
        self.optimizer.step()
        del self.rewards[:]  # 저장된 보상과 로그 확률 리스트, saved_actions를 비움
        del self.saved_log_probs[:]
        del self.saved_actions[:]
