import gym
import numpy as np
import random
import pandas as pd
from copy import deepcopy
from gym import spaces
from Scorecalculator import ScoreCalculator
from Factory import ASCPFactory

class CrewPairingEnv(gym.Env):
    def __init__(self, initial_pairing_set, pairingList, maxFlight):
        super(CrewPairingEnv, self).__init__()
        # 초기 페어링 셋(id) <class 'numpy.ndarray'> (215, 4)
        self.initial_pairing_id_set = initial_pairing_set
        # 초기 페어링 리스트 (객체 형태)
        self.initial_pairing_set = pairingList
        # 갱신되는 pairing_id_set (처음 선언시 빈 array로 선언, 1차원으로 변환된 페어링 셋이 들어감)
        self.pairing_id_set = self.initial_pairing_id_set
        # 갱신되는 pairing_set (처음 선언 시 빈 array로 선언, 객체 페어링 셋이 들어감)
        self.pairing_set = self.initial_pairing_set
        # 페어링 셋의 개수 (현재 데이터에서는 215개, 엑셀 행의 개수)
        self.n_pairings = len(self.initial_pairing_id_set)

        # (페어링 셋의 최대 비행 횟수 -> 들어오는 페어링의 최대 비행 * 2로 수정)
        # 행동 공간 정의를 위해서 필요한 값
        self.max_flights = maxFlight  # len(self.initial_pairing_id_set[0])

        # 강화 학습의 action space와 observation space 설정
        self.action_space = spaces.MultiDiscrete(
            [self.n_pairings, self.max_flights, self.n_pairings, self.max_flights])
        self.observation_space = spaces.Box(low=0, high=self.n_pairings, shape=(
            self.n_pairings, self.max_flights), dtype=np.int16)

    # 강화학습 계산을 위해서는 2차원으로 구성되어있는 전체 페어링 셋을 1차원으로 바꿔줘야 함.
    # (즉, 215개의 행을 하나로 쭉 나열해주는 것음 -> 215*4 = 860개의 원소를 가진 1차원 배열로 바꿔줌)
    # CalculateScore 함수는 2차원 배열의 각 pairing set에 대해서 cost를 계산해주는 함수이므로, 1차원으로 바꿔준 페어링 셋을 다시 2차원으로 바꿔줘야 함. 이를 위한 함수임.

    def reshape_list(self, input_list, num_rows, num_columns):
        total_elements = num_rows * num_columns
        if len(input_list) != total_elements:
            raise ValueError(
                "Input list does not have the required number of elements.")
        reshaped_list = [input_list[i:i+num_columns]
                         for i in range(0, total_elements, num_columns)]
        return reshaped_list

    # 페어링 셋에서 빈 값들은 -1로 채워져있음. (빈 값들도 행동 공간으로 계산을 진행해야 하기 때문)
    # 이때, 두 비행이 변경되는 경우 [-1, -1, 34, -1]의 예시와 같이 제일 첫번째 값이 빈 값이 되는 경우가 발생할 수 있음.
    # 이를 해결하기 위해 비행의 값들을 왼쪽으로 shift 시켜주는 함수임. [34, -1, -1 ,-1]과 같이.
    def shift_minus_ones(self, pairing_set):
        dummyFlight = ASCPFactory.dummyFlight
        idx_list = []
        pairing_id_set = [[flight.id for flight in pairing.pair] # 객체가 아닌, id만 포함된 페어링 셋
                          for pairing in pairing_set]

        pairing_idx = 0
        for pair_id_list in pairing_id_set: # id만 포함된 페어링셋의 하나하나의 페어링에 대하여
            flight_idx = 0 # 
            for flight_id in pair_id_list: # 해당 페어링의 하나하나의 플라이트에 대하여
                if flight_id == -1: # 플라이트 id가 -1 이라면
                    idx_list.append([pairing_idx, flight_idx]) # 해당 플라이트가 속한 페어링 인덱스와 플라이트 인덱슬를 idx_list에 저장
                flight_idx += 1
            pairing_idx += 1

        # idx_list를 역순으로 정렬
        idx_list = sorted(idx_list, key=lambda x: (x[0], x[1]), reverse=True)

        for idx in idx_list:  # 역순으로 pop. id가 -1이었던 객체는 pop되어 나가고, 다시 id가 -1인 더미Flight가 append되어 들어오면서, 중간에 껴있는 더미 Flight가 뒤로보내짐.
            pairing_set[idx[0]].pair.pop(idx[1])
            pairing_set[idx[0]].pair.append(dummyFlight)

        return pairing_set

    # step 함수는 처음에 임의로 선택한 비행의 인덱스와, 그 비행의 인덱스를 바꿀 비행의 인덱스를 받아서, 두 비행의 위치를 서로 바꿔주는 함수임.
    # 바꿔줬을 때, cost가 개선되면 done이라고 하는 종료 조건을 만족하게 되고, 이때까지의 reward를 반환함.
    def step(self, before_idx, after_idx):

        # 현재 데이터에 맞추어 설정해둔 값으로 나중에 다 받아올 수 있게 만들어야 함
        # reshape_list 함수를 위한 값들
        num_flights = self.max_flights
        num_rows = self.n_pairings
        num_columns = num_flights
        # print('------ 바꾸기 전 pairing set ------')

        # 비행의 인덱스를 가지고 페어링 셋의 인덱스로 변환(즉, 특정 비행의 인덱스를 가지고 해당 비행이 들어있는 페어링 셋의 인덱스를 찾음)
        before_pairing_idx = before_idx // num_flights # before_pairing_idx: 처음에 선택한 플라이트가 속해있는 페어링 인덱스
        before_flight_idx = before_idx % num_flights # before_flight_idx : 처음에 선택한 플라이트의 페어링 속 플라이트 인덱스
        after_pairing_idx = after_idx // num_flights # after_pairing_idx: 바꿀 대상 플라이트가 속해있는 페어링 인덱스
        after_flight_idx = after_idx % num_flights # after_flight_idx : 바꿀 대상 플라이트의 페어링 속 플라이트 인덱스

        before_pairing_set = self.pairing_set # 현재 클래스변수 pairing_set을 얕은복사로 가져옴
        after_pairing_set = deepcopy(self.pairing_set) # 현재 클래스변수 pairing_set을 얕은복사로 가져옴

        # flight 교환 이전, 처음 선택한 flight가 속해있는 pairing의 cost 계산
        src_last_hard, src_last_soft = self.calculateScore(
            before_pairing_set[before_pairing_idx])
        # flight 교환 이전, 두번째 선택된 flight가 속해있는 pairing의 cost 계산
        trg_last_hard, trg_last_soft = self.calculateScore(
            after_pairing_set[after_pairing_idx])
        # 두개의 pairing에서의 soft값과 hard값을 가지고 현재 cost 계산
        current_hard = src_last_hard + trg_last_hard
        current_soft = src_last_soft + trg_last_soft

        # //////////////////// 위치 서로 바꾸기 ///////////////////

        temp_flight = after_pairing_set[before_pairing_idx].pair[before_flight_idx]
        after_pairing_set[before_pairing_idx].pair[before_flight_idx] = after_pairing_set[after_pairing_idx].pair[after_flight_idx]
        after_pairing_set[after_pairing_idx].pair[after_flight_idx] = temp_flight

        # 새로 바뀐 페어링 셋에 대해서는 -1이 맨 앞에 위치할 수 없으므로, shift_minus_ones 함수를 통해 -1이 맨 앞에 위치하는 경우를 없애줌.
        reshaped_pairing_set = self.shift_minus_ones(after_pairing_set)

        # flight 교환 이후, 첫번째 선택된 flight가 포함된 pairing의 cost 계산
        src_new_hard, src_new_soft = self.calculateScore(
            reshaped_pairing_set[before_pairing_idx])
        # flight 교환 이후, 두번째 선택된 flight가 포함된 pairing의 cost 계산
        trg_new_hard, trg_new_soft = self.calculateScore(
            reshaped_pairing_set[after_pairing_idx])
        # 두개의 pairing에서의 soft값과 hard값을 가지고 현재 cost 계산
        new_hard = src_new_hard + trg_new_hard
        new_soft = src_new_soft + trg_new_soft

        # 종료 조건을 처음에는 False로 설정해둠.
        done = False

        # 현재의 cost와 바꾼 후의 cost를 비교해서 reward를 설정해줌.
        # hard_reward = current_hard - new_hard
        # Soft Score는 작아질수록 (0에 가까울수록) 더 개선된 것이므로 기존 경우에서 변경된 soft score가 더 작아지면 reward를 줌.
        soft_reward = current_soft - new_soft

        # 종료 조건은 hard reward가 0보다 크고, soft reward가 0보다 크거나 같은 경우임.
        # 즉, hard는 개선이 되거나 유지가 되어야 하고, soft는 개선이 되어야 함.
        done = (soft_reward > 0 and new_hard == 0)

        # 만약 done이 True가 되면, 개선이 되었다는 뜻이므로 
        # 변경을 클래스변수 self.pairing_set에 적용해줌.
        if done is True: 
            df = pd.DataFrame([[flight.id for flight in pairing.pair]
                               for pairing in reshaped_pairing_set])  # (id만 있는 페어링 리스트도 맞춰서 업데이트)
            reshaped_pairing_id_set = df.values  # 2차원의 배열 형식으로 변환

            # self.pairing_set 업데이트. (객체화된 페어링 리스트를 바뀐 페어링으로 업데이트)
            self.pairing_set = reshaped_pairing_set  # 클래스변수 pairing_set도 업데이트

            # 업데이트 되었던, id만 있는 pairing_id_set을 flatten
            self.pairing_id_set = np.array(reshaped_pairing_id_set).flatten()
            return self.pairing_id_set, soft_reward, done, {}

        # 우선 hard reward는 기존의 점수를 유지하기만 하면 되도록 설정해두기 위해 soft reward만 반환하도록 설정해둠.
        # done이 아니라면, 기존의 페어링 셋을 반환해줌.
        else:           
                
            return None, soft_reward, done, {}

    def reset(self):
        """
        input : pairing_set numpy array(2-dim)
        output : pairing_set_flatten numpy array(1-dim), state(randomly selected flight index)
        역할 : 초기 페어링 셋을 받아서 1차원으로 바꿔주고, 그 중 하나의 비행을 랜덤으로 선택해서 state로 반환함.
        """

        self.pairing_id_set = np.array(self.pairing_id_set).flatten() # 현재 pairing_id_set을 numpy 형태로 변환 후 flatten
        random_index = random.choice(np.where(self.pairing_id_set != -1)[0]) # pairing_id_set에서 무작위로 하나의 flight 선택 (첫번째 선택된 Flight)
        state = random_index

        return state

    # 한 pair의 cost를 구하는 것으로 구현해둠.
    def calculateScore(self, pairing):

        calculator = ScoreCalculator(pairing)
        pair_hard_score,pair_soft_score=calculator.calculateScore
       

        return pair_hard_score, pair_soft_score

    def render(self):
        print('Not use this method')
