import gym
import numpy as np
import random
from gym import spaces
from Scorecalculator import ScoreCalculator

# 2023. 09.04 동겸 작성
# 현재: pairing을 numpy로 받아와서 1차원으로 flatten시킨다. -> 랜덤하게 한 flight를 골라, 해당 flight를 select_action 메소드를 불러와 옮긴다 -> 다시
#    다시 2차원 평면으로 만든다.
# Q: 근데, 2차원으로 다시 만들 때 뭘 기준으로 하나의 페어링을 삼는가? F1 F2 F3 F4 F5,,, 이렇게 flatten 되어있는 numpy가 있다면,
#   하나의 페어링을 어떤 기준을 세워 정할 것인지...

# 미래: pairing을 pairing 객체가 포함된 리스트로 받아와서, 한 페어링 내의 특정 flight를 골라(인덱스 형식으로, ex. (3,4))
#     다른 pairing의 특정 인덱스로 옮김. input의 형식이 바뀌어야함. (3,4,2,1) : 3번 페어링의 4번째 플라이트를, 2번 페어링의 1번째 위치로.
# 이렇게 바꾸면 자리를 바꾸고 나서도 페어링의 구조가 확립이 된 상태로 존재할 수 있게됨.

# 아직 손봐야 하는 부분: Scorecalculator.py 인자가 두개인 부분이 의도대로 코딩 안된게 한두개 있는것같음(느낌상). 다시 확인하기
# 변경한부분: DeadheadCost를 Airport의 멤버로 들어가있던걸, Airport에서 빼고 그냥 메소드를 통해 받아오는 것으로 변경함(얘기해봐야함)
#           Flight.py의 destAirport와 originAirport 입력 방식 오류 수정.
#           Aircraft.py, Airport.py, Flight.py, Pairing.py의 getter, setter 무한재귀 현상 수정
#           


class CrewPairingEnv(gym.Env):
    def __init__(self, initial_pairing_set, maxFlight):
        super(CrewPairingEnv, self).__init__()
        # 초기 페어링 셋 <class 'numpy.ndarray'> (215, 4)
        self.initial_pairing_set = initial_pairing_set

        # 갱신되는 pairing_set (처음 선언시 빈 array로 선언, 1차원으로 변환된 페어링 셋이 들어감)
        self.pairing_set = self.initial_pairing_set

        # 페어링 셋의 개수 (현재 데이터에서는 215개, 엑셀 행의 개수)
        self.n_pairings = len(self.initial_pairing_set)

        # (페어링 셋의 최대 비행 횟수 -> 들어오는 페어링의 최대 비행 * 2로 수정)
        # 행동 공간 정의를 위해서 필요한 값
        self.max_flights = maxFlight  # len(self.initial_pairing_set[0])

        # 강화 학습의 action space와 observation space 설정
        self.action_space = spaces.MultiDiscrete(
            [self.n_pairings, self.max_flights, self.n_pairings, self.max_flights])
        self.observation_space = spaces.Box(low=0, high=self.n_pairings, shape=(
            self.n_pairings, self.max_flights), dtype=np.int16)

    # 강화학습 계산을 위해서는 2차원으로 구성되어있는 전체 페어링 셋을 1차원으로 바꿔줘야 함.
    # (즉, 215개의 행을 하나로 쭉 나열해주는 것음 -> 215*4 = 860개의 원소를 가진 1차원 배열로 바꿔줌)
    # CalculateScore 함수는 2차원 배열의 각 pairing set에 대해서 cost를 계산해주는 함수이므로, 1차원으로 바꿔준 페어링 셋을 다시 2차원으로 바꿔줘야 함. 이를 위한 함수임.

    def reshape_list(self, input_list, num_rows, num_columns):
        print(input_list)
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
    def shift_minus_ones(self, lst):
        shifted_list = []
        for sublist in lst:
            new_sublist = [num for num in sublist if num != -1]  # -1을 제외한 숫자들
            new_sublist.extend(
                [-1] * (len(sublist) - len(new_sublist)))  # 남는 자리에 -1 추가
            shifted_list.append(new_sublist)
        return shifted_list

    # step 함수는 처음에 임의로 선택한 비행의 인덱스와, 그 비행의 인덱스를 바꿀 비행의 인덱스를 받아서, 두 비행의 위치를 서로 바꿔주는 함수임.
    # 바꿔줬을 때, cost가 개선되면 done이라고 하는 종료 조건을 만족하게 되고, 이때까지의 reward를 반환함.
    # pylint: disable=W0221
    def step(self, before_idx, after_idx, pairing_set):

        # 현재 데이터에 맞추어 설정해둔 값으로 나중에 다 받아올 수 있게 만들어야 함
        # reshape_list 함수를 위한 값들
        num_flights = self.max_flights
        num_rows = self.n_pairings
        num_columns = num_flights
        before_idx = int(before_idx)
        after_idx = int(after_idx)
        # print('------ 바꾸기 전 pairing set ------')
        # 바꾸기 전 cost 계산
        reshaped_matrix = self.reshape_list(
            pairing_set, num_rows, num_columns)  # 2차원 배열로 변환 하고
        reshaped_matrix = [arr.tolist()
                           for arr in reshaped_matrix]  # numpy array를 list로 변환

        # 비행의 인덱스를 가지고 페어링 셋의 인덱스로 변환(즉, 특정 비행의 인덱스를 가지고 해당 비행이 들어있는 페어링 셋의 인덱스를 찾음)
        before_pairing_idx = before_idx // num_flights
        after_pairing_idx = after_idx // num_flights
        # 페어링 인덱스를 가지고 before pairing의 cost 계산
        src_last_hard, src_last_soft = self.calculateScore(
            reshaped_matrix[before_pairing_idx])
        # 페어링 인덱스를 가지고 after pairing의 cost 계산
        trg_last_hard, trg_last_soft = self.calculateScore(
            reshaped_matrix[after_pairing_idx])
        # 두개의 pairing에서의 soft값과 hard값을 가지고 현재 cost 계산
        current_hard = src_last_hard + trg_last_hard
        current_soft = src_last_soft + trg_last_soft
        # print()
        # print('바꾸기 전 hard cost : ', current_hard,'\t바꾸기 전 soft cost : ', current_soft)

        # //////////////////// 위치 서로 바꾸기 ///////////////////
        # print('------ 바꾼 후 pairing set ------')
        pairing_set[before_idx], pairing_set[after_idx] = pairing_set[after_idx], pairing_set[before_idx]
        reshaped_matrix = self.reshape_list(
            pairing_set, num_rows, num_columns)  # 2차원 배열로 변환 하고
        # reshaped_matrix = [arr.tolist()for arr in reshaped_matrix]  # numpy array를 list로 변환
        # 비행의 인덱스를 가지고 페어링 셋의 인덱스로 변환(즉, 특정 비행의 인덱스를 가지고 해당 비행이 들어있는 페어링 셋의 인덱스를 찾음)
        before_pairing_idx = before_idx // num_flights
        after_pairing_idx = after_idx // num_flights
        # 새로 바뀐 페어링 셋에 대해서는 -1이 맨 앞에 위치할 수 없으므로, shift_minus_ones 함수를 통해 -1이 맨 앞에 위치하는 경우를 없애줌.
        reshaped_matrix = self.shift_minus_ones(reshaped_matrix)
        # 페어링 인덱스를 가지고 before pairing의 cost 계산
        src_new_hard, src_new_soft = self.calculateScore(
            reshaped_matrix[before_pairing_idx])
        # 페어링 인덱스를 가지고 after pairing의 cost 계산
        trg_new_hard, trg_new_soft = self.calculateScore(
            reshaped_matrix[after_pairing_idx])
        # 두개의 pairing에서의 soft값과 hard값을 가지고 현재 cost 계산
        new_hard = src_new_hard + trg_new_hard
        new_soft = src_new_soft + trg_new_soft
        # print('바꾼 후 hard cost : ', new_hard, '\t바꾼 후 soft cost : ', new_soft)

        # 종료 조건을 처음에는 False로 설정해둠.
        done = False

        # 현재의 cost와 바꾼 후의 cost를 비교해서 reward를 설정해줌.
        hard_reward = current_hard - new_hard
        # Soft Score는 작아질수록 (0에 가까울수록) 더 개선된 것이므로 기존 경우에서 변경된 soft score가 더 작아지면 reward를 줌.
        soft_reward = current_soft - new_soft

        # 종료 조건은 hard reward가 0보다 크고, soft reward가 0보다 크거나 같은 경우임.
        # 즉, hard는 개선이 되거나 유지가 되어야 하고, soft는 개선이 되어야 함.
        done = (soft_reward > 0 and hard_reward >= 0)
        # print('hard_reward : ', hard_reward, '\tsoft_reward : ', soft_reward)
        # Return new state, reward, done, and additional info

        # 만약 done이 True가 되면, reshaped_matrix를 shift_minus_ones 함수를 통해 -1이 맨 앞에 위치하는 경우를 없애줌.
        # 그리고 변경된 리스트를 반환해줌
        if done:
            reshaped_matrix = self.shift_minus_ones(reshaped_matrix)
            # reshaped_matritx 다시 flatten하기
            reshaped_matrix = [
                item for sublist in reshaped_matrix for item in sublist]
            reshaped_matrix = np.array(reshaped_matrix)
            return reshaped_matrix, soft_reward, done, {}

        # 우선 hard reward는 기존의 점수를 유지하기만 하면 되도록 설정해두기 위해 soft reward만 반환하도록 설정해둠.
        # done이 아니라면, 기존의 페어링 셋을 반환해줌.
        return None, soft_reward, done, {}

    def reset(self):
        """
        input : pairing_set numpy array(2-dim)
        output : pairing_set_flatten numpy array(1-dim), state(randomly selected flight index)
        역할 : 초기 페어링 셋을 받아서 1차원으로 바꿔주고, 그 중 하나의 비행을 랜덤으로 선택해서 state로 반환함.
        """

        self.pairing_set = self.pairing_set
        random_index = random.choice(np.where(self.pairing_set != -1)[0])
        state = random_index

        return state

    # 한 pair의 cost를 구하는 것으로 구현해둠.
    def calculateScore(self, pairing):

        calculator = ScoreCalculator(pairing)
        pair_hard_score = 0
        pair_soft_score = 0

        # 비행일정의 선후관계 만족 여부 -> 어기는 경우 1000점씩 부여
        pair_hard_score = calculator.airportPossible()
        # 선행 비행 도착지와 후행 비행 출발지 동일 여부 -> 어기는 경우 1000점씩 부여
        pair_hard_score = calculator.timePossible()
        # 선행 비행 기종과 후행 비행 기종 동일 여부 -> 어기는 경우 1000점씩 부여
        pair_hard_score = calculator.aircraftType()
        # 비행 횟수 제약: 비행 횟수가 4회 이상일 시 -> 하드스코어 부여(총 비행횟수 * 100)
        pair_hard_score = calculator.landingTimes()
        # 비행 일수 제약: 페어링 총 기간이 7일 이상일 시 -> 하드스코어 부여((총 길이-7) * 100)
        pair_hard_score = calculator.pairLength()
        pair_hard_score = calculator.continuityPossible()  # 왜 얘만 자바에 두개들어가있는지 모르겠음
        # deadhead cost 계산(Base diff): 출발공항과 도착 공항이 다를 경우 소프트 점수 부여
        pair_soft_score = calculator.baseDiff()
        # 두 비행 사이 간격이 6시간 이상인 경우, 해당 시간만큼의 layover salary를 score를 추가해줌.
        pair_soft_score = calculator.layoverCost()
        # 총 이동근무 cost 계산(MovingWork cost):페어링 길이가 2 이상일 시 - > 소프트스코어 부여(MovingWork cost 발생 시 cost+)
        pair_soft_score = calculator.movingWorkCost()
        # 선행 비행 도착시간과 후행 비행 출발시간의 차이가 1시간 이상인지 여부 -> 어기는 경우 100점씩 부여
        pair_soft_score = calculator.quickTurnCost()
        # 총 호텔숙박비 cost 계산(Hotel cost): 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Hotel cost 발생 시 cost+)
        pair_soft_score = calculator.hotelCost()
        # 두 비행 사이 간격이 3시간 이하인 경우, 만족도 하락 -> min(0,(180-휴식 시간)*1000) 의 합을 score에 추가해줌. -> 이게 무슨 의미인지 잘 모르겠음.
        pair_soft_score = calculator.satisCost()

        return pair_hard_score, pair_soft_score

    def render(self):
        print('Not use this method')
