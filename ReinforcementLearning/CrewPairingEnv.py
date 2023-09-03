import gym
import numpy as np
from gym import spaces
import pandas as pd
import random

# 비행 정보 받아오기 : dataframe 형태로 받아옴
input_flight = pd.read_excel(
    '/home/public/yunairline/ASCP/ReinforcementLearning/dataset/ASCP_Data_Input_new.xlsx', sheet_name='User_Flight', header=2)
# deadhead 정보 받아오기 : dataframe 형태로 받아옴
input_deadhead = pd.read_excel(
    '/home/public/yunairline/ASCP/ReinforcementLearning/dataset/ASCP_Data_Input_new.xlsx', sheet_name='User_Deadhead', header=2)
# salary 정보 받아오기 : dataframe 형태로 받아옴
input_salary = pd.read_excel(
    '/home/public/yunairline/ASCP/ReinforcementLearning/dataset/ASCP_Data_Input_new.xlsx', sheet_name='Program_Cost', header=1)

# DUARTION 열 추가 (기존의 CalculateScore 함수는 시간을 시단위로 계산하게 되어있음 ex. 1시간 30분 -> 1.5시간)
# 날짜 형식 변환
date_format = '%m/%d/%y %H:%M'
input_flight['ORIGIN_DATE'] = pd.to_datetime(
    input_flight['ORIGIN_DATE'], format=date_format)
input_flight['DEST_DATE'] = pd.to_datetime(
    input_flight['DEST_DATE'], format=date_format)
# DURATION 열 계산 (초 단위)
input_flight['DURATION_SECONDS'] = (
    input_flight['DEST_DATE'] - input_flight['ORIGIN_DATE']).dt.total_seconds()
# DURATION 열 변환 (소수로 표현)
input_flight['DURATION'] = input_flight['DURATION_SECONDS'] / \
    3600  # 초를 시간으로 변환


class CrewPairingEnv(gym.Env):
    def __init__(self, initial_pairing_set):
        super(CrewPairingEnv, self).__init__()
        # 초기 페어링 셋
        self.initial_pairing_set = initial_pairing_set
        # 페어링 셋의 개수 (현재 데이터에서는 215개, 엑셀 행의 개수)
        self.n_pairings = len(self.initial_pairing_set)

        # (페어링 셋의 최대 비행 횟수 -> 들어오는 페어링의 최대 비행 * 2로 수정)
        # 행동 공간 정의를 위해서 필요한 값
        self.max_flights = len(self.initial_pairing_set[0])

        # 강화 학습의 action space와 observation space 설정
        self.action_space = spaces.MultiDiscrete(
            [self.n_pairings, self.max_flights, self.n_pairings, self.max_flights])
        self.observation_space = spaces.Box(low=0, high=self.n_pairings, shape=(
            self.n_pairings, self.max_flights), dtype=np.int16)
        
        #print('env init')
        #print('n_pairings : ', self.n_pairings)
        #print('max_flights : ', self.max_flights)

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
    def step(self, before_idx, after_idx, pairing_set):

        # 현재 데이터에 맞추어 설정해둔 값으로 나중에 다 받아올 수 있게 만들어야 함
        # reshape_list 함수를 위한 값들
        num_flights = 4
        num_rows = 215
        num_columns = num_flights
        before_idx = int(before_idx)
        after_idx = int(after_idx)  
        #print('------ 바꾸기 전 pairing set ------')
        # 바꾸기 전 cost 계산
        reshaped_matrix = self.reshape_list(pairing_set, num_rows, num_columns)  # 2차원 배열로 변환 하고
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
        #print()
        #print('바꾸기 전 hard cost : ', current_hard,'\t바꾸기 전 soft cost : ', current_soft)

        # //////////////////// 위치 서로 바꾸기 ///////////////////
        #print('------ 바꾼 후 pairing set ------')
        pairing_set[before_idx], pairing_set[after_idx] = pairing_set[after_idx], pairing_set[before_idx]
        reshaped_matrix = self.reshape_list(
            pairing_set, num_rows, num_columns)  # 2차원 배열로 변환 하고
        reshaped_matrix = [arr.tolist()
                           for arr in reshaped_matrix]  # numpy array를 list로 변환
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
        #print('바꾼 후 hard cost : ', new_hard, '\t바꾼 후 soft cost : ', new_soft)

        # 종료 조건을 처음에는 False로 설정해둠.
        done = False

        # 현재의 cost와 바꾼 후의 cost를 비교해서 reward를 설정해줌.
        hard_reward = current_hard - new_hard
        # Soft Score는 작아질수록 (0에 가까울수록) 더 개선된 것이므로 기존 경우에서 변경된 soft score가 더 작아지면 reward를 줌.
        soft_reward = current_soft - new_soft

        # 종료 조건은 hard reward가 0보다 크고, soft reward가 0보다 크거나 같은 경우임.
        # 즉, hard는 개선이 되거나 유지가 되어야 하고, soft는 개선이 되어야 함.
        done = (soft_reward > 0 and hard_reward >= 0)
        #print('hard_reward : ', hard_reward, '\tsoft_reward : ', soft_reward)
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
        return pairing_set, soft_reward, done, {}

    # 한 pair의 cost를 구하는 것으로 구현해둠.
    def calculateScore(self, pair):
        #print('pair : ', pair, end='\t')
        pair_hard_score = 0
        time_possible_score = 1000
        airport_possible_score = 1000
        aircraft_type_score = 1000
        landing_times = 0
        pair_length_score = 1000
        landing_times_score = 100
        pairing_term_score = 100
        min_break_time_score = 500
        pair_length = sum(1 for flight_index in pair if flight_index != -1)
        pair_soft_score = 0
        deadhead_score = 0
        layover_salary_score = 0
        layover_threshold = 6
        flight_term_threshold = 3
        break_term_threshold = 1
        satisfaction_score = 0

        if pair_length > 1:
            for i in range(0, pair_length-1):
                before_flight = pair[i]
                next_flight = pair[i+1]
                if next_flight == -1:
                    break
                # #print('before_flight : ', before_flight, '\tnext_flight : ', next_flight, end='\t')
                before_flight_arrival_time = input_flight.loc[before_flight, 'DEST_DATE']
                next_flight_departure_time = input_flight.loc[next_flight, 'ORIGIN_DATE']
                flight_term_hours = int(
                    (next_flight_departure_time - before_flight_arrival_time).total_seconds() / 3600)  # 오류 발생으로 인한 표현방식 수정
                before_flight_arrival_airport = input_flight.loc[before_flight, 'DEST']
                next_flight_departure_airport = input_flight.loc[next_flight, 'ORIGIN']
                before_flight_aircraft_type = input_flight.loc[before_flight,
                                                               'AIRCRAFT_TYPE']
                next_flight_aircraft_type = input_flight.loc[next_flight,
                                                             'AIRCRAFT_TYPE']

                # 비행일정의 선후관계 만족 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_arrival_time > next_flight_departure_time:
                    pair_hard_score += time_possible_score
                    #print('[HARD] 선후관계 불만족', end='\t')
                # 선행 비행 도착지와 후행 비행 출발지 동일 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_arrival_airport != next_flight_departure_airport:
                    pair_hard_score += airport_possible_score
                    #print('[HARD] 공항 불만족', end='\t')
                # 선행 비행 기종과 후행 비행 기종 동일 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_aircraft_type != next_flight_aircraft_type:
                    pair_hard_score += aircraft_type_score
                    #print('[HARD] 기종 불만족', end='\t')
                # 총 비행 시간 계산 : 만약 마지막 호출 이라면 next_flight는 없으므로, next_flight 값도 더하기
                landing_times += input_flight.loc[before_flight, 'DURATION']
                if i == pair_length:
                    landing_times += input_flight.loc[next_flight, 'DURATION']
                # 선행 비행 도착시간과 후행 비행 출발시간의 차이가 1시간 이상인지 여부 -> 어기는 경우 100점씩 부여
                if flight_term_hours < break_term_threshold:
                    pair_soft_score += min_break_time_score
                    #print('[SOFT] 휴식시간 불만족 : ', min_break_time_score, end='\t')
                # 두 비행 사이 간격이 6시간 이상인 경우, 해당 시간만큼의 layover salary를 score를 추가해줌.
                if flight_term_hours >= layover_threshold:
                    layover_salary_per_hour = input_salary.loc[input_salary['AIRCRAFT']
                                                               == before_flight_aircraft_type, 'Layover Cost(원/분)'].values[0]
                    layover_salary_score += flight_term_hours*layover_salary_per_hour
                    pair_soft_score += layover_salary_score
                    #print('[SOFT] layover salary 추가 : ',layover_salary_score, end='\t')
                # 두 비행 사이 간격이 3시간 이하인 경우, 만족도 하락 -> min(0,(180-휴식 시간)*1000) 의 합을 score에 추가해줌. -> 이게 무슨 의미인지 잘 모르겠음.
                if flight_term_hours <= flight_term_threshold:
                    satisfaction_score += (flight_term_threshold *
                                           60-flight_term_hours*60)*1000
                    pair_soft_score += satisfaction_score
                    #print('[SOFT] 만족도 하락 : ', satisfaction_score, end='\t')
        elif (pair_length == 1):
            if pair[0] == -1:
                #print('pair', pair)
                landing_times += input_flight.loc[pair[0], 'DURATION']
        else:
            return 0, 0

        # 한 페어링의 기간이 7일 이상인지 여부 -> 어기는 경우 (총 일수 - 7) * 100점씩 부여
        if (pair[0] != -1) & (pair[pair_length-1] != -1):
            start_time = input_flight.loc[pair[0], 'ORIGIN_DATE']
            # #print(pair[pair_length-1],pair_length)
            end_time = input_flight.loc[pair[pair_length-1], 'DEST_DATE']
            # Extract days from Timedelta
            pairing_term_days = (end_time - start_time).days
            if pairing_term_days > 7:
                pair_hard_score += (pairing_term_days-7)*pairing_term_score
                # #print('페어링 기간 불만족',pairing_term_days)
            # 한 페어링 내의 비행 횟수가 5회 이상인지 여부 -> 어기는 경우 1000점씩 부여
            if pair_length >= 5:
                pair_hard_score += pair_length_score
                # #print('비행 횟수 불만족')
            # 페어링의 시작 공항과 끝 공항이 다른 경우 deadhead score를 추가해줌.
            start_airport = input_flight.loc[pair[0], 'ORIGIN']
            end_airport = input_flight.loc[pair[pair_length-1], 'DEST']
            if start_airport != end_airport:
                deadhead_cost = input_deadhead.loc[(input_deadhead['출발 공항'] == end_airport) & (
                    input_deadhead['도착 공항'] == start_airport), 'Deadhead(원)'].values[0]
                #print('[SOFT] deadhead_cost : ', deadhead_cost, end='\t')
                deadhead_score += deadhead_cost
                pair_soft_score += deadhead_score
        else:
            #print('비행 없음')
            return 0, 0
        return pair_hard_score, pair_soft_score

    def reset(self, pairing_set):
        """
        input : pairing_set numpy array(2-dim)
        output : pairing_set_flatten numpy array(1-dim), state(randomly selected flight index)
        역할 : 초기 페어링 셋을 받아서 1차원으로 바꿔주고, 그 중 하나의 비행을 랜덤으로 선택해서 state로 반환함.
        """
        pairing_set_flatten = pairing_set.flatten()
        random_index = random.choice(np.where(pairing_set_flatten != -1)[0])
        state = random_index

        return pairing_set_flatten, state
