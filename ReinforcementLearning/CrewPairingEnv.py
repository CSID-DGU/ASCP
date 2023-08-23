import gym
import numpy as np
from gym import spaces
import pandas as pd
import random

input_flight = pd.read_excel('/home/public/yunairline/dataset/pairingdata/Input-data.xlsx', sheet_name='User_Flight', header=1)
input_deadhead = pd.read_excel('/home/public/yunairline/dataset/pairingdata/Input-data.xlsx', sheet_name='User_Deadhead', header=1)
input_salary = pd.read_excel('/home/public/yunairline/dataset/pairingdata/ASCP_Data_Input.xlsx', sheet_name='Program_Input_Aircraft', header=1)

del input_flight['INDEX']

class CrewPairingEnv(gym.Env):
    def __init__(self, initial_pairing_set):
        super(CrewPairingEnv, self).__init__()
        
        self.initial_pairing_set=initial_pairing_set              
        self.n_pairings=len(initial_pairing_set)
        self.max_flights = 4 #일단 임시로 4로 설정해둠. -> 후에 initial pairing set의 최대 flgiht로 설정할 것.
        self.action_space = spaces.MultiDiscrete([self.n_pairings, self.max_flights, self.n_pairings, self.max_flights])
        self.observation_space = spaces.Box(low=0, high=self.n_pairings, shape=(self.n_pairings, self.max_flights), dtype=np.int16)

    def reshape_list(self, input_list, num_rows, num_columns):
        total_elements = num_rows * num_columns
        if len(input_list) != total_elements:
            raise ValueError("Input list does not have the required number of elements.")

        reshaped_list = [input_list[i:i+num_columns] for i in range(0, total_elements, num_columns)]
        return reshaped_list
    
    def shift_minus_ones(self, lst):
        shifted_list = []
        
        for sublist in lst:
            new_sublist = [num for num in sublist if num != -1]  # -1을 제외한 숫자들
            new_sublist.extend([-1] * (len(sublist) - len(new_sublist)))  # 남는 자리에 -1 추가
            shifted_list.append(new_sublist)
        
        return shifted_list    

    def step(self, before_idx, after_idx):
        pairing_set = self.initial_pairing_set.copy()
        pairing_set = np.array(pairing_set).flatten()
        before_idx = int(before_idx)
        # 현재 데이터에 맞추어 설정해둔 값으로 나중에 다 받아올 수 있게 만들어야 함
        num_flights = 4
        num_rows = 215
        num_columns = num_flights    
            
        # 바꾸기 전 cost 계산
        reshaped_matrix = reshape_list(pairing_set, num_rows, num_columns)
        reshaped_matrix = [arr.tolist() for arr in reshaped_matrix]
        #print(reshaped_matrix)
        before_pairing_idx = before_idx // num_flights
        after_pairing_idx = after_idx // num_flights
        src_last_hard, src_last_soft = self.calculateScore(reshaped_matrix[before_pairing_idx])
        trg_last_hard, trg_last_soft = self.calculateScore(reshaped_matrix[after_pairing_idx])
        current_hard = src_last_hard + trg_last_hard
        current_soft = src_last_soft + trg_last_soft
        print('바꾸기 전 hard cost : ', current_hard, '\t바꾸기 전 soft cost : ', current_soft)
        print('바꾸기 전 pairing set : ', reshaped_matrix[before_pairing_idx], reshaped_matrix[after_pairing_idx])
        
        # 위치 서로 바꾸기
        pairing_set[before_idx], pairing_set[after_idx] = pairing_set[after_idx], pairing_set[before_idx]        
        reshaped_matrix = reshape_list(pairing_set, num_rows, num_columns)
        reshaped_matrix = [arr.tolist() for arr in reshaped_matrix]
        #print(reshaped_matrix)
        before_pairing_idx = before_idx // num_flights
        after_pairing_idx = after_idx // num_flights
        
        reshaped_matrix = shift_minus_ones(reshaped_matrix)

        src_new_hard, src_new_soft = self.calculateScore(reshaped_matrix[before_pairing_idx])
        trg_new_hard, trg_new_soft = self.calculateScore(reshaped_matrix[after_pairing_idx])
        new_hard = src_new_hard + trg_new_hard
        new_soft = src_new_soft + trg_new_soft
        print('바꾼 후 hard cost : ', new_hard, '\t바꾼 후 soft cost : ', new_soft)
        print('바꾼 후 pairing set : ', reshaped_matrix[before_pairing_idx], reshaped_matrix[after_pairing_idx])

        done = False
        
        # Calculate reward based on the improvement of the cost
        hard_reward = current_hard - new_hard
        soft_reward = current_soft - new_soft
        
        # Soft Score는 작아질수록 (0에 가까울수록) 더 개선된 것이므로 기존 경우에서 변경된 soft score가 더 작아지면 reward를 줌.
        done = (soft_reward > 0 and hard_reward >= 0)

        print('hard_reward : ', hard_reward, '\tsoft_reward : ', soft_reward)
        # Return new state, reward, done, and additional info
        # 우선 hard reward는 기존의 점수를 유지하기만 하면 되도록 설정해두기 위해 soft reward만 반환하도록 설정해둠.
        
        # reshaped_matritx flatten하기
        reshaped_matrix = [item for sublist in reshaped_matrix for item in sublist]
        reshaped_matrix = np.array(reshaped_matrix)
        
        return self.pairing_set, soft_reward, done, {}

    # 한 pair의 cost를 구하는 것으로 구현해둠.
    def calculateScore(self, pair):
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
            for i in range(1, pair_length):
                before_flight = pair[i]
                next_flight = pair[i+1]
                if next_flight == -1:
                    break
                
                
                before_flight_arrival_time = input_flight.loc[before_flight, 'DEST_DATE']
                next_flight_departure_time = input_flight.loc[next_flight, 'ORIGIN_DATE']
                flight_term_hours = int((next_flight_departure_time - before_flight_arrival_time).total_seconds() / 3600) # 오류 발생으로 인한 표현방식 수정
                before_flight_arrival_airport = input_flight.loc[before_flight, 'DEST']
                next_flight_departure_airport = input_flight.loc[next_flight, 'ORIGIN']
                before_flight_aircraft_type = input_flight.loc[before_flight, 'AIRCRAFT_TYPE']
                next_flight_aircraft_type = input_flight.loc[next_flight, 'AIRCRAFT_TYPE']

                # 비행일정의 선후관계 만족 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_arrival_time > next_flight_departure_time:
                    pair_hard_score += time_possible_score
                    print('[HARD] 선후관계 불만족', end='\t')
                # 선행 비행 도착지와 후행 비행 출발지 동일 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_arrival_airport != next_flight_departure_airport:
                    pair_hard_score += airport_possible_score
                    print('[HARD] 공항 불만족', end='\t')
                # 선행 비행 기종과 후행 비행 기종 동일 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_aircraft_type != next_flight_aircraft_type:
                    pair_hard_score += aircraft_type_score
                    print('[HARD] 기종 불만족', end='\t')
                # 총 비행 시간 계산 : 만약 마지막 호출 이라면 next_flight는 없으므로, next_flight 값도 더하기
                landing_times += input_flight.loc[before_flight, 'DURATION']
                if i == pair_length:
                    landing_times += input_flight.loc[next_flight, 'DURATION']
                # 선행 비행 도착시간과 후행 비행 출발시간의 차이가 1시간 이상인지 여부 -> 어기는 경우 100점씩 부여
                if flight_term_hours < break_term_threshold:
                    pair_soft_score += min_break_time_score    
                    print('[SOFT] 휴식시간 불만족 : ', min_break_time_score, end='\t')
                # 두 비행 사이 간격이 6시간 이상인 경우, 해당 시간만큼의 layover salary를 score를 추가해줌.
                if flight_term_hours >= layover_threshold:
                    layover_salary_per_hour = input_salary.loc[input_salary['AIRCRAFT'] == before_flight_aircraft_type, 'LAYOVER_SALARY'].values[0]
                    layover_salary_score += flight_term_hours*layover_salary_per_hour
                    pair_soft_score += layover_salary_score
                    print('[SOFT] layover salary 추가 : ', layover_salary_score, end='\t')
                # 두 비행 사이 간격이 3시간 이하인 경우, 만족도 하락 -> min(0,(180-휴식 시간)*1000) 의 합을 score에 추가해줌. -> 이게 무슨 의미인지 잘 모르겠음.
                if flight_term_hours <= flight_term_threshold:
                    satisfaction_score += (flight_term_threshold*60-flight_term_hours*60)*1000
                    pair_soft_score += satisfaction_score
                    print('[SOFT] 만족도 하락 : ', satisfaction_score, end='\t')
        elif (pair_length == 1): 
            if pair[0] == -1:
                print('pair', pair)
            landing_times += input_flight.loc[pair[0], 'DURATION']
        else:
            return 0, 0
        
        # 한 페어링의 기간이 7일 이상인지 여부 -> 어기는 경우 (총 일수 - 7) * 100점씩 부여
        if (pair[0] != -1) & (pair[pair_length-1] != -1):
            start_time = input_flight.loc[pair[0], 'ORIGIN_DATE']
            #print(pair[pair_length-1],pair_length)
            end_time = input_flight.loc[pair[pair_length-1], 'DEST_DATE']
            pairing_term_days = (end_time - start_time).days  # Extract days from Timedelta
            if pairing_term_days > 7:
                pair_hard_score += (pairing_term_days-7)*pairing_term_score
                #print('페어링 기간 불만족',pairing_term_days)
            # 한 페어링 내의 비행 횟수가 5회 이상인지 여부 -> 어기는 경우 1000점씩 부여
            if pair_length >= 5:
                pair_hard_score += pair_length_score
                #print('비행 횟수 불만족')
            # 페어링의 시작 공항과 끝 공항이 다른 경우 deadhead score를 추가해줌.
            start_airport = input_flight.loc[pair[0], 'ORIGIN']
            end_airport = input_flight.loc[pair[pair_length-1], 'DEST']
            if start_airport != end_airport:
                deadhead_cost = input_deadhead.loc[(input_deadhead['ORIGIN'] == end_airport) & (input_deadhead['DEST']==start_airport), 'deadhead'].values[0]
                print('[SOFT] deadhead_cost : ',deadhead_cost, end='\t')
                deadhead_score += deadhead_cost
                pair_soft_score += deadhead_score
        else:
            print('비행 없음')
            return 0, 0

        return pair_hard_score, pair_soft_score

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        initial_pairing_set_flatten = self.initial_pairing_set.flatten()
        random_index = random.choice(np.where(initial_pairing_set_flatten != -1)[0])
        state = random_index

        return initial_pairing_set_flatten, state