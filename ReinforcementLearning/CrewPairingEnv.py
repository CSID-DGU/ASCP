import gym
import numpy as np
from gym import spaces

input_flight = pd.read_excel('/home/public/yunairline/dataset/pairingdata/Input-data.xlsx', sheet_name='User_Flight', header=1)
input_deadhead = pd.read_excel('/home/public/yunairline/dataset/pairingdata/Input-data.xlsx', sheet_name='User_Deadhead', header=1)
input_salary = pd.read_excel('/home/public/yunairline/dataset/pairingdata/ASCP_Data_Input.xlsx', sheet_name='Program_Input_Aircraft', header=1)

class CrewPairingEnv(gym.Env):
    def __init__(self, initial_pairing_set):
        super(CrewPairingEnv, self).__init__()
        
        self.initial_pairing_set=initial_pairing_set
        self.pairing_set = initial_pairing_set
        
        # self.current_cost=self.calculate_cost(self.pairing_set)
        # self.cost_threshold = cost_threshold
        hard_score = 0
        soft_score = 0
        for pair in self.pairing_set:
            pair_hard_score, pair_soft_score = self.calculateScore(self.pairing_set,pair)
            hard_score += pair_hard_score
            soft_score += pair_soft_score
        self.current_hard = hard_score
        self.current_soft = soft_score               

        # Define action and observation space
        # Assuming the observation is the cost of each flight in the pairing set
        # And each action represents moving a flight from one pairing to another
        self.n_pairings=len(initial_pairing_set)
        self.max_flights = 5 #일단 임시로 5로 설정해둠.
        self.action_space = spaces.MultiDiscrete([self.n_pairings, self.max_flights, self.n_pairings, self.max_flights])
        self.observation_space = spaces.Box(low=0, high=self.n_pairings, shape=(self.n_pairings, self.max_flights), dtype=np.int16)


    def step(self, action):

        source_pairing_index, flight_index, target_pairing_index, target_position = action

        # source pairing의 flight를, target pairing의 target position으로 이동시킴
        flight_to_move = self.pairing_set[source_pairing_index].pop(flight_index)
        self.pairing_set[target_pairing_index].insert(target_position, flight_to_move)


        # Current cost of the pairing set
        print('현재 hard cost : ', self.current_hard, '\t현재 soft cost : ', self.current_soft)
        
        # Calculate cost of the new pairing set
        src_new_hard, src_new_soft = self.calculateScore(self.pairing_set,self.pairing_set[source_pairing_index])
        trg_new_hard, trg_new_soft = self.calculateScore(self.pairing_set,self.pairing_set[target_pairing_index])
        new_hard = src_new_hard + trg_new_hard
        new_soft = src_new_soft + trg_new_soft

        # Check if the new cost meets the termination criteria : 기존에는 종료 조건을 임의로 설정했지만 이제는 soft 점수의 개선이 있고 hard 점수는 유지되면 종료하도록 설정해둠.
        # done = new_cost <= self.cost_threshold
        
        done = False
        
        # Calculate reward based on the improvement of the cost
        hard_reward = self.current_hard - new_hard
        soft_reward = self.current_soft - new_soft
        
        # Soft Score는 작아질수록 (0에 가까울수록) 더 개선된 것이므로 기존 경우에서 변경된 soft score가 더 작아지면 reward를 줌.
        done = (soft_reward > 0 and hard_reward == 0)

        # Return new state, reward, done, and additional info
        # 우선 hard reward는 기존의 점수를 유지하기만 하면 되도록 설정해두기 위해 soft reward만 반환하도록 설정해둠.
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
        pair_length = len(pair)-1
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
                before_flight_arrival_time = input_flight.loc[input_flight['INDEX']==before_flight, 'DEST_DATE'].values[0]
                next_flight_departure_time = input_flight.loc[input_flight['INDEX']==next_flight, 'ORIGIN_DATE'].values[0]
                flight_term_hours = int(next_flight_departure_time - before_flight_arrival_time) / 1e9 / 3600
                before_flight_arrival_airport = input_flight.loc[input_flight['INDEX']==before_flight, 'DEST'].values[0]
                next_flight_departure_airport = input_flight.loc[input_flight['INDEX']==next_flight, 'ORIGIN'].values[0]
                before_flight_aircraft_type = input_flight.loc[input_flight['INDEX']==before_flight, 'AIRCRAFT_TYPE'].values[0]
                next_flight_aircraft_type = input_flight.loc[input_flight['INDEX']==next_flight, 'AIRCRAFT_TYPE'].values[0]

                # 비행일정의 선후관계 만족 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_arrival_time > next_flight_departure_time:
                    pair_hard_score += time_possible_score
                    print('선후관계 불만족')
                # 선행 비행 도착지와 후행 비행 출발지 동일 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_arrival_airport != next_flight_departure_airport:
                    pair_hard_score += airport_possible_score
                    print('공항 불만족')
                # 선행 비행 기종과 후행 비행 기종 동일 여부 -> 어기는 경우 1000점씩 부여
                if before_flight_aircraft_type != next_flight_aircraft_type:
                    pair_hard_score += aircraft_type_score
                    print('기종 불만족')
                # 총 비행 시간 계산 : 만약 마지막 호출 이라면 next_flight는 없으므로, next_flight 값도 더하기
                landing_times += input_flight.loc[input_flight['INDEX']==before_flight, 'DURATION'].values[0]
                if i == pair_length:
                    landing_times += input_flight.loc[input_flight['INDEX']==next_flight, 'DURATION'].values[0]
                # 선행 비행 도착시간과 후행 비행 출발시간의 차이가 1시간 이상인지 여부 -> 어기는 경우 100점씩 부여
                if flight_term_hours < break_term_threshold:
                    pair_soft_score += min_break_time_score    
                    #print('휴식시간 불만족')
                # 두 비행 사이 간격이 6시간 이상인 경우, 해당 시간만큼의 layover salary를 score를 추가해줌.
                if flight_term_hours >= layover_threshold:
                    layover_salary_per_hour = input_salary.loc[input_salary['AIRCRAFT'] == before_flight_aircraft_type, 'LAYOVER_SALARY'].values[0]
                    layover_salary_score += flight_term_hours*layover_salary_per_hour
                    pair_soft_score += layover_salary_score
                    #print('layover salary 추가')
                # 두 비행 사이 간격이 3시간 이하인 경우, 만족도 하락 -> min(0,(180-휴식 시간)*1000) 의 합을 score에 추가해줌. -> 이게 무슨 의미인지 잘 모르겠음.
                if flight_term_hours <= flight_term_threshold:
                    satisfaction_score += (flight_term_threshold*60-flight_term_hours*60)*1000
                    pair_soft_score += satisfaction_score
                    #print('만족도 하락')
        else : 
            landing_times += input_flight.loc[input_flight['INDEX']==pair[1], 'DURATION'].values[0]

        # # 한 페어링의 비행 시간이 7시간 이상인지 여부 -> 어기는 경우 (총 비행시간 - 7) * 100점씩 부여
        # if landing_times > 7:
        #     pair_hard_score += (landing_times-7)*landing_times_score
        #     print('비행시간 불만족',landing_times)
        
        # 한 페어링의 기간이 7일 이상인지 여부 -> 어기는 경우 (총 일수 - 7) * 100점씩 부여
        start_time = input_flight.loc[input_flight['INDEX']==pair[1], 'ORIGIN_DATE'].values[0]
        end_time = input_flight.loc[input_flight['INDEX']==pair[pair_length], 'DEST_DATE'].values[0]
        pairing_term_days = int(end_time - start_time) / 1e9 / 3600 / 24
        if pairing_term_days > 7:
            pair_hard_score += (pairing_term_days-7)*pairing_term_score
            print('페어링 기간 불만족',pairing_term_days)
        # 한 페어링 내의 비행 횟수가 5회 이상인지 여부 -> 어기는 경우 1000점씩 부여
        if pair_length >= 5:
            pair_hard_score += pair_length_score
            print('비행 횟수 불만족')
        # 페어링의 시작 공항과 끝 공항이 다른 경우 deadhead score를 추가해줌.
        start_airport = input_flight.loc[input_flight['INDEX']==pair[1], 'ORIGIN'].values[0]
        end_airport = input_flight.loc[input_flight['INDEX']==pair[pair_length], 'DEST'].values[0]
        if start_airport != end_airport:
            deadhead_cost = input_deadhead.loc[(input_deadhead['ORIGIN'] == end_airport) & (input_deadhead['DEST']==start_airport), 'deadhead'].values[0]
            # print('deadhead_cost : ',deadhead_cost)
            deadhead_score += deadhead_cost
            pair_soft_score += deadhead_score
            #print('deadhead 발생')

        print(pair[0],end='\t')
        print('pair_hard_score : ',pair_hard_score, '\tpair_soft_score : ',pair_soft_score)
        # print('landing_times : ',landing_times,end='\t')
        # print('pair_length : ',pair_length,end='\t')
        # print('start_airport :',start_airport,' end_airport :',end_airport)
        return pair_hard_score, pair_soft_score

    def reset(self):
        """
        Reset the state of the environment to an initial state.
        """
        self.pairing_set = self.initial_pairing_set
        hard_score = 0
        soft_score = 0
        for pair in self.pairing_set:
            pair_hard_score, pair_soft_score = self.calculateScore(self.pairing_set,pair)
            hard_score += pair_hard_score
            soft_score += pair_soft_score
        self.current_hard = hard_score
        self.current_soft = soft_score
        return self.pairing_set, self.current_hard, self.current_soft