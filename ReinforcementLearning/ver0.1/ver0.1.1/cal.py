import random
from getData import getInputData

input_flight, input_deadhead, input_salary = getInputData()

# 강화학습 계산을 위해서는 2차원으로 구성되어있는 전체 페어링 셋을 1차원으로 바꿔줘야 함.
# (즉, 215개의 행을 하나로 쭉 나열해주는 것음 -> 215*4 = 860개의 원소를 가진 1차원 배열로 바꿔줌)
# CalculateScore 함수는 2차원 배열의 각 pairing set에 대해서 cost를 계산해주는 함수이므로, 1차원으로 바꿔준 페어링 셋을 다시 2차원으로 바꿔줘야 함. 이를 위한 함수임.


def reshape_list(input_list, num_rows, num_columns):
    total_elements = num_rows * num_columns
    if len(input_list) != total_elements:
        raise ValueError(
            "Input list does not have the required number of elements.")
    reshaped_list = [input_list[i:i+num_columns]
                     for i in range(0, total_elements, num_columns)]
    return reshaped_list


def generate_random_choice(n_pairing, state_history):
    available_numbers = [num for num in range(
        n_pairing) if num not in state_history]

    if not available_numbers:
        print("No available numbers to choose from.")
        return None

    random_choice = random.choice(available_numbers)
    return random_choice

# 하나의 페어링 셋에 대해서 cost를 계산해주는 함수


def calculateScore(pair):
    pair_hard_score = 0
    time_possible_score = 1000
    airport_possible_score = 1000
    aircraft_type_score = 1000
    landing_times = 0
    pair_length_score = 1000
    # landing_times_score = 100
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
            if next_flight == -1 or before_flight == -1:
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
                # print('[HARD] 선후관계 불만족', end='\t')
            # 선행 비행 도착지와 후행 비행 출발지 동일 여부 -> 어기는 경우 1000점씩 부여
            if before_flight_arrival_airport != next_flight_departure_airport:
                pair_hard_score += airport_possible_score
                # print('[HARD] 공항 불만족', end='\t')
            # 선행 비행 기종과 후행 비행 기종 동일 여부 -> 어기는 경우 1000점씩 부여
            if before_flight_aircraft_type != next_flight_aircraft_type:
                pair_hard_score += aircraft_type_score
                # print('[HARD] 기종 불만족', end='\t')
            # 총 비행 시간 계산 : 만약 마지막 호출 이라면 next_flight는 없으므로, next_flight 값도 더하기
            landing_times += input_flight.loc[before_flight, 'DURATION']
            if i == pair_length:
                landing_times += input_flight.loc[next_flight, 'DURATION']
            # 선행 비행 도착시간과 후행 비행 출발시간의 차이가 1시간 이상인지 여부 -> 어기는 경우 100점씩 부여
            if flight_term_hours < break_term_threshold:
                pair_soft_score += min_break_time_score
                # print('[SOFT] 휴식시간 불만족 : ', min_break_time_score, end='\t')
            # 두 비행 사이 간격이 6시간 이상인 경우, 해당 시간만큼의 layover salary를 score를 추가해줌.
            if flight_term_hours >= layover_threshold:
                layover_salary_per_hour = input_salary.loc[input_salary['AIRCRAFT']
                                                           == before_flight_aircraft_type, 'Layover Cost(원/분)'].values[0]
                layover_salary_score += flight_term_hours*layover_salary_per_hour
                pair_soft_score += layover_salary_score
                # print('[SOFT] layover salary 추가 : ',layover_salary_score, end='\t')
            # 두 비행 사이 간격이 3시간 이하인 경우, 만족도 하락 -> min(0,(180-휴식 시간)*1000) 의 합을 score에 추가해줌. -> 이게 무슨 의미인지 잘 모르겠음.
            if flight_term_hours <= flight_term_threshold:
                satisfaction_score += (flight_term_threshold *
                                       60-flight_term_hours*60)*1000
                pair_soft_score += satisfaction_score
                # print('[SOFT] 만족도 하락 : ', satisfaction_score, end='\t')
    elif (pair_length == 1):
        if pair[0] != -1:
            # print('pair', pair)
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
            # print('[SOFT] deadhead_cost : ', deadhead_cost, end='\t')
            deadhead_score += deadhead_cost
            pair_soft_score += deadhead_score
    else:
        # print('비행 없음')
        return 0, 0
    return pair_hard_score, pair_soft_score
