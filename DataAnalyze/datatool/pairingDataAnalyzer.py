import pandas as pd

#데이터 불러오기
pairing = pd.read_excel()
flight = pd.read_csv('flight.csv')
salary = pd.read_csv('salary.csv')
deadhead = pd.read_csv('deadhead.csv')


# 결과를 저장할 analysis 데이터 프레임 생성
analysis = pd.DataFrame(columns=['INDEX', 'flight_time', 'layover_time', 'showup_time', 'deadhead_time', 'start_time', 'end_time','total_time', 'flight_salary', 'layover_salary', 'showup_salary', 'base_salary', 'deadhead_salary','deadhead_cost', 'total_cost'])

from datetime import datetime
import time
from datetime import timedelta

datetime_format = "%Y-%m-%d %H:%M"

for i in range(len(pairing)):
    start = True
    sec_flight_total_time = 0
    showup_cnt = 0
    flight_total_sal = 0
    showup_sal = 0
    deadhead_cost = 0
    deadhead_sal = 0
    index = 'P'+str(i+1)
    showup_hour = 2
    deadhead_term = 0
    deadhead_time = 0
    for air in pairing.iloc[i]:
        # pairing 별 total 값 계산
        air = str(air)
        if (air != 'nan') & (air != 'None'):
            match = flight[flight['INDEX'] == air]
            
            #pairing의 시작시간 및 공항
            if start == True :
                start_time = datetime.strptime(match['origin'].values[0], datetime_format)
                start_port = match['ORIGIN'].values[0]
                pre_log_time = datetime.strptime(match['dest'].values[0], datetime_format)
                showup_cnt = 1
            start = False
            
            # 항공기종 확인
            aircraft = str(match['AIRCRAFT_TYPE'].values[0])
            
            # flight time & salary 계산
            flight_per_sal = salary.loc[salary['AIRCRAFT']==aircraft, 'FLIGHT_SALARY'].values[0]
            flight_time = datetime.strptime(match['dest'].values[0],datetime_format) - datetime.strptime(match['origin'].values[0],datetime_format)
            sec_flight_time = flight_time.total_seconds()
            flight_sal = flight_per_sal * (sec_flight_time/3600)
            sec_flight_total_time += sec_flight_time
            flight_total_sal += flight_sal
            
            #showup 계산을 위한 logtime 개선
            next_log_time = datetime.strptime(match['origin'].values[0], datetime_format)
            log_delta_time = next_log_time - pre_log_time
            pre_log_day = pre_log_time.day
            next_log_day = next_log_time.day
            
            pre_log_time = datetime.strptime(match['dest'].values[0], datetime_format)
            # N 시간 경과시 & 다른 날짜 새로운 쇼업 필요
            showup_term = 5
            if (log_delta_time.total_seconds() > (3600 * showup_term)) & (pre_log_day !=next_log_day):
                showup_cnt += 1
    
    # 총 flight time
    flight_total_time = timedelta(seconds=sec_flight_total_time)
    
    # pairing의 끝시간, 공항과 최종 시간 계산
    end_time = datetime.strptime(match['dest'].values[0], datetime_format)
    end_port = match['DEST'].values[0]
    total_time = end_time - start_time
    sec_total_time = total_time.total_seconds()
    
    # base salary
    base_sal = salary.loc[salary['AIRCRAFT']==aircraft, 'BASE_SALARY'].values[0]
    
    # layover time 및 salary 계산
    sec_layover_time = (sec_total_time - sec_flight_total_time)
    layover_time = timedelta(seconds=sec_layover_time)
    layover_sal =  (sec_layover_time/3600) * salary.loc[salary['AIRCRAFT']==aircraft, 'LAYOVER_SALARY'].values[0]
    
    # deadhead 계산
    if start_port != end_port:
        deadhead_cost = deadhead.loc[(deadhead['ORIGIN'] == end_port) & (deadhead['DEST'] == start_port), 'COST'].values[0]
        deadhead_time = deadhead.loc[(deadhead['ORIGIN'] == end_port) & (deadhead['DEST'] == start_port), 'TIME'].values[0]
        if deadhead_time != 'None' : deadhead_time = datetime.strptime(deadhead_time, '%H:%M:%S').time()
        deadhead_delta = timedelta(hours=deadhead_time.hour, minutes=deadhead_time.minute, seconds=deadhead_time.second)
        total_time = deadhead_delta + total_time
        sec_deadhead_time = deadhead_delta.total_seconds()
        deadhead_sal = (sec_deadhead_time/3600) * salary.loc[salary['AIRCRAFT']==aircraft, 'DEADHEAD_SALARY'].values[0]
    
    # showup 시간 계산
    showup_time = showup_cnt * showup_hour
    showup_sal = flight_per_sal * showup_time
    
    # 최종
    total_cost = flight_total_sal+ layover_sal+ showup_sal+ base_sal+ deadhead_cost

    analysis.loc[i] = [index, flight_total_time, layover_time, showup_time, deadhead_time, start_time, end_time, total_time, flight_total_sal, layover_sal, showup_sal, base_sal,deadhead_sal, deadhead_cost, total_cost]
    analysis.to_csv('analysis_base_5m.csv',index=False)