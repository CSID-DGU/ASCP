# RL 모듈 import
import sys
import pandas as pd
import numpy as np
import openpyxl as openxl
from datetime import datetime
from domain.factory.Factory import Factory
from env.CrewPairingEnv import CrewPairingEnv
from agent.PolicyGradientAgent import PolicyGradientAgent


# argv: inputFileDirectory
if (len(sys.argv) > 1):
    informationCsvFileDirectory = sys.argv[1]
else:
    informationCsvFileDirectory = None
# argv: inputFileName
if (len(sys.argv) > 2):
    inputFileName = sys.argv[2]
else:
    inputFileName = None

# argv: pairingFileName
if (len(sys.argv) > 3):
    pairingXlsxFile = sys.argv[3]
else:
    pairingXlsxFile = None


xls = pd.ExcelFile(informationCsvFileDirectory + inputFileName)  # 엑셀 파일의 모든 시트 읽기
sheet_names = xls.sheet_names  # 엑셀 파일 안의 각 시트 이름을 리스트형태(?)로 가져옴

for sheet_name in sheet_names:  # 각 시트를 순회하며 csv파일로 저장
    df = pd.read_excel(informationCsvFileDirectory+inputFileName,
                       sheet_name=sheet_name)
    csv_filename = f"{'/home/public/ascp_prod/data/crewpairing/input'}/{sheet_name}.csv"
    df.to_csv(csv_filename, index=False)


# aircraft, airport, flight, pairing 항목에 ID 부여
# airport, aircraft, flight 정보는 사용자가 직접 넣은 input 데이터를 활용한다

aircraftList = Factory.createAircraft(informationCsvFileDirectory)
airportList = Factory.createAirport(informationCsvFileDirectory)
flightList = Factory.createFlight(
    informationCsvFileDirectory, airportList, aircraftList)

# pairingList vs. flightIdList vs. flightList
# createPairing 객체에는 pairingXlsxFile, 즉 옵타의 결과 파일을 전달

max_flight=10
pairingList, pairing_matrix = Factory.createPairing(
    pairingXlsxFile, flightList,max_flight)


# 옵타가 먼저 실행되므로 pairingXlssxFile을 항상 받을 수 있다.

#max_flight = len((pd.DataFrame(pairing_matrix)).columns)


# 강화학습 환경/에이전트 설정
env = CrewPairingEnv(pairing_matrix, pairingList, max_flight)
agent = PolicyGradientAgent(env)

"""
하나의 episode에서 하는 일
1. 랜덤하게 flihgt를 선택해서 state를 받음
2. step 진행하여 갱신이 있을 때까지 반복
3. 받은 reward를 저장
4. step이 끝나면 저장한 reward를 통해 policy를 업데이트
"""

tempSoftScore = 0
tempHardScore = 0
tempDf = None

i_episode = 0

while True:
    print('episode : ', i_episode, end='\t')

    # state reset하기
    state_history = []
    # 리셋하는 부분이 꼭 필요함.!!! -> 여기서 변경된 페어링 샛을 기준으로 새로운 state를 생성할 수 있도록 하고 episode가 끝나면 다시 리셋
    # 근데 이부분 구현 안되어있는것같음 - 동겸
    state = env.reset()
    state_history.append(state)  # 한번 선택한 비행은 다시 선택하지 않도록 state_history에 저장
    done = False

    # 하나의 step에서 하는 일
    # 1. state를 받아서 action을 선택
    # 2. 선택한 action을 통해 다음 state와 reward를 받음

    for t in range(1000):
        # print('step : ', t+1)
        state, action = agent.select_action(state)
        # available action이 없는 경우 action가 None이 됨 -> 이 경우에는 step을 종료
        if action == None:
            print('No available actions.')
            break

        # idx를 통해서 flight 정보 출력하기
        state = int(state)
        action = int(action)

        # step 진행 -> 진행하면 env.pairing_set이 갱신됨 -> 안되게 하려면 env.pairing_set을 복사해서 사용해야 함
        step_pairing_set, reward, done, _ = env.step(
            state, action)
        agent.rewards.append(reward)
        # env.pairingset은 done이 True인 경우 갱신된 pairingset으로 바뀜 False인 경우에는 기존의 pairingset을 유지

    agent.update()

    hard_score = 0.0
    soft_score = 0.0
    for pair in env.pairing_set:
        pair_hard_score, pair_soft_score = env.calculateScore(pair)
        hard_score += pair_hard_score
        soft_score += pair_soft_score
    tempSoftScore = soft_score
    # tempSoftScore = -tempSoftScore
    tempHardScore = hard_score
    print('hard score : ', hard_score, 'soft score : ', tempSoftScore)

   # 'pairing index'를 포함한 데이터프레임 생성
    df = pd.DataFrame(env.pairing_id_set)

    # 'pairing index' 열 추가
    df.insert(0, 'pairing index', range(len(df)))

    # -1 값을 빈 값으로 변경
    df = df.replace(-1.0, '')
    tempDf = df

    col_num = len(df.columns)
    score_list = ['soft', soft_score, 'hard', hard_score]
    while len(score_list) < col_num:
        score_list.extend([None])

    # 맨 위 행에 score값 추가
    df.index = df.index + 1
    df = df.sort_index()

    # 칼럼명과 첫번째 행 바꾸기
    tobe_col = df.iloc[0]
    tobe_first = df.columns.tolist()
    tobe_first = [np.nan if isinstance(
        x, (int, float)) else x for x in tobe_first]

    df.columns = tobe_col
    df.loc[0] = tobe_first
    
    if i_episode >= 200000:
        break
    
    i_episode += 1

now = datetime.now() #현재시간
now_str = now.strftime("%Y_%m_%d_%H_%M_%S")

# 파일 이름 설정
file_name = f"/home/public/ascp_prod/data/crewpairing/output/{now_str}-pairingData.xlsx"

# 데이터프레임을 엑셀 파일로 저장
tempDf.to_excel(file_name, sheet_name='Data', index=False)
print(f"Create Output File : {now_str}-pairingData.xlsx")
print("Hard Score : %d" % (tempHardScore))
print("Soft Score : %d" % (tempSoftScore))
