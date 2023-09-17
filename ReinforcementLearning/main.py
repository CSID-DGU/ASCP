# RL 모듈 import
import pandas as pd
import numpy as np
import sys
from Factory import ASCPFactory
import CrewPairingEnv
import PolicyGradientAgent

# arguments reading (if exists)
# argv(data/)
if (len(sys.argv) > 0):
    dataDirPath = sys.argv[0]
else:
    dataDirPath = None

# argv(crewpairing/)
if (len(sys.argv) > 1):
    dataDirName = sys.argv[1]
else:
    dataDirName = None
    
# argv: inputFileName
if (len(sys.argv) > 2):
    informationXlsxFile = sys.argv[2]
else:
    informationXlsxFile = None
    
# argv: pairingFileName
if (len(sys.argv) > 3):
    pairingXlsxFile = sys.argv[3]
else:
    pairingXlsxFile = None
    

# initial_pairing_set = pd.read_excel('/home/public/yunairline/ASCP/ReinforcementLearning/dataset/output.xlsx', header=1)

# 기존 pairing_matrix 생성 코드
# del initial_pairing_set['pairing index']
# max_flight = len(initial_pairing_set.columns)
# column = list(range(1, max_flight+1))
# column_list = [str(item) for item in column]
# initial_pairing_set.columns = column_list
# initial_pairing_set = initial_pairing_set.fillna(-1)
# pairing_matrix = initial_pairing_set.values # 2차원  배열

# aircraft, airport, flight, pairing 항목에 ID 부여
aircraftList = ASCPFactory.createAircraft(informationXlsxFile) # airport, aircraft, flight 정보는 사용자가 직접 넣은 input 데이터를 활용한다
airportList = ASCPFactory.createAirport(informationXlsxFile)
flightList = ASCPFactory.createFlight(informationXlsxFile)

pairingList = ASCPFactory.createPairing(pairingXlsxFile, flightList) # createPairing 객체에는 pairingXlsxFile, 즉 옵타의 결과 파일을 전달
# 옵타가 먼저 실행되므로 pairingXlssxFile을 항상 받을 수 있다.

###### test_code #######
# aircraftList = ASCPFactory.createAircraft() # 시험 input 데이터셋 경로 삽입
# airportList = ASCPFactory.createAirport()
# flightList = ASCPFactory.createFlight()

# initial_pairing_set = "/home/public/yunairline/ASCP/ReinforcementLearning/dataset/output.xlsx" # 시험할 페어링 셋 경로
# pairingList = ASCPFactory.createPairing(initial_pairing_set, flightList)


# 기존의 pairing_matrix 생성 코드
max_flight = len(pairingList.columns)
column = list(range(1, max_flight+1))
column_list = [str(item) for item in column]
pairingList.columns = column_list
pairingList = pairingList.fillna(-1)
pairing_matrix = pairingList.values # 2차원  배열

# 강화학습 환경/에이전트 설정
env = CrewPairingEnv(pairing_matrix)
agent = PolicyGradientAgent(env)

# episode 반복하는 부분
pairing_set = []
for i_episode in range(10):
    print('episode : ', i_episode)
    pairing_set, state = env.reset() # state -> Int type, 페어링셋을 1차원으로 만들어서 반환
    done = False
    trycnt = 0
    while done == False:
        trycnt += 1
        print('trycnt : ', trycnt)
        if trycnt > 50:
            step_pairing_set = pairing_set
            break
        before_idx, after_idx = agent.select_action(state) # before_idx: 기존에 랜덤으로 뽑은 스테이트 / after_idx: 확률 계산을 통해 해당 인덱스와 위치를 바꿔볼 플라이트 인덱스
        step_pairing_set, reward, done, _ = env.step(before_idx, after_idx) # step() 호출
        print('done : ', done, 'reward : ', reward, end='\n\n')
    
    # 개선이 안된 페어링셋이 들어가는 문제가 발생한다!!
    if trycnt <= 50:
        agent.rewards.append(reward)
        if np.any(pairing_set != step_pairing_set):
            print('!!!!!!!!!!! change pairing set !!!!!!!!!!!')
        pairing_set = step_pairing_set
        state = after_idx
        agent.update()

reshape_list = env.reshape_list(pairing_set.tolist(), num_rows=215, num_columns=4) 

hard_score = 0
soft_score = 0
for pair in reshape_list:
    pair_hard_score, pair_soft_score = env.calculateScore(pair)
    hard_score += pair_hard_score
    soft_score += pair_soft_score

#print('hard score : ', hard_score, 'soft score : ', soft_score)

# RL에선 신경 쓸 필요 없는 부분
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

###### test: 엑셀 파일로 저장 #######
df.to_excel('/home/public/yunairline/ASCP/ReinforcementLearning/output/output.xlsx', index=False)
# 파일 출력 시 Main.java의 .getinputstream()이 감지해서 읽어들인다..?

# manager-program과의 연동을 위해 표준입출력(stream) 타입으로 출력해줄 필요 있음
#df.to_csv()