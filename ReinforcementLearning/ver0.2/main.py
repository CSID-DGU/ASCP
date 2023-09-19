# RL 모듈 import
import pandas as pd
import numpy as np
import sys
from Factory import ASCPFactory
from CrewPairingEnv import CrewPairingEnv
from PolicyGradientAgent import PolicyGradientAgent

# arguments reading (if exists)
# argv(data/)

# 실행 방법:
# source /home/public/yunairline/ASCP/ReinforcementLearning/.env/bin/activate
# python /home/public/yunairline/ASCP/ReinforcementLearning/ver0.2/main.py /home/public/yunairline/ASCP/ReinforcementLearning/dataset/ASCP_Data_Input_new.xlsx /home/public/yunairline/ASCP/ReinforcementLearning/dataset/output.xlsx


# if (len(sys.argv) > 0):
#    dataDirPath = sys.argv[0]
# else:
#    dataDirPath = None


# argv(crewpairing/)
# (len(sys.argv) > 1):
#    dataDirName = sys.argv[1]
# else:
#    dataDirName = None


# argv: inputFileName
if (len(sys.argv) > 0):
    informationXlsxFile = sys.argv[1]
else:
    informationXlsxFile = None

# argv: pairingFileName
if (len(sys.argv) > 1):
    pairingXlsxFile = sys.argv[2]
else:
    pairingXlsxFile = None

# aircraft, airport, flight, pairing 항목에 ID 부여
# airport, aircraft, flight 정보는 사용자가 직접 넣은 input 데이터를 활용한다
aircraftList = ASCPFactory.createAircraft(informationXlsxFile)
airportList = ASCPFactory.createAirport(informationXlsxFile)
flightList = ASCPFactory.createFlight(informationXlsxFile)

# pairingList vs. flightIdList vs. flightList
# createPairing 객체에는 pairingXlsxFile, 즉 옵타의 결과 파일을 전달
pairingList = np.array(ASCPFactory.createPairing(pairingXlsxFile, flightList))
# 옵타가 먼저 실행되므로 pairingXlssxFile을 항상 받을 수 있다.

max_flight = len((pd.DataFrame(pairingList)).columns)
print(max_flight)  # 로그 출력용

# 강화학습 환경/에이전트 설정
env = CrewPairingEnv(pairingList, max_flight)
agent = PolicyGradientAgent(env)

# 학습단계. 에피소드를 반복하며 정책 학습시킴.
pairing_set = []
for i_episode in range(10):
    print('episode : ', i_episode)
    # state -> Int type, 페어링셋을 1차원으로 만들어서 반환. pairing_set: flatten된 1차원 pairing 집합
    state = env.reset()
    done = False
    trycnt = 0
    while done == False:
        trycnt += 1

        print('trycnt : ', trycnt)
        if trycnt > 50:  # step이 무한루프에 빠지는 것을 방지.
            break
        # before_idx: 기존에 랜덤으로 뽑은 스테이트 / after_idx: 확률 계산을 통해 해당 인덱스와 위치를 바꿔볼 플라이트 인덱스
        before_idx, after_idx = agent.select_action(state)
        state, reward, done, _ = env.step(
            before_idx, after_idx, state)  # step() 호출
        print('done : ', done, 'reward : ', reward, end='\n\n')

    agent.rewards.append(reward)
    agent.update()


# 테스트 단계. 고정된 정책으로 변경된 pairing_set 구함.

# 테스트 에피소드의 수 설정
test_episodes = 5
maxRewardPairingSet = []
totalRewardHist = []

# 정책은 이미 학습된 상태이므로 별도로 업데이트하지 않음
for i_episode in range(test_episodes):
    state = env.reset()  # 환경을 초기 상태로 리셋
    done = False
    total_reward = 0  # 하나의 에피소드 동안 얻은 총 보상
    trycnt = 0  # 무한루프 방지를 위한 카운터

    while not done:
        trycnt += 1
        if trycnt > 50:  # 무한루프에 빠지는 것을 방지
            break

        # 학습된 정책을 사용하여 행동을 선택
        before_idx, after_idx = agent.select_action(state)
        # 환경에서 행동을 실행하고 새로운 상태와 보상을 얻음
        state, reward, done, _ = env.step(before_idx, after_idx, state)
        total_reward += reward  # 총 보상에 추가

    print(f"Total reward for test episode {i_episode + 1}: {total_reward}\n")
    if len(totalRewardHist) == 0 or totalRewardHist[-1] < total_reward:
        maxRewardPairingSet = state
    totalRewardHist.append(total_reward)

reshape_list = env.reshape_list(
    maxRewardPairingSet.tolist(), num_rows=215, num_columns=4)

hard_score = 0
soft_score = 0
for pair in reshape_list:
    pair_hard_score, pair_soft_score = env.calculateScore(pair)
    hard_score += pair_hard_score
    soft_score += pair_soft_score

# print('hard score : ', hard_score, 'soft score : ', soft_score)

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
df.to_excel(
    '/home/public/yunairline/ASCP/ReinforcementLearning/output/output.xlsx', index=False)
# 파일 출력 시 Main.java의 .getinputstream()이 감지해서 읽어들인다..?

# manager-program과의 연동을 위해 표준입출력(stream) 타입으로 출력해줄 필요 있음
# df.to_csv()
