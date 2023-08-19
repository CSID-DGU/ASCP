import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os

# FlightDataGenerator.py 파일의 상위 경로를 가져옵니다.
current_dir = os.path.dirname(__file__)
main_dir = os.path.dirname(current_dir)

# 상대 경로를 기반으로 파일을 불러옵니다.
btsdata_file_path = os.path.join(main_dir, "dataset", "flightdata", "input", "T_ONTIME_MARKETING.csv")
sfodata_file_path = os.path.join(main_dir, "dataset", "flightdata", "input", "tailnumTocraft.csv")

# csv_file_path를 사용하여 파일을 처리합니다.
btsdata = pd.read_csv(btsdata_file_path)
sfodata = pd.read_csv(sfodata_file_path)

# 각 데이터에서 불필요한 열 삭제
del btsdata['OP_UNIQUE_CARRIER']
sfodata = sfodata[['Tail Number','Aircraft Model']]
sfodata.rename(columns = {'Tail Number':'TAIL_NUM', 'Aircraft Model':'AIRCRAFT_MODEL'},inplace = True)

# TAIL_NUM을 기준으로 두 데이터 합치기
mergeddata = pd.merge(btsdata, sfodata, on = 'TAIL_NUM', how='inner')

converteddata = mergeddata.copy()

# FL_DATE 열의 데이터를 datetime 자료형으로 변환
converteddata['FL_DATE'] = pd.to_datetime(converteddata['FL_DATE'])

# CRS_DEP_TIME과 CRS_ARR_TIME 열을 합쳐서 DEP_TIME과 ARR_TIME 열 생성
def convert_to_time_string(time):
    time_str = str(time)
    return f"{time_str[:-2]}{time_str[-2:]}"

# 함수를 정의하여 시간 데이터를 시간 형식으로 변환
def convert_to_time(time_data):
    time_str = str(time_data).zfill(4)  # 4자리 숫자로 맞추기 위해 0으로 채우기
    hour = int(time_str[:2])
    minute = int(time_str[2:])
    return timedelta(hours=hour, minutes=minute)

converteddata['DEP_TIME'] = converteddata['FL_DATE'] + converteddata['CRS_DEP_TIME'].apply(convert_to_time)

converteddata['DEP_TIME'] = converteddata['DEP_TIME'].apply(convert_to_time_string)
converteddata['DEP_TIME'] = pd.to_datetime(converteddata['DEP_TIME'])

converteddata['ARR_TIME'] = converteddata['FL_DATE'] + converteddata['CRS_ARR_TIME'].apply(convert_to_time)
converteddata['ARR_TIME'] = converteddata['ARR_TIME'].apply(convert_to_time_string)
converteddata['ARR_TIME'] = pd.to_datetime(converteddata['ARR_TIME'])

# ARR_TIME이 DEP_TIME보다 작은 경우, 날짜를 하루 늘리기
converteddata.loc[converteddata['ARR_TIME'] < converteddata['DEP_TIME'], 'ARR_TIME'] += timedelta(days=1)

# 필요없는 중간 열 삭제
converteddata.drop(['FL_DATE', 'CRS_DEP_TIME', 'CRS_ARR_TIME'], axis=1, inplace=True)

# DEP_TIME과 ARR_TIME 자료형을 datetime으로 변환
converteddata['DEP_TIME'] = pd.to_datetime(converteddata['DEP_TIME'])
converteddata['ARR_TIME'] = pd.to_datetime(converteddata['ARR_TIME'])

converteddata['ELAPSED_TIME'] = converteddata['CRS_ELAPSED_TIME']/60
# 필요없는 중간 열 삭제
converteddata.drop(['CRS_ELAPSED_TIME'], axis=1, inplace=True)

# 기간 입력
def get_valid_datetime_input(prompt):
    while True:
        try:
            datetime_input = datetime.strptime(input(prompt), "%Y-%m-%d %H:%M:%S")
            return datetime_input
        except ValueError:
            print("입력 형식이 잘못되었습니다. 다시 입력해주세요.")
            continue

print("#############    원하는 비행의 기간을 입력해주세요   #############")
print('데이터 시작 일시 :',converteddata['DEP_TIME'].min())
print('데이터 종료 일시 :',converteddata['ARR_TIME'].max())
print("입력 예시: 2023-04-01 00:13:00",end='\n\n')
start_cutoff = get_valid_datetime_input("✒ 시작 일시: ")
end_cutoff = get_valid_datetime_input("✒ 종료 일시: ")

dateFdata = converteddata.loc[(converteddata['DEP_TIME'] > start_cutoff) & (converteddata['ARR_TIME'] < end_cutoff)]

# 기종 입력
craftFdata = dateFdata.copy()
craftlist = list(craftFdata['AIRCRAFT_MODEL'].value_counts().index)
craftdict = {}
for index, element in enumerate(sorted(craftlist), start=1):
  craftdict[index] = element

print('\n\n')
print("#############    원하는 비행기의 기종을 입력해주세요   #############")
print("🛫 기종 목록 🛫")
print(craftdict)
craftnum = [int(x) for x in input("\n✒ 항공사의 번호를 입력해주세요 ex) 1 4 5 : ").split()]
selectedcraft = [craftdict[key] for key in craftnum]

craftFdata = craftFdata[craftFdata['AIRCRAFT_MODEL'].isin(selectedcraft)]

# 공항 입력
portFdata = craftFdata.copy()
print('\n\n')
print("#############    원하는 공항의 종류을 입력해주세요   #############")
top_n = int(input("✒ 상위 몇개의 공항을 확인하시겠습니까? : "))
print()
print("🛫 출발 공항 개수 🛫")
originPort = pd.DataFrame(portFdata['ORIGIN'].value_counts()).head(top_n)
originPort.reset_index(inplace=True)
originPort.rename(columns={'index':'PORT','ORIGIN':'#'}, inplace=True)
print(originPort)
print()
print("🛫 도착 공항 개수 🛫")
destPort = pd.DataFrame(portFdata['DEST'].value_counts()).head(top_n)
destPort.reset_index(inplace=True)
destPort.rename(columns={'index':'PORT','DEST':'#'}, inplace=True)
print(destPort)

# 공항을 숫자로 제시하기
portset = set(np.concatenate((originPort['#'].values,destPort['#'].values)))
portdict = {}

for index, element in enumerate(sorted(portset), start=1):
  portdict[index] = element

print("🛫 공항 목록 🛫")
print(portdict)
portnum = [int(x) for x in input("\n✒ 공항의 번호를 입력해주세요 ex) 1 4 5 : ").split()]
selectedport = [portdict[key] for key in portnum]

# ORIGIN 열의 값이 selectedport 리스트에 없는 행 삭제
portFdata = portFdata[portFdata['ORIGIN'].isin(selectedport)]

# DEST 열의 값이 selectedport 리스트에 없는 행 삭제
portFdata = portFdata[portFdata['DEST'].isin(selectedport)]

# 항공사 입력
carrierFdata = portFdata.copy()
carrierlist = list(carrierFdata['MKT_UNIQUE_CARRIER'].value_counts().index)
carrierdict = {}
for index, element in enumerate(sorted(carrierlist), start=1):
  carrierdict[index] = element

print('\n\n')
print("#############    원하는 항공사의 종류을 입력해주세요   #############")
print("🛫 항공사 목록 🛫")
print(carrierdict)
carriernum = [int(x) for x in input("\n✒ 항공사의 번호를 입력해주세요 ex) 1 4 5 : ").split()]
selectedcarrier = [carrierdict[key] for key in carriernum]

carrierFdata = carrierFdata[carrierFdata['MKT_UNIQUE_CARRIER'].isin(selectedcarrier)]

# flight 수 입력
numFdata = carrierFdata.copy()
print('\n\n')
print("#############    비행의 수를 입력해주세요   #############")
print("현재 사용할 수 있는 flight의 수 :",len(numFdata))
num_flight = int(input("\n✒ 원하는 flight의 개수를 입력하세요 : "))
sampleddata = numFdata.sample(n=num_flight, random_state=42)

# 데이터 정보 확인
output_text = ""
output_text += "총 비행의 개수 : " + str(len(sampleddata)) + "\n"
output_text += "비행의 기간\n"
output_text += "비행 시작 일시 : " + str(sampleddata['DEP_TIME'].min()) + "\n"
output_text += "비행 종료 일시 : " + str(sampleddata['ARR_TIME'].max()) + "\n"
output_text += "항공기 기종 별 flight 개수\n" + str(sampleddata['AIRCRAFT_MODEL'].value_counts()) + "\n"
output_text += "출발/도착 공항 별 flight 개수\n" + str(sampleddata['ORIGIN'].value_counts()) + "\n" + str(sampleddata['DEST'].value_counts()) + "\n"
output_text += "항공사 별 flight 개수\n" + str(sampleddata['MKT_UNIQUE_CARRIER'].value_counts()) + "\n"

now = datetime.now()
today = now.strftime("%Y-%m-%d %H:%M:%S")

# 결과를 저장할 파일 경로 설정
output_txt_file_path = os.path.join(main_dir, "dataset", "flightdata","output", str(today)+"_flight_data_summary.txt")
output_csv_file_path = os.path.join(main_dir, "dataset", "flightdata","output", str(today)+"_flight_data.csv")

# 텍스트 파일에 결과 기록
with open(output_txt_file_path, "w") as output_file:
    output_file.write(output_text)

# csv 파일로 결과 저장
sampleddata.to_csv(output_csv_file_path,index=False)

print('\n\n')
print("#############    데이터 저장이 완료되었습니다   #############")
print("flight 데이터 요약 정보가", output_txt_file_path, "에 저장되었습니다.")
print("flight 데이터 csv가", output_csv_file_path, "에 저장되었습니다.")