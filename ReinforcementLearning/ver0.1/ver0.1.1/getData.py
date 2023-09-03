import pandas as pd


def getInputData():
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

    return input_flight, input_deadhead, input_salary
