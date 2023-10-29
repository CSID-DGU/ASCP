
import pandas as pd
from domain.factory.IDProvider import IDProvider
from domain.Aircraft import Aircraft
from domain.Airport import Airport
from domain.Flight import Flight
from domain.Pairing import Pairing


class Factory:

    idprovider = IDProvider()
    dummyAircraft=Aircraft(id=-1, type='xxx',crewNum=0,flightCost=0,layoverCost=0,quickTurnCost=0) # dummyFlight에 부여해줄 dummyCraft 생성
    dummyAirport = Airport(id=-1, name='xxx', hotelCost=0, destCostList={} ) # dummyFlight에 부여해줄 dummyAirport 생성
    dummyFlight = Flight(id=-1, originAirport=dummyAirport, originTime='2000-01-01 00:00:00', destAirport=dummyAirport,
                         destTime='2000-01-01 00:00:00', aircraft=dummyAircraft, TailNumber='')  # 페어링에서 빈자리를 차지할 id가 -1인 플라이트 생성

    @staticmethod
    def createAircraft(excel_path):
        # Input_Data 엑셀 파일의 Program_Cost 시트를 읽어와, DataFrame으로 만듦
        df = pd.read_csv(excel_path+'/Program_Cost.csv', header=1)
        aircraftList = []  # 객체화된 aircraft들을 담을 리스트 선언
        for idx, row in df.iterrows():
            aircraft = Aircraft(
                # 싱글톤인 idprovider 호출하여 aircraft에 id 부여
                id=Factory.idprovider.get_aircraft_id(),
                type=row['AIRCRAFT'],
                crewNum=row['CREW_NUM(명)'],
                flightCost=row['Flight Cost(원/분)'],
                layoverCost=row['Layover Cost(원/분)'],
                quickTurnCost=row['Quick Turn Cost(원/회)']
            )
            aircraftList.append(aircraft)
        return aircraftList

    @staticmethod
    def createAirport(excel_path):
        # Input_Data 엑셀 파일의 User_Hotel 시트를 읽어와, DataFrame으로 만듦
        df = pd.read_csv(excel_path+'/User_Hotel.csv', header=2)
        df2 = pd.read_csv(excel_path+'/User_Deadhead.csv', header=2)
        airportList = []
        for idx, row in df.iterrows():
            origin = row['공항 Code']
            destCostList = []
            for idxx, roww in df2.iterrows():
                if roww['출발 공항'] == origin:
                    destCostList.append([roww['도착 공항'], roww['Deadhead(원)']])
            airport = Airport(
                # 싱글톤인 idprovider 호출하여 airport에 id 부여
                id=Factory.idprovider.get_airport_id(),
                name=row['공항 Code'],
                hotelCost=row['비용(원)'],
                destCostList=destCostList
            )
            airportList.append(airport)
        return airportList

    @staticmethod
    def createFlight(excel_path, airportList=None, aircraftList=None):
        # Input_Data 엑셀 파일의 User_Flight 시트를 읽어와, DataFrame으로 만듦
        df = pd.read_csv(excel_path+'/User_Flight.csv', header=2)
        flightList = []
        for idx, row in df.iterrows():
            origin_airport_obj = next(
                (airport for airport in airportList if airport.name == row['ORIGIN']), None)  # Airport 객체 찾기
            dest_airport_obj = next(
                (airport for airport in airportList if airport.name == row['DEST']), None)  # Airport 객체 찾기
            aircraft_obj = next(
                (aircraft for aircraft in aircraftList if aircraft.type == row['AIRCRAFT_TYPE']), None)
            flight = Flight(
                id=Factory.idprovider.get_flight_id(),  # 싱글톤인 idprovider 호출하여 flight에 id 부여
                TailNumber=row['T/N'],
                originAirport=origin_airport_obj,
                originTime=row['ORIGIN_DATE'],
                destAirport=dest_airport_obj,
                destTime=row['DEST_DATE'],
                aircraft=aircraft_obj
            )
            flightList.append(flight)
        flightList.append(Factory.dummyFlight)
        return flightList

    @staticmethod
    def createPairing(excel_path, flightList, max_flight):
        pairingList = []  # Pairing Set을 저장할 변수
        # 옵타플래너를 돌려 나온 output.xlsx를 읽어옴
        df = pd.read_excel(excel_path)
        del df['Pairing Data']  # 엑셀에서 필요없는 행 Pairing Data 삭제
        # 엑셀이 가지고있는, 전체 Pairing 중 가장 Flight 수가 많은 Pairing의 Flight 수
        # max_flight = len(df.columns)

        # 현재 df의 열 수를 확인
        current_columns = len(df.columns)

        # max_flight개로 늘리기 위한 열 수 계산
        num_columns_to_add = max_flight - current_columns
        if max_flight>current_columns:
            # 각 열을 -1로 초기화하면서 df에 추가
            for _ in range(num_columns_to_add):
                df[str(current_columns + 1)] = -1
                current_columns += 1

        # 열 이름 설정
        df.columns = [str(i) for i in range(1, max_flight+1)]




        #column = list(range(1, max_flight+1))  # column=[1,2, ... ,max_flight]   column=[1,2,3, ... , 10]
        #column_list = [str(item) for item in column]
        #df.columns = column_list  # 읽어온 엑셀파일의 column을 위와 같이 1,2,3,,,, maxflight로 바꾸어줌
        df = df.fillna(-1)  # Nan 자리는 -1로 채움
        pairing_matrix = df.values  # DataFrame을 2차원 배열로 변환
        # m은 행의 수(pairing의 수), n은 열의 수(max_flight).
        m, n = pairing_matrix.shape

        for i in range(0, m):  # 행의 개수만큼(pairing의 수 만큼) 반복함
            # 하나의 pair씩 순차적으로 선택하여 flightIdList에 넣어줌. 리스트는 output 파일의 flight ID로 구성.
            flightIdList = pairing_matrix[i]
            pair = []  # ID가 아닌, 실제 Flight 객체가 들어갈 리스트인 pair 선언

            for j in range(0, len(flightIdList)):
                # Flight List의 id 하나하나와, flightIdList의 id를 비교함
                tempFlight = None
                for k in range(0, len(flightList)):
                    # id가 같은게 발견될 시 해당 flight 객체, 아닐 시 None
                    tempFlight = flightList[k].findByID(flightIdList[j])
                    if tempFlight is not None:  # 만약 id가 같은게 발견됐다면
                        # 해당 Flight 객체를 pair에 append하고 for문 하나'만'(FlightList에서 찾는 반복문) 탈출
                        pair.append(tempFlight)
                        break
            pairing = Pairing(
                id=Factory.idprovider.get_pairing_id(),  # pairing에 id 부여
                pair=pair,  # List['Flight'] 형태의 pair
            )
            pairingList.append(pairing)  # pairingList에, 만들어진 pairing을 넣어줌
        # pairing마다의 Score 계산을 위한 pairingList, 강화학습에서 step을 진행하기 위한 pairing_matrix
        return pairingList, pairing_matrix
