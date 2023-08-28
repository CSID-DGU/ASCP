from Aircraft import Aircraft
from Airport import Airport
from Flight import Flight
from Pairing import Pairing
from IDProvider import IDProvider
import pandas as pd


class ASCPFactory:

    idprovider=IDProvider()

    @staticmethod
    def createAircraft(excel_path, sheet_name='Program_Cost'):
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=2) # Input_Data 엑셀 파일의 Program_Cost 시트를 읽어와, DataFrame으로 만듦
        aircraftList = [] # 객체화된 aircraft들을 담을 리스트 선언
        for idx, row in df.iterrows():
            aircraft = Aircraft(
                id=ASCPFactory.idprovider.get_aircraft_id(), # 싱글톤인 idprovider 호출하여 aircraft에 id 부여
                type=row['AIRCRAFT'],
                crewNum=row['CREW_NUM(명)'],
                flightCost=row['Flight Cost(원/분)'],
                layoverCost=row['Layover Cost(원/분)'],
                quickTurnCost=row['Quick Turn Cost(원/회)']
            )
            aircraftList.append(aircraft)
        return aircraftList
    
    @staticmethod
    def createAirport(excel_path, sheet_name='User_Hotel'):
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=2) # Input_Data 엑셀 파일의 User_Hotel 시트를 읽어와, DataFrame으로 만듦
        airportList = []
        for idx, row in df.iterrows():
            airport = Airport(
                id=ASCPFactory.idprovider.get_airport_id(), # 싱글톤인 idprovider 호출하여 airport에 id 부여
                name=row['공항 Code'],
                hotelCost=row['비용(원)'],
                # deadheadCost=row['DeadheadCost']  잘 몰르겠음. 질문 필요
            )
            airportList.append(airport)
        return airportList
    
    @staticmethod
    def createFlight(excel_path, sheet_name='User_Flight'):
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=2) # Input_Data 엑셀 파일의 User_Flight 시트를 읽어와, DataFrame으로 만듦
        flightList = []
        for idx, row in df.iterrows():
            flight = Flight(
                id=ASCPFactory.idprovider.get_flight_id(), # 싱글톤인 idprovider 호출하여 flight에 id 부여
                TailNumber=row['T/N'],
                originAirport=row['ORIGIN'],
                originTime=row['ORIGIN_DATE'],
                destAirport=row['DEST'],
                destTime=row['DEST_DATE'],
                aircraft=row['AIRCRAFT_TYPE']
            )
            flightList.append(flight)
        return flightList
    
    @staticmethod
    def createPairing(excel_path, flightList):
        pairingList = [] # Pairing Set을 저장할 변수
        df = pd.read_excel(excel_path,header=1) # 옵타플래너를 돌려 나온 output.xlsx를 읽어옴
        del df['pairing index'] # 엑셀에서 필요 없는 한 줄(pairing index) 삭제
        max_flight = len(df.columns) # 엑셀이 가지고있는, 전체 Pairing 중 가장 Flight 수가 많은 Pairing의 Flight 수 
        column = list(range(1, max_flight+1)) # column=[1,2, ... ,max_flight]
        column_list = [str(item) for item in column] 
        df.columns = column_list # 읽어온 엑셀파일의 column을 위와 같이 1,2,3,,,, maxflight로 바꾸어줌
        df = df.fillna(-1) # Nan 자리는 -1로 채움
        pairing_matrix = df.values # DataFrame을 2차원 배열로 변환

        m,n=pairing_matrix.shape # m은 행의 수(pairing의 수), n은 열의 수(max_flight). 

        for i in range(0,m): # 행의 개수만큼(pairing의 수 만큼) 반복함
            flightIdList=pairing_matrix[i] # 하나의 pair씩 순차적으로 선택하여 flightIdList에 넣어줌. 리스트는 output 파일의 flight ID로 구성.
            pair=[] # ID가 아닌, 실제 Flight 객체가 들어갈 리스트인 pair 선언
            
            for j in range(0,len(flightIdList)): 
                for k in range(0,len(flightList)): # Flight List의 id 하나하나와, flightIdList의 id를 비교함
                    tempFlight = flightList[k].findById(flightIdList[j]) # id가 같은게 발견될 시 해당 flight 객체, 아닐 시 None
                    if tempFlight is not None: # 만약 id가 같은게 발견됐다면
                        pair.append(tempFlight) # 해당 Flight 객체를 pair에 append하고 for문 하나'만'(FlightList에서 찾는 반복문) 탈출
                        break
            pairing=Pairing(
                id=ASCPFactory.idprovider.get_pairing_id(), # pairing에 id 부여
                pair=pair, # List['Flight'] 형태의 pair
                totalCost=0 # total Cost
            )
            pairingList.append(pairing) # pairingList에, 만들어진 pairing을 넣어줌
        return pairingList
