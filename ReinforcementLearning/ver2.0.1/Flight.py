from datetime import datetime
from datetime import timedelta


class Flight:
    def __init__(self, idx, TailNumber, originAirport, originTime, destAirport, destTime, aircraft):
        self.id = idx
        self._TailNumber = TailNumber
        self._originAirport = originAirport
        self._destAirport = destAirport
        
        if isinstance(originTime, str):
            self._originTime = datetime.strptime(
                originTime, '%Y-%m-%d %H:%M:%S')
        else:  # Assuming it's a Timestamp object
            self._originTime = originTime

        if isinstance(destTime, str):
            self._destTime = datetime.strptime(
                destTime, '%Y-%m-%d %H:%M:%S')
        else:  # Assuming it's a Timestamp object
            self._destTime = destTime
        self._duration = int(timedelta.total_seconds(
            self._destTime - self._originTime) // 60)
        self._aircraft = aircraft

    # flight 객체를 vector로 변환
    def toVector(self,airport_total,aircraft_total):
        
        parsed_origin_time = int((self._originTime-datetime(2022, 1, 1)).total_seconds() // 60)
        parsed_dest_time = int((self._destTime-datetime(2022, 1, 1)).total_seconds() // 60)
        # A_total만큼 0으로 채워진 리스트 생성
        airport_origin_onehot = [0 for i in range(len(airport_total))]
        airport_dest_onehot = [0 for i in range(len(airport_total))]
        # A_origin의 공항을 A_total의 인덱스로 변환
        for i, airport in enumerate(airport_total):
            if airport == self._originAirport:
                airport_origin_onehot[i] = 1
            if airport == self._destAirport:
                airport_dest_onehot[i] = 1
        # Aircraft_total만큼 0으로 채워진 리스트 생성
        aircraft_onehot = [0 for i in range(len(aircraft_total))]
        # Aircraft의 항공기를 Aircraft_total의 인덱스로 변환
        for i, aircraft in enumerate(aircraft_total):
            if aircraft == self._aircraft:
                aircraft_onehot[i] = 1
        return [parsed_origin_time, parsed_dest_time, self._duration, airport_origin_onehot, airport_dest_onehot, aircraft_onehot]
            
    
    # originTime 기준으로 정렬(originTime이 같다면 destTime 기준으로 정렬)
    def __lt__(self, other):
        if self._originTime == other._originTime:
            return self.destTime < other.destTime
        return self._originTime < other._originTime

    def findByID(self, id):
        if self.id == id:
            return self
        else:
            return None
        
    @property
    def TailNumber(self):
        return self._TailNumber

    @TailNumber.setter
    def TailNumber(self, value):
        self._TailNumber = value

    @property
    def originAirport(self):
        return self._originAirport

    @originAirport.setter
    def originAirport(self, value):
        self._originAirport = value

    @property
    def originTime(self):
        return self._originTime

    @originTime.setter
    def originTime(self, value):
        self._originTime = value

    @property
    def destAirport(self):
        return self._destAirport

    @destAirport.setter
    def destAirport(self, value):
        self._destAirport = value

    @property
    def destTime(self):
        return self._destTime

    @destTime.setter
    def destTime(self, value):
        self._destTime = value

    @property
    def aircraft(self):
        return self._aircraft

    @aircraft.setter
    def aircraft(self, value):
        self._aircraft = value

    @property
    def duration(self):
        return self._duration

    def getIndex(self):
        return "F" + str(self.id)

    def __str__(self):
        return "F" + str(self.id)
