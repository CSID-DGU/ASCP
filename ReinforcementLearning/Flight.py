from datetime import datetime
from datetime import timedelta

class Flight:
    def __init__(self, id, TailNumber, originAirport, originTime, destAirport, destTime, aircraft):
        self.id = id
        self._TailNumber = TailNumber
        self._originAirport = originAirport
        self._originTime = originTime
        self._destAirport = destAirport
        self._destTime = destTime
        self._aircraft = aircraft
        self._flightTime = int((self._destTime - self._originTime).total_seconds() // 60)
    
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
    def flightTime(self):
        return self._flightTime

    def getIndex(self):
        return "F" + str(self.id)

    def __str__(self):
        return "F" + str(self.id)
