from datetime import datetime
from datetime import timedelta


class Flight:
    def __init__(self, id, TailNumber, originAirport, originTime, destAirport, destTime, aircraft):
        self.id = id
        self._TailNumber = TailNumber
        self._originAirport = originAirport

        if isinstance(originTime, str):
            self._originTime = datetime.strptime(
                originTime, '%Y-%m-%d  %I:%M:%S %p').strftime('%m/%d/%y %H:%M')
        else:  # Assuming it's a Timestamp object
            self._originTime = originTime.strftime('%m/%d/%y %H:%M')

        self._destAirport = destAirport

        if isinstance(destTime, str):
            self._destTime = datetime.strptime(
                destTime, '%Y-%m-%d  %I:%M:%S %p').strftime('%m/%d/%y %H:%M')
        else:  # Assuming it's a Timestamp object
            self._destTime = destTime.strftime('%m/%d/%y %H:%M')

        self._aircraft = aircraft
        self._flightTime = int(timedelta.total_seconds(datetime.strptime(
            self._destTime, '%m/%d/%y %H:%M') - datetime.strptime(self._originTime, '%m/%d/%y %H:%M')) // 60)

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
    def flightTime(self):
        return self._flightTime

    def getIndex(self):
        return "F" + str(self.id)

    def __str__(self):
        return "F" + str(self.id)
