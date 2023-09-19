
class Aircraft:
    def __init__(self, id, type, crewNum, flightCost, layoverCost, quickTurnCost):
        self.id = id
        self._type = type
        self._crewNum = crewNum
        self._flightCost = flightCost
        self._layoverCost = layoverCost
        self._quickTurnCost = quickTurnCost
        # self.name = "Aircraft - " + type # 인스턴스 이름

    def find_aircraft(self, aircrafts, name):
        aircraft = next(
            (temp for temp in aircrafts if temp['type'] == name), None)
        if aircraft is None:
            raise ValueError("Aircraft not found")
        return aircraft

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        self._type = value

    @property
    def crewNum(self):
        return self._crewNum

    @crewNum.setter
    def crewNum(self, value):
        self._crewNum = value

    @property
    def flightCost(self):
        return self._flightCost

    @flightCost.setter
    def flightCost(self, value):
        self._flightCost = value

    @property
    def layoverCost(self):
        return self._layoverCost

    @layoverCost.setter
    def layoverCost(self, value):
        self._layoverCost = value

    @property
    def quickTurnCost(self):
        return self._quickTurnCost

    @quickTurnCost.setter
    def quickTurnCost(self, value):
        self._quickTurnCost = value

    def __str__(self):
        return f"Aircraft - {self.type}"
