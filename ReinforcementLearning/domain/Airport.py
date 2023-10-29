class Airport:
    # deadheadCost = dict('name2', 'deadhead') # dictionary 자료형, name/deadhead로 인덱싱

    def __init__(self, id, name, hotelCost, destCostList):
        self.id = id
        self._name = name  # 공항 이름
        self._hotelCost = hotelCost
        self.deadheadCost = {}  # 빈 딕셔너리로 초기화
        for destCost in destCostList:
            self.deadheadCost[destCost[0]] = destCost[1]

    def getDeadheadCost(self, dest):
        return self.deadheadCost[dest.name]

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def hotelCost(self):
        return self._hotelCost

    @hotelCost.setter
    def hotelCost(self, value):
        self._hotelCost = value

    # @deadheadCost.setter
    # def deadheadCost(self, value):
    #   self.deadheadCost = value

    # 인스턴스 이름 설정 (java에서 toString())

    def __str__(self):
        return f"Airport - {self.name}"

    def find_airport(self, airports, name):
        airport = next(
            (airport for airport in airports if airport.name == name), None)
        if airport is None:
            raise ValueError("Airport not found")
        return airport
