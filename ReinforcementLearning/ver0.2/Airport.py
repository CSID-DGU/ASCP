class Airport:
    # deadheadCost = dict('name2', 'deadhead') # dictionary 자료형, name/deadhead로 인덱싱

    def __init__(self, id, name, hotelCost):
        self.id = id
        self._name = name  # 공항 이름
        self._hotelCost = hotelCost

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

    # @property
    # def deadheadCost(self):
    #   return self.deadheadCost

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
