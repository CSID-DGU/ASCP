class Airport:
  # deadheadCost = dict('name2', 'deadhead') # dictionary 자료형, name/deadhead로 인덱싱
  
  def __init__(self, id, name, hotelCost, deadheadCost):
    self.id = id
    self.name = name # 공항 이름
    self.hotelCost = hotelCost
    self.deadheadCost = deadheadCost
    
  # def putDeadhead(name2, deadhead):
  #   deadheadCost = dict(name2, deadhead)
  #   deadheadCost[name2] = deadhead
    
  def putDeadhead(self, name, deadhead):
    self.deadheadCost[name] = deadhead
  
  @property
  def name(self):
    return self.name
  
  @name.setter
  def name(self, value):
    self.name = value
    
  @property
  def hotelCost(self):
    return self.hotelCost
  
  @hotelCost.setter
  def hotelCost(self, value):
    self.hotelCost = value
    
  # @property
  # def deadheadCost(self):
  #   return self.deadheadCost
  
  # @deadheadCost.setter
  # def deadheadCost(self, value):
  #   self.deadheadCost = value
    
    
  #인스턴스 이름 설정 (java에서 toString())
  def __str__(self):
    return f"Airport - {self.name}"
  
  def find_airport(airports, name):
    airport = next((airport for airport in airports if airport.name == name), None)
    if airport is None:
        raise ValueError("Airport not found")
    return airport
  