
class Aircraft:
  def __init__(self, id, type, crewNum, flightCost, layoverCost, quickTurnCost):
    self.id = id
    self.type = type
    self.crewNum = crewNum
    self.flightCost = flightCost
    self.layoverCost = layoverCost
    self.quickTurnCost = quickTurnCost 
    # self.name = "Aircraft - " + type # 인스턴스 이름

  def find_aircraft(aircrafts, name):
    aircraft = next((temp for temp in aircrafts if temp['type'] == name), None)
    if aircraft is None:
        raise ValueError("Aircraft not found")
    return aircraft
  
  @property
  def type(self):
    return self.type
  
  @type.setter
  def type(self, value):
    self.type = value
    
  @property
  def crewNum(self):
    return self.crewNum
  
  @crewNum.setter
  def crewNum(self, value):
    self.crewNum = value
    
  @property
  def flightCost(self):
    return self.flightCost
  
  @flightCost.setter
  def flightCost(self, value):
    self.flightCost = value
    
  @property
  def layoverCost(self):
    return self.layoverCost
  
  @layoverCost.setter
  def layoverCost(self, value):
    self.layoverCost = value
    
  @property
  def quickTurnCost(self):
    return self.quickTurnCost
  
  @quickTurnCost.setter
  def quickTurnCost(self, value):
    self.quickTurnCost = value
  
  def __str__(self):
    return f"Aircraft - {self.type}"