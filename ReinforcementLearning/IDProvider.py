class IDProvider: # id는 중복되면 안되므로, id를 부여하는 객체는 단 하나만 존재해야함. => 싱글톤 패턴 사용. *****id는 0부터 시작
    _instance = None  # 싱글톤 인스턴스를 저장할 클래스 변수. 

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IDProvider, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.aircraft_id = 0
        self.airport_id = 0
        self.flight_id = 0
        self.pairing_id = 0

    def get_aircraft_id(self): 
        result= self.aircraft_id
        self.aircraft_id=self.aircraft_id+1
        return result

    def get_airport_id(self):
        result= self.airport_id
        self.airport_id = self.airport_id + 1
        return result

    def get_flight_id(self):
        result= self.flight_id
        self.flight_id = self.flight_id + 1
        return result

    def get_pairing_id(self):
        result= self.pairing_id
        self.pairing_id = self.pairing_id + 1
        return result
