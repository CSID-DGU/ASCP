class IDProvider:  # id는 중복되면 안되므로, id를 부여하는 객체는 단 하나만 존재해야함. => 싱글톤 패턴 사용. *****id는 0부터 시작
    _instance = None  # 싱글톤 인스턴스를 저장할 클래스 변수.

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(IDProvider, cls).__new__(cls)
            cls._instance.init()
        return cls._instance

    def init(self):
        self.flight_id = 0


    def get_flight_id(self):
        result = self.flight_id
        self.flight_id = self.flight_id + 1
        return result
