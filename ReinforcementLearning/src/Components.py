class Airport:
    graph = {}

    @classmethod
    def add_edge(cls, start, end, dh_cost):
        start, end = tuple(start), tuple(end)   #원-핫인코딩된 리스트 자료이기 때문에 튜플로 변경
        if start not in cls.graph:             #dictionary 구조로 start에 (end, cost) 매핑
            cls.graph[start] = []
        cls.graph[start].append((end, int(dh_cost)))

    @classmethod
    def get_cost(cls, start, end):
        start, end = tuple(start), tuple(end)
        for neighbor, cost in cls.graph[start]:
            if neighbor == end:
                return cost


class Aircraft:
    dic = {}
    
    @classmethod
    def add_type(cls, model, crew_num, layover, quickturn):
        model = tuple(model)
        item = [int(crew_num), int(layover), int(quickturn)]
        cls.dic[model] = item
    
    @classmethod
    def get_cost(cls, model):
        return cls.dic[tuple(model)]
  
    
class Hotel:
    dic = {}
    
    @classmethod
    def add_hotel(cls, airport, cost):
        airport = tuple(airport)
        cls.dic[airport] = int(cost)
    
    @classmethod
    def get_cost(cls, airport):
        return cls.dic[tuple(airport)]