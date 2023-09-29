import math
import sys
import pandas as pd
from Factory import ASCPFactory


class ScoreCalculator:

    def __init__(self, pairing):
        self.pairing = pairing

    # 템플릿 메서드 패턴 적용. calculateScore에는 연산의 뼈대 결정. 이후 서브클래스에서 구현.
    def calculateScore(self):
        hardScore = self.timePossible()+self.airportPossible() + \
            self.continuityPossible()

        softScore = self.baseDiff()+self.layoverCost()+self.movingWorkCost(
        )+self.quickTurnCost()+self.satisCost() +self.hotelCost()

        return hardScore, softScore

    def countFlight(self):  # dummyFlight를 제외한 pairing에 포함된 flight 수를 반환
        cnt = 0
        for i in range(len(self.pairing.pair)):
            if self.pairing.pair[i].id == -1:
                break
            cnt = cnt+1
        return cnt

    # Hard 조건
    # 시간적 선후관계 판단. 틀리다면 Hard score 1000점 부여
    def timePossible(self):
        score = 0
        if self.pairing.getTimeImpossible() == True:
            # print('timePossible')
            score = score+1000
        return score

    # Hard 조건
    # 공간적 선후관계 판단. 틀리다면 Hard score 1000점 부여

    def airportPossible(self):
        score = 0
        if self.pairing.getAirportImpossible() == True:
            score = score+1000
        return score

    # Hard 조건
    # 연속 근무, 연속 휴식에 대한 법적 제약. 틀리다면 Hard 점수 1000 점 부여

    def continuityPossible(self):
        score = 0
        if self.countFlight() >= 2 and self.pairing.getContinuityImpossible() == True:
            score = score+1000
        return score

    #  SOFT
    # deadhead cost 계산(Base diff):
    # 첫 출발공항과 마지막 도착공항이 다를 시 - > 소프트스코어 부여(항공편에 따른 가격)

    def baseDiff(self):
        score = 0
        if self.countFlight() >= 1 and self.pairing.equalBase() == True:
            score = score+self.pairing.getDeadheadCost()
        return score

    # SOFT
    # 총 layover cost 계산(Layover cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(layover 발생 시 cost+)

    def layoverCost(self):
        score = 0
        if self.countFlight() >= 2:
            score = score+self.pairing.getLayoverCost()
        return score

    # SOFT
    # 총 이동근무 cost 계산(MovingWork cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(MovingWork cost 발생 시 cost+)

    def movingWorkCost(self):
        score = 0
        if self.countFlight() >= 2:
            score = score+self.pairing.getMovingWorkCost()
        return score

    # SOFT
    # 총 QuickTurn cost 계산(QuickTurn cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(QuickTurn cost 발생 시 cost+)

    def quickTurnCost(self):
        score = 0
        if self.countFlight() >= 2:
            score = score+self.pairing.getQuickTurnCost()
        return score

    # SOFT
    # 총 호텔숙박비 cost 계산(Hotel cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Hotel cost 발생 시 cost+)

    def hotelCost(self):
        score = 0
        if self.countFlight() >= 2:
            score = score+self.pairing.getHotelCost()
        return score

    # SOFT
    # 승무원 만족도 cost 계산(Satis cost):
    # 승무원의 휴식시간에 따른 만족도를 코스트로 score 부여
    # / 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Satis cost 발생 시 cost+)

    def satisCost(self):
        score = 0
        if self.countFlight() >= 2:
            score = score+self.pairing.getSatisCost()
        return score

