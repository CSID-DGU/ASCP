import math


class ScoreCalculator:

    def __init__(self, pairing):
        self.pairing = pairing

    # 템플릿 메서드 패턴 적용. calculateScore에는 연산의 뼈대 결정. 이후 서브클래스에서 구현.
    def calculateScore(self):
        hardScore = self.timePossible()+self.airportPossible()+self.landingTimes() + \
            self.continuityPossible()  # +self.minBreakTime()+self.pairMinLength()
        softScore = self.baseDiff()+self.layoverCost()+self.movingWorkCost(
        )+self.quickTurnCost()+self.hotelCost()+self.satisCost()
        return hardScore, softScore

    # Hard 조건
    # 시간적 선후관계 판단. 틀리다면 Hard score 1000점 부여

    def timePossible(self):
        score = 0
        if self.pairing.getTimeImpossible() == True:
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
    # 기종 동일성 판단. 틀리다면 Hard 점수 500점 부여
    def aircraftType(self):
        score = 0
        if self.pairing.getAircraftDiff() == True:
            score = score+500
        return score

    # Hard 조건
    # 비행 횟수 제약(Landing times):
    # 비행 횟수가 4회 이상일 시 -> 하드스코어 부여(총 비행횟수 * 100)
    def landingTimes(self):
        score = 0
        if len(self.pairing.pair) > 4:
            score = score+(len(self.pairing.pair)*100)
        return score

    # HARD
    # 비행 일수 제약(Max length):
    # 페어링 총 기간이 7일 이상일 시 -> 하드스코어 부여((총 길이-7) * 100)
    # def pairLength(self):
    #    score = 0
    #        if len(self.pairing.pair) > 2 and self.pairing.getTotalLength() > 7:
    #            score = score + \
    #                int(math.floor((self.pairing.getTotalLength()-7)*100))
    #    return score

    def continuityPossible(self):
        score = 0
        if len(self.pairing.pair) >= 2 and self.pairing.getContinuityImpossible() == True:
            score = score+1000
        return score

    #  SOFT
    # deadhead cost 계산(Base diff):
    # 첫 출발공항과 마지막 도착공항이 다를 시 - > 소프트스코어 부여(항공편에 따른 가격)

    def baseDiff(self):
        score = 0
        if len(self.pairing.pair) >= 1 and self.pairing.equalBase() == True:
            score = score+self.pairing.getDeadheadCost()
        return score

    # SOFT
    # 총 layover cost 계산(Layover cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(layover 발생 시 cost+)
    def layoverCost(self):
        score = 0
        if len(self.pairing.pair) >= 2:
            score = score+self.pairing.getLayoverCost()
        return score

    # SOFT
    # 총 이동근무 cost 계산(MovingWork cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(MovingWork cost 발생 시 cost+)

    def movingWorkCost(self):
        score = 0
        if len(self.pairing.pair) >= 2:
            score = score+self.pairing.getMovingWorkCost()
        return score

    # SOFT
    # 총 QuickTurn cost 계산(QuickTurn cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(QuickTurn cost 발생 시 cost+)
    def quickTurnCost(self):
        score = 0
        if len(self.pairing.pair) >= 2:
            score = score+self.pairing.getQuickTurnCost()
        return score

    # SOFT
    # 총 호텔숙박비 cost 계산(Hotel cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Hotel cost 발생 시 cost+)

    def hotelCost(self):
        score = 0
        if len(self.pairing.pair) >= 2:
            score = score+self.pairing.getHotelCost()
        return score

    # SOFT
    # 승무원 만족도 cost 계산(Satis cost):
    # 승무원의 휴식시간에 따른 만족도를 코스트로 score 부여
    # / 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Satis cost 발생 시 cost+)
    def satisCost(self):
        score = 0
        if len(self.pairing.pair) >= 2:
            score = score+self.pairing.getSatisCost()
        return score

    # 모든 1개 이상인 페어링에 soft 점수를 부여해서 페어링의 수가 줄어드는지 실험
    def testCost(self):
        score = 0
        if len(self.pairing.pair) >= 1:
            score = score+1000000
        return score
