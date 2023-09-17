import math

class ScoreCalculator:

    #템플릿 메서드 패턴 적용. calculateScore에는 연산의 뼈대 결정. 이후 서브클래스에서 구현.
    def calculateScore(self,pairing):
      hardScore=self.airportPossible(pairing)+self.landingTimes(pairing)+self.pairLength(pairing)+self.continuityPossible(pairing)+self.minBreakTime(pairing)+self.pairMinLength(pairing)
      softScore=self.baseDiff(pairing)+self.layoverCost(pairing)+self.movingWorkCost(pairing)+self.quickTurnCost(pairing)+self.hotelCost(pairing)+self.satisCost+self.testCost(pairing)
      return hardScore,softScore
    
    # Hard 조건
    # 공간적 선후관계 판단. 틀리다면 Hard score 1000점 부여
    def airportPossible(pairing):
        score=0
        for pair in pairing:
            if pair.getAirportImpossible()==True:
                score= score+1000
        return score

    # Hard 조건
    # 기종 동일성 판단. 틀리다면 Hard 점수 500점 부여
    def aircraftType(pairing):
        score=0
        for pair in pairing:
            if pair.getAircraftDiff()==True:
                score=score+500
        return score

    # Hard 조건
    # 비행 횟수 제약(Landing times):
    # 비행 횟수가 4회 이상일 시 -> 하드스코어 부여(총 비행횟수 * 100)
    def landingTimes(pairing):
        score=0
        for pair in pairing:
            if len(pair)>4:
                score=score+(len(pair)*100)
        return score

    # HARD
    # 비행 일수 제약(Max length):
    # 페어링 총 기간이 7일 이상일 시 -> 하드스코어 부여((총 길이-7) * 100)
    def pairLength(pairing):
        score=0
        for pair in pairing:
            if len(pair)>2 and pair.getTotalLength()>7:
                score=score+int(math.floor((pairing.getTotalLength()-7)*100))
        return score

  
    def continuityPossible(pairing):
        score=0
        for pair in pairing:
            if len(pair)>=2 and pair.getContinuityImpossible()==True:
                score=score+1000
        return score


    #  SOFT
    # deadhead cost 계산(Base diff):
    # 첫 출발공항과 마지막 도착공항이 다를 시 - > 소프트스코어 부여(항공편에 따른 가격)
    def baseDiff(pairing):
        score=0
        for pair in pairing:
            if len(pair)>=1 and pair.equalBase()==True:
                score=score+pair.getDeadheadCost()
        return score

    # SOFT
    # 총 layover cost 계산(Layover cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(layover 발생 시 cost+)
    def layoverCost(pairing):
        score=0
        for pair in pairing:
            if len(pair)>=2:
                score=score+pair.getLayoverCost()
        return score


    # SOFT
    # 총 이동근무 cost 계산(MovingWork cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(MovingWork cost 발생 시 cost+)
    def movingWorkCost(pairing):
        score=0
        for pair in pairing:
            if len(pair)>=2:
                score=score+pair.geteMovingWorkCost()
        return score

    # SOFT
    # 총 QuickTurn cost 계산(QuickTurn cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(QuickTurn cost 발생 시 cost+)
    def quickTurnCost(pairing):
        score=0
        for pair in pairing:
            if len(pair)>=2:
                score=score+pair.getQuickTurnCost()
        return score
    

    # SOFT
    # 총 호텔숙박비 cost 계산(Hotel cost):
    # 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Hotel cost 발생 시 cost+)
    def hotelCost(pairing):
        score=0
        for pair in pairing:
            if len(pair)>=2:
                score=score+pair.getHotelCost()
        return score

    # SOFT
    # 승무원 만족도 cost 계산(Satis cost):
    # 승무원의 휴식시간에 따른 만족도를 코스트로 score 부여
    # / 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Satis cost 발생 시 cost+)
    def satisCost(pairing):
        score=0
        for pair in pairing:
            if len(pair)>=2:
                score=score+pair.getsatisCost()
        return score

    # 모든 1개 이상인 페어링에 soft 점수를 부여해서 페어링의 수가 줄어드는지 실험
    def testCost(pairing):
        score=0
        for pair in pairing:
            if len(pair)>=1:
                score=score+1000000
        return score


