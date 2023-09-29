from math import floor
from typing import List
from Flight import Flight
import pandas as pd
import math


class Pairing:
    briefingTime = 0
    debriefingTime = 0
    restTime = 0
    LayoverTime = 6 * 60
    QuickTurnaroundTime = 30
    hotelTime = 18 * 60
    hotelMinTime = 720
    checkContinueTime = 60 * 10
    continueMaxTime = 14 * 60  # 14시간
    workMaxTime = 8 * 60
    # 8시간 짜리 하나 만들어야됨

    def __init__(self, id, pair: List['Flight']):
        self.id = id
        self.pair = pair

    @staticmethod
    def setStaticTime(briefingTime, debriefingTime, restTime, LayoverTime, QuickTurnaroundTime):
        Pairing.briefingTime = briefingTime
        Pairing.debriefingTime = debriefingTime
        Pairing.restTime = restTime
        Pairing.LayoverTime = LayoverTime
        Pairing.QuickTurnaroundTime = QuickTurnaroundTime

    def setDebriefingTime(self, debriefingTime):
        Pairing.debriefingTime = debriefingTime

    def setRestTime(self, restTime):
        Pairing.restTime = restTime

    def setLayoverTime(self, layoverTime):
        Pairing.LayoverTime = layoverTime

    def setQuickTurnaroundTime(self, quickTurnaroundTime):
        Pairing.QuickTurnaroundTime = quickTurnaroundTime

    def getTimeImpossible(self):
        """
            /**
            * pairing의 실행 가능 여부 확인(불가능한 경우:true)
            * / 앞 비행이 도착하지 않았는데 이후 비행이 출발했을 경우 판단
            * @return boolean
            */
        """

        for i in range(len(self.pair) - 1):
            # 단, 이후 비행기가 dummyFlight면 무효
            if self.pair[i].destTime > self.pair[i + 1].originTime and self.pair[i+1].id != -1:
                return True
        return False

    def getAirportImpossible(self):
        """
            /**
            * 동일 공항 출발 여부 확인
            * / 도착 공항과 출발 공항이 다를 시 true 반환
            * @return boolean
            */
        """
        for i in range(len(self.pair) - 1):
            # 이후 비행기가 dummyFlight면 무효
            if self.pair[i].destAirport.name != self.pair[i + 1].originAirport.name and self.pair[i+1].id != -1:
                return True
        return False

    def getContinuityImpossible(self):  # 휴식시간 포함 14시간, 휴식시간 빼고 8시간 둘 다 지켜져야함.
        """
            /**
            * 페어링의 최소 휴식시간 보장 여부 검증
            * / 연속되는 비행이 14시간 이상일 시 true 반환(연속: breakTime이 10시간 이하)
            * @return boolean
            */
        """
        totalTime = self.pair[0].flightTime
        workTime = self.pair[0].flightTime

        for i in range(1, len(self.pair)):
            if self.__checkBreakTime(i-1) < self.checkContinueTime:
                totalTime += self.pair[i].flightTime + \
                    self.__checkBreakTime(i-1)
                workTime += self.pair[i].flightTime
            else:
                totalTime = self.pair[i].flightTime
                workTime = self.pair[i].flightTime

            if totalTime > self.continueMaxTime:
                return True
            if workTime > self.workMaxTime:
                return True
        return False

    def getAircraftDiff(self):
        """
            /**
            * pairing의 동일 항공기 여부 검증
            * / 비행들의 항공기가 동일하지 않을 시 true 반환
            * @return boolean
            */
        """
        for i in range(len(self.pair) - 1):
            if self.pair[i].aircraft.type != self.pair[i + 1].aircraft.type:
                return True
        return False

    def equalBase(self):
        """
            /**
            * 처음과 끝 공항의 동일 여부 확인
            * / 처음 출발 공항과 마지막 도착 공항이 다를 시 true
            * @return boolean
            */
        """
        dest = None
        for i in range(len(self.pair)):
            if self.pair[i].id == -1:  # dummyFlihgt 고려.
                break
            dest = self.pair[i].destAirport.name
        return self.pair[0].originAirport.name != dest

    def getSatisCost(self):
        """
            /**
            * 페어링의 총 SatisCost 반환
            * / breakTime이 180보다 작은 경우 발생
            * @return 퀵턴코스트/(시간-퀵턴 기준 시간)
            */
        """
        satisScore = 0

        for i in range(len(self.pair) - 1):
            if self.__isLastFlight(i):
                break
            flight_gap = self.__checkBreakTime(i)
            if flight_gap < self.LayoverTime and flight_gap > self.QuickTurnaroundTime:
                start_cost = self.pair[i].aircraft.quickTurnCost
                plus_score = (start_cost // (self.LayoverTime - self.QuickTurnaroundTime)) * (-flight_gap + self.LayoverTime)
                satisScore += plus_score

        return satisScore

    def getTotalLength(self):
        """
            /**
            * 페어링의 총 갈아 반환 (일)
            * @return 마지막 비행 도착시간 - 처음 비행 시작시간
            */
        """
        startTime = self.pair[0].originTime
        endTime = self.pair[-1].destTime
        totalLength = (endTime - startTime).days
        return totalLength

    def getMovingWorkCost(self):
        """
            /**
            * 페어링의 총 이동근무 cost 반환
            * / 페어링 인원보다 요구 승무원이 적은 비행일 시 발생(maxCrewNum이 기준)
            * @return sum((maxCrewNum - 요구 승무원) * (해당 항공편의 시작 공항 -> 종료 공항에 해당하는 Deadhead Cost))
            */
        """
        movingWorkCost = 0

        for flight in self.pair:
            if flight.id == -1:  # dummyFlight 만날 시 break
                break

            presentCrewNum = flight.aircraft.crewNum

            movingWorkCost += ((self.__getMaxCrewNum() - presentCrewNum)
                               * flight.originAirport.getDeadheadCost(flight.destAirport))

        return movingWorkCost//100

    def getDeadheadCost(self):
        """
            /**
            * 페어링의 deadhead cost 반환
            * / 마지막 도착 공항에서 처음 공항으로 가는데 필요한 deadhead cost 사용
            * @return deadhead cost / 2
            */
        """
        if self.pair[0].id == -1:  # 빈 페어링의 경우 0 반환. 만약 페어링이 비어있더라도 equalBase()가 True 나오므로 필요.
            return 0
        else:
            dest = None
            origin = self.pair[0].originAirport
            for i in range(len(self.pair)):
                if self.pair[i].id != -1:
                    dest = self.pair[i].destAirport
            # 출발공항에 대하여, 도착 공항이 어디인지를 인자로 넘겨주어, 해당하는 deadhead cost를 불러옴.
            deadhead = origin.getDeadheadCost(dest)

        return (deadhead * self.__getMaxCrewNum())//100

    def getLayoverCost(self):
        """
            /**
            * 페어링의 총 LayoverCost 반환
            * 비행편간 간격이 LayoverTime 보다 크거나 같은 경우에만 LayoverCost 발생
            * @return sum(LayoverCost) / 100
            */
        """
        maxLayoverCost = self.pair[0].aircraft.layoverCost
        # pair에 있는 flight 중 최대 maxLayoverCost를 찾음
        for flight in self.pair:
            maxLayoverCost = max(maxLayoverCost, flight.aircraft.layoverCost)

        cost = 0
        flightGapSum=0
        for i in range(len(self.pair) - 1):

            if self.__isLastFlight(i):
                break
            # 단순 dummyFlight라서 return 0이 되는 경우 없게 만듦.
            if self.__checkBreakTime(i) <= 0:
                return 0

            if self.__checkBreakTime(i) >= self.LayoverTime:
                flightGapSum+=self.__checkBreakTime(i)-self.LayoverTime
                cost += (self.__checkBreakTime(i) -
                         self.LayoverTime) * maxLayoverCost
        return cost // 100

    def getQuickTurnCost(self):
        """
            /**
            * 페어링의 총 QuickTurnCost 반환
            * 비행편간 간격이 QuickTurnaroundTime 보다 작은 경우에만 QuickTurnCost 발생
            * @return sum(QuickTurnCost) / 100
            */
        """
        cost = 0
        for i in range(len(self.pair) - 1):
            if self.__isLastFlight(i):
                break
            if self.__checkBreakTime(i) <= 0:
                return 0

            if self.pair[i].aircraft.type == self.pair[i+1].aircraft.type and self.__checkBreakTime(i) < self.QuickTurnaroundTime:
                cost += self.pair[i].aircraft.quickTurnCost

        return cost // 100

    def getHotelCost(self):  
        """
            /**
            * 페어링의 총 HotelCost 반환
            * / 총 인원수를 곱하는 이유 : Flight Cost, Layover Cost, QuickTurn Cost 모두 총 인원에 대한 값으로 계산된 후 입력받음
            * / 휴식시간이 12시간 이상일 경우 1일 숙박,이후 18시간 이상 남을 시 1일 추가 반복
            * @return sum(hotel cost) / 100
            */
        """
        cost = 0
        for i in range(len(self.pair) - 1):
            
            if self.__isLastFlight(i):
                break
            # 만약 비행편 간격이 하나라도 0이라면 유효한 페어링이 아님
            if self.__checkBreakTime(i) <= 0:
                return 0
        
            layover_start_time = self.pair[i].destTime
            layover_finish_time = self.pair[i + 1].originTime

            # layover가 발생했으면 일단 1회 발생, 이후 날짜가 바뀔 때마다 1회씩 발생
            if self.__checkBreakTime(i)>= self.LayoverTime:
                cost += (
                    self.pair[i].destAirport.hotelCost
                    * self.__getMaxCrewNum()
                    * (1 + max(0, (layover_finish_time.date() - layover_start_time.date()).days - 1))
                )     

        return cost // 100

    def __checkBreakTime(self, index):
        """
            /**
            * 비행 사이의 쉬는 시간 계산
            * @return (int) Math.max(0,breakTime)
            */
        """

        breakTime = (self.pair[index+1].originTime -
                     self.pair[index].destTime).total_seconds() // 60
        return max(0, breakTime)

    def __getMaxCrewNum(self):
        """
            /**
            * pairing의 인원을 구하는 메서드
            * @return maxCrewNum
            */
        """
        maxCrewNum = 0
        for flight in self.pair:
            maxCrewNum = max(maxCrewNum, flight.aircraft.crewNum)
        return maxCrewNum

    def __isLastFlight(self, index):
        return self.pair[index].id == -1 or self.pair[index+1].id == -1
