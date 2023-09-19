from math import floor
from typing import List
from Flight import Flight
import pandas as pd


class Pairing:
    briefingTime = 0
    debriefingTime = 0
    restTime = 0
    LayoverTime = 6 * 60
    QuickTurnaroundTime = 0
    hotelTime = 18 * 60
    hotelMinTime = 720
    checkContinueTime = 60 * 10
    continueMaxTime = 14 * 60  # 14시간
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
        for i in range(len(self.pair) - 1):
            if self.pair[i].destTime > self.pair[i + 1].originTime:
                return True
        return False

    def getSatisCost(self):
        satisScore = 0
        for i in range(len(self.pair) - 1):
            if self.checkBreakTime(i) <= 180:
                satisScore += 1000 * (180 - self.checkBreakTime(i))
        return satisScore

    def getContinuityImpossible(self):  # 휴식시간 포함 14시간, 휴식시간 빼고 8시간 둘 다 지켜져야함.
        time = self.pair[0].flightTime

        for i in range(1, len(self.pair)):
            if self.checkBreakTime(i-1) < self.checkContinueTime:
                time += self.pair[i].flightTime + self.checkBreakTime(i-1)
            else:
                time = self.pair[i].flightTime

            if time > self.continueMaxTime:
                return True
        return False

    def getAirportImpossible(self):
        for i in range(len(self.pair) - 1):
            if self.pair[i].destAirport.name != self.pair[i + 1].originAirport.name:
                return True
        return False

    def getAircraftDiff(self):
        for i in range(len(self.pair) - 1):
            if self.pair[i].aircraft.type != self.pair[i + 1].aircraft.type:
                return True
        return False

    def getMovingWorkCost(self):
        maxCrewNum = 0
        movingWorkCost = 0

        for flight in self.pair:
            maxCrewNum = max(maxCrewNum, flight.aircraft.crewNum)

        for flight in self.pair:
            movingWorkCost += (maxCrewNum -
                               flight.aircraft.crewNum) * flight.flightTime * 10

        return movingWorkCost

    def getTotalLength(self):
        startTime = self.pair[0].originTime
        endTime = self.pair[-1].destTime
        totalLength = (endTime - startTime).days
        return totalLength

    def equalBase(self):
        return self.pair[0].originAirport != self.pair[-1].destAirport

    def getDeadheadCost(self):

        if self.pair[0].id == -1:
            return 0
        else:
            dest = None
            origin = self.pair[0].originAirport
            for i in range(len(self.pair)):
                if self.pair[i].id != -1:
                    dest = self.pair[i].destAirport
            if dest is not None:
                # 출발공항에 대하여, 도착 공항이 어디인지를 인자로 넘겨주어, 해당하는 deadhead cost를 불러옴.
                deadhead = origin.getDeadheadCost(dest)

            return deadhead

    def getLayoverCost(self):  # LayoverTime 빠짐!!  # 가장 인원이 많이 타는 비행기를 찾아서 걔를 부여
        if len(self.pair) <= 1:
            return 0

        cost = 0
        for i in range(len(self.pair) - 1):
            if self.checkBreakTime(i) <= 0:
                return 0

            if self.checkBreakTime(i) >= self.LayoverTime:
                cost += (self.checkBreakTime(i) -
                         self.LayoverTime) * self.pair[0].aircraft.layoverCost

        return cost // 100

    def checkBreakTime(self, index):
        breakTime = (self.pair[index+1].originTime -
                     self.pair[index].destTime).total_seconds() // 60
        return max(0, breakTime)

    def getQuickTurnCost(self):  # 가장 인원이 많이 타는 비행기를 찾아서 걔를 부여
        if len(self.pair) <= 1:
            return 0

        cost = 0
        for i in range(len(self.pair) - 1):
            if self.checkBreakTime(i) <= 0:
                return 0

            if self.checkBreakTime(i) < self.QuickTurnaroundTime:
                cost += (self.QuickTurnaroundTime - self.checkBreakTime(i)
                         ) * self.pair[0].aircraft.getQuickTurnCost()

        return cost // 100

    def getHotelCost(self):  # hoteltime, hotelMinTime 빠짐!!!!!!!!!  # 가장 인원이 많이 타는 비행기를 찾아서 걔를 부여
        if len(self.pair) <= 1:
            return 0

        cost = 0
        for i in range(len(self.pair) - 1):
            if self.checkBreakTime(i) <= 0:
                return 0

            flightGap = self.checkBreakTime(i)

            if flightGap >= self.hotelMinTime:
                cost += (
                    self.pair[i + 1].originAirport.hotelCost *
                    self.pair[0].aircraft.crewNum *
                    (1 + floor((flightGap - self.hotelMinTime) / self.hotelTime))
                )

        return cost // 100
