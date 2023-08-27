from Flight import Flight 
from datetime import datetime, timedelta
from typing import List, Dict
from math import floor

class Pairing:
    briefingTime = 0
    debriefingTime = 0
    restTime = 0
    LayoverTime = 0
    QuickTurnaroundTime = 0
    hotelTime = 18 * 60
    hotelMinTime = 720
    checkContinueTime = 60 * 10
    continueMaxTime = 14 * 60
    
    def __init__(self, id, pair: List['Flight'], totalCost: int):
        self.id = id
        self.pair = pair
        self.totalCost = totalCost
    
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
    
    def getContinuityImpossible(self):
        time = self.pair[0].flightTime 

        for i in range(1,len(self.pair)):
            if Pairing.checkBreakTime(i-1) < Pairing.checkContinueTime:
               time += self.pair[i].flightTime + Pairing.checkBreakTime(i-1)
            else:
               time = self.pair[i].flightTime

            if time > Pairing.continueMaxTime:
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
            movingWorkCost += (maxCrewNum - flight.aircraft.crewNum) * flight.flightTime * 10

        return movingWorkCost

    def getTotalLength(self):
        startTime = self.pair[0].originTime
        endTime = self.pair[-1].destTime
        totalLength = (endTime - startTime).days
        return totalLength

    def equalBase(self):
        return self.pair[0].originAirport.name != self.pair[-1].destAirport.name
    
    def getDeadheadCost(self):
        deadheads = self.pair[-1].destAirport.getDeadheadCost()
    
        # dest = self.pair[-1].destAirport.name  제거해야된다고 써있어서 주석처리해둠
        origin = self.pair[0].originAirport.name
    
        return deadheads.get(origin, 0) // 2



    def getLayoverCost(self, LayoverTime):
        if len(self.pair) <= 1:
            return 0

        cost = 0
        for i in range(len(self.pair) - 1):
            if Pairing.checkBreakTime(i, self.pair) <= 0:
                return 0

            if Pairing.checkBreakTime(i, self.pair) >= LayoverTime:
                cost += (Pairing.checkBreakTime(i, self.pair) - LayoverTime) * self.pair[0].aircraft.getLayoverCost()

        return cost // 100


    @staticmethod
    def checkBreakTime(self, index):
        breakTime = (self.pair[index+1].originTime - self.pair[index].destTime).total_seconds() // 60
        return max(0, breakTime)

    def getQuickTurnCost(self, QuickTurnaroundTime):
        if len(self.pair) <= 1:
            return 0

        cost = 0
        for i in range(len(self.pair) - 1):
            if Pairing.checkBreakTime(i, self.pair) <= 0:
                return 0

            if Pairing.checkBreakTime(i, self.pair) < QuickTurnaroundTime:
                cost += (QuickTurnaroundTime - Pairing.checkBreakTime(i, self.pair)) * self.pair[0].aircraft.getQuickTurnCost()

        return cost // 100

    def getHotelCost(self, hotelMinTime, hotelTime):
        if len(self.pair) <= 1:
            return 0

        cost = 0
        for i in range(len(self.pair) - 1):
            if Pairing.checkBreakTime(i, self.pair) <= 0:
                return 0

            flightGap = Pairing.checkBreakTime(i, self.pair)

            if flightGap >= hotelMinTime:
                cost += (
                    self.pair[i + 1].originAirport.getHotelCost() * 
                    self.pair[0].aircraft.crewNum *
                    (1 + floor((flightGap - hotelMinTime) / hotelTime))
                )

        return cost // 100

    def __str__(self, id):
        return f"Pairing - {id} {{ pair = {self.pair} }}"


