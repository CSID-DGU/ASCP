package org.example.domain;


import java.time.LocalDateTime;



public class Flight {
    //비행편 인덱스
    private String flightNumber;
    //출발 공항
    private Airport originAirport;
    //출발 시간
    private LocalDateTime originTime;
    //도착 공항
    private Airport destAirport;
    //도착 시간
    private LocalDateTime destTime;
    //기종
    private Aircraft aircraft;

    public Flight(String flightNumber,Airport originAirport, LocalDateTime originTime, Airport destAirport, LocalDateTime destTime, Aircraft aircraft){
        this.flightNumber = flightNumber;
        this.originAirport = originAirport;
        this.originTime = originTime;
        this.destAirport = destAirport;
        this.destTime =destTime;
        this.aircraft = aircraft;
    }

    public LocalDateTime getOriginTime() {
        return originTime;
    }

    public Airport getDestAirport() {
        return destAirport;
    }

    public Airport getOriginAirport() {
        return originAirport;
    }

    public LocalDateTime getDestTime() {
        return destTime;
    }

    public Aircraft getAircraft() {
        return aircraft;
    }

    public String getFlightNumber() {
        return flightNumber;
    }

    //     Shadow variables
//
//    @PreviousElementShadowVariable(sourceVariableName = "pair")
//    private Flight previousFlight;
//    @NextElementShadowVariable(sourceVariableName = "pair")
//    private Flight nextFlight;


    @Override
    public String toString() {
        return "Flight{" +
                "flightNumber='" + flightNumber + '\'' +
                ", originAirport=" + originAirport +
                ", originTime=" + originTime +
                ", destAirport=" + destAirport +
                ", destTime=" + destTime +
                ", aircraft=" + aircraft +
                '}';
    }
}
