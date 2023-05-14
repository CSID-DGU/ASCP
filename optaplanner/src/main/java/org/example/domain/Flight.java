package org.example.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.kie.api.definition.rule.ActivationListener;
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.variable.NextElementShadowVariable;
import org.optaplanner.core.api.domain.variable.PreviousElementShadowVariable;

import java.time.LocalDateTime;



public class Flight {
    private String flightNumber;
    private Airport originAirport;
    private LocalDateTime originTime;
    private Airport destAirport;
    private LocalDateTime destTime;
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
