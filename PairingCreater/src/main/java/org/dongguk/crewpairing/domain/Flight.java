package org.dongguk.crewpairing.domain;

import lombok.*;
import org.dongguk.common.domain.AbstractPersistable;

import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;

@Getter
@Setter
@AllArgsConstructor
public class Flight extends AbstractPersistable {
    //비행편 T/N
    private String TailNumber;
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
    //운항시간
    private int flightTime;

    public String getIndex() {
        return "F" + id;
    }

    @Override
    public String toString() {
        return "F"+ id;
    }

    @Builder
    public Flight(long id, String TailNumber, Airport originAirport, LocalDateTime originTime, Airport destAirport, LocalDateTime destTime, Aircraft aircraft) {
        super(id);
        this.TailNumber = TailNumber;
        this.originAirport = originAirport;
        this.originTime = originTime;
        this.destAirport = destAirport;
        this.destTime = destTime;
        this.aircraft = aircraft;
        this.flightTime = (int) ChronoUnit.MINUTES.between(originTime, destTime);
    }
}
