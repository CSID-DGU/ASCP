package org.dongguk.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.time.LocalDateTime;

@Getter
@Setter
@AllArgsConstructor
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
