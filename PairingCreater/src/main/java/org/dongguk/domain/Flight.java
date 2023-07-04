package org.dongguk.domain;

import lombok.*;

import java.time.LocalDateTime;

@Getter
@Setter
@Builder
@AllArgsConstructor
@RequiredArgsConstructor
public class Flight {
    //비행편 인덱스
    private String index;
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
                "flightNumber='" + index + '\'' +
                ", originAirport=" + originAirport +
                ", originTime=" + originTime +
                ", destAirport=" + destAirport +
                ", destTime=" + destTime +
                ", aircraft=" + aircraft +
                '}';
    }
}
