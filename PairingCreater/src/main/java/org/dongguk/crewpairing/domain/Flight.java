package org.dongguk.crewpairing.domain;

import lombok.*;
import org.dongguk.common.domain.AbstractPersistable;

import java.time.LocalDateTime;

@Getter
@Setter

@AllArgsConstructor
@RequiredArgsConstructor
public class Flight extends AbstractPersistable {
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
        return index;
    }

    @Builder
    public Flight(long id, String index, Airport originAirport, LocalDateTime originTime, Airport destAirport, LocalDateTime destTime, Aircraft aircraft) {
        super(id);
        this.index = index;
        this.originAirport = originAirport;
        this.originTime = originTime;
        this.destAirport = destAirport;
        this.destTime = destTime;
        this.aircraft = aircraft;
    }
}
