package org.dongguk.domain;

import lombok.*;
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.variable.PlanningListVariable;

import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Getter
@Setter
@Builder
@AllArgsConstructor
@RequiredArgsConstructor
@PlanningEntity
public class Pairing {
    private Integer id;
    //변수로서 작동 된다. Pair 는 Flight 들의 연속이므로 ListVariable 로 작동된다.
    @PlanningListVariable(valueRangeProviderRefs = {"pairing"})
    private List<Flight> pair = new ArrayList<>();
    private Integer totalCost;

    //pair가 시간상 불가능하면 true를 반환
    public boolean getTimeImpossible() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (pair.get(i).getDestTime().isAfter(pair.get(i + 1).getOriginTime())) {
                return true;
            }
        }
        return false;
    }

    //pair가 공간상 불가능하면 true를 반환
    public boolean getAirportImpossible() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (!pair.get(i).getDestAirport().getName().equals(pair.get(i + 1).getOriginAirport().getName())) {
                return true;
            }
        }
        return false;
    }

    //기종이 다 같은지 다 같지 않으면 true 반환
    public boolean getAircraftImpossible() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (!pair.get(i).getAircraft().getName().equals(pair.get(i + 1).getAircraft().getName())) {
                return true;
            }
        }
        return false;
    }

    //pair의 총 길이 반환 (일수)
    public long getTotalLength() {
        return ChronoUnit.DAYS.between(pair.get(0).getOriginTime(), pair.get(pair.size() - 1).getDestTime());
    }

    //첫번쨰 비행과 마지막 비행 Base 비교
    public boolean isBaseSame() {
        return pair.get(0).getOriginAirport().getName().equals(pair.get(pair.size() - 1).getDestAirport().getName());
    }
/*
    private void calculateTotalCost() {
        this.totalCost = 0;
        for (int i = 0; i < pair.size(); i++) {
            // this.totalCost += pair.get(i).getAircraft().getFlightSalary(); //시간 곱해줘야 함
            // cost+=pair.get(i).getAircraft().getLayoverCost(); //조건 필요
            this.totalCost+=pair.get(i).getAircraft().getBaseSalary();
        }
    }
*/
    //마지막 비행 반환
    public Integer getDeadHeadCost() {
        Map<String, Integer> deadheads = pair.get(pair.size() - 1).getDestAirport().getDeadheadCost();

        String dest = pair.get(pair.size() - 1).getDestAirport().getName();
        String origin = pair.get(0).getOriginAirport().getName();

        if (!dest.equals(origin)) {
            return deadheads.get(origin);
        } else {
            return 0;
        }
    }

    public Integer getLayoverCost(){
        if(pair.size() == 0) return 0;

        long flightTime = 0;
        long totalFlight = ChronoUnit.MINUTES.between(pair.get(0).getOriginTime(), pair.get(pair.size()-1).getDestTime());

        for(Flight flight : pair){
            flightTime += ChronoUnit.MINUTES.between(flight.getOriginTime(), flight.getDestTime());
        }
        //(총 페어링 시간)이 (비행 시간의 합)보다 작으면 유효하지 않은 해로 간주함(이 경우 하드 조건에서 배제됨);
        if(totalFlight<flightTime) return 0;
        return (int) (totalFlight-flightTime)*pair.get(0).getAircraft().getLayoverCost()/60;
    }

    @Override
    public String toString() {
        return "Pairing - " + id +
                " { pair=" + pair + " }";
    }
}
