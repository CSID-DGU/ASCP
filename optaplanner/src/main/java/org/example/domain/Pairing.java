package org.example.domain;


import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.valuerange.ValueRangeProvider;
import org.optaplanner.core.api.domain.variable.PlanningListVariable;
import org.optaplanner.core.api.domain.variable.PlanningVariable;

import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;


@PlanningEntity
public class Pairing {

    public Pairing() {
        // 인자가 없는 생성자의 내용을 필요에 따라 추가
    }

    public Pairing(List<Flight> pair, int totalCost) {
        this.pair = pair;
        this.totalCost = totalCost;
    }

    //변수로서 작동 된다. pair는 Flight들의 연속이므로 ListVariable로 작동된다.
    @PlanningListVariable(valueRangeProviderRefs = {"pairing"})
    private List<Flight> pair = new ArrayList<>();

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

    public List<Flight> getPair() {
        return pair;
    }

    private double totalCost;

    // 여기서 Cost계산 해야할 것
    public double getTotalCost() {
        setTotalCost();
        return totalCost;
    }

    public void setTotalCost() {
        this.totalCost = 0;
        for (int i = 0; i < pair.size(); i++) {
            //this.totalCost += pair.get(i).getAircraft().getFlightSalary(); //시간 곱해줘야 함
            //cost+=pair.get(i).getAircraft().getLayoverCost(); //조건 필요
            this.totalCost+=pair.get(i).getAircraft().getBaseSalary();
        }
    }

    //마지막 비행 반환
    public Flight getLastFlight() {
        return pair.get(pair.size() - 1);
    }


    @Override
    public String toString() {
        return "Pairing{" +
                "pair=" + pair +
                ", totalCost=" + totalCost +
                '}';
    }
}
