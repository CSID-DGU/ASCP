package org.example.domain;

import lombok.Getter;
import lombok.Setter;
import org.example.domain.Flight;
import org.example.domain.Pairing;
import org.optaplanner.core.api.domain.solution.PlanningEntityCollectionProperty;
import org.optaplanner.core.api.domain.solution.PlanningScore;
import org.optaplanner.core.api.domain.solution.PlanningSolution;
import org.optaplanner.core.api.domain.solution.ProblemFactCollectionProperty;
import org.optaplanner.core.api.domain.valuerange.ValueRangeProvider;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.buildin.hardsoftlong.HardSoftLongScore;

import java.util.List;

@PlanningSolution
public class PairingSoultion {

    public PairingSoultion() {
        // 인자가 없는 생성자의 내용을 필요에 따라 추가
    }

    //Aircraft에 대한 모든 정보
    @ProblemFactCollectionProperty
    private List<Aircraft> aircraftList;

    //Airport에 대한 모든 정보
    @ProblemFactCollectionProperty
    private List<Airport> airportList;

    //비행편에 대한 모든 정보 / 변수로서 작동 되므로 ValueRangeProvider 필요
    @ValueRangeProvider(id = "pairing")
    @ProblemFactCollectionProperty
    private List<Flight> flightList;
    // solver가 풀어낸 Entity들
    @PlanningEntityCollectionProperty
    private List<Pairing> pairingList;

    //score 변수
    @PlanningScore
    private HardSoftScore score = null;

    //private double TotalPairingCost;
    public PairingSoultion(List<Aircraft> aircraftList, List<Airport> airports, List<Flight> flightList, List<Pairing> pairingList) {
        this.aircraftList = aircraftList;
        this.airportList = airports;
        this.flightList = flightList;
        this.pairingList = pairingList;
    }

    public List<Aircraft> getAircraftList() {
        return aircraftList;
    }

    public List<Airport> getAirportList() {
        return airportList;
    }

    public List<Flight> getFlightList() {
        return flightList;
    }

    public List<Pairing> getPairingList() {
        return pairingList;
    }


    public HardSoftScore getScore() {
        return score;
    }

    //총 코스트
    public double getTotalPairingCost() {
        double total = 0;
        for (int i = 0; i < pairingList.size() - 1; i++) {
            total += pairingList.get(i).getTotalCost();
        }
        return total;
    }
    //코스트 평균
    public double getCostAverage() {
        return getTotalPairingCost() / pairingList.size();
    }

    @Override
    public String toString() {
        return "PairingSoultion{" +
                "aircraftList=" + aircraftList +
                ", \nairportList=" + airportList +
                ",\nflightList=" + flightList +
                ", \npairingList=" + pairingList +
                ", \nscore=" + score +
                '}';
    }

    public void printParingList() {
        for (Pairing pair : pairingList) {
            for (Flight flight : pair.getPair()) {
                System.out.print(flight.getFlightNumber() + " ");
            }
            System.out.println();
        }
    }
}

