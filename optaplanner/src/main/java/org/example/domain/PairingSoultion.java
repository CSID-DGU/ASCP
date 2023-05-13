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

import java.util.List;

@PlanningSolution
public class PairingSoultion {

    public PairingSoultion() {
        // 인자가 없는 생성자의 내용을 필요에 따라 추가
    }

    @ProblemFactCollectionProperty
    private List<Aircraft> aircraftList;

    @ProblemFactCollectionProperty
    private List<Airport> airportList;

    @ProblemFactCollectionProperty
    private List<Flight> flightList;

    @ValueRangeProvider(id = "pairing")
    @PlanningEntityCollectionProperty
    private List<List<Pairing>> pairingList;

    @PlanningScore
    private HardSoftScore score = null;

    public PairingSoultion(List<Aircraft> aircraftList, List<Airport> airports, List<Flight> flightList, List<List<Pairing>> pairingList) {
        this.aircraftList =aircraftList;
        this.airportList = airports;
        this.flightList = flightList;
        this.pairingList = pairingList;
    }
}

