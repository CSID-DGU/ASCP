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
@Getter
@Setter
public class PairingSoultion {

    @ProblemFactCollectionProperty
    private List<Aircraft> aircraftList;

    @ProblemFactCollectionProperty
    private List<Airport> airportList;

    @ProblemFactCollectionProperty
    private List<Flight> flightList;

    @ValueRangeProvider
    @PlanningEntityCollectionProperty
    private List<Pairing> pairingList;

    @PlanningScore
    private HardSoftScore score = null;

    public PairingSoultion(List<Aircraft> aircraftList, List<Airport> airports, List<Flight> flightList) {
    }
}

