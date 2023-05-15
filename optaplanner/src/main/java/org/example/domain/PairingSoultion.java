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
    @ValueRangeProvider(id = "pairing")
    @ProblemFactCollectionProperty
    private List<Flight> flightList;

    @PlanningEntityCollectionProperty
    private List<Pairing> pairingList;

    @PlanningScore
    private HardSoftScore score = null;

    public PairingSoultion(List<Aircraft> aircraftList, List<Airport> airports, List<Flight> flightList, List<Pairing> pairingList) {
        this.aircraftList =aircraftList;
        this.airportList = airports;
        this.flightList = flightList;
        this.pairingList = pairingList;
    }

    public List<Aircraft> getAircraftList(){
        return aircraftList;
    }

    public List<Airport> getAirportList(){
        return airportList;
    }

    public List<Flight> getFlightList(){
        return flightList;
    }

    public List<Pairing> getPairingList() {
        return pairingList;
    }


    public HardSoftScore getScore() {
        return score;
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

    public void printParingList(){
        for(Pairing pair : pairingList){
            for (Flight flight: pair.getPair()){
                System.out.print(flight.getFlightNumber()+" ");
            }
            System.out.println();
        }
    }
}

