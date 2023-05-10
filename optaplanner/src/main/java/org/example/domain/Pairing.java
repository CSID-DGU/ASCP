package org.example.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.example.domain.Flight;
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.valuerange.ValueRangeProvider;
import org.optaplanner.core.api.domain.variable.PlanningListVariable;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@PlanningEntity
public class Pairing {

    @PlanningListVariable(valueRangeProviderRefs = {"flightRange"})
    private List<Flight> pair = new ArrayList<>();
    private double totalCost;

    @ValueRangeProvider(id = "flightRange")
    public List<Flight> getPairRange() {
        List<Flight> validFlights = new ArrayList<>();

        // Implement your logic to generate valid flights
        // Consider time and airport constraints

        // Sort the flights based on their origin time
        pair.sort(Comparator.comparing(Flight::getOriginTime));

        // Iterate through the sorted flights
        for (int i = 0; i < pair.size(); i++) {
            Flight currentFlight = pair.get(i);

            // Check if the current flight is the last flight in the list
            if (i == pair.size() - 1) {
                validFlights.add(currentFlight);
                break;
            }

            Flight nextFlight = pair.get(i + 1);

            // Check if the airport constraint is satisfied
            if (currentFlight.getDestAirport().equals(nextFlight.getOriginAirport())) {
                // Check if the time constraint is satisfied
                if (currentFlight.getDestTime().isBefore(nextFlight.getOriginTime())) {
                    validFlights.add(currentFlight);
                }
            }
        }

        return validFlights;
    }

}
