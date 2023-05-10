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

    @PlanningListVariable
    private List<Flight> pair = new ArrayList<>();
    private double totalCost;

    public List<Flight> getPair() {
        return pair;
    }

    public double getTotalCost() {
        return totalCost;
    }
}
