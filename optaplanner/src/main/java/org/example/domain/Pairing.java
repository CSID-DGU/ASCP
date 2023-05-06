package org.example.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.example.domain.Flight;
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.variable.PlanningListVariable;
import org.optaplanner.core.api.score.buildin.hardsoftlong.HardSoftLongScore;
import org.optaplanner.core.api.score.stream.Constraint;
import org.optaplanner.core.api.score.stream.ConstraintFactory;

import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@PlanningEntity
public class Pairing {

    @PlanningListVariable
    private List<Flight> pair = new ArrayList<>();
    private double totalCost;

    public Flight getFirstFlight(){
        return pair.get(0);
    }
    public Flight getLastFlight(){
        return pair.get(-1);
    }

}
