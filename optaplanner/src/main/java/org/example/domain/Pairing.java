package org.example.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.example.domain.Flight;
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.valuerange.ValueRangeProvider;
import org.optaplanner.core.api.domain.variable.PlanningListVariable;
import org.optaplanner.core.api.domain.variable.PlanningVariable;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;


@PlanningEntity
public class Pairing {

    public Pairing() {
        // 인자가 없는 생성자의 내용을 필요에 따라 추가
    }

    @PlanningVariable(valueRangeProviderRefs = {"pairing"})
    private List<Flight> pair = new ArrayList<>();
    private double totalCost;
    public Pairing(List<Flight> pair, int totalCost){
        this.pair=pair;
        this.totalCost = totalCost;
    }



    public List<Flight> getPair() {
        return pair;
    }

    public double getTotalCost() {
        return totalCost;
    }
}
