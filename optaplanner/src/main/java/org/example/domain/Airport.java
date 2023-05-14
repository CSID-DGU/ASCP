package org.example.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

import java.util.Map;


public class Airport {
    private String name;
    private Map<String,Integer> deadheadCost;

    public Airport(String name,Map<String,Integer> deadheadCost){
        this.name = name;
        this.deadheadCost = deadheadCost;
    }

    public String getName() {
        return name;
    }

    public Map<String, Integer> getDeadheadCost() {
        return deadheadCost;
    }

    @Override
    public String toString() {
        return "Airport{" +
                "name='" + name + '\'' +
                ", deadheadCost=" + deadheadCost +
                '}';
    }
}
