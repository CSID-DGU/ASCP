package org.example.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class Aircraft {
    private String name;
    private int crewNum;
    private int flightSalary;
    private int baseSalary;
    private int layoverCost;

    public Aircraft(String name,int crewNum, int flightSalary, int baseSalary, int layoverCost){}

    public String getName() {
        return name;
    }

    public int getCrewNum() {
        return crewNum;
    }

    public int getFlightSalary() {
        return flightSalary;
    }

    public int getBaseSalary() {
        return baseSalary;
    }

    public int getLayoverCost() {
        return layoverCost;
    }
}
