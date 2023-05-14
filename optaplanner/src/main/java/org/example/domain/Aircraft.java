package org.example.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;


public class Aircraft {
    private String name;
    private int crewNum;
    private int flightSalary;
    private int baseSalary;
    private int layoverCost;

    public Aircraft(String name,int crewNum, int flightSalary, int baseSalary, int layoverCost){
        this.name = name;
        this.crewNum = crewNum;
        this.flightSalary = flightSalary;
        this.baseSalary = baseSalary;
        this.layoverCost = layoverCost;
    }

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

    @Override
    public String toString() {
        return "Aircraft{" +
                "name='" + name + '\'' +
                ", crewNum=" + crewNum +
                ", flightSalary=" + flightSalary +
                ", baseSalary=" + baseSalary +
                ", layoverCost=" + layoverCost +
                '}';
    }
}
