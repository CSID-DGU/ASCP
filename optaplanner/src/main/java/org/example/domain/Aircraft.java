package org.example.domain;


import java.util.List;

public class Aircraft {
    //기종 이름
    private String name;
    //기종 크루원 수
    private int crewNum;
    //비행 수당
    private int flightSalary;
    //기본급
    private int baseSalary;
    //layover 비용
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

    public static Aircraft findAircraftName(List<Aircraft> aircrafts, String name) {
        for (Aircraft aircraft : aircrafts) {
            if (aircraft.getName().equals(name)) {
                return aircraft;
            }
        }
        return null;
    }


}
