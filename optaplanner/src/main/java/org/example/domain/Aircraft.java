package org.example.domain;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
@AllArgsConstructor
public class Aircraft {
    private String name;
    private int crewNum;
    private int flightSalary;
    private int baseSalary;
    private int layoverCost;

}
