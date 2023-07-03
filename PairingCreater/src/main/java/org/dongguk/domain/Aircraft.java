package org.dongguk.domain;

import lombok.*;

import java.util.List;
import java.util.Optional;

@Getter
@Setter
@Builder
@AllArgsConstructor
@RequiredArgsConstructor
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

    public static Aircraft findInAircraftName(List<Aircraft> aircrafts, String name) {
        return aircrafts.stream()
                .filter(temp -> temp.getName().equals(name))
                .findFirst()
                .orElse(null);
    }
}
