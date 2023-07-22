package org.dongguk.crewpairing.domain;

import lombok.*;
import org.dongguk.common.domain.AbstractPersistable;

import java.util.List;

@Getter
@Setter
@AllArgsConstructor
@RequiredArgsConstructor
public class Aircraft extends AbstractPersistable {
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
        return "Aircraft - " + name;
    }

    public static Aircraft findInAircraftName(List<Aircraft> aircrafts, String name) {
        return aircrafts.stream()
                .filter(temp -> temp.getName().equals(name))
                .findFirst()
                .orElse(null);
    }

    @Builder
    public Aircraft(long id, String name, int crewNum, int flightSalary, int baseSalary, int layoverCost) {
        super(id);
        this.name = name;
        this.crewNum = crewNum;
        this.flightSalary = flightSalary;
        this.baseSalary = baseSalary;
        this.layoverCost = layoverCost;
    }
}
