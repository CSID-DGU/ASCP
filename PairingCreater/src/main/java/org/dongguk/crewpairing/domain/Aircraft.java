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
    private String type;
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
        return "Aircraft - " + type;
    }

    public static Aircraft of(List<Aircraft> aircrafts, String name) {
        return aircrafts.stream()
                .filter(temp -> temp.getType().equals(name))
                .findFirst()
                .orElse(null);
    }

    @Builder
    public Aircraft(long id, String type, int crewNum, int flightSalary, int baseSalary, int layoverCost) {
        super(id);
        this.type = type;
        this.crewNum = crewNum;
        this.flightSalary = flightSalary;
        this.baseSalary = baseSalary;
        this.layoverCost = layoverCost;
    }
}
