package org.dongguk.domain;

import lombok.*;
import org.dongguk.AbstractPersistable;

import java.util.List;

@Getter
@Setter
public class Aircraft extends AbstractPersistable {
    // 기종 이름
    final private String type;
    // 기종 크루원 수
    final private int crewNum;
    // 비행 수당
    final private int flightCost;
    // Layover 비용
    final private int layoverCost;
    // QuickTurn 비용
    final private int quickTurnCost;

    @Override
    public String toString() {
        return "Aircraft - " + type;
    }

    public static Aircraft of(final List<Aircraft> aircrafts, final String name) {
        return aircrafts.stream()
                .filter(temp -> temp.getType().equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Aircraft not found"));
    }

    @Builder
    public Aircraft(long id, String type, int crewNum, int flightCost, int layoverCost, int quickTurnCost) {
        this.id = id;
        this.type = type;
        this.crewNum = crewNum;
        this.flightCost = flightCost;
        this.layoverCost = layoverCost;
        this.quickTurnCost = quickTurnCost;
    }
}
