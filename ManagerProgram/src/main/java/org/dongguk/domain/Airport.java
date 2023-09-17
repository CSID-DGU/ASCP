package org.dongguk.domain;

import lombok.*;
import org.dongguk.AbstractPersistable;

import java.util.List;
import java.util.Map;

@Getter
@Setter
@AllArgsConstructor
public class Airport extends AbstractPersistable {
    // 공항 이름
    private String name;
    // 공항에 따른 호텔비용
    private int hotelCost;
    // 공항에 따른 Deadhead Cost 맵 ex) deadheadCost.get("ATL") -> 200
    private Map<String, Integer> deadheadCost;

    @Override
    public String toString() {
        return "Airport - " + name;
    }

    public static Airport of(List<Airport> airports, String name) {
        return airports.stream()
                .filter(airport -> airport.getName().equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Airport not found"));
    }

    @Builder
    public Airport(long id, String name, int hotelCost, Map<String, Integer> deadheadCost) {
        super(id);
        this.name = name;
        this.hotelCost = hotelCost;
        this.deadheadCost = deadheadCost;
    }

    public void putDeadhead(String name, Integer deadhead) {
        deadheadCost.put(name, deadhead);
    }
}
