package org.dongguk.crewpairing.domain;

import lombok.*;
import org.dongguk.common.domain.AbstractPersistable;

import java.util.List;
import java.util.Map;

@Getter
@Setter
@AllArgsConstructor
@RequiredArgsConstructor
public class Airport extends AbstractPersistable {
    //공항 이름
    private String name;
    //공항에 따른 Deadhead Cost 맵 ex) deadheadCost.get("ATL") -> 200
    private Map<String, Integer> deadheadCost;

    @Override
    public String toString() {
        return "Airport - " + name;
    }

    public static Airport findAirportByName(List<Airport> airports, String name) {
        return airports.stream()
                .filter(airport -> airport.getName().equals(name))
                .findFirst()
                .orElse(null);
    }

    @Builder
    public Airport(long id, String name, Map<String, Integer> deadheadCost) {
        super(id);
        this.name = name;
        this.deadheadCost = deadheadCost;
    }
}
