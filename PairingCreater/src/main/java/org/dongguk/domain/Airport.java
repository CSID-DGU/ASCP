package org.dongguk.domain;

import lombok.*;

import java.util.List;
import java.util.Map;

@Getter
@Setter
@Builder
@AllArgsConstructor
@RequiredArgsConstructor
public class Airport {
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
}
