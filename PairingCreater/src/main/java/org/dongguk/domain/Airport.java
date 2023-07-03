package org.dongguk.domain;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

import java.util.List;
import java.util.Map;

@Getter
@Setter
@Builder
@AllArgsConstructor
public class Airport {
    //공항 이름
    private String name;
    //공항에 따른 Deadhead Cost 맵 ex) deadheadCost.get("ATL") -> 200
    private Map<String, Integer> deadheadCost;

    @Override
    public String toString() {
        return "Airport{" +
                "name='" + name + '\'' +
                ", deadheadCost=" + deadheadCost +
                '}';
    }

    public static Airport findAirportByName(List<Airport> airports, String name) {
        return airports.stream()
                .filter(airport -> airport.getName().equals(name))
                .findFirst()
                .orElse(null);
    }
}
