package org.example.domain;



import java.util.List;
import java.util.Map;


public class Airport {
    //공항 이름
    private String name;
    //공항에 따른 deadhead비용 맵 ex) deadheadCost.get("ATL") -> 200
    private Map<String, Integer> deadheadCost;

    public Airport(String name, Map<String, Integer> deadheadCost) {
        this.name = name;
        this.deadheadCost = deadheadCost;
    }

    public String getName() {
        return name;
    }

    public int getDeadheadCost(String s) {
        return deadheadCost.get(s);
    }

    @Override
    public String toString() {
        return "Airport{" +
                "name='" + name + '\'' +
                ", deadheadCost=" + deadheadCost +
                '}';
    }

    public static Airport findAirportByName(List<Airport> airports, String name) {
        for (Airport airport : airports) {
            if (airport.getName().equals(name)) {
                return airport;
            }
        }
        return null;
    }

}
