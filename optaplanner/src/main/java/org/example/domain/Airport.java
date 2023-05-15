package org.example.domain;



import java.util.List;
import java.util.Map;


public class Airport {
    private String name;
    private Map<String, Integer> deadheadCost;

    public Airport(String name, Map<String, Integer> deadheadCost) {
        this.name = name;
        this.deadheadCost = deadheadCost;
    }

    public String getName() {
        return name;
    }

    public Map<String, Integer> getDeadheadCost() {
        return deadheadCost;
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
