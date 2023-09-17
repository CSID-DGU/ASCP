package org.dongguk.crewpairing.domain.factory;

import org.dongguk.crewpairing.domain.Aircraft;
import org.dongguk.crewpairing.domain.Airport;
import org.dongguk.crewpairing.domain.Flight;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DomainFactory {
    private static final List<Aircraft> aircraftList = new ArrayList<>();
    private static final List<Airport> airportList = new ArrayList<>();
    private static final List<Flight> flightList = new ArrayList<>();

    public static void addAircraft(Aircraft aircraft) {
        aircraftList.add(aircraft);
    }

    public static void addAllAirport(Map<String, Airport> airports) {
        airportList.addAll(airports.values());
    }

    public static void addFlight(Flight flight) {
        flightList.add(flight);
    }

    public static Aircraft getAircraft(String name) {
        return aircraftList.stream()
                .filter(aircraft -> aircraft.getType().equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Aircraft not found"));
    }

    public static Airport getAirport(String name) {
        return airportList.stream()
                .filter(airport -> airport.getName().equals(name))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Airport not found"));
    }

    public static Flight getFlight(String serialNumber) {
        return flightList.stream()
                .filter(flight -> flight.getSerialNumber().equals(serialNumber))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException("Flight not found"));
    }

    public static List<Aircraft> getAircraftList() {
        return aircraftList;
    }

    public static List<Airport> getAirportList() {
        return airportList;
    }

    public static List<Flight> getFlightList() {
        return flightList;
    }
}
