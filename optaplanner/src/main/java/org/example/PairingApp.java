package org.example;

import org.example.domain.*;
import org.example.score.ParingConstraintProvider;
import org.optaplanner.core.api.solver.Solver;
import org.optaplanner.core.api.solver.SolverFactory;
import org.optaplanner.core.config.solver.SolverConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

public class PairingApp {

    private static final Logger LOGGER = LoggerFactory.getLogger(PairingApp.class);
    public static void main(String[] args) {
        SolverFactory<PairingSoultion> solverFactory = SolverFactory.create(new SolverConfig()
                .withSolutionClass(PairingSoultion.class)
                .withEntityClasses(Pairing.class)
                .withConstraintProviderClass(ParingConstraintProvider.class)
                .withTerminationSpentLimit(Duration.ofSeconds(5)));

        // Load the problem
        PairingSoultion problem = generateDemoData();

        // Solve the problem
        Solver<PairingSoultion> solver = solverFactory.buildSolver();
        PairingSoultion solution = solver.solve(problem);
        // Visualize the solution
        printPairing(solution);

        System.exit(0);
    }

    private static void printPairing(PairingSoultion pairingSoultion){

        List<Aircraft> aircraftList = pairingSoultion.getAircraftList();
        System.out.println(aircraftList.toString());
        System.out.println(pairingSoultion.getScore());

    }

    public static PairingSoultion generateDemoData() {



        List<Airport> airports = new ArrayList<>();
        Map<String,Integer>[] map = new Map[10];
        map[0] = new HashMap<>();
        map[0].put("LGA",100);
        map[0].put("DTW",200);
        map[0].put("MSP",230);
        map[0].put("SLC",380);

        map[1] = new HashMap<>();
        map[1].put("ATL",100);
        map[1].put("DTW",200);
        map[1].put("MSP",230);
        map[1].put("SLC",380);

        map[2] = new HashMap<>();
        map[2].put("ATL",100);
        map[2].put("LGA",200);
        map[2].put("MSP",230);
        map[2].put("SLC",380);

        map[3] = new HashMap<>();
        map[3].put("ATL",100);
        map[3].put("LGA",200);
        map[3].put("MSP",230);
        map[3].put("SLC",380);

        map[4] = new HashMap<>();
        map[4].put("ATL",100);
        map[4].put("LGA",200);
        map[4].put("DTW",230);
        map[4].put("MSP",380);


        airports.add(new Airport("ATL",map[0]));
        airports.add(new Airport("LGA",map[1]));
        airports.add(new Airport("DTW",map[2]));
        airports.add(new Airport("MSP",map[3]));
        airports.add(new Airport("SLC",map[4]));

        List<Aircraft> aircraftList = new ArrayList<>();
        aircraftList.add(new Aircraft("B767-300",8,184,3220,1610));
        aircraftList.add(new Aircraft("A321-100",6,130,2275,1137));
        aircraftList.add(new Aircraft("A321-200",6,130,2275,1137));
        aircraftList.add(new Aircraft("A320-200",5,110,1925,962));

        List<Flight> flightList = new ArrayList<>();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
        flightList.add(new Flight("F1",airports.get(4),LocalDateTime.parse("2020-01-03 17:30:00", formatter),airports.get(0),LocalDateTime.parse("2020-01-03 17:30:00", formatter),aircraftList.get(2)));
        flightList.add(new Flight("F2",airports.get(0),LocalDateTime.parse("2020-01-05 11:25:00", formatter),airports.get(2),LocalDateTime.parse("2020-01-05 13:21:00", formatter),aircraftList.get(1)));

        List<Pairing> pairingList = new ArrayList<>();
        List<Flight> pair1 = new ArrayList<>();
        pair1.add(flightList.get(0));
        pair1.add(flightList.get(1));
        pairingList.add(new Pairing(pair1,1000));
        return new PairingSoultion(aircraftList,airports,flightList,pairingList);
    }
}
