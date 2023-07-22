package org.dongguk.crewpairing.app;

import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.dongguk.crewpairing.domain.*;
import org.dongguk.crewpairing.persistence.FlightCrewPairingGenerator;
import org.dongguk.crewpairing.util.PairingVisualize;
import org.drools.io.ClassPathResource;
import org.optaplanner.core.api.solver.Solver;
import org.optaplanner.core.api.solver.SolverFactory;

import java.io.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PairingApp {
    public static String SOVLER_CONFIG = "solverConfig.xml";
    public static void main(String[] args) {
        SolverFactory<PairingSolution> solverFactory = SolverFactory.createFromXmlResource("solverConfig.xml");
        FlightCrewPairingGenerator generator = new FlightCrewPairingGenerator();

        // Load the problem
        PairingSolution problem = generator.create(40);

        // Solve the problem
        Solver<PairingSolution> solver = solverFactory.buildSolver();
        PairingSolution solution = solver.solve(problem);

        // Visualize the solution
        System.out.println(solution);
        
        // OutPut Excel
        PairingVisualize.visualize(solution.getPairingList());

        System.exit(0);
    }
}
