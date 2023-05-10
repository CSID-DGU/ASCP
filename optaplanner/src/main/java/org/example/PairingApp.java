package org.example;

import io.vertx.core.impl.logging.LoggerFactory;
import org.example.domain.Aircraft;
import org.example.domain.Pairing;
import org.example.domain.PairingSoultion;
import org.example.score.ParingConstraintProvider;
import org.optaplanner.core.api.solver.Solver;
import org.optaplanner.core.api.solver.SolverFactory;
import org.optaplanner.core.config.solver.SolverConfig;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

public class PairingApp {
    public static void main(String[] args) {
        SolverFactory<PairingSoultion> solverFactory = SolverFactory.create(new SolverConfig()
                .withSolutionClass(PairingSoultion.class)
                .withEntityClasses(Pairing.class)
                .withConstraintProviderClass(ParingConstraintProvider.class)
                // The solver runs only for 5 seconds on this small dataset.
                // It's recommended to run for at least 5 minutes ("5m") otherwise.
                .withTerminationSpentLimit(Duration.ofMinutes(5)));

        // Load the problem
        PairingSoultion problem = generateDemoData();

        // Solve the problem
        Solver<PairingSoultion> solver = solverFactory.buildSolver();
        PairingSoultion solution = solver.solve(problem);

        // Visualize the solution
        //printTimetable(solution);
    }

    public static PairingSoultion generateDemoData() {
        List<Aircraft> aircraftList = new ArrayList<>();
        aircraftList.add(new Aircraft("B767-300",8,184,3220,1610));
        aircraftList.add(new Aircraft("A321-100",6,130,2275,1137));
        aircraftList.add(new Aircraft("A321-200",6,130,2275,1137));
        aircraftList.add(new Aircraft("A320-200",5,110,1925,962));



        return new PairingSoultion();
    }
}
