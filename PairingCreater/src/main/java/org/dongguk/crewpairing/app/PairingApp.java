package org.dongguk.crewpairing.app;

import org.dongguk.common.app.CommonApp;
import org.dongguk.common.business.SolutionBusiness;
import org.dongguk.crewpairing.domain.*;
import org.dongguk.crewpairing.persistence.FlightCrewPairingGenerator;
import org.dongguk.crewpairing.persistence.FlightCrewPairingXlsxFileIO;
import org.dongguk.crewpairing.util.ViewAllConstraint;
import org.optaplanner.core.api.score.ScoreExplanation;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.constraint.ConstraintMatchTotal;
import org.optaplanner.core.api.solver.SolutionManager;
import org.optaplanner.core.api.solver.Solver;
import org.optaplanner.core.api.solver.SolverFactory;
import org.optaplanner.persistence.common.api.domain.solution.SolutionFileIO;

import java.util.*;

public class PairingApp extends CommonApp<PairingSolution> {
    public static final String SOLVER_CONFIG = "airlineCrewSchedulingSolverConfig.xml";
    public static final String DATA_DIR_NAME = "crewpairing";

    public static void main(String[] args) {
        SolutionBusiness<PairingSolution, ?> business = new PairingApp().init().getSolutionBusiness();

        // Input Xlsx File
        business.openSolution(null);

        // Solve By SolverJob
        business.solve(business.getSolution());

        // Solution 출력
        PairingSolution solution = business.getSolution();
        System.out.println(business.getSolution());

        // Output CSV File
        business.saveSolution(null);

        // Check score detail
        SolutionManager<PairingSolution, HardSoftScore> scoreManager = SolutionManager.create(business.getSolverFactory());
        ScoreExplanation<PairingSolution, HardSoftScore> explain = scoreManager.explain(solution);
        Map<String, ConstraintMatchTotal<HardSoftScore>> constraintMatchTotalMap = explain.getConstraintMatchTotalMap();
        ViewAllConstraint.viewAll(constraintMatchTotalMap, solution);

        System.exit(0);
    }

    public PairingApp() {
        super("CrewPairing",
                "Airline Scheduling Crew Pairing",
                SOLVER_CONFIG,
                DATA_DIR_NAME);
    }

    @Override
    public SolutionFileIO<PairingSolution> createSolutionFileIO() {
        return new FlightCrewPairingXlsxFileIO();
    }
}
