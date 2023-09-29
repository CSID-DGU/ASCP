package org.dongguk.crewpairing.app;

import lombok.extern.slf4j.Slf4j;
import org.dongguk.common.app.CommonApp;
import org.dongguk.common.business.SolutionBusiness;
import org.dongguk.crewpairing.domain.*;
import org.dongguk.crewpairing.persistence.FlightCrewPairingXlsxFileIO;
import org.dongguk.crewpairing.util.ViewAllConstraint;
import org.optaplanner.core.api.score.ScoreExplanation;
import org.optaplanner.core.api.score.buildin.hardsoftlong.HardSoftLongScore;
import org.optaplanner.core.api.score.constraint.ConstraintMatchTotal;
import org.optaplanner.core.api.solver.SolutionManager;
import org.optaplanner.persistence.common.api.domain.solution.SolutionFileIO;

import java.util.*;

@Slf4j
public class PairingApp extends CommonApp<PairingSolution> {
    public static final String SOLVER_CONFIG = "solverConfig.xml";

    public static void main(String[] args) {
        String dataDirPath = args.length > 0 ? args[0] : null;
        String dataDirName = args.length > 1 ? args[1] : null;
        String flightSize = args.length > 2 ? args[2] : null;
        String informationXlsxFile = args.length > 3 ? args[3] : null;
        String pairingXlsxFile = args.length > 4 ? args[4] : null;

        assert flightSize != null;
        SolutionBusiness<PairingSolution, ?> business = new PairingApp(dataDirPath, dataDirName, informationXlsxFile)
                .init(Integer.valueOf(flightSize)).getSolutionBusiness();

        // Input Information Xlsx File
        business.openSolution(
                business.getInputFileList()
                        .stream()
                        .filter(inputFile -> inputFile.getName().equals(informationXlsxFile))
                        .findFirst().orElseThrow(() -> new IllegalArgumentException("파일이 존재하지 않습니다.")));

        if (pairingXlsxFile != null) {
            FlightCrewPairingXlsxFileIO xlsxFileIO = new FlightCrewPairingXlsxFileIO();
            List<Flight> flightList = business.getSolution().getFlightList();
            List<Pairing> pairingList = xlsxFileIO.readPairingList(flightList, business.getOutputFileList()
                    .stream()
                    .filter(outputFile -> outputFile.getName().equals(pairingXlsxFile))
                    .findFirst().orElseGet(() -> null));
            business.getSolution().setPairingList(pairingList);
        }

        // Solve By SolverJob
        business.solve(business.getSolution());

        // Solution 출력
        PairingSolution solution = business.getSolution();

        // Output Excel File
        business.saveSolution(null);

        // Check score detail
        SolutionManager<PairingSolution, HardSoftLongScore> scoreManager = SolutionManager.create(business.getSolverFactory());
        ScoreExplanation<PairingSolution, HardSoftLongScore> explain = scoreManager.explain(solution);
        Map<String, ConstraintMatchTotal<HardSoftLongScore>> constraintMatchTotalMap = explain.getConstraintMatchTotalMap();
        ViewAllConstraint.viewAll(constraintMatchTotalMap, solution);
//        ViewAllConstraint.pairingScore(explain);

        System.exit(0);
    }

    public PairingApp(String dataDirPath, String dataDirName, String informationFileName) {
        super("CrewPairing",
                "Airline Scheduling Crew Pairing",
                SOLVER_CONFIG,
                dataDirPath,
                dataDirName,
                informationFileName);
    }

    @Override
    public SolutionFileIO<PairingSolution> createSolutionFileIO() {
        return new FlightCrewPairingXlsxFileIO();
    }
}
