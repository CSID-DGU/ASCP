package org.dongguk.common.persistence;

import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.optaplanner.core.api.score.Score;
import org.optaplanner.core.api.solver.SolverFactory;
import org.optaplanner.core.impl.score.definition.ScoreDefinition;
import org.optaplanner.core.impl.score.director.InnerScoreDirectorFactory;
import org.optaplanner.core.impl.score.director.ScoreDirectorFactory;
import org.optaplanner.core.impl.solver.DefaultSolverFactory;
import org.optaplanner.persistence.common.api.domain.solution.SolutionFileIO;


public abstract class AbstractXlsxSolutionFileIO<Solution_> implements SolutionFileIO<Solution_> {

    @Override
    public String getInputFileExtension() {
        return "xlsx";
    }

    public static abstract class AbstractXlsxReader<Solution_, Score_ extends Score<Score_>> {
        // protected final XSSFWorkbook workbook;
        // [ScoreDirector]는 최적화 문제를 해결하는 데 사용되는 중요한 객체로서, 해를 탐색하고 성능 점수를 계산하며,
        // 이를 통해 탐색 과정에서 최적의 해를 찾기 위한 지침을 제공합니다.
//        protected final ScoreDefinition<Score_> scoreDefinition;

        public AbstractXlsxReader(String solverConfigResource) {
//            SolverFactory<Solution_> solverFactory = SolverFactory.createFromXmlResource(solverConfigResource);
//            ScoreDirectorFactory<Solution_> scoreDirectorFactory = ((DefaultSolverFactory<Solution_>) solverFactory).getScoreDirectorFactory();
//            scoreDefinition = ((InnerScoreDirectorFactory<Solution_, Score_>) scoreDirectorFactory).getScoreDefinition();
        }

        public abstract Solution_ read();
    }
}
