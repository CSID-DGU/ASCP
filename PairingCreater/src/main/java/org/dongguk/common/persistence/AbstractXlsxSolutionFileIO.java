package org.dongguk.common.persistence;

import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.xssf.usermodel.XSSFSheet;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.optaplanner.core.api.score.Score;
import org.optaplanner.persistence.common.api.domain.solution.SolutionFileIO;

import java.util.Iterator;


public abstract class AbstractXlsxSolutionFileIO<Solution_> implements SolutionFileIO<Solution_> {

    @Override
    public String getInputFileExtension() {
        return "xlsx";
    }

    public static abstract class AbstractXlsxReader<Solution_, Score_ extends Score<Score_>> {

        /**
         * 엑셀을 읽고 쓰기 위한 멤버 변수
         */
        protected final XSSFWorkbook workbook;
        protected XSSFSheet currentSheet;
        protected Iterator<Row> currentRowIterator;

        /**
         * [ScoreDirector]는 최적화 문제를 해결하는 데 사용되는 중요한 객체로서, 해를 탐색하고 성능 점수를 계산하며,
         * 이를 통해 탐색 과정에서 최적의 해를 찾기 위한 지침을 제공합니다.
         */
        // protected final ScoreDefinition<Score_> scoreDefinition;

        public AbstractXlsxReader(XSSFWorkbook workbook, String solverConfigResource) {
            this.workbook = workbook;
//            SolverFactory<Solution_> solverFactory = SolverFactory.createFromXmlResource(solverConfigResource);
//            ScoreDirectorFactory<Solution_> scoreDirectorFactory = ((DefaultSolverFactory<Solution_>) solverFactory).getScoreDirectorFactory();
//            scoreDefinition = ((InnerScoreDirectorFactory<Solution_, Score_>) scoreDirectorFactory).getScoreDefinition();
        }

        public abstract Solution_ read();

        protected void nextSheet(String sheetName) {
            currentSheet = workbook.getSheet(sheetName);
            if (currentSheet == null) {
                throw new IllegalStateException("The workbook does not contain a sheet with name ("
                        + sheetName + ").");
            }

            currentRowIterator = currentSheet.rowIterator();
            if (currentRowIterator == null) {
                throw new IllegalStateException("The sheet has no rows.");
            }
        }
    }

    public static abstract class AbstractXlsxWriter<Solution_, Score_ extends Score<Score_>> {
        protected Solution_ solution;
        public AbstractXlsxWriter(Solution_ solution, String solverConfigResource) {
            this.solution = solution;
        }

        public abstract void write();
    }
}
