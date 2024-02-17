package org.dongguk.common.app;

import lombok.Getter;
import lombok.Setter;
import org.dongguk.common.business.SolutionBusiness;
import org.optaplanner.core.api.solver.SolverFactory;
import org.optaplanner.core.config.solver.SolverConfig;
import org.optaplanner.core.config.solver.termination.TerminationConfig;
import org.optaplanner.persistence.common.api.domain.solution.SolutionFileIO;

import java.awt.*;
import java.io.File;

@Getter
@Setter
public abstract class CommonApp<Solution_> extends LoggingMain {
    // Data Directory 경로
    public static final String DATA_DIR_SYSTEM_PROPERTY = "org.dongguk.dataDir";

    protected final String name;
    protected final String description;
    protected final String solverConfigResource;
    protected final String dataDirPath;
    protected final String dataDirName;
    protected final String informationFileName;

    protected SolutionBusiness<Solution_, ?> solutionBusiness;
    protected SolverConfig solverConfig;

    // 우리가 사용할 Data Directory의 하위 경로 지정
    public File determineDataDir() {
        // 우리가 원하는 Data Directory 사용
        File dataDir = new File(dataDirPath, dataDirName);
        if (!dataDir.exists()) {
            throw new IllegalStateException(String.format("해당 Path [%s]에 Data Directory는 존재하지 않습니다", dataDir.getAbsolutePath()));
        }

        return dataDir;
    }


    // 생성자
    protected CommonApp(String name, String description, String solverConfigResource,
                        String dataDirPath, String dataDirName, String informationFileName) {
        this.name = name;
        this.description = description;
        this.solverConfigResource = solverConfigResource;
        this.dataDirPath = dataDirPath;
        this.dataDirName = dataDirName;
        this.informationFileName = informationFileName;
    }

    // 초기화 함수
    public CommonApp<Solution_> init(Integer flightSize) {
        init(null, true, flightSize);
        return this;
    }

    public void init(Component centerForComponent, boolean exitOnClose, Integer flightSize) {
        solutionBusiness = createSolutionBusiness(flightSize);
    }

    private SolutionBusiness<Solution_, ?> createSolutionBusiness(Integer flightSize) {
        // SolverConfig.xml을 읽어서 SolverConfig 객체를 생성 및 종료 조건 설정
        SolverConfig solverConfig = SolverConfig.createFromXmlResource(solverConfigResource);
        //solverConfig.withMoveThreadCount("1");
        //solverConfig.withTerminationConfig(
        //        new TerminationConfig()
        //                .withSecondsSpentLimit(0L));
                        //.withUnimprovedSecondsSpentLimit((long) (8.0 * Math.max(1.0, Math.log10(flightSize))))
                        //.withSecondsSpentLimit((long) (45.0 * Math.max(1.0, Math.log10(flightSize)))));

        // SolutionBusiness 객체 생성
        SolutionBusiness<Solution_, ?> solutionBusiness = new SolutionBusiness<>(this,
                SolverFactory.create(solverConfig));
        solutionBusiness.setDataDir(determineDataDir());
        solutionBusiness.setSolutionFileIO(createSolutionFileIO());
        solutionBusiness.updateDataDirs();
        return solutionBusiness;
    }

    // Solution File IO
    public abstract SolutionFileIO<Solution_> createSolutionFileIO();

    // 사용 유무 모르겠음
    // public interface ExtraAction<Solution_> extends BiConsumer<SolutionBusiness<Solution_, ?>, SolutionPanel<Solution_>> {
    //     String getName();
    // }
}
