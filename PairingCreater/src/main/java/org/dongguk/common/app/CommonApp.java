package org.dongguk.common.app;

import lombok.Getter;
import lombok.Setter;
import org.dongguk.common.business.SolutionBusiness;
import org.optaplanner.core.api.solver.SolverFactory;
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
    protected final String dataDirName;

    protected SolutionBusiness<Solution_, ?> solutionBusiness;

    // 우리가 사용할 Data Directory의 하위 경로 지정
    public static File determineDataDir(String dataDirName) {
        // 만약 기존경로가 없다면 프로젝트 내 data Dir을 사용함
        String dataDirPath = System.getProperty(DATA_DIR_SYSTEM_PROPERTY, "data/");

        // 우리가 원하는 Data Directory 사용
        File dataDir = new File(dataDirPath, dataDirName);
        if (!dataDir.exists()) {
            throw new IllegalStateException(String.format("해당 Path [%s]에 Data Directory는 존재하지 않습니다", dataDir.getAbsolutePath()));
        }

        return dataDir;
    }

    // 생성자
    protected CommonApp(String name, String description, String solverConfigResource, String dataDirName) {
        this.name = name;
        this.description = description;
        this.solverConfigResource = solverConfigResource;
        this.dataDirName = dataDirName;
    }

    // 초기화 함수
    public CommonApp<Solution_> init() {
        init(null, true);
        return this;
    }

    public void init(Component centerForComponent, boolean exitOnClose) {
        solutionBusiness = createSolutionBusiness();
    }

    private SolutionBusiness<Solution_, ?> createSolutionBusiness() {
        SolutionBusiness<Solution_, ?> solutionBusiness = new SolutionBusiness<>(this,
                SolverFactory.createFromXmlResource(solverConfigResource));
        solutionBusiness.setDataDir(determineDataDir(dataDirName));
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
