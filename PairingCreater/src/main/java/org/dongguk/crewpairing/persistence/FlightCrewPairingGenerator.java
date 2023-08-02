package org.dongguk.crewpairing.persistence;

import org.dongguk.common.app.LoggingMain;
import org.dongguk.crewpairing.domain.*;
import org.optaplanner.persistence.common.api.domain.solution.SolutionFileIO;

import java.util.ArrayList;
import java.util.List;

public class FlightCrewPairingGenerator extends LoggingMain {
    private final SolutionFileIO<PairingSolution> solutionFileIO;

    public FlightCrewPairingGenerator() {
        this.solutionFileIO = new FlightCrewPairingXlsxFileIO();
    }

    public List<Pairing> createEntities(List<Flight> flightList) {
        //초기 페어링 Set 구성 어차피 [solver]가 바꿔버려서 의미 없음 아무것도 안넣으면 오류나서 넣는 것
        List<Pairing> pairingList = new ArrayList<>();
        for (int i = 0; i < flightList.size(); i++){
            List<Flight> pair = new ArrayList<>();
            pair.add(flightList.get(i));
            pairingList.add(new Pairing(i, pair,0));
        }

        return pairingList;
    }

    public void createOutput(PairingSolution solution) {
        solutionFileIO.write(solution, null);
    }
}
