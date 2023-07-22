package org.dongguk.domain;

import com.github.javaparser.utils.Pair;
import lombok.*;
import org.dongguk.app.PairingVisualize;
import org.optaplanner.core.api.domain.solution.PlanningEntityCollectionProperty;
import org.optaplanner.core.api.domain.solution.PlanningScore;
import org.optaplanner.core.api.domain.solution.PlanningSolution;
import org.optaplanner.core.api.domain.solution.ProblemFactCollectionProperty;
import org.optaplanner.core.api.domain.valuerange.ValueRangeProvider;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;

import java.util.List;

@Getter
@Setter
@Builder
@AllArgsConstructor
@RequiredArgsConstructor
@PlanningSolution
public class PairingSolution {
    //Aircraft 에 대한 모든 정보
    @ProblemFactCollectionProperty
    private List<Aircraft> aircraftList;

    //Airport 에 대한 모든 정보
    @ProblemFactCollectionProperty
    private List<Airport> airportList;

    //비행편에 대한 모든 정보 / 변수로서 작동 되므로 ValueRangeProvider 필요
    @ValueRangeProvider(id = "pairing")
    @ProblemFactCollectionProperty
    private List<Flight> flightList;

    // solver 가 풀어낸 Entity 들
    @PlanningEntityCollectionProperty
    private List<Pairing> pairingList;

    //score 변수
    @PlanningScore
    private HardSoftScore score = null;

    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();

        builder.append("Score = ").append(score).append("\n");
        for (Pairing pairing : pairingList) {
            builder.append(pairing.toString()).append(PairingVisualize.date2String(pairing)).append("\n");
        }

        return builder.toString();
    }
}
