package org.dongguk.score;

import org.dongguk.domain.Pairing;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.stream.Constraint;
import org.optaplanner.core.api.score.stream.ConstraintFactory;
import org.optaplanner.core.api.score.stream.ConstraintProvider;

public class ParingConstraintProvider implements ConstraintProvider {
    @Override
    public Constraint[] defineConstraints(ConstraintFactory constraintFactory) {
        return new Constraint[]{
                timePossible(constraintFactory),
                airportPossible(constraintFactory),
                aircraftType(constraintFactory),
                landingTimes(constraintFactory),
                pairLength(constraintFactory),
//                pairMinLength(constraintFactory),
                baseDiff(constraintFactory)
        };
    }

    //시간 제약조건
    private Constraint timePossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getTimeImpossible())
                .penalize(HardSoftScore.ofHard(1000))
                .asConstraint("Flight possible");
    }

    //공간 제약 조건
    private Constraint airportPossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getAirportImpossible())
                .penalize(HardSoftScore.ofHard(1000))
                .asConstraint("Airport possible");
    }

    //같은 기종
    private Constraint aircraftType(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter((paring -> paring.getAircraftImpossible()))
                .penalize(HardSoftScore.ofHard(100))
                .asConstraint("Same aircraft");
    }

    //최대 랜딩 횟수
    private Constraint landingTimes(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() > 4)
                .penalize(HardSoftScore.ONE_HARD, pairing -> pairing.getPair().size() * 100)
                .asConstraint("Landing times");
    }

    //페어링 최대 길이
    private Constraint pairLength(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() >= 2 && pairing.getTotalLength() > 7)
                .penalize(HardSoftScore.ONE_HARD, pairing -> (int) (Math.floor(((double) pairing.getTotalLength() - 7)) * 100))
                .asConstraint("Max Length");
    }

    //페어링 최소 길이
//    private Constraint pairMinLength(ConstraintFactory constraintFactory) {
//        return constraintFactory.forEach(Pairing.class)
//                .filter(pairing -> pairing.getPair().size() == 0)
//                .penalize(HardSoftScore.ofHard(100))
//                .asConstraint("Min Length");
//    }

    //base 같아야 함 (soft) == Deadhead
    private Constraint baseDiff(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 1 && !pairing.isBaseSame()))
                .penalize(HardSoftScore.ONE_SOFT, pairing -> pairing.getDeadHeadCost())
                .asConstraint("base Diff");
    }
}
