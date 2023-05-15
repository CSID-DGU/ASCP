package org.example.score;

import org.example.domain.Flight;
import org.example.domain.Pairing;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.buildin.hardsoftlong.HardSoftLongScore;
import org.optaplanner.core.api.score.stream.Constraint;
import org.optaplanner.core.api.score.stream.ConstraintFactory;
import org.optaplanner.core.api.score.stream.ConstraintProvider;
import org.optaplanner.core.api.score.stream.Joiners;

import java.time.temporal.ChronoUnit;

import static org.optaplanner.core.api.score.stream.ConstraintCollectors.count;
import static org.optaplanner.core.api.score.stream.ConstraintCollectors.countDistinct;
import static org.optaplanner.core.api.score.stream.Joiners.*;

public class ParingConstraintProvider implements ConstraintProvider {

    @Override
    public Constraint[] defineConstraints(ConstraintFactory constraintFactory) {
        return new Constraint[] {
                timePossible(constraintFactory),
                aircraftType(constraintFactory),
                flightConflict(constraintFactory),
                //pairLength(constraintFactory),
                baseDiff(constraintFactory)
        };
    }

    //flight 1개 이상
    private Constraint timePossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getTimeImpossbile())
                .penalize(HardSoftScore.ofHard(100))
                .asConstraint("flight possible");
    }

    //같은 기종
    private Constraint aircraftType(ConstraintFactory constraintFactory){
        return constraintFactory.forEach(Pairing.class)
                .groupBy(pairing -> pairing.getPair().get(0).getAircraft().getName(),countDistinct())
                .filter((aircraftType,count) -> count > 1)
                .penalize(HardSoftScore.ofHard(10))
                .asConstraint("same aircraft");
    }

    //최대 랜딩 횟수
    private Constraint flightConflict(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size()>4)
                .penalize(HardSoftScore.ofHard(10))
                .asConstraint("Landing conflict");
    }

    //페어링 최대 길이
    private Constraint pairLength(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> ChronoUnit.DAYS.between(pairing.getPair().get(0).getOriginTime(),pairing.getPair().get(pairing.getPair().size()-1).getDestTime()) > 7)
                .penalize(HardSoftScore.ofHard(10))
                .asConstraint("Max Length");
    }

    //base 같아야 함 (soft)
    private Constraint baseDiff(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().get(0).getOriginAirport() != pairing.getPair().get(pairing.getPair().size()-1).getDestAirport())
                .penalize(HardSoftScore.ofSoft(100))
                .asConstraint("base Diff");
    }



}
