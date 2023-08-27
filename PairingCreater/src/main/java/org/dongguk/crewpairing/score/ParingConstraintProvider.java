package org.dongguk.crewpairing.score;

import org.dongguk.crewpairing.domain.Pairing;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.buildin.hardsoftlong.HardSoftLongScore;
import org.optaplanner.core.api.score.stream.Constraint;
import org.optaplanner.core.api.score.stream.ConstraintFactory;
import org.optaplanner.core.api.score.stream.ConstraintProvider;

public class ParingConstraintProvider implements ConstraintProvider {

    //시간 최소시간
    @Override
    public Constraint[] defineConstraints(ConstraintFactory constraintFactory) {
        return new Constraint[]{
                timePossible(constraintFactory),
                airportPossible(constraintFactory),
                lawPossible(constraintFactory),
                //aircraftType(constraintFactory),
                landingTimes(constraintFactory),
                pairLength(constraintFactory),
//                minBreakTime(constraintFactory),
//                pairMinLength(constraintFactory),
                movingWorkCost(constraintFactory),
                baseDiff(constraintFactory),
                layoverCost(constraintFactory),
                quickTurnCost(constraintFactory),
                hotelCost(constraintFactory),
                satisCost(constraintFactory),
//                testCost(constraintFactory)
        };
    }

    //시간 제약조건
    private Constraint timePossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::getTimeImpossible)
                .penalize(HardSoftLongScore.ofHard(1000))
                .asConstraint("Flight possible");
    }

    //공간 제약 조건
    private Constraint airportPossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::getAirportImpossible)
                .penalize(HardSoftLongScore.ofHard(1000))
                .asConstraint("Airport possible");
    }

    //같은 기종
    private Constraint aircraftType(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter((Pairing::getAircraftDiff))
                .penalize(HardSoftLongScore.ofHard(500))
                .asConstraint("Same aircraft");
    }

    //최대 랜딩 횟수
    private Constraint landingTimes(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() > 4)
                .penalize(HardSoftLongScore.ONE_HARD, pairing -> pairing.getPair().size() * 100)
                .asConstraint("Landing times");
    }

    //페어링 최대 길이
    private Constraint pairLength(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() >= 2 && pairing.getTotalLength() > 7)
                .penalize(HardSoftLongScore.ONE_HARD, pairing -> (int) (Math.floor(((double) pairing.getTotalLength() - 7)) * 100))
                .asConstraint("Max length");
    }

    private Constraint lawPossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() >= 2)
                .filter(Pairing::getLawImpossible)
                .penalize(HardSoftLongScore.ofHard(1000))
                .asConstraint("law possible");
    }
/*
    private Constraint minBreakTime(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .filter((Pairing::minBreakTime))
                .penalize(HardSoftLongScore.ofHard(500))
                .asConstraint("Break Time");
    }
 */

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
                .filter(pairing -> (pairing.getPair().size() >= 1 && pairing.equalBase()))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getDeadHeadCost)
                .asConstraint("Base diff");
    }

    private Constraint layoverCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getLayoverCost)
                .asConstraint("Layover cost");
    }

    private Constraint movingWorkCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() >= 2)
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getMovingWorkCost)
                .asConstraint("MovingWork cost");
    }

    private Constraint quickTurnCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getQuickTurnCost)
                .asConstraint("QuickTurn Cost");
    }

    private Constraint hotelCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getHotelCost)
                .asConstraint("Hotel Cost");
    }

    private Constraint satisCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getSatisCost)
                .asConstraint("Satis cost");
    }

    //모든 1개 이상인 페어링에 soft 점수를 부여해서 페어링의 수가 줄어드는지 실험
    private Constraint testCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 1))
                .penalize(HardSoftLongScore.ofSoft(1000000))
                .asConstraint("Test cost");
    }

}
