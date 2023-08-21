package org.dongguk.crewpairing.score;

import org.dongguk.crewpairing.domain.Pairing;
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
                continuityPossible(constraintFactory),
                //aircraftType(constraintFactory),
                //landingTimes(constraintFactory),
                //pairLength(constraintFactory),
//                minBreakTime(constraintFactory),
//                pairMinLength(constraintFactory),
                movingWorkCost(constraintFactory),
                baseDiff(constraintFactory),
                layoverCost(constraintFactory),
                quickTurnCost(constraintFactory),
                hotelCost(constraintFactory),
                satisCost(constraintFactory)
        };
    }

    /**
     * HARD
     * 시간 제약(Flight possible):
     * TimeImpossible 어긴 제약 -> 하드스코어 부여(1000)
     */
    private Constraint timePossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::getTimeImpossible)
                .penalize(HardSoftLongScore.ofHard(1000))
                .asConstraint("Flight possible");
    }

    /**
     * HARD
     * 공간 제약(Airport possible):
     * AirportImpossible 어긴 제약 -> 하드스코어 부여(1000)
     */
    private Constraint airportPossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(Pairing::getAirportImpossible)
                .penalize(HardSoftLongScore.ofHard(1000))
                .asConstraint("Airport possible");
    }

    /**
     * HARD
     * 기종 제약(Same aircraft):
     * pairing의 항공기 기종이 다를 시 -> 하드스코어 부여(500)
     */
    private Constraint aircraftType(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter((Pairing::getAircraftDiff))
                .penalize(HardSoftLongScore.ofHard(500))
                .asConstraint("Same aircraft");
    }

    /**
     * HARD
     * 비행 횟수 제약(Landing times):
     * 비행 횟수가 4회 이상일 시 -> 하드스코어 부여(총 비행횟수 * 100)
     */
    @Deprecated
    private Constraint landingTimes(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() > 4)
                .penalize(HardSoftLongScore.ONE_HARD, pairing -> pairing.getPair().size() * 100)
                .asConstraint("Landing times");
    }

    /**
     * HARD
     * 비행 일수 제약(Max length):
     * 페어링 총 기간이 7일 이상일 시 -> 하드스코어 부여((총 길이-7) * 100)
     */
    @Deprecated
    private Constraint pairLength(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() >= 2 && pairing.getTotalLength() > 7)
                .penalize(HardSoftLongScore.ONE_HARD, pairing -> (int) (Math.floor(((double) pairing.getTotalLength() - 7)) * 100))
                .asConstraint("Max length");
    }

    /**
     * HARD
     * 비행 일수 제약(Max length):
     * 페어링 총 기간이 7일 이상일 시 -> 하드스코어 부여((총 길이-7) * 100)
     */
    private Constraint continuityPossible(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() >= 2)
                .filter(Pairing::getContinuityImpossible)
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

    /**
     * SOFT
     * deadhead cost 계산(Base diff):
     * 첫 출발공항과 마지막 도착공항이 다를 시 - > 소프트스코어 부여(항공편에 따른 가격)
     * @return getDeadheadCost
     */
    private Constraint baseDiff(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 1 && pairing.equalBase()))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getDeadHeadCost)
                .asConstraint("Base diff");
    }

    /**
     * SOFT
     * 총 layover cost 계산(Layover cost):
     * 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(layover 발생 시 cost+)
     * @return getLayoverCost
     */
    private Constraint layoverCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getLayoverCost)
                .asConstraint("Layover cost");
    }

    /**
     * SOFT
     * 총 이동근무 cost 계산(MovingWork cost):
     * 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(MovingWork cost 발생 시 cost+)
     * @return getMovingWorkCost
     */
    private Constraint movingWorkCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> pairing.getPair().size() >= 2)
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getMovingWorkCost)
                .asConstraint("MovingWork cost");
    }

    /**
     * SOFT
     * 총 QuickTurn cost 계산(QuickTurn cost):
     * 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(QuickTurn cost 발생 시 cost+)
     * @return getMovingWorkCost
     */
    private Constraint quickTurnCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getQuickTurnCost)
                .asConstraint("QuickTurn Cost");
    }

    /**
     * SOFT
     * 총 호텔숙박비 cost 계산(Hotel cost):
     * 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Hotel cost 발생 시 cost+)
     * @return getHotelCost
     */
    private Constraint hotelCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getHotelCost)
                .asConstraint("Hotel Cost");
    }

    /**
     * SOFT
     * 승무원 만족도 cost 계산(Satis cost):
     * 승무원의 휴식시간에 따른 만족도를 코스트로 score 부여
     * / 페어링 길이가 2 이상일 시 - > 소프트스코어 부여(Satis cost 발생 시 cost+)
     * @return getMovingWorkCost
     */
    private Constraint satisCost(ConstraintFactory constraintFactory) {
        return constraintFactory.forEach(Pairing.class)
                .filter(pairing -> (pairing.getPair().size() >= 2))
                .penalize(HardSoftLongScore.ONE_SOFT, Pairing::getSatisCost)
                .asConstraint("Satis cost");
    }
}
