package org.dongguk.crewpairing.util;

import lombok.extern.slf4j.Slf4j;
import org.dongguk.crewpairing.domain.PairingSolution;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.buildin.hardsoftlong.HardSoftLongScore;
import org.optaplanner.core.api.score.constraint.ConstraintMatchTotal;

import java.util.Map;

/**
 * solve 된 결과의 HardSoft score의 detail 확인하기 위한 클래스
 */
@Slf4j
public class ViewAllConstraint {
    /**
     * 각 제약 조건을 몇 번 어겼고, 얼마의 score를 만들어냈는지 확인
     * + 실제 가격이 얼마가 나왔는지 확인 가능
     * @param constraintMatchTotalMap
     * @param solution
     */
    public static void viewAll(Map<String, ConstraintMatchTotal<HardSoftLongScore>> constraintMatchTotalMap, PairingSolution solution){
        for (ConstraintMatchTotal<HardSoftLongScore> constraintMatchTotal : constraintMatchTotalMap.values()) {

            String constraintName = constraintMatchTotal.getConstraintName();
            int constraintMatchCount = constraintMatchTotal.getConstraintMatchCount();
            HardSoftLongScore constraintScore = constraintMatchTotal.getScore();
            long hardScore = constraintScore.hardScore();
            long softScore = constraintScore.softScore();
            long checkScore=0;

            if(hardScore!=0){
                checkScore=-hardScore;
            } else {
                checkScore=-softScore;
            }
            System.out.println("Constraint: " + constraintName + " -> cost " + checkScore +" (violated "+ constraintMatchCount+" times)");
        }

        System.out.println("Hard Score : " + solution.getScore().hardScore());
        System.out.println("Soft Score : " + solution.getScore().softScore());
    }
}
