package org.dongguk.crewpairing.util;

import org.dongguk.crewpairing.domain.PairingSolution;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;
import org.optaplanner.core.api.score.constraint.ConstraintMatchTotal;

import java.util.Map;

public class ViewAllConstraint {

    public static void viewAll(Map<String, ConstraintMatchTotal<HardSoftScore>> constraintMatchTotalMap, PairingSolution solution){
        for (ConstraintMatchTotal<HardSoftScore> constraintMatchTotal : constraintMatchTotalMap.values()) {

            String constraintName = constraintMatchTotal.getConstraintName();
            int constraintMatchCount = constraintMatchTotal.getConstraintMatchCount();
            HardSoftScore constraintScore = constraintMatchTotal.getScore();
            int hardScore = constraintScore.hardScore();
            int softScore = constraintScore.softScore();
            int checkScore=0;

            if(hardScore!=0){
                checkScore=-hardScore;
            } else {
                checkScore=-softScore;
            }
            System.out.println("Constraint: " + constraintName + "-> cost " + checkScore +" (violated "+ constraintMatchCount+" times)");
        }
        System.out.println("total cost: "+(-solution.getScore().softScore()));
    }
}
