package org.example.optional.score;

import org.example.domain.Pairing;
import org.example.domain.PairingSoultion;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;

import org.optaplanner.core.api.score.calculator.EasyScoreCalculator;

import java.util.List;


public class PairingEasyScoreCalculator implements EasyScoreCalculator<PairingSoultion, HardSoftScore> {
    @Override
    public HardSoftScore calculateScore(PairingSoultion solution) {
        List<Pairing> pairingList = solution.getPairingList();

        int hardScore = 0;
        int softScore = 0;

        for (Pairing pairing : pairingList) {
            if (pairing.getTotalCost() > solution.getCostAverage() * 2) {
                hardScore -= (pairing.getTotalCost() - solution.getCostAverage());
            } else if (pairing.getTotalCost() > solution.getCostAverage()) {
                softScore -= (pairing.getTotalCost() - solution.getCostAverage());
            }
        }


        return HardSoftScore.of(hardScore, softScore);
    }
}
