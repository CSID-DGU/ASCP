package org.example.domain;


import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.valuerange.ValueRangeProvider;
import org.optaplanner.core.api.domain.variable.PlanningListVariable;
import org.optaplanner.core.api.domain.variable.PlanningVariable;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;


@PlanningEntity
public class Pairing {

    public Pairing() {
        // 인자가 없는 생성자의 내용을 필요에 따라 추가
    }
    public Pairing(List<Flight> pair, int totalCost){
        this.pair=pair;
        this.totalCost = totalCost;
    }

    @PlanningListVariable(valueRangeProviderRefs = {"pairing"})
    private List<Flight> pair = new ArrayList<>();

    //pair가 시간상 불가능하면 true를 반환
    public boolean getTimeImpossbile(){
        for(int i=0;i<pair.size()-1;i++){
            if(pair.get(i).getDestTime().isAfter(pair.get(i+1).getOriginTime())){
                return true;
            }
        }
        return false;
    }

    public List<Flight> getPair() {
        return pair;
    }
    private double totalCost;
    public double getTotalCost() {
        return totalCost;
    }

    @Override
    public String toString() {
        return "Pairing{" +
                "pair=" + pair +
                ", totalCost=" + totalCost +
                '}';
    }
}
