package org.dongguk.crewpairing.domain;

import lombok.*;
import org.dongguk.common.domain.AbstractPersistable;
import org.optaplanner.core.api.domain.entity.PlanningEntity;
import org.optaplanner.core.api.domain.variable.PlanningListVariable;

import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Getter
@Setter

@AllArgsConstructor
@RequiredArgsConstructor
@PlanningEntity
public class Pairing extends AbstractPersistable {
    //변수로서 작동 된다. Pair 는 Flight 들의 연속이므로 ListVariable 로 작동된다.
    @PlanningListVariable(valueRangeProviderRefs = {"pairing"})
    private List<Flight> pair = new ArrayList<>();
    private Integer totalCost;
    private static int briefingTime;
    private static int debriefingTime;
    private static int restTime;
    private static int LayoverTime;
    private static int QuickTurnaroundTime;

    public static void setStaticTime(int briefingTime,
                        int debriefingTime,
                        int restTime,
                        int LayoverTime,
                        int QuickTurnaroundTime) {
        Pairing.briefingTime = briefingTime;
        Pairing.debriefingTime = debriefingTime;
        Pairing.restTime = restTime;
        Pairing.LayoverTime = LayoverTime;
        Pairing.QuickTurnaroundTime = QuickTurnaroundTime;
    }

    public void setDebriefingTime(int debriefingTime) {
        Pairing.debriefingTime = debriefingTime;
    }

    public void setRestTime(int restTime) {
        Pairing.restTime = restTime;
    }

    public void setLayoverTime(int layoverTime) {
        LayoverTime = layoverTime;
    }

    public void setQuickTurnaroundTime(int quickTurnaroundTime) {
        QuickTurnaroundTime = quickTurnaroundTime;
    }

    @Builder
    public Pairing(long id, List<Flight> pair, Integer totalCost) {
        super(id);
        this.pair = pair;
        this.totalCost = totalCost;
    }

    //pair가 시간상 불가능하면 true를 반환
    public boolean getTimeImpossible() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (pair.get(i).getDestTime().isAfter(pair.get(i + 1).getOriginTime())) {
                return true;
            }
        }
        return false;
    }

    public int checkBreakTime(int index){
        long breakTime = ChronoUnit.MINUTES.between(pair.get(index).getDestTime(), pair.get(index+1).getOriginTime());

        return (int) Math.max(0,breakTime);
    }
    /*
    public boolean minBreakTime(){
        for(int i=0; i<pair.size()-1; i++){
            if(checkBreakTime(i) <= 60) return true;
        }
        return false;
    }
    */

    public Integer getSatisCost(){
        int satisScore = 0;
        for(int i=0; i<pair.size()-1; i++){
            if(checkBreakTime(i) <= 180) satisScore += 1000 * (180-checkBreakTime(i));
        }
        return satisScore;
    }
    public Integer getLayoverCost(){
        int layoverCost = 0;
        for(int i=0; i<pair.size()-1; i++){
            if(checkBreakTime(i) >= 360) layoverCost += checkBreakTime(i) * pair.get(0).getAircraft().getLayoverCost()/100;
        }
        return layoverCost;
    }

    public boolean getLawImpossible(){
        int time = pair.get(0).getFlightTime();

        for(int i=1; i<pair.size(); i++){
            if(checkBreakTime(i-1) < 10*60) time += pair.get(i).getFlightTime() + checkBreakTime(i-1);
            else time = pair.get(i).getFlightTime();

            if(time > 14*60) return true;
        }
        return false;
    }

    //pair가 공간상 불가능하면 true를 반환
    public boolean getAirportImpossible() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (!pair.get(i).getDestAirport().getName().equals(pair.get(i + 1).getOriginAirport().getName())) {
                return true;
            }
        }
        return false;
    }

    //기종이 다 같은지 다 같지 않으면 true 반환
    public boolean getAircraftDiff() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (!pair.get(i).getAircraft().getType().equals(pair.get(i + 1).getAircraft().getType())) {
                return true;
            }
        }
        return false;
    }

    public int getMovingWorkCost(){
        int maxCrewNum = 0;
        for (Flight flight : pair) {
            maxCrewNum = Math.max(maxCrewNum, flight.getAircraft().getCrewNum());
        }
        int movingWorkCost = 0;
        for (Flight flight : pair) {
            //(최대 승무원 수 - 지금 기종의 승무원 수) * 운항시간(분)*100 <-추후 cost 변경
            movingWorkCost += (maxCrewNum-flight.getAircraft().getCrewNum()) * flight.getFlightTime() * 10;
        }
        return movingWorkCost;
    }

    //pair의 총 길이 반환 (일수)
    public long getTotalLength() {
        return ChronoUnit.DAYS.between(pair.get(0).getOriginTime(), pair.get(pair.size() - 1).getDestTime());
    }

    //첫번쨰 비행과 마지막 비행 Base 비교
    public boolean equalBase() {
        return !pair.get(0).getOriginAirport().getName().equals(pair.get(pair.size() - 1).getDestAirport().getName());
    }

    //마지막 비행 반환
    public Integer getDeadHeadCost() {
        Map<String, Integer> deadheads = pair.get(pair.size() - 1).getDestAirport().getDeadheadCost();

        String dest = pair.get(pair.size() - 1).getDestAirport().getName();
        String origin = pair.get(0).getOriginAirport().getName();

        return deadheads.getOrDefault(origin, 0) * 100;
    }
    @Override
    public String toString() {
        return "Pairing - " + id +
                " { pair=" + pair + " }";
    }
}
