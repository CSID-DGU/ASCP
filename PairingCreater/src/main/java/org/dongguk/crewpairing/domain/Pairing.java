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
    public static int briefingTime;
    public static int debriefingTime;
    public static int restTime;
    public static int LayoverTime;
    public static int QuickTurnaroundTime;
    public static int hotelTime = 18 * 60;
    public static int hotelMinTime = 720;
    public final static int checkContinueTime = 60*10;
    public final static int continueMaxTime = 14*60;
    public final static int workMaxTime = 8*60;

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

    /**
     * 페어링의 최소 휴식시간 보장 여부 검증
     * / 연속되는 비행이 14시간 이상일 시 true 반환(연속: breakTime이 10시간 이하)
     * @return boolean
     */
    public boolean getContinuityImpossible(){
        int totalTime = pair.get(0).getFlightTime();
        int workTime = pair.get(0).getFlightTime();

        for(int i=1; i<pair.size(); i++){
            if(checkBreakTime(i-1) < checkContinueTime) {
                totalTime += pair.get(i).getFlightTime() + checkBreakTime(i-1);
                workTime += pair.get(i).getFlightTime();
            }
            else {
                totalTime = pair.get(i).getFlightTime();
                workTime = pair.get(i).getFlightTime();
            }
            if(totalTime > continueMaxTime) return true;
            if(workTime > workMaxTime) return true;
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

        return deadheads.getOrDefault(origin, 0) / 2;
    }

    /**
     * 페어링의 총 LayoverCost 반환
     * 비행편간 간격이 LayoverTime 보다 크거나 같은 경우에만 LayoverCost 발생
     * @return sum(LayoverCost) / 100
     */
    public Integer getLayoverCost(){
        // 페어링의 총 길이가 1개 이하라면 LayoverCost 없음
        if(pair.size() <= 1) return 0;

        int cost = 0;
        for (int i = 0; i < pair.size() - 1; i++) {
            // 만약 비행편 간격이 하나라도 음수라면 유효한 페어링이 아님
            if (checkBreakTime(i) <= 0) {
                return 0;
            }

            // 음수가 아니라면 유효한 페어링이므로 LayoverCost 계산
            if (checkBreakTime(i) >= LayoverTime) {
                cost += (checkBreakTime(i) - LayoverTime) * pair.get(0).getAircraft().getLayoverCost();
            }
        }

        return cost / 100;
    }

    /**
     * 페어링의 총 QuickTurnCost 반환
     * 비행편간 간격이 QuickTurnaroundTime 보다 작은 경우에만 QuickTurnCost 발생
     * @return sum(QuickTurnCost) / 100
     */
    public Integer getQuickTurnCost() {
        // 페어링의 총 길이가 1개 이하라면 QuickTurnCost 없음
        if(pair.size() <= 1) return 0;

        int cost = 0;
        for (int i = 0; i < pair.size() - 1; i++) {
            // 만약 비행편 간격이 하나라도 음수라면 유효한 페어링이 아님
            if (checkBreakTime(i) <= 0) {
                return 0;
            }

            // 음수가 아니라면 유효한 페어링이므로 QuickTurnCost 계산
            if (checkBreakTime(i) < QuickTurnaroundTime) {
                cost += (QuickTurnaroundTime - checkBreakTime(i)) * pair.get(0).getAircraft().getQuickTurnCost();
            }
        }

        return cost / 100;
    }

    /**
     * 페어링의 총 HotelCost 반환
     * 총 인원수를 곱하는 이유 : Flight Cost, Layover Cost, QuickTurn Cost 모두 총 인원에 대한 값으로 계산된 후 입력받음
     * 이걸 굳이 나눠야 싶음(레이오버가 생기면 호텔비용이 생기므로 합쳐도 되지 않을까 싶음)
     * @return HotelCost(Layover가 발생했을 때 [해당 항공기의 인원 수 * 공항의 호텔 비용]를 다 더함) / 10
     */
    public Integer getHotelCost() {
        // 페어링의 총 길이가 1개 이하라면 LayoverCost 없음
        if(pair.size() <= 1) return 0;

        int cost = 0;
        for (int i = 0; i < pair.size() - 1; i++) {
            // 만약 비행편 간격이 하나라도 음수라면 유효한 페어링이 아님
            if (checkBreakTime(i) <= 0) {
                return 0;
            }

            int flightGap = checkBreakTime(i);
            // 음수가 아니라면 유효한 페어링이므로 HotelCost 계산
            if (flightGap >= hotelMinTime) {
                cost += (pair.get(i + 1).getOriginAirport().getHotelCost()
                        * pair.get(0).getAircraft().getCrewNum()
                        * (int) (1 + (int) Math.floor(((float) flightGap - (float) hotelMinTime) / (float) hotelTime)));
            }
        }

        return cost / 100;
    }

    private int checkBreakTime(int index){
        long breakTime = ChronoUnit.MINUTES.between(pair.get(index).getDestTime(), pair.get(index+1).getOriginTime());

        return (int) Math.max(0,breakTime);
    }
    @Override
    public String toString() {
        return "Pairing - " + id +
                " { pair=" + pair + " }";
    }
}
