package org.dongguk.domain;

import lombok.*;
import org.dongguk.AbstractPersistable;

import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

@Getter
@Setter
@AllArgsConstructor
@RequiredArgsConstructor
public class Pairing extends AbstractPersistable {
    //변수로서 작동 된다. Pair 는 Flight 들의 연속이므로 ListVariable 로 작동된다.
    private List<Flight> pair = new ArrayList<>();
    private Integer totalCost;
    private static int briefingTime;
    private static int debriefingTime;
    private static int restTime;
    private static int LayoverTime;
    private static int QuickTurnaroundTime;
    private static int hotelTime;
    private static int hotelMinTime = 720;
    private static int checkContinueTime = 60 * 10;
    private static int continueMaxTime = 14 * 60;
    private static int workMaxTime = 8 * 60;

    public static void setStaticTime(int briefingTime, int debriefingTime,
                                     int restTime, int LayoverTime, int QuickTurnaroundTime,
                                     int hotelTime) {
        Pairing.briefingTime = briefingTime;
        Pairing.debriefingTime = debriefingTime;
        Pairing.restTime = restTime;
        Pairing.LayoverTime = LayoverTime;
        Pairing.QuickTurnaroundTime = QuickTurnaroundTime;
        Pairing.hotelTime = hotelTime;
    }

    @Builder
    public Pairing(long id, List<Flight> pair, Integer totalCost) {
        super(id);
        this.pair = pair;
        this.totalCost = totalCost;
    }

    /**
     * pairing의 실행 가능 여부 확인(불가능한 경우:true)
     * / 앞 비행이 도착하지 않았는데 이후 비행이 출발했을 경우 판단
     * @return boolean
     */
    public boolean isImpossibleTime() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (pair.get(i).getDestTime().isAfter(pair.get(i + 1).getOriginTime())) {
                return true;
            }
        }
        return false;
    }

    /**
     * 동일 공항 출발 여부 확인
     * / 도착 공항과 출발 공항이 다를 시 true 반환
     * @return boolean
     */
    public boolean isImpossibleAirport() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (!pair.get(i).getDestAirport().getName().equals(pair.get(i + 1).getOriginAirport().getName())) {
                return true;
            }
        }
        return false;
    }

    /**
     * 페어링의 최소 휴식시간 보장 여부 검증
     * / 연속되는 비행이 14시간 이상일 시 true 반환(연속: breakTime이 10시간 이하)
     * @return boolean
     */
    public boolean isImpossibleContinuity(){
        int totalTime = pair.get(0).getFlightTime();
        int workTime = pair.get(0).getFlightTime();

        for(int i=1; i<pair.size(); i++){
            if(getFlightGap(i - 1) < checkContinueTime) {
                totalTime += pair.get(i).getFlightTime() + getFlightGap(i - 1);
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

    /**
     * pairing의 동일 항공기 여부 검증
     * / 비행들의 항공기가 동일하지 않을 시 true 반환
     * @return boolean
     */
    public boolean isDifferentAircraft() {
        for (int i = 0; i < pair.size() - 1; i++) {
            if (!pair.get(i).getAircraft().getType().equals(pair.get(i + 1).getAircraft().getType())) {
                return true;
            }
        }
        return false;
    }

    /**
     * 처음과 끝 공항의 동일 여부 확인
     * / 처음 출발 공항과 마지막 도착 공항이 다를 시 true
     * @return boolean
     */
    public boolean isEqualBase() {
        return !pair.get(0).getOriginAirport().getName().equals(pair.get(pair.size() - 1).getDestAirport().getName());
    }

    /**
     * 페어링의 총 SatisCost 반환
     * / breakTime이 180보다 작은 경우 발생
     * @return sum(180 - breakTime)*1000
     */
    public Integer getSatisCost(){
        int satisScore = 0;
        for(int i=0; i<pair.size()-1; i++){
            if(getFlightGap(i) <= 180) satisScore += 1000 * (180 - getFlightGap(i));
        }
        return satisScore;
    }

    /**
     * 페어링의 총 이동근무 cost 반환
     * / 페어링 인원보다 요구 승무원이 적은 비행일 시 발생(maxCrewNum이 기준)
     * @return sum((maxCrewNum - 요구 승무원)*운항시간(분))*10
     */
    public int getMovingWorkCost(){
        int maxCrewNum = 0;
        int movingWorkCost = 0;

        for (Flight flight : pair) {
            maxCrewNum = Math.max(maxCrewNum, flight.getAircraft().getCrewNum());
        }
        for (Flight flight : pair) {
            //(최대 승무원 수 - 지금 기종의 승무원 수) * 운항시간(분)*100 <-추후 cost 변경
            movingWorkCost += (maxCrewNum - flight.getAircraft().getCrewNum()) * flight.getFlightTime() * 10;
        }
        return movingWorkCost;
    }

    /**
     * 페어링의 총 갈아 반환 (일)
     * @return 마지막 비행 도착시간 - 처음 비행 시작시간
     */
    public long getTotalLength() {
        return ChronoUnit.DAYS.between(pair.get(0).getOriginTime(), pair.get(pair.size() - 1).getDestTime());
    }

    /**
     * 페어링의 deadhead cost 반환
     * / 마지막 도착 공항에서 처음 공항으로 가는데 필요한 deadhead cost 사용
     * @return deadhead cost / 2
     */
    public Integer getDeadHeadCost() {
        Map<String, Integer> deadheads = pair.get(pair.size() - 1).getDestAirport().getDeadheadCost();
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
            if (getFlightGap(i) <= 0) {
                return 0;
            }

            // 음수가 아니라면 유효한 페어링이므로 LayoverCost 계산
            if (getFlightGap(i) >= LayoverTime) {
                cost += (getFlightGap(i) - LayoverTime) * pair.get(0).getAircraft().getLayoverCost();
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
            if (getFlightGap(i) <= 0) {
                return 0;
            }

            // 음수가 아니라면 유효한 페어링이므로 QuickTurnCost 계산
            if (getFlightGap(i) < QuickTurnaroundTime) {
                cost += (QuickTurnaroundTime - getFlightGap(i)) * pair.get(0).getAircraft().getQuickTurnCost();
            }
        }

        return cost / 100;
    }

    /**
     * 페어링의 총 HotelCost 반환
     * / 총 인원수를 곱하는 이유 : Flight Cost, Layover Cost, QuickTurn Cost 모두 총 인원에 대한 값으로 계산된 후 입력받음
     * / 휴식시간이 12시간 이상일 경우 1일 숙박,이후 18시간 이상 남을 시 1일 추가 반복
     * @return sum(hotel cost) / 100
     */
    public Integer getHotelCost() {
        // 페어링의 총 길이가 1개 이하라면 LayoverCost 없음
        if(pair.size() <= 1) return 0;

        int cost = 0;
        for (int i = 0; i < pair.size() - 1; i++) {
            // 만약 비행편 간격이 하나라도 음수라면 유효한 페어링이 아님
            if (getFlightGap(i) <= 0) {
                return 0;
            }

            int flightGap = getFlightGap(i);
            // 음수가 아니라면 유효한 페어링이므로 HotelCost 계산
            if (flightGap >= hotelMinTime) {
                cost += (pair.get(i + 1).getOriginAirport().getHotelCost()
                        * pair.get(0).getAircraft().getCrewNum() //비행마다 crew num 다를 수도 있어서 max crew로 수정해야 할 듯)
                        * (int) (1 + (int) Math.floor(((float) flightGap - (float) hotelMinTime) / (float) hotelTime)));
            }
        }

        return cost / 100;
    }

    /**
     * 비행 사이의 쉬는 시간 계산
     * @return (int) Math.max(0,breakTime)
     */
    private int getFlightGap(int index){ //수정 필요
        long breakTime = ChronoUnit.MINUTES.between(pair.get(index).getDestTime(), pair.get(index+1).getOriginTime());

        return (int) Math.max(0, breakTime);
    }

    @Override
    public String toString() {
        return "Pairing - " + id +
                " { pair=" + pair + " }";
    }
}
