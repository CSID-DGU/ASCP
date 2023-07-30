package org.dongguk.crewpairing.persistence;

import lombok.Getter;
import lombok.Setter;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.dongguk.common.persistence.AbstractXlsxSolutionFileIO;
import org.dongguk.crewpairing.app.PairingApp;
import org.dongguk.crewpairing.domain.*;
import org.drools.io.ClassPathResource;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;

import java.io.*;
import java.time.LocalDateTime;
import java.time.temporal.ChronoUnit;
import java.util.*;

import static java.lang.String.valueOf;

public class FlightCrewPairingXlsxFileIO extends AbstractXlsxSolutionFileIO<PairingSolution> {
    @Override
    public String getOutputFileExtension() {
        return super.getOutputFileExtension();
    }

    @Override
    public PairingSolution read(File inputFile) {
        return new FlightCrewPairingXlsxReader().read();
    }

    @Override
    public void write(PairingSolution pairingSolution, File file) {
        new FlightCrewPairingXlsxWriter(pairingSolution).write();
    }

    @Getter
    @Setter
    private static class FlightCrewPairingXlsxReader extends AbstractXlsxReader<PairingSolution, HardSoftScore> {
        private List<Aircraft> aircraftList;
        private List<Airport> airportList;
        private List<Flight> flightList;

        public FlightCrewPairingXlsxReader() {
            super(PairingApp.SOVLER_CONFIG);
        }

        @Override
        public PairingSolution read() {
            readAircraft();
            readAirport();
            readFlight();
            return PairingSolution.builder()
                    .aircraftList(aircraftList)
                    .airportList(airportList)
                    .flightList(flightList).build();
        }

        private void readAircraft() {
            aircraftList = new ArrayList<>();
            //Aircraft class 정보 읽기
            InputStream inputStream = null;

            try {
                inputStream = new ClassPathResource("salary.xlsx").getInputStream();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            Workbook workbook;

            try {
                workbook = new XSSFWorkbook(inputStream);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            int cnt = 0;

            Sheet sheet = workbook.getSheetAt(0);
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);
                String aircraft = row.getCell(0).getStringCellValue();
                int crewNum = (int)(row.getCell(1).getNumericCellValue());
                int flightSalary = (int)(row.getCell(2).getNumericCellValue());
                int baseSalary = (int)(row.getCell(3).getNumericCellValue());
                int layoverSalary = (int)(row.getCell(4).getNumericCellValue());

                aircraftList.add(Aircraft.builder()
                        .id(cnt++)
                        .name(aircraft)
                        .crewNum(crewNum)
                        .flightSalary(flightSalary)
                        .baseSalary(baseSalary)
                        .layoverCost(layoverSalary)
                        .build());
            }
        }

        private void readAirport() {
            airportList = new ArrayList<>();
            //Airport class 정보 읽기
            InputStream inputStream = null;

            try {
                inputStream = new ClassPathResource("deadhead.xlsx").getInputStream();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            Workbook workbook;

            try {
                workbook = new XSSFWorkbook(inputStream);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            int cnt_index = 0;
            Map<String, Integer>[] map = new Map[54];
            map[0] = new HashMap<>();
            int cnt = 0;

            Sheet sheet = workbook.getSheetAt(0);
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);
                String origin = row.getCell(0).getStringCellValue();
                String dest = row.getCell(1).getStringCellValue();
                int deadhead = (int)(row.getCell(2).getNumericCellValue());
                map[cnt].put(dest,deadhead);
                if(sheet.getRow(i+1)== null || origin != sheet.getRow(i+1).getCell(0).getStringCellValue()){
                    airportList.add(Airport.builder()
                            .id(cnt_index++)
                            .name(row.getCell(0).getStringCellValue())
                            .deadheadCost(map[cnt++]).build());

                    map[cnt] = new HashMap<>();
                }
            }
        }

        private void readFlight() {
            flightList = new ArrayList<>();
            //Flight class 정보 읽기
            InputStream inputStream = null;

            try {
                inputStream = new ClassPathResource("flight.xlsx").getInputStream();
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            Workbook workbook;

            try {
                workbook = new XSSFWorkbook(inputStream);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }

            Sheet sheet = workbook.getSheetAt(0);
            int cnt = 0;
            for (int i = 1; i <= sheet.getLastRowNum(); i++) {
                Row row = sheet.getRow(i);
                String index = row.getCell(0).getStringCellValue();
                String origin = row.getCell(1).getStringCellValue();
                LocalDateTime originDate = row.getCell(2).getLocalDateTimeCellValue();
                String dest = row.getCell(3).getStringCellValue();
                LocalDateTime destDate = row.getCell(4).getLocalDateTimeCellValue();
                String aircraft = row.getCell(6).getStringCellValue();

                flightList.add(Flight.builder()
                        .id(cnt++)
                        .index(index)
                        .originAirport(Airport.findAirportByName(airportList, origin))
                        .originTime(originDate)
                        .destAirport(Airport.findAirportByName(airportList, dest))
                        .destTime(destDate)
                        .aircraft(Aircraft.findInAircraftName(aircraftList,aircraft)).build());
            }
        }


    }

    @Getter
    @Setter
    private static class FlightCrewPairingXlsxWriter extends AbstractXlsxWriter<PairingSolution, HardSoftScore> {

        public FlightCrewPairingXlsxWriter(PairingSolution pairingSolution) {
            super(pairingSolution, PairingApp.SOVLER_CONFIG);
        }

        @Override
        public void write() {
            List<Pairing> pairingList = solution.getPairingList();
            //첫 항공기의 출발시간을 기준으로 정렬
            pairingList.removeIf(pairing -> pairing.getPair().isEmpty());
            pairingList.sort(Comparator.comparing(a -> a.getPair().get(0).getOriginTime()));

            //첫 항공기의 출발시간~마지막 항공기의 도착 시간까지 타임 테이블 생성
            StringBuilder text = new StringBuilder();
            LocalDateTime f = pairingList.get(0).getPair().get(0).getOriginTime();
            LocalDateTime firstTime = stripMinutes(f);
            LocalDateTime l = firstTime;

            for (Pairing pair: pairingList){
                for(Flight flight : pair.getPair()){
                    l = l.isAfter(flight.getDestTime()) ? l : flight.getDestTime();
                }
            }
            LocalDateTime lastTime = stripMinutes(l);

            //첫 줄에 날짜 단위 입력
            f = firstTime;
            text.append(",").append(f).append(",");
            f = f.plusHours(1);
            do {
                if(f.getHour()==0) text.append(f);
                text.append(",");

                f = f.plusHours(1);
            } while (!f.equals(lastTime));
            text.append("\n");

            //두번째 줄에 시간 단위 입력
            text.append(",");
            f = firstTime;
            do {
                text.append(f.getHour());
                text.append(":00,");

                f = f.plusHours(1);
            } while (!f.equals(lastTime));
            text.append("\n");

            //타임 테이블의 내용 작성
            for(Pairing pairing : pairingList){
                text.append("SET");
                text.append(pairingList.indexOf(pairing));
                text.append(",");
                String s = buildTable(pairing.getPair(), firstTime);
                text.append(s);
            }

            //csv 파일로 출력
            try (FileWriter fw = new FileWriter("visualized-data.csv")) {
                fw.write(text.toString());
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        //출발시간과 도착 시간의 차이를 구하며 csv format 에 맞는 text 생성.
        private static String buildTable(List<Flight> pairing, LocalDateTime firstTime){
            StringBuilder sb = new StringBuilder();
            for(Flight flight : pairing){
                int a = (int) ChronoUnit.HOURS.between(firstTime, stripMinutes(flight.getOriginTime()));
                sb.append(",".repeat(Math.max(0,a)));
                sb.append(flight.getOriginAirport().getName());
                sb.append(",");
                int b = (int) ChronoUnit.HOURS.between(flight.getOriginTime(), stripMinutes(flight.getDestTime()));
                sb.append("#######,".repeat(Math.max(0,b-1)));
                sb.append(flight.getDestAirport().getName());

                firstTime = stripMinutes(flight.getDestTime());
            }
            sb.append("\n");

            return valueOf(sb);
        }

        //분 단위를 버림함
        private static LocalDateTime stripMinutes(LocalDateTime l){
            return LocalDateTime.of(l.getYear(), l.getMonth(), l.getDayOfMonth(), l.getHour(), 0);
        }
    }
}
