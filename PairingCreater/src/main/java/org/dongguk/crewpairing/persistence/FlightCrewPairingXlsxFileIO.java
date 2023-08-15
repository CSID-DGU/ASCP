package org.dongguk.crewpairing.persistence;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.xssf.usermodel.XSSFRow;
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

@Slf4j
public class FlightCrewPairingXlsxFileIO extends AbstractXlsxSolutionFileIO<PairingSolution> {

    @Override
    public String getOutputFileExtension() {
        return super.getOutputFileExtension();
    }

    @Override
    public PairingSolution read(File inputFile) {
        try (InputStream in = new ClassPathResource("ASCP_Data_Input.xlsx").getInputStream()) {
            XSSFWorkbook workbook = new XSSFWorkbook(in);
            return new FlightCrewPairingXlsxReader(workbook).read();
        } catch (IOException | RuntimeException e) {
            log.error("{} {}", e.getMessage(), "Input File Error. Please Input File Format");
            System.exit(1);
        }

        // 절대 나오면 안되는 null을 정의
        return null;
    }

    @Override
    public void write(PairingSolution pairingSolution, File file) {
        new FlightCrewPairingXlsxWriter(pairingSolution).write();
    }

    @Getter
    @Setter
    private static class FlightCrewPairingXlsxReader extends AbstractXlsxReader<PairingSolution, HardSoftScore> {
        private final List<Aircraft> aircraftList = new ArrayList<>();
        private final List<Airport> airportList = new ArrayList<>();
        private final List<Flight> flightList = new ArrayList<>();

        private final Map<String, Airport> airportMap = new HashMap<>();

        public FlightCrewPairingXlsxReader(XSSFWorkbook workbook) {
            super(workbook, PairingApp.SOLVER_CONFIG);
        }

        @Override
        public PairingSolution read() {
            readAircraft();
            readAirport();
            readDeadhead();
            readFlight();
            createEntities();
            return PairingSolution.builder()
                    .aircraftList(aircraftList)
                    .airportList(airportList)
                    .flightList(flightList)
                    .pairingList(createEntities()).build();
        }

        private List<Pairing> createEntities() {
            //초기 페어링 Set 구성 어차피 [solver]가 바꿔버려서 의미 없음 아무것도 안넣으면 오류나서 넣는 것
            List<Pairing> pairingList = new ArrayList<>();
            for (int i = 0; i < flightList.size(); i++) {
                List<Flight> pair = new ArrayList<>();
                pair.add(flightList.get(i));
                pairingList.add(new Pairing(i, pair, 0));
            }

            return pairingList;
        }

        private void readAircraft() {
            aircraftList.clear();
            nextSheet("Program_Input_Aircraft");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    aircraftList.add(Aircraft.builder()
                            .id(indexCnt++)
                            .type(row.getCell(0).getStringCellValue())
                            .crewNum((int) row.getCell(1).getNumericCellValue())
                            .flightSalary((int) row.getCell(2).getNumericCellValue())
                            .baseSalary((int) row.getCell(3).getNumericCellValue())
                            .layoverCost((int) row.getCell(4).getNumericCellValue()).build());
                } catch (IllegalStateException e) {
                    log.info("Finish Read Aircraft File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        private void readAirport() {
            airportMap.clear();
            nextSheet("Program_Input_Airport");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    airportMap.put(row.getCell(0).getStringCellValue(), Airport.builder()
                            .id(indexCnt++)
                            .name(row.getCell(0).getStringCellValue())
                            .deadheadCost(new HashMap<>()).build());
                } catch (IllegalStateException e) {
                    log.info("Finish Read Airport File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        private void readDeadhead() {
            airportList.clear();
            nextSheet("User_Deadhead");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // Header 스킵

            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    String origin = row.getCell(0).getStringCellValue();

                    if (origin.isBlank()) {
                        continue;
                    }

                    airportMap.get(origin)
                            .putDeadhead(row.getCell(1).getStringCellValue(), (int) row.getCell(2).getNumericCellValue());
                } catch (IllegalStateException e) {
                    log.info("Finish Read DeadHead File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            airportList.addAll(airportMap.values());
        }

        private void readFlight() {
            flightList.clear();
            nextSheet("User_Flight");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    flightList.add(Flight.builder()
                            .id(indexCnt++)
                            .TailNumber(row.getCell(0).getStringCellValue())
                            .originAirport(Airport.of(airportList, row.getCell(1).getStringCellValue()))
                            .originTime(row.getCell(2).getLocalDateTimeCellValue())
                            .destAirport(Airport.of(airportList, row.getCell(3).getStringCellValue()))
                            .destTime(row.getCell(4).getLocalDateTimeCellValue())
                            .aircraft(Aircraft.of(aircraftList, row.getCell(6).getStringCellValue())).build());
                } catch (IllegalStateException e) {
                    log.info("Finish Read Flight File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

        @Getter
        @Setter
        private static class FlightCrewPairingXlsxWriter extends AbstractXlsxWriter<PairingSolution, HardSoftScore> {

            public FlightCrewPairingXlsxWriter(PairingSolution pairingSolution) {
                super(pairingSolution, PairingApp.SOLVER_CONFIG);
            }

            public void output() {
                List<Pairing> pairingList = solution.getPairingList();
                StringBuilder text = new StringBuilder();

                for (Pairing pairing : pairingList) {
                    text.append(pairingList.indexOf(pairing)).append(",");
                    for (Flight flight : pairing.getPair()) {
                        text.append(flight.getIndex()).append(",");
                    }
                    text.append("\n");
                }

                try (FileWriter fw = new FileWriter("src/main/output/output-data.csv")) {
                    fw.write(text.toString());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            @Override
            public void write() {
                output();

                List<Pairing> pairingList = solution.getPairingList();
                //첫 항공기의 출발시간을 기준으로 정렬
                pairingList.removeIf(pairing -> pairing.getPair().isEmpty());
                pairingList.sort(Comparator.comparing(a -> a.getPair().get(0).getOriginTime()));

                //첫 항공기의 출발시간~마지막 항공기의 도착 시간까지 타임 테이블 생성
                StringBuilder text = new StringBuilder();
                LocalDateTime f = pairingList.get(0).getPair().get(0).getOriginTime();
                LocalDateTime firstTime = stripMinutes(f);
                LocalDateTime l = firstTime;

                for (Pairing pairing : pairingList) {
                    for (Flight flight : pairing.getPair()) {
                        l = l.isAfter(flight.getDestTime()) ? l : flight.getDestTime();
                    }
                }
                LocalDateTime lastTime = stripMinutes(l);

                //첫 줄에 날짜 단위 입력
                f = firstTime;
                text.append(",,").append(f).append(",");
                f = f.plusHours(1);
                do {
                    if (f.getHour() == 0) text.append(f);
                    text.append(",");

                    f = f.plusHours(1);
                } while (!f.equals(lastTime));
                text.append("\n");

                //두번째 줄에 시간 단위 입력
                text.append("INDEX,TYPE,");
                f = firstTime;
                do {
                    text.append(f.getHour());
                    text.append(":00,");

                    f = f.plusHours(1);
                } while (!f.equals(lastTime));
                text.append("\n");

                //타임 테이블의 내용 작성
                for (Pairing pairing : pairingList) {
                    text.append("SET").append(pairingList.indexOf(pairing)).append(",");
                    text.append(pairing.getPair().get(0).getAircraft().getType()).append(",");
                    String s = buildTable(pairing.getPair(), firstTime);
                    text.append(s);
                }

                //csv 파일로 출력
                try (FileWriter fw = new FileWriter("src/main/output/visualized-data.csv")) {
                    fw.write(text.toString());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }

            //출발시간과 도착 시간의 차이를 구하며 csv format 에 맞는 text 생성.
            private static String buildTable(List<Flight> pairing, LocalDateTime firstTime) {
                StringBuilder sb = new StringBuilder();
                for (Flight flight : pairing) {
                    int a = (int) ChronoUnit.HOURS.between(firstTime, stripMinutes(flight.getOriginTime()));
                    sb.append(",".repeat(Math.max(0, a)));
                    sb.append(flight.getOriginAirport().getName());
                    sb.append(",");
                    int b = (int) ChronoUnit.HOURS.between(flight.getOriginTime(), stripMinutes(flight.getDestTime()));
                    sb.append("#######,".repeat(Math.max(0, b - 1)));
                    sb.append(flight.getDestAirport().getName());

                    firstTime = stripMinutes(flight.getDestTime());
                }
                sb.append("\n");

                return valueOf(sb);
            }

            //분 단위를 버림함
            private static LocalDateTime stripMinutes(LocalDateTime l) {
                return LocalDateTime.of(l.getYear(), l.getMonth(), l.getDayOfMonth(), l.getHour(), 0);
            }
        }
    }
