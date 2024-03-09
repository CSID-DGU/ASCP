package org.dongguk.crewpairing.persistence;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.*;
import org.dongguk.common.persistence.AbstractXlsxSolutionFileIO;
import org.dongguk.crewpairing.app.PairingApp;
import org.dongguk.crewpairing.domain.*;
import org.dongguk.crewpairing.domain.factory.DomainFactory;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
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
    public PairingSolution read(File informationXlsxFile) {
        try (InputStream in = new BufferedInputStream(new FileInputStream(informationXlsxFile))) {
            XSSFWorkbook workbook = new XSSFWorkbook(in);
            return new FlightCrewPairingXlsxReader(workbook).read();
        } catch (IOException | RuntimeException e) {
            log.error("{} {}", e.getMessage(), "Input File Error. Please Input File Format");
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    public List<Pairing> readPairingList(List<Flight> flightList, File pairingXlsxFile) {
        try (InputStream in = new BufferedInputStream(new FileInputStream(pairingXlsxFile))) {
            XSSFWorkbook workbook = new XSSFWorkbook(in);
            return new FlightCrewPairingXlsxReader(workbook).readPairingSet(flightList);
        } catch (IOException | RuntimeException e) {
            log.error("{} {}", e.getMessage(), "Input File Error. Please Input File Format");
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    @Override
    public void write(PairingSolution pairingSolution, File file) {
        new FlightCrewPairingXlsxWriter(pairingSolution).write();
    }

    @Getter
    private static class FlightCrewPairingXlsxReader extends AbstractXlsxReader<PairingSolution, HardSoftScore> {
        private final Map<String, Airport> airportMap = new HashMap<>();

        public FlightCrewPairingXlsxReader(XSSFWorkbook workbook) {
            super(workbook, PairingApp.SOLVER_CONFIG);
        }

        @Override
        public PairingSolution read() {
            readTimeData();
            log.debug("Complete Read Time Data");
            readAircraft();         // 수정
            log.debug("Complete Read Aircraft Data");
            readAirport();          // 수정
            log.debug("Complete Read Airport Data");
            readDeadhead();         // 수정
            log.debug("Complete Read Deadhead Data");
            readFlight();
            log.debug("Complete Read Flight Data");
            return PairingSolution.builder()
                    .aircraftList(DomainFactory.getAircraftList())
                    .airportList(DomainFactory.getAirportList())
                    .flightList(DomainFactory.getFlightList())
                    .pairingList(createEntities()).build();
        }

        public List<Pairing> readPairingSet(List<Flight> inputFlightList) {
            List<Pairing> list = new ArrayList<>();

            nextSheet("Sheet");    // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵

            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                Iterator<Cell> currentCellIterator =  row.cellIterator();

                int indexCnt = (int) currentCellIterator.next().getNumericCellValue();
                List<Flight> pair = new ArrayList<>();
                while (currentCellIterator.hasNext()) {
                    Cell cell = currentCellIterator.next();
                    if (cell.getCellType() == CellType.BLANK || cell.getCellType() == CellType.STRING) {
                        break;
                    }

                    pair.add(inputFlightList.get((int) cell.getNumericCellValue()));
                }

                list.add(new Pairing(indexCnt, pair, 0));
            }

            log.debug("Complete Read Pairing Data");
            return list;
        }

        private void readTimeData() {
            nextSheet("User_Time");      // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵
            currentRowIterator.next();              // 빈행  스킵
            currentRowIterator.next();              // 보조제목  스킵
            currentRowIterator.next();              // Header  스킵

            XSSFRow row = (XSSFRow) currentRowIterator.next();

            Pairing.setStaticTime((int) row.getCell(1).getNumericCellValue(),
                    (int) row.getCell(2).getNumericCellValue(),
                    (int) row.getCell(3).getNumericCellValue(),
                    (int) row.getCell(4).getNumericCellValue(),
                    (int) row.getCell(5).getNumericCellValue());

            log.info("Complete Read Time Data");
        }

        private void readAircraft() {
            nextSheet("Program_Cost");    // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵
            currentRowIterator.next();              // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    DomainFactory.addAircraft(Aircraft.builder()
                            .id(indexCnt++)
                            .type(row.getCell(0).getStringCellValue())
                            .crewNum((int) row.getCell(1).getNumericCellValue())
                            .flightCost((int) row.getCell(2).getNumericCellValue())
                            .layoverCost((int) row.getCell(3).getNumericCellValue())
                            .quickTurnCost((int) row.getCell(4).getNumericCellValue()).build());
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
            nextSheet("User_Hotel");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // 빈 행 스킵
            currentRowIterator.next();  // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    if (row.getCell(0).getStringCellValue().isBlank()) {
                        continue;
                    }

                    airportMap.put(row.getCell(0).getStringCellValue(), Airport.builder()
                            .id(indexCnt++)
                            .name(row.getCell(0).getStringCellValue())
                            .hotelCost((int) row.getCell(1).getNumericCellValue())
                            .deadheadCost(new HashMap<>()).build());
                } catch (IllegalStateException e) {
                    log.info("Finish Read Airport File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            log.info("Complete Read Airport Data");
        }

        private void readDeadhead() {
            nextSheet("User_Deadhead");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // 빈 행 스킵
            currentRowIterator.next();  // Header 스킵

            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    String origin = row.getCell(0).getStringCellValue();

                    if (origin.isBlank()) {
                        continue;
                    }

                    airportMap.get(origin)
                            .putDeadhead(row.getCell(1).getStringCellValue(),
                                    (int) row.getCell(2).getNumericCellValue());
                } catch (IllegalStateException e) {
                    log.info("Finish Read DeadHead File");
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }

            DomainFactory.addAllAirport(airportMap);
            log.info("Complete Read Deadhead Data");
        }

        private void readFlight() {
            nextSheet("User_Flight");    // Sheet 고르기
            currentRowIterator.next();  // 주제목 스킵
            currentRowIterator.next();  // 빈 행 스킵
            currentRowIterator.next();  // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    DomainFactory.addFlight(Flight.builder()
                            .id(indexCnt++)
                            .serialNumber(row.getCell(0).getStringCellValue())
                            .tailNumber(row.getCell(1).getStringCellValue())
                            .originAirport(DomainFactory.getAirport(row.getCell(2).getStringCellValue()))
                            .originTime(row.getCell(3).getLocalDateTimeCellValue())
                            .destAirport(DomainFactory.getAirport(row.getCell(4).getStringCellValue()))
                            .destTime(row.getCell(5).getLocalDateTimeCellValue())
                            .aircraft(DomainFactory.getAircraft(row.getCell(6).getStringCellValue())).build());
                } catch (IllegalStateException e) {
                    log.info("Finish Read Flight File");
                    break;
                } catch (Exception e) {
                    log.error("{}", e.getMessage());
                    throw new RuntimeException(e);
                }
            }
        }

        private List<Pairing> createEntities() {
            List<Flight> flightList = DomainFactory.getFlightList();

            //초기 페어링 Set 구성 어차피 [solver]가 바꿔버려서 의미 없음 아무것도 안넣으면 오류나서 넣는 것
            List<Pairing> pairingList = new ArrayList<>();
            for (int i = 0; i < flightList.size(); i++) {
                List<Flight> pair = new ArrayList<>();
                pair.add(flightList.get(i));
                pairingList.add(new Pairing(i, pair, 0));
            }

            return pairingList;
        }
    }

        @Getter
        private static class FlightCrewPairingXlsxWriter extends AbstractXlsxWriter<PairingSolution, HardSoftScore> {

            public FlightCrewPairingXlsxWriter(PairingSolution pairingSolution) {
                super(pairingSolution, PairingApp.SOLVER_CONFIG);
            }

            @Override
            public void write() {
                String timeStr = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy_MM_dd_HH_mm_ss"));
//                exportPairingData(timeStr);
//                exportUserData(timeStr);
                exportUserData2(timeStr);
            }

            private void exportPairingData(String timeStr) {
                String fileName = timeStr + "-pairingData.xlsx";

                try (XSSFWorkbook workbook = new XSSFWorkbook()) {
                    XSSFSheet sheet = workbook.createSheet("Data");

                    List<Pairing> pairingList = solution.getPairingList();

                    //Pairing index 셀 스타일(우측 테두리)
                    CellStyle rightBorder = workbook.createCellStyle();
                    rightBorder.setAlignment(HorizontalAlignment.CENTER);
                    rightBorder.setBorderRight(BorderStyle.THIN);

                    Row row = sheet.createRow(0);
                    Cell cell = row.createCell(0);
                    cell.setCellValue("Pairing Data");

                    //Pairing data 테이블
                    int rowIdx = 1;
                    for(Pairing pairing : pairingList){
                        row = sheet.createRow(rowIdx);
                        cell = row.createCell(0);
                        cell.setCellValue(rowIdx-1);
                        cell.setCellStyle(rightBorder);

                        for(int i=0; i<pairing.getPair().size(); i++){
                            cell = row.createCell(i+1);
                            cell.setCellValue(pairing.getPair().get(i).getId());
                        }
                        rowIdx++;
                    }

                    try (FileOutputStream fo = new FileOutputStream("./data/crewpairing/output/" + fileName)) {
                        workbook.write(fo);
                    }
                }catch (IOException e){
                    e.printStackTrace();
                }

                System.out.println("Create Output File : " + fileName);
            }

            public void exportUserData(String timeStr) {
                String fileName = timeStr + "-userData1.xlsx";
                try (XSSFWorkbook workbook = new XSSFWorkbook()) {
                    XSSFSheet sheet = workbook.createSheet("Data");

                    List<Pairing> pairingList = solution.getPairingList();
                    pairingList.removeIf(pairing -> pairing.getPair().isEmpty());
                    pairingList.sort(Comparator.comparing(a -> a.getPair().get(0).getOriginTime()));

                    LocalDateTime firstTime = pairingList.get(0).getPair().get(0).getOriginTime();
                    LocalDateTime lastTime = firstTime;

                    //셀 스타일 모음
                    CellStyle headerStyle = workbook.createCellStyle();
                    Font headerFont = workbook.createFont();
                    headerFont.setBold(true);
                    headerStyle.setFont(headerFont);
                    headerStyle.setBorderBottom(BorderStyle.DOUBLE);
                    headerStyle.setFillForegroundColor(new XSSFColor(new byte[]{(byte) 226, (byte) 239, (byte) 217}, null));
                    headerStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);
                    headerStyle.setAlignment(HorizontalAlignment.CENTER);

                    CellStyle centerStyle = workbook.createCellStyle();
                    centerStyle.setAlignment(HorizontalAlignment.CENTER);
                    centerStyle.setBorderRight(BorderStyle.THIN);

                    //타임 테이블 헤더 작성
                    Row row = sheet.createRow(0);
                    Cell cell = row.createCell(0);
                    cell.setCellValue("Pairing SET");
                    cell.setCellStyle(headerStyle);
                    sheet.autoSizeColumn(0);

                    //가장 늦게 끝나는 페어링을 헤더의 마지막 날짜로 잡기 위함
                    for (Pairing pairing : pairingList) {
                        for (Flight flight : pairing.getPair()) {
                            lastTime = lastTime.isAfter(flight.getDestTime()) ? lastTime : flight.getDestTime();
                        }
                    }

                    int days = 0;
                    for (LocalDateTime f = firstTime; ChronoUnit.DAYS.between(f.toLocalDate(), lastTime.toLocalDate()) >= 0; f = f.plusDays(1)) {
                        days += 1;
                        String MMdd = f.format(DateTimeFormatter.ofPattern("MM/dd"));
                        cell = row.createCell(days);
                        cell.setCellValue(MMdd);
                        cell.setCellStyle(headerStyle);
                    }

                    //타임 테이블 내용 작성
                    XSSFColor[] colors = {
                            new XSSFColor(new byte[]{(byte) 255, (byte) 242, (byte) 204}, null),
                            new XSSFColor(new byte[]{(byte) 221, (byte) 235, (byte) 247}, null)
                    };

                    for (int i = 0; i < pairingList.size(); i++) {
                        row = sheet.createRow(i + 1);
                        cell = row.createCell(0);
                        cell.setCellValue("SET" + i);
                        cell.setCellStyle(centerStyle);

                        //Pairing의 flight에 대해서, 첫번째 비행과의 날짜 차이 k만큼 떨어진 셀에 내용 입력
                        for (Flight flight : pairingList.get(i).getPair()) {

                            //도착 시간이 24시를 넘어가는 경우 (날짜 차이)*24 + 도착시간 으로 표시
                            int daysGap = (int) ChronoUnit.DAYS.between(flight.getOriginTime().toLocalDate(), flight.getDestTime().toLocalDate());
                            int destHour = flight.getDestTime().getHour();
                            int destMin = flight.getDestTime().getMinute();
                            String tn = flight.getTailNumber();
                            String oriTime = flight.getOriginTime().format(DateTimeFormatter.ofPattern("HH:mm"));
                            String dstTime = String.format("%02d:%02d", daysGap * 24 + destHour, destMin);
                            String oriApt = flight.getOriginAirport().getName();
                            String dstApt = flight.getDestAirport().getName();

                            String text = "    [" + tn + "] " + "[ " + oriTime + " ~ " + dstTime + " ] " + "[" + oriApt + " -> " + dstApt + "]";

                            int k = (int) ChronoUnit.DAYS.between(firstTime.toLocalDate(), flight.getOriginTime().toLocalDate()) + 1;

                            //이미 셀에 값이 있다면 내용 추가
                            if (row.getCell(k) == null) {
                                cell = row.createCell(k);
                                cell.setCellValue(text);
                            } else {
                                StringBuilder sb = new StringBuilder(cell.getStringCellValue());
                                cell.setCellValue(sb.append("    /").append(text).toString());
                            }

                            XSSFColor currentColor = ((k % 2 == 0) && (i % 2 == 0)) || ((k % 2 == 1) && (i % 2 == 1)) ? colors[1] : colors[0];

                            CellStyle contentStyle = workbook.createCellStyle();
                            Font contentfont = workbook.createFont();
                            contentfont.setFontHeightInPoints((short) 9);
                            contentStyle.setAlignment(HorizontalAlignment.LEFT);
                            contentStyle.setBorderBottom(BorderStyle.DASH_DOT_DOT);
                            contentStyle.setBorderLeft(BorderStyle.DASH_DOT_DOT);
                            contentStyle.setFont(contentfont);
                            contentStyle.setFillForegroundColor(currentColor);
                            contentStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);

                            cell.setCellStyle(contentStyle);
                            sheet.autoSizeColumn(k);
                        }
                    }

                    try (FileOutputStream fo = new FileOutputStream("./data/crewpairing/output/" + fileName)) {
                        workbook.write(fo);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }

            public void exportUserData2(String timeStr) {
                String fileName = timeStr + "-userData2.xlsx";
                try (XSSFWorkbook workbook = new XSSFWorkbook()) {
                    XSSFSheet sheet = workbook.createSheet("Data");

                    List<Pairing> pairingList = solution.getPairingList();

                    //셀 스타일 모음
                    CellStyle headerStyle = workbook.createCellStyle();
                    Font headerFont = workbook.createFont();
                    headerFont.setBold(true);
                    headerStyle.setFont(headerFont);
                    headerStyle.setBorderBottom(BorderStyle.DOUBLE);
                    headerStyle.setFillForegroundColor(new XSSFColor(new byte[] {(byte) 226,(byte) 239,(byte) 217}, null));
                    headerStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);
                    headerStyle.setAlignment(HorizontalAlignment.CENTER);

                    CellStyle rightBorder = workbook.createCellStyle();
                    rightBorder.setBorderRight(BorderStyle.THIN);
                    rightBorder.setAlignment(HorizontalAlignment.CENTER);

                    Row row = sheet.createRow(0);
                    Cell cell = row.createCell(0);
                    cell.setCellValue("Pairing SET");
                    cell.setCellStyle(headerStyle);
                    sheet.autoSizeColumn(0);

                    // 타임 테이블 내용 작성
                    XSSFColor[] colors = {
                            new XSSFColor(new byte[]{(byte) 255, (byte) 242, (byte) 204}, null),
                            new XSSFColor(new byte[]{(byte) 221, (byte) 235, (byte) 247}, null)
                    };

                    //pairing의 최대 길이를 테이블의 길이로 설정
                    int maxCell = 0;
                    for (int i = 0; i < pairingList.size(); i++) {
                        row = sheet.createRow(i + 1);
                        cell = row.createCell(0);
                        cell.setCellValue("SET " + i);
                        cell.setCellStyle(rightBorder);
                        maxCell = Math.max(maxCell, pairingList.get(i).getPair().size());

                        int k = 0;
                        for (Flight flight : pairingList.get(i).getPair()) {
                            String tn = flight.getTailNumber();
                            String oriTime = flight.getOriginTime().format(DateTimeFormatter.ofPattern("MM/dd HH:mm"));
                            String dstTime = flight.getDestTime().format(DateTimeFormatter.ofPattern("MM/dd HH:mm"));
                            String oriApt = flight.getOriginAirport().getName();
                            String dstApt = flight.getDestAirport().getName();
                            String text = "  ["+tn+"] " + "[ "+oriTime+" ~ "+dstTime+" ] " + "[" + oriApt + " -> " + dstApt +"]";
                            cell = row.createCell(++k);
                            cell.setCellValue(text);

                            //바둑판 형식으로 색 칠하기, 셀 스타일을 위로 빼서 색만 바꿀 경우 적용이 안됨. 적용할 때 마다 새로 만들어야 함.
                            XSSFColor currentColor = ((k % 2 == 0) && (i % 2 == 0)) || ((k % 2 == 1) && (i % 2 == 1)) ? colors[1] : colors[0];

                            CellStyle contentStyle = workbook.createCellStyle();
                            Font contentfont = workbook.createFont();
                            contentfont.setFontHeightInPoints((short) 9);
                            contentStyle.setAlignment(HorizontalAlignment.LEFT);
                            contentStyle.setBorderBottom(BorderStyle.DASH_DOT_DOT);
                            contentStyle.setBorderLeft(BorderStyle.DASH_DOT_DOT);
                            contentStyle.setFont(contentfont);
                            contentStyle.setFillForegroundColor(currentColor);
                            contentStyle.setFillPattern(FillPatternType.SOLID_FOREGROUND);

                            cell.setCellStyle(contentStyle);

                            sheet.autoSizeColumn(k);
                        }
                    }

                    row = sheet.getRow(0);
                    for(int i=1; i<=maxCell; i++){
                        cell = row.createCell(i);
                        cell.setCellValue("Flight" + i);
                        cell.setCellStyle(headerStyle);
                    }

                    try (FileOutputStream fo = new FileOutputStream("./data/crewpairing/output/" + fileName)) {
                        workbook.write(fo);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
}

