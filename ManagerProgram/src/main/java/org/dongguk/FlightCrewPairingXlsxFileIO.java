package org.dongguk;

import lombok.Getter;
import lombok.NoArgsConstructor;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.*;
import org.dongguk.domain.Aircraft;
import org.dongguk.domain.Airport;
import org.dongguk.domain.Flight;
import org.dongguk.domain.Pairing;

import java.io.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.temporal.ChronoUnit;
import java.util.*;

public class FlightCrewPairingXlsxFileIO {

    public static List<Flight> read(File informationXlsxFile) {
        try (InputStream in = new BufferedInputStream(new FileInputStream(informationXlsxFile))) {
            XSSFWorkbook workbook = new XSSFWorkbook(in);
            return new FlightCrewPairingXlsxReader(workbook).read();
        } catch (IOException | RuntimeException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    public static List<Pairing> readPairing(List<Flight> flightList, File informationXlsxFile) {
        try (InputStream in = new BufferedInputStream(new FileInputStream(informationXlsxFile))) {
            XSSFWorkbook workbook = new XSSFWorkbook(in);
            return new FlightCrewPairingXlsxReader(workbook).readPairingSet(flightList);
        } catch (IOException | RuntimeException e) {
            throw new RuntimeException(e);
        }
    }
    public static void write(List<Pairing> pairingList) {
        new FlightCrewPairingXlsxReader.FlightCrewPairingXlsxWriter().write(pairingList);
    }
    @Getter
    private static class FlightCrewPairingXlsxReader {
        private final List<Aircraft> aircraftList = new ArrayList<>();
        private final List<Airport> airportList = new ArrayList<>();
        private final List<Flight> flightList = new ArrayList<>();

        private final Map<String, Airport> airportMap = new HashMap<>();
        private int exchangeRate;

        protected XSSFWorkbook workbook;
        protected XSSFSheet currentSheet;
        protected Iterator<Row> currentRowIterator;

        public FlightCrewPairingXlsxReader(XSSFWorkbook workbook) {
            this.workbook = workbook;
        }

        protected void nextSheet(String sheetName) {
            currentSheet = workbook.getSheet(sheetName);
            if (currentSheet == null) {
                throw new IllegalStateException("The workbook does not contain a sheet with name ("
                        + sheetName + ").");
            }

            currentRowIterator = currentSheet.rowIterator();
            if (currentRowIterator == null) {
                throw new IllegalStateException("The sheet has no rows.");
            }
        }

        public List<Flight> read() {
//            readTimeData();
            readAircraft();         // 수정
            readAirport();          // 수정
            readDeadhead();         // 수정
            readFlight();

            return flightList;
        }

        public List<Pairing> readPairingSet(List<Flight> inputFlightList) {
            List<Pairing> list = new ArrayList<>();
            nextSheet("Data");    // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵

            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                Iterator<Cell> currentCellIterator = row.cellIterator();

                int indexCnt = (int) currentCellIterator.next().getNumericCellValue();
                List<Flight> pair = new ArrayList<>();
                while (currentCellIterator.hasNext()) {
                    Cell cell = currentCellIterator.next();
                    if (cell.getCellType() == CellType.BLANK || cell.getCellType() == CellType.STRING || cell.getNumericCellValue() == 0) {
                        break;
                    }

                    pair.add(inputFlightList.get((int) cell.getNumericCellValue()));
                }

                list.add(new Pairing(indexCnt, pair, 0));
            }

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
                    (int) row.getCell(5).getNumericCellValue(),
                    (int) row.getCell(6).getNumericCellValue());
        }

        private void readExchangeRate() {
            nextSheet("User_Cost");       // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵
            currentRowIterator.next();              // 빈 행 스킵

            exchangeRate = (int) currentRowIterator.next().getCell(12).getNumericCellValue();
        }

        private void readAircraft() {
            aircraftList.clear();
            nextSheet("Program_Cost");    // Sheet 고르기
            currentRowIterator.next();              // 주제목 스킵
            currentRowIterator.next();              // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    aircraftList.add(Aircraft.builder()
                            .id(indexCnt++)
                            .type(row.getCell(0).getStringCellValue())
                            .crewNum((int) row.getCell(1).getNumericCellValue())
                            .flightCost((int) row.getCell(2).getNumericCellValue())
                            .layoverCost((int) row.getCell(3).getNumericCellValue())
                            .quickTurnCost((int) row.getCell(4).getNumericCellValue()).build());
                } catch (IllegalStateException e) {
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
            currentRowIterator.next();  // 빈 행 스킵
            currentRowIterator.next();  // Header 스킵

            int indexCnt = 0;
            while (currentRowIterator.hasNext()) {
                XSSFRow row = (XSSFRow) currentRowIterator.next();

                try {
                    flightList.add(Flight.builder()
                            .id(indexCnt++)
                            .serialNumber(row.getCell(0).getStringCellValue())
                            .tailNumber(row.getCell(1).getStringCellValue())
                            .originAirport(Airport.of(airportList, row.getCell(2).getStringCellValue()))
                            .originTime(row.getCell(3).getLocalDateTimeCellValue())
                            .destAirport(Airport.of(airportList, row.getCell(4).getStringCellValue()))
                            .destTime(row.getCell(5).getLocalDateTimeCellValue())
                            .aircraft(Aircraft.of(aircraftList, row.getCell(6).getStringCellValue())).build());
                } catch (IllegalStateException e) {
                    break;
                } catch (Exception e) {
                    throw new RuntimeException(e);
                }
            }
        }

        @Getter
        @NoArgsConstructor
        private static class FlightCrewPairingXlsxWriter {
            public void write(List<Pairing> pairingList) {
                String timeStr = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy_MM_dd_HH_mm_ss"));
                exportUserData1(timeStr, pairingList);
            }

            public void exportUserData1(String timeStr, List<Pairing> inputPairingList) {
                String fileName = timeStr + "-userData1.xlsx";
                try (XSSFWorkbook workbook = new XSSFWorkbook()) {
                    XSSFSheet sheet = workbook.createSheet("Data");

                    List<Pairing> pairingList = inputPairingList;
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

                    try (FileOutputStream fo = new FileOutputStream("./data/crewpairing/finish/" + fileName)) {
                        workbook.write(fo);
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
        }
    }
}
