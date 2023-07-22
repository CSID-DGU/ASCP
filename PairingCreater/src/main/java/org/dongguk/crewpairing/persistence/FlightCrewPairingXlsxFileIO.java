package org.dongguk.crewpairing.persistence;

import lombok.Getter;
import lombok.Setter;
import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.dongguk.common.persistence.AbstractXlsxSolutionFileIO;
import org.dongguk.crewpairing.app.PairingApp;
import org.dongguk.crewpairing.domain.Aircraft;
import org.dongguk.crewpairing.domain.Airport;
import org.dongguk.crewpairing.domain.Flight;
import org.dongguk.crewpairing.domain.PairingSolution;
import org.drools.io.ClassPathResource;
import org.optaplanner.core.api.score.buildin.hardsoft.HardSoftScore;

import java.io.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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
}
