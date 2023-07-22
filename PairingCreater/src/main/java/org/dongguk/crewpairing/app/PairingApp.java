package org.dongguk.crewpairing.app;

import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.dongguk.crewpairing.domain.*;
import org.dongguk.crewpairing.util.PairingVisualize;
import org.drools.io.ClassPathResource;
import org.optaplanner.core.api.solver.Solver;
import org.optaplanner.core.api.solver.SolverFactory;

import java.io.*;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class PairingApp {
    public static void main(String[] args) {

        SolverFactory<PairingSolution> solverFactory = SolverFactory.createFromXmlResource("solverConfig.xml");

        // Load the problem
        PairingSolution problem = generateDemoData(35);

        // Solve the problem
        Solver<PairingSolution> solver = solverFactory.buildSolver();
        PairingSolution solution = solver.solve(problem);

        // Visualize the solution
        printPairing(solution);
        
        // OutPut Excel
        PairingVisualize.visualize(solution.getPairingList());

        System.exit(0);
    }

    private static void printPairing(PairingSolution pairingSoultion){
        //결과 출력
        System.out.println(pairingSoultion);

    }

    public static List<Airport> readDeadHead(){
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
        List<Airport> airports = new ArrayList<>();
        Map<String, Integer>[] map = new Map[54];
        map[0] = new HashMap<>();
        int cnt = 0;
        // 데이터를 변환합니다.
        List<Airport> dataList = new ArrayList<>();
        Sheet sheet = workbook.getSheetAt(0);
        for (int i = 1; i <= sheet.getLastRowNum(); i++) {
            Row row = sheet.getRow(i);
            String origin = row.getCell(0).getStringCellValue();
            String dest = row.getCell(1).getStringCellValue();
            int deadhead = (int)(row.getCell(2).getNumericCellValue());
            map[cnt].put(dest,deadhead);
            if(sheet.getRow(i+1)== null || origin != sheet.getRow(i+1).getCell(0).getStringCellValue()){
                airports.add(Airport.builder()
                        .id(cnt_index++)
                        .name(row.getCell(0).getStringCellValue())
                        .deadheadCost(map[cnt++]).build());

                map[cnt] = new HashMap<>();
            }

        }
        return airports;
    }

    public static List<Aircraft> readSalary(){
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

        List<Aircraft> aircraftList = new ArrayList<>();

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
        return aircraftList;
    }

    public static List<Flight> readFlight(List<Aircraft> aircraftList, List<Airport> airportList){
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

        List<Flight> flightList = new ArrayList<>();

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
        return flightList;
    }

    public static PairingSolution generateDemoData(int totalpair) {


        List<Airport> airportList = readDeadHead();
        List<Aircraft> aircraftList = readSalary();
        List<Flight> flightList = readFlight(aircraftList, airportList);


        List<Pairing> pairingList = new ArrayList<>();
        //초기 페어링 Set 구성 어차피 solver가 바꿔버려서 의미 없음 아무것도 안넣으면 오류나서 넣는 것
        for (int i=0;i<totalpair;i++){
            List<Flight> pair = new ArrayList<>();
            pair.add(flightList.get(i));
            pairingList.add(new Pairing(i, pair,0));
        }

        return PairingSolution.builder()
                .id(0)
                .aircraftList(aircraftList)
                .airportList(airportList)
                .flightList(flightList)
                .pairingList(pairingList).build();
    }
}
