package org.example;


import org.apache.poi.ss.usermodel.Row;
import org.apache.poi.ss.usermodel.Sheet;
import org.apache.poi.ss.usermodel.Workbook;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import org.example.domain.*;
import org.example.score.ParingConstraintProvider;
import org.optaplanner.core.api.solver.Solver;
import org.optaplanner.core.api.solver.SolverFactory;
import org.optaplanner.core.config.solver.SolverConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.time.LocalDateTime;

public class PairingApp {

    private static final Logger LOGGER = LoggerFactory.getLogger(PairingApp.class);
    public static void main(String[] args) {
        SolverFactory<PairingSoultion> solverFactory = SolverFactory.create(new SolverConfig()
                .withSolutionClass(PairingSoultion.class)
                .withEntityClasses(Pairing.class)
                .withConstraintProviderClass(ParingConstraintProvider.class)
                .withTerminationSpentLimit(Duration.ofSeconds(5)));

        // Load the problem
        PairingSoultion problem = generateDemoData();

        // Solve the problem
        Solver<PairingSoultion> solver = solverFactory.buildSolver();
        PairingSoultion solution = solver.solve(problem);
        // Visualize the solution
        printPairing(solution);

        System.exit(0);
    }

    private static void printPairing(PairingSoultion pairingSoultion){


        System.out.println(pairingSoultion);
        pairingSoultion.printParingList();

    }

    public static List<Airport> readAirport(){
        FileInputStream fileInputStream;

        {
            try {
                fileInputStream = new FileInputStream(new File("deadhead.xlsx"));
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        }

        Workbook workbook;

        {
            try {
                workbook = new XSSFWorkbook(fileInputStream);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        List<Airport> airports = new ArrayList<>();
        Map<String, Integer>[] map = new Map[10];
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
            if(sheet.getRow(i+1)== null || origin!= sheet.getRow(i+1).getCell(0).getStringCellValue()){
                airports.add(new Airport(row.getCell(0).getStringCellValue(),map[cnt]));
                cnt++;
                map[cnt] = new HashMap<>();
            }

        }
        return airports;
    }

    public static List<Aircraft> readAircraft(){
        FileInputStream fileInputStream;

        {
            try {
                fileInputStream = new FileInputStream(new File("salary.xlsx"));
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        }

        Workbook workbook;

        {
            try {
                workbook = new XSSFWorkbook(fileInputStream);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        List<Aircraft> aircraftList = new ArrayList<>();

        Sheet sheet = workbook.getSheetAt(0);
        for (int i = 1; i <= sheet.getLastRowNum(); i++) {
            Row row = sheet.getRow(i);
            String aircraft = row.getCell(0).getStringCellValue();
            int crewNum = (int)(row.getCell(1).getNumericCellValue());
            int flightSalary = (int)(row.getCell(2).getNumericCellValue());
            int baseSalary = (int)(row.getCell(3).getNumericCellValue());
            int layoverSalary = (int)(row.getCell(4).getNumericCellValue());
            aircraftList.add(new Aircraft(aircraft,crewNum,flightSalary,baseSalary,layoverSalary));

        }
        return aircraftList;
    }

    public static List<Flight> readFlight(List<Aircraft> aircraftList, List<Airport> airportList){
        FileInputStream fileInputStream;

        {
            try {
                fileInputStream = new FileInputStream(new File("flight.xlsx"));
            } catch (FileNotFoundException e) {
                throw new RuntimeException(e);
            }
        }

        Workbook workbook;

        {
            try {
                workbook = new XSSFWorkbook(fileInputStream);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }

        List<Flight> flightList = new ArrayList<>();

        Sheet sheet = workbook.getSheetAt(0);
        for (int i = 1; i <= sheet.getLastRowNum(); i++) {
            Row row = sheet.getRow(i);
            String index = row.getCell(0).getStringCellValue();
            String origin = row.getCell(1).getStringCellValue();
            LocalDateTime originDate = row.getCell(2).getLocalDateTimeCellValue();
            String dest = row.getCell(3).getStringCellValue();
            LocalDateTime destDate = row.getCell(4).getLocalDateTimeCellValue();
            String aircraft = row.getCell(6).getStringCellValue();

            flightList.add(new Flight(index,Airport.findAirportByName(airportList,origin), originDate, Airport.findAirportByName(airportList,dest), destDate, Aircraft.findAircraftName(aircraftList,aircraft)));

        }
        return flightList;
    }

    public static PairingSoultion generateDemoData() {


        List<Airport> airportList = readAirport();
        List<Aircraft> aircraftList = readAircraft();
        List<Flight> flightList = readFlight(aircraftList,airportList);
        //DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

        List<Pairing> pairingList = new ArrayList<>();
        for (int i=0;i<5;i++){
            List<Flight> pair = new ArrayList<>();
            pair.add(flightList.get(i));
            pairingList.add(new Pairing(pair,0));
        }

        return new PairingSoultion(aircraftList,airportList,flightList,pairingList);
    }
}
