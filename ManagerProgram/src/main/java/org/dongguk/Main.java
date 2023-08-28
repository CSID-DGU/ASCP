package org.dongguk;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class Main {
    public static void main(String[] args) {
        // 총 사이클 횟수와 informationFile 이름을 입력받는다.
        int count = args.length > 0 ? Integer.parseInt(args[0]) : 1;
        String inputFileName = args.length > 1 ? args[1] : "input_base.xlsx";
        String pairingFileName = null;

        // Hard, Soft 점수를 저장할 리스트 생성
        List<Long> otHardScore = new ArrayList<>();
        List<Long> otSoftScore = new ArrayList<>();

        List<Long> rlHardScore = new ArrayList<>();
        List<Long> rlSoftScore = new ArrayList<>();

        // count만큼 사이클을 돌면서 Hard, Soft 점수를 저장
        while(count-- > 0) {
            try {
                ProcessBuilder pb = CommandUtil.getJavaCommand(inputFileName, pairingFileName);

                System.out.println("Running " + pb.command());
                Process p = pb.start();

                BufferedReader br = new BufferedReader(new InputStreamReader( p.getInputStream() ));
                String line = null;
                while( (line = br.readLine()) != null ){
                    if (line.startsWith("Create Output File : ")) {
                        pairingFileName = line.substring(21);
                    }

                    if (line.startsWith("Hard Score : ")) {
                        otHardScore.add(Long.parseLong(line.substring(13)));
                    }

                    if (line.startsWith("Soft Score : ")) {
                        otSoftScore.add(Long.parseLong(line.substring(13)));
                    }

                    System.out.println(line);
                }
            } catch (Exception e) {
                e.printStackTrace();
            }

//            try {
//                ProcessBuilder pb = CommandUtil.getPythonCommand(inputFileName, pairingFileName);
//
//                System.out.println("Running " + pb.command());
//                Process p = pb.start();
//
//                BufferedReader br = new BufferedReader(new InputStreamReader( p.getInputStream() ));
//                String line = null;
//                while( (line = br.readLine()) != null ){
//                    if (line.startsWith("Create Output File : ")) {
//                        pairingFileName = line.substring(21);
//                    }
//
//                    if (line.startsWith("Hard Score : ")) {
//                        rlHardScore.add(Long.parseLong(line.substring(13)));
//                    }
//
//                    if (line.startsWith("Soft Score : ")) {
//                        rlSoftScore.add(Long.parseLong(line.substring(13)));
//                    }
//
//                    System.out.println(line);
//                }
//            } catch (Exception e) {
//                e.printStackTrace();
//            }
        }

        for (int j = 0; j < otHardScore.size(); j++) {
            // 인덱스와 같이 Hard, Soft 점수 출력
            System.out.println("[ OT - " + j + " ] " + "Hard Score : " + otHardScore.get(j) + ", Soft Score : " + otSoftScore.get(j));
//            System.out.println("[ RL - " + j + " ] " + "Hard Score : " + rlHardScore.get(j) + ", Soft Score : " + rlSoftScore.get(j));
        }
    }
}