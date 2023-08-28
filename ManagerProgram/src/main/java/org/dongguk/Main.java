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
        for (int i = 0; i < count; i++) {
            // Half Cycle: OptaPlanner 실행
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

//            // Half Cycle: Reinforcement Learning 실행
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
//
//            // 만약 사이클에서 변화가 없다면 종료(현재는 점수 변화가 없다면 끝나지만, 추후에는 다른 조건을 추가할 예정)
//            if (otSoftScore.get(i) - rlSoftScore.get(i) == 0) {
//                break;
//            }

            // 임시 종료 조건(OptaPlanner만 실행할 경우)
            if (i != 0 && otSoftScore.get(i) - otSoftScore.get(i - 1) == 0) {
                break;
            }
        }

        for (int i = 0; i < otHardScore.size(); i++) {
            // 인덱스와 같이 Hard, Soft 점수 출력
            System.out.println("[ OT - " + i + " ] " + "Hard Score : " + otHardScore.get(i) + ", Soft Score : " + otSoftScore.get(i));
//            System.out.println("[ RL - " + i + " ] " + "Hard Score : " + rlHardScore.get(i) + ", Soft Score : " + rlSoftScore.get(i));
        }
    }
}