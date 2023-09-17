package org.dongguk;

public class CommandUtil {
    public static ProcessBuilder getJavaCommand(String flightSize, String inputFileName, String pairingFileName) {
        if (pairingFileName != null) {
            return new ProcessBuilder("java", "-jar", "crew-pairing.jar",
                    "data/", "crewpairing/" , flightSize,
                    inputFileName, pairingFileName);
        } else {
            return new ProcessBuilder("java", "-jar", "crew-pairing.jar",
                    "data/", "crewpairing/", flightSize,
                    inputFileName);
        }
    }

    public static ProcessBuilder getPythonCommand(String inputFileName, String pairingFileName) {
        if (pairingFileName != null) {
            return new ProcessBuilder("python3", "./RL/main_final.py",
                    "./data/crewpairing/input/", inputFileName ,
                    "/home/public/airline1/data/crewpairing/output/" + pairingFileName);
        } else {
            throw new IllegalArgumentException("파일명을 입력해주세요.");
        }
    }
    /**
     * python3 ./RL/main_final.py ./data/crewpairing/input/ input_500.xlsx ./data/crewpairing/output/2023_09_06_12_04_07-pairingData.xlsx
     */
}
