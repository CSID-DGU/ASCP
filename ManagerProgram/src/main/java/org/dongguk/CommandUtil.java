package org.dongguk;

public class CommandUtil {
    public static ProcessBuilder getJavaCommand(String inputFileName, String pairingFileName) {
        if (pairingFileName != null) {
            return new ProcessBuilder("java", "-jar", "crew-pairing.jar",
                    "data/", "crewpairing/" ,
                    inputFileName, pairingFileName);
        } else {
            return new ProcessBuilder("java", "-jar", "crew-pairing.jar",
                    "data/", "crewpairing/",
                    inputFileName);
        }
    }

    public static ProcessBuilder getPythonCommand(String inputFileName, String pairingFileName) {
        if (pairingFileName != null) {
            return new ProcessBuilder("python3", "main.py",
                    "data/", "crewpairing/" ,
                    inputFileName, pairingFileName);
        } else {
            return new ProcessBuilder("python3", "main.py",
                    "data/", "crewpairing/" ,
                    inputFileName);
        }
    }
}
