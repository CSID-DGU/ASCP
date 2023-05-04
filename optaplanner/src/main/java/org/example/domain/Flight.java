package org.example.domain;

import jdk.vm.ci.meta.Local;
import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import org.kie.api.definition.rule.ActivationListener;

import java.time.LocalDateTime;


@Getter
@Setter
@AllArgsConstructor
public class Flight {
    private String flightNumber;
    private Airport originAirport;
    private Airport destAirport;
    private LocalDateTime originTime;
    private LocalDateTime destTime;
    private Aircraft aircraft;
    private double cost;

}
