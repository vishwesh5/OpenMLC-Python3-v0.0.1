EESchema Schematic File Version 2
LIBS:power
LIBS:device
LIBS:transistors
LIBS:conn
LIBS:linear
LIBS:regul
LIBS:74xx
LIBS:cmos4000
LIBS:adc-dac
LIBS:memory
LIBS:xilinx
LIBS:microcontrollers
LIBS:dsp
LIBS:microchip
LIBS:analog_switches
LIBS:motorola
LIBS:texas
LIBS:intel
LIBS:audio
LIBS:interface
LIBS:digital-audio
LIBS:philips
LIBS:display
LIBS:cypress
LIBS:siliconi
LIBS:opto
LIBS:atmel
LIBS:contrib
LIBS:valves
LIBS:myLib
LIBS:new_project-cache
EELAYER 25 0
EELAYER END
$Descr A4 11693 8268
encoding utf-8
Sheet 1 1
Title "Tutorial1"
Date ""
Rev ""
Comp ""
Comment1 ""
Comment2 ""
Comment3 ""
Comment4 ""
$EndDescr
$Comp
L R R2
U 1 1 5800221C
P 4850 3250
F 0 "R2" V 4930 3250 50  0000 C CNN
F 1 "1k" V 4850 3250 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM10mm" V 4780 3250 50  0001 C CNN
F 3 "" H 4850 3250 50  0000 C CNN
	1    4850 3250
	0    1    1    0   
$EndComp
$Comp
L R R3
U 1 1 5800235D
P 5150 3650
F 0 "R3" V 5230 3650 50  0000 C CNN
F 1 "1k" V 5150 3650 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM10mm" V 5080 3650 50  0001 C CNN
F 3 "" H 5150 3650 50  0000 C CNN
	1    5150 3650
	1    0    0    -1  
$EndComp
$Comp
L R R1
U 1 1 580023F0
P 4850 3050
F 0 "R1" V 4930 3050 50  0000 C CNN
F 1 "1k" V 4850 3050 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM10mm" V 4780 3050 50  0001 C CNN
F 3 "" H 4850 3050 50  0000 C CNN
	1    4850 3050
	0    1    1    0   
$EndComp
$Comp
L R R4
U 1 1 5800248B
P 5750 2650
F 0 "R4" V 5830 2650 50  0000 C CNN
F 1 "1k" V 5750 2650 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM10mm" V 5680 2650 50  0001 C CNN
F 3 "" H 5750 2650 50  0000 C CNN
	1    5750 2650
	0    1    1    0   
$EndComp
Wire Wire Line
	5150 2650 5150 3150
Wire Wire Line
	5150 2650 5600 2650
Connection ~ 5150 3050
Wire Wire Line
	6200 2650 6200 3150
Wire Wire Line
	5150 3250 5150 3500
Connection ~ 5150 3250
Wire Wire Line
	5150 4050 5150 3800
$Comp
L ZENER D1
U 1 1 58017DCD
P 7000 2950
F 0 "D1" H 7000 3050 50  0000 C CNN
F 1 "ZENER" H 7000 2850 50  0000 C CNN
F 2 "Discret:D3" H 7000 2950 50  0001 C CNN
F 3 "" H 7000 2950 50  0000 C CNN
	1    7000 2950
	0    1    1    0   
$EndComp
Wire Wire Line
	5150 3950 7000 3950
Connection ~ 5150 3950
Connection ~ 6200 2650
$Comp
L GND #PWR01
U 1 1 58018568
P 5150 4050
F 0 "#PWR01" H 5150 3800 50  0001 C CNN
F 1 "GND" H 5150 3900 50  0000 C CNN
F 2 "" H 5150 4050 50  0000 C CNN
F 3 "" H 5150 4050 50  0000 C CNN
	1    5150 4050
	1    0    0    -1  
$EndComp
$Comp
L PWR_FLAG #FLG02
U 1 1 58018EC0
P 6400 4250
F 0 "#FLG02" H 6400 4345 50  0001 C CNN
F 1 "PWR_FLAG" H 6400 4430 50  0000 C CNN
F 2 "" H 6400 4250 50  0000 C CNN
F 3 "" H 6400 4250 50  0000 C CNN
	1    6400 4250
	1    0    0    -1  
$EndComp
$Comp
L GND #PWR03
U 1 1 58018EE0
P 6400 4250
F 0 "#PWR03" H 6400 4000 50  0001 C CNN
F 1 "GND" H 6400 4100 50  0000 C CNN
F 2 "" H 6400 4250 50  0000 C CNN
F 3 "" H 6400 4250 50  0000 C CNN
	1    6400 4250
	1    0    0    -1  
$EndComp
$Comp
L +15V #PWR04
U 1 1 58018F39
P 5500 2750
F 0 "#PWR04" H 5500 2600 50  0001 C CNN
F 1 "+15V" H 5500 2890 50  0000 C CNN
F 2 "" H 5500 2750 50  0000 C CNN
F 3 "" H 5500 2750 50  0000 C CNN
	1    5500 2750
	1    0    0    -1  
$EndComp
$Comp
L -15V #PWR5
U 1 1 58018F66
P 5950 4250
F 0 "#PWR5" H 5950 4350 50  0001 C CNN
F 1 "-15V" H 5950 4400 50  0000 C CNN
F 2 "" H 5950 4250 50  0000 C CNN
F 3 "" H 5950 4250 50  0000 C CNN
	1    5950 4250
	-1   0    0    1   
$EndComp
$Comp
L +15V #PWR05
U 1 1 58019044
P 5550 4250
F 0 "#PWR05" H 5550 4100 50  0001 C CNN
F 1 "+15V" H 5550 4390 50  0000 C CNN
F 2 "" H 5550 4250 50  0000 C CNN
F 3 "" H 5550 4250 50  0000 C CNN
	1    5550 4250
	-1   0    0    1   
$EndComp
$Comp
L -15V #PWR3
U 1 1 580190A2
P 5500 3650
F 0 "#PWR3" H 5500 3750 50  0001 C CNN
F 1 "-15V" H 5500 3800 50  0000 C CNN
F 2 "" H 5500 3650 50  0000 C CNN
F 3 "" H 5500 3650 50  0000 C CNN
	1    5500 3650
	-1   0    0    1   
$EndComp
$Comp
L PWR_FLAG #FLG06
U 1 1 580190D1
P 5550 4250
F 0 "#FLG06" H 5550 4345 50  0001 C CNN
F 1 "PWR_FLAG" H 5550 4430 50  0000 C CNN
F 2 "" H 5550 4250 50  0000 C CNN
F 3 "" H 5550 4250 50  0000 C CNN
	1    5550 4250
	1    0    0    -1  
$EndComp
$Comp
L PWR_FLAG #FLG07
U 1 1 58019400
P 5950 4250
F 0 "#FLG07" H 5950 4345 50  0001 C CNN
F 1 "PWR_FLAG" H 5950 4430 50  0000 C CNN
F 2 "" H 5950 4250 50  0000 C CNN
F 3 "" H 5950 4250 50  0000 C CNN
	1    5950 4250
	1    0    0    -1  
$EndComp
Wire Wire Line
	5500 3650 5500 3450
Wire Wire Line
	5500 2850 5500 2750
Wire Wire Line
	4700 3250 4150 3250
Wire Wire Line
	4700 3050 4150 3050
Text Label 4300 3050 0    60   ~ 0
V1_LBL
Text Label 4300 3250 0    60   ~ 0
V2_LBL
Wire Wire Line
	8100 4200 7650 4200
Text Label 8050 4200 2    60   ~ 0
V1_LBL
Wire Wire Line
	8100 4100 7650 4100
Text Label 8050 4100 2    60   ~ 0
V2_LBL
Wire Wire Line
	4500 4100 4050 4100
Wire Wire Line
	8100 4750 7650 4750
Text Label 4500 4100 2    60   ~ 0
GND_LBL
Text Label 8100 4750 2    60   ~ 0
15V_LBL
$Comp
L GND #PWR08
U 1 1 58024DAA
P 7350 3100
F 0 "#PWR08" H 7350 2850 50  0001 C CNN
F 1 "GND" H 7350 2950 50  0000 C CNN
F 2 "" H 7350 3100 50  0000 C CNN
F 3 "" H 7350 3100 50  0000 C CNN
	1    7350 3100
	0    1    1    0   
$EndComp
$Comp
L -15V #PWR8
U 1 1 58024DB0
P 7600 3400
F 0 "#PWR8" H 7600 3500 50  0001 C CNN
F 1 "-15V" H 7600 3550 50  0000 C CNN
F 2 "" H 7600 3400 50  0000 C CNN
F 3 "" H 7600 3400 50  0000 C CNN
	1    7600 3400
	0    -1   -1   0   
$EndComp
$Comp
L +15V #PWR09
U 1 1 58024DB6
P 7650 2850
F 0 "#PWR09" H 7650 2700 50  0001 C CNN
F 1 "+15V" H 7650 2990 50  0000 C CNN
F 2 "" H 7650 2850 50  0000 C CNN
F 3 "" H 7650 2850 50  0000 C CNN
	1    7650 2850
	0    -1   -1   0   
$EndComp
Wire Wire Line
	7650 2850 8200 2850
Wire Wire Line
	7600 3400 8150 3400
Wire Wire Line
	7350 3100 7900 3100
Text Label 7700 2850 0    60   ~ 0
15V_LBL
Text Label 7600 3400 0    60   ~ 0
-15V_LBL
Text Label 7400 3100 0    60   ~ 0
GND_LBL
Wire Wire Line
	8100 4650 7650 4650
Text Label 8100 4650 2    60   ~ 0
-15V_LBL
$Comp
L LM741 U1
U 1 1 580B82A7
P 5600 3150
F 0 "U1" H 5600 3400 50  0000 L CNN
F 1 "LM741" H 5600 3300 50  0000 L CNN
F 2 "Housings_DIP:DIP-8_W7.62mm" H 5650 3200 50  0001 C CNN
F 3 "" H 5750 3300 50  0000 C CNN
	1    5600 3150
	1    0    0    -1  
$EndComp
Wire Wire Line
	5700 3450 6000 3450
Wire Wire Line
	5600 3450 5600 3600
Wire Wire Line
	5600 3600 6000 3600
Text Label 5800 3450 0    60   ~ 0
N1
Text Label 5800 3600 0    60   ~ 0
N2
Wire Wire Line
	5900 2650 6450 2650
$Comp
L R R5
U 1 1 580B849B
P 6600 2650
F 0 "R5" V 6680 2650 50  0000 C CNN
F 1 "68" V 6600 2650 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM10mm" V 6530 2650 50  0001 C CNN
F 3 "" H 6600 2650 50  0000 C CNN
	1    6600 2650
	0    1    1    0   
$EndComp
Wire Wire Line
	6750 2650 7350 2650
$Comp
L POT RV1
U 1 1 580B853A
P 4050 3700
F 0 "RV1" H 4050 3620 50  0000 C CNN
F 1 "POT" H 4050 3700 50  0000 C CNN
F 2 "Potentiometers:Potentiometer_Alps-RK163-single_15mm" H 4050 3700 50  0001 C CNN
F 3 "" H 4050 3700 50  0000 C CNN
	1    4050 3700
	1    0    0    -1  
$EndComp
Wire Wire Line
	3900 3700 3500 3700
Wire Wire Line
	4200 3700 4650 3700
Wire Wire Line
	4050 3550 4050 3450
Text Label 3250 3450 0    60   ~ 0
-15V_LBL
Text Label 3650 3700 0    60   ~ 0
N1
Text Label 4350 3700 0    60   ~ 0
N2
$Comp
L SW_PUSH SW1
U 1 1 580B8C61
P 7000 3550
F 0 "SW1" H 7150 3660 50  0000 C CNN
F 1 "SW_PUSH" H 7000 3470 50  0000 C CNN
F 2 "Buttons_Switches_ThroughHole:SW_DIP_x2_Piano" H 7000 3550 50  0001 C CNN
F 3 "" H 7000 3550 50  0000 C CNN
	1    7000 3550
	0    -1   -1   0   
$EndComp
Wire Wire Line
	7000 2650 7000 2750
Wire Wire Line
	7000 3250 7000 3150
Wire Wire Line
	7000 3950 7000 3850
$Comp
L CP C2
U 1 1 580B9345
P 8200 3000
F 0 "C2" H 8225 3100 50  0000 L CNN
F 1 "0.1u" H 8225 2900 50  0000 L CNN
F 2 "Capacitors_ThroughHole:C_Radial_D5_L6_P2.5" H 8238 2850 50  0001 C CNN
F 3 "" H 8200 3000 50  0000 C CNN
	1    8200 3000
	1    0    0    -1  
$EndComp
$Comp
L CP C1
U 1 1 580B94BF
P 8150 3550
F 0 "C1" H 8175 3650 50  0000 L CNN
F 1 "0.1u" H 8175 3450 50  0000 L CNN
F 2 "Capacitors_ThroughHole:C_Radial_D5_L6_P2.5" H 8188 3400 50  0001 C CNN
F 3 "" H 8150 3550 50  0000 C CNN
	1    8150 3550
	1    0    0    -1  
$EndComp
$Comp
L GND #PWR010
U 1 1 580B950E
P 8200 3150
F 0 "#PWR010" H 8200 2900 50  0001 C CNN
F 1 "GND" H 8200 3000 50  0000 C CNN
F 2 "" H 8200 3150 50  0000 C CNN
F 3 "" H 8200 3150 50  0000 C CNN
	1    8200 3150
	1    0    0    -1  
$EndComp
$Comp
L GND #PWR011
U 1 1 580B9546
P 8150 3700
F 0 "#PWR011" H 8150 3450 50  0001 C CNN
F 1 "GND" H 8150 3550 50  0000 C CNN
F 2 "" H 8150 3700 50  0000 C CNN
F 3 "" H 8150 3700 50  0000 C CNN
	1    8150 3700
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X05 P1
U 1 1 580BA377
P 4700 4200
F 0 "P1" H 4700 4500 50  0000 C CNN
F 1 "CONN_01X05" V 4800 4200 50  0000 C CNN
F 2 "Pin_Headers:Pin_Header_Straight_1x05" H 4700 4200 50  0001 C CNN
F 3 "" H 4700 4200 50  0000 C CNN
	1    4700 4200
	1    0    0    -1  
$EndComp
Wire Wire Line
	4500 4000 4050 4000
Text Label 4500 4000 2    60   ~ 0
GND_LBL
Wire Wire Line
	4500 4300 4050 4300
Text Label 4500 4300 2    60   ~ 0
GND_LBL
Wire Wire Line
	4500 4200 4050 4200
Text Label 4500 4200 2    60   ~ 0
GND_LBL
Wire Wire Line
	4500 4400 4050 4400
Text Label 4500 4400 2    60   ~ 0
GND_LBL
$Comp
L CONN_01X03 P2
U 1 1 580BAA09
P 7300 4250
F 0 "P2" H 7300 4450 50  0000 C CNN
F 1 "CONN_01X03" V 7400 4250 50  0000 C CNN
F 2 "Pin_Headers:Pin_Header_Straight_1x03" H 7300 4250 50  0001 C CNN
F 3 "" H 7300 4250 50  0000 C CNN
	1    7300 4250
	1    0    0    -1  
$EndComp
Connection ~ 6200 3150
Wire Wire Line
	7100 4150 6750 4150
Wire Wire Line
	7100 4250 6750 4250
Wire Wire Line
	7100 4350 6750 4350
Text Label 6900 4150 0    60   ~ 0
VO
Text Label 6900 4250 0    60   ~ 0
VO
Text Label 6900 4350 0    60   ~ 0
VO
$Comp
L CONN_01X02 P3
U 1 1 580BB0E9
P 8300 4150
F 0 "P3" H 8300 4300 50  0000 C CNN
F 1 "CONN_01X02" V 8400 4150 50  0000 C CNN
F 2 "Pin_Headers:Pin_Header_Straight_1x02" H 8300 4150 50  0001 C CNN
F 3 "" H 8300 4150 50  0000 C CNN
	1    8300 4150
	1    0    0    -1  
$EndComp
$Comp
L CONN_01X02 P4
U 1 1 580BB13C
P 8300 4700
F 0 "P4" H 8300 4850 50  0000 C CNN
F 1 "CONN_01X02" V 8400 4700 50  0000 C CNN
F 2 "Pin_Headers:Pin_Header_Straight_1x02" H 8300 4700 50  0001 C CNN
F 3 "" H 8300 4700 50  0000 C CNN
	1    8300 4700
	1    0    0    -1  
$EndComp
$Comp
L R R6
U 1 1 580BC3FB
P 3850 3450
F 0 "R6" V 3930 3450 50  0000 C CNN
F 1 "1.5k" V 3850 3450 50  0000 C CNN
F 2 "Resistors_ThroughHole:Resistor_Horizontal_RM10mm" V 3780 3450 50  0001 C CNN
F 3 "" H 3850 3450 50  0000 C CNN
	1    3850 3450
	0    1    1    0   
$EndComp
Wire Wire Line
	4050 3450 4000 3450
Wire Wire Line
	3700 3450 3200 3450
Wire Wire Line
	6200 3150 5900 3150
Connection ~ 7000 2650
Text Label 7150 2650 0    60   ~ 0
VO
Wire Wire Line
	5000 3250 5000 3150
Wire Wire Line
	5000 3150 5150 3150
Wire Wire Line
	5300 3050 5150 3050
Wire Wire Line
	5000 3050 5100 3050
Wire Wire Line
	5100 3050 5100 3250
Wire Wire Line
	5100 3250 5300 3250
$EndSCHEMATC
