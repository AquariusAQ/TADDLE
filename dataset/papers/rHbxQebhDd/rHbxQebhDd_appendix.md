## A ABBREVIATIONS AND NOTATIONS

This appendix provides a comprehensive list of the abbreviations and notations used throughout the paper.

A.1 FLIGHT-RELATED NOTATIONS

• F: Set of all flights

• origin: Origin of a flight.

• dest: Destination of a flight.

• SET: Scheduled Elapsed Time.

• SAT: Scheduled Arrival Time.

• SDT: Scheduled Departure Time.

• ADT: Actual Departure Time.

• AET: Actual Elapsed Time.

• AAT: Actual Arrival Time.

A.2 Cost-related Notations

• c: nominal cost.

• rc: Reliable cost.

### A.3 METRICS

• P-index: P-index, a new metric introduced to measure the predictive power of the model.

•  $ C^{td} $ : Total Delay Cost index.

### A.4 TIME AND SURVIVAL MODEL

•  $ t_{q} $ : Query time.

• f: Survival model function.

• S: Survival function.

•  $ p_{=} $ : Probability of being within the acceptable delay range.

• MAE $ _{>} $ , MAE $ _{=} $ , MAE $ _{<} $ : Mean Absolute Errors above, within, and below the predicted threshold.

•  $ \hat{q} $ : Predicted time of an event.

### A.5 DATASET, NETWORK, AND CONSTRAINTS

• D: Dataset of flights and connections.

• C: Constraints set.

• N: Flight network.

• A: Set of arcs in the network.

• c: Crew base.

•  $ \hat{F} $ : Pruned set of flights.

• δ: Sit or Connection time between flights (assumed to be 60 mins).

### A.6 Pairing-related Notations

• P: Set of all pairings.

•  $ P_{opt} $ : Set of optimal pairings.

• p: Single pairing.

• R: Actual elapsed distribution.

### A.7 KDE AND RELIABILITY

•  $ KDE(\text{matched flights}) $ : Kernel Density Estimation for matched flights.

• r: Reliability score of a flight connection.

•  $ \phi $ : Cost function adjusted for reliability.

## B SUMMARY OF SURVIVAL MODELS

<div style="text-align: center;">Table 5: Summary of Survival Analysis Methods</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'>Model Type</td><td style='text-align: center;'>Prop. Constraint</td><td style='text-align: center;'>Main Benefit</td></tr><tr><td style='text-align: center;'>Cox Proportional Regression Cox (1972)</td><td style='text-align: center;'>Continuous</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Most Interpretable</td></tr><tr><td style='text-align: center;'>DeepSurv Katzman et al. (2018)</td><td style='text-align: center;'>Continuous</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Handles non-linearity (Uses NN for reg Cox)</td></tr><tr><td style='text-align: center;'>Cox-Time Kvamme et al. (2019)</td><td style='text-align: center;'>Continuous</td><td style='text-align: center;'>No</td><td style='text-align: center;'>Extends Cox Reg beyond prop. hazards</td></tr><tr><td style='text-align: center;'>Cox-CC Kvamme et al. (2019)</td><td style='text-align: center;'>Continuous</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Proportional version of Cox-Time</td></tr><tr><td style='text-align: center;'>Random Survival Forests Ishwaran et al. (2008b)</td><td style='text-align: center;'>Continuous</td><td style='text-align: center;'>No</td><td style='text-align: center;'>Handles interactions and non-linearity</td></tr><tr><td style='text-align: center;'>DeepHit Lee et al. (2018)</td><td style='text-align: center;'>Discrete</td><td style='text-align: center;'>No</td><td style='text-align: center;'>Best discriminative ability (C-index)</td></tr></table>

## C INSTANCE GENERATION

To set up a crew pairing experiment, we first generate an instance of crew operations consisting of a closed network of flights and connections over a fixed period. The crew starts and finishes at a specified crew base. Specifically, an instance is described by a network N containing a set of flights  $ \hat{F} $ , connections  $ \hat{A} $ , and costs  $ \hat{C} $ . Using a set of spatiotemporal constraints C, we construct N as follows:

1. Filter D according to C to obtain a set of flights  $ F_{0} $ .

2. Construct an initial network  $ N_{0} $  by applying space and time constraints to  $ F_{0} $ . A connection between flight i and flight j is feasible if  $ \text{origin}_{j} = \text{dest}_{i} $  and  $ \delta_{min} \leq \text{SDT}_{j} - \text{SAT}_{i} \leq \delta_{max} $ , where,  $ \delta_{min}, \delta_{max} $  are the minimum and maximum connection times.



3. Prune  $ N_{0} $  to remove redundant flights and extract the subgraph N describing our instance. We create an instance for December 2-5, 2019, using the flight operations of Endeavor Air between all the airports in the network on these dates, with John F. Kennedy International Airport (JFK) as the crew base.



<div style="text-align: center;"><img src="imgs/img_in_image_box_212_550_1006_830.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 6: Block diagram for Instance Generation</div>


Filtering: During the filtering phase, we specify space and time constraints to select relevant flights from the dataset. The constraints include the specific dates and the crew base for starting and ending operations. This phase aims to narrow down the vast dataset to a manageable subset that is relevant to the instance we want to create. By applying these constraints, we extract a filtered set of flights  $ F_{0} $  from the dataset D, such that  $ F_{0} = \{i \in D | i  satisfies  C\} $ . This step ensures that only flights within the specified dates and that either start or end at the crew base are included in the instance.

Constructing a Network: In the network construction phase, we form connections between the filtered flights. The connections represent possible pairings of flights that a crew can operate within the given constraints. The network includes nodes for the crew base and connections between flights that are feasible based on time constraints. Specifically, a connection between flight i and flight j is feasible if the destination of i matches the origin of j, and the time difference between the scheduled departure time of flight j and the scheduled arrival time of flight i falls within the allowable connection time range. The cost for a connection is calculated as  $ SDT_{j} - SAT_{i} + SET_{i} $ , where SDT is the scheduled departure time, SAT is the scheduled arrival time, and SET is the scheduled elapsed time.

Network Pruning: The pruning phase ensures that the network remains practical and feasible for crew pairings. During this phase, we remove redundant or infeasible flights and connections that do not contribute to viable pairings. This is done by identifying and retaining only those flights that can form a continuous path from the origin to the destination crew base. Flights that do not participate in any such path are pruned out.

## D FLIGHT MATCHING CRITERIA AND LATE AIRCRAFT DELAY

The matched flights are identified based on the following criteria:

<div style="text-align: center;">Table 6: Sensitivity analysis of  $ \alpha $ . Higher  $ \alpha $  penalizes longer connection times. Headers: DFC - Deadhead Flying Cost, DCC - Deadhead Connection Cost, TFC - Total Flying Cost, TCC - Total Connection Cost, TC - Total Cost.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">$ \alpha $</td><td colspan="2">Deadheading Cost</td><td colspan="3">Total Cost</td></tr><tr><td style='text-align: center;'>DFC</td><td style='text-align: center;'>DCC</td><td style='text-align: center;'>TFC</td><td style='text-align: center;'>TCC</td><td style='text-align: center;'>TC</td></tr><tr><td style='text-align: center;'>1</td><td style='text-align: center;'>1657</td><td style='text-align: center;'>13244</td><td style='text-align: center;'>27942</td><td style='text-align: center;'>216189</td><td style='text-align: center;'>244131</td></tr><tr><td style='text-align: center;'>0.5</td><td style='text-align: center;'>260 (-84.30)</td><td style='text-align: center;'>1922 (-85.49)</td><td style='text-align: center;'>26545 (-5.00)</td><td style='text-align: center;'>217586 (0.65)</td><td style='text-align: center;'>244131 (0.00)</td></tr><tr><td style='text-align: center;'>2</td><td style='text-align: center;'>2948 (77.91)</td><td style='text-align: center;'>17436 (31.62)</td><td style='text-align: center;'>29233 (4.63)</td><td style='text-align: center;'>214933 (-0.58)</td><td style='text-align: center;'>244166 (0.01)</td></tr><tr><td style='text-align: center;'>3</td><td style='text-align: center;'>3099 (87.01)</td><td style='text-align: center;'>17130 (29.41)</td><td style='text-align: center;'>29384 (5.18)</td><td style='text-align: center;'>214777 (-0.65)</td><td style='text-align: center;'>244161 (0.01)</td></tr><tr><td style='text-align: center;'>4</td><td style='text-align: center;'>3119 (88.17)</td><td style='text-align: center;'>17406 (31.48)</td><td style='text-align: center;'>29404 (5.23)</td><td style='text-align: center;'>214727 (-0.68)</td><td style='text-align: center;'>244131 (0.00)</td></tr><tr><td style='text-align: center;'>5</td><td style='text-align: center;'>3119 (88.17)</td><td style='text-align: center;'>16211 (22.38)</td><td style='text-align: center;'>29404 (5.23)</td><td style='text-align: center;'>214762 (-0.66)</td><td style='text-align: center;'>244166 (0.01)</td></tr></table>

• Origin: The airport from which the flight departs.

• Destination: The airport to which the flight arrives.

• Time of Day of the Scheduled Arrival: The time of day when the flight is scheduled to arrive. This can be segmented into different periods, such as morning (06:00 AM - 11:59 AM), afternoon (12:00 PM - 04:59 PM), evening (05:00 PM - 10:59 PM), and night (11:00 PM - 05:59 AM).

The LateAircraftDelay is not included in the initial delay estimation to avoid double-counting, as this category represents delays caused by propagation through aircraft connections. This delay is simulated separately following Antunes et al. (2019), where the delay is approximated as the difference between actual and scheduled arrival times, minus the LateAircraftDelay.

## E SENSITIVITY ANALYSIS OF  $ \alpha $ 

Sensitivity analysis of  $ \alpha $ : For  $ \alpha = 0.5 $ , deadhead fly and connection costs drop significantly (-84.3% and -85.49%, respectively), with only a slight increase in total connection cost (+0.65%) as shown in Table 6. As  $ \alpha $  increases to 1, costs rise, with deadhead fly cost increasing by 77.91% and connection cost by 31.62%. At  $ \alpha = 3 $ , deadhead fly cost peaks (+87.01%), with minimal changes for higher values. Deadheads increase from 2 at  $ \alpha = 0.5 $  to 28 at  $ \alpha \geq 3 $ , indicating diminishing returns beyond this point.

## F Detailed Scenario Notations

<div style="text-align: center;">Table 7: Simulation Scenarios Based on Percentage Irregularity and Level of Delay. R: Regular Operations, IR: Irregular Operations. For IR runs, delay values are sampled from the specified percentile using KDE.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Scenario Notation</td><td style='text-align: center;'>Description</td></tr><tr><td style='text-align: center;'>100R 0</td><td style='text-align: center;'>100% of runs are R</td></tr><tr><td style='text-align: center;'>75R, 25IR 70</td><td style='text-align: center;'>75% of runs are R, 25% from &gt;70 percentile</td></tr><tr><td style='text-align: center;'>75R, 25IR 80</td><td style='text-align: center;'>75% of runs are R, 25% from &gt;80 percentile</td></tr><tr><td style='text-align: center;'>75R, 25IR 90</td><td style='text-align: center;'>75% of runs are R, 25% from &gt;90 percentile</td></tr><tr><td style='text-align: center;'>50R, 50IR 70</td><td style='text-align: center;'>50% of runs are R, 50% from &gt;70 percentile</td></tr><tr><td style='text-align: center;'>50R, 50IR 80</td><td style='text-align: center;'>50% of runs are R, 50% from &gt;80 percentile</td></tr><tr><td style='text-align: center;'>50R, 50IR 90</td><td style='text-align: center;'>50% of runs are R, 50% from &gt;90 percentile</td></tr><tr><td style='text-align: center;'>25R, 75IR 70</td><td style='text-align: center;'>25% of runs are R, 75% from &gt;70 percentile</td></tr><tr><td style='text-align: center;'>25R, 75IR 80</td><td style='text-align: center;'>25% of runs are R, 75% from &gt;80 percentile</td></tr><tr><td style='text-align: center;'>25R, 75IR 90</td><td style='text-align: center;'>25% of runs are R, 75% from &gt;90 percentile</td></tr><tr><td style='text-align: center;'>100IR 70</td><td style='text-align: center;'>100% sample delay from &gt;70 percentile</td></tr></table>

## G DETAILED PERFORMANCE COMPARISON

As we analyze the results, Figure 7 clearly shows the trends in performance as irregular operations increase. The total propagated delays (TPGD) become more severe as both irregularity levels and the percentiles of delay rise, demonstrating the greater importance of incorporating reliability into decision-making, especially when met with disruptions.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_321_1007_849.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 7: Total Propagated Delays for 75R,25IR Scenarios (70, 80, 90). N: Nominal, R: Reliable. The "75R,25IR" denotes the percentage of regular and irregular runs, respectively. The numbers 70, 80, and 90 indicate the level of delay in each scenario, which increases from left to right. Total Propagated Delays show significant improvements as irregularity increases or as the level of delay rises.</div>


<div style="text-align: center;">Table 8: Total Propagated Delays for 25R, 75IR Scenarios with  $ L = (70, 80, 90) $ ; pth represents the pth percentile of the TPGD (Total Propagated Delays); N: Nominal, R: Reliable, with percentage change in Reliable relative to Nominal. Improvements where R < N are highlighted.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">pth</td><td colspan="2">25R, 75IR-70</td><td colspan="2">25R, 75IR-80</td><td colspan="2">25R, 75IR-90</td></tr><tr><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td></tr><tr><td style='text-align: center;'>90</td><td style='text-align: center;'>425.80</td><td style='text-align: center;'>679.90 (59.68  $ \uparrow $ )</td><td style='text-align: center;'>594.80</td><td style='text-align: center;'>817.80 (37.49  $ \uparrow $ )</td><td style='text-align: center;'>2031.30</td><td style='text-align: center;'>1330.50 (-34.50  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>91</td><td style='text-align: center;'>440.20</td><td style='text-align: center;'>710.32 (61.36  $ \uparrow $ )</td><td style='text-align: center;'>809.81</td><td style='text-align: center;'>834.99 (3.11  $ \uparrow $ )</td><td style='text-align: center;'>2094.39</td><td style='text-align: center;'>1459.65 (-30.31  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>92</td><td style='text-align: center;'>536.04</td><td style='text-align: center;'>757.20 (41.26  $ \uparrow $ )</td><td style='text-align: center;'>822.24</td><td style='text-align: center;'>849.96 (3.37  $ \uparrow $ )</td><td style='text-align: center;'>2347.72</td><td style='text-align: center;'>1552.12 (-33.89  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>93</td><td style='text-align: center;'>801.56</td><td style='text-align: center;'>795.68 (-0.73  $ \downarrow $ )</td><td style='text-align: center;'>871.91</td><td style='text-align: center;'>908.05 (4.14  $ \uparrow $ )</td><td style='text-align: center;'>2432.42</td><td style='text-align: center;'>1728.52 (-28.94  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>94</td><td style='text-align: center;'>813.32</td><td style='text-align: center;'>818.60 (0.65  $ \uparrow $ )</td><td style='text-align: center;'>885.80</td><td style='text-align: center;'>927.28 (4.68  $ \uparrow $ )</td><td style='text-align: center;'>2583.38</td><td style='text-align: center;'>1928.35 (-31.41  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>95</td><td style='text-align: center;'>881.20</td><td style='text-align: center;'>832.70 (-5.50  $ \downarrow $ )</td><td style='text-align: center;'>919.10</td><td style='text-align: center;'>1011.45 (10.05  $ \uparrow $ )</td><td style='text-align: center;'>3413.10</td><td style='text-align: center;'>1981.84 (-42.00  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>898.76</td><td style='text-align: center;'>924.68 (2.88  $ \uparrow $ )</td><td style='text-align: center;'>1025.44</td><td style='text-align: center;'>1040.80 (1.50  $ \uparrow $ )</td><td style='text-align: center;'>3589.52</td><td style='text-align: center;'>2196.58 (-44.79  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>97</td><td style='text-align: center;'>1235.57</td><td style='text-align: center;'>990.74 (-19.82  $ \downarrow $ )</td><td style='text-align: center;'>1257.88</td><td style='text-align: center;'>1084.96 (6.51  $ \uparrow $ )</td><td style='text-align: center;'>3675.56</td><td style='text-align: center;'>2287.96 (-37.79  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>98</td><td style='text-align: center;'>1469.66</td><td style='text-align: center;'>1117.66 (-24.03  $ \downarrow $ )</td><td style='text-align: center;'>1469.66</td><td style='text-align: center;'>1117.66 (10.06  $ \uparrow $ )</td><td style='text-align: center;'>3729.20</td><td style='text-align: center;'>2678.17 (-28.09  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>99</td><td style='text-align: center;'>2542.62</td><td style='text-align: center;'>1125.16 (-55.75  $ \downarrow $ )</td><td style='text-align: center;'>2543.02</td><td style='text-align: center;'>1208.14 (-52.42  $ \downarrow $ )</td><td style='text-align: center;'>3886.91</td><td style='text-align: center;'>2695.00 (-30.45  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>100</td><td style='text-align: center;'>3693.00</td><td style='text-align: center;'>2032.00 (-45.00  $ \downarrow $ )</td><td style='text-align: center;'>3733.00</td><td style='text-align: center;'>2113.00 (-43.42  $ \downarrow $ )</td><td style='text-align: center;'>3977.00</td><td style='text-align: center;'>2695.00 (-32.24  $ \downarrow $ )</td></tr></table>

<div style="text-align: center;">Table 9: Total Propagated Delays for 100R - 0 and 100IR - 70 scenarios; pth represents the pth percentile of the TPGD (Total Propagated Delays); N: Nominal, R: Reliable, with percentage change in Reliable relative to Nominal. Improvements where R < N are highlighted.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>pth</td><td colspan="2">100R-0</td><td colspan="2">100IR-70</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td></tr><tr><td style='text-align: center;'>90</td><td style='text-align: center;'>237.40</td><td style='text-align: center;'>253.00 ( $ 6.57\% \uparrow $ )</td><td style='text-align: center;'>801.30</td><td style='text-align: center;'>710.80 ( $ -11.30\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>91</td><td style='text-align: center;'>241.00</td><td style='text-align: center;'>266.41 ( $ 10.55\% \uparrow $ )</td><td style='text-align: center;'>804.45</td><td style='text-align: center;'>757.60 ( $ -5.83\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>92</td><td style='text-align: center;'>245.72</td><td style='text-align: center;'>312.28 ( $ 27.09\% \uparrow $ )</td><td style='text-align: center;'>814.76</td><td style='text-align: center;'>795.92 ( $ -2.31\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>93</td><td style='text-align: center;'>300.77</td><td style='text-align: center;'>328.47 ( $ 9.22\% \uparrow $ )</td><td style='text-align: center;'>881.28</td><td style='text-align: center;'>818.70 ( $ -7.10\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>94</td><td style='text-align: center;'>313.76</td><td style='text-align: center;'>351.54 ( $ 12.05\% \uparrow $ )</td><td style='text-align: center;'>885.66</td><td style='text-align: center;'>837.66 ( $ -5.42\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>95</td><td style='text-align: center;'>359.20</td><td style='text-align: center;'>423.35 ( $ 17.86\% \uparrow $ )</td><td style='text-align: center;'>912.65</td><td style='text-align: center;'>991.90 ( $ 8.67\% \uparrow $ )</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>403.36</td><td style='text-align: center;'>734.00 ( $ 82.01\% \uparrow $ )</td><td style='text-align: center;'>1237.76</td><td style='text-align: center;'>1049.76 ( $ -15.18\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>97</td><td style='text-align: center;'>473.35</td><td style='text-align: center;'>739.64 ( $ 56.24\% \uparrow $ )</td><td style='text-align: center;'>1491.35</td><td style='text-align: center;'>1116.39 ( $ -25.12\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>98</td><td style='text-align: center;'>937.52</td><td style='text-align: center;'>922.22 ( $ -1.63\% \downarrow $ )</td><td style='text-align: center;'>2894.20</td><td style='text-align: center;'>1131.94 ( $ -60.88\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>99</td><td style='text-align: center;'>2534.97</td><td style='text-align: center;'>934.86 ( $ -63.12\% \downarrow $ )</td><td style='text-align: center;'>2960.40</td><td style='text-align: center;'>1283.56 ( $ -56.64\% \downarrow $ )</td></tr><tr><td style='text-align: center;'>100</td><td style='text-align: center;'>2928.00</td><td style='text-align: center;'>1119.00 ( $ -61.78\% \downarrow $ )</td><td style='text-align: center;'>3693.00</td><td style='text-align: center;'>2032.00 ( $ -45.00\% \downarrow $ )</td></tr></table>