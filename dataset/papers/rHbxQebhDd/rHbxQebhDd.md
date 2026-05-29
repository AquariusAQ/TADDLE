

# UNCERTAINTY-AWARE COLUMN GENERATION FOR CREW PAIRING OPTIMIZATION USING SURVIVAL ANALYSIS

Anonymous authors

Paper under double-blind review

## ABSTRACT

The crew pairing problem (CPP) is central to optimal planning and scheduling of operations in the airline industry, where the objective is to assign crews to cover a flight schedule at minimal cost while adhering to various logistical, personnel, and policy constraints. Despite the implementation of optimized schedules, operations are frequently disrupted by unforeseen events. This vulnerability stems from the deterministic nature of the CPP's base formulation, which fails to account for the uncertainties inherent in real-world operations. Existing solutions either aim to safeguard against a specified level of uncertainty or focus on worst-case scenarios. To this end, we propose a reliability-centric CPP formulation amenable to solution by column-generation (CG) SurvCG, that leverages survival analysis for dynamic quantification of uncertainty using the operation patterns in historical data. Applied to CPP, SurvCG forecasts and incorporates flight connection reliability into the optimization process. Through rigorous experiments on a large-scale first-of-its-kind real-world instance under regular and irregular operating conditions, we demonstrate that SurvCG achieves unprecedented improvements (up to 61%) over baseline in terms of total propagated delays, establishing SurvCG as the first data-driven solution for uncertainty-aware reliable scheduling.

## 1 INTRODUCTION

Airline operations planning involves complex decision making on optimal flight scheduling, aircraft assignment, and crew pairing. A pairing is a sequence of flights assigned to a crew under strict rules governed by aviation regulating bodies and union policies. Crew expenses are a major component of airline costs and are highly sensitive to disruptions, since airlines incur additional costs due to any delayed flights, swaps or call-ins (IATA). In this context, the crew pairing problem (CPP) is critical for determining an optimal set of pairings with minimum crew cost under the said constraints. $ ^{1} $ 

The CPP is a highly constrained NP-hard combinatorial optimization problem (Aydemir-Karadag et al., 2013; Lu & Gzara, 2015; Deveci & Demirel, 2018), and is typically modelled as a set-partitioning problem, and, the state-of-art solution methods are based on the column generation (CG) method (Zeren & Özkol, 2016; Quesnel et al., 2020). CPP is often solved by minimizing planned costs based on known schedules and flight times assuming no disruption, months prior to the actual day of operations; also used as the benchmark for evaluation, referred to as the nominal cost ((Erdoğan et al., 2015; Eltoukhy et al., 2017)). However, unforeseen events, such as crew absenteeism and adverse weather, introduce disruption, making the actual costs considerably higher than the planned ones (Antunes et al., 2019). Such disruptions cause pairings to violate operational constraints, such as union regulations, requiring overtime, crew swaps, or additional crews, often leading to deadheading or last-minute crew assignments to maintain coverage, resulting in inflated costs, carbon emissions, and customer dissatisfaction (Huang et al., 2020).

Using historical flight data at the time of planning can help with uncertainty management and crew utilization (Sohoni et al. (2011)). To this end, current approaches introduce uncertainty in CPP by

<div style="text-align: center;"><img src="imgs/img_in_image_box_287_144_918_486.jpg" alt="Image" width="51%" /></div>


<div style="text-align: center;">Figure 1: SurvCG integrates survival analysis with column generation for optimal crew pairing. The reliabilities  $ r_{ij} $  can be pre-computed or can be queried on-the-fly for the optimization for scalability.</div>


modelling flight delay using predefined intervals or historical data (Lu & Gzara (2015), Antunes et al. (2019)). Nevertheless, they struggle to account for real-time disruptions, since limited information available during planning makes it challenging to accurately predict delays, especially on a new flight route or time. Moreover, while minor prediction errors lead to missed connections, historical averages result in an overly conservative plan, exacerbating operational inefficiencies and increasing costs. Schaefer et al. (2005) used simulation to estimate operational crew costs but didn't incorporate this into optimization, emphasizing the need to consider these costs during planning.

Given this, there is a need to inform the CG-based CPP optimization via data-driven predictions, allowing it to trade-off crew utilization and on-time performance for a selected reliability, which virtually leads to planning costs better reflecting operational ones. To this end, we propose SurvCG – a survival analysis-based CG algorithm that utilizes survival analysis predicted flight arrival likelihood between two candidate connections  $ (reliability r_{ij}) $  in the cost function evaluation; see Fig. 1. Here, the refined cost  $ \phi(c_{p}^{k}) $  captures not only the planned scheduled costs but also historical flight connection reliability – to the best of our knowledge, the first work to incorporate data-driven reliability for this task. Note that, while for this exposition we use this specific cost function, SurvCG's modular structure allows for the use of data-driven reliabilities in other linear and nonlinear cost functions. Our evaluations on a real-world on-time performance dataset at different levels of irregular (disrupted) operations reveals that SurvCG leads to significant performance gains in total propagated delays, especially in the challenging higher-percentiles of delays, in some cases reducing the delays by 61% – an unprecedented advancement enabled by our data-driven optimization method. This is because, as opposed to historical averages, SurvCG can handle the long tail of delay distributions. Our overall contributions may be summarized as follows:

1. Data-driven reliability-based optimization formulation: SurvCG combines survival analysis with column generation, integrating reliability into the crew pairing optimization, ensuring schedules are rigorously penalized for low reliability, thus optimizing for both efficiency and robustness. To the best of our knowledge, this is the first approach to explicitly quantify real-world uncertainties using time-to-event models.

2. Introduce P-Index to measure the predictive ability of time-to-event models. Conventional survival metrics (e.g., C-index, Brier score) only consider event ordering, inadequate when exact event timing is crucial. We propose the P-index to assess model precision.

3. Instance generation and rigorous analysis at different levels of disruptions: Using real-world flight data, we generate instances and run extensive simulations to establish the superior properties of SurvCG under various operational conditions. Our public dataset-based instance offers the first benchmark for this task to drive advancements in the area. https://anonymous.4open.science/r/SurvCG-Instance-67C6/

### 1.1 RELATED WORKS

Traditional crew pairing models are deterministic and fail to account for disruptions such as weather, delays, or maintenance issues. To this end, stochastic programming has been applied to introduce

randomness and develop more disruption-resilient solutions. For instance, Yen & Birge (2006) aimed to minimize expected total costs through stochastic programming, though scalability remains an issue. Ionescu & Kliewer (2011) and Dück et al. (2012) extended this work by incorporating crew swaps and combining crew scheduling with aircraft assignment for operational resilience. Schaefer et al. (2005) estimate crew pairing costs via simulation, without reflecting this in the optimization. They propose heuristic-based improvements over nominal solution using penalties on undesirable features.

Robust optimization approaches, which account for uncertainty by modeling worst-case scenarios, have also been explored. These models introduce additional constraints and variables, increasing computational complexity. Antunes et al. (2019) developed a robust crew pairing model that accounts for delay propagation and operational disruptions, while Lu & Gzara (2015) proposed a robust optimization approach using Lagrangian relaxation to handle crew costs under worst-case conditions. However, these methods often rely on historical delays or predefined uncertainty sets. This is limiting because they average are not sufficient to capture the long-tail distribution of delays, and therefore these methods' ability to handle irregular disruptions in the real-world.

Time-to-event modeling is used in a number of domains while survival analysis is a popular choice in clinical studies to analyze disease progression Collett (2015); In & Lee (2018); George et al. (2014), in engineering it is referred to as reliability engineering; we use survival analysis w.l.o.g. since recent developments use this terminology. Survival analysis has been successfully applied in traffic incident modeling Nam & Mannering (2000); Hojati et al. (2014); Li et al. (2020) and predictive maintenance Vianna & Yoneyama (2017); Verhagen & De Boer (2018). A standard approach in survival analysis is to use the Cox proportional hazard (CoxPH) model Cox (1972), which is a semi-parametric model that assumes that 1) the logarithm of risk (hazard) of an event has a linear dependence on their covariates Breslow (1975), and 2) hazard of two data samples remains constant over time, known as the proportional hazard assumption. The linearity and the proportional hazard assumptions are limiting in real-world applications, and works such as Liestbl et al. (1994); Faraggi & Simon (1995); Ishwaran et al. (2008a), and more recently neural network-based models Katzman et al. (2018); Lee et al. (2018); Zhong et al. (2021) have become popular.

In the context of aviation, survival analysis has been used to analyze individual flight delays, such as factors influencing delay recovery in an airline Wong & Tsai (2012), and assess delays in South Korea's air transportation Kim & Bae (2021). To the best of our knowledge, time-to-event models have not been used to predict connection reliability or in any optimization settings.

## 2 SURVIVAL ANALYSIS FOR FLIGHT CONNECTION RELIABILITY

This section develops time-to-event (TTE) terminology for survival analysis in the context of flight events, such as arrivals and departures, with examples. We then introduce flight connection reliability forecasting based on a flight’s likelihood of arriving within the necessary time window.

### 2.1 METHODOLOGY FOR RELIABILITY PREDICTION

The success of a connection depends on whether the flight i lands within a feasible window before flight j's departure, to allow crew to transition to the next flight. Consequently, delays in flight i can disrupt the entire sequence to be completed by a crew, causing reassignments or missed connections. Hence, we will define the reliability based on flight i's timely arrival within the connection window. To this end, we use survival analysis to determine the probability that a flight i can connect to a subsequent flight j departing at scheduled departure time,  $ SDT_{j} $ . The time-to-event (TTE) for flight i is defined as  $ TTE_{i} = AAT_{i} - SDT_{i} $ , where  $ AAT_{i} $  is the actual arrival time, demonstrated as follows.

### Example 2.1

Consider a flight from New York (JFK) to Los Angeles (LAX) with a  $ \left(\mathrm{SDT}_{i}\right) $  of 08:00 AM and an  $ \left(\mathrm{AAT}_{i}\right) $  of 11:30 AM, both in Central Time (CT). TTE for this flight would be  $ TTE_{i}=11:30\ AM-08:00\ AM=3.5 $  hours. If the subsequent flight j from LAX to San Francisco (SFO) is scheduled to depart at 12:00 PM  $ \left(\mathrm{SDT}_{j}\right) $ , we need to determine if the connection between these flights is feasible given the TTE.

We represent each flight, i by a tuple  $ (\mathbf{z}_{i}, y_{i}, D_{i}) $ , where  $ z_{i} \in R^{d} $  has features like origin, destination,  $ SAT_{i} $ , aircraft type/model etc.,  $ y_{i} $  is the TTE, and  $ D_{i} $  is the event indicator. The event  $ D_{i} $  is 1 if the flight lands by time t, otherwise 0. The survival function for a flight with features z at time t is:

 $$ \begin{aligned}S(t\mid\mathbf{z})&=\mathbb{P}(flight landing beyond time t\mid flight^{\prime}s features\mathbf{z})&=\mathbb{P}(T>t\mid\mathbf{Z}=\mathbf{z})\\ &=1-\mathbb{P}(flight landing within time t\mid flight^{\prime}s features\mathbf{z})&=1-\mathbb{P}(T\leq t\mid\mathbf{Z}=\mathbf{z})\\ \end{aligned} $$ 

Here, Z and T are random variables corresponding to features z, and the associated TTE.

To predict the reliability  $ r_{ij} $  of two flights i and j in a sequence, we aim to estimate the probability that flight i will arrive in time for the subsequent flight j to depart. Specifically, we need the probability  $ r_{ij} $  that the flight i to land in time  $ t_{q} = SDT_{j} - SDT_{i} - \delta_{min} $ , where  $ \delta_{min} $  is minimum sit time between flights. Using (1), we estimate  $ r_{ij} $  by querying the estimated survival function  $ \widehat{S}(t \mid \mathbf{z}_{\mathbf{i}}) $  as

 $$ r_{ij}=\mathbb{P}(T\leq t_{q}\mid\mathbf{Z}=\mathbf{z}_{i})=1-\widehat{S}(t_{q}\mid\mathbf{z}_{i}),where t_{q}=\mathrm{SDT}_{j}-\mathrm{SDT}_{i}-\delta_{min}. $$ 

The survival function  $ \widehat{S}(t) $  can be estimated using a non-parametric Kaplan & Meier (1958) estimator from empirical data as follows, where  $ t_{1}, t_{2}, \ldots, t_{L} $  are unique times of flight landing,  $ d_{i} $  denotes the flights that landed at time  $ t_{i} $ ,  $ n_{i} $  be the flights that could possibly land at time  $ t_{i} $  and  $ 1\{\cdot\} $  is the indicator function. However, this cannot be used at time points without event observations.

 $$ \widehat{S}(t)=\prod_{i=1}^{L}\left(1-\frac{d_{i}}{n_{i}}\right)^{1\{t_{i}\leq t\}},\mathrm{w h e r e}d_{i}=\sum_{j=1}^{n}\mathbb{1}\{y_{j}=t_{i}\}D_{j},n_{i}=\sum_{j=1}^{n}\mathbb{1}\{y_{j}\geq t_{i}\} $$ 

Therefore, semi-parametric methods, such as CoxPH Cox (1972) and DeepSurv Katzman et al. (2018) which can capture non-linearity when modeling covariates, are a popular choice to tackle such cases since since these can provide continuous survival estimates that extend beyond the observed event times. CoxTime Kvamme et al. (2019) further extends the CoxPH model by allowing the risk score to vary with time. The hazard function is defined below, where  $ f(\mathbf{z}, t; \theta) $  is a time-dependent neural net.

 $$ h(t\mid\mathbf{z})=h_{0}(t)\exp\left(f(\mathbf{z},t;\theta)\right). $$ 

This model relaxes the proportional hazards assumption by allowing the effect of covariates on hazard to vary over time. We found CoxTime's performance to be competitive and hence use it for this exposition. In general, any time-to-event survival model which preserves the probability interpretation can be used with SurvCG; See (Moore, 2016; Freedman, 2008) for a primer on survival analysis.

### 2.2 P-INDEX: EVALUATING THE PREDICTIVE PERFORMANCE OF SURVIVAL MODELS

The most commonly used evaluation metric for survival models is concordance index or C-index Harrell et al. (1982). It quantifies the rank correlation between the actual time-to-event (TTE) and the model's predictions. However, C-index has been found to be less effective for evaluating models that violate the proportional hazards' assumption Antolini et al. (2005). Hence, we first use the time-dependent C-index,  $ C^{td} $  given by Antolini et al. (2005), as used in Kvamme et al. (2019). For an estimate  $ \hat{S}(t \mid \mathbf{z}) $ ,  $ C^{td} $  estimates the probability that the predicted TTE  $ T_{i} $  for flight i is less than the TTE  $ T_{j} $  for flight j, given that  $ T_{i} $  is less than or equal to  $ T_{j} $  as

 $$ C^{t d}=P(\hat{S}(T_{i}\mid\mathbf{z}_{i})<\hat{S}(T_{j}\mid\mathbf{z}_{j})\mid T_{i}\leq T_{j},D_{i}=1). $$ 

However, since our interest extends beyond the discriminative ability of the model; we focus on the accuracy with which probabilities derived from the predicted survival curve are mapped to specific times. As illustrated by (2), the reliability of a connection  $ r_{ij} $  is obtained by querying a time  $ t_{q} $  against the predicted survival function  $ \widehat{S}(t \mid \mathbf{z}) $ . This mapping is crucial, as errors in the probability estimation for specific times impact the decision to choose a connection over the other. Therefore, to evaluate the accuracy of this mapping, we introduce P-index, defined as:

 $$ P-index=\frac{\sum_{n=1}^{N}\mathbb{1}(\left|\hat{t}_{q}^{n}-t_{q}^{*,n}\right|\leq\varepsilon)}{n}. $$ 

Here, with a discretization N of a predicted survival function for the i-th flight (with certain origin, destination, and  $ TTE_{i} $ ), for each  $ n \in N $ , the indicator is 1 if the predicted query time  $ \hat{t}_{q}^{n} $  is within a

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_173_577_444.jpg" alt="Image" width="29%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_619_174_979_444.jpg" alt="Image" width="29%" /></div>


<div style="text-align: center;">(a) Within:  $ \hat{t}_{q}^{n} $  for flight 1 is within  $ \varepsilon $ </div>


<div style="text-align: center;">(b) Not within:  $ \hat{t}_{q}^{n} $  for flight 2 is outside  $ \varepsilon $ .</div>


Figure 2: P-index: Ratio of predictions within  $ \varepsilon $  to total predictions across query times  $ t_{q} $ . predefined error margin  $ \varepsilon $  of the actual query time  $ t_{q}^{*n} $ , and 0 otherwise. Here,  $ t_{q}^{*n} $  is the estimate of the event time for flight i based on the non-parametric Kaplan-Meier estimator  $ \hat{S}_{KM}(t_{q}^{*}|z_{i}) $ . The  $ \hat{t}_{q}^{n} $  is determined by the time at which the estimated survival probability by the model  $ \hat{S}(t|\mathbf{z}_{i}) $  matches the Kaplan-Meier survival probability  $ p_{KM} = \hat{S}_{KM}(t_{q}^{*n}|\mathbf{z}_{i}) $ . Figure 2 illustrates cases where  $ \hat{t}_{q}^{n} $  is within and not within  $ \varepsilon $ . Additionally, to quantify the deviations we compute the Mean Absolute Error (MAE) for predictions within, above, and below the actual query times:

 $$ \mathbf{MAE}_{l}=\frac{\sum_{i=1}^{n}\left|\hat{t}_{q}^{n}-t_{q}^{*,n}\right|\cdot\mathbb{1}_{\theta}}{\mathbf{card}(\theta)},\quad\theta\in\{\mathrm{Within}(\equiv),\mathrm{Above}(>),\mathrm{Below}(<)\} $$ 

where  $ 1_{\theta} $  is the indicator function for each condition  $ \theta $ , and  $ \text{card}(\theta) $  is the number of instances satisfying the condition  $ \theta $ . Above represents  $ \hat{t}_{q}^{n} > t_{q}^{*,n} $ , and Below represents  $ \hat{t}_{q}^{n} < t_{q}^{*,n} $ .

## 3 SURVIVAL-BASED COLUMN GENERATION (SURVCG)

We model the crew pairing problem for a set of crews K on a flight network  $  G = (N, A)  $ ; where N includes origin and destination nodes O and D representing the crew base at the start and end of the schedule, respectively, and flight nodes  $ N \setminus \{O, D\} $  corresponding to the set of flights F in the schedule. A flight node is defined by the flight's origin and destination airports, and respective departure and arrival times. The set A comprises three types of arcs: arc  $ (O, i) $  if crew k can start its schedule with flight i, arc  $ (i, D) $  if crew k can end its schedule with flight i, and arc  $ (i, j) $  for sequential flights i and j where the destination airport of i is the same as the origin airport of j and the minimum sit time is met. The latter is the minimum time needed for the crew to transition between two consecutive flights in their pairing. The set  $ P^{k} $  represents all possible pairings for crew  $ k \in K $ , where pairing  $ p \in P^{k} $  has an associated cost  $ c_{p}^{k} $ . The binary parameter  $ a_{ip} $  is 1 if flight  $ i \in F $  is covered by pairing,  $ p \in P^{k} $  and 0 otherwise. The binary decision variable  $ x_{p}^{k} $  equals 1 if pairing  $ p \in P^{k} $  is selected for crew,  $ k \in K $  and 0 otherwise. The set-covering formulation of the reliable crew pairing problem RCPP is given by:

 $$ \mathrm{[RCPP]:minimize}\quad\sum_{k\in K}\sum_{p\in P^{k}}\phi(c_{p}^{k}) $$ 

 $$  subject to\quad\sum_{k\in K}\sum_{p\in P^{k}}a_{ip}x_{p}^{k}\geq1\quad\forall i\in\mathcal{F} $$ 

 $$ \sum_{p\in P^{k}}x_{p}^{k}=1\quad\forall k\in K $$ 

 $$ x_{p}^{k}\in\{0,1\}\quad\forall k\in K,\forall p\in P^{k} $$ 

The objective function (8) minimizes the total cost of selecting  $ |K| $  crew pairings using the reliability-integrated cost function  $ \phi(\cdot) $ . Constraints (9) ensure that each flight  $ i \in F $  is covered by at least one

<div style="text-align: center;"><img src="imgs/img_in_image_box_218_171_987_358.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">Figure 3: Network and Constraints: The first panel shows the flight network. The middle one violates Constraint 2 (node 1 uncovered), while the last violates Constraint 3 (Crew 2 without pairing).</div>


pairing. Constraint (10) makes sure that each crew  $ k \in K $  is assigned exactly one pairing from their respective set  $ P^{k} $ . Figure 3 illustrates constraint effects on a 4-flight, 2-crew network.

The reliable cost function  $ \phi $  reflects not only the cost of the pairing but also the reliability of the flight connections within. The cost of a pairing  $ c_{p}^{k} $  for pairing p and crew k, and its reliability-integrated cost,  $ \phi(c_{p}^{k}) $  are defined on the arcs A of the flight network as follows, with the cost of including arc  $ (i,j) $  being  $ c_{ij} $  in pairing  $ p \in P^{k} $ ; the explicit expressions of these costs are detailed next.

 $$ c_{p}^{k}=\sum_{(i,j)\in p}c_{ij},\quad\forall p\in P^{k};\quad\phi(c_{p}^{k})=\sum_{(i,j)\in p}\phi_{ij}(c_{ij}),\quad\forall p\in P^{k}. $$ 

### 3.1 NOMINAL VS RELIABLE COST FUNCTIONS

The cost of arc  $ (i,j) $ ,  $ c_{ij} $ , corresponds to the cost of the crew covering flight j after flight i in their pairing, and is calculated apriori in function of scheduled departure and arrival times. As such, it is referred to as the nominal cost to distinguish from the actual crew cost, which is based on actual departure and arrival times.  $ c_{ij} $  is calculated based on the elapsed time of flight i and the connection time between the two consecutive flights i and j, expressed as:

 $$ c_{ij}=(c_{i}^{e}+\alpha\cdot c_{ij}^{c}),where c_{i}^{e}=SET_{i},c_{ij}^{c}=SDT_{j}-SAT_{i} $$ 

The scaling factor  $ \alpha $  penalises longer connection times, making pairings with shorter layovers between flights more attractive from a cost perspective. Specifically,  $ \alpha $  reflects the operational priorities, such as reducing crew downtime or enforcing extended layovers. The nominal cost function 13 is not equivalent but mimics the pay-and-credit model, which itself does not accurately reflect the complexity of crew pay in practice. Our function is motivated by discussions with an industry partner.

The nominal cost  $ c_{ij} $  assumes perfect operation of the airlines, which is rarely the case. Delays in one flight may cause a cascading effect of delays and disruptions in subsequent flights in the pairing, leading to actual costs significantly different from the nominal. The reliable cost function  $ \phi_{ij}(c_{ij}) $  that we propose makes use of the reliability score of a connection to augment the nominal cost and account for delays and disruptions under a push-back recovery policy that is commonly used in the literature (Schaefer et al. (2005); Antunes et al. (2019); Lu & Gzara (2015)). It is expressed as:

 $$ r c_{i j}=\phi_{i j}(c_{i j})=c_{i j}(\lambda_{1}e^{-\lambda_{2}r_{i j}}+1)-c_{i j}(\lambda_{1}e^{-\lambda_{2}}) $$ 

where  $ \lambda_{1} $  is a parameter that adjusts how significantly the reliability score impacts the reliable cost. A higher value of  $ \lambda_{1} $  increases the sensitivity of the cost adjustment to changes in  $ r_{ij} $ . The parameter  $ \lambda_{2} $  controls the rate of exponential decay, i.e., the rate of increase of the reliable cost as  $ r_{ij} $  decreases. A larger  $ \lambda_{2} $  results in a steeper decay curve, which more aggressively penalizes lower  $ r_{ij} $  values. The term  $ c_{ij}(\lambda_{1}e^{-\lambda_{2}}) $  is the vertical adjustment, it shifts the cost function such that the reliable cost equals the nominal cost under ideal conditions ( $ r_{ij}=1 $ ). The effects of varying  $ \lambda_{1} $  and  $ \lambda_{2} $  on the reliable cost function are illustrated in Figure 4.

The reliable cost function  $ \phi_{ij}(c_{ij}) $  captures the trade-offs between reliability and cost efficiency and is nonlinear in  $ c_{ij} $ . Furthermore, similar incorporation of reliability is possible into a pay-and-credit costing model. Consequently,  $ \phi(c_{p}^{k}) $  may be calculated for a given pairing  $ p \in P^{k} $  and the objective function of [RCPP] remains linear in the decision variable  $ x_{p}^{k} $ . Hence, the linear programming

<div style="text-align: center;">Table 1: Comparison of survival models.  $ \eta $  is the learning rate, and B is the batch size.  $ C^{td} $  is the time-dependent C-Index. CoxTime shows the best performance across all metrics. MAE in minutes.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>$ \eta $</td><td style='text-align: center;'>B</td><td style='text-align: center;'>$ C^{td} $</td><td style='text-align: center;'>P-index</td><td style='text-align: center;'>MAE $ \equiv $</td><td style='text-align: center;'>MAE $ &gt; $</td><td style='text-align: center;'>MAE $ &lt; $</td></tr><tr><td style='text-align: center;'>CoxPH</td><td style='text-align: center;'>0.001</td><td style='text-align: center;'>16</td><td style='text-align: center;'>0.787</td><td style='text-align: center;'>0.494</td><td style='text-align: center;'>8.057</td><td style='text-align: center;'>38.594</td><td style='text-align: center;'>10.261</td></tr><tr><td style='text-align: center;'>DeepSurv</td><td style='text-align: center;'>0.0001</td><td style='text-align: center;'>64</td><td style='text-align: center;'>0.795</td><td style='text-align: center;'>0.532</td><td style='text-align: center;'>8.440</td><td style='text-align: center;'>48.374</td><td style='text-align: center;'>8.090</td></tr><tr><td style='text-align: center;'>CoxTime</td><td style='text-align: center;'>0.0001</td><td style='text-align: center;'>32</td><td style='text-align: center;'>0.807</td><td style='text-align: center;'>0.911</td><td style='text-align: center;'>4.839</td><td style='text-align: center;'>21.227</td><td style='text-align: center;'>4.730</td></tr></table>

relaxation of [RCPP] may be solved by column generation, where the subproblem is a shortest path problem with modified reliable costs on the arcs. Once the relaxation is solved, one has to apply branch-and-price in order to obtain the optimal solution. It is known that solving [RCPP] restricted on the set of generated pairings by CG is usually optimal or very close to optimal. In our experiments, we observed an optimality gap around 0.05%. We note that both [RCPP] and its nominal version CPP defined on the nominal costs are solved using CG. All the instances with varying cost function and parameter settings were solved using CPLEX v22.1.1 using docplex python API on a resource with 4 CPU cores, 25 GB RAM, and all computational times were less than 30mins.

## 4 EXPERIMENTAL RESULTS

This section validates SurvCG by outlining the dataset, hyperparameter tuning, and model performance. It details instance generation, solution comparisons, and simulations, providing quantitative evidence of the approach's robustness and effectiveness.

### 4.1 SURVIVAL MODEL TESTING AND RELIABILITY PREDICTION

Dataset: We train the survival analysis model using the Bureau of Transportation Statistics (2024) (BTS) On-Time Performance dataset for flight operations of Endeavor Air in 2019. There are 97294 total flights, of which 80% are used for training and 20% are used for testing. Specifically, for each flight i in the data, the feature set  $ z_{i} $  consists of spatiotemporal attributes and aircraft information including the day of week, aircraft age/model, origin, destination, and scheduled departure/arrival.

Model Implementation: We implement DeepSurv (Katzman et al., 2018), CoxPH (Cox, 1972), and CoxTime (Kvamme et al., 2019). DeepSurv and CoxPH assume proportionality; CoxTime doesn’t.

Hyperparameter Tuning + Performance Comparison: We perform hyperparameter tuning for each model to find an optimal combination of learning rate lr, and batch size b. For evaluation, we primarily consider the P-index(6) for evaluating model performance and also consider the  $ C^{td} $  (5). The optimal hyperparameter configurations and evaluation results are shown in Table 1.

Reliability Prediction: We use CoxTime (Kvamme et al. (2019)) for reliability prediction due to its superior performance across all quantitative metrics. Notably, CoxTime achieved an P-index of 0.911, which indicates its robust capability to accurately predict survival functions. Further, this precision is crucial for predicting if delayed flights will meet minimum crew connection times  $ \delta_{min} $ .

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_1160_603_1399.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(a) Varying  $ \lambda_{1} $  for  $ \lambda_{2}=5.5 $ </div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_619_1161_1004_1398.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(b) Varying  $ \lambda_{2} $  for  $ \lambda_{1}=30 $ </div>


<div style="text-align: center;">Figure 4: Effects of varying  $ \lambda_{1} $  and  $ \lambda_{2} $  on the Reliability Integrated Cost Function</div>


### 4.2 COMPARISON OF SOLUTIONS

Setup: Given an instance (see Appendix C for detailed instance generation), we run the CG algorithm for the nominal cost function with  $ \alpha = 2 $ , as defined in equation 13, meaning that the sit time is penalized twice compared to the elapsed time. For the reliable solutions, we use two configurations with  $ \lambda_{1} = 10 $ ,  $ \lambda_{2} = 3 $  and  $ \lambda_{1} = 20 $ ,  $ \lambda_{2} = 4 $ , as defined in equation 14.

Cost comparison: We compute the cost of the solutions based on the nominal arc cost (13) with  $ \alpha=1 $  to ensure nominal and reliable solutions are comparable. The constraint (9) allows for a flight node to be covered more than once, resulting in deadheading, where crew members are transported as passengers, incurring additional costs. Reliable solutions show an increase in deadhead flying costs by up to 5.93% compared to the nominal solution. However, these reliable solutions exhibit significant reductions in deadhead connection costs, with decreases up to 13.58%. This results in marginally lower total costs by up to 0.003%, as seen in Table 2. In comparison, Antolini et al. (2005) report a 1-3% increase in planned costs for their robust solutions.

The number of deadheads in the nominal  $ \alpha=2 $  solution is 28, based on the flight frequency count (230 nodes covered once, 2 nodes twice, 9 nodes thrice, and 2 nodes four times). For the reliable solution  $ \alpha=2 $ ,  $ \lambda_{1}=10 $ ,  $ \lambda_{2}=3 $ , the number of deadheads is 24 (232 nodes covered once, 7 nodes thrice, 2 nodes four times, and 2 nodes five times). The configuration  $ \alpha=2 $ ,  $ \lambda_{1}=20 $ ,  $ \lambda_{2}=4 $  also results in 24 deadheads. The deadheading is lower for reliable solutions, as reflected by deadhead costs. Additionally, severity of deadheading into a node is also much lower for reliable solutions.

### 4.3 SIMULATION

Setup: Given a crew pairing solution, either nominal or reliable  $ (\lambda_{1}=20,\lambda_{2}=4) $ , where a pairing  $ p\in P_{opt} $  covers flights  $ F_{p} $ , we simulate by obtaining an actual elapsed time for  $ i\inF_{p} $ . The simulation follows Antunes et al. (2019), where the actual elapsed time  $ AET_{i} $  is given by:

 $$ \mathrm{AET}_{i}=\mathrm{SET}_{i}+\epsilon_{i},\quad\epsilon_{i}\sim\mathrm{Kernel Density Estimation}_{i}^{\mathrm{arrival delay}}\mathrm{on matched flights} $$ 

For details on how matched flights are identified, refer to Appendix D.

Design of Experiments: We simulate across multiple scenarios by varying two controls: the percentage of irregular operations and the severity of these delays/irregular operations. These variables determine from where the  $ \epsilon_{i} $  will be sampled for the matched flights. Each flight is simulated across 100 runs. The percentage of irregular operations (\% IR) indicates the proportion of runs (realizations of the pairings) that experience irregularities, while the level of delay specifies the severity of these irregularities. Delay values for irregular operations (IR) are sampled from a specified percentile of the delay distribution using Kernel Density Estimation (KDE). These parameters govern how the flight delays  $ \epsilon_{i} $  are sampled for each realization. Scenarios are denoted using the format mR, nIR-L, where  $ m\% $  of the runs are regular,  $ n\% $  are irregular, and L represents the percentile beyond which delays are considered. For a full description of the scenarios, refer to Appendix F.

Metrics: We evaluate simulation outcomes using Total Propagated Delay (TPGD), which quantifies delays carried from one flight segment to the next, capturing the cascading effects of delays.

 $$ \mathrm{TPGD}=\sum_{i}\mathrm{pgd}_{i},\quad\mathrm{pgd}_{i}=\begin{cases}\Delta-(\mathrm{SDT}_{i+1}-\mathrm{AAT}_{i}),&\mathrm{if}\mathrm{SDT}_{i+1}-\mathrm{AAT}_{i}<\Delta\\ 0,&\mathrm{otherwise}\end{cases} $$ 

<div style="text-align: center;">Table 2: Comparison of Deadheading and Total Costs for Nominal (N) and Reliable (R) solutions  $ (\alpha = 2) $ . Changes in costs (in parentheses) are percentages compared to the nominal solution. Headers: DFC - Deadhead Flying Cost, DCC - Deadhead Connection Cost, TFC - Total Flying Cost, TCC - Total Connection Cost, TC - Total Cost. R1-  $ (\lambda_{1}, \lambda_{2}) = (10, 3) $ , R2-  $ (\lambda_{1}, \lambda_{2}) = (20, 4) $ .</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Solution</td><td colspan="2">Deadheading Cost</td><td colspan="3">Total Cost</td></tr><tr><td style='text-align: center;'>DFC</td><td style='text-align: center;'>DCC</td><td style='text-align: center;'>TFC</td><td style='text-align: center;'>TCC</td><td style='text-align: center;'>TC</td></tr><tr><td style='text-align: center;'>Nominal</td><td style='text-align: center;'>2948.0</td><td style='text-align: center;'>17436.0</td><td style='text-align: center;'>29233.0</td><td style='text-align: center;'>214933.0</td><td style='text-align: center;'>244166.0</td></tr><tr><td style='text-align: center;'>Reliable 1</td><td style='text-align: center;'>3123.0 (5.93)</td><td style='text-align: center;'>16155.0 (-7.35)</td><td style='text-align: center;'>29408.0 (0.60)</td><td style='text-align: center;'>214750.0 (-0.09)</td><td style='text-align: center;'>244158.0 (-0.003)</td></tr><tr><td style='text-align: center;'>Reliable 2</td><td style='text-align: center;'>3123.0 (5.93)</td><td style='text-align: center;'>15015.0 (-13.58)</td><td style='text-align: center;'>29408.0 (0.60)</td><td style='text-align: center;'>214750.0 (-0.09)</td><td style='text-align: center;'>244158.0 (-0.003)</td></tr></table>

<div style="text-align: center;">Table 3: Total Propagated Delays for 75R,25IR Scenarios with  $ L = (70, 80, 90) $ ; pth represents the pth percentile of the TPGD (Total Propagated Delays); N: Nominal, R: Reliable, with percentage change in Reliable relative to Nominal. Improvements where R < N are highlighted.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">pth</td><td colspan="2">75R,25IR-70</td><td colspan="2">75R,25IR-80</td><td colspan="2">75R,25IR-90</td></tr><tr><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td></tr><tr><td style='text-align: center;'>90</td><td style='text-align: center;'>358.80</td><td style='text-align: center;'>394.40 (9.94%  $ \uparrow $ )</td><td style='text-align: center;'>476.00</td><td style='text-align: center;'>515.70 (8.37%  $ \uparrow $ )</td><td style='text-align: center;'>1080.90</td><td style='text-align: center;'>831.10 (-23.11%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>91</td><td style='text-align: center;'>382.65</td><td style='text-align: center;'>409.07 (6.90%  $ \uparrow $ )</td><td style='text-align: center;'>583.91</td><td style='text-align: center;'>680.31 (16.53%  $ \uparrow $ )</td><td style='text-align: center;'>1119.24</td><td style='text-align: center;'>922.99 (-17.54%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>92</td><td style='text-align: center;'>461.76</td><td style='text-align: center;'>430.40 (-6.79%  $ \downarrow $ )</td><td style='text-align: center;'>858.80</td><td style='text-align: center;'>734.00 (-14.53%  $ \downarrow $ )</td><td style='text-align: center;'>1155.52</td><td style='text-align: center;'>939.24 (-18.75%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>93</td><td style='text-align: center;'>484.52</td><td style='text-align: center;'>437.03 (-9.80%  $ \downarrow $ )</td><td style='text-align: center;'>891.98</td><td style='text-align: center;'>745.62 (-16.39%  $ \downarrow $ )</td><td style='text-align: center;'>1218.96</td><td style='text-align: center;'>1011.49 (-17.02%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>94</td><td style='text-align: center;'>535.16</td><td style='text-align: center;'>480.20 (-10.27%  $ \downarrow $ )</td><td style='text-align: center;'>905.96</td><td style='text-align: center;'>901.20 (-0.53%  $ \downarrow $ )</td><td style='text-align: center;'>1526.22</td><td style='text-align: center;'>1022.86 (-33.00%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>95</td><td style='text-align: center;'>808.60</td><td style='text-align: center;'>734.00 (-9.21%  $ \downarrow $ )</td><td style='text-align: center;'>921.75</td><td style='text-align: center;'>920.65 (-0.12%  $ \downarrow $ )</td><td style='text-align: center;'>1562.25</td><td style='text-align: center;'>1100.00 (-29.60%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>896.36</td><td style='text-align: center;'>741.96 (-17.21%  $ \downarrow $ )</td><td style='text-align: center;'>1011.32</td><td style='text-align: center;'>934.96 (-7.56%  $ \downarrow $ )</td><td style='text-align: center;'>1639.68</td><td style='text-align: center;'>1121.36 (-31.61%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>97</td><td style='text-align: center;'>964.64</td><td style='text-align: center;'>938.58 (-2.71%  $ \downarrow $ )</td><td style='text-align: center;'>2822.09</td><td style='text-align: center;'>986.11 (-65.06%  $ \downarrow $ )</td><td style='text-align: center;'>2930.61</td><td style='text-align: center;'>1179.86 (-59.74%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>98</td><td style='text-align: center;'>2893.70</td><td style='text-align: center;'>1119.20 (-61.32%  $ \downarrow $ )</td><td style='text-align: center;'>2922.12</td><td style='text-align: center;'>1120.12 (-61.65%  $ \downarrow $ )</td><td style='text-align: center;'>3015.38</td><td style='text-align: center;'>1241.38 (-58.84%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>99</td><td style='text-align: center;'>2928.25</td><td style='text-align: center;'>1130.47 (-61.40%  $ \downarrow $ )</td><td style='text-align: center;'>2928.98</td><td style='text-align: center;'>1176.07 (-59.87%  $ \downarrow $ )</td><td style='text-align: center;'>3037.12</td><td style='text-align: center;'>1309.16 (-56.89%  $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>100</td><td style='text-align: center;'>2953.00</td><td style='text-align: center;'>1276.00 (-56.78%  $ \downarrow $ )</td><td style='text-align: center;'>3026.00</td><td style='text-align: center;'>1282.00 (-57.65%  $ \downarrow $ )</td><td style='text-align: center;'>3346.00</td><td style='text-align: center;'>1325.00 (-60.39%  $ \downarrow $ )</td></tr></table>

<div style="text-align: center;">Table 4: Total Propagated Delays for 50R,50IR Scenarios with  $ L = (70, 80, 90) $ ; pth represents the pth percentile of the TPGD (Total Propagated Delays); N: Nominal, R: Reliable, with percentage change in Reliable relative to Nominal. Improvements where R < N are highlighted.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">pth</td><td colspan="2">50R,50IR-70</td><td colspan="2">50R,50IR-80</td><td colspan="2">50R,50IR-90</td></tr><tr><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td><td style='text-align: center;'>N</td><td style='text-align: center;'>R</td></tr><tr><td style='text-align: center;'>90</td><td style='text-align: center;'>402.30</td><td style='text-align: center;'>424.60 (5.55% $ \uparrow $ )</td><td style='text-align: center;'>530.00</td><td style='text-align: center;'>681.40 (28.59% $ \uparrow $ )</td><td style='text-align: center;'>1591.70</td><td style='text-align: center;'>1217.80 (-23.51% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>91</td><td style='text-align: center;'>406.80</td><td style='text-align: center;'>430.45 (5.82% $ \uparrow $ )</td><td style='text-align: center;'>558.26</td><td style='text-align: center;'>745.93 (33.62% $ \uparrow $ )</td><td style='text-align: center;'>1677.56</td><td style='text-align: center;'>1288.81 (-23.20% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>92</td><td style='text-align: center;'>425.64</td><td style='text-align: center;'>437.32 (2.75% $ \uparrow $ )</td><td style='text-align: center;'>595.00</td><td style='text-align: center;'>823.28 (38.37% $ \uparrow $ )</td><td style='text-align: center;'>2030.44</td><td style='text-align: center;'>1298.60 (-36.02% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>93</td><td style='text-align: center;'>436.43</td><td style='text-align: center;'>480.94 (10.19% $ \uparrow $ )</td><td style='text-align: center;'>871.91</td><td style='text-align: center;'>908.82 (4.23% $ \uparrow $ )</td><td style='text-align: center;'>2088.97</td><td style='text-align: center;'>1317.56 (-36.95% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>94</td><td style='text-align: center;'>483.86</td><td style='text-align: center;'>708.88 (46.51% $ \uparrow $ )</td><td style='text-align: center;'>884.42</td><td style='text-align: center;'>935.94 (5.83% $ \uparrow $ )</td><td style='text-align: center;'>2346.04</td><td style='text-align: center;'>1349.06 (-42.48% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>95</td><td style='text-align: center;'>531.40</td><td style='text-align: center;'>757.20 (42.43% $ \uparrow $ )</td><td style='text-align: center;'>891.70</td><td style='text-align: center;'>983.40 (10.28% $ \uparrow $ )</td><td style='text-align: center;'>2454.50</td><td style='text-align: center;'>1727.80 (-29.63% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>881.60</td><td style='text-align: center;'>822.60 (-6.68% $ \downarrow $ )</td><td style='text-align: center;'>905.64</td><td style='text-align: center;'>1011.16 (11.67% $ \uparrow $ )</td><td style='text-align: center;'>3037.84</td><td style='text-align: center;'>1770.44 (-41.69% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>97</td><td style='text-align: center;'>896.27</td><td style='text-align: center;'>934.68 (4.28% $ \uparrow $ )</td><td style='text-align: center;'>923.85</td><td style='text-align: center;'>1043.80 (12.99% $ \uparrow $ )</td><td style='text-align: center;'>3588.64</td><td style='text-align: center;'>1979.63 (-44.82% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>98</td><td style='text-align: center;'>944.76</td><td style='text-align: center;'>994.74 (5.29% $ \uparrow $ )</td><td style='text-align: center;'>1054.12</td><td style='text-align: center;'>1200.66 (13.91% $ \uparrow $ )</td><td style='text-align: center;'>3678.24</td><td style='text-align: center;'>2203.68 (-40.09% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>99</td><td style='text-align: center;'>2901.00</td><td style='text-align: center;'>1283.56 (-55.74% $ \downarrow $ )</td><td style='text-align: center;'>2930.11</td><td style='text-align: center;'>1290.31 (-55.96% $ \downarrow $ )</td><td style='text-align: center;'>3886.91</td><td style='text-align: center;'>2678.17 (-31.08% $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>100</td><td style='text-align: center;'>3693.00</td><td style='text-align: center;'>2032.00 (-45.00% $ \downarrow $ )</td><td style='text-align: center;'>3733.00</td><td style='text-align: center;'>2113.00 (-43.42% $ \downarrow $ )</td><td style='text-align: center;'>3977.00</td><td style='text-align: center;'>2695.00 (-32.24% $ \downarrow $ )</td></tr></table>

where  $ pgd_{i} $  is the propagated delay for each flight i in the pairings,  $ SDT_{i+1} $  is the scheduled departure time of the next flight  $ i+1 $ ,  $ AAT_{i} $  is the actual arrival time of the current flight i, and  $ \Delta $  is the minimum sit time between flights.

Results: We simulate both nominal and reliable solutions separately for 100 seeds each and analyze the results across the scenarios. Reliable crew pairing solutions outperform nominal solutions, particularly in scenarios with higher irregular operations and higher levels of delay. The scenarios with 75% regular operations and 25% irregular operations (75R, 25IR) at various severity levels (70th, 80th, and 90th percentiles) represent situations where delays were greater than the  $ pth $  percentile used to simulate delays. These scenarios are designed to test the robustness of crew pairing solutions under mixed operational conditions, highlighting how well they handle varying degrees of irregularity.

When comparing reliable and nominal solutions for 75R, 25IR scenarios in Table 3 scenarios across different severities of delay, it becomes evident that reliable solutions generally outperform nominal ones, especially at higher percentiles. For the 75R, 25IR-70 scenario, the Total Propagated Delay (TPGD) at the 99th percentile for reliable solutions is 1130.47, while for nominal solutions it is significantly higher at 2928.25. As illustrated in Figure 5, this performance gap between reliable and nominal solutions grows as the severity of delays increases, with the gap widening notably for scenarios involving delays sampled from higher percentiles (P70, P80, P90).

In the 75R, 25IR-80 scenario, the trend continues with reliable solutions showing a TPGD of 1223.89 at the 99th percentile, compared to 2998.11 for nominal solutions. Even as the severity of delays

<div style="text-align: center;"><img src="imgs/img_in_chart_box_234_163_493_410.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_493_163_737_410.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_738_163_983_410.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">Figure 5: Performance Gap Between nominal and Reliable Solutions for 75R,25IR Across Increasing Delay Severity (P70, P80, P90). Here, 'P#': delays of Irregular runs are sampled from #-th percentile.</div>


increases, reliable solutions maintain a lower TPGD, demonstrating their effectiveness in mitigating the impact of severity of delays. In the 75R, 25IR-90 scenario, the TPGD at the 99th percentile for reliable solutions is 1309.16 mins, whereas for nominal solutions it is much higher at 3037.12 mins. Reliable solutions consistently show better performance across these mixed operational scenarios, with a significant reduction in TPGD at higher percentiles compared to nominal solutions, showing an improvement of at least 1000 mins (TPGD) for 98th, 99th and 100th percentiles.

As we transition from less irregular operations 100IR to more irregular operations 100IR, the magnitude of the delays increases significantly. From Table 4, we observe that for 50R, 50IR, scenarios with more severity of delay shows greater performance improvements. For instance, L = 90 percentile, R achieves a minimum reduction of 23% across all the upper percentiles of TPGD. For scenarios 75R, 25IR-L=(70, 80, 90), 100R and 100IR, the detailed performance comparison between the reliable and nominal solutions shows similar trends of improvement and can be further summarized in Appendix G, highlighting robustness of SurvCG in highly irregular scenarios.

## 5 Discussion

We introduce a data-driven approach for the Crew Pairing Problem (CPP) to tackle the uncertainty in real-world planning using an exposition of crew pairing in aviation operations. Our results indicate the tremendous potential to impact operational efficiencies in the real world by leveraging historical on time performance data. We accomplish this by incorporating the reliabilities, predicted using survival analysis – a popular time-to-event model, in the cost function of the CPP task and combining this with column generation algorithm – the state-of-the-art algorithm to solve CPP. SurvCG significantly reduces total propagated delay and deadheading connection costs compared to the nominal solution.

Reliable solutions also show a significant reduction in higher percentiles across various scenarios, demonstrating their robustness under mixed and highly irregular conditions. While recent methods Antunes et al. (2019) report a reduction of 18 – 20% as compared to the nominal solution in terms of total propagated delays, we demonstrate that SurvCG can lead to a reduction of up to approx. 60% over nominal on this metric, that too under the challenging irregular operating conditions. For instance, in the 75R, 25IR-70 scenario in Table 3, reliable solutions save 1797.78 minutes in TPGD at the 99th percentile. Similar trends are observed in the 75R, 25IR-80 and 75R, 25IR-90 scenarios, with savings of 1955.12 minutes and 1724.70 minutes, respectively. Our solution also reduces certain costs and significantly decreases deadheading, resulting in lower operational expenses over the nominal.

SurvCG, is also, to the best of our knowledge, the first algorithm to incorporate data-driven reliabilities for this long-term planning problem. As a result, this investigation also lays the foundations of developing other machine learning for optimization methods. While, a limitation of our approach is that the optimal pairings obtained on solving the CPP using column generation algorithm depend on how accurate the reliability predictions are from the survival model. It is worth noting that while we use a specific survival analysis model – CoxTime, which worked well for this dataset, SurvCG is not constrained to using this model and practitioners can incorporate any appropriate time-to-event model. Future work on this thread can extend the use of reliabilities for other optimizations, or even combine this with recent works on reinforcement learning for column generation Chi et al. (2022). Overall, our work opens new avenues to usher operational efficiencies in the aviation industry and beyond in the backdrop of high competition, climate impacts, and customer retention.

## REFERENCES

Mahyar Alimian, Mohammad Saidi-Mehrabad, and Armin Jabbarzadeh. A robust integrated production and preventive maintenance planning model for multi-state systems with uncertain demand and common cause failures. Journal of Manufacturing Systems, 50:263–277, 2019.

Laura Antolini, Patrizia Boracchi, and Elia Biganzoli. A time-dependent discrimination index for survival data. Statistics in medicine, 24(24):3927–3944, 2005.

David Antunes, Vikrant Vaze, and António Pais Antunes. A robust pairing model for airline crew scheduling. Transportation science, 53(6):1751–1771, 2019.

Ayyuce Aydemir-Karadag, Berna Dengiz, and Ahmet Bolat. Crew pairing optimization based on hybrid approaches. Computers & Industrial Engineering, 65(1):87–96, 2013.

Michael Ball, Cynthia Barnhart, George Nemhauser, and Amedeo Odoni. Air transportation: Irregular operations and control. Handbooks in operations research and management science, 14:1–67, 2007.

N. E. Breslow. Analysis of survival data under the proportional hazards model. International Statistical Review, pp. 45–57, 1975.

Bureau of Transportation Statistics. On-time: Reporting carrier on-time performance (1987-present). https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr, 2024. URL https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoyr_VQ=FGJ&QO_fu146_anzr=b0-gvzr. Accessed: 2024-05-20.

Cheng Chi, Amine Aboussalah, Elias Khalil, Juyoung Wang, and Zoha Sherkat-Masoumi. A deep reinforcement learning framework for column generation. Advances in Neural Information Processing Systems, 35:9633–9644, 2022.

D. Collett. Modelling survival data in medical research. CRC press, 2015.

David R Cox. Regression models and life-tables. Journal of the Royal Statistical Society: Series B (Methodological), 34(2):187–202, 1972.

Muhammet Deveci and Nihan Cetin Demirel. A survey of the literature on airline crew scheduling. Engineering Applications of Artificial Intelligence, 74:54–69, 2018.

Viktor Dück, Lucian Ionescu, Natalia Kliewer, and Leena Suhl. Increasing stability of crew and aircraft schedules. Transportation research part C: emerging technologies, 20(1):47–61, 2012.

Abdelrahman EE Eltoukhy, Felix TS Chan, and Sai Ho Chung. Airline schedule planning: a review and future directions. Industrial Management & Data Systems, 117(6):1201–1243, 2017.

Güneş Erdoğan, Mohamed Haouari, Melda Örmeci Matoglu, and Okan Örsan Özener. Solving a large-scale crew pairing problem. Journal of the Operational Research Society, 66:1742–1754, 2015.

D. Faraggi and R. Simon. A neural network model for survival data. Stat. in medicine, 1995.

David A Freedman. Survival analysis: A primer. The American Statistician, 62(2):110–119, 2008.

Brandon J. George, S. Seals, and I. Aban. Survival analysis and regression models. Journal of Nuclear Cardiology, 21:686–694, 2014. doi: 10.1007/s12350-014-9908-2.

Frank E Harrell, Robert M Califf, David B Pryor, Kerry L Lee, and Robert A Rosati. Evaluating the yield of medical tests. Jama, 247(18):2543–2546, 1982.

Ahmad Tavassoli Hojati, Luis Ferreira, Simon Washington, Phil Charles, and Ameneh Shobeirinejad. Modelling total duration of traffic incidents including incident detection and recovery time. Accident Analysis & Prevention, 71:296–305, 2014.

Fei Huang, Dequn Zhou, Jin-Li Hu, and Qunwei Wang. Integrated airline productivity performance evaluation with  $ CO_{2} $  emissions and flight delays. Journal of Air Transport Management, 84:101770, 2020.

International Air Transport Association (IATA). Unveiling the biggest airline costs, 2024. URL https://www.iata.org/en/publications/newsletters/iata-knowledge-hub/unveiling-the-biggest-airline-costs/.

J. In and Dong Kyu Lee. Survival analysis: Part i — analysis of time-to-event. Korean Journal of Anesthesiology, 71:182 – 191, 2018. doi: 10.4097/kja.d.18.00067.

Lucian Ionescu and Natalia Kliewer. Increasing flexibility of airline crew schedules. Procedia-Social and Behavioral Sciences, 20:1019–1028, 2011.

H. Ishwaran, Udaya B. Kogalur, Eugene H. Blackstone, and Michael S. Lauer. Random survival forests. The Annals of Applied Statistics, 2(3), 2008a. doi: 10.1214/08-AOAS169. URL https://doi.org/10.1214/08-AOAS169.

Hemant Ishwaran, Udaya B Kogalur, Eugene H Blackstone, and Michael S Lauer. Random survival forests. 2008b.

Edward L Kaplan and Paul Meier. Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282):457–481, 1958.

Jared L. Katzman, Uri Shaham, Alexander Cloninger, and et al. Deepsurv: personalized treatment recommender system using a cox proportional hazards deep neural network. BMC Medical Research Methodology, 18:24, 2018. doi: 10.1186/s12874-018-0482-1.

Myeonghyeon Kim and Jiheon Bae. Modeling the flight departure delay using survival analysis in south korea. Journal of Air Transport Management, 91:101996, 2021.

Håvard Kvamme, Ørnulf Borgan, and Ida Scheel. Time-to-event prediction with neural networks and Cox regression. Journal of machine learning research, 20(129):1–30, 2019.

Changhee Lee, William Zame, Jinsung Yoon, and Mihaela Van Der Schaar. Deephit: A deep learning approach to survival analysis with competing risks. In Proceedings of the AAAI conference on artificial intelligence, volume 32, 2018.

Xiaobing Li, Jun Liu, Asad Khattak, and Shashi Nambisan. Sequential prediction for large-scale traffic incident duration: Application and comparison of survival models. Transportation research record, 2674(1):79–93, 2020.

K. Liestbl, P. K. Andersen, and U. Andersen. Survival analysis and neural nets. Statistics in medicine, 13(12):1189–1200, 1994.

Da Lu and Fatma Gzara. The robust crew pairing problem: model and solution methodology. Journal of Global Optimization, 62:29–54, 2015.

Dirk F Moore. Applied survival analysis using R, volume 473. Springer, 2016.

Amirhossein Moosavi and Sadoullah Ebrahimnejad. Scheduling of elective patients considering upstream and downstream units and emergency demand using robust optimization. Computers & Industrial Engineering, 120:216–233, 2018.

Doohee Nam and Fred Mannering. An exploratory hazard-based analysis of highway incident duration. Transportation Research Part A: Policy and Practice, 34(2):85–102, 2000.

Frédéric Quesnel, Guy Desaulniers, and François Soumis. A branch-and-price heuristic for the crew pairing problem with language constraints. European Journal of Operational Research, 283(3):1040–1054, 2020.

Andrew J. Schaefer, Ellis L. Johnson, Anton J. Kleywegt, and George L. Nemhauser. Airline crew scheduling under uncertainty. Transportation Science, 39:340–348, 2005.

Milind Sohoni, Yu-Ching Lee, and Diego Klabjan. Robust airline scheduling under block-time uncertainty. Transportation Science, 45(4):451–464, 2011.

Wim JC Verhagen and Lennaert WM De Boer. Predictive maintenance for aircraft components using proportional hazard models. Journal of Industrial Information Integration, 12:23–30, 2018.

Wlamir Olivares Loesch Vianna and Takashi Yoneyama. Predictive maintenance optimization for aircraft redundant systems subjected to multiple wear profiles. IEEE Systems Journal, 12(2):1170–1181, 2017.

Jinn-Tsai Wong and Shy-Chang Tsai. A survival model for flight delay propagation. Journal of Air Transport Management, 23:5–11, 2012.

Cheng-Lung Wu. Airline operations and delay management: Insights from airline economics, networks and strategic schedule planning. Routledge, 2016.

Joyce W Yen and John R Birge. A stochastic programming approach to the airline crew scheduling problem. Transportation Science, 40(1):3–14, 2006.

Bahadır Zeren and Ibrahim Özkol. A novel column generation strategy for large scale airline crew pairing problems. Expert Systems with Applications, 55:133–144, 2016.

Q. Zhong, J. W. Mueller, and J. Wang. Deep extended hazard models for survival analysis. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://proceedings.neurips.cc/paper/2021/file/7f6caf1f0ba788cd7953d817724c2b6e-Paper.pdf.

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