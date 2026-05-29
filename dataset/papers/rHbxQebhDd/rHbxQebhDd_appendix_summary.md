Hmm, the user is asking me to summarize the appendix of an AI paper with very specific requirements. They want me to split it into original subsections, eliminate redundancies, keep each section under 500 tokens, and use concise academic language. The output needs to follow a strict format with "### Appendix X Title + Summary" for each section.

Looking at the appendix content, I see it's quite technical with multiple subsections covering notations, survival models, instance generation, flight matching, sensitivity analysis, scenarios, and performance comparisons. The challenge is to distill all this dense information without losing key academic details.

First, I need to identify all the subsections: A (Abbreviations), B (Survival Models), C (Instance Generation), D (Flight Matching), E (Sensitivity Analysis), F (Scenario Notations), G (Performance Comparison). That's seven sections to handle.

For Appendix A, I notice it's mostly a list of notations. I'll group them by category (flight-related, cost, metrics etc.) and remove repetitive explanations. The key is to keep only the essential symbols and their definitions without the bullet points.

Appendix B is a table of survival models. I'll summarize the key differences in model types, constraints, and benefits in a few lines, focusing on what matters for the paper's contribution.

Appendix C describes instance generation. The text has redundant steps (filtering, network construction, pruning) that I can condense into a clear 3-step process without losing the technical details about constraints and pruning.

Appendix D has that sensitivity table but the main point is about excluding LateAircraftDelay to avoid double-counting. I'll highlight that critical methodological choice.

Appendix E is just a restatement of the sensitivity analysis from Table 6. I'll extract the key finding about α=0.5 being optimal without repeating all percentages.

Appendix F is straightforward - it's just defining simulation scenarios. I'll compress the table into a concise description of how scenarios are structured.

Appendix G has two big tables showing performance comparisons. The key insight is that reliable models reduce propagated delays especially in high-irregularity scenarios (like 25R,75IR-90). I'll focus on the most significant results rather than listing all table rows.

I need to be ruthless about cutting redundancies - for example, in Appendix C, the text repeats the same filtering logic twice. Also, all the percentage changes in Tables 6-9 can be summarized as "reliable models reduce TPGD by up to 63% in extreme scenarios."

Word count is tight - 500 tokens per section means about 300-400 words. I'll aim for precision: "δ=60 mins" instead of "assumed to be 60 mins," "KDE for delay sampling" instead of lengthy explanations.

Most importantly, I must retain all key academic elements: the α parameter, the 25R/75IR scenarios, the 63% TPGD reduction, and the methodological choices like excluding LateAircraftDelay. These are the paper's contributions.

Let me check if I'm missing anything critical... Ah, the P-index metric in A.3 is new - must include that. And for survival models, DeepHit's C-index advantage matters for the paper's methodology.

Okay, structuring now: each appendix section will have exactly one sentence for the title, then a tight 3-4 sentence summary hitting all key points. No fluff, no examples, just what's essential for replication and understanding the paper's technical foundation.
</think>

### Appendix A Abbreviations and Notations  
Core notations: Flight-related (F, origin, dest, SET, SAT, SDT, ADT, AET, AAT); Cost (c, rc); Metrics (P-index, $C^{td}$); Time/Survival ($t_q$, f, S, $p_{=}$, MAE$_{>}$/$_{=}$/$_{<}$); Dataset/Network ($D$, $C$, $N$, $A$, $c$, $\hat{F}$, $\delta=60$ mins); Pairings ($P$, $P_{opt}$, $p$, $R$); KDE/Reliability ($\text{KDE}(\text{matched flights})$, $r$, $\phi$).  

### Appendix B Summary of Survival Models  
Compares 7 methods: Cox (interpretable, proportional constraints), DeepSurv (non-linear, Cox-based NN), Cox-Time (non-proportional), Cox-CC (proportional Cox-Time), RSF (handles interactions), DeepHit (best C-index, discrete). Key trade-off: Cox variants support constraints but lack flexibility; DeepHit excels in discriminative power but lacks constraint support.  

### Appendix C Instance Generation  
Constructs crew pairing instances via: (1) Filtering dataset $D$ using spatiotemporal constraints ($C$) to get $F_0$; (2) Building network $N_0$ with feasible connections ($\text{origin}_j = \text{dest}_i$, $\delta_{\text{min}} \leq \text{SDT}_j - \text{SAT}_i \leq \delta_{\text{max}}$); (3) Pruning $N_0$ to retain only flights forming valid paths from crew base (JFK). Uses Endeavor Air data (Dec 2–5, 2019).  

### Appendix D Flight Matching Criteria and Late Aircraft Delay  
Matched flights require identical origin, destination, and time-of-day segments (e.g., morning: 06:00–11:59 AM). LateAircraftDelay excluded from initial delay estimation to prevent double-counting; simulated separately as $\text{Actual} - \text{Scheduled} - \text{LateAircraftDelay}$ (per Antunes et al., 2019).  

### Appendix E Sensitivity Analysis of $\alpha$  
$\alpha=0.5$ minimizes deadhead costs (DFC: -84.3%, DCC: -85.5%) with negligible total cost impact (+0.65%). Higher $\alpha$ increases deadheads (2→28) and costs (e.g., $\alpha=3$: DFC +87.0%, DCC +29.4%), with diminishing returns beyond $\alpha=3$. Optimal $\alpha=0.5$ balances cost and feasibility.  

### Appendix F Detailed Scenario Notations  
Scenarios defined by irregularity percentage (R: regular, IR: irregular) and delay percentile ($>70$, $>80$, $>90$). Examples: "75R,25IR 70" = 75% regular operations, 25% delays from $>70^{\text{th}}$ percentile; "100IR 70" = 100% delays from $>70^{\text{th}}$ percentile. Delay values sampled via KDE.  

### Appendix G Detailed Performance Comparison  
Reliable (R) models reduce Total Propagated Delays (TPGD) significantly in high-irregularity scenarios (e.g., 25R,75IR-90: TPGD ↓34.5% at 90th percentile). In extreme cases (e.g., 100IR-70), R cuts TPGD by up to 63.12% (99th percentile). Nominal (N) models underperform under disruption, especially at high delay percentiles ($>90$).