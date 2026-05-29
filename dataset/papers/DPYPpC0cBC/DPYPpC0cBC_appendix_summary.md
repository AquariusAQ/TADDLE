Hmm, the user is asking me to summarize the appendix of an AI paper about a logistics routing problem called LRP (Location-Routing Problem). They want me to split it into subsections, keep each under 500 tokens, eliminate redundancies, and use academic language.

Looking at the appendix content, it's quite detailed with four main sections: A.1 to A.4 about methodology, and B.1 to B.4 about experiments. I need to be very concise while preserving key technical details.

For A.1 (LRP Configuration), I'll focus on the three constraints - customer demand, vehicle capacity, and depot supply - noting which are hard vs soft constraints. The assumptions about single depot serving and no order splitting are crucial too.

A.2 (MDP Formulation) is dense with math. I'll distill the state/action definitions and highlight the key equation for capacity update. The γ=1 discount factor is important to mention since it's unusual.

A.3 (Multi-depot Mask Mechanism) has that complex Algorithm 1. I'll summarize the four scenarios for selection domains instead of listing all code steps. The batch-wise mask operation is the core innovation here.

A.4 (Training Details) has two algorithms. I'll emphasize the pre-training with baseline comparison and the dual-mode training distinction between Gaussian and exact position modes.

For the experimental section:
B.1 (Hyperparameters) needs the key numbers - 100 epochs, batch sizes, GPU specs. The heuristic method parameters can be condensed to just the most critical values.

B.2 (Depot Visualization) shows an important insight about distribution complexity increasing with scale. I'll capture that trend without the figure details.

B.3 (Route Balancing) has that key equation for the balance penalty. The table comparison shows clear effectiveness, so I'll mention the penalty reduction (e.g., 0.758→0.008) as evidence.

B.4 (Discussion) is more narrative. I'll focus on the framework's novelty (proactive depot generation vs selection) and the two key limitations/future directions about adaptive depot count and constraint expansion.

I need to watch token counts carefully. Each subsection must stay under 500 tokens. I'll prioritize mathematical formulations and empirical results over procedural details. The phrase "eliminates redundant expressions" means I can cut phrases like "as depicted in Fig. 3" since the summary stands alone.

The user is clearly an academic who needs this for a paper or review, so precision matters more than readability. I'll use terms like "hard constraints" and "soft constraint" without explanation since the audience will know.

For the output format, I'll strictly follow "### Appendix X Title + Summary" as requested. No markdown, just clean academic text. I'll count tokens as I go to stay under limit.

*checks token count* The draft I'm imagining for A.2 is about 450 tokens - good. For B.3, the table data can be summarized in one sentence about penalty reduction. The "5th column" reference in the original is redundant for a summary.

Biggest challenge is A.3's algorithm. I'll describe the four scenarios in one sentence each rather than listing them. The mask matrix operation is the key point, not the code steps.

*double-checks* Yes, the user said "retains key academic information" so I must keep the 0.008 vs 0.758 comparison in B.3 - that's the evidence for the method's effectiveness.

Okay, time to distill this systematically. I'll start with A.1 and move down, keeping each section tight.
</think>

### Appendix A.1 LRP Configuration  
The LRP adheres to three core assumptions: (1) Each customer is served by exactly one depot with no intermediate load transfers; (2) Orders cannot be split across vehicles; (3) Vehicle count is unconstrained but minimized in the objective. Constraints include: (i) Hard: Vehicle capacity must cover next customer demand; (ii) Hard: Cumulative route demand ≤ vehicle capacity; (iii) Soft: Total depot dispatch ≤ maximum supply (penalized in objective). The first two ensure feasibility; the third is a cost penalty.  

### Appendix A.2 MDP Formulation  
The LRP solution is modeled as an MDP tuple $(\mathbf{S}, \mathbf{A}, \mathbf{P}, \mathbf{R}, \gamma)$ with $\gamma=1$. States $\mathbf{S}$ are $(G, D_t, \mathbf{v}_t, Q_t)$, where $G$ is graph data, $D_t$ is current depot, $\mathbf{v}_t$ is current customer, $Q_t$ is remaining capacity. Actions $\mathbf{A}$ select next feasible point (unserved customer or depot), updating $Q_{t+1}$ via:  
$$Q_{t+1} = \begin{cases} Q_t - q_e & \text{if visiting unserved customer } e, \\ Q & \text{if returning to depot}. \end{cases}$$  
Rewards $\mathbf{R}$ accumulate travel distance $d_{ij}$ (except depot-to-depot moves). The MDP encodes feasible routes as permutations, later reverted to vehicle routes.  

### Appendix A.3 Multi-depot Mask Mechanism  
A batch-wise mask dynamically updates feasible actions per instance based on four scenarios:  
(i) At depot, all tasks done → stay at depot;  
(ii) At depot, tasks pending → choose unserved customers/unplanned depots (excl. current depot);  
(iii) At customer, all tasks done → return to depot;  
(iv) At customer, tasks pending → choose unserved customers/departure depot.  
The mask matrix efficiently enforces these per batch, avoiding per-instance operations. Key steps: (1) Mask served customers/unsatisfiable demands; (2) Initially mask all depots, reveal departure depot; (3) For scenario (ii), mask departure depot and reveal unplanned depots.  

### Appendix A.4 MDLRAM Pre-training & DGM Dual-mode Training  
**MDLRAM pre-training (100 epochs, batch size 512)**: Uses probabilistic sampling for actions. A fixed baseline network stores best parameters; MDLRAM updates parameters only if superior (verified via paired t-test).  
**DGM dual-mode training (100 epochs)**:  
- *Multivariate Gaussian mode*: Main-batch size $B_{\text{main}}=32$ (scale 20), sub-batch $B_{\text{sub}}=128$. DGM samples depot sets; MDLRAM acts as critic.  
- *Exact Position mode*: $B_{\text{main}}=512$; no sampling. DGM directly outputs depot coordinates.  
Both modes use 100 epochs, evaluate every 100 iterations on 20 batches.  

### Appendix B.1 Hyperparameters  
**MDLRAM**: 100 epochs, 2500 batches (batch size 512; 256 for scale 100). **DGM**: 100 epochs, 2500 main-batches. Gaussian mode: $B_{\text{main}}=32$ (16 for scale 100), $B_{\text{sub}}=128/64/32$ (scales 20/50/100). Exact mode: $B_{\text{main}}=512$ (256 for scale 100). **Architecture**: 3 attention layers, 8 heads, embedding size 128. All trained on single A40 GPU.  
*Heuristics*: ALNS (destroy: 10–40% random, 5–10 worst nodes; repair: random/greedy/regret; rewards: $r_1=30$, $r_2=20$, $r_3=10$, $r_4=-10$); GA (pop. 100, mut. prob. 0.2, crossover 0.6); TS (tabu step 30).  

### Appendix B.2 Depot Distribution Visualization  
DGM’s Gaussian mode generates multivariate depot distributions. For scale $m=3,n=20$, the 6D distribution splits into three 2D Gaussians (Fig. 4). As scale increases ($m=6,n=50$; $m=9,n=100$), distributions become complex, not decomposable into discrete clusters. This implies large-scale depot correlations require extensive sampling for optimal coverage, increasing computational cost.  

### Appendix B.3 MDLRAM Route Balancing  
MDLRAM fine-tuned with balance penalty $\tilde{L}_{\text{Sel}} = L_{\text{Sel}} + \sum_{k,k'} |l_k - \frac{\rho_k}{\rho_{k'}} l_{k'}|$ achieves proportional route lengths ($\rho_k=1$). On all scales, balance penalty drops significantly (e.g., scale 20: 0.758 → 0.008), with minimal total length increase (e.g., scale 20 case 1: 3,487 → 2,781). Table 4 confirms balanced depot routes under the new objective.  

### Appendix B.4 Further Discussion  
The framework shifts from depot *selection* to *generation* via DGM (Gaussian/exact modes), enabling proactive depot placement. MDLRAM then plans routes from generated depots, minimizing total cost. Key strengths: cost reduction in real-world logistics (e.g., disaster relief), modular design for LRP variants (e.g., route balancing in B.3).  
**Limitation**: DGM generates fixed depot count; adaptive count based on demand is needed.  
**Future work**: Incorporate depot constraints (e.g., minimum distance, forbidden zones), and extend to multi-task routing for dynamic logistics.