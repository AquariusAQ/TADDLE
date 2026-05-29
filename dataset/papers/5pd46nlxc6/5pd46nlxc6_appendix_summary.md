Hmm, the user is asking me to summarize an academic paper's appendix section with very specific formatting requirements. They want me to split it into original subsections, keep each summary under 500 tokens, eliminate redundancies, and maintain key academic information. The output format must be "### Appendix [Letter] Title + Summary content."

Looking at the appendix content, I see it covers several technical sections: representational complexity, expressiveness proofs, limitations, hyperparameters, environmental impact, domain descriptions, and missing plots. I need to process each subsection carefully.

For Appendix A, it's about representational complexity of MAC-IGM and MACADV-IGM. The key points are the formal limitation of value factorization in Dec-POMDPs, the proposed AVF framework's architectural preservation, and three mathematical propositions proving expressiveness relationships. I'll condense this to the core findings without the proof details.

Appendix B focuses on AVF-QPLEX-D0's expressiveness. The main contribution is extending QPLEX's universal approximation to macro-actions. I'll highlight the equivalence proof between AVF-QPLEX-D0 and Mac-IGM without the lengthy equations.

Appendix C covers limitations and broader impact. The three limitations (representation constraints, scalability, fixed macro-actions) and the argument about macro-actions enabling real-world MARL scaling are crucial. I'll keep it concise.

Appendix D lists hyperparameters. The table is dense, but I can summarize the key parameters used (learning rate, gamma, batch size etc.) without listing every candidate value.

Appendix E discusses environmental impact. The key numbers are 360 hours computation and 10.39kg CO2 emissions - I'll present these facts cleanly.

Appendix F describes three environments (Box Pushing, Warehouse Delivery, Capture Target). For each, I'll extract the essential task mechanics and observation/action structures without the visual descriptions.

Appendix G shows missing plots. Since it's just referencing figures, I'll note the key result (normalized returns across algorithms) without describing each chart.

I need to be extremely economical with words - each summary must stay under 500 tokens. I'll prioritize: 1) mathematical claims 2) experimental parameters 3) key conclusions. Will skip all figure references and redundant phrases like "as shown in" or "we can conclude."

The user seems to be a researcher or academic who needs this for a paper review or literature survey. They're being very precise about formatting, so I'll strictly follow their requested output structure. No fluff - just distilled technical content.

Let me count tokens as I draft each section. For Appendix A, I'll focus on the core proposition that F^IGM ⊂ F^Mac-IGM and how AVF preserves expressiveness. The proofs can be omitted since they're in the main text.

For Appendix B, the critical point is that AVF-QPLEX-D0 achieves full expressiveness over Mac-IGM via universal approximation. The weight constraints (positive weights) are important to mention.

I notice Appendix D has a table with many hyperparameters - I'll group them by category (training, network, mixer) rather than listing all values. The key is that they used standard values from prior work with minor tuning.

Environmental impact in Appendix E is straightforward: 360 hours → 10.39kg CO2. No need for the calculator details.

For the environment descriptions in Appendix F, I'll use one sentence per domain capturing: 1) core objective 2) key action types 3) observation scope. The box pushing example can be summarized as "Two agents push big box to goal (reward +300) vs small box (+20)."

Finally, Appendix G is just a reference to figures showing performance - I'll state the main finding (AVF variants outperform baselines) without describing each plot.

*checks token counts* Each summary should be around 300-400 tokens to be safe. I'll avoid any markdown and keep academic terms precise ("Dec-POMDP" not "decentralized partially observable MDP").
</think>

### Appendix A REPRESENTATIONAL COMPLEXITY OF MAC-IGM AND MACADV-IGM  
The appendix formalizes limitations of value factorization in Dec-POMDPs and MacDec-POMDPs. Key insight: per-agent value ordering may conflict with joint value ordering when macro-actions replace primitive actions. AVF preserves the representational expressiveness of base factorization methods (VDN, QMIX, QPLEX) by conditioning on state during training or storing history in recurrent layers. Propositions prove: (1) $F^{IGM} \subset F^{Mac-IGM}$ (Mac-IGM subsumes IGM due to larger action space), (2) $F^{Mac-IGM} \equiv F^{MacAdv-IGM}$ (MacAdv-IGM consistency matches Mac-IGM), and (3) $F^{IGM} \equiv F^{Adv-IGM} \subset F^{Mac-IGM} \equiv F^{MacAdv-IGM}$. AVF maintains original methods' expressiveness classes under macro-action settings.  

### Appendix B EXPRESSIVENESS OF AVF-QPLEX-D0  
AVF-QPLEX-D0 achieves full expressiveness over Mac-IGM, extending QPLEX’s universal approximation to macro-actions. The proof demonstrates equivalence between the class of functions representable by AVF-QPLEX-D0 ($\mathcal{Q}^{MIX}$) and Mac-IGM ($Q^{Mac-IGM}$). Key steps: (1) Positive weights in transformation/mixer modules ensure action selection consistency; (2) Constructing transformed utilities $V_i^T, A_i^T$ and joint values $V^{MIX}, A^{MIX}$ satisfies Mac-IGM constraints; (3) Inclusion $\mathcal{Q}^{MIX} \subseteq Q^{Mac-IGM}$ and $Q^{Mac-IGM} \subseteq \mathcal{Q}^{MIX}$ holds under universal function approximation. AVF-QPLEX-D0 thus represents all functions satisfying Mac-IGM.  

### Appendix C LIMITATIONS AND BROADER IMPACT  
**Limitations:** (1) AVF inherits Dec-POMDP representation limitations from base factorization methods; (2) Joint macro-state scalability may degrade with many agents (tested up to 10 agents); (3) MacDec-POMDPs assume fixed macro-actions, unlike primitive MARL. **Broader impact:** Macro-actions enable higher-level decision-making for complex real-world tasks (e.g., object manipulation), improving explainability and scalability over primitive actions. AVF extends MARL to solve larger coordination problems than prior methods.  

### Appendix D HYPER-PARAMETERS  
Hyperparameters were tuned via grid search (Table 4) and finalized per Table 5. Shared parameters: learning rate (5e-4–2.5e-5), $\gamma$ (0.9), ASVB size (2500), batch size (32–64), sampling trajectory size (10–25), Polyak averaging ($\omega$=0.995). Algorithm-specific: AVF-QMIX used mixer embed. size=32; AVF-QPLEX used advantage hypernet embed. size=32. Joint reward scheme (Section 2.2) outperformed alternatives.  

### Appendix E ENVIRONMENTAL IMPACT  
Experiments used a private cluster (carbon efficiency ≈0.275 kgCO₂eq/kWh) for ≈360 cumulative compute hours. Total emissions ≈10.39 kgCO₂eq (calculated via Machine Learning Impact calculator), offset via Freedom. Emphasizes need for sample-efficient MARL to reduce environmental footprint.  

### Appendix F DOMAIN DESCRIPTION  
**F.1 Box Pushing (BP):** Two agents push big box (+300) or small box (+20) to goal. Primitive actions: move/turn/stay; macro-actions: Go-to-Box, Push. Observations: front cell state (empty, teammate, box). Episode ends at goal reach or 100 steps.  
**F.2 Warehouse Tool Delivery (WTD):** Robots deliver tools to humans in 4 work phases. Macro-actions: Go-W(i), Get-Tool, Search-Tool(i), Pass-to-M(i). Rewards: +100 for on-time delivery, -20 for delay, -10 for invalid pass. Four variants (WTD-S to WTD-F) with 1–4 humans.  
**F.3 Capture Target (CT):** 10 agents capture moving target. Primitive actions: move/turn/stay; macro-actions: Move-to-T, Stay. Reward: +1 only if all agents capture target simultaneously (60-step horizon).  

### Appendix G MISSING PLOTS FROM SECTION 5  
Figure 10 shows normalized average return at convergence: AVF-VDN/D0, AVF-QMIX/D0, AVF-QPLEX/D0 outperform baselines (Dec-MADDQN, Cen-MADDQN, Mac-IAICC, HAVEN) across environments. Figures 11–16 detail training curves (AVF variants vs. baselines), confirming AVF-D0’s stability and superiority. AVF-QPLEX-D0 achieves highest returns in all tasks.