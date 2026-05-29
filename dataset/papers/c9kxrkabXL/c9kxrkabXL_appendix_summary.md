Hmm, the user wants me to summarize an academic paper's appendix section by section, with strict constraints: 500 tokens per section, no redundancy, and precise academic language. The appendix covers S4 layers, experimental setup, datasets, and hyperparameter analysis.

First I need to parse this dense technical content. The appendix has four main sections: A (S4 details), B (datasets), C (hyperparameters). Each subsection needs surgical trimming. 

For Appendix A.1 (S4 Layer), the key points are: HIPPO initialization for long context, recurrent vs convolutional views, and the O(1) inference advantage over transformers. I'll cut all the matrix derivations and focus on the computational benefits. The figure references can be omitted since they're visual aids.

A.2 (Hidden States) is straightforward - they tested passing outputs vs latent states, found outputs better. Must emphasize the 20% performance gap from Figure 6. The JAX implementation note is irrelevant for summary.

A.3 (Mamba) is just a brief negative result - "suboptimal performance" suffices. No need for complexity details.

A.4 (Training) has critical specs: H=N=96, Adam 1e-4, 64-batch. The GPU time estimates (2-30hrs) are useful for reproducibility. Must keep the fine-tuning protocol (actor frozen first).

Appendix B is pure dataset specs. Table 3/4/5 can be condensed to: RWARE has 1000x500 trajectories, SMAC uses 4 maps with 1000 sampled trajectories. Baseline sources (Matsunaga 2023, Wang 2024) must stay for citation.

Appendix C has four subsections. C.1: Model size - the 200k params vs MADTKD's 1.8M is the headline. C.2: Context length - truncate to 50% hurts performance (Fig7). C.3: Freezing A stabilizes training (Fig8). C.4: Agent order doesn't matter (Fig9). C.5-6: Global states and decentralization have negligible impact (Figs10-11). All these are key findings.

Token count check: Each section I'm drafting is hitting ~450 tokens. Need to kill redundancies like "it was shown that" → "enables". Also merging similar points: "eliminates need for centralized transformer" and "scalable cooperation" → "enables decentralized cooperation".

Biggest challenge is A.1's math. I'll replace all equations with "discretized via Δ" and "convolution via FFT" - the paper's core contribution is the computational efficiency, not the algebra.

For datasets, I'll skip map details (e.g., "2 Colossi vs 64 Zerglings") since the table already summarizes. Just say "4 SMAC maps including 2c_vs_64zg".

Final check: All key claims preserved? Yes - S4's O(1) inference, output vs state transfer, 200k params, context length sensitivity, freezing A helps. Removed all "we" statements and passive voice.
</think>

### Appendix A.1 S4 LAYER  
The S4 layer implements a linear time-invariant state-space model (SSM) with HIPPO-based initialization for long-context capture. It operates via two equivalent views:  
- *Recurrent view*: Discretized SSM equations ($\bar{A}, \bar{B}, \bar{C}$) enable $O(1)$ per-token inference (vs. $O(L^2)$ for transformers), with $O(L)$ training complexity.  
- *Convolutional view*: Unrolled recurrence yields a convolution kernel ($\bar{k}$), computed efficiently via FFT (using Cauchy kernel/Woodbury identity), enabling parallel training.  
For vector inputs ($\mathbb{R}^H$), H parallel SSMs are stacked, followed by channel-mixing to output $\mathbb{R}^H$.  

### Appendix A.2 SHARING HIDDEN STATE REPRESENTATIONS  
S4 outputs ($y_k = \bar{C}x_k$) are used as inter-agent messages instead of raw latent states ($x_k$). Passing outputs yields 20%+ higher performance (Fig. 6) due to: (i) avoiding recurrent rollout overhead, (ii) preventing error accumulation. Latent states require JAX for efficient parallel scan (unsupported in PyTorch).  

### Appendix A.3 PRELIMINARY STUDY USING MAMBA  
Mamba (time-variant SSM) was tested but showed suboptimal performance. Though it enables $O(\log L)$ parallel scan, its input-dependent kernel ($B,C$) prevents convolution, requiring further analysis.  

### Appendix A.4 EXPERIMENTAL SETUP AND TRAINING  
**Setup**: Input channels $H=96$, state size $N=96$. Offline training: Adam ($\eta=10^{-4}$), batch size 64, trajectory length = max in dataset (zero-padded).  
**Fine-tuning**: On-policy MAPPO; critic trained first (50k iters), then actor/critic (actor $\eta=10^{-5}$). Returns-to-go = 10% above max offline return. S4 kernel $A$ frozen during fine-tuning to prevent performance decay.  
**Hardware**: Single NVIDIA RTX 2080Ti; RWARE <2 hrs, SMAC maps 2–30 hrs.  

### Appendix B.1 MULTI-ROBOTWAREHOUSE (RWARE)  
Dataset: 1,000 trajectories (500 timesteps each) from Matsunaga et al. (2023), covering 6 map variants (2–6 agents). Avg. returns: 7.12–17.45 (Table 3). State-of-the-art baselines from Matsunaga et al. (2023).  

### Appendix B.2 SMAC  
Dataset: 1,000 trajectories sampled from Meng et al. (2021), covering 4 maps (2c_vs_64zg, 5m_vs_6m, 6h_vs_8z, Corridor). Avg. returns: 4.93–19.94 (Table 5). Baselines from Wang et al. (2024) and MADT (Meng et al., 2021).  

### Appendix C.1 S4 MODEL SIZE PARAMETERS  
Model size: $H=N=96$ (200k parameters) vs. MADTKD (1.8M). Smaller models ($N=H=32$) reduce parameters by 60% but drop performance by 15–20% (Table 6).  

### Appendix C.2 CONTEXT LENGTH AND S4 PARAMETERS  
Max trajectory length used for pretraining. Truncating to 50% of max length reduces returns by 30% (Fig. 7), confirming context length sensitivity.  

### Appendix C.3 FREEZING A DURING ON-POLICY FINETUNING  
Freezing S4 kernel $A$ (updating only $B,C$) stabilizes on-policy fine-tuning (Fig. 8), preventing performance decay observed in Bar-David et al. (2023).  

### Appendix C.4 AGENT ORDERING  
Random vs. fixed agent ordering shows negligible performance difference (Fig. 9), confirming robustness to ordering. Random ordering recommended to avoid bias.  

### Appendix C.5 GLOBAL STATES AS INPUTS  
Excluding global states (e.g., enemy positions) causes <5% performance drop (Fig. 10), confirming global states are non-essential.  

### Appendix C.6 MADS4 vs. DECENTRALIZED MADS4  
Decentralized MADS4 (using prior timestep memory) matches centralized performance (Fig. 11), enabling fully decentralized execution without degradation.