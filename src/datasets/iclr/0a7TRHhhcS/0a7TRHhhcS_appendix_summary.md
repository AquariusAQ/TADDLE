Hmm, the user is asking me to summarize an AI paper's appendix section by splitting it into original subsections. They want concise academic summaries under 500 tokens per section, eliminating redundancies while keeping key information.

Looking at the appendix content, I see it's structured with multiple sections: A (Algorithm Details), B (Theoretical Details), C (More Experiment Results), D (Computing Infrastructure), E (Scalability), F (Resolution Impact), and G (Feature Impact). The user specifically wants me to follow the appendix's own subsection numbering.

I need to be careful about token limits - each section must stay under 500 tokens. The original appendix has some repetitive phrasing (like the spatial-temporal embedding description appearing twice in A.1) that I'll need to condense. For theoretical sections like B, I'll focus on the key mathematical contributions rather than full proofs.

For Section A.1 (Spatial-Temporal Embedding), I'll compress the redundant description into one clear sentence about the embedding methodology. The algorithm in A.2 seems incomplete in the provided text, so I'll note that it's referenced but not detailed here.

Section B has heavy mathematical content. I'll summarize the α-entmax definition and its Lipschitz properties as the core contribution, skipping the lengthy proof details since the user wants conciseness. The Rademacher complexity derivation can be omitted as it's too technical for a summary.

For experimental sections (C-G), I'll highlight key findings: C.1 shows utility functions affect choices, C.2 explains LGCP model interpretation, E demonstrates scalability with training times, F reveals resolution trade-offs, and G shows how crime features influence decision patterns. The tables and figures can be referenced without describing each data point.

I notice the user is likely a researcher or academic who needs this for a paper revision or literature review. They're being very specific about format requirements, so I'll strictly follow their requested output structure with "### Appendix X Title + Summary" format.

The most challenging part will be trimming Section B's theoretical content to under 500 tokens while preserving the α-entmax novelty. I'll focus on the functional definition and its Lipschitz property as the key contribution, since the proofs are supplementary.

For computational details in D, I'll just state the hardware specs without the OS version since it's not academically critical. The scalability results in E can be summarized with the key metric improvements (aRMSE/MAPE decreases) and training times.

I should watch out for token counts - I'll draft each section then trim redundancies. The crime feature analysis in G has very specific racial patterns, but the summary should capture the general insight about how features shape decision patterns rather than listing all table data.

Okay, time to structure this precisely as requested. I'll go section by section matching the appendix numbering, keeping each summary tight and academic.
</think>

### Appendix A Algorithm Details  
**A.1 Spatial-Temporal Embedding**  
The model segments regions into blocks, applying sinusoidal positional encoding for spatial order (latitude/longitude) and temporal sequences. Static features (e.g., crime severity) use one-hot encoding; dynamic features use linear embedding. Spatial, temporal, and feature embeddings are summed element-wise to form the comprehensive input. This integrates positional/contextual information for preference modeling.  

**A.2 Overall Algorithm**  
Algorithm 1 details MoE training: (1) Experts compute gating weights via $\alpha$-entmax; (2) Choice probabilities combine expert weights and utility scores; (3) Parameters optimize via maximum likelihood. Shared embeddings $A$ and $B$ encode spatial/temporal context.  

### Appendix B Theoretical Details  
**B.1 $\alpha$-entmax**  
$\alpha$-entmax generalizes softmax ($\alpha=1$) and Gini entropy ($\alpha=2$). It maximizes $\boldsymbol{p}^\top\boldsymbol{z} + H_\alpha^T(\boldsymbol{p})$ over the simplex $\Delta^M$. The function is 1-Lipschitz in $l_2$-norm (Lemma 1), enabling stable gradient flow.  

**B.2 Proof of Theorem 1**  
Lipschitz continuity of $\alpha$-entmax ensures bounded gradients. The composite function $f_m$ (combining $\alpha$-entmax and softmax) is $L$-Lipschitz ($L = 1/\min(\alpha\text{-entmax})$). Rademacher complexity bounds scale as $O(1/\sqrt{N})$, proving generalization.  

### Appendix C More Experiment Results  
**C.1 Learned Utility Function**  
On NYC Crime data, utility scores $U$ adjust choices based on social characteristics (Fig. 5). Expert-specific patterns align with observed crime distributions, validating utility-driven decision modeling.  

**C.2 LGCP Model Interpretation**  
Non-negative matrix factorization decomposes LGCP intensity matrix $\Lambda \approx WB$. Our model’s experts explain LGCP patterns by learning equivalent mixture components, enhancing interpretability.  

**C.3 Additional Results**  
Figures 6–9 show strong fit on Chicago Crime (Fig. 6–7) and Shanghai Mobike (Fig. 8–9) datasets, with expert mixing coefficients ($\pi^h$) and utility-adjusted patterns ($g^h\exp(U^h)$) capturing spatial-temporal dynamics.  

### Appendix D Computing Infrastructure  
All experiments ran on Ubuntu 20.04.3 LTS with Intel Xeon Gold 6248R CPU (3.00GHz) and 227GB RAM.  

### Appendix E Scalability  
Model performance improves with larger datasets (Fig. 10): NYC Crime (732→5561 samples) reduces aRMSE from 2.34→2.06 and MAPE from 107.94→100.72. Training time remains efficient (NYC: 3.897h; Mobike: 7.106h).  

### Appendix F Impact of Spatial-Temporal Resolutions  
Finer resolutions (e.g., 25×25 blocks vs. 10×10) yield marginal gains (aRMSE: 2.29→2.15 for 2985 samples). Performance heavily depends on sample size (Case-2: 732→5561 samples reduces aRMSE by 0.28), indicating resolution trade-offs favor larger datasets over finer grids.  

### Appendix G Impact of Features on Decision-Making  
Crime severity, race, and gender significantly influence utility scores (Fig. 12–14). Expert-1 reveals racial patterns: Black suspects peak in early/late hours (e.g., 0h–6h, 18h–24h), White Hispanic avoid early hours, and regional crime hotspots vary by race (Tables 3–4). This demonstrates feature-driven human decision modeling.