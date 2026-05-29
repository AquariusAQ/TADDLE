Hmm, the user is asking me to summarize the appendix of an AI paper with very specific requirements. They want me to split it into original subsections, eliminate redundancies, keep each section under 500 tokens, and use concise academic language. The output format must be "### Appendix [Letter] Title + Summary content."

Looking at the appendix content, I see it has several sections: A (More Related Works), B (Interpretation of Likelihood Cancellation), C (Sufficient Statistics), D (Experimental Details), and E (Ablation Studies). I need to handle each one carefully.

For Appendix A, I'll focus on the key distinction between methods with and without training label access. The original text has some repetitive phrasing about "data assumptions" that I can condense. I'll highlight the two main categories and mention representative works without listing all citations.

Appendix B is about VAE likelihood cancellation. The explanation gets technical with the math, but the core insight is that reconstruction errors (decoder) and semantic features (encoder) interact in OOD detection. I'll simplify the explanation to focus on why likelihood cancellation happens and its practical implication.

Appendix C covers sufficient statistics via Fisher-Neyman factorization. The mathematical details are dense, so I'll distill it to the key principle: sufficient statistics allow inference without full data, and the paper uses this to justify their VAE statistics. I'll skip the MLE derivation since it's not central to their method.

Appendix D is the longest with subsections. I'll need to be ruthless about trimming:
- D.1.1: The latent dimension trade-off is crucial but the original has redundant explanations. I'll state the encoder/decoder conflict clearly.
- D.1.2: The two-VAE solution is the key innovation. I'll emphasize pairing high/low dimensions and the statistics they extract.
- D.1.3: The fixed decoder covariance is a minor implementation detail - just note it's simplified.
- D.1.4: The training modifications (MMD, KL variants) showed no significant gains, so I'll mention they were tested but not adopted.
- D.2 and D.3: Feature processing details can be condensed to "quantile transform and lp/lq norms for better copula estimation."

Appendix E has ablation studies. Table 3 shows individual statistics hurt performance, so I'll state that combination is essential. Table 4 proves lower latent dimensions don't help (contrary to hypothesis), which is a key finding.

I'm counting tokens carefully - each summary must stay under 500. For example, Appendix D's summary will be tight by focusing on: (1) VAE architecture choices, (2) the two-VAE strategy as solution to trade-off, (3) key implementation tweaks. The ablation tables can be summarized in 1-2 sentences each.

The user is clearly an academic who needs this for research - probably preparing a paper review or literature survey. They'll appreciate precision over flair. I'll avoid any markdown in the output as requested, and strictly follow the "### Appendix X Title" format.

*checks token counts* 
- Appendix A: ~250 tokens
- Appendix B: ~200 tokens
- Appendix C: ~150 tokens
- Appendix D: ~450 tokens (pushing limits but critical)
- Appendix E: ~200 tokens

All under 500. I'll remove all "we" statements to sound more objective, and cut phrases like "as discussed in Section D.1.3" since the appendix is self-contained.

Noticing one tricky part: In Appendix D.1.4, they tried multiple loss variants but found no difference. The original says "we still include for future work" - I'll omit that as it's not core to their method. Academic summaries should focus on what was adopted, not what was discarded.

For the tables, I won't reproduce numbers but will state the key takeaway (e.g., "Table 4 shows lower dimensions hurt performance"). The user can check the paper for exact values.

*double-checks* 
- Removed all citations (e.g., "Hendrycks & Gimpel, 2016" → "prior work")
- Cut 50% of descriptive phrases in D.1.1
- Simplified the Fisher-Neyman theorem to its practical implication
- Made sure every sentence adds new information

This should meet the user's need for a clean, publication-ready appendix summary. They'll likely use this to quickly grasp the paper's technical depth without wading through appendices.
</think>

### Appendix A More Related Works  
This section categorizes OOD detection methods by data assumptions. Prior work assumes either access to training labels (e.g., classifier-based methods like Hendrycks & Gimpel, 2016) or no label access. Without labels, methods further assume batched OOD data (Nalisnick et al., 2019) or streaming single-point OOD data. The latter often relies on implicit data structure knowledge (e.g., image compressors for structure, Ren et al., 2019). The paper’s method operates without training labels and handles single-point OOD data, avoiding structural assumptions.  

### Appendix B Interpretation of Likelihood Cancellation  
VAE likelihood estimation combines decoder reconstruction ($p_\theta(\mathbf{x}|\mathbf{z})$) and encoder prior ($p(\mathbf{z})$). For OOD data ($\mathbf{x}_{\text{OOD}}$), lower-level features may resemble ID data (high $p_\theta(\mathbf{x}|\mathbf{z})$), but semantics differ (low $p(\mathbf{z})$). However, $p_\theta(\mathbf{x})$ is dominated by lower-level features, masking $p(\mathbf{z})$’s OOD signal. Conversely, reconstruction error ($u(\mathbf{x})$) flags $\mathbf{x}_{\text{OOD}}$ but may be obscured if $p(\mathbf{z})$ is unusually high. Thus, likelihood cancellation prevents direct OOD detection via $p_\theta(\mathbf{x})$.  

### Appendix C Sufficient Statistics and Where to Find Them  
Sufficient statistics $T(\mathbf{x})$ for likelihood $p(\mathbf{x}|\psi)$ satisfy $p(\mathbf{x}|\psi) = f(\mathbf{x})g_\psi(T(\mathbf{x}))$ (Fisher-Neyman theorem). They enable inference without full data, as $\psi_{\text{MLE}}$ depends only on $T(\mathbf{x})$. For Gaussian VAEs, minimal sufficient statistics are $\mu_{\mathbf{z}}(\mathbf{x})$, $\sigma_{\mathbf{z}}(\mathbf{x})$, and $\mu_{\mathbf{x}}(\mathbf{z})$ (decoder mean). The paper leverages these for OOD detection, avoiding decoder variance as it converges to a fixed scalar.  

### Appendix D Experimental Details  
**D.1 VAE Architecture and Training**: Trained VAEs with latent dimensions $\{1, 2, 5, 10, 100, 1000, 2000, 3096, 5000, 10000\}$, selecting the best validation loss checkpoint. For LPath-1M, used all dimensions; for LPath-2M, paired high-dimensional (3096–10000) and low-dimensional (1–5) VAEs. Categorical decoders were tested but performed similarly to Gaussian decoders.  

**D.1.1 Dimensionality Trade-off**: High latent dimensions improve encoder discrimination ($v(\mathbf{x})$, $w(\mathbf{x})$), while low dimensions amplify decoder reconstruction errors ($u(\mathbf{x})$). A single VAE cannot optimize both.  

**D.1.2 Pairing VAEs**: Using two VAEs (high/low dimension) overcomes the trade-off. High-dimensional VAE provides $v(\mathbf{x}) = \|\mu_{\mathbf{z}}(\mathbf{x})\|_2$, $w(\mathbf{x}) = \|\sigma_{\mathbf{z}}(\mathbf{x})\|_2$; low-dimensional VAE provides $u(\mathbf{x}) = \|\mathbf{x} - \hat{\mathbf{x}}\|_2$, $s(\mathbf{x})$. Combined, they achieve state-of-the-art OOD detection (e.g., CIFAR-10 ID vs. CIFAR-100 OOD).  

**D.1.3–D.1.4**: Decoder covariance fixed (no learnable scalar); training loss modified with KL divergence and MMD for latent concentration, but no significant AUROC gains over standard loss.  

**D.2–D.3**: Statistics $u(\mathbf{x})$, $v(\mathbf{x})$, $w(\mathbf{x})$ processed via quantile transform and de-correlation for COPOD. Using $l^p$/$l^q$ norms (small/large $p,q$) instead of $l^2$ slightly improved AUC (Table 2).  

### Appendix E Ablation Studies  
**E.1 Individual Statistics**: Using $u$, $v$, or $w$ alone significantly reduced performance (Table 3), confirming their complementary roles.  

**E.2 MD vs. COPOD**: Replacing COPOD with MD (Lee et al., 2018) yielded comparable scores, confirming $u,v,w$ as the primary performance driver.  

**E.3 Latent Dimensions**: Lower dimensions (e.g., 1, 2) hurt performance ($v(\mathbf{x})$ AUC: 0.39–0.63 vs. 0.76 at 3096; Table 4), refuting the hypothesis that reduced dimensionality enhances OOD discrimination.