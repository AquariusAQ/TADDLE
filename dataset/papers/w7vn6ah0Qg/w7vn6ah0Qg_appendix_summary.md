Hmm, the user is asking me to summarize an AI paper's appendix by splitting it into subsections, with strict constraints: keep each section under 500 tokens, eliminate redundancies, and retain key academic information. The appendix has multiple sections (A to F) covering algorithms, related work, architecture, proofs, decomposition details, and experimental results.

First, I need to analyze each appendix section carefully. For Appendix A (Algorithm), the key is to distill the KokerNet workflow without getting bogged down in repetitive steps. The algorithm has three main components: stationary/non-stationary decomposition, global/local Koopman operators, and distribution constraints. I'll focus on the core logic flow and omit redundant loop descriptions.

Appendix B (Related Work) is a literature review. I should compress the taxonomy of non-stationary forecasting methods (stationarization vs. decomposition) while highlighting the paper's positioning against prior work like RevIN and Koopa. Need to cut the excessive citations and merge similar concepts.

Appendix C (Architecture) has a figure description. Since the figure isn't visible, I'll summarize the key insight: how the model splits time series into stationary/non-stationary parts and applies different Koopman operators with distribution constraints. Must avoid describing the image itself.

Appendix D (Proofs) is dense. I'll extract only the essential theorems: Theorem 1 about spectral approximation (critical for the kernel trick) and Theorem 2 about Koopman eigenfunction approximation. The lengthy derivations can be condensed to their implications.

Appendix E (Decomposition) explains the time series disentanglement process. The core is the Fourier-based decomposition method (FT⁻¹(S_α, FT(X_T))), but I'll skip the seasonal_decompose implementation details since they're not novel.

Appendix F (Experiments) has multiple subsections. For F.1, I'll summarize the key multivariate results (Table 5) by noting KokerNet's competitiveness. F.2 and F.3 contain ablation studies - I'll highlight the main finding that stationary component proportion affects performance, and that decomposition is crucial (Table 8). F.4's efficiency comparison (Figure 7) can be reduced to "competitive training speed vs. baselines."

Now, checking token counts: Each section must stay under 500 tokens. I'll aim for 300-400 to be safe. Need to ruthlessly cut phrases like "we first," "as shown in," and redundant method names. For example, in Appendix D, instead of quoting the full proof, state "Theorem 1 proves M samples suffice for Δ-spectral approximation of kernel matrix K."

Also noting the user's academic context: They're likely a researcher or reviewer who needs this for quick reference, so precision matters more than flowery language. Must retain all key terms (Koopman operators, spectral approximation, non-stationary components) but drop examples like "M4 dataset" unless critical.

Biggest challenge is Appendix D's proofs. I'll focus on what the theorems *mean* for the model's validity rather than the math. For instance: "Theorem 2 shows g_ω approximates Koopman eigenfunctions with T→∞, justifying the operator learning."

Finally, verifying against the constraints: No markdown, strict section headers, and under 500 tokens per part. I'll count tokens as I draft. For Appendix F.3, the Table 8 comparison is vital - I'll summarize the key takeaway (decomposition improves results across datasets) without listing all numbers.

*Double-checking*: Did I miss any critical element? The "distribution constraint" in Appendix A is central to the method, so I'll emphasize that in the algorithm summary. Also confirming that all cited methods (RevIN, Koopa) are properly contextualized in Appendix B.
</think>

### Appendix A Algorithm  
KokerNet decomposes time series $X_T$ into stationary ($X_s$) and non-stationary ($X_{ns}$) components using index $S_v$. For $X_s$, it learns a global Koopman operator $K_s$ with stationary distribution constraints: forecasts $Z_s^{fore} = K_s Z_s^{back}$ via encoder $g_\Theta$, decodes $\hat{X}_s^{fore} = \Phi_{de}(Z_s^{fore})$, and computes loss $L_{dis}^s$. For $X_{ns}$, it learns a local Koopman operator $K_{ns}$ with non-stationary constraints: uses historical distributions to forecast $Z_{fore} = K_{ns} Z_{back}$, decodes $\hat{X}_{ns}^{fore} = \Psi_{de}(Z_{fore})$, and aligns distributions via $L_{dis}^{align}$. A distribution Koopman operator $K_{dis}$ further refines non-stationary forecasts. Total loss $L_{KokerNet} = L_{fore}^s + L_{dis}^s + L_{fore}^{ns} + L_{dis}^{align} + L_{dis}^{rec}$ drives convergence.  

### Appendix B Related Works in Non-Stationary Time Series Forecasting  
Non-stationary time series forecasting methods fall into two categories. *Stationarization methods* (e.g., RevIN, Non-stationary Transformer) transform series into stationary ones via normalization or de-stationary attention. *Decomposition methods* (e.g., KNF, Koopa) split series into time-invariant (global dynamics) and time-variant (local dynamics) components using Koopman operators. KokerNet extends this by jointly learning global ($K_s$) and local ($K_{ns}$) Koopman operators with distribution constraints, addressing both stationary and non-stationary patterns without explicit stationarization.  

### Appendix C Architecture of Koopman Operator Learning and Distribution Constraint  
The model decomposes $X_T$ into stationary ($X_s$) and non-stationary ($X_{ns}$) components via index $S_v$. For $X_s$, a global Koopman operator $K_s$ learns stationary dynamics under distribution constraints. For $X_{ns}$, a local Koopman operator $K_{ns}$ learns time-variant dynamics, with historical distributions processed by $K_{dis}$ to model distribution shifts. The architecture integrates encoder $g_\Theta$, decoders $\Phi_{de}$/$\Psi_{de}$, and distribution alignment to preserve statistical properties during forecasting.  

### Appendix D Theoretical Results  
**Theorem 1**: With $M \geq \frac{2\delta(3\sqrt{n}+2\Delta)}{3\Delta^2} \ln \frac{8\sqrt{n}}{\rho}$ samples, $ZZ^\top$ is a $\Delta$-spectral approximation of kernel matrix $K$, ensuring accurate Koopman operator learning.  
**Theorem 2**: The measurement function $g_\omega$ approximates Koopman eigenfunctions with $\epsilon$-accuracy when $T \geq \sqrt{\frac{2}{M}} \frac{3\omega_{\max}}{\epsilon}$. As $T \to \infty$, $g_\omega$ becomes an exact eigenfunction of the Koopman operator, justifying the model’s theoretical foundation for capturing time-series dynamics.  

### Appendix E Time Series Decomposition  
Time series $X_T$ is decomposed into stationary ($X_s$) and non-stationary ($X_{ns}$) components via Fourier transform. First, global trends/seasonality are removed (using `seasonal_decompose`), yielding residuals $X_T^r$. The residuals are segmented, and $S_v$ (via Kolmogorov-Smirnov test) determines stationary/non-stationary proportions. Frequency spectrum is sorted; top $\alpha\%$ frequencies form $S_\alpha$ (stationary), remaining frequencies form $S \setminus S_\alpha$ (non-stationary). Decomposition: $X_s = \text{FT}^{-1}(S_\alpha, \text{FT}(X_T))$, $X_{ns} = \text{FT}^{-1}(S \setminus S_\alpha, \text{FT}(X_T))$.  

### Appendix F Additional Experimental Results  
**F.1 Multivariate Forecasting**: KokerNet achieves competitive results on ETTh1/ETTm1/ETTm2 (Table 5), outperforming baselines (Ns.Transformer, Autoformer) in MSE/MAE across forecasting horizons $H \in \{48, 96, 144, 192\}$.  
**F.2 Stationary Proportion & Segmentation**: MSE decreases as stationary component proportion increases (Figure 5). KokerNet excels on low-seasonality datasets (e.g., 12.71% sMAPE reduction on Yearly M4 data), confirming higher non-stationarity benefits decomposition.  
**F.3 Decomposition Effectiveness**: Ablation (Table 8) shows decomposition (Predef) outperforms single-component models ($K_s$/$K_{ns}$) across all datasets, proving necessity of disentangling stationary/non-stationary patterns.  
**F.4 Efficiency**: KokerNet achieves comparable training speed/memory to baselines (Figure 7), confirming scalability.