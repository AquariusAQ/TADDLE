Hmm, the user is asking me to summarize the appendix of an AI paper titled "Efficient Time Series Forecasting via Hyper-Complex Models and Frequency Aggregation." They want me to split it into subsections, keep each section under 500 tokens, eliminate redundancies, and use concise academic language.

Looking at the appendix content, I see it's quite detailed with multiple sections: Notations & Symbols (A), Additional Experimental Details (B), Additional Information on HC Numbers (C), and Additional Ablation Studies (D). Each has several subsections.

For Section A (Notations & Symbols), I need to condense the symbol tables. The key is to capture all critical symbols without listing every single one. I'll focus on the most important ones like batch size (B), lookback window (L), embedding size (E), and the hyper-complex related terms (M, N_FFT, etc.). The dimension table can be summarized by highlighting the key tensor shapes.

Section B covers datasets, baselines, implementation details, metrics, and normalization. I should extract the essential dataset characteristics (features, timesteps, frequency) and list the key baselines without repeating their descriptions. The hyperparameter table can be simplified to show the most relevant values per dataset.

Section C is about hyper-complex number systems (complex, quaternions, sedenions). This is quite technical, so I'll need to distill the core mathematical properties and how they're applied in the model (HC-MLP variants). I'll skip the detailed multiplication formulas but keep the key differences between bases.

Section D has multiple ablation studies. I'll group these by theme: parameter sensitivity (embedding size, windows, FFT resolution), lookback window effects, real/imaginary component analysis, HC model comparisons, neighborhood aggregation, complexity analysis, and visualizations. For each, I'll extract the main findings without listing all table values.

I need to be extremely concise - under 500 tokens per section. That means cutting all redundant phrases like "we provide" or "as shown in the table." I'll use precise academic terms and focus on what's novel or critical for understanding the method.

The user is clearly an academic researcher who needs this for quick reference, so I'll prioritize technical accuracy over readability. No fluff. For example, instead of "The model uses min-max normalization," I'll say "Min-max normalization (0-1 range) applied to inputs."

I notice some sections have image references (Figures 7-13) which I can omit since they're not part of the textual content. The key is the data they represent.

For the HC numbers section (C), I'll emphasize that the paper extends beyond complex numbers to quaternions and sedenions, noting that norm preservation holds for bases 2/4 but not 16. That's the critical academic point.

For the ablation studies (D), I'll highlight the most significant findings: optimal embedding sizes around 64-128, window count p=13-33 being effective, and that HC-MLP with p=4 (octonions) often outperforms others.

I'll be careful to maintain the exact technical terms from the paper (like "STFT windows" not "frequency windows") and keep all mathematical notations correct (e.g., ℂ for complex numbers).

Finally, I'll strictly follow the requested output format: "### Appendix [Letter] [Title] + Summary" for each subsection. No extra commentary - just the distilled content.
</think>

### Appendix A Notations & Symbols  
Core symbols: Batch size (B), lookback window (L), features (D), prediction horizon (T), embedding size (E), top-M frequencies (M), STFT windows (P), FFT bins (N_FFT). Key tensors: Input X ∈ ℝ^(B×L×D), embedded X_E ∈ ℝ^(B×L×D×E), STFT windows C_i ∈ ℂ^(B×N_FFT×D×E), frequency-aggregated C_i^in/out ∈ ℂ^(B×M×D×E), reconstructed X_rec ∈ ℝ^(B×L×D×E). Dimensions follow tensor flow: X_E expands features via embedding, C_i^in retains top-M magnitudes, X_rec reconstructs time-domain output. Hyper-complex weights W_{i→j}, B_{i→j} ∈ ℂ^(E×E)/ℂ^E capture frequency energy shifts.  

### Appendix B Additional Experimental Details  
Datasets: 13 real-world TSF benchmarks (e.g., Weather: 21 features, 52,696 timesteps, 10m freq; Traffic: 862 features, 17,544 timesteps, 1h freq; Electricity: 321 features, 26,304 timesteps, 15m freq; ETT: 7 features, 17,420/69,680 timesteps, 1h/15m freq). Baselines: Informer, Reformer, Autoformer, FEDformer, LTSF-Linear, PatchTST, FreTS (used as foundation). Hyperparameters: Batch size (4–16), embedding size (32–128), hidden size (256), N_FFT (6–48), STFT windows (3–33), M (4–M_max), epochs (10). Metrics: MSE (training), MAE/RMSE (evaluation). Normalization: Min-max (0–1) applied to inputs.  

### Appendix C Additional Information on HC Numbers and Models  
Hyper-complex (HC) numbers generalize complex numbers (base 2) to higher dimensions:  
- **Base 2 (Complex)**: Multiplication preserves norm; HC-MLP reduces to FreTS.  
- **Base 4 (Quaternions)**: Multiplication preserves norm; HC-MLP (QuatMLP) uses 2 complex components.  
- **Base 16 (Sedenions)**: Multiplication *does not* preserve norm; HC-MLP (SedMLP) uses 8 complex components.  
HC-MLP variants (QuatMLP, OctMLP, SedMLP) operate on STFT windows with base-specific multiplication rules. Norm preservation holds for bases 2/4 but fails for 16, impacting stability.  

### Appendix D Additional Ablation Studies  
**Parameter Sensitivity**: Optimal embedding size (E=64–128) reduces loss; excessive E causes overfitting. STFT windows (p=13–33) and FFT resolution (N_FFT=16–32) significantly impact performance.  
**Lookback Window**: Longer lookback (L=96–192) improves accuracy for T=96/192 but degrades for T=720.  
**Real vs. Imaginary**: Using both components (normal) outperforms isolating real/imaginary parts (MAE/RMSE similar across variants).  
**HC-MLP Comparison**: OctMLP (p=4) and SedMLP (p=8) consistently outperform QuatMLP (p=2) and BasicMLP (non-HC) across datasets (e.g., ETTm1: OctMLP MAE=0.0496 vs. BasicMLP=0.0546 at T=336).  
**Neighborhood Aggregation**: 1-neighbor WM-MLP (optimal) outperforms 2/3-neighbor variants (e.g., ETTh1 T=96: RMSE=0.084 vs. 0.095).  
**Complexity**: WM-MLP/HC-MLP achieve O(L log(L/p)) training time (vs. O(L²) for Transformer), with HC-MLP reducing memory by 2/3.