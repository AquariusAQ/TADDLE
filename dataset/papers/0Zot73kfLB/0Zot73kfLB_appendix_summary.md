Hmm, the user is asking me to summarize the appendix of an AI paper about 3D Gaussian Splatting with motion modeling. They want it split into subsections with specific constraints: each section must focus on core methods, experimental details, data/parameters, or supplementary arguments; eliminate redundancies; keep under 500 tokens per section; use concise academic language; and retain key information.

Looking at the appendix content, it's quite extensive with 17 subsections. I need to carefully categorize them into the requested types. Let me analyze each section:

For core methods:
- A.1: Vanilla 3DGS rendering equation (essential foundation)
- A.2: Auxiliary Deformation Field implementation (key innovation)
- A.3: Translation and Rotation System (core physics module)

For experimental details:
- A.4: Training/evaluation resources (practical specs)
- A.5: Incremental learning experiments (validation method)

For data/parameters:
- A.6: Dataset details (Dynamic Multipart, GoPro)
- A.7, A.8, A.9: Ablation studies (parameter sensitivity)

For supplementary arguments:
- A.10 to A.16: Additional results (quantitative/qualitative)
- A.17: Segmentation results (complementary analysis)

I notice some sections are very repetitive (like multiple ablation tables), so I'll consolidate those under "Ablation Studies" to avoid token waste. The GoPro dataset details in A.6 should be merged with A.14 since they're about the same dataset.

Important to cut all redundant phrases like "we refer readers to" or "as discussed in main context." Must preserve all key metrics (PSNR/SSIM/LPIPS) but present them concisely. For example, in A.5's incremental learning, I'll just state the performance gap without repeating the table structure.

Token count is critical - I'll aim for 450 tokens per section. The rendering equation in A.1 can be summarized in 3 lines instead of the full derivation. For A.2's deformation field, I'll focus on the MLP architecture and output format without listing every embedding degree.

Wait, the user specified "each section focuses on core methods, experimental details..." so I shouldn't merge A.7/A.8/A.9 into one. Better to keep them separate but make each summary tight. A.7 is ablation on time-dependent physics, A.8 on Taylor expansion order, A.9 on interpolation strategies - these are distinct experimental angles.

For the datasets in A.6, I'll list only the key challenges per scene instead of full descriptions. "Hyperbolic Slot: stick rotating through hyperbolic slot" captures the essence.

*checks token count* The original appendix is massive - I'll need to be ruthless. For example, in A.14's GoPro results, I can say "GVFi outperforms DefGS by 5.8 PSNR on extrapolation" instead of listing all numbers.

*verifying academic terms* Must keep "quaternion", "covariance matrix", "Spherical Harmonics" but can drop "differentiable rasterization pipeline" since it's standard in 3DGS papers.

*double-checking constraints* User said "eliminates redundant expressions" - so I'll cut all "as shown in Table X" references and just state the conclusion. "Table 5 shows GVFi maintains stable performance while DefGS degrades" is sufficient.

Okay, structuring now:
- A.1: Rendering equation (core)
- A.2: Deformation field (core)
- A.3: Translation/rotation system (core)
- A.4: Training resources (experimental)
- A.5: Incremental learning (experimental)
- A.6: Dataset details (data)
- A.7-9: Ablation studies (experimental)
- A.10-16: Results (supplementary)
- A.17: Segmentation (supplementary)

But A.10-16 are all result tables - I'll group them as "Additional Results" to save tokens. The user said "supplementary arguments" so results count as supplementary.

*counting tokens in draft* For A.1 summary: 120 tokens. Good. A.2: 140 tokens. A.3: 100 tokens. All under limit.

Final check: Did I miss any key claim? The paper's main contribution is the physics-based motion modeling, so A.2 and A.3 summaries must highlight the deformation field and translation/rotation system clearly. The ablation studies prove their necessity - that's covered in A.7-9.

*noting potential trap* User said "retains key academic information" - so I must include the 3δt finding from A.8 and the 2nd-order Taylor expansion being optimal. Those are crucial.

Okay, time to write. Strictly 500 tokens max per section, no fluff.
</think>

### Appendix A.1 PRELIMINARY FOR VANILLA 3DGS  
3D Gaussian Splatting (Kerbl et al., 2023) models scenes via 3D Gaussians parameterized by position $\mathbf{P}$, orientation (quaternion $\mathbf{r}$), scaling $\mathbf{s}$, opacity $\sigma$, and color $\mathbf{c}$ (SH-based). The covariance $\Sigma = \mathbf{R}\mathbf{S}\mathbf{S}^T\mathbf{R}^T$ enables Gaussian evaluation $G(\mathbf{x}) = \exp(-\frac{1}{2}(\mathbf{x}-\mathbf{P})^T\Sigma^{-1}(\mathbf{x}-\mathbf{P}))$. Rendering uses differentiable rasterization: projected 2D position $\mathbf{P}' = \mathbf{JW}\mathbf{P}$ and covariance $\Sigma' = \mathbf{JW}\Sigma\mathbf{W}^T\mathbf{J}^T$, followed by $\alpha$-blending $\mathbf{C}(\mu) = \sum_i T_i \alpha_i \mathbf{c}_i$, where $T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$ and $\alpha_i = \sigma_i G_i(\mu)$.  

### Appendix A.2 IMPLEMENTATION DETAILS OF AUXILIARY DEFORMATION FIELD  
The deformation field $f_{\text{defo}}(\mathbf{x}_0, t): \mathbb{R}^4 \to \mathbb{R}^{10}$ uses an 8-layer MLP (256 units, ResNet at layer 4) with 8-degree positional embedding for $\mathbf{x}_0$ and 5-degree for $t$. It outputs $\delta\mathbf{x}$ (translation), $\delta\mathbf{r}$ (rotation quaternion), and $\delta\mathbf{z}$ (log-scale difference). Deformed Gaussians at time $t$ are: $\mathbf{x}_t = \mathbf{x}_0 + \delta\mathbf{x}$, $\mathbf{r}_t = \delta\mathbf{r} \circ \mathbf{r}_0$, $\mathbf{s}_t = \mathbf{s}_0 \odot \exp(\delta\mathbf{z})$.  

### Appendix A.3 IMPLEMENTATION DETAILS OF TRANSLATION AND ROTATION SYSTEM  
The physics module $f_{\text{trd}}(\mathbf{x}): \mathbb{R}^3 \to \mathbb{R}^{13}$ is a 6-layer MLP (128 units, ReLU) with 5-degree positional embedding. It predicts for each particle: center $\mathbf{P}_c$, velocity $\mathbf{v}_c$, acceleration $\mathbf{a}_c$, angular velocity $\mathbf{w}_p$ (magnitude $\omega = \|\mathbf{w}_p\|_2$), and axis $\hat{\mathbf{k}} = \mathbf{w}_p / \omega$. Dynamics are derived via Taylor expansion (2nd order).  

### Appendix A.4 TRAINING AND EVALUATION RESOURCES  
Training uses 40k–1.6M Gaussians per scene. Model adds 5% training time vs. DefGS (27 min vs. 25 min on bat), 367.4kB storage, and 15% slower rendering (32 fps vs. 40 fps on bat). All experiments run on a single NVIDIA 3090 24G GPU.  

### Appendix A.5 INCREMENTAL LEARNING FOR SELF-PROPELLED OBJECTS  
Incremental training on Dynamic Object Dataset (t=0→0.75) shows GVFi maintains stable performance (avg. PSNR 40.44) while DefGS degrades (avg. 35.28) due to overfitting. GVFi achieves consistent interpolation/extrapolation (e.g., 39.97 PSNR at t=0.75→0.90 vs. DefGS’s 27.90).  

### Appendix A.6 ADDITIONAL DATASET DETAILS  
**Dynamic Multipart**: 4 objects—Foldingchair (3 rotating parts), Hyperbolic Slot (stick through hyperbolic slot), Satellite (rotating doors), Stove (closing cover). **GoPro**: 4 real scenes—Box (drawer closure), Hammer (contact motion), Collision (two moving objects), Wrist Rest (deformable object).  

### Appendix A.7–A.9 ABLATION STUDIES  
**A.7**: Time-dependent $f_{\text{trd}}$ (input $\mathbf{x},t$) improves extrapolation (PSNR 30.72 vs. 29.44) but not interpolation. **A.8**: 2nd-order Taylor expansion (vs. 1st/3rd) optimizes performance (Dynamic Object: PSNR 38.79, Extrapolation 28.76; Dynamic Multipart: PSNR 34.81, Extrapolation 30.72). **A.9**: Direct $f_{\text{defo}}$ (strategy 1) matches strategy 3 (34.81 PSNR), while autoregressive $f_{\text{trd}}$ (strategy 2) degrades (30.31 PSNR).  

### Appendix A.10–A.16 ADDITIONAL RESULTS  
GVFi outperforms baselines across datasets. **Dynamic Object**: GVFi achieves 39.63 PSNR (interpolation) and 24.35 PSNR (extrapolation) vs. DefGS (38.75/17.06). **Dynamic Multipart**: GVFi reaches 37.91 PSNR (interpolation) and 27.87 PSNR (extrapolation). **GoPro**: GVFi attains 26.28 PSNR (extrapolation) vs. DefGS (21.19). Longer extrapolation (t=0.75→1.5) remains physically plausible.  

### Appendix A.17 ADDITIONAL SEGMENTATION RESULTS  
Object/part segmentation via K-means clustering on learned physics parameters yields high-quality masks (Figures 5–7), with clear separation of dynamic components (e.g., rotating doors in Satellite, moving parts in Foldingchair).