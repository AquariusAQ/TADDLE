Okay, the user has shared a lengthy appendix section from an AI paper and wants me to summarize it by splitting into original subsections. They need concise academic summaries under 500 tokens per section, eliminating redundancies while keeping key information.

Hmm, this is quite a technical appendix covering kernel methods, MMD, characteristic kernels, and experimental details for PPM-Reg. The user is clearly an academic researcher who needs precise, distilled information - probably for literature review or paper writing. They're being very specific about formatting requirements, so I must adhere strictly to their instructions.

Let me break this down section by section. First, Appendix A is about kernel basics and MMD. I'll focus on the key definitions: feature maps, RKHS, universality/characteristicness, and how MMD relates to kernel mean embeddings. Need to cut the lengthy proofs but keep the core mathematical relationships.

Appendix B proves characteristic kernels on Ω. The main points are Lemma 1 about RKHS inclusion, Theorem 4 on universality transfer, and Theorem 5 showing measure containment. The duality arguments are crucial but I'll skip the proof details.

Appendix C covers weak topology metrization. The key is Theorem 6 linking MMD to weak convergence, and Corollary 2 showing equivalence with Wasserstein metric. Must emphasize the compact Polish space condition.

Appendix D proves smoothness of PPM-Reg. Theorem 7 is central - showing Fréchet differentiability under C¹ assumptions. I'll skip the Bochner integral details but keep the derivative expression.

Appendix E has two parts: shape matching (E.1) and imperfect convergence (E.2). For E.1, I'll summarize the experimental setup (512-point clouds, 4 loss variants) and key result (PPM-Reg preserves topology). For E.2, the penalty term and PD distance comparison are essential.

Appendices F and G cover image generation and SSL experiments. Tables 4-11 contain critical metrics - I'll extract the best-performing λ/σ combinations and highlight the 12.52% accuracy jump in SVHN. Must note the dataset sizes (CelebA 202k vs LSUN 2.2M) as context for the topological complexity argument.

Appendix H is just network diagrams - no summary needed since it's purely visual.

*checks token counts* Each section must stay under 500 tokens. I'll be ruthless with redundancies: e.g., "we refer the reader to" can go, and all lemma proofs can be condensed to "proves X" statements. The mathematical notation must stay precise though - no dumbing down.

*double-checks* User said "retains key academic information" so I'll keep all theorem numbers and critical parameters (like λ=1, σ=0.1 in experiments). The "why it matters" for each section should be implied through the summary focus.

*mental note* Avoid phrases like "as shown in" - just state the result. And absolutely no new information beyond the appendix. This is pure distillation.

*starts drafting* For Appendix A: 1) Feature map → kernel definition 2) RKHS via Moore-Aronszajn 3) Characteristic kernels enable MMD = ∥Φ(μ)-Φ(ν)∥. Cut all examples.

*checks length* Current draft for A is 198 tokens - good. B is 212... wait, need to merge the universality/characteristicness parts. Theorem 4 and Corollary 1 are connected - I'll say "Theorem 4 establishes universality; Corollary 1 extends to characteristicness via duality."

*validates against paper* Yes, the appendix does use duality for characteristicness. Must include "duality theorem of Simon-Gabriel & Schölkopf (2018)" for academic rigor.

*proceeds methodically* Each subsection gets exactly one tight paragraph. For experimental sections (E-G), I'll lead with the key finding before details. "PPM-Reg consistently reduces PD distance" not "Figure 5 shows..."

*last check* User wants "supplementary arguments" in summaries - so for Appendix E.2, I'll include the penalty term equation since it's central to the imperfect convergence analysis. But skip the β=80 value unless critical (it is, for convergence behavior).

*token count anxiety* 500 tokens is tight for Appendix B's dense math. Will sacrifice some lemma notation: "Lemma 1 shows H_Ω ⊃ {ℓ·f | f∈H}" instead of writing the full series.

*confirms* All summaries will be under 500 tokens. Ready to output in the exact requested format.
</think>

### Appendix A Background on Kernels and MMD  
This appendix defines kernel methods foundational to MMD. A feature map $\Phi: \mathcal{X} \to \mathcal{H}$ embeds data into a Hilbert space $\mathcal{H}$, inducing a positive-definite kernel $k(x,y) = \langle \Phi(x), \Phi(y) \rangle_{\mathcal{H}}$. The reproducing kernel Hilbert space (RKHS) $\mathcal{H}$ satisfies the reproducing property $\langle f, k(x,\cdot) \rangle = f(x)$. A kernel is *universal* if $\mathcal{H}$ is dense in $C(\mathcal{X})$, and *characteristic* if the kernel mean embedding $\Phi(\mu) = \int \Phi(x) d\mu(x)$ is injective. The maximum mean discrepancy (MMD) for characteristic kernels is $\text{MMD}_k(\mu,\nu) = \|\Phi(\mu) - \Phi(\nu)\|_{\mathcal{H}}$, metrizing weak convergence on $\mathcal{P}(\mathcal{X})$.  

### Appendix B Characteristic Kernels on $\Omega$  
Theorem 1 proves $k_\Omega$ is characteristic on $\Omega$. Lemma 1 shows $\mathcal{H}_\Omega \supset \{g: g(b,\ell) = \ell \cdot f(b,\ell), f \in \mathcal{H}\}$. Theorem 4 establishes $k_\Omega$ is universal w.r.t. $C_{\text{lin}}(\Omega_T)$ (linear-growth continuous functions). Corollary 1 extends this to characteristicness w.r.t. $C_{\text{lin}}(\Omega)^*$. Theorem 5 shows $\mathcal{M}_{\text{lin}}(\Omega) \subset C_{\text{lin}}(\Omega)^*$, and Lemma 3 proves $\mathcal{P}(\Omega) \cong \mathcal{P}_{\text{lin}}(\Omega)$ via homeomorphism $\psi(\mu) = \mu - \mu(*)\delta_*$. Thus, $k_\Omega$ is characteristic on $\mathcal{P}(\Omega)$.  

### Appendix C Metrizing Weak Topology  
Theorem 6 proves MMD metrizes weak convergence on $\mathcal{P}(\Omega)$: $\text{MMD}_k(\nu_n, \nu) \to 0$ iff $\nu_n(g) \to \nu(g)$ for all $g \in C(\Omega)$. This follows from $\Omega$ being compact Polish and $k_\Omega$ being continuous/bounded. Corollary 2 states MMD and $p$-Wasserstein metrics induce the same topology on $\mathcal{P}(\Omega)$, as both metrize weak convergence (Villani, 2009).  

### Appendix D PPM Regularizer Has Continuous Gradients  
Theorem 7 shows $\mathfrak{H}(\theta) = \Phi_\Omega(\text{PPM}_q(h_\theta(\mu)))$ is $C^1$ if $\mu$ has $C^1$ density and $H(\theta,x) = h_\theta(x)$ is $C^1$. The Fréchet derivative is $\frac{\partial}{\partial \theta_i} \mathfrak{H}(\theta) = \int_{(\mathbb{R}^L)^{2k+2}} \Phi_\Omega \circ \text{PH}_k(x) \frac{\partial f(\theta,x)}{\partial \theta_i} dx$, continuous due to $f$'s $C^1$ property. Theorem 3 follows directly, as the norm is continuously differentiable away from origin.  

### Appendix E Shape Matching  
**E.1 Implementation**: 512-point clouds matched via gradient descent (momentum 0.9, step size 0.05). Reference shapes: unit circle (Fig. 4a,e) and two intersecting circles (Fig. 4b,f). Losses: Cramer, MMD, Cramer+PPM-Reg ($\lambda=1.6$), MMD+PPM-Reg ($\lambda=5$). PPM-Reg hyperparameters: $\lambda=1$, $\lambda_0=1$, $\lambda_1=6000$, $\sigma=0.1$, $s=2000$. PPM-Reg preserves topological features (Fig. 4b,d,f,h).  
**E.2 Imperfect Convergence**: Penalty $f_p = \frac{\lambda_p}{\beta} \ln(1 + e^{\beta(c_\delta - \|c_{\text{train}} - c_{\text{ref}}\|_2)})$ restricts centroid distance. For $c_\delta \in \{0, 0.04, 0.12\}$, MMD+PPM-Reg consistently achieves lower 2-Wasserstein distance to 1D persistence diagrams (Fig. 5) than MMD alone, even with imperfect convergence.  

### Appendix F Unconditional Image Generation  
**F.1 Implementation**: ResNet generators/discriminators (Fig. 8). Ablation (Tables 4–5): $\lambda \in \{1,5,10\}$, $\sigma \in \{0.05,0.5,1.0\}$. PPM-Reg ($\lambda_0=0.001$, $\lambda_1=0.6$, $s=1024$) improves CMMD, $FD_{\text{Dinov2}}$, and $WD_{\text{latent}}$ on AnimeFace/CelebA. Best results at $\lambda=10$ ($\sigma=0.5$) for AnimeFace; $\lambda=5$ ($\sigma=0.5$) for CelebA.  
**F.2 Further Experiments**: 64×64 generation (Fig. 9). PPM-Reg (Fig. 6, Table 6) improves SSL on CelebA (CMMD: 0.46 vs. 0.52) and LSUN Kitchen (CMMD: 1.31 vs. 1.56). LSUN’s complexity yields larger gains (12.52% accuracy increase vs. Baseline on SVHN, Table 10).  

### Appendix G Semi-Supervised Learning  
**G.1 Implementation**: 64D noise input (Fig. 10). PPM-Reg ($\lambda_0=1$, $\lambda_1=90$, $s=1024$) improves classification on Fashion-MNIST/Kuzushiji-MNIST (Tables 7–8). Best at $\lambda=0.025$ ($\sigma=0.1$) for Fashion-MNIST (76.84% vs. 75.39% Baseline).  
**G.2 Further Experiments**: SVHN (Fig. 11) with 400/600 labels. PPM-Reg (Table 11) achieves 57.20% (400 labels) vs. Baseline 44.68%, a 12.52% gain. Consistent improvement across datasets confirms PPM-Reg enhances latent encoding for SSL.  

### Appendix H Network Architecture  
Figures 8–11 detail architectures:  
- **Unconditional (32×32)**: Generator (a) uses ResNet; discriminator (b) outputs 128D latent (Fig. 8).  
- **Unconditional (64×64)**: Modified ResNet for higher resolution (Fig. 9).  
- **SSL (MNIST variants)**: Generator (a), discriminator (b), classifier (c) (Fig. 10).  
- **SSL (SVHN)**: Adapted for color/complexity (Fig. 11).  
All use Adam ($\beta_1=0.0/0.5$, $\beta_2=0.99/99$), batch size 192.