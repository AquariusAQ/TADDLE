Hmm, the user is asking me to summarize the appendix of an AI paper with very specific requirements. They want me to split it into original subsections, eliminate redundancies, keep each section under 500 tokens, and use concise academic language. The output format must be "### Appendix A Title + Summary content" etc.

Looking at the appendix content, I see it has several sections: A (Algorithm), B (SSL Methods Overview), C (Experimental Details), and D (Theoretical Analysis). I need to handle each carefully.

For Appendix A (Algorithm), I'll focus on the three-phase process: unsupervised feature learning, classifier training, and permutation testing. I'll cut the verbose variable descriptions and keep only essential steps. The key is showing how SSL-C2ST differs from standard C2ST by adding the autoencoder pre-training phase.

Appendix B is about SSL method categories. I notice it explains transductive vs inductive learning in relation to two-sample testing - that's crucial. I'll condense the five SSL categories into one-line definitions, emphasizing why transductive learning is necessary for their problem. Must remove the repeated "we will" phrases.

Appendix C has many subsections. For C.1 (Datasets), I'll summarize HDGM levels and the three benchmark datasets (HDGM, MNIST, ImageNet) with their key parameters. For C.2 (Implementation), I'll contrast C2ST vs SSL-C2ST in one sentence. C.3 (SSL-C2ST-M) can be simplified to its core innovation: using MMD on features instead of labels. C.4 (HDGM Details) is redundant with C.1 so I'll skip it. C.5-C.9 are minor - I'll compress computing resources, sequential testing discussion, and future work into minimal points.

Appendix D (Theoretical) is dense. Theorem 5.1 proof shows test power optimization via minimizing ε. I'll extract the key insight: SSL-C2ST maximizes power by reducing ε (inability to distinguish distributions). Theorem 5.3 proof about error bounds can be summarized as "proves generalization with VC-dimension bounds."

I need to watch token counts strictly. Each section must be under 500 tokens. I'll aim for 300-400 to be safe. Also must remove all redundant phrases like "we present" or "it is clear that" - academic summaries should be direct.

The user seems to be a researcher or grad student preparing a paper submission. They're probably time-pressed and need a clean appendix summary for their revision. The strict token limit suggests journal constraints. I should ensure every word adds value - no fluff.

For the output format, I'll strictly follow "### Appendix X Title + Summary" without any extra text. No markdown beyond the headers as requested. Will count tokens carefully before finalizing.
</think>

### Appendix A Algorithm  
SSL-C2ST trains a classifier via three phases: (1) Unsupervised feature learning using an autoencoder on pooled unlabeled data $S_{\text{unl}} = S_P \cup S_Q$, minimizing MSE reconstruction loss; (2) Supervised classifier training on labeled data $S^{\text{tr}} = (S_P^{\text{tr}}, \mathbf{0}) \cup (S_Q^{\text{tr}}, \mathbf{1})$ with cross-entropy loss; (3) Permutation test on $S^{\text{te}}$ using classifier accuracy to compute $p$-value. The method leverages transductive learning, using unlabeled test data for representation learning before classification.  

### Appendix B Overview of Major Categories of SOTA SSL Methods  
SSL methods fall into five categories: (1) *Consistency Regularization* enforces smoothness via input perturbations; (2) *Pseudo-Labelling* assigns high-confidence predictions as labels; (3) *Graph-Based* propagates labels via similarity graphs; (4) *Generative Models* learn data distributions for reconstruction; (5) *Hybrid* combines techniques. For two-sample testing, transductive learning is essential (vs. inductive), as SSL-C2ST requires learning from the specific unlabeled data used in testing. Most SOTA SSL methods fail here due to inductive assumptions.  

### Appendix C Experimental Details  
**C.1 Datasets**: HDGM-Easy/Medium/Hard (synthetic Gaussian mixtures with varying cluster separation); HDGM-S (null, $H_0$) and HDGM-D (alternative, $H_1$); MNIST vs. MNIST-Fake (image); ImageNet vs. ImageNet-Fake (large-scale image). HDGM levels test SSL-C2ST’s robustness to high overlap.  
**C.2 Implementation**: SSL-C2ST adds unsupervised pre-training (autoencoder) to C2ST’s two phases. C2ST uses supervised training only; SSL-C2ST uses unlabeled data for feature extraction before classification.  
**C.3 SSL-C2ST-M**: Uses MMD on classifier features (not labels) for test statistic. For HDGM, computes mean probability differences; for images, uses hidden-layer features (size 100) for MMD.  
**C.4–C.9**: HDGM synthesis details (Table 3) omitted as redundant; computing resources (NVIDIA GPUs); sequential testing comparison (C2ST-Sequential(5) underperforms SSL-C2ST); future work (replacing autoencoder with VAEs); reproducibility (code link).  

### Appendix D Theoretical Analysis  
**D.1 Theorem 5.1 Proof**: Under $H_0$, test statistic $\hat{t} \sim \mathcal{N}(1/2, 1/(4n_{\text{te}}))$; under $H_1$, $\hat{t} \sim \mathcal{N}(1-\epsilon, \epsilon(1-\epsilon)/n_{\text{te}})$. Test power $\pi = \Phi\left(\frac{(1/2 - \epsilon)\sqrt{n_{\text{te}}} - \Phi^{-1}(1-\alpha)/2}{\sqrt{\epsilon - \epsilon^2}}\right)$. Maximizing power requires minimizing $\epsilon$ (inability to distinguish $P$ and $Q$), proven via optimization. Type-I error is controlled at $\alpha$.  
**D.2 Theorem 5.3 Proof**: Shows generalization bound: empirical error $\hat{\epsilon} \leq \epsilon + \Delta/2 + \sqrt{\ln(4s/\delta)/(8m_{\text{u}})}$ with high probability, where $\epsilon$ is true inability, $m_{\text{u}}$ is unlabeled data size, and $s$ is VC-dimension. Ensures SSL-C2ST’s error minimization generalizes.