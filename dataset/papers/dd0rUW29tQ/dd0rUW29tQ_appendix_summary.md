Hmm, the user wants me to summarize an AI paper's appendix by splitting it into subsections with specific constraints. They need concise summaries under 500 tokens per section, eliminating redundancies while keeping key academic information. The output format must be "### Appendix X Title + Summary content."

Looking at the appendix content, I see it covers multiple technical sections about GeNIe and GeNIe-Ada methods. I need to carefully categorize each subsection:

First, A.1 analyzes class probabilities - I'll focus on how GeNIe generates boundary samples (r=0.6-0.7) with class-confusion evidence from SVM probabilities. The key is the 0.2-0.45 overlap in Fig A1/A2.

A.2 proves augmentation independence from test sets via embedding distances (Fig A3). Must mention the higher mean distance for augmentation-test pairs.

A.3 compares augmentation combinations - the critical finding is GeNIe-Ada's 76.8% (1-shot) vs Mixed+MixUp's 65.7%, with MixUp+GeNIe dropping to 74.8%.

A.4 tests backbone effects - DeiT-B gives marginal 0.7% gain over ResNet (Table A2), but they note it's not practical.

A.5 is pseudocode - I'll skip since it's implementation, but note the NUM_INFERENCE_STEPS=50 and partial noising.

A.6 covers textual inversion fine-tuning - GeNIe+TxtInv beats DAFusion (73.9% vs 64.3% on ResNet-18), crucial for diffusion model unawareness.

A.7 gives computational stats - GeNIe-Ada is ~2.1x slower than Txt2Img (9.22s vs 4.12s), with r-dependent speedup.

A.8 details fine-grained experiments - datasets (CUB200, Food101 etc.), 1-shot per class, superclass prompts.

A.9 shows tiered-ImageNet results - GeNIe-Ada leads at 76.9% (1-shot) vs Text2Img's 75.0%.

A.10-12 are supplementary: A.10 compares long-tail methods (GeNIe-Ada at 71.5% overall), A.11 shows semantic trajectories (Fig A5), A.12 explains feature retention via noise control (Fig A4), A.13 analyzes bidirectional noise effects (Fig A6), A.14 has extra visualizations.

I'll compress each to ~400 tokens by:
- Cutting methodological repetitions (e.g., "we compute" → "results show")
- Removing figure references (keep only key findings)
- Using precise terms like "class-confusion" instead of "classifier struggles"
- Preserving all metrics and comparative gains

For A.1, I'll emphasize the r=0.6-0.7 boundary evidence. For A.3, highlight the 0.6% Mixed vs 0.9% MixUp drop. Must keep all backbone comparisons (ResNet-18/50) where relevant.

*checks token count* Need to be ruthless - e.g., in A.12, "DINOv2 encoder" → "DINOv2 embeddings" saves space. Avoid "we observe" → "results show."

*verifies academic terms* "hard-negative" is correct jargon, "semantic trajectory" is standard. Will keep "UniSiam" as it's the baseline method.

*double-checks tables* For A.9, the 76.9% (ResNet-34) is the key result to include, not all baselines.

Final output will strictly follow the ### Appendix X format with no markdown. Each summary must stand alone with self-contained key points.
</think>

### Appendix A.1 Analyzing GeNIe, GeNIe-Ada's Class-Probabilities  
GeNIe and GeNIe-Ada generate hard-negative samples near class decision boundaries by partial noising (noise ratio $r$). Class probabilities $P(Y_S|X_r)$ and $P(Y_T|X_r)$ for $r \in \{0.5,0.6,0.7,0.8,0.9\}$ (Fig. A1) show significant overlap at $r=0.6,0.7$, indicating class-confusion. GeNIe-Ada’s optimal $r^*$ (Fig. A2) yields overlap at 0.2–0.45, confirming samples lie near boundaries. This empirically validates GeNIe’s core mechanism for enhancing decision boundaries.  

### Appendix A.2 Independence of Generated Augmentations from Downstream Test Sets  
Augmentation embeddings (from GeNIe) show higher mean distance to test-set embeddings than train-test pairs (Fig. A3). UMAP visualization confirms clear separation between test and augmented samples, proving no overlap. This ensures evaluation integrity, as augmentations are independent of downstream test sets.  

### Appendix A.3 Additional Augmentation Comparisons  
On mini-ImageNet (5-way, 1-shot/5-shot), Mixed augmentation (combining Weak/Strong) yields marginal gains (0.6% max). MixUp + GeNIe drops performance by 0.9% (74.8% vs. 75.5% for GeNIe alone). GeNIe-Ada achieves 76.8% (1-shot) and 85.9% (5-shot) on ResNet-18, outperforming all traditional augmentation combinations (Table A1).  

### Appendix A.4 Effect of Backbone for Noise Ratio Selector in GeNIe-Ada  
Using DeiT-B (ImageNet-1K pretrained) for noise ratio selection instead of ResNet-18 improves GeNIe-Ada by 0.7% (1-shot: 77.5% vs. 76.8%). This confirms stronger feature extractors enhance augmentation quality, though the paper uses ResNet-18 for practicality.  

### Appendix A.6 Impact of GeNIe with Fine-Tuning  
GeNIe+TxtInv (textual inversion fine-tuning) achieves 73.9% (1-shot) on ResNet-18, significantly outperforming DAFusion (64.3%). This demonstrates GeNIe’s robustness when the diffusion model lacks target-class knowledge, as fine-tuning enables target-category embedding learning without label access.  

### Appendix A.7 Computational Complexity of GeNIe and GeNIe-Ada  
GeNIe’s runtime scales as $1/r$ times Txt2Img (e.g., $r=0.5$: 2.17s vs. 4.12s). GeNIe-Ada (scanning $r \in [0.6,0.8]$) requires 2.1× Txt2Img runtime (9.22s vs. 4.12s), including backbone inference (Table A4).  

### Appendix A.8 Details of Fine-Grained Few-Shot Classification  
Experiments on Food101 (101 classes), CUB200 (200 birds), Cars196 (196 cars), and FGVC-Aircraft (41 aircraft) use superclass-enhanced prompts (e.g., "a burger, a type of food"). 19 samples per class are generated, with Top-1 accuracy averaged over 100 episodes (Table A5).  

### Appendix A.9 Few-Shot Classification with ResNet-34 on tiered-ImageNet  
GeNIe-Ada achieves 76.9% (1-shot) and 86.3% (5-shot) on tiered-ImageNet, surpassing all baselines (e.g., Text2Img: 75.0%/85.4%). This confirms GeNIe-Ada’s superiority in unsupervised few-shot learning (Table A6).  

### Appendix A.10 Additional Details of Long-Tail Experiments  
On ImageNet-LT, GeNIe-Ada improves "Few" set accuracy by 11.7% over LiVT (42.7% → 52.7%) and 4.4% over Text2Img (48.3% → 52.7%) (Table A8). It achieves 71.5% overall accuracy (ResNet-50), outperforming all long-tail baselines.  

### Appendix A.12 Further Analysis of Semantic Shifts Using GeNIe  
GeNIe augmentations transition semantically from source (mushroom) to target (volcano) classes as $r$ increases (Fig. A5). PCA of DINOv2 embeddings shows sparse point distributions at $r \in [0.4,0.6]$, confirming per-sample optimal $r$ selection in GeNIe-Ada.  

### Appendix A.13 How GeNIe Controls Feature Retention  
GeNIe preserves source image features via partial noising ($r$) and contradictory text prompts (Fig. A4). Low $r$ (e.g., $r=0.7$) retains source context; high $r$ ($r \geq 0.9$) loses it. Contradictory prompts (e.g., "dog" for bird source) enable semantic switching to target classes.  

### Appendix A.14 Analyzing Noise Effects in Bi-Directional Transformations  
Applying GeNIe twice (mushroom→volcano→mushroom) shows lower $r$ in the first step preserves more source features, yielding final images closer to the original mushroom (Fig. A6). This validates noise ratio’s role in feature retention during semantic transitions.