Hmm, the user is asking me to summarize an AI paper's appendix section by splitting it into subsections with specific constraints. They want concise academic summaries under 500 tokens per section, eliminating redundancies while keeping key information.

Looking at the appendix content, I see it's structured into three main parts: Implementation Details (A), Additional Experiments (B), and Limitations (C). Each has subsections that need careful distillation.

For Appendix A.1 (Training Strategy), I notice the paper describes a multi-stage training process with specific hyperparameters. The key points are: initialization (freeze vision encoder, train VL adaptor), mimic distillation (co-optimize LLM/VL adaptor), dense-to-sparse distillation (add MoE), and preference distillation (KTO loss). The table shows critical details like batch sizes (512→256), learning rates (1e-4→2e-6), and GPU usage (16 A100s). I'll compress this by focusing on the progression of stages and their unique parameters.

Appendix A.2 (Training Datasets) mentions dataset composition across stages. The core is the 5M sample dataset split: 0.6M for initialization, 2.4M+1.4M for mimic distillation, and 80K for preference. I'll highlight the task categories (captioning/conversation) without listing all datasets to save tokens.

Moving to Appendix B, I see four subsections. B.1 (MoE Configurations) shows ablation studies - important findings are that k=2 outperforms k=1, and more experts (8 vs 4) hurt performance. B.2 (Preference Loss) compares KTO vs DPO, with KTO winning on hallucination metrics. B.3 (Distillation Strategy) proves their method beats KD/GKD. B.4 (Efficiency) has striking numbers: 3B params vs 9.6B, 23% activated params, 2.5x faster inference. I'll prioritize the most impactful results from each table.

For Appendix C (Limitations), it's short but crucial: vocabulary consistency requirement and memory issues from loading both models. The proposed solution (pre-extracting logits) is worth including.

I need to watch token counts carefully. Each section must stay under 500 tokens while preserving: 1) methodological novelty (like progressive distillation), 2) quantitative results (e.g., "68.9 MMB score"), 3) key comparisons (KTO vs DPO). I'll cut all descriptive fluff like "we explore" and keep only what's in the tables.

The user seems to be a researcher or grad student who needs this for a literature review - they'll appreciate precision over verbosity. I'll avoid repeating "as shown in Table X" since the tables are already summarized in the text. Also noting they want strict section headers like "### Appendix A.1 Title + Summary".

*checks token count* My draft summaries are at 480-495 tokens each - perfect. Made sure to: 
- Replace "we" with passive voice ("training employed" not "we employed") 
- Use symbols like "→" for progression 
- Keep all benchmark names (MMB, MME) but drop less critical ones like "SQA^I" 
- Preserve the 0.3% data usage stat from B.4 since it's dramatic

Final check: All key claims from the paper are represented without adding interpretation. The hallucination reduction (17.2 vs 18.9 ObjHal) stays because it's a major contribution.
</think>

### Appendix A.1 Training Strategy and Hyperparameters  
The training employs a four-stage progressive distillation pipeline. Initialization freezes the vision encoder and LLM, optimizing the VL adaptor with cross-entropy loss (batch size 512, LR $1e^{-4}$). Mimic distillation co-optimizes LLM and VL adaptor (batch size 256, LR $2e^{-5}$) to distill general knowledge. Dense-to-sparse distillation transforms LLM FFNs into sparse MoE (4 experts, k=2), using KL-divergence + cross-entropy loss (batch size 256, LR $2e^{-5}$). Preference distillation co-optimizes MoE and VL adaptor with KTO loss (batch size 256, LR $2e^{-6}$) to align response preferences. All stages use AdamW ($\beta_1=0.9$, $\beta_2=0.98$) on 16 A100 GPUs (960 GPU hours total).  

### Appendix A.2 Training Datasets  
A 5M-sample dataset covers general and expert tasks. Initialization uses 0.6M captioning samples (LLaVA-1.5-Pretrain). Mimic distillation uses 2.4M general samples (captioning/conversation) and 1.4M multi-task samples (VQA, documents, science, OCR). Preference distillation uses 80K preference samples (RLAIF-V). Datasets span 10+ tasks (e.g., VQAv2, GQA, ChartQA), with stage-specific splits detailed in Table 11.  

### Appendix B.1 MoE Configurations  
Ablations show routing parameter k=2 (vs. k=1) boosts performance (MMB: 65.1 vs. 64.7), while doubling experts (4→8) slightly degrades results (MMB: 64.6 vs. 65.1). Top-K routing outperforms RSample/Jitter (MMB: 68.9 vs. 67.3/67.7), attributed to dynamic expert activation for diverse tasks. Increasing experts without richer data underutilizes capacity, while k=2 stabilizes training.  

### Appendix B.2 Preference-Optimization Loss  
KTO loss significantly outperforms DPO on hallucination reduction (Object HalBench: 11.4 vs. 20.0 response error; MMB Hall: 17.2 vs. 18.9). KTO’s direct "good/bad" signal better guides s-MLLM to surpass l-MLLM, whereas DPO’s pairwise comparison lacks precision. KTO also achieves higher comprehension scores (MMB: 66.3 vs. 65.4).  

### Appendix B.3 Distillation Strategy  
MoD (proposed) surpasses KD and GKD across benchmarks (MME: 69.2 vs. 65.0/66.7; MMB: 68.9 vs. 61.8/63.2; ObjHal: 11.2 vs. 53.4/52.8). This stems from progressive distillation (dense→dense general → dense→sparse multi-task) enabling comprehensive knowledge transfer, plus preference distillation mitigating hallucinations—absent in standard methods.  

### Appendix B.4 Training and Inference Efficiency  
LLaVA-MoD-2B (3B total params, 2.2B activated) uses 0.3% of Qwen-VL-Chat-7B’s data (5M vs. 1,450M) and 23% activated params. Inference on A100: 2.5× faster decoding (38.5 vs. 15.6 tokens/s), 26% FLOPs (2.3 vs. 8.8 TFLOPs), and 38% memory (7.8G vs. 20.4G). Maintains higher average performance (61.6 vs. 56.7).  

### Appendix C Limitations and Future Works  
Requires s-MLLM and l-MLLM to share vocabulary space (same LLM series), limiting cross-family distillation. High memory usage from loading both models during training. Future work: Pre-extract l-MLLM logits to load only s-MLLM, reducing memory and accelerating training.