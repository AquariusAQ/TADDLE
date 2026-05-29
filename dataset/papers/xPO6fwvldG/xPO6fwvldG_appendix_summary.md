Hmm, the user is asking me to summarize the appendix of an AI paper with very specific requirements. They want me to split it into original subsections, eliminate redundancies, keep each section under 500 tokens, and use concise academic language. The output format must be "### Appendix X Title + Summary content."

Looking at the appendix content, I see it has several sections: Ethics Statement, Reproducibility Statement, and then Appendix A with subsections A.1 to A.7. I need to handle each one carefully.

For the Ethics Statement, the key points are: the method doesn't involve human subjects but dataset includes Objaverse with human models; potential ethical concerns about unauthorized reconstruction of protected models or human-related 3D data. I'll condense this to the essential ethical considerations without the fluff.

The Reproducibility Statement mentions they provide detailed method descriptions, use public codebases (latent diffusion, TorchSparse), and have training details. I'll highlight the key reproducibility elements without listing every codebase.

Now for Appendix A subsections:

A.1 More Qualitative Results: Just mentions additional figures showing restoration results across subtasks. I can summarize this as "Additional qualitative results demonstrating model performance across all subtasks (completion, super-resolution, refinement) are provided in Fig. 8."

A.2 Network Architecture: This is technical. I need to capture the H-VAE and H-LDM structures concisely. Key points: H-VAE uses cascaded stages with spatial2column downsampling, patch-wise ResBlocks; H-LDM uses U-Net with stage-wise conditional inputs. I'll avoid repeating implementation details.

A.3 Training and Inference Details: Contains lots of numbers (GPUs, epochs, times). I'll summarize the key training specs: 8x A100 GPUs, four training phases (H-VAE, noise-robust encoder, two H-LDM stages), and inference speeds. The table about precomputation speedup (20x) is important to include.

A.4 Evaluation Protocol: Explains metrics (MMD, TMD, AMD) and adaptation to benchmarks. I'll focus on the core metrics and how they handle resolution mismatches.

A.5 Dataset Details: This is lengthy with construction methods for different degradation types (noise-free completion, noisy completion, refinement). I'll condense to: "Dataset constructed with four degradation scenarios (super-resolution, noise-free completion, noisy completion, noisy refinement) at resolutions 32-256, using TSDF grids standardized to 256 resolution."

A.6 3DQD Benchmark Details: Explains differences from their method and preprocessing. Key point: "Adapted 3DQD data to 256-res TSDF by rebuilding partial grids and reprocessing outputs."

A.7 PatchComplete Details: Similar to A.6 but for PatchComplete. Important to note their preprocessing workaround for low-res artifacts and the novel category results.

I need to be ruthless about cutting redundancies. For example, in A.5, they describe resolution upsampling multiple times - I'll just state the standardization to 256-res. Also, the tables (6-9) have numbers I can reference without listing all values.

Word count is critical - each summary must stay under 500 tokens. I'll aim for 300-400 per section. For instance, A.3 has many numbers but I can group them: "H-VAE: 200 epochs, 14 days; Noise-robust encoder: 90 epochs, 6 days; H-LDM stages: 400/200 epochs, 3.5/5 days with 20x speedup via precomputation."

For academic tone, I'll use precise terms like "TSDF grids," "sparse convolutions," "denoiser," and avoid phrases like "we present." Must retain key citations (e.g., Rombach et al., 2022) but not list all.

Finally, I'll verify each summary against the original to ensure no key info is lost - like the ethical concern about human models in Objaverse, or the 20x speedup in Table 6.

*checks token counts mentally* Okay, this structure should work. Now to write it concisely.
</think>

### Appendix A.1 More Qualitative Results  
Additional qualitative results demonstrating model performance across all subtasks (completion, super-resolution, refinement) are provided in Fig. 8. Results show robust restoration of diverse geometric structures under varying degradation scenarios, including noise-free and noisy inputs at multiple resolutions.  

### Appendix A.2 Network Architecture  
Hierarchical VAE (H-VAE) employs cascaded stages with spatial2column downsampling for high-resolution inputs, followed by patch-wise residual blocks (sparse convolutions) for latent encoding. Hierarchical Latent Diffusion Model (H-LDM) uses U-Net denoisers at each stage, conditioning on upsampled latents from prior stages. Conditional inputs (e.g., text) are injected via cross-attention. Both architectures leverage sparse convolutional operations for efficiency.  

### Appendix A.3 Training and Inference Details  
Trained on 8×A100 GPUs: H-VAE (200 epochs, 14 days), noise-robust encoder (90 epochs, 6 days), H-LDM stages (400/200 epochs, 3.5/5 days). Precomputation with noise-robust encoder accelerates H-LDM training 20× (Table 6). Inference: 1st stage (10 timesteps/sec, 10s/sampling, 8GB VRAM), 2nd stage (6.5 timesteps/sec, 15s/sampling, 12GB VRAM).  

### Appendix A.4 Evaluation Protocol  
Metrics: MMD (quality, min CD to GT), TMD (diversity, avg CD among samples), AMD (fidelity approximation due to missing GT regions). Low-res benchmarks (e.g., 3DQD, PatchComplete) are upsampled to 256-res for evaluation.  

### Appendix A.5 General Shape Restoration Dataset  
Constructed with four degradation scenarios: (1) Super-resolution (resolutions 32–256), (2) Noise-free completion (partial shapes), (3) Noisy completion (sparse views + depth/camera noise), (4) Noisy refinement (full coverage + noise). All inputs standardized to 256-res TSDF grids. Resolution impacts performance (Table 7: AMD/MMD improve with higher resolution).  

### Appendix A.6 3DQD Benchmark Adaptation  
3DQD uses 64-res SDF grids from non-manifold meshes (Xu & Barbič, 2020), causing surface deficiencies. Adapted by rebuilding partial TSDF grids at 256-res and reprocessing outputs to match 3DQD’s offset-surface evaluation. Model trained on general data, evaluated in canonical pose.  

### Appendix A.7 PatchComplete Benchmark Adaptation  
PatchComplete uses 32-res TSDF grids from single-view depth fusion. Preprocessing avoids scale inconsistencies by re-fusing with padding and removing boundary artifacts. Fine-tuned on ShapeNet/ScanNet subsets (novel categories), with results reported via MMD (best sample). Outperforms baselines on most categories (Table 8–9).