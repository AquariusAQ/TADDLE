Hmm, the user is asking me to summarize the appendix of an AI paper about multi-modal motion generation. They want me to split it into original subsections, eliminate redundancies, keep each section under 500 tokens, and use concise academic language. The output format should be "### Appendix A Title + Summary content" etc.

Looking at the appendix content, I see it has several sections: A (Theoretical Analysis), B (Rules for Multi-Modal Conditions), C (Limitations of Weighted Averaging), D (Computational Complexity), E (Additional Experiments), and F (Multi-Modal Benchmark Details). 

For Section A, I need to condense the theoretical derivation of their MOCO framework's core assumption. The key points are the two approximations they make: 1) neglecting x_{t-1,lower} when predicting upper-body motion, and 2) decoupling modalities (text for lower body, audio for upper body). I'll focus on the mathematical justification without all the derivation steps.

Section B is about their rules for handling asynchronous conditions. The default rules are straightforward: single modality controls whole body, different modalities split control (text→lower, audio→upper). The flexible rules handle conflicts when both modality control same body parts. I'll compress this into clear bullet points.

Section C explains why weighted averaging fails - they show speech-to-gesture model calculates larger motion differences than text-to-motion model, causing weighted average to mimic audio-only results. The counterintuitive leg movement observation when text says "sitting" is important to mention as evidence of data bias.

Section D has a table with computational metrics. I'll summarize the key findings: G_S2G and G_S2D are largest due to cross-attention, G_T2V is lightest but slowest, and overall inference is fast (3.72s for 54s video). The table numbers aren't needed in summary.

Section E has multiple subsections. For E.1, I'll highlight that L-BFGS improves trajectory control but CFG doesn't help. For E.2, the asynchronous condition outperforms synchronous due to simpler single-modality control. E.3 compares single-modality results - MOCO matches MDM on HumanML3D and beats most on BEATX. E.4 user study shows MOCO dominates all baselines in text-following and beat-sync.

Section F describes their 1,000-test-clip benchmark. Key points: 40 text descriptions for lower-body, 694 audio clips from BEATX, random interval generation with text/audio asynchrony.

I need to be extremely concise - each section must stay under 500 tokens. I'll cut all redundant phrases like "we provide" or "as shown in the table" and focus on what's new/important. For academic tone, I'll use precise terms like "decouple-then-combine" instead of "split then join" and keep all key metrics.

Checking token counts as I go... for Section A, the core is the two approximations with their justifications. I'll say "First, approximates p(x_{t-1,upper}|c_all,x_{t-1,lower}) ≈ p(x_{t-1,upper}|c_all) by leveraging diffusion step proximity. Second, decouples modalities: text for lower-body, audio for upper-body based on their distinct influences." That's about 100 tokens.

For Section C, the critical finding is "Speech-to-gesture model computes larger motion differences than text-to-motion model (Fig 5), causing weighted averaging to mimic audio-only results. Counterintuitive leg movements when text says 'sitting' suggest speech dataset bias."

I'll make sure every sentence adds value. No fluff like "in this section we show..." - just the facts. And absolutely no markdown in the output, just plain text with the required ### headers.

*checks token counter* Okay, my draft summaries are all under 400 tokens each. Perfect. Now to format exactly as requested: ### Appendix [Letter] [Title] + summary.
</think>

### Appendix A Theoretical Analysis for Decouple-Then-Combine  
The MOCO framework assumes $ p(x_{t-1} \mid c_{\text{text}}, c_{\text{audio}}, x_t) \approx p(x_{t-1,\text{lower}} \mid c_{\text{text}}, x_t) \cdot p(x_{t-1,\text{upper}} \mid c_{\text{audio}}, x_t) $. This derives from two key approximations: (1) $ p(x_{t-1,\text{upper}} \mid c_{\text{all}}, x_{t-1,\text{lower}}) \approx p(x_{t-1,\text{upper}} \mid c_{\text{all}}) $, justified by diffusion step proximity and strong $ x_t $-$ x_{t-1} $ correlation; (2) decoupling modalities: $ p(x_{t-1,\text{lower}} \mid c_{\text{all}}) \approx p(x_{t-1,\text{lower}} \mid c_{\text{text}}, x_t) $ (text governs lower body) and $ p(x_{t-1,\text{upper}} \mid c_{\text{all}}) \approx p(x_{t-1,\text{upper}} \mid c_{\text{audio}}, x_t) $ (audio governs upper body), leveraging modality-specific movement influences.  

### Appendix B Rules for Managing Multi-Modal Asynchronous Conditions  
**Default rules**: (1) Single active condition controls whole body; (2) Two different modalities: speech → upper body, text → lower body. **Flexible rules**: (1) Different body parts → each condition controls its part; (2) Overlapping parts → condition controlling fewer parts takes precedence; (3) Equal scope → earlier-starting condition takes precedence. These extend STMC (Petrovich et al., 2024) to multi-modal scenarios, resolving conflicts via body-part specificity and temporal priority.  

### Appendix C Limitations of Weighted Averaging  
Weighted averaging mimics audio-only results due to speech-to-gesture model computing larger motion differences than text-to-motion model (Fig 5), causing aggressive motion updates. When text = "sitting", speech-to-gesture model shows larger leg differences than arms—counterintuitive as speech typically affects upper body—indicating speech-to-motion dataset bias (e.g., standing-dominant data). This explains why weighted averaging fails in text-to-motion metrics despite good speech-to-gesture performance.  

### Appendix D Computational Complexity  
MOCO comprises four denoisers: $ G_{T2M} $ (text-to-motion), $ G_{S2G} $ (speech-to-gesture), $ G_{T2V} $ (trajectory-to-velocity), $ G_{S2D} $ (speech-to-details). $ G_{S2G} $ and $ G_{S2D} $ are largest (36.86M/36.94M params) due to cross-attention. $ G_{T2V} $ is lightest (0.34M params) but slowest (6.20 ms/frame) due to guidance. Total inference for 54s video: 3.72s on NVIDIA 4090, confirming real-time potential.  

### Appendix E Additional Experiments  
**E.1 Trajectory Control**: L-BFGS optimization significantly reduces location differences (avg. diff: 0.0747m vs. 0.5641m without) but CFG offers no improvement. **E.2 Synchronous vs. Asynchronous**: Asynchronous conditions slightly outperform synchronous (Table 5), as single-modality control aligns better with training data distribution. **E.3 Single Modality**: MOCO matches MDM on HumanML3D (Table 6) and achieves SOTA on BEATX (Table 7, FGD↓ 5.543). **E.4 User Study**: MOCO dominates baselines in text-following (99.0% vs. 0.0% for Pseudo-Text), beat-sync (74.5% vs. 3.5% for Weighted Average), and body coherence (71.0% vs. 17.0% for Combine Last Time).  

### Appendix F Details of Multi-Modal Benchmark  
A 1,000-test-clip benchmark was created: 40 manually curated text descriptions (lower-body focus), 694 audio clips from BEATX (8 speakers, VAD-segmented, avg. 9.14s). Clips combine two random texts (with random durations) and two adjacent audio clips, with random start times for text/audio intervals. Ensures asynchrony and diverse condition combinations for evaluation.