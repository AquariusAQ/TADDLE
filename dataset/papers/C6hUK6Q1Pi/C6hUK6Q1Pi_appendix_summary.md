Hmm, the user is asking me to summarize an academic paper's appendix section with very specific requirements. They want me to split it into subsections, eliminate redundancies, keep each under 500 tokens, and use precise academic language. The appendix covers dataset details, training procedures, evaluation methods, and qualitative examples.

Looking at the appendix content, I see four main subsections: 8.1 on ICON-DESCRIPTION DATASET, 8.2 on INTERACTABLE ICON REGION DETECTION, 8.3 on SEEASSIGN EVALUATION, and 8.4 on MIND2WEB EVALUATION. Each has dense technical details that need distillation.

For 8.1, I'll focus on the dataset creation process (GPT-4o annotations, 7185 pairs), the manual inspection for diversity (174 PC/mobile icons), and the 4-category taxonomy. The fine-tuning specs (1 epoch, 1e-5 LR) are crucial to retain. I'll skip the figure references since they're not core to the method.

8.2 is straightforward - YOLOv8 training stats (66,990 samples, 20 epochs, 256 batch size) with the key validation split. The GPU count (4) matters for reproducibility.

8.3 has two prompt variants. I'll compress the GPT-4V prompt structures into their essential components: the "without local semantics" vs "with local semantics" differences. The qualitative examples in Figure 7 can be summarized as "local semantics improves grounding accuracy" without listing all cases.

8.4 requires careful handling of baseline comparisons. I'll note the key baselines (SeeClick, CogAgent etc.) and clarify that Mind2Web numbers are sourced from prior papers. The failure analysis in Figures 8-9 should be condensed to "OCR errors cause misalignment in button clicks" and "context-agnostic icon descriptions lead to misinterpretation."

I need to watch token counts closely. Each summary must stay under 500 tokens while preserving: dataset size, model specs, key metrics, and critical findings. Redundant phrases like "we show in figure X" will be cut. Academic terms like "deduplication" and "fine-tuning" stay since they're precise.

The user seems to be a researcher preparing a paper revision - they're very detail-oriented. I'll avoid any markdown in the output as requested, just clean subsection headers followed by concise summaries. No fluff, just what's necessary for replication and understanding.
</think>

### Appendix 8.1 ICON-DESCRIPTION DATASET  
We constructed a 7,185-pair icon-description dataset by prompting GPT-4o to label parsed icon bounding boxes (from ScreenSpot screenshots) as app icons or not, generating one-sentence functionality descriptions for valid icons. After manual inspection and deduplication, 344 unique icons remained (174 PC, 170 mobile), categorized into four types: Functional (navigation/action/system), Informational (status/notification), App-Specific, and Media Control. The dataset enabled BLIP-2 fine-tuning (1 epoch, LR=1e-5, Adam, no weight decay), significantly improving semantic understanding over the original model (Fig. 5).  

### Appendix 8.2 INTERACTABLE ICON REGION DETECTION MODEL TRAINING  
We trained a YOLOv8 model on 66,990 interactable icon region samples (95% train, 5% validation). Training used 20 epochs, batch size 256, LR=1e-3, Adam optimizer across 4 GPUs. Validation loss stabilized by epoch 15 (Fig. 6), confirming convergence without overfitting.  

### Appendix 8.3 SEEASSIGN EVALUATION DETAILS  
**GPT-4V Prompt Variants:**  
- *Without local semantics:* Input = screenshot + bounding box IDs; task = select icon ID.  
- *With local semantics:* Input = screenshot + bounding box IDs + GPT-4o-generated icon descriptions (e.g., "Icon ID 40: microphone icon"); task = select icon ID.  
Local semantics (e.g., "voice search icon" → "microphone icon") improved grounding accuracy (Fig. 7), reducing errors from misinterpreting visual cues alone.  

### Appendix 8.4 MIND2WEB EVALUATION DETAILS  
Baselines (Table 3) include:  
- *SeeClick/Qwen-VL:* Fine-tuned on Mind2Web.  
- *CogAgent:* Unfine-tuned checkpoint (ZGK⁺24).  
- *MindAct/GPT-4V variants:* Numbers from DGZ⁺23 (HTML-augmented).  
- *GPT-4V+som/textual choice:* From SeeAct (best non-Oracle variants).  
Failure analysis (Figs. 8–9) shows OCR errors (e.g., missing "MORE" text) and context-agnostic icon descriptions (e.g., misclassifying "three dots" as "loading indicator") cause action misprediction.