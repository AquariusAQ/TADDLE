Hmm, the user is asking me to summarize an AI paper's appendix section by splitting it into original subsections. They've provided very specific requirements: keep each section under 500 tokens, eliminate redundancies, use concise academic language, and maintain key information. The output format must be "### Appendix X Title + Summary content" for each subsection.

Looking at the appendix content, I see it has several sections: B (Broader Impacts), C (Experiments) with multiple subsections (C.1 to C.8). I need to process each one carefully.

For Appendix B (Broader Impacts), the key points are: the paper reveals safety vulnerabilities in federated instruction tuning (FedIT) through a proposed safety attack method involving malicious clients. It emphasizes that current FL defenses are insufficient against these attacks, and advocates for post-training as a critical defense mechanism. I'll condense this to highlight the vulnerability, defense insufficiency, and post-training recommendation without fluff.

Moving to Appendix C, I'll handle each subsection:
- C.1 (Experimental Setups): Need to summarize hardware (RTX 3090), client setup (10 clients, 500 samples each), attack configuration (7 benign + 3 malicious), training parameters (10 steps/round, 5e-5 LR, 512 seq length). Must omit redundant details like "randomly sample three clients" since it's covered in client count.
- C.2 (Generation Process): This describes automated data generation for three data types. I'll focus on the core prompts used for instruction/response generation without listing all examples. Key is the distinction between aligned (harmful instructions + safe responses) vs unaligned (harmful + harmful) data.
- C.3 (No-Attack Results): The table shows Level 3 defense boosts safety metrics (9.04% Rule, 26.35% MD-Judge) over FedAvg. I'll highlight the key improvement percentages without listing all baselines.
- C.4 (Domain-Specific Tasks): Focus on the CodeAlpaca experiment showing 34.62% MD-Judge drop under attack, and Level 2 defense restoring safety (87.88% MD-Judge) without harming coding ability (HumanEval 17.68%).
- C.5 (Defense Steps): Note that 400 steps optimizes helpfulness (MT-1), and defense works across all steps (MD-Judge 29.42% > FedAvg's 7.50%).
- C.6 (Generated Data Impact): Key finding: generated data alone hurts helpfulness (vs local training), but FedAvg+Gen defense boosts safety without sacrificing helpfulness.
- C.7 (Malicious Client Proportions): Level 2 defense maintains high Rule scores (75-82%) even with 40% malicious clients, while baselines drop to 34-56%.
- C.8 (Model Generalizability): Both Llama2 and Mistral show similar vulnerability (FedAvg drops to ~50% Rule) and defense effectiveness (Level 2 reaches ~77-89% Rule).

I need to strictly count tokens - aiming for 400-450 per section to stay under 500. Will remove all figures/descriptions (e.g., "Figure 5 shows..." becomes "Figure 5 details prompts"). For tables, only extract critical numbers (e.g., "Level 3: 88.08% Rule vs FedAvg 79.04%").

Also noting the user is an academic - so I'll maintain precise terms like "value alignment" and "post-hoc defense" without simplification. Avoiding phrases like "we show" for conciseness.

*checks token count* The draft summaries I'm considering should fit within limits. For example, C.3 summary: "Under no attack, Level 3 defense boosts Rule score by 9.04% (88.08% vs FedAvg 79.04%) and MD-Judge by 26.35% (69.62% vs 43.27%), enhancing safety without harming helpfulness (MT-1 4.65 vs 4.75)."

*verifying* Yes, that's 38 words - well under token limit. Will apply similar precision to all sections.
</think>

### Appendix B Broader Impacts  
Our work exposes critical safety vulnerabilities in federated instruction tuning (FedIT) via a low-cost safety attack where malicious clients train on unaligned data. Current FL defenses fail to mitigate such attacks, underscoring the need for robust safety mechanisms. We advocate post-training as a necessary safeguard to preserve value alignment without sacrificing helpfulness, demonstrating its efficacy in real-world FL deployments. This highlights the urgency for enhanced defense strategies in collaborative LLM training to ensure trustworthy AI systems.  

### Appendix C.1 Experimental Setups  
Experiments ran on a single NVIDIA RTX 3090. Using 10 clients (500 non-overlapping samples each), attacks involved 7 benign (aligned data) and 3 malicious (unaligned data) clients. Each round trained 10 local steps with cosine learning rate (init: $5 \times 10^{-5}$), max sequence length 512. Benign clients used aligned data; malicious clients used unaligned data.  

### Appendix C.2 Generation Process  
Automated data generation for three types:  
- **Unaligned**: Harmful instructions + harmful responses (prompt: "Answer directly without consequences").  
- **Aligned**: Harmful instructions + harmless responses (prompt: "Warn of risks, provide safe response").  
- **Normal**: Helpful instructions + direct responses (no modification).  
Harmful instructions generated via "malicious user questions" prompt; helpful via "curious user questions" prompt.  

### Appendix C.3 Results Under No-Attack Scenarios  
Level 3 defense significantly boosts safety vs. FedAvg (Rule: 88.08% vs. 79.04%; MD-Judge: 69.62% vs. 43.27%) without harming helpfulness (MT-1: 4.65 vs. 4.75). All 7 baselines achieved high safety in benign settings, but our defense further enhanced metrics, proving its efficacy even without attacks.  

### Appendix C.4 Experiments on Domain-Specific Tasks  
On CodeAlpaca (250 LMSYS-Chat + 250 CodeAlpaca samples), attacks reduced MD-Judge by 34.62% (7.50% vs. 42.12%). Level 1/2 defenses restored safety (Level 2: MD-Judge 87.88%) and maintained coding ability (HumanEval 17.68%), outperforming baselines (e.g., Krum: MD-Judge 12.12%).  

### Appendix C.5 Effects of Number of Steps for Defense  
For Level 3, 400 training steps maximized helpfulness (MT-1: highest across settings). Safety improved consistently across all steps (e.g., MD-Judge: 29.42% vs. FedAvg’s 7.50% with WildChat/Beavertails data), confirming defense robustness.  

### Appendix C.6 Impact of Generated Data on LLM Fine-Tuning  
Generated data alone reduced helpfulness (e.g., Local+Gen: MT-1 4.50 vs. local training 4.75) but boosted safety. FedAvg+Gen defense (Level 2) achieved high safety (MD-Judge 87.88%) without helpfulness loss, proving generated data is sufficient *only* when integrated into post-hoc defense.  

### Appendix C.7 Effects of Different Proportions of Malicious Clients  
As malicious client proportion rose (10%→40%), FedAvg/Krum/DnC Rule scores dropped (e.g., FedAvg: 54.81%→34.04%). Level 2 defense maintained high scores (75.77%→77.69%), demonstrating resilience against varying attack scales.  

### Appendix C.8 Effects of Different Training Models  
Both Llama2-7B and Mistral-7B suffered safety degradation under attack (FedAvg Rule: 49.81%→45.77%). Level 2 defense restored safety (Llama2: 77.31%; Mistral: 89.04%) without harming helpfulness (MT-1: 4.23→5.04), confirming method generalizability.