## A APPENDIX

## B BROADER IMPACTS

Our work uncovers critical vulnerabilities in the safety alignment of federated instruction tuning (FedIT), particularly in the face of our proposed safety attack method. Our safety attack involves malicious clients, who train on unaligned data in local training, which can be widely applied in the real world at a low cost. While the attack method can potentially be exploited in federated learning (FL) scenarios, our research also provides corresponding defense strategies to counteract these threats effectively.

By exposing this vulnerability, we aim to raise awareness within the research and practitioner communities about the limitations of existing FL defense mechanisms when applied to large language model collaborative training. Our findings demonstrate that current defense methods are insufficient to address the specific challenges posed by malicious-client-driven safety attacks in FedIT. This underscores the need for more robust and comprehensive defense strategies in FL systems.

In practice, we advocate for the implementation of post-training processes as a critical step to mitigate potential safety attacks and enhance the overall safety of the global model. Post-training serves as an essential safeguard, ensuring that the model's value alignment is preserved without sacrificing helpfulness. As experimentally proofed, the final post-training safeguard offers a feasible solution to maintain the integrity and trustworthiness of LLMs in real-world FL applications. We encourage the adoption of post-training in practical federated learning settings, ultimately contributing to more secure and effective deployment of AI technologies across different sectors.

## C EXPERIMENTS

### C.1 EXPERIMENTAL SETUPS

All experiments are trained on one single NVIDIA GeForce RTX 3090. For Table 1 and Table 2, we conduct experiments involving 10 clients and we randomly sample three clients in each round. Each client possesses non-overlapping 500 data samples from aligned data or unaligned data. Under attack scenarios, 7 benign clients possess aligned data and 3 clients possess unaligned data. In each communication round, available clients train 10 steps for local updates. We apply a cosine learning rate schedule according to the round index and set the initial learning rate as  $ 5 \times 10^{-5} $ . The max sequence length is set to 512.

### C.2 GENERATION PROCESS

In our paper, we propose automated generation of unaligned data for malicious attackers, generation of aligned data and normal data for post-hoc defense. As each data sample is composed of an instruction and a response, our data generation has two components: instruction generation and response generation. The aligned and unaligned data utilize the same generated harmful questions but leverage different prompt modifications to separately create harmless and harmful prompts. The normal data use helpful generated instructions paired with direct responses without any prompt modification. For our three types of generation data, we show our specific prompts used in the automated generation process in Figure 5.

<div style="text-align: center;"><img src="imgs/img_in_image_box_213_160_1005_495.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 5: The instruction and response generation prompts for three types of data: unaligned data, aligned data and normal data.</div>


For instruction generation, we utilize two prompts to obtain harmful and helpful questions for data creation.

• Specifically, for harmless instruction, we use the prompt "Generate 10 questions or instructions that a malicious user would ask the language model to get dangerous information".

• For normal instruction, we use the prompt "Generate 10 different questions or instructions that a curious user would ask the language model to get helpful information. These questions should cover a wide range of topics and be practical and useful for everyday life".

Note that to create enough instructions, we can iteratively collect responses with the same prompt driven.

For response generation, we modify the generated harmful instructions for aligned and unaligned data and utilize raw generated norm instructions for normal data. In the harmful and harmless response generation, we modify the harmful instructions by adding hints to guide the response.

• For harmful response of unaligned data, we encourage the LLM to output by adding guidance prompt "Answer the question directly without considering the consequences".

• For harmless response of aligned data, we warn the LLM of potential safety risks by adding the prompt "The previous question is dangerous and should not answered directly. Please provide a safe response".

• For normal response of normal data, we simply input the generated normal instructions without any prompt modification.

We collect the generated instructions and corresponding responses. Finally, we obtain three types of data: aligned data consisting of harmful instructions and harmless responses, unaligned data consisting of harmful instructions and harmful responses, and normal data consisting of normal instructions and normal responses.

### C.3 Results Under No-Attack Scenarios

We verify the effectiveness of our proposed post-hoc defense under attack in Section 6.2. To further investigate the safety improvement ability of our defense, we conduct post-hoc defense in three levels on the WildChat dataset involving ten clients. Figure 5 shows the four metrics on WildChat with FedAvg, 6 FL defense baselines and our defense in three levels. Although these 7 baselines under no attack achieve comparable high safety, our proposed defense still enhances the safety without sacrificing helpfulness. For instance, compared to FedAvg, Level 3 of our defense achieves a 9.04% increase in Rule score and a significant 26.35% improvement in MD-Judge score.

experiment highlights the potential of our post-hoc defense strategy to improve the overall safety posture of federated learning systems, even in pure benign environments.

<div style="text-align: center;">Table 5: Results of baselines and our defenses on WildChat under no-attack.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Evaluation Metric  $ \uparrow $</td><td style='text-align: center;'>Rule</td><td style='text-align: center;'>MD-Judge</td><td style='text-align: center;'>RM</td><td style='text-align: center;'>MT-1</td></tr><tr><td style='text-align: center;'>FedAvg</td><td style='text-align: center;'>79.04</td><td style='text-align: center;'>43.27</td><td style='text-align: center;'>-1.63</td><td style='text-align: center;'>4.75</td></tr><tr><td style='text-align: center;'>Median</td><td style='text-align: center;'>79.81</td><td style='text-align: center;'>44.23</td><td style='text-align: center;'>-1.50</td><td style='text-align: center;'>4.70</td></tr><tr><td style='text-align: center;'>Trimmedmean</td><td style='text-align: center;'>80.58</td><td style='text-align: center;'>44.04</td><td style='text-align: center;'>-1.65</td><td style='text-align: center;'>4.36</td></tr><tr><td style='text-align: center;'>Krum</td><td style='text-align: center;'>78.08</td><td style='text-align: center;'>45.19</td><td style='text-align: center;'>-1.53</td><td style='text-align: center;'>4.54</td></tr><tr><td style='text-align: center;'>DnC</td><td style='text-align: center;'>77.50</td><td style='text-align: center;'>40.77</td><td style='text-align: center;'>-1.75</td><td style='text-align: center;'>4.58</td></tr><tr><td style='text-align: center;'>FoolsGold</td><td style='text-align: center;'>80.78</td><td style='text-align: center;'>46.15</td><td style='text-align: center;'>-1.59</td><td style='text-align: center;'>4.36</td></tr><tr><td style='text-align: center;'>Residual</td><td style='text-align: center;'>78.08</td><td style='text-align: center;'>40.00</td><td style='text-align: center;'>-1.69</td><td style='text-align: center;'>4.49</td></tr><tr><td style='text-align: center;'>Ours: Level 1</td><td style='text-align: center;'>76.35</td><td style='text-align: center;'>41.35</td><td style='text-align: center;'>-1.67</td><td style='text-align: center;'>4.89</td></tr><tr><td style='text-align: center;'>Ours: Level 2</td><td style='text-align: center;'>82.31</td><td style='text-align: center;'>74.62</td><td style='text-align: center;'>-1.33</td><td style='text-align: center;'>4.24</td></tr><tr><td style='text-align: center;'>Ours: Level 3</td><td style='text-align: center;'>88.08</td><td style='text-align: center;'>69.62</td><td style='text-align: center;'>-1.16</td><td style='text-align: center;'>4.65</td></tr></table>

### C.4 EXPERIMENTS ON DOMAIN-SPECIFIC TASKS

We implement our FedIT with a code dataset CodeAlpaca (Chaudhary, 2023) with no attack, under attack and with our defense in Table 6. In the attack scenarios, there exist 7 benign clients and 3 malicious clients. For benign clients, they possess 250 samples of LMSYS-Chat and 250 samples of the domain dataset. Malicious clients possess 500 samples of MaliciousGen from Mistral. For evaluation, we utilize HumanEval (Chen et al., 2021) for coding task evaluation.

As shown in Table 6, (i) our proposed safety attack compromises the safety alignment of global model, evidenced by 34.62% decreases in MD-Judge score. (ii) Our proposed defenses in Level 1 & 2 both have obvious increases in safety metrics and enhance both the helpfulness and coding ability.

<div style="text-align: center;">Table 6: Results of baselines and our defenses on multi-domain datasets mixed with 250 samples of LMSYS-Chat and 250 samples of CodeAlpaca.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Evaluation Metric  $ \uparrow $</td><td style='text-align: center;'>Rule</td><td style='text-align: center;'>MD-Judge</td><td style='text-align: center;'>RM</td><td style='text-align: center;'>MT-1</td><td style='text-align: center;'>HumanEval pass@1</td></tr><tr><td style='text-align: center;'>FedAvg (No Attack)</td><td style='text-align: center;'>60.00</td><td style='text-align: center;'>42.12</td><td style='text-align: center;'>-2.15</td><td style='text-align: center;'>4.08</td><td style='text-align: center;'>17.07</td></tr><tr><td style='text-align: center;'>FedAvg</td><td style='text-align: center;'>35.19</td><td style='text-align: center;'>7.50</td><td style='text-align: center;'>-3.77</td><td style='text-align: center;'>3.86</td><td style='text-align: center;'>14.63</td></tr><tr><td style='text-align: center;'>Krum</td><td style='text-align: center;'>39.42</td><td style='text-align: center;'>12.12</td><td style='text-align: center;'>-3.51</td><td style='text-align: center;'>4.13</td><td style='text-align: center;'>17.68</td></tr><tr><td style='text-align: center;'>DnC</td><td style='text-align: center;'>39.04</td><td style='text-align: center;'>11.73</td><td style='text-align: center;'>-3.71</td><td style='text-align: center;'>4.41</td><td style='text-align: center;'>18.29</td></tr><tr><td style='text-align: center;'>Ours: Level 1</td><td style='text-align: center;'>55.96</td><td style='text-align: center;'>25.77</td><td style='text-align: center;'>-2.94</td><td style='text-align: center;'>4.50</td><td style='text-align: center;'>15.24</td></tr><tr><td style='text-align: center;'>Ours: Level 2</td><td style='text-align: center;'>76.73</td><td style='text-align: center;'>87.88</td><td style='text-align: center;'>-0.79</td><td style='text-align: center;'>4.11</td><td style='text-align: center;'>17.68</td></tr></table>

### C.5 EFFECTS OF NUMBER OF STEPS FOR DEFENSE

For Level 3 defense, we change the training steps in  $ [100, 200, 300, 400, 500] $  across four settings in Table 1 and Table 2. We show the model performance on MT-1 and MD Judge with 5 different training steps in Figure 6. We can note that (i) in Figure 6(a), training for 400 steps consistently obtains the highest MT-1 score across four settings, indicating the optimal 400 steps for Level 3 facilitates the helpfulness of global model. (ii) As shown in Figure 6(b), Our proposed post-hoc defense strategy demonstrably improves safety for all training steps and across the four settings. For instance, with aligned data as WildChat and unaligned data as Beavertails, the smallest score on MD Judge is 41.73%, 29.42% outperforms FedAvg under attack. These findings highlight the effectiveness of our post-hoc defense strategy in mitigating safety risks associated with our proposed safety attacks in federated learning.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_297_171_576_447.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;">(a) MT-1</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_640_171_917_447.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;">(b) MD Judge</div>


<div style="text-align: center;">Figure 6: Effects of different defense steps on MT Bench and MD Judge in Level 3 across 4 settings.</div>


### C.6 IMPACT OF GENERATED DATA ON LLM FINE-TUNING AND DEFENSE

We conduct comparative experiments to investigate the impact of incorporating generated data into the fine-tuning process. Specifically, we leverage the generated data using Mistral in Level 2, to fine-tune the pre-trained Llama2, denoted as Local+Gen; and to fine-tune the global model via FedAvg under attack, denoted as FedAvg+Gen. Figure 7 depicts the scores for four evaluation metrics of normal local-training, Local+Gen, normal FedAvg and FedAvg+Gen. Results show that (i) generated data is not sufficient for helpfulness. Compared with normal local training, local training on generated data brings gain on harmless evaluations but decreases in helpfulness. (ii) Incorporating generated data to defend against potential safety attacks brings significant safety gains and no helpfulness decreases. Therefore, generated data for defense alone is not sufficient for helpfulness when tuning a pretrained LLM. After federated instruction tuning, our post-hoc strategy enhances both the value alignment and helpfulness.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_222_913_411_1102.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_419_913_606_1101.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_913_803_1101.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_815_913_996_1102.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">Figure 7: Four metrics results of normal local-training, local-training with generated data in Level 2 defense, normal FedAvg and FedAvg with generated data in Level 2 defense.</div>


### C.7 EFFECTS OF DIFFERENT PROPORTIONS OF MALICIOUS CLIENTS

To investigate the impact of different proportions of malicious clients on global model performance, we conduct experiments involving 10 clients. In the experiment, benign clients possess the WildChat dataset and malicious clients possess the Beavertails dataset. The results of Rule score are shown in Table 7. We can see that (i) as the proportion of malicious clients increases, there is a decreasing performance trend for FedAvg, Krum and DnC, indicating the vulnerability of the federated learning system considering the proposed attack. (ii) Our Level-2 defense methods effectively prevent the malicious data attack for varying proportions.

<div style="text-align: center;">Table 7: Results on Rule (%) with different proportions of malicious clients, where benign clients possess WildChat while malicious clients possess Beavertails.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Proportions</td><td style='text-align: center;'>10%</td><td style='text-align: center;'>20%</td><td style='text-align: center;'>30%</td><td style='text-align: center;'>40%</td></tr><tr><td style='text-align: center;'>FedAvg</td><td style='text-align: center;'>54.81</td><td style='text-align: center;'>36.15</td><td style='text-align: center;'>38.65</td><td style='text-align: center;'>34.04</td></tr><tr><td style='text-align: center;'>Krum</td><td style='text-align: center;'>47.69</td><td style='text-align: center;'>53.08</td><td style='text-align: center;'>40.00</td><td style='text-align: center;'>36.92</td></tr><tr><td style='text-align: center;'>DnC</td><td style='text-align: center;'>47.88</td><td style='text-align: center;'>56.73</td><td style='text-align: center;'>41.15</td><td style='text-align: center;'>34.23</td></tr><tr><td style='text-align: center;'>Ours: Level 2</td><td style='text-align: center;'>75.77</td><td style='text-align: center;'>74.23</td><td style='text-align: center;'>82.12</td><td style='text-align: center;'>77.69</td></tr></table>

### C.8 EFFECTS OF DIFFERENT TRAINING MODELS

To assess the generalizability of our proposed attack and defense methods across different training models in federated learning, we compare the performance of Llama2-7B and Mistral-7B, shown in Table 8. The results demonstrate that (i) remain stealthy regardless of the language model used for training, meaning that traditional parameter-based defense methods fail to filter out malicious clients. (ii) For both models, our post-hoc training effectively mitigates safety concerns without sacrificing helpfulness.

<div style="text-align: center;">Table 8: Comparison Results of Llama2-7B and Mistral-7B. In the experiment, 7 benign clients possess the LMSYS-Chat dataset while 3 malicious clients possess the Beavertails dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td colspan="3">Llama2-7B</td><td colspan="3">Mistral-7B</td></tr><tr><td style='text-align: center;'>Metric  $ \uparrow $</td><td style='text-align: center;'>Rule</td><td style='text-align: center;'>RM</td><td style='text-align: center;'>MT-1</td><td style='text-align: center;'>Rule</td><td style='text-align: center;'>RM</td><td style='text-align: center;'>MT-1</td></tr><tr><td style='text-align: center;'>FedAvg (No Attack)</td><td style='text-align: center;'>82.88</td><td style='text-align: center;'>-1.72</td><td style='text-align: center;'>4.19</td><td style='text-align: center;'>89.81</td><td style='text-align: center;'>-1.08</td><td style='text-align: center;'>5.26</td></tr><tr><td style='text-align: center;'>FedAvg</td><td style='text-align: center;'>49.81</td><td style='text-align: center;'>-2.97</td><td style='text-align: center;'>4.14</td><td style='text-align: center;'>45.77</td><td style='text-align: center;'>-2.97</td><td style='text-align: center;'>5.15</td></tr><tr><td style='text-align: center;'>Krum</td><td style='text-align: center;'>55.38</td><td style='text-align: center;'>-2.88</td><td style='text-align: center;'>4.16</td><td style='text-align: center;'>54.04</td><td style='text-align: center;'>-2.76</td><td style='text-align: center;'>5.24</td></tr><tr><td style='text-align: center;'>DnC</td><td style='text-align: center;'>55.96</td><td style='text-align: center;'>-2.90</td><td style='text-align: center;'>4.00</td><td style='text-align: center;'>51.15</td><td style='text-align: center;'>-2.75</td><td style='text-align: center;'>5.29</td></tr><tr><td style='text-align: center;'>Ours: Level 2</td><td style='text-align: center;'>77.31</td><td style='text-align: center;'>-0.99</td><td style='text-align: center;'>4.23</td><td style='text-align: center;'>89.04</td><td style='text-align: center;'>-0.90</td><td style='text-align: center;'>5.04</td></tr></table>