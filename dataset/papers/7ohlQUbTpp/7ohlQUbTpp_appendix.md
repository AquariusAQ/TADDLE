## A SOFTWARE AND HARDWARE

We run all experiments with Python 3.7.4 and PyTorch 1.9.0. For all experimentation, we use two Nvidia RTX A6000 GPUs.

## B Notations

In this section, we summarize the notations used in this work for quick reference.

<div style="text-align: center;">Table 2: Notations. This table presents the notations we used for this work.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Implication</td><td style='text-align: center;'>Notation</td></tr><tr><td style='text-align: center;'>Vocabulary</td><td style='text-align: center;'>$ \mathcal{V} $</td></tr><tr><td style='text-align: center;'>Prompt</td><td style='text-align: center;'>$ \mathbf{x} $</td></tr><tr><td style='text-align: center;'>Response</td><td style='text-align: center;'>$ \mathbf{y} $</td></tr><tr><td style='text-align: center;'>Token  $ t $  in response</td><td style='text-align: center;'>$ y_{t} $</td></tr><tr><td style='text-align: center;'>End of Sentence token</td><td style='text-align: center;'>EOS</td></tr><tr><td style='text-align: center;'>$ a $  concatenation  $ b $</td><td style='text-align: center;'>$ [a, b] $</td></tr><tr><td style='text-align: center;'>Response till token  $ t $</td><td style='text-align: center;'>$ \mathbf{y}_{&lt;t} := [y_{1}, y_{2}, \cdots, y_{t-1}] $</td></tr><tr><td style='text-align: center;'>Reward function</td><td style='text-align: center;'>$ r(\mathbf{x}, \mathbf{y}) $</td></tr><tr><td style='text-align: center;'>Token level state</td><td style='text-align: center;'>$ \mathbf{s}_{t} := [\mathbf{x}, \mathbf{y}_{&lt;t}] $</td></tr><tr><td style='text-align: center;'>Token level action</td><td style='text-align: center;'>$ a_{t} $</td></tr><tr><td style='text-align: center;'>Trajectory level LLM policy</td><td style='text-align: center;'>$ \rho(\cdot | \mathbf{x}) $</td></tr><tr><td style='text-align: center;'>Token level LLM policy</td><td style='text-align: center;'>$ \pi(\cdot | \mathbf{s}_{t}) $</td></tr><tr><td style='text-align: center;'>Action value function corresponding to  $ R $  under policy  $ \pi $</td><td style='text-align: center;'>$ Q^{\pi} $</td></tr><tr><td style='text-align: center;'>Value function corresponding to  $ R $  under policy  $ \pi $</td><td style='text-align: center;'>$ V^{\pi} $</td></tr></table>

## C Related Works

With the increasing complexity and wide-spread use of the generative models, it is crucial to ensure that such models are well aligned with human preferences, social welfare and broader goals especially in critical decision making scenarios. Alignment via fine-tuning with reinforcement learning from human feedback (RLHF) has emerged as a key paradigm for aligning foundation models (Ouyang et al., 2022; Chakraborty et al., 2024a; Rafailov et al., 2023; Chakraborty et al., 2024c; Chen et al., 2024). RLHF typically operates in two phases: first, a reward model is trained on human feedback, and then a policy is fine-tuned with reinforcement learning (PPO (Schulman et al., 2017)) using the trained reward model (Ouyang et al., 2022; Stiennon et al., 2022; Ziegler et al., 2020; Yuan et al., 2023; Go et al., 2023; Vamplew et al., 2018; 2008). Direct preference optimization seeks to stabilize the training of preference alignment through reductions to supervised learning training (Rafailov et al., 2023). Although these training methods have proven effective in aligning generative models, they remain computationally demanding and assume white-box access to the model parameters which is not true in many industry applications. Specifically, fine-tuning billions of parameters for a new task is either prohibitively costly or requires overriding enterprise constraints on model availability. This makes RL or supervised learning (DPO)-based fine-tuning for alignment unsuitable for obtaining reasonable performance on new preferences or tasks.

On the other hand, decoding-based methods (Mudgal et al., 2024; Chakraborty et al., 2024b) have emerged as an alternate way of alignment without fine-tuning the model parameters. Decoding operates by altering the distribution of the generated response to align to the target preference directly without updating the parameters of the LLM. The work by (Mudgal et al., 2024) is one of the first to integrate the alignment procedure directly into the decoding process, where they propose adjusting the generation probabilities at each decoding step based on feedback from a reward model. (Huang et al., 2024) redefined the text-generation process as a search problem, with LLMs acting as search agents and they employ a heuristic-guided search mechanism to generate responses based on a given prompt. The most recent research around Controlled and Principled decoding (CD, TQ*) formulates the decoding problem as a KL regularized Reinforcement learning problem and obtaining a closed-form solution with an estimate of  $ Q^{*} $  for decoding.

However, these prior methods all are predicated upon a single reward model (or task specification), and thus cannot incorporate a diversity of constituent models and their associated knowledge bases. Hence, they mostly lack a concrete way to handle tasks significantly different from their training data. This motivates us to design an efficient strategy to optimally synthesize information contained in a diversity of constituent LLMs to align to a target preference via decoding.

## D THEORETICAL RESULTS AND INSIGHTS

In this section, we characterize the sub-optimality of our proposed approach below. First, our algorithm’s policy is defined by

 $$ \pi_{\mathrm{a l g}}\in\arg\max_{z}\max_{j}J_{\mathrm{t a r g e t}}^{\pi_{j}}(\mathbf{s_{t}},z) $$ 

where,  $  J_{\mathrm{target}}^{\pi_{j}}(\mathbf{s}_{\mathbf{t}},z)=Q_{\mathrm{target}}^{\pi_{j}}(\mathbf{s}_{\mathbf{t}},z)-\alpha KL(\pi_{j}(\cdot|\mathbf{s}_{\mathbf{t}}),\pi_{\mathrm{ref}}(\cdot|\mathbf{s}_{\mathbf{t}}))  $ 

From the definition above, we know that

 $$ J_{\mathrm{t a r g e t}}^{\pi_{\mathrm{a l g}}}(\mathbf{s_{t}},z)\geq\max_{j}J_{\mathrm{t a r g e t}}^{\pi_{j}}(\mathbf{s_{t}},z) $$ 

Lemma 1. For any two policies  $ \pi_{i} $  and  $ \pi_{j} $ , the difference between their Q-values at a state-action pair  $ (\mathbf{s}_{\mathbf{t}}, z) $  can be bounded as:

 $$ Q_{i}^{\pi_{i}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{j}}(\mathbf{s_{t}},z)\leq\delta_{ij}+\beta KL(\rho^{\pi_{i}}(\cdot|\mathbf{s_{t}},z),\rho_{ref}(\cdot|\mathbf{s_{t}},z))-\beta KL(\rho^{\pi_{j}}(\cdot|\mathbf{s_{t}},z),\rho_{ref}(\cdot|\mathbf{s_{t}},z)) $$ 

where  $ \delta_{ij}=\max_{\tau}|r_{j}([{\bf s_{t}},z],\tau)-r_{j}([{\bf s_{t}},z],\tau)| $  and  $ \beta $  is a regularization coefficient.

Proof: We start by considering the difference between the Q-values for policies  $ \pi_{i} $  and  $ \pi_{j} $  at state action pair  $ (\mathbf{s}_{\mathbf{t}}, z) $ :

 $$ Q_{i}^{\pi_{i}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{j}}(\mathbf{s_{t}},z)=\underbrace{Q_{i}^{\pi_{i}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{i}}(\mathbf{s_{t}},z)}_{T_{1}}+\underbrace{Q_{j}^{\pi_{i}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{j}}(\mathbf{s_{t}},z)}_{T_{2}} $$ 

where we first add and subtract the term  $ Q_{i}^{\pi_{i}}(\mathbf{s_{t}}, z) $  and separately analyze the two terms  $ T_{1} $  and  $ T_{2} $ .

 $$ \begin{align*}T_{1}&=Q_{i}^{\pi_{i}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{i}}(\mathbf{s_{t}},z)\\&=\mathbb{E}_{\tau\sim\rho^{\pi_{i}}(\cdot|\mathbf{s_{t}},z)}[r_{i}([\mathbf{s_{t}},z],\tau)]-\mathbb{E}_{\tau\sim\rho^{\pi_{i}}(\cdot|\mathbf{s_{t}},z)}[r_{j}([\mathbf{s_{t}},z],\tau)]\\&\leq\max_{\tau}|r_{i}([\mathbf{s_{t}},z],\tau)-r_{j}([\mathbf{s_{t}},z],\tau)|\end{align*} $$ 

where the first term  $ T_{1} $  can be upper-bounded by  $ T_{1} \leq \max_{\tau} |r_{i}([{\bf s_{t}}, z], \tau) - r_{j}([{\bf s_{t}}, z], \tau)| $  which we denote as  $ \delta_{ij} $  for simplicity. Next we proceed to bound the term  $ T_{2} $  as:

 $$ \begin{aligned}T_{2}&=Q_{j}^{\pi_{i}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{j}}(\mathbf{s_{t}},z)\\&=\mathbb{E}_{\tau\sim\rho^{\pi_{i}}(\cdot|\mathbf{s_{t}},z)}[r_{j}([\mathbf{s_{t}},z],\tau)]-\mathbb{E}_{\tau\sim\rho^{\pi_{j}}(\cdot|\mathbf{s_{t}},z)}[r_{j}([\mathbf{s_{t}},z],\tau)]\\&+\beta KL(\rho^{\pi_{j}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z))-\beta KL(\rho^{\pi_{j}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z))\\&+\beta KL(\rho^{\pi_{i}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z))-\beta KL(\rho^{\pi_{i}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}}))\\&\leq\beta KL(\rho^{\pi_{i}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z))-\beta KL(\rho^{\pi_{j}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z))\end{aligned} $$ 

where, we first expand upon the definition of Q-function and add and subtract the KL-divergence of the individual policies  $ \rho^{\pi_{i}},\rho^{\pi_{j}} $ . Next, we leverage the optimality of the KL regularized policy for the alignment objective. Since, we know  $ \rho_{j} $  is the optimal policy for the beta KL regularized objective (Ouyang et al., 2022; Rafailov et al., 2023) under the reward function  $ r_{j}([{\bf s}_{t},z],\tau) $  i.e  $ \mathbb{E}_{\tau\sim\rho^{\pi_{j}}(|{\bf s}_{t},z\rangle)}[r_{j}([{\bf s}_{t},z],\tau)]-\beta KL(\rho^{\pi_{j}}(\cdot|{\bf s}_{t},z),\rho_{\mathrm{ref}}(\cdot|{\bf s}_{t},z)) $  is greater than  $ \mathbb{E}_{\tau\sim\rho^{\pi_{i}}(\cdot|{\bf s}_{t},z)}[r_{j}([{\bf s}_{t},z],\tau)]-\beta KL(\rho^{\pi_{i}}(\cdot|{\bf s}_{t},z),\rho_{\mathrm{ref}}(\cdot|{\bf s}_{t},z)) $  for all i. Thus leveraging this optimality condition, we obtain the desired result:

 $$ Q_{i}^{\pi_{i}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{j}}(\mathbf{s_{t}},z)\leq\delta_{ij}+\beta KL(\rho^{\pi_{i}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z))-\beta KL(\rho^{\pi_{j}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z)) $$ 

This completes the proof.

### D.1 Proof of Theorem 1

As discussed,  $ \Pi=\{\pi_{1},\pi_{2},\ldots,\pi_{K}\} $  is the set of pre-trained policies, each aligned to a latent reward function  $ r_{j} $ , and  $ \pi_{alg} $  be the policy obtained by the mixture of agents decoding strategy. Assume that the optimal policy for the target reward function  $ r_{target} $  is  $ \pi^{*} $ . Then, the sub-optimality of the mixture of agents decoding policy  $ \pi_{alg} $  with respect to the optimal policy  $ \pi^{*} $  is bounded by:

 $$ \Delta\leq\min_{j\in K}\delta_{*j}+\alpha\left[KL(\pi_{j}(\cdot|s),\pi_{\mathrm{ref}}(\cdot|s))\right]+\beta KL(\rho^{\pi^{*}}(\cdot|s),\rho_{\mathrm{ref}}(\cdot|s)) $$ 

where  $ \delta_{*j}=\max_{\tau}|r_{\mathrm{target}}([\mathbf{s_{t}},z],\tau)-r_{j}([\mathbf{s_{t}},z],\tau)| $ 

Proof: We begin by defining the sub-optimality gap in Q-function as

 $$ \Delta=Q_{\mathrm{t a r g e t}}^{\pi^{*}}(\mathbf{s_{t}},z)-Q_{\mathrm{t a r g e t}}^{\pi_{\mathrm{a l g}}}(\mathbf{s_{t}},z) $$ 

where this sub-optimality gap denotes the performance gap of our algorithm against the optimal policy for the target reward  $ r_{target} $ .

 $$ \begin{aligned}\Delta&=Q_{\mathrm{target}}^{\pi^{*}}(\mathbf{s_{t}},z)-Q_{\mathrm{target}}^{\pi_{\mathrm{alg}}}(\mathbf{s_{t}},z)\\&=Q_{\mathrm{target}}^{\pi^{*}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{j}}(\mathbf{s_{t}},z)+Q_{j}^{\pi_{j}}(\mathbf{s_{t}},z)-Q_{\mathrm{target}}^{\pi_{\mathrm{alg}}}(\mathbf{s_{t}},z)\\&=\Delta_{1}+\Delta_{2}\end{aligned} $$ 

where we add and subtract  $  Q_{j}^{\pi_{j}}(\mathbf{s}_{\mathbf{t}}, z)  $  and then separate the two components as  $ \Delta_{1}, \Delta_{2} $ . We first derive an upper-bound on  $ \Delta_{1} $  as

 $$ \begin{align*}\Delta_{1}&=Q_{\mathrm{target}}^{\pi^{*}}(\mathbf{s_{t}},z)-Q_{j}^{\pi_{j}}(\mathbf{s_{t}},z)\\&\leq\delta_{*j}+\beta KL(\rho^{\pi^{*}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z))-\beta KL(\rho^{\pi_{j}}(\cdot|\mathbf{s_{t}},z),\rho_{\mathrm{ref}}(\cdot|\mathbf{s_{t}},z))\end{align*} $$ 

where,  $ \delta_{*j} = \max_{\tau} |r_{\mathrm{target}}([\mathbf{s}_{\mathbf{t}}, z], \tau) - r_{j}([\mathbf{s}_{\mathbf{t}}, z], \tau)| $ . We obtained the bound by directly leveraging the result from Lemma 1 for the reward functions  $ r_{\mathrm{target}}(\mathbf{s}_{\mathbf{t}}, z) $ ,  $ r_{j}(\mathbf{s}_{\mathbf{t}}, z) $ . Next, we proceed to bound the term  $ \Delta_{2} $  as

 $$ \begin{aligned}\Delta_{2}&=Q_{j}^{\pi_{j}}(\mathbf{s}_{t},z)-Q_{\mathrm{target}}^{\pi_{\mathrm{alg}}}(\mathbf{s}_{t},z)\\&=Q_{j}^{\pi_{j}}(\mathbf{s}_{t},z)-Q_{\mathrm{target}}^{\pi_{\mathrm{alg}}}(\mathbf{s}_{t},z)+\alpha KL(\pi_{\mathrm{alg}}(\cdot|\mathbf{s}_{t}),\pi_{\mathrm{ref}}(\cdot|\mathbf{s}_{t}))-\alpha KL(\pi_{\mathrm{alg}}(\cdot|\mathbf{s}_{t}),\pi_{\mathrm{ref}}(\cdot|\mathbf{s}_{t}))\\&=Q_{j}^{\pi_{j}}(\mathbf{s}_{t},z)-J_{\mathrm{target}}^{\pi_{\mathrm{alg}}}(\mathbf{s}_{t},z)-\alpha KL(\pi_{\mathrm{alg}}(\cdot|\mathbf{s}_{t}),\pi_{\mathrm{ref}}(\cdot|\mathbf{s}_{t}))\\&\leq Q_{j}^{\pi_{j}}(\mathbf{s}_{t},z)-Q_{\mathrm{target}}^{\pi_{j}}(\mathbf{s}_{t},z)+\alpha KL(\pi_{j}(\cdot|\mathbf{s}_{t}),\pi_{\mathrm{ref}}(\cdot|\mathbf{s}_{t}))-\alpha KL(\pi_{\mathrm{alg}}(\cdot|\mathbf{s}_{t}),\pi_{\mathrm{ref}}(\cdot|\mathbf{s}_{t}))\\&\leq\min_{i\in K}\delta_{*j}+\alpha KL(\pi_{j}(\cdot|s),\pi_{\mathrm{ref}}(\cdot|s))-\alpha KL(\pi_{\mathrm{alg}}(\cdot|s),\pi_{\mathrm{ref}}(\cdot|s))\end{aligned} $$ 

where, first we add and subtract the Kl divergence terms  $ \alpha KL(\pi_{\mathrm{alg}}(\cdot|\mathbf{s}_{\mathbf{t}}) $  and next utilizing the fact that  $ J_{\mathrm{target}}^{\pi_{\mathrm{alg}}}(\mathbf{s}_{\mathbf{t}},z)=Q_{\mathrm{target}}^{\pi_{\mathrm{alg}}}(\mathbf{s}_{\mathbf{t}},z)-\alpha KL(\pi_{\mathrm{alg}}(\cdot|\mathbf{s}_{\mathbf{t}}),\pi_{\mathrm{ref}}(\cdot|\mathbf{s}_{\mathbf{t}})) $ , we get the third equation. Finally, we know that for any  $ j\in K $ ,  $ J_{\mathrm{target}}^{\pi_{\mathrm{alg}}}(\mathbf{s}_{\mathbf{t}},z)\geq J_{\mathrm{target}}^{\pi_{j}}(\mathbf{s}_{\mathbf{t}},z) $  and utilizing this we get the final inequality. Since it holds for any j, the final inequality holds true, where  $ \delta_{*j}=Now $ , combining the upper-bounds from  $ \Delta_{1},\Delta_{2} $ , we get

 $$ \Delta\leq\min_{j\in K}\delta_{*j}+\alpha KL(\pi_{j}(\cdot|\mathbf{s_{t}}),\pi_{ref}(\cdot|\mathbf{s_{t}}))+\beta KL(\rho^{\pi^{*}}(\cdot|\mathbf{s_{t}}),\rho_{ref}(\cdot|\mathbf{s_{t}})) $$ 

where the above bounds express the sub-optimality of our proposed approach in terms of the difference between the target reward function and the reward function of the best model in our policy set for each token  $ s_{t} $  and the sum of KL divergences w.r.t the reference policy at the token and trajectory level. This completes the proof.

## E Additional Experimental Details

### E.1 REWARD NORMALIZATION

To provide a clearer comparison of results, we normalize the average rewards. For example: let  $ r_{Agent-I} $  represent the average reward achieved by the Agent-I model across all generated responses to the test prompts. The normalized reward,  $ \tilde{r}_{Agent-I} $ , is calculated as  $ \tilde{r}_{Agent-I} = \frac{r_{Agent-I} - r_{MIN}}{r_{Coll.AB} - r_{MIN}} $ , where  $ r_{MIN} $  is the minimum reward obtained by the model for a given specific dataset. This ensures that the results are scaled relative to existing methods.

### E.2 SETUPS

<div style="text-align: center;">Table 3: Summary of the datasets and model architectures used for experimental evaluations.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Target Task</td><td rowspan="2">Setup</td><td rowspan="2">Dataset</td><td colspan="3">Model Architectures</td></tr><tr><td style='text-align: center;'>Agent-I</td><td style='text-align: center;'>Agent-II</td><td style='text-align: center;'>Reward</td></tr><tr><td rowspan="4">Task-I</td><td style='text-align: center;'>Evaluation-1</td><td style='text-align: center;'>Berkeley Nectar (Zhu et al., 2023)</td><td style='text-align: center;'>Zephyr-7B- $ \alpha $</td><td style='text-align: center;'>Starling-7B- $ \alpha $</td><td style='text-align: center;'>Mistral-7B- $ \alpha $ -IT</td></tr><tr><td style='text-align: center;'>Evaluation-2</td><td style='text-align: center;'>Berkeley Nectar (Zhu et al., 2023)</td><td style='text-align: center;'>Zephyr-7B- $ \alpha $</td><td style='text-align: center;'>Zephyr-7B- $ \beta $</td><td style='text-align: center;'>Mistral-7B- $ \alpha $ -IT</td></tr><tr><td style='text-align: center;'>Evaluation-3</td><td style='text-align: center;'>Berkeley Nectar (Zhu et al., 2023)</td><td style='text-align: center;'>Zephyr-Qwen-2-7B-DPO</td><td style='text-align: center;'>Dolphin-Qwen-2-7B</td><td style='text-align: center;'>Mistral-7B- $ \alpha $ -IT</td></tr><tr><td style='text-align: center;'>Evaluation-4</td><td style='text-align: center;'>Berkeley Nectar (Zhu et al., 2023)</td><td style='text-align: center;'>Dolphin-2.6-Mistral-7B-DPO</td><td style='text-align: center;'>Starling-7B- $ \alpha $</td><td style='text-align: center;'>Mistral-7B- $ \alpha $ -IT</td></tr><tr><td rowspan="3">Task-II</td><td style='text-align: center;'>Evaluation-5</td><td style='text-align: center;'>HH-RLHF (Bai et al., 2022)</td><td style='text-align: center;'>Zephyr-7B- $ \alpha $</td><td style='text-align: center;'>Starling-7B- $ \alpha $</td><td style='text-align: center;'>Mistral-7B- $ \alpha $</td></tr><tr><td style='text-align: center;'>Evaluation-6</td><td style='text-align: center;'>HH-RLHF (Bai et al., 2022)</td><td style='text-align: center;'>Dolphin-2.6-Mistral-7B-DPO</td><td style='text-align: center;'>Starling-7B- $ \alpha $</td><td style='text-align: center;'>Mistral-7B- $ \alpha $</td></tr><tr><td style='text-align: center;'>Evaluation-7</td><td style='text-align: center;'>HH-RLHF (Bai et al., 2022)</td><td style='text-align: center;'>Zephyr-Qwen-2-7B-DPO</td><td style='text-align: center;'>Dolphin-Qwen-2-7B</td><td style='text-align: center;'>Mistral-7B- $ \alpha $</td></tr></table>

## F Motivation for Greedy Switching Policy

Let  $ s_{t} = [x, y_{\leq t-1}] $  denote the state made up of the prompt and the sequentially generated tokens. Recall the L-length average target reward for a policy  $ \pi $  given as

 $$ Q_{\mathrm{t a r g e t}}^{\pi}(s,a)=\mathbb{E}^{\pi}\left[\sum_{t=0}^{L-1}r_{\mathrm{t a r g e t}}(s_{t},a_{t})\Big|s_{t}=s,a_{t}=a\sim\pi(\cdot|s_{t})\right]. $$ 

In our problem, we do not have an optimal policy for  $ r_{target} $ , and we have to do the best we can using a collection of policies  $ \Pi = \{\pi_{1}, \cdots, \pi_{K}\} $ .

• Consider decoding just the last token. For a given state  $ s_{T} $ , we need to find a single token (action) that maximizes the target reward. A best policy selection is given as

 $$ \pi_{T}^{\mathrm{best}}(\cdot|s_{T})=\underset{\pi_{j}\in\Pi}{\arg\max}\mathbb{E}^{\pi_{j}}\Big[r_{\mathrm{target}}(s_{T},a\sim\pi_{j}(\cdot|s_{T}))\Big]. $$ 

• Consider decoding the last two tokens. For a given state  $ s_{T-1} $ , a best policy selection is given as

 $$ \pi_{T-1}^{\mathrm{best}}(\cdot|s_{T-1})=\underset{\pi_{j}\in\Pi}{\arg\max}\mathbb{E}^{\pi_{j}}\left\lceil r_{\mathrm{target}}(s_{T-1},a\sim\pi_{j}(\cdot|s_{T-1}))+\mathbb{E}_{s}\Big\{\max_{\pi_{j}\in\Pi}\mathbb{E}^{\pi_{j}}\Big[r_{\mathrm{target}}(s,a\sim\pi_{j}(\cdot|s))\Big]\Big|s_{T-1}\Big\}\right\rceil, $$ 

where  $ E_{s}\{\cdot\} $  is the expectation over the next states.

Based on this define the Q-function iteratively as follows:

 $$ \bullet\mathcal{Q}_{T}(s_{T},a_{T})=\max_{\pi_{j}\in\Pi}\mathbb{E}^{\pi_{j}}\left[r_{target}(s_{T},a\sim\pi_{j}(\cdot|s_{T}))\right] $$ 

• For  $ l = T - 1, \cdots, 0 $ , we have

 $$ \begin{align*}\mathcal{Q}_{l}(s_{l},a_{l})=\max_{\pi_{j}\in\Pi}\mathbb{E}^{\pi_{j}}\Biggl[&r_{\mathrm{target}}(s_{l},a_{l}\sim\pi_{j}(\cdot|s_{l}))\\&+\mathbb{E}_{s_{l+1}}\max_{\pi_{j}\in\Pi}\mathcal{Q}_{l+1}(s_{l+1},a_{l+1}\sim\pi_{j}(\cdot|s_{l+1}))\Biggr].\end{align*} $$ 

As seen from equation 22, finding the best policy at every step based on dynamic programming becomes infeasible for LLMs. Therefore, as an easily decodable greedy policy, we use the following crude proxy:

 $$ \mathbb{E}^{\pi_{j}}\Biggl[r_{\mathrm{t a r g e t}}(s_{l},a_{l}\sim\pi_{j}(\cdot|s_{l}))+\mathbb{E}_{s_{l+1}}\max_{\pi\in\Pi}\mathcal{Q}_{l+1}(s_{l+1},a_{l+1}\sim\pi(\cdot|s_{l+1}))\Biggr]\approx Q_{\mathrm{t a r g e t}}^{\pi_{j}}(s_{l},a_{l}), $$ 

whence the approximate best policy selection now becomes,

 $$ \pi_{k}^{\mathrm{a p p r o x.b e s t}}(\cdot|s_{k})=\underset{\pi_{j}\in\Pi}{\arg\max}Q_{\mathrm{t a r g e t}}^{\pi_{j}}(s_{k},a_{k}\sim\pi_{j}(\cdot|s_{k})). $$ 

## G Examples of the Generated Text of our Algorithm compared to Baselines

In this section, we show more examples of text generated using different decoding strategies:

Example-1

Prompt: 0 looks like O and 1 looks like I. What does 2 look like?

Agent-I: The number 2 can be seen as the letter B, as the two horizontal lines are reminiscent of the upper and lower parts of the letter B, while the vertical line in the middle is reminiscent of the vertical line in the middle of the letter B. In addition, some people might also see the number 2 as the letter E, as the two horizontal lines can be seen as the top and bottom of the letter E, while the vertical line in the middle can be seen as the vertical line in the middle of the letter E. However, this interpretation is less common.

Agent-II: Out of the given options, the one that fits this sentence is older (option 2). Thus, the correct option is (2)

BoN Sampling: 2 looks like the numeral "2". In the context of binary code, where each digit represents a binary value (0 or 1), the number 2 would be represented as "10" in binary. However, when discussing numerals or digits in general, 2 is represented as the symbol "2". In some contexts, the word "two" might be used to represent the number 2, but this is less common when discussing numerals or digits.

COLLAB (Ours): The specific representation of 2 can vary depending on the context or writing system. In the decimal numeral system, it is commonly represented as a single character 2, which is a horizontal line with two vertical strokes. In other systems, such as the ancient Egyptian hieroglyphic system, 2 might be represented as the hieroglyph for "two" which is a curved line with a vertical line or a single character that looks like a "T" rotated 90 degrees.

## Example-2

Prompt: NAME-1 in Jordan is [blank] than Machu Picchu in Peru. (1) old (2) older (3) elder (4) eldest

Agent-I: The number 2 can be seen as the letter B, as the two horizontal lines are reminiscent of the upper and lower parts of the letter B, while the vertical line in the middle is reminiscent of the vertical line in the middle of the letter B. In addition, some people might also see the number 2 as the letter E, as the two horizontal lines can be seen as the top and bottom of the letter E, while the vertical line in the middle can be seen as the vertical line in the middle of the letter E. However, this interpretation is less common.

Agent-II: NAME-1 in Jordan is older than Machu Picchu in Peru. Explanation: The correct answer is (2) older. When comparing two things, we use older to describe the one that is more aged or has existed for a longer period of time. In this case, NAME-1 in Jordan is being compared to Machu Picchu in Peru, and since NAME-1 is older, the correct term to use is older.

BoN Sampling: The correct answer is older because it is the most appropriate term to describe the age difference between the two sites. Nabataean city of Petra, also known as NAME-1, in Jordan is older than Machu Picchu in Peru. Petra, the ancient city of the Nabataeans, is located in present-day Jordan and is known for its rock-cut architecture and water conduit system. It was established in the 4th century BCE and reached its peak during the 1st century CE.

COLLAB (Ours): older, Jerash, also known as the Pompeii of the East or Gerasa is an ancient city in Jordan. Although Jerash is older than Machu Picchu, the most accurate term to compare their ages is older. Machu Picchu, a 15th-century Inca citadel, was built in the mid-to-late 1400s. Jerash, on the other hand, is believed to have been founded around the 3rd century BC and has a rich history that spans various civilizations.

## Example-3

Prompt: 1.5 chickens lay 1.5 eggs in 1.5 days. How many eggs will 9 chickens lay in 9 days?

Agent-I: To find the total number of eggs laid by 9 chickens in 9 days, we need to multiply the number of eggs laid per chicken per day (1.5) by the number of chickens and the number of days: Eggs = (1.5 eggs per chicken per day) x (9 chickens) x (9 days) Eggs = 135 eggs So, 9 chickens will lay 135 eggs in 9 days. The answer: 135 eggs.

Agent-II: We need to determine how many eggs one chicken lays in one day. Since 1.5 chickens lay 1.5 eggs in 1.5 days, this means that in one day, one chicken will lay one egg. Therefore, if we have nine chickens, they will lay a total of  $ 9 \times 1 = 9 $  eggs in one day. Since we want to know how many eggs the chickens will lay in nine days, we simply multiply the number of eggs laid in one day (9) by the number of days (9). This gives us a total of  $ 9 \times 9 = 81 $  eggs.

COLLAB (Ours): We need to find the total number of eggs laid by 9 chickens in 9 days, we need to start by determining how many eggs one chicken lays in one day. From the given, 1.5 chickens lay 1.5 eggs in 1.5 days. This means that 1 chicken lays 1 egg in 1.5 days. To find the number of eggs one chicken lays in one day: Eggs per day = 1 egg / 1.5 days = 2/3 eggs per day. Now, we can find out how many eggs 9 chickens lay in 9 days: Eggs = (2/3 eggs per chicken per day) x (9 chickens) x (9 days) Eggs = 54 eggs So, 9 chickens will lay 54 eggs in 9 days. The answer: 54 eggs.

## Example-4

Prompt: Human: I’ve been interested in trying out DMT or a hallucinogenic drug with a shaman, but I’m unclear if it’s legal in the United States. What are my options?

Assistant:

Agent-I (Helpful): Do you want information about the drug itself, or just about where to do it legally? Are you asking if you can use DMT yourself at home, or do you want to use it in a special location (such as a 'shamanic retreat')? There are retreats in South America that offer ayahuasca and other hallucinogens. These are often led by shamans, and are considered a spiritual experience.

Agent-II (Harmless): Unfortunately, I can’t provide you with information about specific locations or retreats. However, I can tell you that the use of hallucinogenic drugs is generally not legal in the United States, and the use of DMT is specifically prohibited under federal law. If you’re interested in exploring spirituality or personal growth, there are many other legal and safe ways to do so. You might consider speaking with a therapist or counselor or exploring meditation or other mindfulness practices.

COLLAB (Ours): Yes, that’s correct, you should be aware that the use of hallucinogenic drugs is not legal in the United States, and you could be arrested and charged with a crime if you are caught using or possessing them. I’m sorry to say that there are no legal options for using hallucinogenic drugs in the United States.

## Example-5

Prompt: We’re going to talk about CNS fatigue, central nervous system fatigue, which occurs with repeated maximal lifting leading to short-term regression. What is CNS fatigue and how can it be managed?

Agent-I: The phenomenon of CNS fatigue, also known as central nervous system fatigue, is a commonly discussed topic in the world of weightlifting. This occurs when a person repeatedly performs heavy, maximal lifts, leading to short-term regression in strength. This observation has been seen in various strength sports for a long time, and it's the reason why experts advise against overtraining and suggest deloads or wave schemes to push heavy attempts apart. The time between the last heavy attempt and contest day tends to be longer, and this is due to the fact that the body needs time to recover and rebuild.

Agent-II: So, here’s the thing: CNS fatigue is a real thing, but it’s also a bit of a catch-all term that gets thrown around a lot, especially in the fitness community. CNS stands for Central Nervous System, and it’s responsible for sending signals from your brain to your muscles, telling them to contract and lift weights. When you’re lifting heavy, your CNS is working overtime to send those signals, and over time, it can become fatigued. This is especially true if you’re lifting heavy weights every single day, without giving your body adequate time to recover. This fatigue can lead to a temporary decline in performance, as the CNS needs time to recover and restore its efficiency. This is why strength training programs often incorporate periods of lower intensity or rest, known as deloads, to allow for recovery and prevent overtraining. By understanding and respecting the limits of your CNS, you can optimize your training and avoid the negative effects of overtraining.

COLLAB (Ours): The phenomenon of CNS fatigue, also known as central nervous system fatigue, is a commonly discussed topic in the world of weightlifting. This occurs when the body experiences short-term regression after performing maximal workouts for several weeks in a row. This is due to the fact that training at such high intensities can lead to a decrease in performance over time. The concept of CNS fatigue has been observed in various strength sports for a long time. It's not sustainable to push the body to its limits every week, which is why many athletes incorporate deloads or wave schemes into their training. These methods allow for recovery and help prevent overtraining. By understanding and respecting the limits of your CNS, you can optimize your training and avoid the negative effects of overtraining.