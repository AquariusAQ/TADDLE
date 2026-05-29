# MULTI-STEP PREFERENCE OPTIMIZATION VIA TWOplayer Markov Games

Anonymous authors

Paper under double-blind review

## ABSTRACT

Reinforcement Learning from Human Feedback (RLHF) has been highly successful in aligning large language models with human preferences. While prevalent methods like DPO have demonstrated strong performance, they frame interactions with the language model as a bandit problem, which limits their applicability in real-world scenarios where multi-turn conversations are common. Additionally, DPO relies on the Bradley-Terry model assumption, which does not adequately capture the non-transitive nature of human preferences. In this paper, we address these challenges by modeling the alignment problem as a two-player constant-sum Markov game, where each player seeks to maximize their winning rate against the other across all steps of the conversation. Our approach Multi-step Preference Optimization (MPO) is built upon the natural actor-critic framework (Peters & Schaal, 2008). We further develop OMPO based on the optimistic online gradient descent algorithm (Rakhlin & Sridharan, 2013; Joulani et al., 2017). Theoretically, we provide a rigorous analysis for both algorithms on convergence and show that OMPO requires  $ \mathcal{O}(\epsilon^{-1}) $  policy updates to converge to an  $ \epsilon $ -approximate Nash equilibrium. We also validate the effectiveness of our method through experiments on the multi-turn conversations dataset in MT-bench-101.

## 1 INTRODUCTION

In recent years, the integration of large-language models (LLMs) (Brown et al., 2020; Achiam et al., 2023; Team et al., 2023) into various applications has highlighted the need for advanced preference alignment methods (Ziegler et al., 2019; Stiennon et al., 2020; Bai et al., 2022; Ouyang et al., 2022; Rafailov et al., 2023). As models increasingly engage in complex decision making or reasoning scenarios, e.g., GPT-4o and o1 $ ^{1} $ , the ability to align their outputs with user preferences has received more attention. However, existing works on reinforcement learning from human feedback (RLHF) focus mostly on one-step preference (Rafailov et al., 2023; Meng et al., 2024; Munos et al., 2024; Azar et al., 2024; Wu et al., 2024; Zhang et al., 2024), which neglects indispensable intermediate preferences within the answer and limits the model's alignment ability. For example, in multi-round conversations, alignment must occur at each turn to meet user needs. Similarly, in mathematical reasoning with chain-of-thought prompting, step-by-step validation is essential to ensure accuracy in the final result. The reliance on final-output feedback in most existing RLHF methods (Wang et al., 2023; Shani et al., 2024) neglects these intermediate steps, highlighting the need for multi-step preference optimization to enhance alignment capabilities.

Meanwhile, earlier alignment methods e.g., DPO and its variants step-DPO (Lai et al., 2024; Lu et al., 2024), typically model the pairwise preference by the Bradley-Terry model (Bradley & Terry, 1952), which assigns a score for each answer based on its preference. This assumption of the model cannot capture the non-transitive preference, which is often observed in the averaged human preferences from the population (Tversky, 1969; Gardner, 1970). While a recent line of work has modeled the alignment process under the framework of general preference (Azar et al., 2024; Munos et al., 2024; Wu et al., 2024; Rosset et al., 2024), and thus bypasses the BT model assumption, the challenge of multi-step preference optimization remains underexplored.

In this paper, we first address this gap by formulating multi-step general preference optimization within the framework of two-player Markov games (Shapley, 1953), where each player seeks to

maximize their winning rate against the other across all steps of the conversation. Next, we introduce Multi-step Preference Optimization (MPO) drawing on insights from the natural actor-critic framework (Peters & Schaal, 2008). We further develop OMPO which leverages the optimistic online gradient descent algorithm and benefits from improved theoretical guarantees (Rakhlin & Sridharan, 2013; Joulani et al., 2017). Theoretically, we provide rigorous analysis for both algorithms on the convergence to Nash equilibrium. Empirically, we demonstrate the effectiveness of our approach through experiments on multi-turn conversation datasets, such as MT-bench-101. We firmly believe that our framework and approach can enhance the responsiveness of LLMs to user feedback.

Based on our discussions above, we summarize the contributions as follows:

• We formulate multi-step preference optimization as a two-player partially observable Markov game. Unlike Wang et al. (2023); Swamy et al. (2024); Shani et al. (2024) who focus on the preference feedback at the final state, we assume that the preference signal is received at each step. Such feedback allows the model to better identify which steps are correct or erroneous, potentially enhancing learning efficiency and accuracy.

• We propose Multi-step Preference Optimization (MPO) based on the natural actor-critic framework and Optimistic Multi-step Preference Optimization (OMPO), built upon the optimistic online gradient descent. Theoretically, we show that OMPO requires  $ \mathcal{O}(\epsilon^{-1}) $  policy updates to converge to an  $ \epsilon $ -approximate Nash equilibrium, compared to  $ \mathcal{O}(\epsilon^{-2}) $  by the algorithms provided in Wang et al. (2023); Swamy et al. (2024); Shani et al. (2024). Our result cannot be trivially extended by Alacaoglu et al. (2022) due to the partially observable nature of Markov game. Interestingly, we bypass this difficulty by deriving our OMPO that parameterizes the game over occupancy measures.

• We provide practical implementations of both MPO and OMPO for LLM alignment. Numerical results show that the proposed methods achieve considerable improvement on multi-turn conversation datasets, such as MT-bench-101, compared to the multi-step variant of DPO.

The remaining part of this paper is organized as follows: Sec. 2 provides a comprehensive review and discussion of related work. In Sec. 3, we introduce the problem setting for the investigated multistep RLHF. Sec. 4.1 and Sec. 4.2 introduce the proposed MPO and OMPO and provide a theoretical convergence analysis. Experimental results are present in Sec. 5. Conclusion, limitation, and future work are discussed in Sec. 6.

## 2 RELATED WORK

RLHF under Bradley-Terry model. Over the years, significant strides have been made towards developing RLHF algorithms from various perspectives under the Bradley-Terry model. Bradley & Terry (1952). Earlier RLHF pipelines usually included supervised fine-tuning, learning a reward model, and reinforcement learning optimization with PPO (Ziegler et al., 2019; Stiennon et al., 2020; Bai et al., 2022; Ouyang et al., 2022). Due to the instability and scaling issues of such a pipeline, direct alignment methods such as DPO have been proposed to bypass the training of the reward model (Rafailov et al., 2023). Several follow-up methods, such as generalized preference optimization (GPO, Tang et al., 2024), use offline preference data to directly optimize pairwise preferences against a fixed opponent. A number of works have proposed reference-model-free method (Meng et al., 2024; Hong et al., 2024). In Meng et al. (2024), the impact of sequence length is mitigated by averaging the likelihood over the length of the sequence. In the multi-step scenario, several multi-step variants of DPO are introduced in the math reasoning task. Lu et al. (2024) initiate from an intermediate step in a correct reasoning process and increase the temperature to produce a flawed reasoning path leading to an incorrect answer. Meanwhile, Lai et al. (2024) leverage GPT-4 to detect the first incorrect step in a multi-step reasoning trajectory, then regenerate from that point to obtain the correct path. Together, these serve as the pair of samples for DPO.

RLHF under general preferences. The reward model in the Bradley-Terry model inherently implies transitivity in preferences. However, human preferences, especially the resulting averaged human preferences from populations, are usually nontransitive (Tversky, 1969; Gardner, 1970). To this end, Azar et al. (2024) outline a general framework for RLHF starting from general preference optimization and shows that DPO is a special case with the assumption of Bradley-Terry model. They further proposed IPO without such an assumption. Subsequently, Munos et al. (2024) try to solve the alignment of non-transitive general preferences using two-player nash learning in a bandit

setting. In their work, preferences are regularized through KL divergence to a reference policy, and they prove the convergence of the last iterative. In Swamy et al. (2024), multi-step alignment is considered while preference signals are only applied at the final step. Swamy et al. (2024) do not demonstrate the effectiveness of this framework in large language models. Wu et al. (2024) propose SPPO, studying bandit alignment under general preferences. They introduce a novel loss function that increases the log-likelihood of the selected response while decreasing that of the rejected response, in contrast to DPO. Rosset et al. (2024) start with the nash learning framework and propose Online DPO, which is an iterative version of DPO. Wang et al. (2023) provide theoretical analysis on multi-step RLHF under general preference while practice application is not explored. In Wang et al. (2023), the preference signal is given for the entire trajectory of an MDP while in this paper it is step-wise. Shani et al. (2024) study multi-step alignment under general preferences. However, unlike their approach where only preferences at the final states are considered, our work is built on a two-player Markov game which assumes that human preference is received at each step rather than only at the final step. Additionally, we leverage the optimistic online gradient descent to achieve a better convergence rate than Wang et al. (2023); Shani et al. (2024), and utilize Monte Carlo estimation with a small-scale pairwise reward model, avoiding the need for an additional function approximator for the critic network.

Two-player Markov game & optimistic online gradient descent. Two-player Markov games have been widely studied since the seminal work (Shapley, 1953). Particularly relevant to our work is the research line on policy gradient algorithms for two-player Markov games such as Daskalakis et al. (2020); Wei et al. (2021); Alacaoglu et al. (2022). Our OMPO is strictly related to the idea of optimistic online gradient descent (Popov, 1980; Chiang et al., 2012; Rakhlin & Sridharan, 2013) originally proposed in online learning to achieve small regret in case of slow varying loss sequences. Our update that uses only one projection per update was proposed in Joulani et al. (2017). The name of our method is due to a similar algorithm introduced in the context of variational inequalities by Malitsky & Tam (2020).

## 3 MULTI-STEP RLHF AS TWO-PLAYER MARKOV GAMES

### 3.1 NOTATION

We define the prompt to the language model as x and the answer from the language model as a. For a multi-turn conversation with turn H, the prompts and the answers are denoted by  $ x_{h} $  and  $ a_{h}, \forall h \in [H] $ . The concatenation of a prompt x and an answer a is denoted by  $ [x, a] $  and can be generalized to the concatenation of multiple prompts and answers, e.g.,  $ [x_{1}, a_{1}, \ldots, x_{H}, a_{H}] $ . For any two sentences, e.g.,  $ [x, a] $  and  $ [x', a'] $ , we define a preference oracle as  $ o([x, a] \succ [x', a']) \in \{0, 1\} $ , which can provide preference feedback with 0-1 scores, where 1 means the conversation  $ [x, a] $  is preferred and 0 otherwise. We denote  $ \mathbb{P}([x, a] \succ [x', a》） = \mathbb{E}[o([x, a] \succ [x', a》）] $  as the probability that the conversation  $ [x, a] $  is preferred over  $ [x', a'] $ . Moreover, we have  $ \mathbb{P}([x, a] \succ [x', a》） = 1 - \mathbb{P}([x', a'] \succ [x, a]) $ . An autoregressive language model is denoted by  $ \pi(a|x) $  which receives input x and generates answer a. We denote the KL divergence of two probability distributions p and q by  $ D(p||q) $ . The Bregman Divergences between two points are denoted by  $ \mathbb{D}(p||q) $ . The sigmoid function is defined by  $ \sigma(z) := \frac{1}{1 + e^{-z}} $ . Detailed definitions for the notations are summarized in Appx. A.

### 3.2 PROBLEM FORMULATION OF MULTI-STEP RLHF

In this section, we introduce the problem setting for multi-step RLHF and we defer the preliminaries on single-step RLHF to Appx. B. Specifically, we can cast the multi-step alignment process as a finite-horizon Markov Decision Process (MDP). We define  $ s_{h} = [x_{1}, a_{1}, \ldots, x_{h-1}, a_{h-1}, x_{h}] $  as the state at h > 1. We define the action  $ a_{h} $  as the answer given  $ s_{h} $ . Particularly, we have  $ s_{1} = x_{1} $ . The prompt in the next state is sampled under the transition  $ x_{h+1} \sim f(\cdot | s_{h}, a_{h}) $ , which is equivalent to  $ s_{h+1} \sim f(\cdot | s_{h}, a_{h}) $ . The equivalence comes from the fact  $ s_{h+1} = [s_{h}, a_{h}, x_{h+1}] $  by using the concatenation operator between sentences. The terminal state is  $ s_{H+1} $ . Our setting covers a number of alignment problems, and we list some examples below.

Example 1 (Single-step alignment). In single-step alignment, a language model receives one prompt and outputs one answer. Our framework covers the single-step alignment by dissecting the answer into single tokens. Specifically, we set  $ x_{1} $  as the prompt,  $ x_{2}, \ldots, x_{H+1} $  as empty sentences,

and the answer  $ a_{h} $  at each turn consists of only one token. Then the horizon H is the number of tokens in the answer. The transition between each state is deterministic.

Example 2 (Chain-of-thought reasoning alignment). In the chain-of-thought reasoning, the horizon H denotes the number of reasoning steps, where  $ x_{1} $  is the initial prompt and  $ x_{2}, \ldots, x_{H+1} $  are empty. Each  $ a_{h} $  corresponds to a reasoning step. The transition between each state is deterministic.

Example 3 (Mutli-turn conversation alignment). In multi-turn conversation, the horizon H denotes the total number of turns in the conversation. In the h-th turn,  $ x_{h} $  is the prompt, and  $ a_{h} $  is the answer. The prompt in the terminal state,  $ x_{H+1} $ , is an empty sentence. The transition between each state can be deterministic or stochastic.

Next, we define the pair-wise reward function of two state-action pairs as the preference of two trajectories:

 $$ r(s_{h},a_{h},s_{h}^{\prime},a_{h}^{\prime})=\mathbb{P}([s_{h},a_{h}]\succ[s_{h}^{\prime},a_{h}^{\prime}]). $$ 

Upon this point, we can define the MDP as a tuple  $ \mathcal{M} = (\mathcal{S}, \mathcal{A}, f, r, \nu_{1}, H) $ , where S is the state space, A is the action space, H is the horizon (total steps), the initial state distribution  $ \nu_{1} $  is a distribution over the initial prompt  $ x_{1} $ . Note that in a two-player game environment, each state in S is a pair of  $ s_{h} $  and  $ s_{h}^{\prime} $  generated by two policies. Our goal is to identify the Nash equilibrium (or von Neumann winner) of the following two-player constant-sum Markov game:

 $$ (\pi^{*},\pi^{*})=\arg\max_{\pi}\min_{\pi^{\prime}}\mathbb{E}_{s_{1}\sim\nu_{1},s_{h},a_{h},s_{h}^{\prime},a_{h}^{\prime}}\Big[\sum_{h=1}^{H}r(s_{h},a_{h},s_{h}^{\prime},a_{h}^{\prime})\Big], $$ 

 $$  where s_{1}=s_{1}^{\prime}=x_{1},a_{h}\sim\pi(\cdot|s_{h}),a_{h}^{\prime}\sim\pi^{\prime}(\cdot|s_{h}^{\prime}),s_{h}\sim f(\cdot|s_{h-1},a_{h-1}),s_{h}^{\prime}\sim f(\cdot|s_{h-1}^{\prime},a_{h-1}^{\prime}). $$ 

Here we make a few remarks on the benefit of incorporating human preferences at each step. More detail on the motivation can be found at Appx. G.

Remark 1. If two conversations of H turns,  $ s_{H+1} $  and  $ s_{H+1}^{\prime} $ , are globally similar but differ in the early turns (e.g.,  $ s_{2} $  are better than  $ s_{2}^{\prime} $ ), more credit should be assigned to  $ s_{H+1} $ , encouraging the model to align with it. This follows the principle that humans typically master simpler and earlier tasks before progressing to more complex ones.

Remark 2. From a practical standpoint, including per-step preference data generates a richer dataset for training, helping the model learn which reasoning steps are correct or wrong. This incremental feedback can enhance overall performance by reinforcing the importance of foundational steps in reasoning.

Next, we present some additional notation. We define the pair-wise value function as follows

 $$ V_{h}^{\pi,\pi^{\prime}}(s,s^{\prime})=\mathbb{E}\Big[\sum_{\hat{h}=h}^{H}r(s_{\hat{h}},a_{\hat{h}},s_{\hat{h}}^{\prime},a_{\hat{h}}^{\prime})|s_{h}=s,s_{h}^{\prime}=s^{\prime}\Big], $$ 

where  $ a_{\hat{h}} \sim \pi_{\hat{h}}(\cdot | s_{\hat{h}}) $ ,  $ a'_{\hat{h}} \sim \pi'_{\hat{h}}(\cdot | s'_{\hat{h}}) $ ,  $ s_{\hat{h}+1} \sim f(\cdot | s_{\hat{h}}, a_{\hat{h}}) $ , and  $ s'_{\hat{h}+1} \sim f(\cdot | s'_{\hat{h}}, a'_{\hat{h}}) $ . We will often denote  $ V_{1}^{\pi,\pi'} $  omitting the subscript, i.e., as  $ V^{\pi,\pi'} $ . Moreover, notice that we consider potentially non-stationary policies, i.e., they are indexed by h. We denote by  $ \pi $  the non-stationary policy and by  $ \pi_{h} $  the distribution over actions at step h corresponding to the non-stationary policy  $ \pi $ .

We define the pair-wise Q-function as follows:



 $$ Q_{h}^{\pi,\pi^{\prime}}(s,a,s^{\prime},a^{\prime})=r(s,a,s^{\prime},a^{\prime})+\mathbb{E}\Big[\sum_{\hat{h}=h+1}^{H}r(s_{\hat{h}},a_{\hat{h}},s_{\hat{h}}^{\prime},a_{\hat{h}}^{\prime})\Big], $$ 

where $s_{\hat{h}+1}\sim f(\cdot|s_{\hat{h}},a_{\hat{h}})$ and $s_{\hat{h}+1}^{\prime}\sim f(\cdot|s_{\hat{h}}^{\prime},a_{\hat{h}}^{\prime})$.

Lemma 1. (Adapted from Puterman (1994)) The pair-wise value function and pair-wise Q-value function satisfy the following Bellman equation for all  $ h \in [H] $ .

 $$ Q_{h}^{\pi,\pi^{\prime}}(s,a,s^{\prime},a^{\prime})=r(s,a,s^{\prime},a^{\prime})+\mathbb{E}_{\hat{s}\sim f(\cdot|s,a),\bar{s}\sim f(\cdot|s^{\prime},a^{\prime})}[V_{h+1}^{\pi,\pi^{\prime}}(\hat{s},\bar{s})]. $$ 

 $$ V_{h}^{\pi,\pi^{\prime}}(s,s^{\prime})=\mathbb{E}_{a\sim\pi_{h}(\cdot|s),a^{\prime}\sim\pi_{h}^{\prime}(\cdot|s^{\prime})}Q_{h}^{\pi,\pi^{\prime}}(s,a,s^{\prime},a^{\prime}). $$ 

By Lemma 1, we can rewrite Game as follows:

 $$ \left(\pi^{*},\pi^{*}\right)=\arg\max_{\pi}\min_{\pi^{\prime}}\mathbb{E}\Big[\sum_{h=1}^{H}r(s_{h},a_{h},s_{h}^{\prime},a_{h}^{\prime})\Big]=\arg\max_{\pi}\min_{\pi^{\prime}}\mathbb{E}_{s_{1}\sim\nu_{1}}V^{\pi,\pi^{\prime}}(s_{1},s_{1}). $$ 

Given the above notation, we can formalize our objective. We look for a policy  $ \pi $  satisfying the following definition of approximate equilibrium.

Definition 1 ( $ \epsilon $ -approximate Nash equilibrium). A policy  $ \pi $  is said to be an approximate Nash equilibrium if it holds that

 $$ \left\langle\nu_{1},V^{\pi,\pi}\right\rangle-\min_{\bar{\pi}\in\Pi}\left\langle\nu_{1},V^{\pi,\bar{\pi}}\right\rangle\leq\epsilon, $$ 

and

 $$ \max_{\bar{\pi}\in\Pi}\left\langle\nu_{1},V^{\bar{\pi},\pi}\right\rangle-\left\langle\nu_{1},V^{\pi,\pi}\right\rangle\leq\epsilon. $$ 

Definition 2 (Occupancy measures). Given the policy  $ \pi $ , the occupancy measure of  $ \pi $ , is defined at stage h as  $ d_{h}^{\pi}(s,a)=\operatorname*{Pr}(s_{h}=s,a_{h}=a) $  where  $ s_{1}=x_{1}\sim\nu_{1},a_{h}\sim\pi_{h}(\cdot|s_{h}),s_{h}\sim f(\cdot|s_{h-1},a_{h-1}) $ . We also define  $ d_{h}^{\pi}(s,a)|_{s_{1}}=\operatorname*{Pr}(s_{h}=s,a_{h}=a|s_{1}=s_{1}) $ . In addition, given the policies  $ \pi,\bar{\pi} $ , the occupancy measure of  $ (\pi,\bar{\pi}) $  at stage h is defined as  $ d_{h}^{\pi,\pi}(s,a,s',a')=\operatorname*{Pr}(s_{h}=s,a_{h}=a,s_{h}'=s',a_{h}'=a') $ , where  $ s_{1}=s_{1}^{\prime}=x_{1}\sim\nu_{1},a_{h}\sim\pi(\cdot|s_{h}),a_{h}^{\prime}\sim\pi^{\prime}(\cdot|s_{h}^{\prime}) $ ,  $ s_{h}\sim f(\cdot|s_{h-1},a_{h-1}) $ , and  $ s_{h}^{\prime}\sim f(\cdot|s_{h-1}^{\prime},a_{h-1}^{\prime}) $ .

Remark: The value function at the initial state can be represented as an inner product between the reward function and the occupancy measure, i.e.,  $ V^{\pi,\pi} = \sum_{h=1}^{H} \left\langle r, d_{h}^{\pi,\pi} \right\rangle $ . Given the structure of the game where the sequences of sentences and answers are generated independently by the two agents given the initial state  $ s_{1} $ , the occupancy measure at each step can be factorized as the product of the two agents occupancy measures given  $ s_{1} $ . In particular, we have  $ d_{h}^{\pi,\pi}(s, a, s', a')|s_{1} = d_{h}^{\pi}(s, a)|s_{1} \cdot d_{h}^{\pi}(s', a')|s_{1} $  for all h, s, a,  $ s', a' $ .

## 4 METHOD

We first develop our method Multi-Step Preference Optimization (MPO) based on the natural actor-critical framework (Peters & Schaal, 2008; Alacaoglu et al., 2022) in Sec. 4.1. Next, we introduce Optimistic Multi-Step Preference Optimization, dubbed OMPo, in Sec. 4.2. The framework is inspired by the idea of optimism used in online learning and in min-max optimization with improved theoretical guarantees (Popov, 1980; Chiang et al., 2012; Rakhlin & Sridharan, 2013).

### 4.1 MPO WITH NATURAL ACTOR-CRITIC

This section presents our first method to find an approximate solution to Game. In order to find an  $ \epsilon $ -approximate Nash equilibrium, the MPO method builds upon the next lemma which decomposes the difference of two value functions to the Q function at each step. The lemma 2 is the extension of Kakade & Langford (2002) to the multi-agent setting where the dynamics are controlled independently by each player but the reward depends on the joint-state action tuple.

Lemma 2 (Value difference lemma (Adapted from Kakade & Langford (2002))). For a finite horizon MDP with initial distribution  $ \nu_{1} $  it holds that:

 $$ \left\langle\nu_{1},V^{\pi,\bar{\pi}}-V^{\pi^{\prime},\bar{\pi}}\right\rangle=\mathbb{E}_{s_{1}\sim\nu_{1}}\sum_{h=1}^{H}\mathbb{E}_{s\sim d_{h}^{\pi}|s_{1}}\left[\left\langle\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\bar{\pi}}|s_{1}}Q_{h}^{\pi^{\prime},\bar{\pi}}(s,\cdot,s^{\prime},a^{\prime}),\pi_{h}(\cdot|s,s_{1})-\pi_{h}^{\prime}(\cdot|s,s_{1})\right\rangle\right]. $$ 

The proof can be found at Appx. D.2. In our setting, the initial state  $ s_{1} $  is a deterministic function of the state s so we can remove  $ s_{1} $  from the conditioning in the policy $ ^{2} $ . To highlight this fact we

Algorithm 1 MPO (Theory Version)
input: reference policy  $ \pi^{1} $ , preference oracle P, learning rate  $ \beta = \sqrt{\frac{\log \pi^{-1}}{TH^{2}}} $ , total iteration T for  $ t = 1, 2, \ldots, T $  do
 $ \pi_{h}^{t+1}(a|s) \propto \pi_{h}^{t}(a|s) \exp \left[ \beta E_{s',a' \sim d_{h}^{t} | s_{1}(s)} Q_{h}^{t, \pi^{t}}(s, a, s', a') \right] \quad \forall h \in [H], \quad \forall s, a. $ 
end for
output:  $ \bar{\pi}^{T} $  (such that  $ d_{h}^{\bar{\pi}^{T}} = \frac{1}{T} \sum_{t=1}^{T} d_{h}^{t} $ ,  $ \forall h \in [H] $ ).

Algorithm 2 MPO (Practical version)

input: reference policy  $ \pi^{1} $ , preference oracle P, learning rate  $ \beta $ , number of generated samples K, horizon H, total iteration T.

for  $ t = 1, 2, \ldots, T $  do

Generates response by sampling  $ s_{1}^{1} \sim \nu_{1} $  and  $ a_{h}^{1} \sim \pi^{t}(\cdot | s_{h}^{1}) $  for  $ h \in [H] $ .

Clear the dataset buffer  $ D_{t} $ .

for  $ h = 1, 2, \ldots, H $  do

Set  $ s_{h}^{K} = \ldots, s_{h}^{2} = s_{h}^{1} $ .

Generate K - 1 conversations by sampling  $ a_{\hat{h}}^{2:K} \sim \pi^{t}(\cdot | s_{\hat{h}}^{2:K}) $  for  $ \hat{h} \in [h, H] $ .

Estimate  $ \mathbb{E}_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}}) $ ,  $ \forall k, k^{\prime} \in [K] $  via Eq. (5) with query to P.

Form the data pair  $ \{(s_{h}^{1}, a_{h}^{k}, \mathbb{E}_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}})\}_{k \in [K]} $ , add to  $ D_{t} $ .

end for

Optimize  $ \pi_{t+1} $  over  $ D_{t} $  according to

 $ \pi^{t+1} \leftarrow \arg \min_{\pi} \mathbb{E}\left(\log \left(\frac{\pi(a_{h}^{k} | s_{h}^{1})}{\pi^{t}(a_{h}^{k} | s_{h}^{1})}\right) - \beta \left(\mathbb{E}_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}}) - \frac{H - h + 1}{2}\right)\right)^{2} $ .

end for

output:  $ \pi^{T+1} $ 

denote as  $ s_{1}(s) $  the only initial state that can lead to s. By setting  $ \pi' = \overline{\pi} = \pi^{t} $  in Lemma 2 and  $ \pi = \pi^{\star} $  and summing from t = 1 to T we obtain:

 $$ \sum_{t=1}^{T}\left\langle\nu_{1},V^{\pi^{\star},\pi^{t}}-V^{\pi^{t},\pi^{t}}\right\rangle=\mathbb{E}_{s_{1}\sim\nu_{1}}\sum_{h=1}^{H}\sum_{t=1}^{T}\mathbb{E}_{s\sim d_{h}^{\pi^{\star}}|s_{1}}\left[\left\langle\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}}Q_{h}^{\pi^{t},\pi^{t}}(s,\cdot,s^{\prime},a^{\prime}),\pi_{h}^{\star}(\cdot|s)-\pi_{h}^{t}(\cdot|s)\right\rangle\right] $$ 

Since the sum over t commutes with the expectation, we see that we can decompose the global regret  $ \sum_{t=1}^{T}\left\langle\nu_{1},V^{\pi^{\star},\pi^{t}}-V^{\pi^{t},\pi^{t}}\right\rangle $  into a weighted sum of local regrets at each stage  $ h\in[H] $ , i.e.,  $ \mathbb{E}_{s\sim d_{h}^{\pi^{*}}|s_{1}}\left[\sum_{t=1}^{T}\left\langle\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}}Q_{h}^{\pi^{t},\pi^{t}}(s,\cdot,s^{\prime},a^{\prime}),\pi_{h}^{\star}(\cdot|s)-\pi_{h}^{t}(\cdot|s)\right\rangle\right] $ . Therefore, we can control the global regret implementing at each state online mirror descent updates (Warmuth et al. 1997, Orabona 2023, Chapter 6, Cesa-Bianchi & Lugosi 2006), i.e., implementing the following update:

 $$ \pi_{h}^{t+1}(\cdot|s)=\underset{\pi}{\arg\max}\langle\pi(\cdot|s),\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}(s)}Q_{h}^{\pi^{t},\pi^{t}}(s,\cdot,s^{\prime},a^{\prime})\rangle-\beta D(\pi(\cdot|s)||\pi_{h}^{t}(\cdot|s)), $$ 

where  $ \beta $  is a learning rate. The solution has the following form:

 $$ \pi_{h}^{t+1}(a|s)\propto\pi_{h}^{t}(a|s)\exp\{\beta\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}(s)}Q_{h}^{\pi^{t},\pi^{t}}(s,a,s^{\prime},a^{\prime})\}, $$ 

which corresponds to natural actor-critic (Peters & Schaal, 2008) that utilizes a softmax-based method for updating policies. The number of policy updates needed by the ideal version of MPO (see Alg. 1) can be bounded as follows and the proof can be found at Appx. D.3.

Theorem 4. Consider Algorithm 1 and assume that the reference policy is uniformly lower bounded by  $ \pi $ , then there exists a policy  $ \bar{\pi}^{T} $  such that  $ d_{h}^{\bar{\pi}^{T}} = \frac{1}{T} \sum_{t=1}^{T} d_{h}^{\pi^{t}}, \forall h \in [H] $ , and it holds that for  $ T = \frac{16H^{4} \log \underline{\pi}^{-1}}{\epsilon^{2}} $  the policy pair  $ (\bar{\pi}^{T}, \bar{\pi}^{T}) $  is an  $ \epsilon $ -approximate Nash equilibrium. Therefore, Algorithm 1 outputs an  $ \epsilon $ -approximate Nash equilibrium after  $ \frac{16H^{4} \log \underline{\pi}^{-1}}{\epsilon^{2}} $  policy updates.

Algorithm 3 OMPO (Theory Version)

input: occupancy measure of reference policy  $ \pi^{1} $  denoted as  $ d^{1} $ , preference oracle P (i.e. reward function r), learning rate  $ \beta $ , Bregman divergence D, iteration T

for  $ t = 1, 2, \ldots, T $  do

 $ d_{h}^{t+1} = \arg \max_{d \in \mathcal{F}_{s_{1}}} \beta \left\langle d, 2 \mathbb{E}_{s', a' \sim d_{h}^{t}} r(\cdot, \cdot, s', a') - \mathbb{E}_{s', a' \sim d_{h}^{t-1}} r(\cdot, \cdot, s', a') \right\rangle - \mathbb{D}(d, d_{h}^{t}) \quad \forall h \in [H] \quad \forall s_{1} $ 

end for

 $ \pi_{h}^{\text{out}}(a|s) = \frac{\bar{d}_{h}(s, a|s_{1})}{\sum_{a} \bar{d}_{h}(s, a|s_{1})} $  with  $ \bar{d}_{h} = T^{-1} \sum_{t=1}^{T} d_{h}^{t} $  for all  $ h \in [H] $  for the unique  $ s_{1} $  from which s is reachable.

Output:  $ \pi^{\text{out}} $ 

Remark 3. The above result generalizes the  $ \mathcal{O}(H^{2}\epsilon^{-2}) $  bound on the policy updates proven in Swamy et al. (2024) in the setting of terminal-only reward. The additional  $ H^{2} $  factor in our theorem is due to considering rewards that are not terminal-only. In Theorem 5 we show that Algorithm 3 improves the number of policy updates needed to converge to an  $ \epsilon $ -approximate Nash equilibrium to  $ \mathcal{O}(H\epsilon^{-1}) $ .

Practical relaxations. For the above theorem, MPO requires the access of the Q function, which is unknown. Next, we are going to develop a practical algorithm to efficiently estimate the Q function and implement Eq. (2). Equivalently, Eq. (2) can be written as

 $$ \pi_{h}^{t+1}(a|s)=\frac{\pi_{h}^{t}(a|s)\exp\{\beta\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}(s)}Q_{h}^{\pi^{t},\pi^{t}}(s,a,s^{\prime},a^{\prime})\}}{Z_{h}^{t}(s)}, $$ 

where  $ Z_{h}^{t}(s) $  is the partition function. Next, we express Eq. (3) as follows:

 $$ \log\frac{\pi_{h}^{t+1}(a|s)}{\pi_{h}^{t}(a|s)}=\beta\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}(s)}Q_{h}^{\pi^{t},\pi^{t}}(s,a,s^{\prime},a^{\prime})-\log Z_{h}^{t}(s). $$ 

Next, we approximate Eq. (4) with an approximate solution of the following optimization program

 $$ \pi^{t+1}=\arg\min_{\pi}\sum_{h=1}^{H}\mathbb{E}_{\substack{s_{1}\sim\nu_{1}\\ (s_{h},a_{h})\sim d_{h}^{\pi^{t}}|s_{1}}}\Biggl[\log\frac{\pi(a_{h}|s_{h})}{\pi_{h}^{t}(a_{h}|s_{h})}-\left(\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}}Q_{h}^{\pi^{t},\pi^{t}}(s_{h},a_{h},s^{\prime},a^{\prime})-\log Z_{h}^{t}(s_{h})\right)\Biggr]^{2}. $$ 

Unfortunately, solving the above minimization exactly is out of hope. The first difficulty is the efficient estimation of  $ \mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}}Q_{h}^{\pi^{t},\pi^{t}}(s_{h},a_{h},s^{\prime},a^{\prime}) $ . In particular, since  $ s^{\prime} $  and s are sampled from the same distribution, we will sample  $ a^{\prime} $  from the state  $ s_{h} $  and use the Monte Carlo estimator:

 $$ \mathbb{E}_{a^{\prime}\sim\pi^{t}(\cdot|s_{h})}Q_{h}^{\pi^{t},\pi^{t}}(s_{h},a_{h},s_{h},a^{\prime})\approx\frac{1}{K}\sum_{k=1}^{K}\sum_{\hat{h}=h}^{H}\mathbb{P}([s_{\hat{h},k},a_{\hat{h},k}],[s_{\hat{h},k}^{\prime},a_{\hat{h},k}^{\prime}]), $$ 

where the sequences  $ \left\{(s_{\hat{h},k},a_{\hat{h},k},s_{\hat{h},k}^{\prime},a_{\hat{h},k}^{\prime})\right\}_{\hat{h}=h}^{H} $  for  $ k\in[K] $  are generated by rollouts of the policies pair  $ (\pi^{t},\pi^{t}) $ . The second difficulty is  $ Z_{h}^{t}(s) $ , which is difficult to compute for large action spaces. In all states s, we replace  $ \log Z_{h}^{t}(s) $  with  $ \beta^{\frac{H-h+1}{2}} $ .

Remark 4. The heuristics is motivated by the next observation. If the preference between  $ a_{h} $  and  $ a_{h}^{\prime} $  in Eq. (5) results in a tie, then with such  $ \log Z_{h}^{t}(s) $ , the solution of Eq. (5) is  $ \pi^{t+1} = \pi^{t} $ , leaving the model unchanged.

In summary, we provide a practical version of MPO in Alg. 2. In practice, we used a stationary policy that we find to be sufficient to obtain convincing results.

### 4.2 OPTIMISTIC MPO: OMPO

In this section, we propose an alternative algorithm based on the optimistic gradient descent method $ ^{3} $  by reformulating the optimization problem over occupancy measures. Here, we show that opti-

Algorithm 4 OMPO (Practical version)

input: reference policy  $ \pi^{1} $ , preference oracle P, learning rate  $ \beta $ , number of generated samples K, horizon H, total iteration T, tunable bias term  $ \tau $ .

for  $ t = 1, 2, \ldots, T $  do

Generates response by sampling  $ s_{1}^{1} \sim \nu_{1} $  and  $ a_{h}^{1} \sim \pi^{t}(\cdot | s_{h}^{1}) $  for  $ h \in [H] $ .

Clear the dataset buffer  $ D_{t} $ .

for  $ h = 1, 2, \ldots, H $  do

Set  $ s_{h}^{K} = \ldots, s_{h}^{2} = s_{h}^{1} $ .

Generate K - 1 conversations by sampling  $ a_{\tilde{h}}^{2:K} \sim \pi^{t}(\cdot | s_{h}^{2:K}) $  for  $ \hat{h} \in [h, H] $ .

Estimate  $ E_{a_{h}^{k}} Q^{\pi^{t}, \pi^{t}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}}) \forall k, k^{\prime} \in [K] $  via Eq. (5).

if t > 1 then

Estimate  $ E_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t-1}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}}) \forall k, k^{\prime} \in [K] $  via Eq. (5).

Add  $ \{(s_{h}^{1}, a_{h}^{k}, E_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}}), E_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t-1}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}})\}_{k \in [K]} $  into  $ D_{t} $ .

else

Add  $ \{(s_{h}^{1}, a_{h}^{k}, E_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}})\} $  into  $ D_{t} $ .

end if

end for

if t > 1 then

Optimize  $ \pi_{t+1} $  over  $ D_{t} $  according to

 $ \pi^{t+1} \leftarrow \arg \min_{\pi} \mathbb{E}\left(\log\left(\frac{\pi(a_{h}^{k} | s_{h}^{1})}{\pi^{t}(a_{h}^{k} | s_{h}^{1})}\right) - \beta\left(2E_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}}) - E_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t-1}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}}) - \tau\right)\right)^{2} $ .

else

Optimize  $ \pi_{t+1} $  over  $ D_{t} $  according to

 $ \pi^{t+1} \leftarrow \arg \min_{\pi} \mathbb{E}\left(\log\left(\frac{\pi(a_{h}^{k} | s_{h}^{1})}{\pi^{t}(a_{h}^{k} | s_{h}^{1})}\right) - \beta\left(E_{a_{h}^{k^{\prime}}} Q^{\pi^{t}, \pi^{t}}(s_{h}^{1}, a_{h}^{k}, s_{h}^{1}, a_{h}^{k^{\prime}}) - \frac{H - h + 1}{2}\right)\right)^{2} $ .

end if

end for

output:  $ \pi^{T+1} $ 

mistic online mirror descent with one projection (Joulani et al., 2017) with an appropriately chosen regularizer can be used to solve approximately the following program which corresponds to Game lifted to the space of conditional occupancy measures.

 $$ (d^{\star},d^{\star})=\underset{d\in\tilde{\mathcal{F}}}{\arg\max}\min_{d^{\prime}\in\tilde{\mathcal{F}}}\mathbb{E}_{s_{1}\sim\nu_{1}}\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}d_{h}(s,a|s_{1})r(s,a,s^{\prime},a^{\prime})d_{h}^{\prime}(s^{\prime},a^{\prime}|s_{1}), $$ 

where  $ \tilde{F} $  is the product set of the Bellman flow constraints for a particular initial state, i.e.  $ \tilde{F} = \times_{s_1 \in \operatorname{supp}(\nu_1)} F_{s_1} $ . We also introduced the Bellman flow constraints for a specific initial state  $ F_{s_1} = \left\{ d = (d_1, \ldots, d_H) : \sum_a d_{h+1}(s, a) = \sum_{s', a'} f(s|s', a') d_h(s', a'), d_1(s) = 1 \left\{ s = s_1 \right\} \right\} $ . The policy pair  $ (\pi^*, \pi^*) $  solution of Game can be retrieved from the occupancy measure pair  $ (d^* $ ,  $ d^*) $  as  $ \pi^\star(a|s) = \frac{d^*\left(s, a|s_1\right)}{\sum_a d^*\left(s, a|s_1\right)} $ . Our idea is to apply the optimistic algorithm from Joulani et al. (2017) to the reformulation of Game over occupancy measures, we present the resulting algorithm, i.e., OMPO, in Alg. 3.

Remark 5. In a partially observable Markov game, lifting the problem to the occupancy measures turns out to be fundamentally important for enabling each agent to learn a policy conditioned only on their own state. This is different from the standard literature on Markov Games (Daskalakis et al., 2020; Wei et al., 2021; Alacaoglu et al., 2022), which assumes that both agents share a common state.

As the next theorem shows, in the ideal case where the updates can be computed exactly, Alg. 3 finds an  $ \epsilon $ -approximate Nash equilibrium using fewer updates compared to Alg. 1 and to (Swamy et al., 2024, Algorithm 1). The proof can be found at Appx. D.4.

<div style="text-align: center;">Table 1: Evaluation results on MT-bench-101 dataset. Mistral-7B-Instruct is selected as the base model. We can observe that both of the proposed algorithms MPO and OMPO considerably outperform the baseline in terms of the score (the higher the better).</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3">Model</td><td colspan="5">Perceptivity</td><td colspan="5">Adaptability</td><td colspan="2">Interactivity</td></tr><tr><td rowspan="2">Avg.</td><td style='text-align: center;'>Memory</td><td colspan="2">Understanding</td><td style='text-align: center;'>Interference</td><td style='text-align: center;'>Rephrasing</td><td colspan="2">Reflection</td><td colspan="2">Reasoning</td><td colspan="2">Questioning</td></tr><tr><td style='text-align: center;'>CM</td><td style='text-align: center;'>SI</td><td style='text-align: center;'>AR</td><td style='text-align: center;'>TS</td><td style='text-align: center;'>CC</td><td style='text-align: center;'>CR</td><td style='text-align: center;'>FR</td><td style='text-align: center;'>SC</td><td style='text-align: center;'>SA</td><td style='text-align: center;'>MR</td><td style='text-align: center;'>GR</td></tr><tr><td style='text-align: center;'>Base (Mistral-7B-Instruct)</td><td style='text-align: center;'>6.223</td><td style='text-align: center;'>7.202</td><td style='text-align: center;'>7.141</td><td style='text-align: center;'>7.477</td><td style='text-align: center;'>7.839</td><td style='text-align: center;'>8.294</td><td style='text-align: center;'>6.526</td><td style='text-align: center;'>6.480</td><td style='text-align: center;'>4.123</td><td style='text-align: center;'>4.836</td><td style='text-align: center;'>4.455</td><td style='text-align: center;'>5.061</td></tr><tr><td style='text-align: center;'>DPO (iter=1)</td><td style='text-align: center;'>6.361</td><td style='text-align: center;'>7.889</td><td style='text-align: center;'>6.483</td><td style='text-align: center;'>7.699</td><td style='text-align: center;'>8.149</td><td style='text-align: center;'>8.973</td><td style='text-align: center;'>7.098</td><td style='text-align: center;'>7.423</td><td style='text-align: center;'>3.448</td><td style='text-align: center;'>6.123</td><td style='text-align: center;'>3.421</td><td style='text-align: center;'>4.492</td></tr><tr><td style='text-align: center;'>DPO (iter=2)</td><td style='text-align: center;'>6.327</td><td style='text-align: center;'>7.611</td><td style='text-align: center;'>6.206</td><td style='text-align: center;'>8.106</td><td style='text-align: center;'>8.052</td><td style='text-align: center;'>9.111</td><td style='text-align: center;'>6.670</td><td style='text-align: center;'>7.153</td><td style='text-align: center;'>3.494</td><td style='text-align: center;'>5.884</td><td style='text-align: center;'>3.360</td><td style='text-align: center;'>4.691</td></tr><tr><td style='text-align: center;'>DPO (iter=3)</td><td style='text-align: center;'>5.391</td><td style='text-align: center;'>6.019</td><td style='text-align: center;'>4.521</td><td style='text-align: center;'>6.890</td><td style='text-align: center;'>6.631</td><td style='text-align: center;'>8.177</td><td style='text-align: center;'>5.437</td><td style='text-align: center;'>5.723</td><td style='text-align: center;'>3.448</td><td style='text-align: center;'>5.295</td><td style='text-align: center;'>3.142</td><td style='text-align: center;'>4.015</td></tr><tr><td style='text-align: center;'>SPPO (iter=1)</td><td style='text-align: center;'>6.475</td><td style='text-align: center;'>7.432</td><td style='text-align: center;'>7.464</td><td style='text-align: center;'>7.714</td><td style='text-align: center;'>8.353</td><td style='text-align: center;'>8.580</td><td style='text-align: center;'>6.917</td><td style='text-align: center;'>6.714</td><td style='text-align: center;'>4.136</td><td style='text-align: center;'>5.055</td><td style='text-align: center;'>4.403</td><td style='text-align: center;'>5.400</td></tr><tr><td style='text-align: center;'>SPPO (iter=2)</td><td style='text-align: center;'>6.541</td><td style='text-align: center;'>7.516</td><td style='text-align: center;'>7.496</td><td style='text-align: center;'>7.808</td><td style='text-align: center;'>8.313</td><td style='text-align: center;'>8.731</td><td style='text-align: center;'>7.077</td><td style='text-align: center;'>6.867</td><td style='text-align: center;'>4.136</td><td style='text-align: center;'>5.281</td><td style='text-align: center;'>4.488</td><td style='text-align: center;'>5.477</td></tr><tr><td style='text-align: center;'>SPPO (iter=3)</td><td style='text-align: center;'>6.577</td><td style='text-align: center;'>7.575</td><td style='text-align: center;'>7.547</td><td style='text-align: center;'>7.944</td><td style='text-align: center;'>8.365</td><td style='text-align: center;'>8.797</td><td style='text-align: center;'>7.040</td><td style='text-align: center;'>6.865</td><td style='text-align: center;'>4.442</td><td style='text-align: center;'>5.185</td><td style='text-align: center;'>4.346</td><td style='text-align: center;'>5.394</td></tr><tr><td style='text-align: center;'>Step-DPO (iter=1)</td><td style='text-align: center;'>6.433</td><td style='text-align: center;'>7.463</td><td style='text-align: center;'>7.054</td><td style='text-align: center;'>7.790</td><td style='text-align: center;'>8.157</td><td style='text-align: center;'>8.593</td><td style='text-align: center;'>6.827</td><td style='text-align: center;'>6.748</td><td style='text-align: center;'>4.234</td><td style='text-align: center;'>4.849</td><td style='text-align: center;'>4.236</td><td style='text-align: center;'>5.519</td></tr><tr><td style='text-align: center;'>Step-DPO (iter=2)</td><td style='text-align: center;'>6.553</td><td style='text-align: center;'>7.616</td><td style='text-align: center;'>7.043</td><td style='text-align: center;'>7.925</td><td style='text-align: center;'>8.147</td><td style='text-align: center;'>8.662</td><td style='text-align: center;'>6.790</td><td style='text-align: center;'>6.878</td><td style='text-align: center;'>4.331</td><td style='text-align: center;'>5.048</td><td style='text-align: center;'>4.366</td><td style='text-align: center;'>5.734</td></tr><tr><td style='text-align: center;'>Step-DPO (iter=3)</td><td style='text-align: center;'>6.442</td><td style='text-align: center;'>7.665</td><td style='text-align: center;'>7.023</td><td style='text-align: center;'>7.767</td><td style='text-align: center;'>8.016</td><td style='text-align: center;'>8.589</td><td style='text-align: center;'>6.723</td><td style='text-align: center;'>6.581</td><td style='text-align: center;'>4.305</td><td style='text-align: center;'>5.014</td><td style='text-align: center;'>4.153</td><td style='text-align: center;'>5.453</td></tr><tr><td style='text-align: center;'>MPO (iter=1)</td><td style='text-align: center;'>6.630</td><td style='text-align: center;'>7.624</td><td style='text-align: center;'>7.846</td><td style='text-align: center;'>8.085</td><td style='text-align: center;'>8.398</td><td style='text-align: center;'>8.947</td><td style='text-align: center;'>7.105</td><td style='text-align: center;'>7.286</td><td style='text-align: center;'>4.208</td><td style='text-align: center;'>4.993</td><td style='text-align: center;'>4.377</td><td style='text-align: center;'>5.264</td></tr><tr><td style='text-align: center;'>MPO (iter=2)</td><td style='text-align: center;'>6.735</td><td style='text-align: center;'>7.838</td><td style='text-align: center;'>7.723</td><td style='text-align: center;'>8.196</td><td style='text-align: center;'>8.590</td><td style='text-align: center;'>9.027</td><td style='text-align: center;'>7.347</td><td style='text-align: center;'>7.209</td><td style='text-align: center;'>4.240</td><td style='text-align: center;'>5.137</td><td style='text-align: center;'>4.469</td><td style='text-align: center;'>5.531</td></tr><tr><td style='text-align: center;'>MPO (iter=3)</td><td style='text-align: center;'>6.733</td><td style='text-align: center;'>7.868</td><td style='text-align: center;'>7.686</td><td style='text-align: center;'>8.289</td><td style='text-align: center;'>8.510</td><td style='text-align: center;'>9.078</td><td style='text-align: center;'>7.330</td><td style='text-align: center;'>7.529</td><td style='text-align: center;'>4.461</td><td style='text-align: center;'>4.829</td><td style='text-align: center;'>4.225</td><td style='text-align: center;'>5.366</td></tr><tr><td style='text-align: center;'>MPO (iter=2)</td><td style='text-align: center;'>6.736</td><td style='text-align: center;'>7.733</td><td style='text-align: center;'>7.723</td><td style='text-align: center;'>8.257</td><td style='text-align: center;'>8.478</td><td style='text-align: center;'>9.122</td><td style='text-align: center;'>7.300</td><td style='text-align: center;'>7.421</td><td style='text-align: center;'>4.123</td><td style='text-align: center;'>5.288</td><td style='text-align: center;'>4.506</td><td style='text-align: center;'>5.513</td></tr><tr><td style='text-align: center;'>MPO (iter=3)</td><td style='text-align: center;'>6.776</td><td style='text-align: center;'>7.649</td><td style='text-align: center;'>7.792</td><td style='text-align: center;'>8.281</td><td style='text-align: center;'>8.578</td><td style='text-align: center;'>9.136</td><td style='text-align: center;'>7.424</td><td style='text-align: center;'>7.635</td><td style='text-align: center;'>4.377</td><td style='text-align: center;'>5.308</td><td style='text-align: center;'>4.312</td><td style='text-align: center;'>5.455</td></tr></table>

Theorem 5 (Convergence of OMPO). Consider Algorithm 3 and let us assume that the occupancy measure of the reference policy is uniformly lower bounded by d. Moreover, let D be  $ 1/\lambda $  strongly convex, i.e.  $ \mathbb{D}(p||q) \geq \frac{\|p-q\|_{\beta}^{2}}{2\lambda} $ . Then, by setting  $ T = \frac{10H\log d^{-1}}{\beta\epsilon} $  and  $ \beta \leq \frac{1}{\sqrt{2\lambda}} $ , we ensure that  $ (\pi^{\mathrm{out}}, \pi^{\mathrm{out}}) $ , i.e. the output of Algorithm 3 is an  $ \epsilon $ -approximate Nash equilibrium. Therefore, we need at most  $ \frac{10H\log d^{-1}}{\beta\epsilon} $  policy updates.

In addition, not only Swamy et al. (2024, Algorithm 1) but also OMPO can be implemented using only one player since in a constant sum game, the max and min player produce the same iterates. The result is formalized as follows and the proof is deferred to Appx. D.5.

Theorem 6. Consider a constant sum two-player Markov games with reward such that  $ r(s,a,s',a') = 1 - r(s',a',s,a) $ , then for each  $ s_{1} \in \operatorname{supp}(\nu_{1}) $  the updates for d in Alg. 3 co-incides with the updates for the min player that uses the updates

 $$ d_{h}^{t+1}(a|s)=\underset{d\in\mathcal{F}_{s_{1}}}{\arg\min}\beta\left\langle d,2\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{t}}r(s^{\prime},a^{\prime},\cdot,\cdot)-\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{t-1}}r(s^{\prime},a^{\prime},\cdot,\cdot)\right\rangle+\mathbb{D}(d,d_{h}^{t}). $$ 

Furthermore, we can avoid the projection over the set F implementing this update on the policy space (see Appendix E). We achieve such results following the techniques developed in Bas-Serrano et al. (2021); Viano et al. (2022).

For the first iteration, we initialize  $ d_{h}^{0} $  to be equal to  $ d_{h}^{1} $  for all h. That is, at the first iteration, we use the same update rule as in MPO. After the first iteration, we apply similar techniques as in MPO by estimating the Q function and we use a tunable parameter to approximate the  $ \log Z $  term. We illustrate the practical algorithm in Alg. 4.

## 5 EXPERIMENTS

In this section, we test the proposed algorithms with multi-turn conversations in MT-bench-101 (Bai et al., 2024). Additional experimental detail, ablation studies, and experiments on math reasoning tasks are deferred to Appx. F. We choose Mistral-7B-Instruct-v0.2 as the base model (Jiang et al., 2023). We use a pre-trained PairRM $ ^{4} $  as the preference oracle. Specifically, given two conversations  $ [s_{h}, a_{h}] $  and  $ [s_{h}^{\prime}, a_{h}^{\prime}] $ , PairRM will return a score that indicates the probability that  $ [s_{h}, a_{h}] $ 

<div style="text-align: center;"><img src="imgs/img_in_chart_box_235_192_636_441.jpg" alt="Image" width="32%" /></div>


<div style="text-align: center;">(a) Radar chart on different categories.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_655_174_968_436.jpg" alt="Image" width="25%" /></div>


<div style="text-align: center;">(b) Winning rate against the base model.</div>


<div style="text-align: center;">Figure 1: (a): Result of OMPO on the MT-bench-101 dataset; (b) Winning rate against the base model with different approximations for the Q functions. When optimizing  $ a_{h} $  at the h step, only considering the preference of  $ s_{h} $  is sufficient compared to using  $ s_{h}, \ldots, s_{H+1} $ .</div>


is better than  $ [s_{h}^{\prime}, a_{h}^{\prime}] $ , which can be used to consider as the preference oracle P defined in the previous section. We select iterative DPO (Dong et al., 2024), iterative SPPO (Wu et al., 2024), and iterative Step-DPO as our baselines. For both iterative DPO and iterative SPPO, we sample K = 5 complete conversations starting from  $ s_{1} $ , and estimate the winning rate  $ \mathbb{P}([s_{H+1}^{k}, a_{H+1}^{k}]\succ(s_{H+1}^{k^{\prime}}, a_{H+1}^{k^{\prime}}])\forall k, k^{\prime}\in[K] $ . Then we select both the best and worst conversations according to their winning rates against others, which is defined as  $ \frac{1}{K}\sum_{k^{\prime}=1}^{K}\mathbb{P}([s_{H+1}^{k}, a_{H+1}^{k}]\succ[s_{H+1}^{k^{\prime}}, a_{H+1}^{k^{\prime}}]) $  for the conversation  $ [s_{H+1}^{k}, a_{H+1}^{k}] $ . Such a pair is used to train DPO while the winning rate is used to train SPPO. For both Step-DPO, MPO, and OMPO, we do the same strategy with starting at  $ s_{h} $ . In MPO, and OMPO, we estimate  $ Q(s_{h}, a_{h}, s_{h}, a_{h}^{\prime}) $  by  $ \mathbb{P}([s_{h}, a_{h}],[s_{h}, a_{h}^{\prime}]) $  to enhance the efficiency. For OMPO, the  $ Q^{\pi^{t},\pi^{t-1}} $  term is estimated by calculating the winning rate between two answers (the best and the worst) generated by the current policy  $ \pi^{t} $  and the five answers previously generated by  $ \pi^{t-1} $ , the  $ \tau $  is selected as zero. Each method is trained with epochs number selected from  $ \{1, 2\} $ , learning rates from  $ \{5e-6, 5e-7\} $ , and  $ \beta $  values from  $ \{0.1, 0.01, 0.001\} $ . The final model is chosen based on the highest winning rate against the base model, as determined by the PairRM model. We use full-parameter fine-tuning for all methods with bf16 precision. A batch size of 64 is used. The maximum output length and maximum prompt length during training are both set as 2048. We use AdamW optimizer (Loshchilov & Hutter, 2019) and cosine learning rate schedule (Loshchilov & Hutter, 2017) with a warmup ratio of 0.1. Each round of dialogue is rated on a scale of 1 to 10 by GPT-4o mini, with the mean score reported for each dialogue. All methods are run for a total of 3 iterations. The results are summarized in Tab. 1, showing significant improvements over the baselines with the proposed MPO and OMPO approaches. In Fig. 1(a), we present the Radar chart on different categories and we can see that the proposed OMPO leads to improvements generally along the iterations. Fig. 1(b) shows that using the entire trajectory to estimate the Q function can lead to subtle improvement at the first two iterations while it finally achieves a similar winning rate when compared to the one that only uses one step.

## 6 CONCLUSION

This work presents a novel framework to enhance the preference alignment of large language models in multi-step settings by casting the alignment process as a two-player Markov game. We introduce novel algorithms based on natural actor-critic and optimistic online gradient descent, supported by both theoretical analysis and empirical results. However, the limitations of this work include the finite-horizon assumption in our theoretical framework, which may not fully capture real-world conversations or reasoning processes that often span with different steps instead of a fixed step H. Additionally, our practical algorithm requires querying a preference oracle, which may limit its applicability in cases where such preference oracles are unavailable or when collecting human feedback is costly. Future work should explore extending the theoretical framework to infinite-horizon settings and finding more scalable methods for gathering preference feedback.

## ETHICS STATEMENT

Our work focuses on algorithmic innovations related to reinforcement learning with human feedback. We do not create any new benchmarks for human preferences nor solicit human preferences for this study. As such, we do not expect any potential violations of ethical standards, including those concerning the use of human data. Our contributions are primarily methodological and theoretical analysis of the convergence, and we have taken care to ensure that our work complies with all relevant ethical guidelines.

## REPRODUCIBILITY STATEMENT

In this work, we have provided the details on the experimental setup and the description of the dataset at Sec. 5 and Appx. F.1. The dataset and language models used in this work are publicly available. The source code of MPO and OMPo will be made public in the camera-ready version. Regarding the theoretical results, we have clearly mentioned all of the assumptions, and all the complete proofs can be found at Appx. D and Appx. E.