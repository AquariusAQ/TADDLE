

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

## REFERENCES

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Ahmet Alacaoglu, Luca Viano, Niao He, and Volkan Cevher. A natural actor-critic framework for zero-sum markov games. In International Conference on Machine Learning, pp. 307–366. PMLR, 2022.

Mohammad Gheshlaghi Azar, Zhaohan Daniel Guo, Bilal Piot, Remi Munos, Mark Rowland, Michal Valko, and Daniele Calandriello. A general theoretical paradigm to understand learning from human preferences. In International Conference on Artificial Intelligence and Statistics, pp. 4447–4455. PMLR, 2024.

Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jiaheng Liu, Zhanhui Zhou, Zhuoran Lin, Wenbo Su, Tiezheng Ge, Bo Zheng, et al. Mt-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues. arXiv preprint arXiv:2402.14762, 2024.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862, 2022.

Joan Bas-Serrano, Sebastian Curi, Andreas Krause, and Gergely Neu. Logistic q-learning. In International conference on artificial intelligence and statistics, pp. 3610–3618. PMLR, 2021.

Ralph Allan Bradley and Milton E Terry. Rank analysis of incomplete block designs: I. the method of paired comparisons. Biometrika, 39(3/4):324–345, 1952.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

Nicolo Cesa-Bianchi and Gábor Lugosi. Prediction, learning, and games. Cambridge university press, 2006.

Chao-Kai Chiang, Tianbao Yang, Chia-Jung Lee, Mehrdad Mahdavi, Chi-Jen Lu, Rong Jin, and Shenghuo Zhu. Online optimization with gradual variations. In Shie Mannor, Nathan Srebro, and Robert C. Williamson (eds.), Proceedings of the 25th Annual Conference on Learning Theory, volume 23 of Proceedings of Machine Learning Research, pp. 6.1–6.20, Edinburgh, Scotland, 25–27 Jun 2012. PMLR. URL https://proceedings.mlr.press/v23/chiang12.html.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Constantinos Daskalakis, Dylan J Foster, and Noah Golowich. Independent policy gradient methods for competitive reinforcement learning. Advances in neural information processing systems, 33:5527–5540, 2020.

Hanze Dong, Wei Xiong, Bo Pang, Haoxiang Wang, Han Zhao, Yingbo Zhou, Nan Jiang, Doyen Sahoo, Caiming Xiong, and Tong Zhang. Rlhf workflow: From reward modeling to online rlhf. arXiv preprint arXiv:2405.07863, 2024.

Roy Fox, Ari Pakman, and Naftali Tishby. Taming the noise in reinforcement learning via soft updates. arXiv preprint arXiv:1512.08562, 2015.

Yoav Freund and Robert E Schapire. Adaptive game playing using multiplicative weights. Games and Economic Behavior, 29(1-2):79–103, 1999.

Martin Gardner. Mathematical games. Scientific american, 222(6):132–140, 1970.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.

Jiwoo Hong, Noah Lee, and James Thorne. Orpo: Monolithic preference optimization without reference model. arXiv preprint arXiv:2403.07691, 2(4):5, 2024.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

Pooria Joulani, András György, and Csaba Szepesvári. A modular analysis of adaptive (non-)convex optimization: Optimism, composite objectives, and variational bounds. In Steve Hanneke and Lev Reyzin (eds.), Proceedings of the 28th International Conference on Algorithmic Learning Theory, volume 76 of Proceedings of Machine Learning Research, pp. 681–720. PMLR, 15–17 Oct 2017. URL https://proceedings.mlr.press/v76/joulani17a.html.

Sham Kakade and John Langford. Approximately optimal approximate reinforcement learning. In Proceedings of the Nineteenth International Conference on Machine Learning, pp. 267–274, 2002.

Xin Lai, Zhuotao Tian, Yukang Chen, Senqiao Yang, Xiangru Peng, and Jiaya Jia. Step-dpo: Stepwise preference optimization for long-chain reasoning of llms. arXiv preprint arXiv:2406.18629, 2024.

Orin Levy, Alon Cohen, Asaf Cassel, and Yishay Mansour. Efficient rate optimal regret for adversarial contextual mdps using online function approximation. In International Conference on Machine Learning, pp. 19287–19314. PMLR, 2023.

Aiwei Liu, Haoping Bai, Zhiyun Lu, Yanchao Sun, Xiang Kong, Simon Wang, Jiulong Shan, Albin Madappally Jose, Xiaojiang Liu, Lijie Wen, et al. Tis-dpo: Token-level importance sampling for direct preference optimization with estimated weights. arXiv preprint arXiv:2410.04350, 2024a.

Haoxiong Liu, Yifan Zhang, Yifan Luo, and Andrew Chi-Chih Yao. Augmenting math word problems via iterative question composing. In ICLR 2024 Workshop on Navigating and Addressing Data Problems for Foundation Models, 2024b. URL https://openreview.net/forum?id=OasPFqWyTA.

Ilya Loshchilov and Frank Hutter. SGDR: Stochastic gradient descent with warm restarts. In International Conference on Learning Representations, 2017. URL https://openreview.net/forum?id=Skq89Scxx.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7.

Zimu Lu, Aojun Zhou, Ke Wang, Houxing Ren, Weikang Shi, Junting Pan, and Mingjie Zhan. Step-controlled dpo: Leveraging stepwise error for enhanced mathematical reasoning. arXiv preprint arXiv:2407.00782, 2024.

Yura Malitsky and Matthew K Tam. A forward-backward splitting method for monotone inclusions without cocoercivity. SIAM Journal on Optimization, 30(2):1451–1472, 2020.

Yu Meng, Mengzhou Xia, and Danqi Chen. Simpo: Simple preference optimization with a reference-free reward. arXiv preprint arXiv:2405.14734, 2024.

Rémi Munos, Michal Valko, Daniele Calandriello, Mohammad Gheshlaghi Azar, Mark Rowland, Zhaohan Daniel Guo, Yunhao Tang, Matthieu Geist, Thomas Mesnard, Andrea Michi, et al. Nash learning from human feedback. In Forty-first International Conference on Machine Learning, 2024.

Gergely Neu and Julia Olkhovskaya. Online learning in mdps with linear function approximation and bandit feedback. Advances in Neural Information Processing Systems, 34:10407–10417, 2021.

Gergely Neu, Anders Jonsson, and Vicenç Gómez. A unified view of entropy-regularized markov decision processes, 2017. URL https://arxiv.org/abs/1705.07798.

Francesco Orabona. A modern introduction to online learning, 2023. URL https://arxiv.org/abs/1912.13213.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

Jan Peters and Stefan Schaal. Natural actor-critic. Neurocomputing, 71(7-9):1180–1190, 2008.

Leonid Denisovich Popov. A modification of the arrow-hurwitz method of search for saddle points. Mat. Zametki, 28(5):777–784, 1980.

M. L. Puterman. Markov Decision Processes: Discrete Stochastic Dynamic Programming. John Wiley & Sons, Inc., USA, 1st edition, 1994.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems, 36, 2023.

Rafael Rafailov, Joey Hejna, Ryan Park, and Chelsea Finn. From $r$ to $q^*$: Your language model is secretly a q-function. In First Conference on Language Modeling, 2024. URL https://openreview.net/forum?id=kEVcNxtqXk.

Alexander Rakhlin and Karthik Sridharan. Online learning with predictable sequences. In Conference on Learning Theory, pp. 993–1019. PMLR, 2013.

Corby Rosset, Ching-An Cheng, Arindam Mitra, Michael Santacroce, Ahmed Awadallah, and Tengyang Xie. Direct nash optimization: Teaching language models to self-improve with general preferences. arXiv preprint arXiv:2404.03715, 2024.

Lior Shani, Aviv Rosenberg, Asaf Cassel, Oran Lang, Daniele Calandriello, Avital Zipori, Hila Noga, Orgad Keller, Bilal Piot, Idan Szpektor, et al. Multi-turn reinforcement learning from preference human feedback. arXiv preprint arXiv:2405.14655, 2024.

Lloyd S Shapley. Stochastic games. Proceedings of the national academy of sciences, 39(10):1095–1100, 1953.

Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008–3021, 2020.

Gokul Swamy, Christoph Dann, Rahul Kidambi, Steven Wu, and Alekh Agarwal. A minimaximalist approach to reinforcement learning from human feedback. In Forty-first International Conference on Machine Learning, 2024.

Yunhao Tang, Zhaohan Daniel Guo, Zeyu Zheng, Daniele Calandriello, Rémi Munos, Mark Rowland, Pierre Harvey Richemond, Michal Valko, Bernardo Ávila Pires, and Bilal Piot. Generalized preference optimization: A unified approach to offline alignment. arXiv preprint arXiv:2402.05749, 2024.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Amos Tversky. Intransitivity of preferences. Psychological review, 76(1):31, 1969.

Luca Viano, Angeliki Kamoutsi, Gergely Neu, Igor Krawczuk, and Volkan Cevher. Proximal point imitation learning. Advances in Neural Information Processing Systems, 35:24309–24326, 2022.

Yuanhao Wang, Qinghua Liu, and Chi Jin. Is rlhf more difficult than standard rl? a theoretical perspective. Advances in Neural Information Processing Systems, 2023.

Manfred K Warmuth, Arun K Jagota, et al. Continuous and discrete-time nonlinear gradient descent: Relative loss bounds and convergence. In Electronic proceedings of the 5th International Symposium on Artificial Intelligence and Mathematics, volume 326. Citeseer, 1997.

Chen-Yu Wei, Chung-Wei Lee, Mengxiao Zhang, and Haipeng Luo. Last-iterate convergence of decentralized optimistic gradient descent/ascent in infinite-horizon competitive markov games. In Conference on Learning Theory, pp. 4259–4299. PMLR, 2021.

Yue Wu, Zhiqing Sun, Huizhuo Yuan, Kaixuan Ji, Yiming Yang, and Quanquan Gu. Self-play preference optimization for language model alignment. arXiv preprint arXiv:2405.00675, 2024.

Longhui Yu, Weisen Jiang, Han Shi, Jincheng YU, Zhengying Liu, Yu Zhang, James Kwok, Zhenguo Li, Adrian Weller, and Weiyang Liu. Metamath: Bootstrap your own mathematical questions for large language models. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=N8N0hgNDRt.

Yongcheng Zeng, Guoqing Liu, Weiyu Ma, Ning Yang, Haifeng Zhang, and Jun Wang. Token-level direct preference optimization. In Forty-first International Conference on Machine Learning, 2024.

Yuheng Zhang, Dian Yu, Baolin Peng, Linfeng Song, Ye Tian, Mingyue Huo, Nan Jiang, Haitao Mi, and Dong Yu. Iterative nash policy optimization: Aligning llms with general preferences via no-regret learning. arXiv preprint arXiv:2407.00617, 2024.

Brian D Ziebart. Modeling purposeful adaptive behavior with the principle of maximum causal entropy. Carnegie Mellon University, 2010.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593, 2019.

## CONTENTS OF THE APPENDIX

The Appendix is organized as follows:

• In Appx. A, we summarize the symbols and notation used in this paper.

• Preliminaries on single-step RLHF can be found in Appx. B.

• In Appx. D, we provide the proofs for the theoretical results.

• Appx. E shows the implementation of Algorithm 3 with updates over policies.

• Appx. F.1 provides an overview of the MT-bench 101 benchmark in the experiment.

## A Symbols and Notation

We include the core symbols and notation in Tab. 2 to facilitate the understanding of our work.

<div style="text-align: center;">Table 2: Core symbols and notations used in this paper.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Symbol</td><td style='text-align: center;'>Dimension(s) &amp; range</td><td style='text-align: center;'>Definition</td></tr><tr><td style='text-align: center;'>x_{h}</td><td style='text-align: center;'>-</td><td style='text-align: center;'>Prompt at step  $ h $</td></tr><tr><td style='text-align: center;'>a_{h}</td><td style='text-align: center;'>-</td><td style='text-align: center;'>Answer (action) at step  $ h $</td></tr><tr><td style='text-align: center;'>s_{h}</td><td style='text-align: center;'>-</td><td style='text-align: center;'>State at step  $ h $</td></tr><tr><td style='text-align: center;'>s_{1}(s_{h})</td><td style='text-align: center;'>-</td><td style='text-align: center;'>The only initial state that can lead to  $ s_{h} $</td></tr><tr><td style='text-align: center;'>\pi</td><td style='text-align: center;'></td><td style='text-align: center;'>Language model (policy)</td></tr><tr><td style='text-align: center;'>\nu_{1}</td><td style='text-align: center;'></td><td style='text-align: center;'>Initial distribution of state  $ s_{1} $</td></tr><tr><td style='text-align: center;'>d_{h}^{\pi}(s,a)</td><td style='text-align: center;'>[0,1]</td><td style='text-align: center;'>Occupancy measure of \pi at stage  $ h $</td></tr><tr><td style='text-align: center;'>f</td><td style='text-align: center;'></td><td style='text-align: center;'>Transition function</td></tr><tr><td style='text-align: center;'>\Pr(s_{h}=s,a_{h}=a)</td><td style='text-align: center;'>[0,1]</td><td style='text-align: center;'>Joint probability of  $ s_{h}=a $  and  $ a_{h}=a $</td></tr><tr><td style='text-align: center;'>o</td><td style='text-align: center;'>{0,1}</td><td style='text-align: center;'>Preference oracle</td></tr><tr><td style='text-align: center;'>\mathbb{P}([s,a],[s&#x27;,a&#x27;])</td><td style='text-align: center;'>[0,1]</td><td style='text-align: center;'>Winning probability of [s,a] against [s&#x27;,a&#x27;]</td></tr><tr><td style='text-align: center;'>D(p||q)</td><td style='text-align: center;'></td><td style='text-align: center;'>KL divergence of two probability distributions  $ p $  and  $ q $</td></tr><tr><td style='text-align: center;'>\mathbb{D}(p||q)</td><td style='text-align: center;'></td><td style='text-align: center;'>Bregman Divergences between two points  $ q $  and  $ p $ .</td></tr><tr><td style='text-align: center;'>D_{t}</td><td style='text-align: center;'></td><td style='text-align: center;'>Dataset buffet at iteration  $ t $</td></tr><tr><td style='text-align: center;'>\Delta_{x}</td><td style='text-align: center;'>[0,1]^{|\mathcal{X}|}</td><td style='text-align: center;'>Set of probability distributions over the set \mathcal{X}</td></tr><tr><td style='text-align: center;'>\mathcal{O}, \mathcal{O}, \Omega $  and \Theta</td><td style='text-align: center;'>-</td><td style='text-align: center;'>Standard Bachmann-Landau order notation</td></tr></table>

We additionally use a compact notation for representing the Bellman flow constraints. We denote by  $ E \in R^{|S| \times |A||S|} $  the matrix such that  $ (Ez)(s,a) = z(s) $  for all vectors  $ z \in R^{|S|} $ . Additionally, we denote by F the matrix such that  $ (Fz)(s,a) = \sum_{s'} f(s'|s,a)z(s') $  for all vectors  $ z \in R^{|S|} $ .

## B PRELIMINARY ON SINGLE-STEP RLHF

In this section, we review the earlier methods in single-step RLHF. Classical RLHF methods (Ziegler et al., 2019; Ouyang et al., 2022) assume that the preference oracle can be expressed by an underlying Bradley-Terry (BT) reward model (Bradley & Terry, 1952), i.e.,

 $$ \mathbb{P}([x_{1},a_{1}]\succ[x_{1},a_{1}^{\prime}])=\sigma(r(x_{1},a_{1})-r(x_{1},a_{1}^{\prime})). $$ 

Thus, one can first learn a reward model and optimize the policy based on the following KL-constrained RL objective with PPO:

 $$ \pi^{\star}=\underset{\pi}{\arg\max}\mathbb{E}_{x_{1}\sim\nu_{1},a_{1}\sim\pi(\cdot|x_{1})}(r(x_{1},a_{1})-\beta D(\pi(\cdot|x_{1})||\pi_{\mathrm{ref}}(\cdot|x_{1}))), $$ 

where  $ \beta $  is a parameter controlling the deviation from the reference model  $ \pi_{ref} $ . Another line of work, e.g., DPO (Rafailov et al., 2023) avoids explicit reward modeling and optimizes the following objective over pair-wise preference data  $ (x_{1}, a_{1}^{w}, a_{1}^{l}) $ .

 $$ \pi^{\star}=\underset{\pi}{\arg\max}\mathbb{E}_{(x_{1},a_{1}^{w},a_{1}^{l})\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi(a_{1}^{w}|x_{1})}{\pi_{1}(a_{1}^{w}|x_{1})}-\beta\log\frac{\pi(a_{1}^{l}|x_{1})}{\pi_{1}(a_{1}^{l}|x_{1})}\right)\right]. $$ 

More recently, several studies (Swamy et al., 2024; Munos et al., 2024; Wu et al., 2024; Zhang et al., 2024; Rosset et al., 2024) have circumvented the Bradley-Terry (BT) assumption by directly modeling the general oracle P, avoiding the reliance on the reward model which is transitive. Specifically, the goal is to identify the Nash equilibrium (or von Neumann winner) of the following two-player constant-sum game:

 $$ \begin{array}{r}{(\pi^{*},\pi^{*})=\arg\underset{\pi}{\operatorname*{m a x}}\underset{\pi^{\prime}}{\operatorname*{m i n}}\mathbb{E}_{x_{1}\sim\nu_{1},a_{1}\sim\pi(\cdot|x_{1}),a_{1}^{\prime}\sim\pi^{\prime}(\cdot|x_{1})}\mathbb{P}\big([x_{1},a_{1}]\succ[x_{1},a_{1}^{\prime}]\big).}\end{array} $$ 

## C ADDITIONAL DISCUSSION ON RELATED WORK

### C.1 RELATED WORK ON TOKEN-LEVEL PREFERENCE OPTIMIZATION

A line of work formulates the alignment of contextual bandit problems in LLMs (Example.1) from token-level MDPs perspective (Rafailov et al., 2024; Zeng et al., 2024; Liu et al., 2024a). In Rafailov et al. (2024), by defining the reward at each token before the terminal token as the generation likelihood and using the maximum entropy RL objective, the authors derive the original objective of DPO from a new perspective that incorporates token-level rewards. Zeng et al. (2024) assume that the reward for a response can be decomposed into token-level rewards at each token. Then they design a token-level objective function based on Trust Region Policy Optimization, adding token-level KL divergence constraints to the DPO objective in the final algorithm. More recently, Liu et al. (2024a) study how the difference in average rewards between chosen and rejected responses affects the optimization stability, designing a new algorithm where importance sampling weights are assigned to each token-level reward. There are two main differences between the multi-step alignment approach in our work and those in previous work. First, while Rafailov et al. (2024); Zeng et al. (2024); Liu et al. (2024a) develop alignment methods based on the Bradley-Terry model with transitive rewards, our framework is motivated by a two-player game with relative rewards. Secondly, although Rafailov et al. (2024); Zeng et al. (2024); Liu et al. (2024a) formulate the alignment process as an MDP, their final objective is tailored to a contextual bandit problem in LLMs. In contrast, our objective is designed for a multi-step alignment problem, suited for multi-turn conversation or chain-of-thought reasoning.

### C.2 DISCUSSION ON THE DIFFERENCE FROM SPPO

Next, we elaborate on the difference with SPPO (Wu et al., 2024) below: Firstly, the theoretical analysis of the proposed MPO differs from that of SPPO due to differences in the settings. SPPO considers the contextual bandit problem and builds its analysis based on the game matrix from Freund & Schapire (1999). In our case, however, we frame the problem as a Markov game and employ a distinct theoretical analysis apart from Freund & Schapire (1999). Specifically, in our proof, we (i) use the performance difference lemma to rewrite the global regret as weighted average of local regrets and (ii) control the local regrets with multiplicative weights updates. Secondly, a new algorithm, OMPO, is developed in this work with a novel theoretical guarantee. In the case where the horizon H = 1, the update of OMPO reduces to

 $$ \pi^{t+1}(a|s)\propto\pi^{t}(a|s)\exp\left[\beta(2\mathbb{P}(a\succ\pi^{t}(\cdot|s))-\mathbb{P}(a\succ\pi^{t-1}(\cdot|s)))\right], $$ 

while the update of SPPO is

 $$ \pi^{t+1}(a|s)\propto\pi^{t}(a|s)\exp\left[\beta(\mathbb{P}(a\succ\pi^{t}(\cdot|s)))\right]. $$ 

As a result, OMPO enables  $ \mathcal{O}(\epsilon^{-1}) $  policy updates to converge to an  $ \epsilon $ -approximate Nash equilibrium instead of  $ \mathcal{O}(\epsilon^{-2}) $ , according to our theoretical analysis.

## D PROOFS

### D.1 Proof of Lemma 1

Proof. By the definition of the state action value function for the policy pair  $ (\pi, \pi^{\prime}) $  we have that

 $$ Q_{h}^{\pi,\pi^{\prime}}(s,a,s^{\prime},a^{\prime})=r(s,a,s^{\prime},a^{\prime})+\mathbb{E}\Big[\sum_{h^{\prime}=h+1}^{H}r(s_{h^{\prime}},a_{h^{\prime}},s_{h^{\prime}}^{\prime},a_{h^{\prime}}^{\prime})\Big]. $$ 

Now, using tower property of the expectation we have that

 $$ \begin{aligned}&Q_{h}^{\pi,\pi^{\prime}}(s,a,s^{\prime},a^{\prime})\\ &=r(s,a,s^{\prime},a^{\prime})+\mathbb{E}_{s^{\prime\prime}\sim f(\cdot|s,a),\bar{s}\sim f(\cdot|s^{\prime},a^{\prime})}\Big[\mathbb{E}\Big[\sum_{h^{\prime}=h+1}^{H}r(s_{h^{\prime}},a_{h^{\prime}},s_{h^{\prime}}^{\prime},a_{h^{\prime}}^{\prime})|s_{h+1}=s^{\prime\prime},s_{h+1}^{\prime}=\bar{s}\Big]\Big]\\ &=r(s,a,s^{\prime},a^{\prime})+\mathbb{E}_{s^{\prime\prime}\sim f(\cdot|s,a),\bar{s}\sim f(\cdot|s^{\prime},a^{\prime})}\Big[V^{\pi,\pi^{\prime}}(s^{\prime\prime},\bar{s})\Big],\\ \end{aligned} $$ 

where the last equality follows from the definition of the state value function.

### D.2 Proof of Lemma 2

Proof. Let us consider the Bellman equation in vectorial form for the policy pair  $ (\pi^{\prime},\bar{\pi}) $ , that is

 $$ r_{h}+F V_{h+1}^{\pi^{\prime},\bar{\pi}}=Q_{h}^{\pi^{\prime},\bar{\pi}}, $$ 

where F denoted the transition matrix induced by the transition function  $ f : S^{2} \times A \to \Delta_{S \times S} $ . Now, multiplying by the occupancy measure of the policy pair  $ (\pi, \bar{\pi}) $  at stage h we obtain

 $$ \left\langle d_{h}^{\pi,\bar{\pi}},r_{h}\right\rangle+\left\langle d_{h}^{\pi,\bar{\pi}},F V_{h+1}^{\pi^{\prime},\bar{\pi}}\right\rangle=\left\langle d_{h}^{\pi,\bar{\pi}},Q_{h}^{\pi^{\prime},\bar{\pi}}\right\rangle. $$ 

At this point, using the Bellman flow constraints Puterman (1994), it holds that

 $$ F^{T}d_{h}^{\pi,\bar{\pi}}=E^{T}d_{h+1}^{\pi,\bar{\pi}}, $$ 

where $E\in\mathbb{R}^{|S|^{2}|\mathcal{A}|\times|\mathcal{S}|^{2}}$ such that $(E^{T}V)(s,a)=V(s)$ for all $V\in\mathbb{R}^{|S|^{2}}$. Plugging this equality in the Bellman equation above we obtain

 $$ \left\langle d_{h}^{\pi,\bar{\pi}},r_{h}\right\rangle+\left\langle d_{h+1}^{\pi,\bar{\pi}},E V_{h+1}^{\pi^{\prime},\bar{\pi}}\right\rangle=\left\langle d_{h}^{\pi,\bar{\pi}},Q_{h}^{\pi^{\prime},\bar{\pi}}\right\rangle. $$ 

Now, subtracting on both sides  $ \left\langle d_{h}^{\pi,\bar{\pi}}, EV_{h}^{\pi',\bar{\pi}} \right\rangle $  and rearranging, it holds that

 $$ \left\langle d_{h}^{\pi,\bar{\pi}},r_{h}\right\rangle+\left\langle d_{h+1}^{\pi,\bar{\pi}},E V_{h+1}^{\pi^{\prime},\bar{\pi}}\right\rangle-\left\langle d_{h}^{\pi,\bar{\pi}},E V_{h}^{\pi^{\prime},\bar{\pi}}\right\rangle=\left\langle d_{h}^{\pi,\bar{\pi}},Q_{h}^{\pi^{\prime},\bar{\pi}}-E V_{h}^{\pi^{\prime},\bar{\pi}}\right\rangle. $$ 

After this, taking sum from h = 1 to H and recognizing that for all policy pairs  $ (\pi, \pi') $  it holds that  $ V_{H+1}^{\pi, \pi'} = 0 $ , it holds that

 $$ \sum_{h=1}^{H}\left\langle d_{h}^{\pi,\bar{\pi}},r_{h}\right\rangle-\left\langle d_{1}^{\pi,\bar{\pi}},E V_{1}^{\pi^{\prime},\bar{\pi}}\right\rangle=\sum_{h=1}^{H}\left\langle d_{h}^{\pi,\bar{\pi}},Q_{h}^{\pi^{\prime},\bar{\pi}}-E V_{h}^{\pi^{\prime},\bar{\pi}}\right\rangle. $$ 

Then, notice that for all policies  $ \pi $ ,  $ \bar{\pi} $  it holds that  $ \sum_{h=1}^{H}\left\langle d_{h}^{\pi,\bar{\pi}},r_{h}\right\rangle=\left\langle\nu_{1},V^{\pi,\bar{\pi}}\right\rangle $ . Plugging in these observations, we get

 $$ \left\langle\nu_{1},V^{\pi,\bar{\pi}}-V^{\pi^{\prime},\bar{\pi}}\right\rangle=\sum_{h=1}^{H}\left\langle d_{h}^{\pi,\bar{\pi}},Q_{h}^{\pi^{\prime},\bar{\pi}}-E V_{h}^{\pi^{\prime},\bar{\pi}}\right\rangle. $$ 

Therefore, expanding the expectation, and noticing that  $ d_{h}^{\pi,\bar{\pi}}(s,a,s',a'|s_{1}) = d_{h}^{\pi}(s,a|s_{1})d_{h}^{\bar{\pi}}(s',a'|s_{1}) $  for all  $ h,s,a,s',a' $  and conditioning  $ s_{1} $ , we get that

 $$ \begin{align*}&\left\langle\nu_{1},V^{\pi,\bar{\pi}}-V^{\pi^{\prime},\bar{\pi}}\right\rangle\\&=\mathbb{E}_{s_{1}\sim\nu_{1}}\sum_{h=1}^{H}\mathbb{E}_{s\sim d_{h}^{\pi}|s_{1}}\left[\left\langle\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\bar{\pi}}|s_{1}}Q_{h}^{\pi^{\prime},\bar{\pi}}(s,\cdot,s^{\prime},a^{\prime}),\pi_{h}(\cdot|s,s_{1})-\pi_{h}^{\prime}(\cdot|s,s_{1})\right\rangle\right].\end{align*} $$ 

#### D.3 Proof of Thm. 4

Proof. We set  $ \bar{\pi}_{h}^{T}(a_{h}|s_{h})=\frac{\sum_{t=1}^{T}d_{h}^{\pi^{t}}(s_{h},a_{h})}{\sum_{t=1}^{T}d_{h}^{\pi^{t}}(s_{h})} $ , where  $ d(s) $  is the marginal distribution of  $ d(s,a) $  on state s, and  $ \bar{\pi}^{T}=(\bar{\pi}_{h}^{T})_{h=1}^{H} $ . We shows that  $ d_{h}^{\bar{\pi}^{T}}=\frac{1}{T}\sum_{t=1}^{T}d_{h}^{\pi^{t}} $  by induction. h=1 holds by definition. Assuming on step h, the equation holds, we have

 $$ \begin{align*}d_{h+1}^{T}(s_{h+1},a_{h+1})&=d_{h+1}^{\bar{\pi}^{T}}(s_{h+1})\bar{\pi}_{h+1}^{T}(a_{h+1}|s_{h+1})\\&=\sum_{s_{h},a_{h}\sim\bar{\pi}_{h}^{T}(\cdot|s_{h})}d_{h}^{\bar{\pi}^{T}}(s_{h},a_{h})f(s_{h+1}|s_{h},a_{h})\bar{\pi}_{h+1}^{T}(a_{h+1}|s_{h+1})\\&=\sum_{s_{h},a_{h}\sim\bar{\pi}_{h}^{T}(\cdot|s_{h})}\frac{1}{T}\sum_{t=1}^{T}d_{h}^{\pi^{t}}(s_{h},a_{h})f(s_{h+1}|s_{h},a_{h})\bar{\pi}_{h+1}^{T}(a_{h+1}|s_{h+1})\\&=\frac{1}{T}\sum_{t=1}^{T}d_{h+1}^{\pi^{t}}(s_{h+1})\bar{\pi}_{h+1}^{T}(a_{h+1}|s_{h+1})\\&=\frac{1}{T}\sum_{t=1}^{T}d_{h+1}^{\pi^{t}}(s_{h+1},a_{h+1}),\end{align*} $$ 

where the last equation holds by definition of  $ \bar{\pi}_{h+1}^{T} $ . Therefore,  $ h+1 $  holds, and the  $ \bar{\pi}^{T} $  satisfy all equations for  $ h\in[H] $ .

Using the value difference Lemma 2 we have that for any $\pi^{\star}\in\Pi$

 $$ \begin{align*}&\left\langle\nu_{1},V^{\pi^{*},\pi^{t}}-V^{\pi^{t},\pi^{t}}\right\rangle\\&=\mathbb{E}_{s_{1}\sim\nu_{1}}\sum_{h=1}^{H}\mathbb{E}_{s\sim d_{h}^{\pi^{*}}\mid s_{1}}\left[\left\langle\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}\mid s_{1}}Q_{h}^{\pi^{t},\pi^{t}}(s,\cdot,s^{\prime},a^{\prime}),\pi_{h}^{\star}(\cdot|s)-\pi_{h}^{t}(\cdot|s)\right\rangle\right].\end{align*} $$ 

Therefore, summing over t from t = 1 to T we obtain

 $$ \begin{align*}&\sum_{t=1}^{T}\left\langle\nu_{1},V^{\pi^{\star},\pi^{t}}-V^{\pi^{t},\pi^{t}}\right\rangle\\&=\mathbb{E}_{s_{1}\sim\nu_{1}}\sum_{h=1}^{H}\mathbb{E}_{s\sim d_{h}^{\pi^{\star}}|s_{1}}\left[\sum_{t=1}^{T}\left\langle\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}}Q_{h}^{\pi^{t},\pi^{t}}(s,\cdot,s^{\prime},a^{\prime}),\pi_{h}^{\star}(\cdot|s)-\pi_{h}^{t}(\cdot|s)\right\rangle\right].\end{align*} $$ 

Therefore, we need to control the local regrets at each state s with loss  $ \ell_{h}^{t}(s,s_{1}):=\mathbb{E}_{s^{\prime},a^{\prime}\sim d_{h}^{\pi^{t}}|s_{1}}Q_{h}^{\pi^{t},\pi^{t}}(s,\cdot,s^{\prime},a^{\prime}) $ . To this end, we can invoke a standard convergence result for online mirror descent (Orabona, 2023, Theorem 6.10) we obtain that at each state we have

 $$ \sum_{t=1}^{T}\left\langle\ell_{h}^{t}(s,s_{1}),\pi^{\star}(\cdot|s)-\pi^{t}(\cdot|s)\right\rangle\leq\frac{D(\pi^{\star}(\cdot|s),\pi^{1}(\cdot|s))}{\beta}+\beta\sum_{t=1}^{T}\|\ell_{h}^{t}(s,s_{1})\|_{\infty}^{2}. $$ 

Now, noticing that we have  $ \|\ell_{h}^{t}(s,s_{1})\|_{\infty}\leq H $  it holds that

 $$ \sum_{t=1}^{T}\left\langle\ell_{h}^{t}(s),\pi_{h}^{\star}(\cdot|s)-\pi_{h}^{t}(\cdot|s)\right\rangle\leq\frac{D(\pi_{h}^{\star}(\cdot|s),\pi_{h}^{1}(\cdot|s))}{\beta}+\beta T H^{2}. $$ 

Finally, using the assumption that  $ \pi^{1}(a|s) \geq \pi $  for all  $ s, a \in S \times A $  it holds that  $ D(\pi^{\star}(\cdot|s), \pi^{1}(\cdot|s)) \leq \log \pi^{-1} $ . Therefore, choosing  $ \beta = \sqrt{\frac{\log \pi^{-1}}{TH^{2}}} $  it holds that

 $$ \sum_{t=1}^{T}\left\langle\ell_{h}^{t}(s,s_{1}),\pi^{\star}(\cdot|s)-\pi^{t}(\cdot|s)\right\rangle\leq2H\sqrt{T\log\underline{\pi}^{-1}}. $$ 

Thus, we conclude that

 $$ \sum_{t=1}^{T}\left\langle\nu_{1},V^{\pi^{\star},\pi^{t}}-V^{\pi^{t},\pi^{t}}\right\rangle\leq2H^{2}\sqrt{T\log\underline{\pi}^{-1}}. $$ 

By the antisimmetry of the game, the same proof steps

 $$ \sum_{t=1}^{T}\left\langle\nu_{1},V^{\pi^{t},\pi^{t}}-V^{\pi^{t},\bar{\pi}^{\star}}\right\rangle\leq2H^{2}\sqrt{T\log\underline{\pi}^{-1}}. $$ 

Therefore, it holds that for all  $ \pi^{\star} $ ,  $ \bar{\pi}^{\star} \in \Pi $ 

 $$ \sum_{t=1}^{T}\left\langle\nu_{1},V^{\pi^{\star},\pi^{t}}-V^{\pi^{t},\pi^{\star}}\right\rangle\leq4H^{2}\sqrt{T\log\underline{\pi}^{-1}}. $$ 

Then, define  $ \bar{\pi}^{T} $  the trajectory level mixture policy as in Swamy et al. (2024), i.e. such that  $ d_{h}^{\bar{\pi}^{T}} = \frac{1}{T} \sum_{t=1}^{T} d_{h}^{\pi^{t}} $  for all stages  $ h \in [H] $ . This implies that  $ V^{\bar{\pi}^{T}, \pi^{\star}} = \frac{1}{T} \sum_{t=1}^{T} V^{\pi^{t}, \pi^{\star}} $ , and  $ V^{\pi^{\star}, \bar{\pi}^{T}} = \frac{1}{T} \sum_{t=1}^{T} V^{\pi^{\star}, \pi_{t}} $ .

Therefore, we have that

 $$ \left\langle\nu_{1},V^{\pi^{\star},\bar{\pi}^{T}}-V^{\bar{\pi}^{T},\bar{\pi}^{\star}}\right\rangle\leq4H^{2}\sqrt{\frac{\log\underline{\pi}^{-1}}{T}}. $$ 

Finally, selecting $\pi^{\star}=\left\langle\nu_{1},\arg\max_{\pi\in\Pi}V^{\pi,\bar{\pi}^{T}}\right\rangle$ and $\bar{\pi}^{\star}=\left\langle\nu_{1},\arg\min_{\pi\in\Pi}V^{\bar{\pi}^{T},\pi}\right\rangle$, we obtain that

 $$ \max_{\pi\in\Pi}\left\langle\nu_{1},V^{\pi,\bar{\pi}^{T}}\right\rangle-\min_{\pi\in\Pi}\left\langle\nu_{1},V^{\bar{\pi}^{T},\pi}\right\rangle\leq4H^{2}\sqrt{\frac{\log\underline{\pi}^{-1}}{T}}. $$ 

This implies that

 $$ \left\langle\nu_{1},V^{\bar{\pi}^{T},\bar{\pi}^{T}}\right\rangle-\min_{\pi\in\Pi}\left\langle\nu_{1},V^{\bar{\pi}^{T},\pi}\right\rangle\leq4H^{2}\sqrt{\frac{\log\underline{\pi}^{-1}}{T}}, $$ 

and

 $$ \max_{\pi\in\Pi}\left\langle\nu_{1},V^{\pi,\bar{\pi}^{T}}\right\rangle-\left\langle\nu_{1},V^{\bar{\pi}^{T},\bar{\pi}^{T}}\right\rangle\leq4H^{2}\sqrt{\frac{\log\underline{\pi}^{-1}}{T}}, $$ 

Therefore, setting  $ T = \frac{16H^{4} \log \pi^{-1}}{\epsilon^{2}} $  we obtain an  $ \epsilon $ -approximate Nash equilibrium.

### D.4 Proof of Theorem 5

Proof. The optimization problem

 $$ \underset{d\in\tilde{\mathcal{F}}}{\arg\max}\min_{d^{\prime}\in\tilde{\mathcal{F}}}\mathbb{E}_{s_{1}\sim\nu_{1}}\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}d_{h}(s,a|s_{1})r(s,a,s^{\prime},a^{\prime})d_{h}^{\prime}(s^{\prime},a^{\prime}|s_{1}) $$ 

can be carried out individually over possible initial states. That is for each  $ s_{1} \in \operatorname{supp}(\nu_{1}) $  we aim at solving

 $$ \underset{d\in\mathcal{F}_{s_{1}}}{\arg\max}\min_{d^{\prime}\in\mathcal{F}_{s_{1}}}\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}d_{h}(s,a|s_{1})r(s,a,s^{\prime},a^{\prime})d_{h}^{\prime}(s^{\prime},a^{\prime}|s_{1}) $$ 

To this end for any  $ s_{1} $ , we consider  $ \phi_{h}^{t} \in F $  and  $ \psi_{h}^{t} \in F $  which are generated by the following updates

 $$ \phi_{h}^{t+1}=\underset{\phi\in\mathcal{F}_{s_{1}}}{\arg\max}\beta\left\langle\phi,2\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t}}r_{h}(\cdot,\cdot,s^{\prime},a^{\prime})-\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t-1}}r_{h}(\cdot,\cdot,s^{\prime},a^{\prime})\right\rangle-\mathbb{D}(\phi,\phi_{h}^{t}), $$ 

and

 $$ \psi_{h}^{t+1}=\underset{\psi\in\mathcal{F}_{s_{1}}}{\arg\min}\beta\left\langle\psi,2\mathbb{E}_{s^{\prime},a^{\prime}\sim\phi^{t}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)-\mathbb{E}_{s^{\prime},a^{\prime}\sim\phi^{t-1}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)\right\rangle+\mathbb{D}(\psi,\psi_{h}^{t}), $$ 

In order to prove convergence to an  $ \epsilon $ -approximate Nash equilibrium, we need to control the quantity

 $$ \mathrm{G a p}_{s_{1}}=\frac{1}{T}\sum_{h=1}^{H}\sum_{t=1}^{T}\left\langle\theta_{h}^{t},\phi_{h}^{t}-\phi_{h}^{\star}\right\rangle+\frac{1}{T}\sum_{h=1}^{H}\sum_{t=1}^{T}\left\langle\zeta_{h}^{t},\psi_{h}^{t}-\psi_{h}^{\star}\right\rangle, $$ 

for  $ \theta_{h}^{t}(s,a)=\sum_{s^{\prime},a^{\prime}}\psi_{h}^{t}(s^{\prime},a^{\prime})r_{h}(s,a,s^{\prime},a^{\prime}) $  and  $ \zeta_{h}^{t}(s^{\prime},a^{\prime})=-\sum_{s,a}\phi_{h}^{t}(s,a)r_{h}(s,a,s^{\prime},a^{\prime}) $ . At this point, we bound the local regret term with the OMPO update. We have that for any  $ \phi_{h}\inF $ 

 $$ \begin{align*}\beta\left\langle2\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}-\phi_{h}^{t+1}\right\rangle&=\beta\left\langle\theta_{h}^{t}-\theta_{h}^{t+1},\phi_{h}-\phi_{h}^{t+1}\right\rangle\\&\quad+\beta\left\langle\theta_{h}^{t}+\theta_{h}^{t+1}-\theta_{h}^{t-1},\phi_{h}-\phi_{h}^{t+1}\right\rangle\\&=\beta\left\langle\theta_{h}^{t}-\theta_{h}^{t+1},\phi_{h}-\phi_{h}^{t+1}\right\rangle\\&\quad+\beta\left\langle\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}-\phi_{h}^{t}\right\rangle\\&\quad+\beta\left\langle\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}^{t}-\phi_{h}^{t+1}\right\rangle\\&\quad+\beta\left\langle\theta_{h}^{t+1},\phi_{h}-\phi_{h}^{t+1}\right\rangle.\end{align*} $$ 

At this point, we work on the third summand above

 $$ \beta\left\langle\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}^{t}-\phi_{h}^{t+1}\right\rangle\leq\beta^{2}\lambda\|\theta_{h}^{t}-\theta_{h}^{t-1}\|_{\infty}^{2}+\frac{1}{4\lambda}\|\phi_{h}^{t}-\phi_{h}^{t+1}\|_{1}^{2}. $$ 

In addition, we have that  $ \|\theta_{h}^{t}-\theta_{h}^{t-1}\|_{\infty}\leq\|\psi_{h}^{t}-\psi_{h}^{t-1}\|_{1} $  and we can apply the  $ 1/\lambda $  strong convexity of D, we obtain

 $$ \beta\left\langle\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}^{t}-\phi_{h}^{t+1}\right\rangle\leq\lambda\beta^{2}\|\psi_{h}^{t}-\psi_{h}^{t-1}\|_{1}^{2}+\frac{1}{2}\mathbb{D}(\phi_{h}^{t+1},\phi_{h}^{t}). $$ 

On the other hand, by the three point identity we have that for all  $ \phi \in F $ 

 $$ \mathbb{D}(\phi_{h},\phi_{h}^{t+1})=\mathbb{D}(\phi_{h},\phi_{h}^{t})-\mathbb{D}(\phi_{h}^{t+1},\phi_{h}^{t})+\left\langle\nabla\mathbb{D}(\phi_{h}^{t+1},\phi_{h}^{t}),\phi_{h}^{t+1}-\phi_{h}\right\rangle $$ 

Then, using the property of the update rule, we obtain that

 $$ \left\langle\nabla\mathbb{D}(\phi_{h}^{t+1},\phi_{h}^{t}),\phi_{h}^{t+1}-\phi_{h}\right\rangle\leq\beta\left\langle2\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}-\phi_{h}^{t+1}\right\rangle. $$ 

Putting all the pieces together we have that

 $$ \begin{align*}\mathbb{D}(\phi_{h},\phi_{h}^{t+1})&\leq\mathbb{D}(\phi_{h},\phi_{h}^{t})-\mathbb{D}(\phi_{h}^{t+1},\phi_{h}^{t})+\beta\left\langle2\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}-\phi_{h}^{t+1}(\cdot|s)\right\rangle\\&\leq\mathbb{D}(\phi_{h},\phi_{h}^{t})-\mathbb{D}(\phi_{h}^{t+1},\phi_{h}^{t})\\&\quad+\beta\left\langle\theta_{h}^{t}-\theta_{h}^{t+1},\phi_{h}-\phi_{h}^{t+1}\right\rangle\\&\quad+\beta\left\langle\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}-\phi_{h}^{t}\right\rangle\\&\quad+\beta^{2}\|\psi_{h}^{t}-\psi_{h}^{t-1}\|_{1}^{2}+\frac{1}{2}\mathbb{D}(\phi_{h}^{t+1},\phi_{h}^{t})\\&\quad+\beta\left\langle\theta_{h}^{t+1},\phi_{h}-\phi_{h}^{t+1}\right\rangle.\end{align*} $$ 

Now, rearranging the terms we get

 $$ \begin{align*}\beta\left\langle\theta_{h}^{t+1},\phi_{h}-\phi_{h}^{t+1}\right\rangle&\leq\mathbb{D}(\phi_{h},\phi_{h}^{t})-\mathbb{D}(\phi_{h},\phi_{h}^{t+1})-\frac{1}{2}\mathbb{D}(\phi_{h}^{t+1},\phi_{h}^{t})\\&\quad+\beta\left\langle\theta_{h}^{t}-\theta_{h}^{t+1},\phi_{h}-\phi_{h}^{t+1}\right\rangle\\&\quad+\beta\left\langle\theta_{h}^{t}-\theta_{h}^{t-1},\phi_{h}-\phi_{h}^{t}\right\rangle\\&\quad+\beta^{2}\lambda\|\psi_{h}^{t}-\psi_{h}^{t-1}\|_{1}^{2}.\end{align*} $$ 

Now, denoting  $ \Phi_{\phi}^{t} := \mathbb{D}(\phi_{h}, \phi_{h}^{t}) + \beta \left\langle \theta_{h}^{t} - \theta_{h}^{t-1}, \phi_{h} - \phi_{h}^{t} \right\rangle $  and summing over t we obtain

 $$ \beta\sum_{t=1}^{T}\left\langle\theta_{h}^{t},\phi_{h}-\phi_{h}^{t}\right\rangle\leq\sum_{t=1}^{T}\Phi_{\phi}^{t-1}-\Phi_{\phi}^{t}-\frac{1}{2}\sum_{t=1}^{T}\mathbb{D}(\phi_{h}^{t},\phi_{h}^{t-1})+\beta^{2}\lambda\sum_{t=1}^{T}\|\psi_{h}^{t-1}-\psi_{h}^{t-2}\|_{1}^{2}. $$ 

Similarly we get

 $$ \beta\sum_{t=1}^{T}\left\langle\zeta^{t}(s,\cdot),\psi_{h}^{t}-\psi_{h}^{t}\right\rangle\leq\sum_{t=1}^{T}\Phi_{\psi}^{t-1}-\Phi_{\psi}^{t}-\frac{1}{2}\sum_{t=1}^{T}\mathbb{D}(\psi_{h}^{t},\psi_{h}^{t-1})+\beta^{2}\lambda\sum_{t=1}^{T}\|\phi_{h}^{t-1}-\psi_{h}^{t-2}\|_{1}^{2}. $$ 

Now, using  $ 1/\lambda $  strong convexity of D and summing the two terms we have that

 $$ \begin{align*}\beta T\mathrm{Gap}_{s_{1},h}\leq\Phi^{0}-\Phi^{T-1}-\frac{1}{2}\sum_{t=1}^{T}(\mathbb{D}(\psi_{h}^{t},\psi_{h}^{t-1})+\mathbb{D}(\phi_{h}^{t},\phi_{h}^{t-1}))\\+2\beta^{2}\lambda\sum_{t=1}^{T}(\mathbb{D}(\psi_{h}^{t-1},\psi_{h}^{t-2})+\mathbb{D}(\phi_{h}^{t-1},\phi_{h}^{t-2})),\end{align*} $$ 

with  $ \Phi^{t}=\Phi_{\phi}^{t}+\Phi_{\psi}^{t} $ . At this point, setting  $ \beta\leq\frac{1}{\sqrt{2\lambda}} $ , we obtain a telescopic sum

 $$ \begin{aligned}&\beta T\mathrm{G a p}_{s_{1},h}\\&\leq\Phi^{0}-\Phi^{T-1}-\frac{1}{2}\sum_{t=1}^{T}(\mathbb{D}(\psi_{h}^{t},\psi_{h}^{t-1})+\mathbb{D}(\phi_{h}^{t},\phi_{h}^{t-1})-\mathbb{D}(\psi_{h}^{t-1},\psi_{h}^{t-2})-\mathbb{D}(\phi_{h}^{t-1},\phi_{h}^{t-2}))\\&\leq\Phi^{0}-\Phi^{T-1}+\frac{1}{2}\left(\mathbb{D}(\psi_{h}^{1},\psi_{h}^{0})+\mathbb{D}(\phi_{h}^{1},\phi_{h}^{0})\right).\\ \end{aligned} $$ 

Now recalling that by assumption the occupancy measure of the reference policy is lower bounded, i.e.  $ d^{\pi^{1}} \geq \underline{d} $ , we can upper bound  $ \Phi^{0} - \Phi^{T} \leq 2 \log \underline{d}^{-1} + 8 \beta $  that allows to conclude that for all  $ n \in [N] $  and setting  $ \psi_{h}^{0} = \psi_{h}^{1} $  and  $ \phi_{h}^{1} = \phi_{h}^{0} $ ,

 $$ \mathrm{G a p}_{s_{1},h}\leq\frac{2\log\underline{d}^{-1}+8\beta}{\beta T}\leq\frac{10\log\underline{d}^{-1}}{\beta T}. $$ 

Now, notice that Gap can be rewritten as

 $$ \begin{aligned}Gap_{s_{1}}&=\sum_{h=1}^{H}Gap_{s_{1},h}\\&=\frac{1}{T}\sum_{t=1}^{T}\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}\psi_{h}^{\star}(s^{\prime},a^{\prime})r_{h}(s,a,s^{\prime},a^{\prime})\phi_{h}^{t}(s,a)\\&\quad-\frac{1}{T}\sum_{t=1}^{T}\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}\psi_{h}^{t}(s^{\prime},a^{\prime})r_{h}(s,a,s^{\prime},a^{\prime})\phi_{h}^{\star}(s,a)\\&=\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}\psi_{h}^{\star}(s^{\prime},a^{\prime})r_{h}(s,a,s^{\prime},a^{\prime})\frac{1}{T}\sum_{t=1}^{T}\phi_{h}^{t}(s,a)\\&\quad-\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}\frac{1}{T}\sum_{t=1}^{T}\psi_{h}^{t}(s^{\prime},a^{\prime})r_{h}(s,a,s^{\prime},a^{\prime})\phi_{h}^{\star}(s,a)\\&=\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}\psi_{h}^{\star}(s^{\prime},a^{\prime})r_{h}(s,a,s^{\prime},a^{\prime})\bar{\phi}_{h}(s,a)-\sum_{h=1}^{H}\sum_{s,a,s^{\prime},a^{\prime}}\bar{\psi}_{h}(s^{\prime},a^{\prime})r_{h}(s,a,s^{\prime},a^{\prime})\phi_{h}^{\star}(s,a).\end{aligned} $$ 

At this point, let us define  $ \pi_{\phi}^{\mathrm{out}}(a|s)=\frac{\bar{\phi}(s,a)}{\sum_{a}\bar{\phi}(s,a)} $  and  $ \pi_{\psi}^{\mathrm{out}}(a|s)=\frac{\bar{\psi}(s,a)}{\sum_{a}\bar{\psi}(s,a)} $ . For such policies and by appropriate choice for  $ \psi^{\star} $  and  $ \phi^{\star} $  it follows that

 $$ \mathrm{G a p}_{s_{1}}=\max_{\psi}V^{\pi_{\phi}^{\mathrm{o u t}},\psi}(s_{1})-\min_{\phi}V^{\phi,\pi_{\psi}^{\mathrm{o u t}}}(s_{1}). $$ 

By the bound on  $ Gap_{s_{1}} $  for each  $ s_{1} \in supp(\nu_{1}) $ , it follows that

 $$ \left\langle\nu_{1},\max_{\psi}V^{\pi_{\phi}^{\mathrm{out}},\psi}-\min_{\phi}V^{\phi,\pi_{\psi}^{\mathrm{out}}}\right\rangle=\mathbb{E}_{s_{1}\sim\nu_{1}}\mathrm{G a p}_{s_{1}}\leq\frac{10H\log\underline{d}^{-1}}{\beta T}, $$ 

therefore  $ T \geq \frac{10H \log d^{-1}}{\beta \epsilon} $ . The proof is concluded invoking Thm. 6 that ensures that the policies  $ \pi_{\psi}^{out} $  and  $ \pi_{\phi}^{out} $  coincide.

### D.5 Proof of Theorem 6

Proof. Let us consider two players performing the following updates

and

 $$ \begin{aligned}&\phi_{h}^{t+1}=\underset{\phi\in\mathcal{F}_{s_{1}}}{\arg\max}\beta\left\langle\phi,2\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t}}r_{h}(\cdot,\cdot,s^{\prime},a^{\prime})-\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t-1}}r_{h}(\cdot,\cdot,s^{\prime},a^{\prime})\right\rangle-\mathbb{D}(\phi,\phi_{h}^{t}),\\ &\\&\psi_{h}^{t+1}=\underset{\psi\in\mathcal{F}_{s_{1}}}{\arg\min}\beta\left\langle\psi,2\mathbb{E}_{s^{\prime},a^{\prime}\sim\phi^{t}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)-\mathbb{E}_{s^{\prime},a^{\prime}\sim\phi^{t-1}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)\right\rangle+\mathbb{D}(\psi,\psi_{h}^{t}).\\ \end{aligned} $$ 

The goal is to proof that the iterates generated by the two updates are identical. We will prove this fact by induction. The base case holds by initialization which gives  $ \phi_{h}^{0} = \psi_{h}^{0} $  for all  $ h \in [H] $ . Then, let us assume by the induction step that  $ \psi_{h}^{t} = \phi_{h}^{t} $  for all  $ h \in [H] $ , then

 $$ \begin{aligned}&\phi_{h}^{t+1}\\ &=\arg\max_{\phi\in\mathcal{F}_{s_{1}}}\beta\left\langle\phi,2\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t}}r_{h}(\cdot,\cdot,s^{\prime},a^{\prime})-\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t-1}}r_{h}(\cdot,\cdot,s^{\prime},a^{\prime})\right\rangle-\mathbb{D}(\phi,\phi_{h}^{t})\\ &=\arg\max_{\phi\in\mathcal{F}_{s_{1}}}\beta\left\langle\phi,-2\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)+\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t-1}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)\right\rangle-\mathbb{D}(\phi,\phi_{h}^{t})+\beta\left\langle\phi,\mathbf{1}\right\rangle\\ &(Antisymmetric Reward)\\ &=\underset{\phi\in\mathcal{F}_{s_{1}}}{\arg\max}\beta\left\langle\phi,-2\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)+\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t-1}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)\right\rangle-\mathbb{D}(\phi,\phi_{h}^{t})+\beta\\ &(Normalization of\phi)\\ &=\underset{\phi\in\mathcal{F}_{s_{1}}}{\arg\max}\beta\left\langle\phi,-2\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)+\mathbb{E}_{s^{\prime},a^{\prime}\sim\psi^{t-1}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)\right\rangle-\mathbb{D}(\phi,\phi_{h}^{t})\\ &(\beta does not depend on\phi)\\ &=\underset{\phi\in\mathcal{F}_{s_{1}}}{\arg\max}\beta\left\langle\phi,-2\mathbb{E}_{s^{\prime},a^{\prime}\sim\phi^{t}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)+\mathbb{E}_{s^{\prime},a^{\prime}\sim\phi^{t-1}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)\right\rangle-\mathbb{D}(\phi,\psi_{h}^{t})\\ &(Inductive Hypothesis)\\ &=\underset{\psi\in\mathcal{F}_{s_{1}}}{\arg\min}\beta\left\langle\psi,2\mathbb{E}_{s^{\prime},a^{\prime}\sim\phi^{t}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)-\mathbb{E}_{s^{\prime},a^{\prime}\sim\phi^{t-1}}r_{h}(s^{\prime},a^{\prime},\cdot,\cdot)\right\rangle+\mathbb{D}(\psi,\psi_{h}^{t})\\ &(Renaming the optimization variable and\underset{x}{\arg\max}f(x)=\underset{x}{\arg\min}-f(x))\\ &=\psi_{h}^{t+1}.\\ \end{aligned} $$ 

## E IMPLEMENTATION OF ALGORITHM 3 WITH UPDATES OVER POLICIES

In this section, we explain how the update in Algorithm 3 for different choices of D. In both cases, we will derive an update that can be summarized by following template. Let us define  $  r_{h}^{t}(s,a) = \mathbb{E}_{s',a'\sim d_{h}^{t}} r(s,a,s',a')  $  and  $  r_{h}^{t-1}(s,a) = \mathbb{E}_{s',a'\sim d_{h}^{t-1}} r(s,a,s',a')  $ 

• Compute the  $ Q_{h}^{t} $  function corresponding to the reward function  $ 2r_{h}^{t} - r_{h}^{t-1} $  minimizing a loss function that depends on the choice of D.

• Update the policy as

 $$ \pi_{h}^{t+1}(a|s)\propto\pi_{h}^{t}(a|s)\exp\left(\beta Q_{h}^{t}(s,a)\right). $$ 

Finally, in Appx. E.3 we show that for D being the conditional relative entropy and for  $ \beta $  small enough the value function  $ Q_{h}^{t} $  is well approximated by the standard Bellman equations.

Remark 6. Both choices of the Bregman divergence are 1 strongly convex so Thm. 5 applies with  $ \lambda = 1 $ .

In the following we consider a generic reward function  $ \tilde{r} $ . In our setting, we will apply the following results for  $ \tilde{r}_{h}=2r_{h}^{t}-r_{h}^{t-1} $  in order to implement the updates of Alg. 3 for the different values of h and t.

### E.1 D CHOSEN AS THE SUM OF CONDITIONAL AND RELATIVE ENTROPY

In this section, we explain how to implement the occupancy measure update in Algorithm 3 over policies. We use the machinery for single agent MDPs introduced in Bas-Serrano et al. (2021). In particular, we consider the Bregman divergence given by the sum of the relative entropy  $ D(d,d^{\prime})=\sum_{s,a}d(s,a)\log\left(\frac{d(s,a)}{d^{\prime}(s,a)}\right) $  and of the conditional relative entropy given, i.e.  $ H(d,d^{\prime})=\sum_{s,a}d(s,a)\log\left(\frac{\pi_{d}(a|s)}{\pi_{d^{\prime}}(a|s)}\right) $  with  $ \pi_{d}(a|s)=d(s,a)/\sum_{a}d(s,a) $ . Under this choice for D, the update of Algorithm 3 for particular values of h, t,  $ s_{1} $  corresponds to the solution of the following optimization program

 $$ \begin{align*}d_{h}^{t+1}=\underset{d\in\Delta^{H}}{\arg\max}\sum_{h=1}^{H}\langle d_{h},\tilde{r}_{h}\rangle-\frac{1}{\beta}D(d_{h},d_{h}^{t})-\frac{1}{\beta}H(d_{h},d_{h}^{t}),\\ s.t.\quad E^{T}d_{h}=F^{T}d_{h-1}\quad\forall h\in[H].\end{align*} $$ 

Theorem 7. The policy  $ \pi_{h}^{t+1} $  with occupancy measure  $ d_{h}^{t+1} $  defined in Eq. (Update I) can be computed as follows

 $$ \pi_{h}^{t+1}(a|s)\propto\pi_{h}^{t}(a|s)\exp\left(\beta Q_{h}^{t}(s,a)\right), $$ 

where  $ Q_{h}^{t} $  is the minimizer of the following loss

 $$ \frac{1}{\beta}\sum_{h=1}^{H}\log\sum_{s,a}\mu_{h}^{t}(s,a)\exp\left(\beta(2\tilde{r}_{h}+P V_{h+1}-Q_{h})(s,a)\right)+\left\langle\nu_{1},V_{1}\right\rangle, $$ 

while  $ V_{h+1}^{t} $  is given by the following closed form.

 $$ V_{h+1}^{t}(s)=\frac{1}{\beta}\log\sum_{a}\pi_{h}^{t}(a|s)\exp(\beta Q_{h+1}^{t}(s,a)). $$ 

Proof. Let us introduce an auxiliary variable  $ \mu_{h} = d_{h} $  for all  $ h \in [H] $ , then we can rewrite the optimization program as

 $$ \begin{align*}\underset{d\in\Delta^{H}}{\arg\max}\max_{\mu\in\Delta^{H}}\sum_{h=1}^{H}&\langle\mu_{h},\tilde{r}_{h}\rangle-\frac{1}{\beta}D(\mu_{h},\mu_{h}^{t})-\frac{1}{\beta}H(d_{h},d_{h}^{t}),\\ s.t.\quad&E^{T}d_{h}=F^{T}\mu_{h-1}\quad\forall h\in[H],\\ s.t.\quad&\mu_{h}=d_{h}\quad\forall h\in[H].\end{align*} $$ 

Then, by Lagrangian duality we have that

 $$ \begin{align*}\max_{d\in\Delta^{H}}&\max_{\mu\in\Delta^{H}}\min_{Q,V}\sum_{h=1}^{H}\left\langle\mu_{h},\tilde{r}\right\rangle-\frac{1}{\beta}D(\mu_{h},\mu_{h}^{t})-\frac{1}{\beta}H(d_{h},d_{h}^{t})\\&+\left\langle-E^{T}d_{h}+F^{T}\mu_{h-1},V_{h}\right\rangle+\left\langle Q_{h},d_{h}-\mu_{h}\right\rangle\\&=\max_{d\in\Delta^{H}}\max_{\mu\in\Delta^{H}}\min_{Q,V}\sum_{h=1}^{H}\left\langle\mu_{h},\tilde{r}+FV_{h+1}-Q_{h}\right\rangle+\left\langle d_{h},Q_{h}-EV_{h}\right\rangle\\&\quad-\frac{1}{\beta}D(\mu_{h},\mu_{h}^{t})-\frac{1}{\beta}H(d_{h},d_{h}^{t})\\&\quad+\left\langle\nu_{1},V_{1}\right\rangle=\mathcal{L}^{\star}.\end{align*} $$ 

Then, by Lagrangian duality, we have that the objective is unchanged by swapping the min and max

 $$ \begin{align*}\mathcal{L}^{\star}&=\min_{Q,V}\max_{d\in\Delta^{H}}\max_{\mu\in\Delta^{H}}\sum_{h=1}^{H}\langle\mu_{h},\tilde{r}_{h}+F V_{h+1}-Q_{h}\rangle+\langle d_{h},Q_{h}-E V_{h}\rangle\\&-\frac{1}{\beta}D(\mu_{h},\mu_{h}^{t})-\frac{1}{\beta}H(d_{h},d_{h}^{t})+\langle\nu_{1},V_{1}\rangle.\end{align*} $$ 

The inner maximization is solved by the following values

 $$ \begin{align*}\mu_{h}^{+}(Q,V)&\propto\mu_{h}^{t}\odot\exp\left(\beta(\tilde{r}_{h}+F V_{h+1}-Q_{h})\right),\\\pi_{h}^{+}(Q,V;s)&\propto\pi_{h}^{t}(\cdot|s)\odot\exp\left(\beta(Q_{h}(s,\cdot)-V_{h}(s))\right),\end{align*} $$ 

where  $ \odot $  denotes the elementwise product between vectors. Then, replacing these values in the Lagrangian and parameterizing the functions  $ V_{h} $  by the functions  $ Q_{h} $  to ensure normalization of the policy, i.e.  $ V_{h}(s) = \frac{1}{\beta} \log \sum_{a} \pi_{h}^{t}(a|s) \exp(\beta Q_{h}(s,a)) $  we have that

 $$ \mathcal{L}^{\star}=\min_{Q}\frac{1}{\beta}\sum_{h=1}^{H}\log\sum_{s,a}\mu_{h}^{t}(s,a)\exp\left(\beta(\tilde{r}_{h}+F V_{h+1}-Q_{h})(s,a)\right)+\left\langle\nu_{1},V_{1}\right\rangle. $$ 

Therefore, denoting

 $$ Q_{h}^{t}=\underset{Q}{\arg\min}\frac{1}{\beta}\sum_{h=1}^{H}\log\sum_{s,a}\mu_{h}^{t}(s,a)\exp\left(\beta(\tilde{r}_{h}+F V_{h+1}-Q_{h})(s,a)\right)+\left\langle\nu_{1},V_{1}\right\rangle, $$ 

and  $ V_{h}^{t} = \frac{1}{\beta} \log \sum_{a} \pi_{h}^{t}(a|s) \exp(\beta Q_{h}^{t}(s,a)) $ , we have that the policy  $ \pi_{h}^{t+1}(\cdot|s) = \pi_{h}^{+}(Q^{t}, V^{t}; s) $  has occupancy measure equal to  $ d_{h}^{t+1} $  for all  $ h \in [H] $ . This is because by the constraints of the problem we have that  $ d_{h}^{t+1} $  satisfies the Bellman flow constraints and that the policy  $ \pi_{h}^{t+1} $  satisfies  $ \pi_{h}^{t+1}(a|s) = d_{h}^{t}(s,a) / \sum_{a} d_{h}^{t}(s,a) $ . ☐

#### E.2 D CHOSEN AS CONDITIONAL RELATIVE ENTROPY NEU ET AL. (2017)

In this section, we study the update considering D chosen as sum of the conditional relative entropy over the stages  $ h' $  s.t.  $ 1 \leq h' \leq h $ , i.e. we study the following update. $ ^{5} $ 

 $$ \begin{aligned}d^{t+1}=\underset{d\in\Delta^{H}}{\arg\max}\sum_{h=1}^{H}\left(\langle d_{h},\tilde{r}_{h}\rangle-\frac{1}{\beta}\sum_{h^{\prime}=1}^{h}H(d_{h^{\prime}},d_{h^{\prime}}^{t})\right),\\ s.t.\quad E^{T}d_{h}=F^{T}d_{h-1}\quad\forall h\in[H].\end{aligned} $$ 

Theorem 8. The policy  $ \pi_{h}^{t+1} $  with occupancy measure  $ d_{h}^{t+1} $  defined in Eq. (6) can be computed as follows

 $$ \pi_{h}^{t+1}(a|s)\propto\pi_{h}^{t}(a|s)\exp\left(\frac{\beta}{H-h+1}(Q_{h}^{t}(s,a))\right), $$ 

where  $ Q_{h}^{t} $  and  $ V_{h+1}^{t} $  satisfies the following recursion

 $$ \begin{aligned}&Q_{h}^{t}=\tilde{r}_{h}+F V_{h+1}^{t}\\&V_{h+1}^{t}(s)=\frac{H-h+1}{\beta}\log\sum_{a}\pi_{h}^{t}(a|s)\exp\left(\frac{\beta}{H-h+1}Q_{h+1}^{t}(s,a)\right).\\ \end{aligned} $$ 

Remark 7. The above recurrences are sometimes called soft Bellman equations Ziebart (2010); Fox et al. (2015).

Proof. Let us introduce an auxiliary variable $\mu_{h}=d_{h}$ for all $h\in[H]$, then we can rewrite the optimization program as

 $$ \begin{aligned}\underset{d\in\Delta^{H}}{\arg\max}\underset{\mu}{\max}\sum_{h=1}^{H}&\left(\langle\mu_{h},\tilde{r}_{h}\rangle-\frac{1}{\beta}\sum_{h^{\prime}=1}^{h}H(d_{h^{\prime}},d_{h^{\prime}}^{t})\right)\\ s.t.\quad&E^{T}d_{h}=F^{T}\mu_{h-1}\quad\forall h\in[H]\\ s.t.\quad&\mu_{h}=d_{h}\quad\forall h\in[H].\end{aligned} $$ 

Notice that importantly, we do not constraint the variable  $ \mu $ . Then, by Lagrangian duality we have that

 $$ \begin{align*}\max_{d\in\Delta^{H}}&\max_{\mu}\min_{Q,V}\sum_{h=1}^{H}\left\langle\mu_{h},\tilde{r}_{h}\right\rangle-\frac{1}{\beta}\sum_{h^{\prime}=1}^{h}H(d_{h^{\prime}},d_{h^{\prime}}^{t})\\&+\left\langle-E^{T}d_{h}+F^{T}\mu_{h-1},V_{h}\right\rangle+\left\langle Q_{h},d_{h}-\mu_{h}\right\rangle\\&=\max_{d\in\Delta^{H}}\max_{\mu}\min_{Q,V}\sum_{h=1}^{H}\left\langle\mu_{h},\tilde{r}_{h}+FV_{h+1}-Q_{h}\right\rangle+\left\langle d_{h},Q_{h}-EV_{h}\right\rangle\\&-\frac{1}{\beta}\sum_{h^{\prime}=1}^{h}H(d_{h^{\prime}},d_{h^{\prime}}^{t})+\left\langle\nu_{1},V_{1}\right\rangle\\&=\min_{Q,V}\max_{d\in\Delta^{H}}\max_{\mu}\sum_{h=1}^{H}\left\langle\mu_{h},\tilde{r}_{h}+FV_{h+1}-Q_{h}\right\rangle+\left\langle d_{h},Q_{h}-EV_{h}\right\rangle\\&-\frac{H-h+1}{\beta}H(d_{h},d_{h}^{t})+\left\langle\nu_{1},V_{1}\right\rangle=\tilde{\mathcal{L}}^{\star},\end{align*} $$ 

where the last equality holds by Lagrangian duality and by  $ \sum_{h=1}^{H}\sum_{h^{\prime}=1}^{h}H(d_{h^{\prime}},d_{h^{\prime}}^{t})=\sum_{h=1}^{H}(H-h+1)H(d_{h^{\prime}},d_{h^{\prime}}^{t}) $ . Now since  $ \mu $  is unconstrained we have that  $ \max_{\mu}\sum_{h=1}^{H}\langle\mu_{h},\tilde{r}_{h}+FV_{h+1}-Q_{h}\rangle $  is equivalent to impose the constraint  $ \tilde{r}_{h}+FV_{h+1}=Q_{h} $  for all  $ h\in[H] $ . Moreover, as in the proof of Thm. 7 the optimal  $ d_{h} $  needs to satisfy that  $ \pi_{d_{h}}(a|s)=d_{h}(s,a)/\sum_{a}d_{h}(s,a) $  is equal to  $ \pi_{h}^{+}(Q,V;s)=\pi_{h}^{t}(\cdot|s)\odot\exp\left(\frac{\beta}{H-h+1}(Q_{h}(s,\cdot)-V_{h}(s))\right) $  for  $ V_{h}(s)=\frac{H-h+1}{\beta}\log\sum_{a}\pi_{h}^{t}(a|s)\exp\left(\frac{\beta}{H-h+1}Q_{h}(s,a)\right) $ . Plugging in, these facts in the expression for  $ \tilde{L}^{\star} $ , we have that

 $$ \tilde{\mathcal{L}}^{\star}=\min_{Q}\left\langle\nu_{1},V_{1}\right\rangle\quad s.t.\quad\tilde{r}_{h}+FV_{h+1}=Q_{h}\quad\forall h\in[H]. $$ 

Since the above problem as only one feasible point, we have that the solution is the sequence  $ Q_{h}^{t} $  satisfying the recursion  $ \tilde{r}_{h} + FV_{h+1}^{t} = Q_{h}^{t} $  with  $ V_{h}^{t}(s) = \frac{H-h+1}{\beta}\log\sum_{a}\pi_{h}^{t}(a|s)\exp\left(\frac{\beta}{H-h+1}Q_{h}^{t}(s,a)\right) $ .

### E.3 APPROXIMATING SOFT BELLMAN EQUATIONS BY STANDARD BELLMAN EQUATIONS

Unfortunately, implementing the update for the V value as in Theorem 7 is often numerically instable. In this section, we show a practical approximation which is easy to implement and shown to be accurate for  $ \beta $  sufficiently small.

Theorem 9. Let us denote  $ \beta_{h}=\frac{\beta}{H-h+1} $  and let us assume that the values  $ Q_{h}^{t} $  generated by the soft Bellman equations in Thm. 8 are uniformly upper bounded by  $ Q_{max} $ , and let us choose  $ \beta_{h}\leq\frac{1}{Q_{max}} $  for all  $ h\in[H] $ . Then, it holds that

 $$ \begin{align*}\left\langle\pi_{h}^{t}(\cdot|s),Q_{h}^{t}(s,\cdot)\right\rangle\leq\frac{1}{\beta_{h}}\log\sum_{a}\pi_{h}^{t}(a|s)\exp(\beta_{h}Q_{h}^{t}(s,a))\leq\left\langle\pi_{h}^{t}(\cdot|s),Q_{h}^{t}(s,\cdot)\right\rangle+\beta_{h}Q_{\max}^{2}.\end{align*} $$ 

Proof.

 $$ \begin{align*}\frac{1}{\beta_{h}}\log\sum_{a}\pi_{h}^{t}(a|s)\exp(\beta_{h} Q_{h}^{t}(s,a))&\geq\frac{1}{\beta_{h}}\sum_{a}\pi_{h}^{t}(a|s)\log\exp(\beta_{h} Q_{h}^{t}(s,a))\\&=\left\langle\pi_{h}^{t}(\cdot|s),Q_{h}^{t}(s,\cdot)\right\rangle,\end{align*} $$ 

where the above inequality holds for Jensen's. For the upper bound, we first use the inequality  $ e^{x} \leq 1 + x + x^{2} $  for  $ x \leq 1 $  we have that

 $$ \begin{align*}&\frac{1}{\beta_{h}}\log\sum_{a}\pi_{h}^{t}\exp(\beta_{h} Q_{h}^{t}(s,a))\\&\leq\frac{1}{\beta_{h}}\log\sum_{a}\pi_{h}^{t}(1+\beta_{h} Q_{h}^{t}(s,a)+\beta_{h}^{2}Q_{\max}^{2})\quad(Using Q_{h}^{t}(s,a)\leq Q_{\max})\\&=\frac{1}{\beta_{h}}\log(1+\beta_{h}\sum_{a}\pi_{h}^{t}(a|s)Q_{h}^{t}(s,a)+\beta_{h}^{2}Q_{\max}^{2})\\&\leq\frac{1}{\beta_{h}}\left(\sum_{a}\pi_{h}^{t}(a|s)\beta_{h} Q_{h}^{t}(s,a)+\beta_{h}^{2}Q_{\max}^{2}\right)\quad(Using\log(1+x)\leq x)\\&\leq\left\langle\pi_{h}^{t}(\cdot|s),Q_{h}^{t}(s,\cdot)\right\rangle+\beta_{h} Q_{\max}^{2}.\end{align*} $$ 

Remark 8. Given this result, in the implementation for deep RL experiment, i.e. Algorithm 4 we compute the standard Q value satisfying the standard Bellman equations (given in Lemma 1) rather than the soft Bellman equation in Thm. 7. In virtue of Thm. 9, the approximation is good for  $ \beta $  reasonably small.

## F ADDITIONAL EXPERIMENT

### F.1 EXPERIMENT IN MT-BENCH 101

The tasks in MT-bench 101 include Context Memory (CM), Anaphora Resolution (AR), Separate Input (SI), Topic Shift (TS), Content Confusion (CC), Content Rephrasing (CR), Format Rephrasing (FR), Self-correction (SC), Self-affirmation (SA), Mathematical Reasoning (MR), General Reasoning (GR), Instruction Clarification (IC), and Proactive Interaction (PI). We list the description of each task in Tab. 3. The default evaluation mode of MT-bench 101 is that the GPT model requires to access the conversation based on the given ground truth of previous steps, provided in MT-bench 101. However, in our problem setting, the answers among the conversation is also generated by the model. We use “gpt-4o-mini-2024-07-18” to evaluate the conversation. The maximum output length and maximum sequence length of gpt-4o are set as 4096. We use a batch size of 8 with a temperature of 0.8. We use the same prompt for gpt-4o as in Bai et al. (2024). Our experiment is conducted on 4 H200 GPUs. We use the PyTorch platform and the Transformer Reinforcement Learning (TRL) for finetuning.

<div style="text-align: center;">Table 3: A detailed description of each task in MT-bench 101 (taken from Bai et al. (2024).)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Task</td><td style='text-align: center;'>Abbr.</td><td style='text-align: center;'>Description</td></tr><tr><td style='text-align: center;'>Context Memory</td><td style='text-align: center;'>CM</td><td style='text-align: center;'>Recall early dialogue details to address the user&#x27;s current question.</td></tr><tr><td style='text-align: center;'>Anaphora Resolution</td><td style='text-align: center;'>AR</td><td style='text-align: center;'>Identify pronoun referents throughout a multi-turn dialogue.</td></tr><tr><td style='text-align: center;'>Separate Input</td><td style='text-align: center;'>SI</td><td style='text-align: center;'>The first turn outlines the task requirements and the following turns specify the task input.</td></tr><tr><td style='text-align: center;'>Topic Shift</td><td style='text-align: center;'>TS</td><td style='text-align: center;'>Recognize and focus on the new topic when users unpredictably switch topics.</td></tr><tr><td style='text-align: center;'>Content Confusion</td><td style='text-align: center;'>CC</td><td style='text-align: center;'>Avoid interference from similar-looking queries with distinct meanings in the dialogue&#x27;s history.</td></tr><tr><td style='text-align: center;'>Content Rephrasing</td><td style='text-align: center;'>CR</td><td style='text-align: center;'>Rephrase the content of the last response according to the user&#x27;s newest requirement.</td></tr><tr><td style='text-align: center;'>Format Rephrasing</td><td style='text-align: center;'>FR</td><td style='text-align: center;'>Rephrase the format of the last response according to the user&#x27;s newest requirement.</td></tr><tr><td style='text-align: center;'>Self-correction</td><td style='text-align: center;'>SC</td><td style='text-align: center;'>Recorrect the last response according to the user feedback.</td></tr><tr><td style='text-align: center;'>Self-affirmation</td><td style='text-align: center;'>SA</td><td style='text-align: center;'>Preserve the last response against inaccurate user feedback.</td></tr><tr><td style='text-align: center;'>Mathematical Reasoning</td><td style='text-align: center;'>MR</td><td style='text-align: center;'>Collaboratively solve complex mathematical problems with users across dialogue turns.</td></tr><tr><td style='text-align: center;'>General Reasoning</td><td style='text-align: center;'>GR</td><td style='text-align: center;'>Collaboratively solve complex general reasoning problems with users across dialogue turns.</td></tr><tr><td style='text-align: center;'>Instruction Clarification</td><td style='text-align: center;'>IC</td><td style='text-align: center;'>Seek clarification by asking further questions on ambiguous user queries.</td></tr><tr><td style='text-align: center;'>Proactive Interaction</td><td style='text-align: center;'>PI</td><td style='text-align: center;'>Propose questions in reaction to user statements to spark their interest to continue the dialogue.</td></tr></table>

Next, we provide the comparison between the proposed MPO and IPO (Azar et al., 2024), which also uses the squared loss and bypasses the BT model assumption. We run both IPO and MPO for one iteration. The results in Tab. 4 show that MPO achieves a higher average score than IPO.

<div style="text-align: center;">Table 4: Comparison between MPO and IPO in MT-BENCH 101 dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3">Model</td><td rowspan="3">Avg.</td><td colspan="5">Perceptivity</td><td colspan="5">Adaptability</td><td colspan="2">Interactivity</td></tr><tr><td style='text-align: center;'>Memory</td><td colspan="2">Understanding</td><td colspan="2">Interference</td><td style='text-align: center;'>Rephrasing</td><td colspan="2">Reflection</td><td colspan="2">Reasoning</td><td colspan="2">Questioning</td></tr><tr><td style='text-align: center;'>CM</td><td style='text-align: center;'>SI</td><td style='text-align: center;'>AR</td><td style='text-align: center;'>TS</td><td style='text-align: center;'>CC</td><td style='text-align: center;'>CR</td><td style='text-align: center;'>FR</td><td style='text-align: center;'>SC</td><td style='text-align: center;'>SA</td><td style='text-align: center;'>MR</td><td style='text-align: center;'>GR</td><td style='text-align: center;'>IC</td></tr><tr><td style='text-align: center;'>Base (Mistral-7B-Instruct)</td><td style='text-align: center;'>6.223</td><td style='text-align: center;'>7.202</td><td style='text-align: center;'>7.141</td><td style='text-align: center;'>7.477</td><td style='text-align: center;'>7.839</td><td style='text-align: center;'>8.294</td><td style='text-align: center;'>6.526</td><td style='text-align: center;'>6.480</td><td style='text-align: center;'>4.123</td><td style='text-align: center;'>4.836</td><td style='text-align: center;'>4.455</td><td style='text-align: center;'>5.061</td><td style='text-align: center;'>5.818</td></tr><tr><td style='text-align: center;'>IPO</td><td style='text-align: center;'>6.498</td><td style='text-align: center;'>7.518</td><td style='text-align: center;'>7.480</td><td style='text-align: center;'>7.759</td><td style='text-align: center;'>7.952</td><td style='text-align: center;'>8.652</td><td style='text-align: center;'>6.892</td><td style='text-align: center;'>6.768</td><td style='text-align: center;'>4.390</td><td style='text-align: center;'>5.185</td><td style='text-align: center;'>4.313</td><td style='text-align: center;'>5.378</td><td style='text-align: center;'>6.146</td></tr><tr><td style='text-align: center;'>MPO</td><td style='text-align: center;'>6.630</td><td style='text-align: center;'>7.624</td><td style='text-align: center;'>7.846</td><td style='text-align: center;'>8.085</td><td style='text-align: center;'>8.398</td><td style='text-align: center;'>8.947</td><td style='text-align: center;'>7.105</td><td style='text-align: center;'>7.286</td><td style='text-align: center;'>4.208</td><td style='text-align: center;'>4.993</td><td style='text-align: center;'>4.377</td><td style='text-align: center;'>5.264</td><td style='text-align: center;'>6.179</td></tr></table>

We now present an ablation study to evaluate the benefits of incorporating terminal rewards. Using MPO, we compare two approaches for optimizing  $ a_{h} $ : one computes the preference signal based on the terminal state  $ s_{H+1} $ , while the other uses the immediate next state  $ s_{h} $ . The results within one iteration for the MT-Bench 101 dataset are shown in Tab. 5, and those for the GSM/Math experiments are provided in Tab. 6. Our findings reveal that using the terminal state  $ s_{H+1} $  performs worse than using the immediate state  $ s_{h} $  in MT-Bench 101. In contrast, the difference in performance is negligible in the GSM/Math tasks. The underlying reason is that in multi-turn conversational datasets, especially when adjacent questions are not closely related, relying on preferences derived from the terminal state can introduce noise. However, in math and reasoning tasks, the terminal state often captures the final answer, making it more critical. Moreover, using  $ s_{H+1} $  for preference signals is significantly more computationally expensive than using  $ s_{h} $ , due to the extended sequence length. Consequently, we conclude that adapting the choice of terminal preference or intermediate preference on the task's characteristics is crucial for balancing performance and efficiency.

### F.2 Tabular Experiment

<div style="text-align: center;"><img src="imgs/img_in_chart_box_427_878_798_1131.jpg" alt="Image" width="30%" /></div>


<div style="text-align: center;">Figure 2: Results in the tabular experiments. Curves are averages across 10 different randomly generated environments. The error bars report one standard deviation.</div>


The setting of our large-scale experiments does not match the assumptions under which Thm. 5 is proven. In particular, in the large scale experiments the state action value functions cannot be computed exactly. In this section, we consider a synthetic experiment in which the state action functions can be computed exactly for both OMPo and MPO. We generate 10 random gridworlds with a number of states and actions sample uniformly from the intervals  $ [1, 100] $  and  $ [2, 10] $ . We plot the exploitability computed as

 $$ \left\langle\nu_{1},\underset{\pi}{\max}V^{\pi,\pi^{k}}-V^{\pi^{k}\pi^{k}}\right\rangle $$ 

which is a standard metric to evaluate the distance from a Nash equilibrium. In particular, when  $ (\pi^{k},\pi^{k}) $  is a Nash equilibrium, the exploitability is 0. We can see that OMPO achieves very low exploitability after 100 updates while 2000 updates are needed by MPO. In this case, where the

<div style="text-align: center;">Table 5: Ablation on terminal reward in MT-BENCH 101 dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3">Model</td><td rowspan="3">Avg.</td><td colspan="4">Perceptivity</td><td colspan="5">Adaptability</td><td style='text-align: center;'>Interactivity</td></tr><tr><td style='text-align: center;'>Memory</td><td colspan="2">Understanding</td><td style='text-align: center;'>Interference</td><td style='text-align: center;'>Rephrasing</td><td style='text-align: center;'>Reflection</td><td colspan="2">Reasoning</td><td colspan="2">Questioning</td></tr><tr><td style='text-align: center;'>CM</td><td style='text-align: center;'>SI</td><td style='text-align: center;'>AR</td><td style='text-align: center;'>TS</td><td style='text-align: center;'>CC</td><td style='text-align: center;'>CR</td><td style='text-align: center;'>SC</td><td style='text-align: center;'>SA</td><td style='text-align: center;'>MR</td><td style='text-align: center;'>GR</td></tr><tr><td style='text-align: center;'>Base (Mistral-7B-Instruct)</td><td style='text-align: center;'>6.223</td><td style='text-align: center;'>7.202</td><td style='text-align: center;'>7.141</td><td style='text-align: center;'>7.477</td><td style='text-align: center;'>7.839</td><td style='text-align: center;'>8.294</td><td style='text-align: center;'>6.526</td><td style='text-align: center;'>6.480</td><td style='text-align: center;'>4.123</td><td style='text-align: center;'>4.836</td><td style='text-align: center;'>5.061</td></tr><tr><td style='text-align: center;'>MPO (intermediate reward)</td><td style='text-align: center;'>6.630</td><td style='text-align: center;'>7.624</td><td style='text-align: center;'>7.846</td><td style='text-align: center;'>8.085</td><td style='text-align: center;'>8.398</td><td style='text-align: center;'>8.947</td><td style='text-align: center;'>7.105</td><td style='text-align: center;'>7.286</td><td style='text-align: center;'>4.208</td><td style='text-align: center;'>4.993</td><td style='text-align: center;'>5.264</td></tr><tr><td style='text-align: center;'>MPO (terminal reward)</td><td style='text-align: center;'>6.459</td><td style='text-align: center;'>7.536</td><td style='text-align: center;'>7.328</td><td style='text-align: center;'>7.643</td><td style='text-align: center;'>8.084</td><td style='text-align: center;'>8.518</td><td style='text-align: center;'>6.847</td><td style='text-align: center;'>6.883</td><td style='text-align: center;'>4.357</td><td style='text-align: center;'>4.863</td><td style='text-align: center;'>5.542</td></tr></table>

<div style="text-align: center;">Table 6: Ablation on terminal reward in MATH and GSM8K dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'>GSM8K</td><td style='text-align: center;'>Math</td></tr><tr><td style='text-align: center;'>Base (Qwen2-7B-Instruct)</td><td style='text-align: center;'>0.8559</td><td style='text-align: center;'>0.5538</td></tr><tr><td style='text-align: center;'>MPO (intermediate reward)</td><td style='text-align: center;'>0.8734</td><td style='text-align: center;'>0.5720</td></tr><tr><td style='text-align: center;'>MPO (terminal reward)</td><td style='text-align: center;'>0.8734</td><td style='text-align: center;'>0.5734</td></tr></table>

Q functions can be computed exactly, we can appreciate the faster convergence rate of OMPO as described by Thm. 5.

### F.3 EXPERIMENT ON MATH REASONING TASKS

As discussed in Appx. B, our framework can also cover the alignment of chain-of-thought reasoning. In this section, we validate the proposed methods on math reasoning tasks. We select two widely used datasets: MATH Hendrycks et al. (2021) and GSM8K Cobbe et al. (2021). We use Qwen2-7B-Instruct as the base model and follow the same evaluation procedure as in Lai et al. (2024). We adopt the dataset for alignment from Lai et al. (2024), which contains 10795 samples of augmented mathematical problems from MetaMath (Yu et al., 2024) and MMIQC (Liu et al., 2024b) $ ^{6} $ . For step-DPO, we use the checkpoint provided in Lai et al. (2024). For both MPO and OMPO, we perform full-parameter finetuning for 1 epoch with learning rate  $ 5e^{-7} $  and  $ \beta $  tuned in the range of  $ \{0.1, 0.01, 0.001\} $ . For both MPO and OMPO, we select the Llama-3-based model as the preference oracle $ ^{7} $  and set the  $ \log z $  as set as 0.5. The final state with the answer is important in this task so we only use the terminal reward (see Tab. 6 for comparison). We use AdamW optimizer (Loshchilov & Hutter, 2019) and cosine learning rate schedule (Loshchilov & Hutter, 2017) with a warmup ratio of 0.1. The experiment is conducted on 4 A100-SXM4-80GB GPUs. The result is provided in Tab. 7, showing that the proposed methods achieve performance comparable to step-DPO (Lai et al., 2024). Notably, MPO and OMPO do not require the ground truth label of the dataset during fine-tuning while Lai et al. (2024) requires it. Additionally, MPO and OMPO need only a Llama3-based pair-preference model to compare two answers. Step-DPO requires GPT-4 to identify the incorrect reasoning step in an answer, which is a considerably more difficult task than comparison.

<div style="text-align: center;">Table 7: Performance of math reasoning on MATH and GSM8K dataset across various models. MPO and OMPA achieve comparable performance comparable to step-DPO without requiring the ground truth label of the dataset during fine-tuning while Lai et al. (2024) requires. Additionally, MPO and OMPA only need access to an oracle Llama-3 to compare two answers whereas step-DPO Lai et al. (2024) requires GPT-4 to locate the identify the incorrect reasoning step in an answer, which is a considerably more difficult task than comparison.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'>GSM8K</td><td style='text-align: center;'>Math</td></tr><tr><td style='text-align: center;'>Base (Qwen2-7B-Instruct)</td><td style='text-align: center;'>0.8559</td><td style='text-align: center;'>0.5538</td></tr><tr><td style='text-align: center;'>Step-DPO (Lai et al., 2024)</td><td style='text-align: center;'>0.8680</td><td style='text-align: center;'>0.5836</td></tr><tr><td style='text-align: center;'>MPO (iter=1)</td><td style='text-align: center;'>0.8734</td><td style='text-align: center;'>0.5734</td></tr><tr><td style='text-align: center;'>MPO (iter=2)</td><td style='text-align: center;'>0.8734</td><td style='text-align: center;'>0.5786</td></tr><tr><td style='text-align: center;'>OMPO (iter=2)</td><td style='text-align: center;'>0.8779</td><td style='text-align: center;'>0.5786</td></tr></table>

## G MOTIVATION OF CONSIDERING INTERMEDIATE REWARD

In this section, we elaborate on the motivation for considering intermediate rewards at each turn instead of only terminal rewards.

In multi-turn conversation tasks, such as MT-bench 101 (Bai et al., 2024), the user asks questions  $ x_{1} $ ,  $ x_{2} $ ,  $ x_{3} $ , and receives answers  $ a_{1} $ ,  $ a_{2} $ ,  $ a_{3} $ . When  $ x_{2} $  is not closely related to  $ x_{1} $ , aligning the first step using feedback among different  $ a_{1} $  is much more helpful than using the sequence  $ [a_{1}, x_{2}, a_{2}] $ , where  $ x_{2} $ ,  $ a_{2} $  can be considered as noise.

In mathematical reasoning tasks, as mentioned in Lai et al. (2024), some cases yield correct final answers but contain errors in intermediate reasoning steps. Consequently, Lai et al. (2024) filter out such samples using GPT-4. For example, consider a case where the reasoning steps yield a correct final answer but include an error:  $ [a_{1}^{correct}, a_{2}^{wrong}, a_{3}^{correct}] $ , where  $ a_{2}^{wrong} $  is incorrect while all of the other steps and the final answer  $ a_{3}^{correct} $  is correct. When there is another response,  $ [a_{1}^{correct}, a_{2}^{correct}, a_{3}^{correct}] $  with all correct steps, using only terminal signal for aligning step 2 might not guarantee that  $ a_{2}^{correct} \succ a_{2}^{wrong} $  because both of final answers are correct, especially when there is only an incorrect step among long reasoning steps. In contrast, an intermediate signal would clearly indicate  $ a_{2}^{correct} \succ a_{2}^{wrong} $ , accurately reflecting the quality of the intermediate steps. In practice, if the final signal is important, e.g., in math reasoning task, then we can use only the terminal reward or the average of terminal reward and intermediate reward, otherwise one can just use the intermediate reward, which is cheaper to collect as compared to assigning reward until the terminal state.