I need to summarize the appendix of an AI paper by splitting it into the original subsections. The appendix contains several sections: A (Symbols and Notation), B (Preliminary on Single-Step RLHF), C (Additional Discussion on Related Work), D (Proofs), E (Implementation of Algorithm 3 with Updates Over Policies), F (Additional Experiment), and G (Motivation of Considering Intermediate Reward).

I'll summarize each section concisely, keeping within 500 tokens per section, using accurate academic language, and eliminating redundant expressions.

Let me go through each section:

### Appendix A: Symbols and Notation
This section provides a table of core symbols and notation used in the paper. I'll summarize the key symbols without listing all of them.

### Appendix B: Preliminary on Single-Step RLHF
This section reviews earlier methods in single-step RLHF, including classical RLHF methods with Bradley-Terry reward models, DPO, and recent work avoiding the BT assumption.

### Appendix C: Additional Discussion on Related Work
This section has two subsections:
- C.1: Related work on token-level preference optimization
- C.2: Discussion on the difference from SPPO

### Appendix D: Proofs
This section contains proofs for theoretical results, including Lemma 1, Lemma 2, Theorem 4, Theorem 5, and Theorem 6.

### Appendix E: Implementation of Algorithm 3 with Updates Over Policies
This section explains how to implement Algorithm 3 for different choices of Bregman divergence (D).

### Appendix F: Additional Experiment
This section presents additional experimental results, including:
- F.1: Experiment in MT-bench 101
- F.2: Tabular Experiment
- F.3: Experiment on Math Reasoning Tasks

### Appendix G: Motivation of Considering Intermediate Reward
This section explains why intermediate rewards are considered instead of only terminal rewards.

Now I'll write concise summaries for each section, keeping within the token limit.

Let me start with Appendix A:

### Appendix A: Symbols and Notation
This appendix provides core symbols and notation used in the paper. Key symbols include:
- $x_h$: Prompt at step $h$
- $a_h$: Answer (action) at step $h$
- $s_h$: State at step $h$
- $\pi$: Language model (policy)
- $\nu_1$: Initial state distribution
- $d_h^\pi(s,a)$: Occupancy measure of $\pi$ at stage $h$
- $f$: Transition function
- $o$: Preference oracle
- $\mathbb{P}([s,a],[s',a'])$: Winning probability of $[s,a]$ against $[s',a']$
- $D(p||q)$: KL divergence
- $\mathbb{D}(p||q)$: Bregman Divergence
- $D_t$: Dataset buffet at iteration $t$
- $\Delta_x$: Set of probability distributions over set $\mathcal{X}$

The appendix also introduces compact notation for Bellman flow constraints using matrices $E$ and $F$.

Now for Appendix B:

### Appendix B: Preliminary on Single-Step RLHF
This section reviews single-step RLHF methods. Classical RLHF assumes a Bradley-Terry (BT) reward model, optimizing the KL-constrained RL objective:
$$\pi^* = \arg\max_\pi \mathbb{E}_{x_1\sim\nu_1,a_1\sim\pi(\cdot|x_1)}(r(x_1,a_1) - \beta D(\pi(\cdot|x_1)||\pi_{\text{ref}}(\cdot|x_1)))$$
DPO avoids explicit reward modeling, optimizing:
$$\pi^* = \arg\max_\pi \mathbb{E}_{(x_1,a_1^w,a_1^l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi(a_1^w|x_1)}{\pi_1(a_1^w|x_1)} - \beta\log\frac{\pi(a_1^l|x_1)}{\pi_1(a_1^l|x_1)}\right)\right]$$
Recent work circumvents the BT assumption by modeling the general oracle $\mathbb{P}$ directly, seeking the Nash equilibrium of a two-player constant-sum game:
$$(\pi^*,\pi^*) = \arg\max_\pi\min_{\pi'}\mathbb{E}_{x_1\sim\nu_1,a_1\sim\pi(\cdot|x_1),a_1'\sim\pi'(\cdot|x_1)}\mathbb{P}([x_1,a_1]\succ[x_1,a_1'])$$

Now for Appendix C:

### Appendix C: Additional Discussion on Related Work
#### C.1: Related Work on Token-Level Preference Optimization
This subsection compares multi-step alignment approaches with previous work. While Rafailov et al. (2024), Zeng et al. (2024), and Liu et al. (2024a) develop alignment methods based on the BT model with transitive rewards, the authors' framework is motivated by a two-player game with relative rewards. Additionally, while previous work formulates alignment as an MDP, their final objective is tailored to contextual bandit problems, whereas the authors' objective is designed for multi-step alignment.

#### C.2: Discussion on the Difference from SPPO
The theoretical analysis differs from SPPO due to different settings: SPPO considers contextual bandits and builds analysis based on Freund & Schapire (1999), while the authors frame the problem as a Markov game. The authors' proof uses the performance difference lemma and multiplicative weights updates. The OMPO algorithm updates as:
$$\pi^{t+1}(a|s)\propto\pi^t(a|s)\exp\left[\beta(2\mathbb{P}(a\succ\pi^t(\cdot|s))-\mathbb{P}(a\succ\pi^{t-1}(\cdot|s)))\right]$$
while SPPO updates as:
$$\pi^{t+1}(a|s)\propto\pi^t(a|s)\exp\left[\beta(\mathbb{P}(a\succ\pi^t(\cdot|s)))\right]$$
OMPO achieves $\mathcal{O}(\epsilon^{-1})$ policy updates to converge to an $\epsilon$-approximate Nash equilibrium, compared to SPPO's $\mathcal{O}(\epsilon^{-2})$.

Now for Appendix D:

### Appendix D: Proofs
#### D.1: Proof of Lemma 1
This proof shows the Bellman equation for the state-action value function $Q_h^{\pi,\pi'}(s,a,s',a')$ in vectorial form.

#### D.2: Proof of Lemma 2
This proof uses the Bellman equation and Bellman flow constraints to derive the value difference lemma:
$$\langle\nu_1,V^{\pi,\bar{\pi}}-V^{\pi',\bar{\pi}}\rangle = \sum_{h=1}^H\langle d_h^{\pi,\bar{\pi}},Q_h^{\pi',\bar{\pi}}-EV_h^{\pi',\bar{\pi}}\rangle$$

#### D.3: Proof of Theorem 4
This proof shows that the OMPO algorithm achieves an $\epsilon$-approximate Nash equilibrium in $T = \frac{16H^4 \log \pi^{-1}}{\epsilon^2}$ iterations using the value difference lemma and online mirror descent.

#### D.4: Proof of Theorem 5
This proof demonstrates convergence to an $\epsilon$-approximate Nash equilibrium for the occupancy measure update with $\beta = \sqrt{\frac{\log \pi^{-1}}{TH^2}}$.

#### D.5: Proof of Theorem 6
This proof shows that the iterates generated by the two updates in Theorem 5 are identical, proving that the policies $\pi_\phi^{\text{out}}$ and $\pi_\psi^{\text{out}}$ coincide.

Now for Appendix E:

### Appendix E: Implementation of Algorithm 3 with Updates Over Policies
This appendix explains how to implement Algorithm 3 for different choices of Bregman divergence (D).

#### E.1: D Chosen as the Sum of Conditional and Relative Entropy
For D as the sum of relative entropy and conditional relative entropy, the policy update is:
$$\pi_h^{t+1}(a|s)\propto\pi_h^t(a|s)\exp\left(\beta Q_h^t(s,a)\right)$$
where $Q_h^t$ minimizes a specific loss function.

#### E.2: D Chosen as Conditional Relative Entropy
For D as conditional relative entropy, the policy update is:
$$\pi_h^{t+1}(a|s)\propto\pi_h^t(a|s)\exp\left(\frac{\beta}{H-h+1}Q_h^t(s,a)\right)$$
with $Q_h^t$ satisfying soft Bellman equations.

#### E.3: Approximating Soft Bellman Equations by Standard Bellman Equations
This subsection shows that for small $\beta$, the soft Bellman equations can be approximated by standard Bellman equations with a small error bound.

Now for Appendix F:

### Appendix F: Additional Experiment
#### F.1: Experiment in MT-Bench 101
The authors compare MPO and IPO on MT-bench 101, showing MPO achieves higher average scores. They also conduct an ablation study on terminal vs. intermediate rewards, finding that intermediate rewards perform better for conversational tasks while terminal rewards are more important for math reasoning tasks.

#### F.2: Tabular Experiment
This section presents a synthetic experiment with gridworlds, showing OMPO achieves low exploitability (distance from Nash equilibrium) after 100 updates, while MPO requires 2000 updates.

#### F.3: Experiment on Math Reasoning Tasks
The authors validate their methods on MATH and GSM8K datasets. MPO and OMPO achieve comparable performance to step-DPO without requiring ground truth labels during fine-tuning, and only need a Llama-3-based model to compare answers, unlike step-DPO which requires GPT-4 to identify incorrect reasoning steps.

Now for Appendix G:

### Appendix G: Motivation of Considering Intermediate Reward
This section explains why intermediate rewards are considered instead of only terminal rewards. In multi-turn conversations, aligning each step using feedback on individual responses is more helpful than using the entire sequence, especially when subsequent questions aren't closely related. In math reasoning tasks, intermediate rewards can identify errors in reasoning steps even when the final answer is correct, which terminal rewards might miss. The authors recommend using intermediate rewards for most tasks, with terminal rewards only for math reasoning where the final answer is critical.

Now I'll write the final summaries, making sure each is under 500 tokens and concise.
</think>

### Appendix A Symbols and Notation
This appendix defines core symbols and notation used throughout the paper. Key symbols include: $x_h$ (prompt at step $h$), $a_h$ (answer/action at step $h$), $s_h$ (state at step $h$), $\pi$ (language model policy), $\nu_1$ (initial state distribution), $d_h^\pi(s,a)$ (occupancy measure), $f$ (transition function), $o$ (preference oracle), $\mathbb{P}([s,a],[s',a'])$ (winning probability), $D(p||q)$ (KL divergence), $\mathbb{D}(p||q)$ (Bregman divergence), $D_t$ (dataset buffet), and $\Delta_x$ (probability distributions over set $\mathcal{X}$). The appendix also introduces compact notation for Bellman flow constraints using matrices $E$ and $F$.

### Appendix B Preliminary on Single-Step RLHF
This section reviews single-step RLHF methods. Classical RLHF assumes a Bradley-Terry (BT) reward model, optimizing the KL-constrained RL objective: $\pi^* = \arg\max_\pi \mathbb{E}_{x_1\sim\nu_1,a_1\sim\pi(\cdot|x_1)}(r(x_1,a_1) - \beta D(\pi(\cdot|x_1)||\pi_{\text{ref}}(\cdot|x_1)))$. DPO avoids explicit reward modeling, optimizing: $\pi^* = \arg\max_\pi \mathbb{E}_{(x_1,a_1^w,a_1^l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi(a_1^w|x_1)}{\pi_1(a_1^w|x_1)} - \beta\log\frac{\pi(a_1^l|x_1)}{\pi_1(a_1^l|x_1)}\right)\right]$. Recent work circumvents the BT assumption by directly modeling the general oracle $\mathbb{P}$, seeking the Nash equilibrium of a two-player constant-sum game: $(\pi^*,\pi^*) = \arg\max_\pi\min_{\pi'}\mathbb{E}_{x_1\sim\nu_1,a_1\sim\pi(\cdot|x_1),a_1'\sim\pi'(\cdot|x_1)}\mathbb{P}([x_1,a_1]\succ[x_1,a_1'])$.

### Appendix C Additional Discussion on Related Work
#### C.1 Related Work on Token-Level Preference Optimization
Previous work (Rafailov et al., 2024; Zeng et al., 2024; Liu et al., 2024a) formulates alignment as token-level MDPs based on the BT model with transitive rewards. Their objectives are tailored to contextual bandit problems in LLMs, while the authors' framework is designed for multi-step alignment problems. The key difference is that the authors' approach is motivated by a two-player game with relative rewards rather than the BT assumption.

#### C.2 Discussion on the Difference from SPPO
The authors' theoretical analysis differs from SPPO (Wu et al., 2024) as they frame the problem as a Markov game rather than a contextual bandit. Their proof uses the performance difference lemma and multiplicative weights updates. The OMPO update: $\pi^{t+1}(a|s)\propto\pi^t(a|s)\exp\left[\beta(2\mathbb{P}(a\succ\pi^t(\cdot|s))-\mathbb{P}(a\succ\pi^{t-1}(\cdot|s)))\right]$ achieves $\mathcal{O}(\epsilon^{-1})$ policy updates to converge to an $\epsilon$-approximate Nash equilibrium, compared to SPPO's $\mathcal{O}(\epsilon^{-2})$.

### Appendix D Proofs
#### D.1 Proof of Lemma 1
Derives the Bellman equation for the state-action value function $Q_h^{\pi,\pi'}(s,a,s',a')$ using the tower property of expectation.

#### D.2 Proof of Lemma 2
Uses Bellman flow constraints to derive the value difference lemma: $\langle\nu_1,V^{\pi,\bar{\pi}}-V^{\pi',\bar{\pi}}\rangle = \sum_{h=1}^H\langle d_h^{\pi,\bar{\pi}},Q_h^{\pi',\bar{\pi}}-EV_h^{\pi',\bar{\pi}}\rangle$.

#### D.3 Proof of Theorem 4
Shows OMPO achieves an $\epsilon$-approximate Nash equilibrium in $T = \frac{16H^4 \log \pi^{-1}}{\epsilon^2}$ iterations using online mirror descent and the value difference lemma.

#### D.4 Proof of Theorem 5
Demonstrates convergence to an $\epsilon$-approximate Nash equilibrium for the occupancy measure update with $\beta = \sqrt{\frac{\log \pi^{-1}}{TH^2}}$.

#### D.5 Proof of Theorem 6
Proves the iterates generated by the two updates in Theorem 5 are identical, showing policies $\pi_\phi^{\text{out}}$ and $\pi_\psi^{\text{out}}$ coincide.

### Appendix E Implementation of Algorithm 3 with Updates Over Policies
#### E.1 D Chosen as the Sum of Conditional and Relative Entropy
For D as sum of relative entropy and conditional relative entropy, the policy update is $\pi_h^{t+1}(a|s)\propto\pi_h^t(a|s)\exp\left(\beta Q_h^t(s,a)\right)$, where $Q_h^t$ minimizes a specific loss function.

#### E.2 D Chosen as Conditional Relative Entropy
For D as conditional relative entropy, the policy update is $\pi_h^{t+1}(a|s)\propto\pi_h^t(a|s)\exp\left(\frac{\beta}{H-h+1}Q_h^t(s,a)\right)$, with $Q_h^t$ satisfying soft Bellman equations.

#### E.3 Approximating Soft Bellman Equations by Standard Bellman Equations
Shows that for small $\beta$, soft Bellman equations can be approximated by standard Bellman equations with error bound $\beta_h Q_{\max}^2$, making implementation numerically stable.

### Appendix F Additional Experiment
#### F.1 Experiment in MT-Bench 101
MPO outperforms IPO on MT-bench 101. Ablation shows intermediate rewards (vs. terminal rewards) perform better for conversational tasks, while terminal rewards are more important for math reasoning. Using terminal rewards is computationally more expensive.

#### F.2 Tabular Experiment
Synthetic gridworld experiments show OMPO achieves low exploitability (distance from Nash equilibrium) after 100 updates, while MPO requires 2000 updates, confirming the theoretical convergence rate.

#### F.3 Experiment on Math Reasoning Tasks
MPO and OMPO achieve comparable performance to step-DPO on MATH and GSM8K without requiring ground truth labels during fine-tuning. They only need a Llama-3-based model to compare answers, unlike step-DPO which requires GPT-4 to identify incorrect reasoning steps.

### Appendix G Motivation of Considering Intermediate Reward
Intermediate rewards are preferred over terminal rewards for most tasks. In multi-turn conversations, aligning each step using individual response feedback is more helpful than using the entire sequence, especially when subsequent questions aren't closely related. In math reasoning, intermediate rewards can identify errors in reasoning steps even when the final answer is correct, which terminal rewards might miss. The authors recommend using intermediate rewards for most tasks, with terminal rewards only for math reasoning where the final answer is critical.