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