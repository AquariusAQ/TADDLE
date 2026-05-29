# ON THE SAMPLE COMPLEXITY OF POLICY GRADIENT ALGORITHM WITH OCCUPANCY APPROXIMATION FOR GENERAL UTILITY REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

## ABSTRACT

Reinforcement learning with general utilities has recently gained attention thanks to its ability to unify several problems, including imitation learning, pure exploration, and safe RL. However, prior work for solving this general problem in a unified way has only focused on the tabular setting. This is restrictive when considering larger state-action spaces because of the need to estimate occupancy measures during policy optimization. In this work, we address this issue and propose to approximate occupancy measures within a function approximation class using maximum likelihood estimation (MLE). We propose a simple policy gradient algorithm (PG-OMA) where an actor updates the policy parameters to maximize the general utility objective whereas a critic approximates the occupancy measure using MLE. We provide a statistical complexity analysis of PG-OMA showing that our occupancy measure estimation error only scales with the dimension of our function approximation class rather than the size of the state action space. Under suitable assumptions, we establish first order stationarity and global optimality performance bounds for the proposed PG-OMA algorithm for nonconcave and concave general utilities respectively. We complement our methodological and theoretical findings with promising empirical results showing the scalability potential of our approach compared to existing tabular count-based approaches.

## 1 INTRODUCTION

Reinforcement learning with general utilities (RLGU) has emerged as a general framework to unify a range of RL applications where the objective of the RL agent cannot be simply cast as a standard expected cumulative reward (Zhang et al., 2022). For instance, in imitation learning, the objective is to learn a policy by minimizing the divergence between the state-action occupancy measure induced by the policy and expert demonstrations (Ho & Ermon, 2016). In pure exploration, the goal is to learn a policy to explore the state space in a reward-free setting by maximizing the entropy of the state occupancy measure induced by the agent's policy (Hazan et al., 2019). Other examples include risk-averse and constrained RL (Garcia & Fernández, 2015), diverse skills discovery (Eysenbach et al., 2019), and experiment design (Mutny et al., 2023).

It is well known that the standard RL objective can be written as a linear functional of the occupancy measure. To capture all the aforementioned applications, the RLGU objective is a possibly nonlinear functional of the state action occupancy measure induced by the policy (Zhang et al., 2022). Due to non-linearity, policy gradient algorithms for solving RLGU problems face the major bottleneck of occupancy measure estimation. Prior works (Hazan et al., 2019; Zhang et al., 2022) have focused on the tabular setting where the state action occupancy measure needs to be estimated for each state action pair using Monte Carlo estimation via sampling trajectories. However, this setting is restrictive for larger state and actions spaces where tabular methods will become intractable due to the curse of dimensionality. This scalability issue stands as an important challenge to overcome to establish RLGU as a general unified framework for which efficient algorithms exist to solve its larger state action space instances. We refer the reader to Figure 1 for an illustration of the challenge motivating our work. Our goal is to address this scalability challenge by proposing a simple algorithm for the general and flexible RLGU framework. In the standard RL setting, several approaches using function approximation have been fruitfully used to approximate action-value functions and scale.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_274_162_501_337.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_496_161_736_337.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_735_161_949_337.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">Figure 1: A Motivating Example: This figure shows the scalability performance of state-of-the-art count-based method of Zhang et al. (2021) in the RLGU setting for a specific application of learning from demonstration (detailed in Sec. 5). We consider three settings easy, medium, and hard and report the episode reward returns. The "easy" setting (left) has  $ 10^{2} $  states, the medium setting (middle) features  $ 10^{3} $  states, and the harder setting (right) comprises  $ 10^{4} $  states. In the easy setting, the count-based method performs relatively well, as expected, since it aims to precisely estimate the occupancy measure (we employ a batch size of B = 100 for estimating the occupancy in each episode). However, as we transition to larger state space settings, it fails to perform due to scalability issues in estimating occupancy measures. This renders the existing general utility RL approach practically inapplicable. The red dotted line shows the oracle's performance.</div>


to large state-action spaces. However, to the best of our knowledge, this issue remains open for RL problems with general utilities. To this end, we summarize our contributions as follows.

Main contributions. In this work, we propose to go beyond the tabular setting in solving RL problems with general utilities. Our contributions are summarized as follows:

• We propose a new policy gradient algorithm, PG-OMA, to solve RLGU where an actor performs policy parameter updates whereas a critic approximates the state-action occupancy measure via maximum likelihood estimation (MLE) within a function approximation class (cf. Sec. 3).

• Theoretical results. We analyze the sample complexity of our algorithm under suitable assumptions. Our analysis relies on a total variation performance bound for occupancy measure approximation via MLE which scales with the dimension of the parameters of the function approximation class rather than the state action space size. Using this result, we establish first-order stationarity and global optimality guarantees for our algorithm for nonconcave and concave general utilities respectively (cf. Sec. 4).

• Experimental evaluations. We conduct experiments on discrete and continuous state-action space environments for learning from demonstration tasks (cf. Sec. 5) to complement our theoretical analysis and show the scalability potential of our approach compared to existing tabular count-based approaches.

Related Works. The general framework of RLGU, also known as convex RL, has been recently introduced in the literature Hazan et al. (2019); Zhang et al. (2021); Zahavy et al. (2021); Geist et al. (2022). Hazan et al. (2019) initially focused on the particular instance of maximum entropy exploration problem and Zhang et al. (2020) proposed a variational policy gradient method to solve the RLGU problem. Zhang et al. (2021) then introduced a simpler (variance-reduced) policy gradient method to solve the (possibly nonconcave) RL problem with general utilities using a simpler policy gradient theorem (see also Kumar et al. (2022)). Later, Barakat et al. (2023) proposed an even simpler single-loop normalized policy gradient algorithm to solve RLGU. Zahavy et al. (2021) leveraged Fenchel duality to cast the convex RL problem into a saddle-point problem that can be solved using standard RL algorithms. In a line of works, Mutti et al. (2022b;a; 2023) formulated the convex RL problem in finite trials instead of infinite realizations and considered an objective which is any convex function of the empirical state distribution computed from a finite number of realizations. Ying et al. (2023a) introduced policy-based primal-dual methods for solving convex constrained CMDPs and Ying et al. (2023b) further addressed a multi-agent RL problem with general utilities. All the aforementioned works focus on the tabular setting. In particular, most of these works use a count-based Monte Carlo estimate of the occupancy measure that cannot scale to large state-action spaces. More recently, Huang et al. (2023) provided sample-efficient online/offline RL.

algorithms with density features in low-rank MDPs for occupancy estimation. See appendix A for an extended related work discussion.

Notations. For a given finite set X, we use the notation  $ |X| $  for its cardinality and  $ \Delta(\mathcal{X}) $  for the space of probability distributions over X. We equip any Euclidean space with its standard inner product denoted by  $ \langle\cdot,\cdot\rangle $ . The notation  $ \|\cdot\| $  refers to both the standard 2-norm for vectors and the spectral norm for matrices. For any vector  $ \lambda\inR^{d} $  where d is an integer, the notation  $ \lambda\geq0 $  means that all the coordinates of the vector  $ \lambda $  are non-negative. We interchangeably denote functions  $ f:X\toR $  over a finite set X as vectors  $ f\inR^{|X|} $  with components  $ f(x) $  with a slight abuse of notations.

## 2 PROBLEM FORMULATION

MDP with General Utility. Consider a discrete-time discounted Markov Decision Process (MDP)  $ (\mathcal{S}, \mathcal{A}, \mathcal{P}, F, \rho, \gamma) $ , where S and A are finite state and action spaces respectively,  $ \mathcal{P}: S \times A \to \Delta(\mathcal{S}) $  is the state transition probability kernel,  $ F: \mathcal{M}(\mathcal{X}) \to \mathbb{R} $  is a general utility function defined over the space of measures  $ \mathcal{M}(\mathcal{X}) $  on the product state-action space  $ X := S \times A $ ,  $ \rho $  is the initial state distribution, and  $ \gamma \in (0, 1) $  is the discount factor. A stationary policy  $ \pi : \mathcal{S} \to \Delta(\mathcal{A}) $  maps each state  $ s \in S $  to a distribution  $ \pi(\cdot | s) $  over the action space A. The set of all stationary policies is denoted by  $ \Pi $ . At each time step  $ t \in N $  in a state  $ s_t \in S $ , the RL agent chooses an action  $ a_t \in A $  with probability  $ \pi(a_t | s_t) $  and then environment transitions to a state  $ s_{t+1} \in S $  with probability  $ \mathcal{P}(s_{t+1} | s_t, a_t) $ . We denote by  $ P_{\rho,\pi} $  the probability distribution of the Markov chain  $ (s_t, a_t)_{t \in \mathbb{N}} $  induced by the policy  $ \pi $  with initial state distribution  $ \rho $ . We use the notation  $ E_{\rho,\pi} $  (or often simply E) for the associated expectation. We define for any policy  $ \pi \in \Pi $  the (normalized) state and state-action occupancy measures  $ d^\pi \in \mathcal{M}(\mathcal{S}, \lambda^\pi \in \mathcal{M}(\mathcal{S} \times \mathcal{A})) $  respectively as:

 $$ d^{\pi}(s):=\left(1-\gamma\right)\sum_{t=0}^{+\infty}\gamma^{t}\mathbb{P}_{\rho,\pi}(s_{t}=s);\quad\lambda^{\pi}(s,a):=d^{\pi}(s)\pi(a|s). $$ 

The general utility function F assigns a real to each occupancy measure  $ \lambda^{\pi} $  induced by a policy  $ \pi\in\Pi $ . We note that  $ \lambda^{\pi} $  will also be seen as a vector of the Euclidean space  $ R^{|S|\cdot|A|} $ . In the rest of this work, we will consider a class of policies parametrized by a vector  $ \theta\inR^{d} $  for some fixed integer  $ d\inN $ . We shall denote by  $ \pi_{\theta}\in\Pi $  such a policy in this class.

Policy optimization. The goal of the RL agent is to find a policy  $ \pi_{\theta} $  solving the problem:

 $$ \operatorname*{m a x}_{\theta\in\mathbb{R}^{d}}F(\lambda^{\pi_{\theta}}), $$ 

where  $ \lambda $  is defined in (1), F is a smooth function supposed to be upper bounded and  $ F^{\star} $  is used to denote the maximum in (2). The agent has access to trajectories of finite length H generated from the MDP under the initial distribution  $ \rho $  and the policy  $ \pi_{\theta} $ . In particular, provided a time horizon H and a policy  $ \pi_{\theta} $  with  $ \theta \in R^{d} $ , the learning agent can simulate a trajectory  $ \tau = (s_{0}, a_{0}, \cdots, s_{H-1}, a_{H-1}) $  from the MDP when the state transition kernel P is unknown. This general utility problem was described, for instance, in Zhang et al. (2021) (see also Kumar et al. (2022)). Recall that the standard RL problem corresponds to the particular case where the general utility function is a linear function, i.e.,  $ F(\lambda^{\pi_{\theta}}) = \langle r, \lambda^{\pi_{\theta}} \rangle $  for some vector  $ r \in R^{|S| \cdot |A|} $ , in which case we recover the expected return function as an objective:

 $$ V^{\pi_{\theta}}(r):=\mathbb{E}_{\rho,\pi_{\theta}}\left[\sum_{t=0}^{+\infty}\gamma^{t}r(s_{t},a_{t})\right]. $$ 

Examples. We provide two motivating examples of the RLGU framework as follows.

(1) Pure Exploration: The problem consists in finding a policy to explore a state space in the absence of a reward signal. A natural objective is to search for a policy that maximizes the entropy of the induced distribution over the state space. In this case, we have  $ F(\lambda^{\pi_{\theta}}) = -\sum_{s \in \mathcal{S}} \mu^{\pi_{\theta}}(s) \log \mu^{\pi_{\theta}}(s) $  where for every  $ s \in \mathcal{S} $ ,  $ \mu^{\pi_{\theta}}(s) := (1 - \gamma) \sum_{a \in \mathcal{A}} \lambda^{\pi_{\theta}}(s, a) $ . See Hazan et al. (2019).

(2) Learning from Demonstrations: The goal is to learn a policy from expert behavior trajectories or demonstrations. A formulation of such a problem consists in minimizing the Kullback-Leibler

divergence w.r.t. the expert's occupancy measure induced by an unknown policy  $ \pi_{E} $ , in which case  $ F(\lambda^{\pi_{\theta}})=\langle\lambda^{\pi_{\theta}},r\rangle-c\mathrm{KL}(\lambda^{\pi_{\theta}}||\lambda^{\pi_{E}}) $ . See Ho & Ermon (2016); Kang et al. (2018).

Remark 1. We prefer the terminology of ‘RL with general utilities’ to ‘convex RL’ since the objective may even be nonconvex in the occupancy measure in full generality. Although our focus in this work is on concave utilities in experiments, we provide first-order stationarity theoretical guarantees for the nonconcave case. While the convex RL literature exclusively focuses on the case of concave utilities, a lot of applications of interest do not fall under this umbrella and inherently involve nonconcave utilities. We provide several such examples in Appendix C.

## 3 POLICY GRADIENT ALGORITHM WITH OCCUPANCY MEASURE APPROXIMATION (PG-OMA)

In this section, we propose a policy gradient algorithm to solve the policy optimization problem  $ (2) $  with general utilities for larger state-action spaces. We start by elaborating on the challenges faced to solve such a large-scale problem. Section 3.1 mainly contains known material from the recent literature (Zhang et al., 2021), we report it here separately from the problem formulation in section 2 to motivate our algorithmic design. The rest of the section presents our algorithmic contributions.

### 3.1 POLICY GRADIENT THEOREM AND CHALLENGES FOR LARGE-SCALE RLGU

Policy Gradient for RLGU. Following the exposition in Zhang et al. (2021); Barakat et al. (2023), we derive the policy gradient for the general utility objective. For convenience, we use the notation  $ \lambda(\theta) $  for  $ \lambda^{\pi_{\theta}} $ . Since the cumulative reward can be rewritten more compactly  $ V^{\pi_{\theta}}(r) = \langle \lambda^{\pi_{\theta}}, r \rangle $ , it follows from the policy gradient theorem that:

 $$ [\nabla_{\theta}\lambda(\theta)]^{T}r=\nabla_{\theta}V^{\pi_{\theta}}(r)=\mathbb{E}_{\rho,\pi_{\theta}}\left[\sum_{t=0}^{+\infty}\gamma^{t}r(s_{t},a_{t})\sum_{t^{\prime}=0}^{t}\nabla\log\pi_{\theta}(a_{t^{\prime}}|s_{t^{\prime}})\right], $$ 

where  $ \nabla_{\theta}\lambda(\theta) $  is the Jacobian matrix of the vector mapping  $ \lambda(\theta) $ . Using the chain rule, we have

 $$ \nabla_{\theta}F(\lambda(\theta))=[\nabla_{\theta}\lambda(\theta)]^{T}\nabla_{\lambda}F(\lambda(\theta))=\nabla_{\theta}V^{\pi_{\theta}}(r)|_{r=\nabla_{\lambda}F(\lambda(\theta))}. $$ 

The classical policy gradient in the standard RL setting uses rewards which are obtained via interaction with the environment. In RLGU, there is no reward function but rather a pseudoreward  $ \nabla_{\lambda}F(\lambda(\theta)) $  depending on the unknown occupancy measure induced by the policy.

Stochastic Policy Gradient. In view of performing a stochastic policy gradient algorithm, we would like to estimate the policy gradient  $ \nabla_{\theta}F(\lambda(\theta)) $  in (5). We can use the standard reinforce estimator suggested by Eq. (4). Define for every reward function r (which is also seen as a vector in  $ R^{|S|\times|A|} $ ), every  $ \theta\inR^{d} $  and every H-length trajectory  $ \tau $  simulated from the MDP with policy  $ \pi_{\theta} $  and initial distribution  $ \rho $  the (truncated) policy gradient estimate:

 $$ g(\tau,\theta,r)=\sum_{t=0}^{H-1}\left(\sum_{h=t}^{H-1}\gamma^{h}r(s_{h},a_{h})\right)\nabla\log\pi_{\theta}(a_{t}|s_{t}). $$ 

Given (5), we also need to estimate the state-action occupancy measure  $ \lambda(\theta) $  (when F is nonlinear) $ ^{1} $ . Prior work has exclusively focused on the tabular setting using a Monte-Carlo estimate of this occupancy measure  $ \lambda^{\pi_{\theta}} = \lambda(\theta) $  (see (1)) truncated at the horizon H by  $ \lambda(\tau) = \sum_{h=0}^{H-1} \gamma^{h} \delta_{s_{h}, a_{h}} $  where for every  $ (s, a) \in \mathcal{S} \times \mathcal{A} $ ,  $ \delta_{s, a} \in R^{|\mathcal{S}| \times |\mathcal{A}|} $  is a vector of the canonical basis of  $ R^{|\mathcal{S}| \times |\mathcal{A}|} $ , i.e., the vector whose only non-zero entry is the  $ (s, a) $ -th entry which is equal to 1, and  $ \tau = \{(s_{h}, a_{h})\}_{0 \leq h \leq H-1} $  is a trajectory of length H generated by the MDP controlled by the policy  $ \pi_{\theta} $ .

Challenges for Large-scale RLGU. One of the main challenges in solving the general utility problem (2) via a policy gradient algorithm based on (5) is to estimate the unknown state-action occupancy measure  $ \lambda(\theta) $  in large scale settings involving huge state and action spaces. This problem is

arguably more delicate than that of estimating action-value functions in cumulative expected reward RL problems. First, while action-value functions satisfy a forward Bellman equation, occupancy measures satisfy a backward Bellman flow equation. This fundamental difference makes it hard to design stochastic algorithms minimizing mean-square Bellman errors as it is customary in algorithms using function approximation to solve standard RL problems (see end of appendix A for further explanations). Second and foremost, while prior work has used Monte Carlo estimates for this quantity, such count-based estimates are not tractable beyond small tabular settings. Indeed, for very large state-action spaces, it is not tractable to compute and store a table of count-based estimates of the true occupancy measure containing all the values for all the state-action pairs. In the next section, we propose an approach to tackle this issue.

Remark 2. (Extension to continuous state-action spaces) Our algorithm can be used in the continuous (compact) state-action space setting since it only relies on using policy gradients and MLE which are both scalable. We stick to the discrete state action space notation for ease of exposition to avoid the technical measure theoretical formalism to address the continuous setting in full mathematical rigor.

### 3.2 Occupancy Measure Estimation

In this section, we address the challenge of occupancy measure estimation in large state action spaces. Given a policy  $ \pi_{\theta} $ , our goal is to estimate the unknown occupancy measure  $ d^{\pi_{\theta}} $  induced by this policy using state samples obtained from executing the policy. Since the normalized occupancy measure is a probability distribution, we propose to perform maximum likelihood estimation. Before presenting this procedure, we elaborate on the motivation behind approximating the occupancy measure by a parametrized distribution in a given function class of neural networks for example.

Motivation. Besides the practical motivation of using distribution approximation to scale to larger state-action space settings, we provide some theoretical motivation. Recall that action-value functions are linear in the feature map for linear (or low-rank) MDPs for solving standard cumulative sum RL problems (see Proposition 2.3 in Jin et al. (2019)). Similarly, it turns out that state-occupancy measures are linear (or affine in the discounted setting) in density features in low-rank MDPs. We refer the reader to Appendix B for a proof of this statement (see also Lemma 16, 17 in Huang et al. (2023)). Therefore, in this case, it is natural to approximate occupancy measures via linear function approximation using some density features. More generally, for an arbitrary MDP, we propose to approximate the (normalized) state occupancy measure  $ d^{\pi_{\theta}} $  induced by a policy  $ \pi_{\theta} $  directly by a probability distribution in a certain parametric class of probability distributions:

 $$ \Lambda:=\{p_{\omega}\in\Delta(\mathcal{S})\mid\omega\in\Omega\subseteq\mathbb{R}^{m}\}, $$ 

where for instance  $ m \ll |S| $ . An example of such a parametrization for a given  $ \omega \in R^{m} $  is the softmax  $ \sigma_{\omega} $  defined over the state space by  $ \sigma_{\omega}(s) := \exp(\psi_{\omega}(s))/Z(\omega) $ , where  $ Z(\omega) := \sum_{s' \in \mathcal{S}} \exp(\psi_{\omega}(s')) $  and where  $ \psi_{\omega} : S \to R $  is a given mapping which can be a neural network in practice. For continuous state spaces, practitioners can consider for instance Gaussian mixture models with means and covariance matrices encoded by trainable neural networks.

Maximum Likelihood Estimation (MLE). For simplicity, we suppose we have access to i.i.d. state samples following the distribution  $ d^{\pi_{\theta}} $  throughout our exposition. We refer the reader to Appendix D.1 for a discussion about how to sample such states. Given the parametric distribution class  $ \Lambda $  defined in (7) and a data set  $ D := \{s_{i}\}_{i=1,\cdots,n} \in S^{n} $  of n i.i.d. state samples following the distribution  $ d^{\pi_{\theta}} $  induced by the current policy  $ \pi_{\theta} $ , we construct the standard MLE

 $$ \hat{d}^{\pi_{\theta}}:=p_{\omega^{*}}\;,\qquad\omega^{*}\in\underset{\omega\in\Omega}{\arg\max}\frac{1}{n}\sum_{i=1}^{n}\log p_{\omega}(s_{i})\;. $$ 

An estimator of the state-action occupancy measure  $ \lambda^{\pi_{\theta}} $  is then given by  $ \hat{\lambda}^{\pi_{\theta}}(s,a)=\hat{d}^{\pi_{\theta}}(s)\pi_{\theta}(a|s) $  for any  $ s\inA,a\inA $  (see (1)). Using MLE is important for our scalability goal. Barakat et al. (2023) recently proposed a different procedure based on mean square error estimation. Please see appendix A for a detailed comparison with this work highlighting the merits of our approach. In practice, a neural network learns the parameters of a chosen parametrized distribution class for approximating the true occupancy measure by maximizing the log-likelihood loss (8) over the samples generated (see appendix D.1 for sampling).

### 3.3 PROPOSED ALGORITHM

Based on our discussion in sections 3.1 and 3.2, we propose a simple stochastic policy gradient algorithm which consists of two main steps:

(i) Compute an approximation of the unknown state-action occupancy measure  $ \lambda^{\pi_{\theta}} \in R^{|S| \times |A|} $  for a fixed parameter  $ \theta \in R^{d} $  with MLE using collected state samples (see (8));

(ii) Perform stochastic policy gradient ascent using the stochastic policy gradient defined in (6) using the estimated occupancy measure computed in the first step.

The resulting algorithm is Algorithm 1 which is model-free as we do not estimate the transition kernel.

Algorithm 1 PG for RLGU with Occupancy Measure Approximation (PG-OMA)

1: Input: $\theta_0 \in \mathbb{R}^d$, $T, N \geq 1$, $\alpha > 0$, $H$.

2: for $t = 0, \ldots, T - 1$ do

// Occupancy approximation for pseudo-reward learning

3: Compute the MLE estimator $\hat{\lambda}_t = \hat{d}^{\pi_{\theta_t}} \cdot \pi_{\theta_t}$ using policy $\pi_{\theta_t}$ (see (8)).

4: $\hat{r}_t = \nabla_\lambda F(\hat{\lambda}_t)$

// Policy parameter update

5: Sample a batch of $N$ independent trajectories $(\tau_t^{(i)})_{1 \leq i \leq N}$ of length $H$ using $\pi_{\theta_t}$.

6: $\theta_{t+1} = \theta_t + \frac{\alpha}{N} \sum_{i=1}^N g(\tau_t^{(i)}, \theta_t, \hat{r}_t)$ (see (6))

7: end for

8: Return: $\theta_T$

Remark 3. When running Algorithm 1, note that the vector  $ \hat{\lambda}_{t}\inR^{|S|\times|A|} $  (and hence the vector  $ r_{t} $ ) is not computed for all state-action pairs. Indeed, at each iteration, one does only need to compute  $ (r_{t}(s_{h}^{(t)},a_{h}^{(t)}))_{0\leq h\leq H-1} $  where  $ \tau_{t}=(s_{h}^{(t)},a_{h}^{(t)})_{0\leq h\leq H-1} $  to obtain the stochastic policy gradient  $ g(\tau_{t},\theta_{t},r_{t-1}) $  as defined in (6).

Our occupancy measure estimation step can be seen as a critic for pseudo-reward learning. Notice though that this critic is not approximating a value function like in standard RL but rather the occupancy measure which is a distribution.

## 4 CONVERGENCE AND SAMPLE COMPLEXITY ANALYSIS

### 4.1 STATISTICAL COMPLEXITY OF OCCUPANCY MEASURE ESTIMATION

In this section, we suppose we are given a data set of i.i.d. state-action pair samples following the (normalized) occupancy measure  $ \lambda^{\pi} $  induced by a fixed given policy  $ \pi $ . As previously explained, we approximate  $ \lambda^{\pi} $  by a function (or parametrized density) in the function class  $ \Lambda $  defined in (7). We make the following assumption to control the complexity of our function approximation class.

Assumption 1 (Function approximation class regularity). The following holds true:

(i) (parameter compactness) The set $\Omega$ is compact, we denote by $B_{\omega}:=\max_{\omega\in\Omega}\|\omega\|_{\infty}$;

(ii) (realizability) The (normalized) occupancy measure to be estimated satisfies: $\lambda^{\pi} \in \Lambda$;

 $$ \begin{array}{r l}&{(i i i)\mathrm{~(L i p s c h i t z n e s s)}\forall\omega,\bar{\omega}\in\Omega,\forall x\in\mathcal{X},\exists L(x)\in\mathbb{R}s.t.\left|p_{\omega}(x)-p_{\bar{\omega}}(x)\right|\leq L(x)\|\bar{\omega}-}\\ &{\quad\omega\|_{\infty}w i t h B_{L}:=\int_{\mathcal{X}}L(x)d x<+\infty.}\end{array} $$ 

Assumption 1 is satisfied for instance for the class of generalized linear models, i.e.  $ \Lambda := \{p_{\omega}(x) = g(\omega^{T}\phi(x)), \forall x \in \mathcal{X}: p_{\omega} \in \Delta(\mathcal{X}), \omega \in \Omega\} $  where  $ g: R \to [0,1] $  is an increasing Lipschitz continuous function and  $ \phi: X \to R^d $  is a given feature map s.t.  $ \int \|\phi(x)\|_1 dx \leq B_L $  for some  $ B_L > 0 $ . Notice that features can be normalized appropriately to satisfy the assumption. A similar assumption has been made in the case of linear MDPs in (Huang et al., 2023, Assumption 1). The realizability assumption can be relaxed at the price of incurring a misspecification error.

We now state our sample complexity result for occupancy measure estimation via MLE in view of our PG sample complexity analysis. This result relies on arguments developed in the statistics literature Van de Geer (2000); Zhang (2006). These techniques were adapted to the RL setting for low-rank MDPs in e.g. Agarwal et al. (2020). Our proof builds on Huang et al. (2023) which we slightly adapt for our purpose (see Appendix D.2).

Proposition 1. Let Assumption 1 hold true. Then for any $\delta > 0$, the MLE $\hat{\lambda}^{\pi_{\theta}}$ defined using (8) satisfies with probability at least $1 - \delta$,

 $$ \begin{array}{r l}&{\mathit{p i l i t y a t l e a s t1-\delta,}}\\ &{\|\hat{\lambda}^{\pi_{\theta}}-\lambda^{\pi_{\theta}}\|_{1}\leq6\sqrt{\frac{12m\log\left(\frac{2\lceil B_{\omega}B_{L}n\rceil}{\delta}\right)}{n}}.}\end{array} $$ 

The above result translates into a sample complexity of  $ \tilde{\mathcal{O}}(m\varepsilon^{-2}) $  to guarantee an  $ \varepsilon $ -approximation of the true occupancy measure (in the  $ l_{1} $ -norm distance) using samples. We highlight that our sample complexity only depends on the dimension m of the parameter space and does not scale with the size of the state-action space. Hence the MLE procedure we use is the key ingredient to scale our algorithm to large state-action spaces. To the best of our knowledge, existing algorithms for solving the RLGU problem (with nonlinear utility functions) are limited to the restrictive tabular setting.

### 4.2 GUARANTEES FOR POLICY GRADIENT WITH OCCUPANCY MEASURE APPROXIMATION

In this section, we establish sample complexity guarantees for Algorithm 1. We start by introducing the assumptions required for our results and discuss their relevance.

Assumption 2 (Policy parametrization). The following holds for every  $ (s,a)\in\mathcal{S}\times\mathcal{A} $ . For every  $ \theta\in\mathbb{R}^{d} $ ,  $ \pi_{\theta}(a|s)>0 $ . Moreover, the function  $ \theta\mapsto\pi_{\theta}(a|s) $  is continuously differentiable and the score function  $ \theta\mapsto\nabla\log\pi_{\theta}(a|s) $  is bounded by some positive constant B.

This standard assumption is satisfied for instance by the common softmax policy parametrization defined for every  $ \theta\in\mathbb{R}^{d} $ ,  $ (s,a)\in\mathcal{S}\times\mathcal{A} $  by  $ \pi_{\theta}(a|s)=\frac{\exp(\psi(s,a;\theta))}{\sum_{a^{\prime}\in\mathcal{A}}\exp(\psi(s,a^{\prime};\theta))} $ , where  $ \psi:S\timesA\timesR^{d}\toR $  is a smooth function such that the map  $ \psi(s,a;\cdot) $  is twice continuously differentiable for every  $ (s,a)\in\mathcal{S}\times\mathcal{A} $  and for which there exist  $ l_{\psi},L_{\psi}>0 $  s.t. (i)  $ \max_{s\in\mathcal{S},a\in\mathcal{A}}\sup_{\theta}\|\nabla\psi(s,a;\theta)\|\leq l_{\psi} $  and (ii)  $ \max_{s\in\mathcal{S},a\in\mathcal{A}}\sup_{\theta}\|\nabla^{2}\psi(s,a;\theta)\|\leq L_{\psi} $ .

Assumption 3 (General utility smoothness). There exist constants  $ l_{\lambda}, L_{\lambda} > 0 $  s.t. for all  $ \lambda_{1}, \lambda_{2} \in \Lambda $ ,  $ \|\nabla_{\lambda} F(\lambda_{1})\|_{2} \leq l_{\lambda} $  and  $ \|\nabla_{\lambda} F(\lambda_{1}) - \nabla_{\lambda} F(\lambda_{2})\|_{2} \leq L_{\lambda}\|\lambda_{1} - \lambda_{2}\|_{2} $ .

Under Assumptions 2 and 3, the function  $ \theta\mapsto F(\lambda^{\pi_{\theta}}) $  is  $ L_{\theta} $ -smooth (see Lemma 3 for the expression). Using this property, the next result shows that our algorithm enjoys a first-order stationary guarantee in terms of the non-convex general utility objective.

Theorem 1. (Nonconcave general utility) Let Assumptions 2, 3 hold. Then the iterates generated by Algorithm 1 with step sizes $\alpha_{t} \leq 1/(2L_{\theta})$ and $T \geq 1$ iterations satisfy:

 $$ \mathbb{E}[\|\nabla_{\theta}F(\lambda^{\pi_{\theta_{\tau}}})\|^{2}]\leq\frac{16(F^{\star}-\mathbb{E}[F(\lambda^{\pi_{\theta_{1}}})])}{\alpha T}+\frac{C_{1}}{N}+C_{2}\mathbb{E}[\|\hat{\lambda}_{\tau}-\lambda^{\pi_{\theta_{\tau}}}\|_{2}^{2}], $$ 

where $\tau$ is a uniform random variable over $\{1,\cdots,T\}$ and expectation is w.r.t. all randomness (in $(\theta_{t})$ and $\tau$).

The above upper bound shows a decomposition of the first order stationarity error into three terms: the first two are the typical errors incurred by PG methods whereas the third one is due to occupancy measure approximation. In particular, choosing the number of iterations T, the batch size N (of sampled trajectories) appropriately and the number n of samples used in MLE for occupancy measure approximation, we obtain the following sample complexity result.

Corollary 1. Let Assumptions 1, 2, 3 hold. Setting the number of iterations to  $ T = \mathcal{O}(\epsilon^{-1}) $ , the batch size for PG to  $ N = \mathcal{O}(\epsilon^{-1}) $  and the number of samples for occupancy measure MLE to  $ n = \mathcal{O}(m\epsilon^{-1}) $  for some precision  $ \epsilon > 0 $  in Theorem 1, it holds that  $ \mathbb{E}[\|\nabla_{\theta}F(\lambda^{\pi_{\theta_{\tau}}})\|^{2}] \leq \epsilon $ . The total sample complexity is then  $ T(N + n) = \mathcal{O}(m\epsilon^{-2}) $ .

In several applications in RLGU, the utility function F is concave w.r.t. its occupancy measure variable. We now turn to proving global performance bounds under this particular setting.

Assumption 4 (Concavity). The utility function  $ F : \Lambda \to R $  is concave.

Notice that the general utility objective is in general nonconcave w.r.t. the policy parameter  $ \theta $ . Despite this non-concavity, we can exploit the so-called hidden convexity (concavity in our setting) of the problem Zhang et al. (2021). We require an additional regularity assumption on the policy parametrization which has been previously made in Zhang et al. (2021); Ying et al. (2023a); Barakat et al. (2023). While this assumption holds for a tabular policy parametrization, it is delicate to relax it further, see e.g. (Barakat et al., 2023, Appendix C) for a discussion.

Assumption 5 (Policy overparametrization). For the softmax policy parametrization defined above, the following three requirements hold: (i) For any  $ \theta \in R^{d} $ , there exist relative neighborhoods  $ U_{\theta} \subset R^{d} $  and  $ \mathcal{V}_{\lambda(\theta)} \subset \Lambda $  respectively containing  $ \theta $  and  $ \lambda(\theta) $  s.t. the restriction  $ \lambda|_{U_{\theta}} $  forms a bijection between  $ U_{\theta} $  and  $ \mathcal{V}_{\lambda(\theta)} $ ; (ii) There exists l > 0 s.t. for every  $ \theta \in R^{d} $ , the inverse  $ (\lambda|_{U_{\theta}})^{-1} $  is l-Lipschitz continuous; (iii) There exists  $ \bar{\eta} > 0 $  s.t. for every positive real  $ \eta \leq \bar{\eta} $ ,  $ (1 - \eta)\lambda(\theta) + \eta\lambda(\theta^{*}) \in \mathcal{V}_{\lambda(\theta)} $  where  $ \pi_{\theta^{*}} $  is the optimal policy.

The following result makes use of the concavity of the utility function F to obtain a global optimality guarantee for the iterates of our algorithm under the assumption that the occupancy measures induced by the policies encountered during the run of the algorithm are uniformly well-approximated.

Theorem 2. (Concave general utility) Let Assumptions 2 to 5 hold. Assume further that there exists  $ \epsilon_{MLE} > 0 $  s.t.  $ \mathbb{E}[\|\hat{\lambda}_{t} - \lambda(\theta_{t})\|_{2}^{2}] \leq \epsilon_{MLE} $  uniformly over  $ T \geq 1 $  iterations of Algorithm 1 with step sizes  $ \alpha_{t} \leq 1/(2L_{\theta}) $ . Then the iterate output  $ \theta_{T} $  of Algorithm 1 satisfies for any  $ \eta < \bar{\eta} $ ,

 $$ \mathbb{E}[F^{\star}-F(\lambda(\theta_{T}))]\leq(1-\eta)^{T}\delta_{0}+C_{3}\frac{\eta}{\alpha}+C_{4}\frac{\alpha}{\eta}\left(\frac{1}{N}+\epsilon_{M L E}\right), $$ 

for some positive constants  $ C_{3}, C_{4} $  explicit in Appendix D.4, (48) and  $ \delta_{0} := \mathbb{E}[F^{\star} - F(\lambda(\theta_{0}))] $ 

Using the above result, we derive the following sample complexity guarantee by specifying the step size and number of iterations of our algorithm as well as large enough batch size and number of samples for MLE using Proposition 1.

Corollary 2. Let Assumptions 1 to 5 hold. For any given precision  $ \epsilon > 0 $ , set  $ T = \frac{1}{\eta} \log(\frac{\delta_{0}}{\epsilon}) $ ,  $ \alpha = \mathcal{O}(\epsilon) $ ,  $ \eta = \mathcal{O}(\epsilon^{2}) $ ,  $ N = \mathcal{O}(\epsilon^{-2}) $  and  $ n = \mathcal{O}(m\epsilon^{-2}) $ , then the total sample complexity to obtain  $ \mathbb{E}[F^{\star} - F(\lambda(\theta_{t}))] \leq \epsilon $  is given by  $ T(N + n) = \mathcal{O}(m\epsilon^{-4}) $ .

## 5 Proof of Concept Experiments

In this section, we investigate the capability of the proposed PG-OMA in terms of scaling with respect to the dimensionality of the state space when solving RLGU problems. In this work, we perform initial proof of concept experiments on simulation environments such as MPE (Multi-Agent Particle Environment (Lowe et al., 2017)) and SMAC (StarCraft Multi-Agent Challenge (Samvelyan et al., 2019)). We provide additional details about the experiments in Appendix E. We consider the problem of learning from demonstrations as defined in Example 2 in Sec. 2 and show results in discrete and continuous state space settings. Before presenting our experimental results, we want to emphasize that our experiments serve as evidence of the potential of the proposed approach in addressing scalability challenges in RLGU. We do not claim to surpass the state-of-the-art performance in solving specific tasks (of learning from demonstration) within the MPE and SMAC environments. In contrast to prior work which mostly designed tailored algorithms for specific single tasks, note that our algorithm can be used for any RLGU problem.

(1) Discrete Spaces. To further demonstrate the effectiveness of our proposed approach, we conducted experiments in a  $ 10 \times 10 $  gridworld environment with varying numbers of agents tasked with reaching distinct goal positions (refer to Figure 4 in the Appendix for a detailed gridworld description). It is important to note that as the number of agents in the environment increases, the

<div style="text-align: center;"><img src="imgs/img_in_chart_box_274_168_950_374.jpg" alt="Image" width="55%" /></div>


<div style="text-align: center;">(a) Experiments with optimal demonstrations</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_274_427_949_634.jpg" alt="Image" width="55%" /></div>


<div style="text-align: center;">(b) Experiments with suboptimal demonstrations</div>


Figure 2: (a) This figure compares the convergence of our proposed approach across three distinct settings: easy, medium, and hard. The "easy" setting has  $ 10^{2} $  states, the medium setting features  $ 10^{3} $  states, and the harder setting comprises  $ 10^{4} $  states. In the easy setting, the count-based method performs relatively well, as expected, since it aims to precisely estimate the occupancy measure (we employ a batch size of B = 100 for estimating the occupancy in each episode). However, as we transition to larger state space settings, our proposed method outperforms the count-based approach significantly. (b) We conducted tests with suboptimal demonstrations and show that our proposed algorithm remains effective. The shaded area is a tolerance interval (with mean and standard deviation) built from running the experiment with 5 different seeds.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_292_931_608_1139.jpg" alt="Image" width="25%" /></div>


<div style="text-align: center;">(a) MPE navigation environment.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_926_930_1140.jpg" alt="Image" width="25%" /></div>


<div style="text-align: center;">(b) SMAC environment.</div>


<div style="text-align: center;">Figure 3: This figure shows the effectiveness of our proposed approach in continuous state space environments, such as MPE and SMAC environments. For MPE, we plot the performance of the base method via discretization of the state space, which clearly results in suboptimal results. We only report the results of our proposed approach for SMAC as the count-based baseline was intractable.</div>


joint state and action space grows exponentially. Additionally, we consider a sparse reward setting, where agents receive a non-zero reward only when they successfully reach their respective goals; otherwise, the reward remains zero. This sparse reward setup makes the problem hard, requiring the incorporation of demonstrations from experts to guide learning, aligning with the RLGU problem outlined in Section 2. Figure 2 summarizes the effectiveness of the proposed approach as compared to the count-based estimation method.

(2) Continuous Spaces. We also conducted experiments to demonstrate the effectiveness of the proposed approach in continuous spaces on (a) the cooperative navigation task from the multi-agent particle environment (MPE) and (b) the SMAC environment based on the StarCraft II game, we consider 3sv4z, which features 3 Stalkers (allies) versus 4 Zealots (enemies). For comparison, we discretize the state space to perform the count-based estimation method as a baseline. Figure 3 presents the training curves of both methods.

## 6 CONCLUSION

In this paper, we proposed a simple policy gradient algorithm for RLGU to address the fundamental challenge of scaling to larger state-action spaces beyond the tabular setting. Our approach hinges on using MLE for approximating occupancy measures to construct a stochastic policy gradient. We proved that our MLE procedure enjoys a sample complexity which only scales with the dimension of the parameters in our function approximation class rather than the size of the state-action space which might even be continuous. Under suitable assumptions, we also provided convergence guarantees for our algorithm to first-order stationarity and global optimality respectively. We hope this work will stimulate further research in view of designing efficient and scalable algorithms for solving real-world problems.