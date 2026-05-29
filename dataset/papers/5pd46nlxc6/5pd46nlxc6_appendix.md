## APPENDICES

## A REPRESENTATIONAL COMPLEXITY OF MAC-IGM AND MACADV-IGM

As discussed by VDN and QMIX (Sunehag et al., 2018; Rashid et al., 2018), common value factorization approaches cannot guarantee representing their respective classes of true value functions in a Dec-POMDP. The same limitation holds in MacDec-POMDPs; agents' observations do not represent the full state in partially observable settings. Similarly, per-agent value function ordering can (potentially) be wrong in a macro-action context. Formally, given an agent i at a time step t it could happen that:

 $$ Q_{i}(\hat{h}^{i},m^{i})>Q_{i}(\hat{h}^{i},m^{\prime i})when Q(s,(\boldsymbol{m}^{-i},m^{i}))<Q\left(s,(\boldsymbol{m}^{-i},m^{\prime i})\right) $$ 

where  $ m^{-a} $  is the joint action of all the agents excluding i. However, there are several ways to alleviate such an issue. First, it is possible to condition per-agent action values (or state and advantage values) with state information during offline training as in QMIX (Rashid et al., 2018), QPLEX (Wang et al., 2021). Moreover, if can not assume that  $ (\hat{\pmb{h}},\pmb{m}) $  (i.e., the joint macro-history and action) is sufficient to fully model  $ Q(s,\pmb{m}) $  (which is a common assumption in prior factorization approaches), we can potentially store additional history-related information in recurrent layers (Sunehag et al., 2018).

### A.1 REPRESENTATIONAL EXPRESSIVENESS OF AVF ALGORITHMS

The proposed AVF framework does not change the architectural design of the chosen factorization method. Hence, the algorithms investigated in Section 5, namely AVF- $ \{VDN, QMIX, QPLEX\} $ , maintain the same considerations of the original factorization methods in terms of representational expressiveness.

In particular, AVF-VDN can factorize arbitrary joint macro-action value functions that can be additively decomposed into individual utilities. AVF-QMIX extends the family of factorizable functions to non-linear monotonic combinations. Finally, AVF-QPLEX does not involve architectural constraints and is capable of achieving the entire class of functions satisfying the underlying IGM.

#### A.1.1 OMITTED PROOFS IN SECTION 3

Proposition 3.2. Denoting with

 $$ F^{IGM}=\left\{\left(Q^{IGM}:\boldsymbol{H}\times\mathcal{U}\rightarrow\mathbb{R}^{|\mathcal{U}|},\left\langle Q_{i}^{IGM}:H^{i}\times U^{i}\rightarrow\mathbb{R}^{|U^{i}|}\right\rangle_{i\in\mathcal{N}}\right)\mid Eq.1holds\right\} $$ 

 $$ F^{Mac-IGM}=\left\{\left(Q^{Mac-IGM}:\hat{\boldsymbol{H}}\times\mathcal{M}\rightarrow\mathbb{R}^{|\mathcal{M}|},\left\langle Q_{i}^{Mac-IGM}:\hat{\boldsymbol{H}}^{i}\times\boldsymbol{M}^{i}\rightarrow\mathbb{R}^{|\boldsymbol{M}^{i}|}\right\rangle_{i\in\mathcal{N}}\right)\mid Eq.6holds\right\} $$ 

the class of functions satisfying IGM and Mac-IGM respectively, then:

 $$ F^{IGM}\subset F^{Mac-IGM} $$ 

Proof. MacDec-POMDPs extends Dec-POMDPs by replacing the primitive actions available to each agent with option-based macro-actions. However, as shown in (Amato et al., 2019), the macro-action set contains primitive actions to guarantee the same globally optimal policy:

 $$ U^{i}\subset M^{i},\forall i\in\mathcal{N} $$ 

Meaning that  $ \forall i\inN,\left|M_{i}\right|>\left|U_{i}\right| $ , which implies  $ |M|>|U| $ . It also follows that  $ O\subseteq\hat{O} $  as a MacDec-POMDP is, in the limit where only primitive actions are selected, equivalent to a Dec-POMDP. For these reasons, we can conclude that  $ |\hat{H}\timesM|>|H\timesU| $  (i.e., the domain over which primitive action-value functions are defined is smaller than the domain over which macro-action-value functions are defined). Hence,  $ F^{IGM}\subset F^{Mac-IGM} $ . ☐

Proposition 3.4. The consistency requirement of MacAdv-IGM in Eq. 10 is equivalent to the Mac-IGM one in Eq. 6. Hence, denoting with

 $$ F^{MacAdv-IGM}=\left\{\left(Q^{MacAdv-IGM}:\hat{\boldsymbol{H}}\times\mathcal{M}\rightarrow\mathbb{R}^{|\mathcal{M}|},\langle Q_{i}^{MacAdv-IGM}:\hat{\boldsymbol{H}}^{i}\times\boldsymbol{M}^{i}\rightarrow\mathbb{R}^{|\boldsymbol{M}^{i}|}\rangle_{i\in\mathcal{N}}\right)\mid Eq10holds\right\} $$ 

the class of functions satisfying MacAdv-IGM, we can conclude that  $ F^{Mac-IGM} \equiv F^{MacAdv-IGM} $ 

Proof. Given a joint macro-history  $ \hat{h} \in \hat{H} $  on which  $ \langle Q_{i}(\hat{h}^{i}, m^{i}) \rangle_{i \in \mathcal{N}} $  satisfies Mac-IGM for  $ Q(\hat{h}, m \mid m_{-}) $ , we show Eq. 10 represents the same consistency constraint as Eq. 6. By applying the dueling decomposition from (Wang et al., 2016), we know  $ Q(\hat{h}, m \mid m_{-}) = V(\hat{h}) + A(\hat{h}, m \mid m_{-}) $ , and  $ Q_{i}(\hat{h}^{i}, m^{i}) = V(\hat{h}^{i}) + A_{i}(\hat{h}^{i}, m^{i}) $ ,  $ \forall i \in N $ . Hence, the state-value functions defined over macro-histories do not influence the action selection process. For the joint value, we can thus conclude that:

 $$ \underset{\boldsymbol{m}\in\mathcal{M}}{\arg\max}Q(\hat{\boldsymbol{h}},\boldsymbol{m}\mid\boldsymbol{m}_{-})=\underset{\boldsymbol{m}\in\mathcal{M}}{\arg\max}V(\hat{\boldsymbol{h}})+A(\hat{\boldsymbol{h}},\boldsymbol{m}\mid\boldsymbol{m}_{-})\\=\underset{\boldsymbol{m}\in\mathcal{M}}{\arg\max}A(\hat{\boldsymbol{h}},\boldsymbol{m}\mid\boldsymbol{m}_{-}) $$ 

Similarly, for the individual values:

 $$ \begin{aligned}\forall i\in\mathcal{N},&\begin{cases}\arg\max_{m^{i}\in M^{i}}Q_{i}(\hat{h}^{i},m^{i})&if M^{i}\in\mathcal{M}_{+}\\\boldsymbol{m}_{-}^{i}&otherwise\end{cases}\\&=\begin{cases}\arg\max_{m^{i}\in M^{i}}V(\hat{h}^{i})+A_{i}(\hat{h}^{i},m^{i})&if M^{i}\in\mathcal{M}_{+}\\\boldsymbol{m}_{-}^{i}&otherwise\end{cases}\\&=\begin{cases}\arg\max_{m^{i}\in M^{i}}A_{i}(\hat{h}^{i},m^{i})&if M^{i}\in\mathcal{M}_{+}\\\boldsymbol{m}_{-}^{i}&otherwise\end{cases}\end{aligned} $$ 

Broadly speaking, we know the history values act as a constant for both the joint and local estimation and do not influence the arg max operator. By combining Eq. 19, 20, we conclude the equivalence between Eq. 6, 10.

Proposition 3.5. Denoting with  $ F^{\{Adv-IGM,MacAdv-IGM\}} $  the classes of functions satisfying Adv-IGM and MacAdv-IGM, respectively, then:

 $$ F^{IGM}\equiv F^{Adv-IGM}\subset F^{Mac-IGM}\equiv F^{MacAdv-IGM}. $$ 

Proof. The result naturally follows from Proposition 3.2, 3.5, and the result of (Wang et al., 2021) that showed the equivalence between the class of functions represented by the primitive IGM and Adv-IGM. In more detail, from the latter we know  $ F^{IGM} \equiv F^{Adv-IGM} $ . Moreover, Proposition 3.2 showed us that  $ F^{IGM} \subset F^{Mac-IGM} $ , from which follows that  $ F^{Adv-IGM} \subset F^{Mac-IGM} $ . In addition, Proposition 3.5 showed us that  $ F^{Mac-IGM} \equiv F^{MacAdv-IGM} $ . Combining these results, we conclude the relationship in Eq. 21. ☐

## B EXPRESSIVENESS OF AVF-OPLEX-D0

In this section, we show how the design of AVF algorithms allows the underlying factorization architecture to maintain the same class of expressiveness as their primitive counterparts (e.g., additive functions, monotonic functions), but with respect to Mac-IGM. Let us prove the full expressiveness AVF-QPLEX-D0 over Mac-IGM as an explanatory example, extending the full expressiveness of QPLEX over IGM of the primitive case.

Proposition 3.5. Given the universal function approximation of neural networks, the function class that AVF-QPLEX-D0 can realize is equivalent to what is induced by Mac-IGM.

Proof. The proof extends the synchronous, primitive action proof of Wang et al. (2021). The main difference is related to the conditional action-value functions learned by AVF-QPLEX-D0, which allows it to maintain action selection consistency and correct updates over asynchronous macro-action-based agents.

First, let us define the utilities deriving from the transformation and mixer modules of AVF-QPLEX-D0. For clarity, we recall these components implement the same operations as the original QPLEX (shown in Fig. 6), but in the asynchronous macro-actions context.

<div style="text-align: center;"><img src="imgs/img_in_image_box_249_161_967_527.jpg" alt="Image" width="58%" /></div>


<div style="text-align: center;">Figure 6: Primitive actions-based QPLEX architecture (image credit: Wang et al. (2021)). (a) Mixing network; (b) QPLEX architecture; (c) Individual utility and transformation networks.</div>


At any step t, consider the set of terminated macro-action spaces  $ M_{t,+} $  and the ongoing macro actions  $ m_{t,-} $  defined as in Def. 3.1. For each agent i, AVF-QPLEX-D0 first decomposes its utility  $ Q_{i}(\hat{h}_{t}^{i}, m_{t}^{i}|m_{t-1}^{i}) $  as follows:

 $$ \begin{aligned}&V_{i}(\hat{h}_{t}^{i})=\begin{cases}\max_{m^{i}}Q_{i}(\hat{h}_{t}^{i},m^{i})&if M^{i}\in\mathcal{M}_{+,t}\\Q_{i}(\hat{h}_{t}^{i},m_{t-1}^{i})&otherwise\end{cases},\\&A_{i}^{(\hat{h}_{t}^{i},m_{t}^{i}|m_{t-1}^{i})}=Q_{i}(\hat{h}_{t}^{i},m_{t}^{i}|m_{t-1}^{i})-V_{i}(\hat{h}_{t}^{i}).\\ \end{aligned} $$ 

The transformation module then outputs the following transformed utilities:

 $$ \begin{aligned}&V_{i}^{T}(\hat{\boldsymbol{h}}_{t})=w_{i}(\hat{\boldsymbol{h}}_{t})V_{i}(\hat{h}_{t}^{i})+b_{i}(\hat{\boldsymbol{h}}_{t}),\\&A_{i}^{T}(\hat{\boldsymbol{h}}_{t},m_{t}^{i}|m_{t-1}^{i})=w_{i}(\hat{\boldsymbol{h}}_{t})A_{i}(\hat{h}_{t}^{i},m_{t}^{i}|m_{t-1}^{i}),\\ \end{aligned} $$ 

and the mixer module combines all the agents' utilities into the following joint utilities:

 $$ \begin{aligned}&V^{MIX}(\hat{\boldsymbol{h}}_{t})=\sum_{i\in\mathcal{N}}V_{i}^{T}(\hat{\boldsymbol{h}}_{t}),\\&A^{MIX}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})=\sum_{i\in\mathcal{N}}\lambda_{i}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})A_{i}^{T}(\hat{\boldsymbol{h}}_{t},m_{t}^{i}|m_{t-1}^{i}),\\ \end{aligned} $$ 

to finally output the joint value  $  Q(\hat{h}_{t}, m_{t}|m_{t,-})  $  defined as:

 $$ Q^{M I X}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})=V^{M I X}(\hat{\boldsymbol{h}}_{t})+A^{M I X}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-}). $$ 

We can now prove the full expressiveness of AVF-QPLEX-D0 over Mac-IGM. Assume AVF-QPLEX-D0's network size is sufficient to satisfy the universal function approximation theorem (Csáji, 2001). Denote the joint  $ Q^{MIX}, A^{MIX}, V^{MIX} $ , transformed  $ Q_{i}^{T}, A_{i}^{T}, V_{i}^{T} $ , and individual  $ Q_{i}, A_{i}, V_{i} $  macro-action, macro-observation, advantage macro-history-based value functions and utilities learned by AVF-QPLEX-D0, respectively. Moreover, Let the class of action-value functions that the algorithms can represent be  $ Q^{MIX} $  defined as:

 $$ \mathcal{Q}^{M I X}=\left\{\left(Q^{M I X},\langle Q_{i}\rangle_{i\in\mathcal{N}}\right)|\mathrm{E q s.22,23,24,25a r e~s a t i s f i e d}\right\}, $$ 

and let  $ Q^{Mac-IGM} $  be the class of macro-action-value functions represented by Mac-IGM (Eq. 15).

Firstly, we note the multiplicative weights in both the transformation and mixer modules are all positive to satisfy action selection consistency. Secondly, we prove  $ Q^{MIX} = Q^{Mac-IGM} $  by demonstrating the inclusion in the two directions  $ Q^{Mac-IGM} \subseteq Q^{MIX} $  and  $ Q^{Mac-IGM} \supseteq Q^{MIX} $ .

1.  $ Q^{Mac-IGM} \subseteq Q^{MIX} $ : For any  $ \left(Q^{Mac-IGM}, \langle Q_{i}^{Mac-IGM} \rangle_{i \in \mathcal{N}}\right) \in Q^{Mac-IGM} $  we construct  $ Q^{MIX} = Q^{Mac-IGM} $  and  $ \langle Q_{i} \rangle_{i \in \mathcal{N}} = \langle Q_{i}^{Mac-IGM} \rangle_{i \in \mathcal{N}} $ , deriving  $ A_{i}, V_{i}, A^{MIX}, V^{MIX} $  by Eqs. 22, 24 and constructing the transformed values connecting joint and individual ones as:

 $$ Q_{i}^{T}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})=\frac{Q^{M I X}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})}{|\mathcal{N}|}, $$ 

 $$ V_{i}^{T}(\hat{\boldsymbol{h}}_{t})=\max_{\boldsymbol{m}^{\prime}}Q_{i}^{T}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}^{\prime}|\boldsymbol{m}_{t,-}),\quad A_{i}^{T}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})=Q_{i}^{T}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})-V_{i}^{T}(\hat{\boldsymbol{h}}_{t}). $$ 

According to the fact that $\forall m^{*}\in\mathcal{M}^{*}(\hat{h}),m\in\mathcal{M}\setminus\mathcal{M}^{*}(\hat{h}),i\in\mathcal{N}$

 $$ A^{M I X}(\hat{\boldsymbol{h}},\boldsymbol{m}^{*}|\boldsymbol{m}_{-})=A_{i}(\hat{h}^{i},m^{i,*}|\boldsymbol{m}_{-}^{i})=0, $$ 

 $$ A^{M I X}(\hat{\boldsymbol{h}},\boldsymbol{m}|\boldsymbol{m}_{-})<0,A_{i}(\hat{h}^{i},m^{i}|\boldsymbol{m}_{-}^{i})<0, $$ 

where  $ \mathcal{M}^{*}(\hat{\boldsymbol{h}})=\{\boldsymbol{m}|\boldsymbol{m}\in\mathcal{M},Q^{MIX}(\hat{\boldsymbol{h}},\boldsymbol{m}|\boldsymbol{m}_{-})=V^{MIX}(\hat{\boldsymbol{h}})\} $ , and by setting:

 $$ w_{i}(\hat{\boldsymbol{h}})=1,\quad b_{i}(\hat{\boldsymbol{h}})=V_{i}^{T}(\hat{\boldsymbol{h}})-V_{i}(\hat{h}_{i}), $$ 

 $$ \lambda_{i}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})=\begin{cases}\frac{A_{i}^{T}(\hat{\boldsymbol{h}}_{t},\boldsymbol{m}_{t}|\boldsymbol{m}_{t,-})}{A_{i}(\hat{h}_{t}^{i},m_{t}^{i}|\boldsymbol{m}_{t,-}^{i})}&if A_{i}(\hat{h}_{t}^{i},m_{t}^{i}|\boldsymbol{m}_{t,-}^{i})<0,\\1&otherwise.\end{cases} $$ 

we conclude that  $ \left(Q^{MIX},\langle Q_{i}\rangle_{i\in\mathcal{N}}\right)\in Q^{Mac-IGM} $ , meaning that  $ Q^{Mac-IGM}\subseteq Q^{MIX} $ 

2.  $ Q^{Mac-IGM} \supseteq Q^{MIX} $ : For any  $ \left(Q^{MIX}, \langle Q_i \rangle_{i \in \mathcal{N}}\right) \in Q^{MIX} $ , following the above fact regarding non-positive advantage functions/utilities,  $ \forall \hat{h} \in \hat{H}, i \in N $ , let:

 $$ A_{i}^{M I X^{*}}(\hat{h}^{i})=\{m^{i}|m^{i}\in\mathcal{M}^{i},A_{i}(\hat{h}^{i},m^{i}|\boldsymbol{m}_{-}^{i})=0\}. $$ 

Combining the positivity of the weights $\langle w_{i},\lambda_{i}\rangle_{i\in\mathcal{N}}$ with Eqs. 22, 23, 24, 25, we can derive $\forall\hat{\boldsymbol{h}}\in\hat{\boldsymbol{H}},m^{i,*}\in A_{i}^{MIX^{*}}(\hat{h}^{i}),m^{i}\in\mathcal{M}\setminus A_{i}^{MIX^{*}}(\hat{h}^{i}),i\in\mathcal{N}$:

 $$ \begin{aligned}A_{i}(\hat{h}^{i},m^{i,*}|\boldsymbol{m}_{-}^{i})&=0\quad and\quad A_{i}(\hat{h}^{i},m^{i}|\boldsymbol{m}_{-}^{i})<0\\A_{i}^{T}(\hat{\boldsymbol{h}},m^{i,*}|\boldsymbol{m}_{-}^{i})&=w_{i}(\hat{\boldsymbol{h}})A_{i}(\hat{h}^{i},m^{i,*}|\boldsymbol{m}_{-}^{i})=0\quad and\\&\quad A_{i}^{T}(\hat{\boldsymbol{h}},m^{i}|\boldsymbol{m}_{-}^{i})=w_{i}(\hat{\boldsymbol{h}})A_{i}(\hat{h}^{i},m^{i}|\boldsymbol{m}_{-}^{i})<0\\A^{MIX}(\hat{\boldsymbol{h}},\boldsymbol{m}^{*}|\boldsymbol{m}_{-})&=\lambda_{i}(\hat{\boldsymbol{h}},\boldsymbol{m}^{*}|\boldsymbol{m}_{-})A_{i}^{T}(\hat{\boldsymbol{h}},m^{i,*}|\boldsymbol{m}_{-}^{i})=0\quad and\\&\quad A^{MIX}(\hat{\boldsymbol{h}},\boldsymbol{m}|\boldsymbol{m}_{-})=\lambda_{i}(\hat{\boldsymbol{h}},\boldsymbol{m}|\boldsymbol{m}_{-})A_{i}^{T}(\hat{\boldsymbol{h}},m^{i}|\boldsymbol{m}_{-}^{i})<0.\end{aligned} $$ 

 $$ Q^{M I X}=Q^{M a c-I G M},\langle Q_{i}\rangle_{i\in\mathcal{N}}= $$ 

 $$ \langle Q_{i}^{Mac-IGM}\rangle_{i\in\mathcal{N}} $$ 

 $$ \left(Q^{Mac-IGM},\langle Q_{i}^{Mac-IGM}\rangle_{i\in\mathcal{N}}\right)\in\mathcal{Q}^{MIX} $$ 

 $$ Q^{MIX}\subseteq Q^{Mac-IGM} $$ 

Under the assumption that AVF-QPLEX-D0's neural networks provide universal function approximation, the joint macro-action-value function class that AVF-QPLEX-D0 can represent is thus equivalent to what is induced by Mac-IGM.

## C LIMITATIONS AND BROADER IMPACT

Limitations. We identify three limitations in our work. First, most factorization approaches cannot guarantee to fully represent their respective classes of value functions in a Dec-POMDP (Sunehag et al., 2018; Rashid et al., 2018; 2020); the same limitation holds in AVF-based algorithms that maintain the same representation expressiveness of the original methods. Second, AVF methods employing the joint macro-state could have scalability issues when considering many agents. While such a problem does not arise in our experiments with up to 10 agents, it is possible to train an encoder to reduce the dimensionality of the joint macro-state. Third, MacDec-POMDPs assume

that macro-actions are known and fixed. This is the same as assuming primitive actions are given in a primitive MARL domain. Moreover, asynchronous settings are common in the real world but have been rarely studied in the MARL literature. For this reason, principled methods are needed for the MacDec-POMDP case before extending them to learn macro-actions (e.g., by employing skill discovery approaches (Eysenbach et al., 2019)).

Broader impact. Regarding the broader impact of our work, we do believe macro-actions have the potential to scale MARL into the real world. Temporally extended actions enable decision-making at a higher level and naturally represent complex real-world behavior (e.g., lifting an object). That can exploit existing robust controllers or be defined by a (human) expert, making them more explainable than other sequences of primitive actions. By extending MARL algorithms to the macro-action case, realistic multi-agent coordination problems can be solved that are orders of magnitude larger than problems solved by previous primitive MARL algorithms.

## D HYPER-PARAMETERS

Regarding the considered baselines, we employed the original authors' implementations and parameters (Sunehag et al., 2018; Rashid et al., 2018; Wang et al., 2021; Xiao et al., 2020a; 2022). Table 4 lists all the hyper-parameters considered in our initial grid search for tuning the algorithms employed in Section 5. We separate algorithm-specific parameters (e.g., for the mixer of AVF-QMIX, AVF-QPLEX) with a horizontal line at the end of the table. We tested different joint reward schemes for macro-actions (e.g., only considering the max/min values and time horizon among the agents, averaging them). Still, the original joint scheme in Section 2.2 resulted in the best performance.

<div style="text-align: center;">Table 4: Hyper-parameters candidate for initial grid search tuning.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Learning rate</td><td style='text-align: center;'>5e-4, 2.5e-4, 2.5e-5</td></tr><tr><td style='text-align: center;'>$ \gamma $</td><td style='text-align: center;'>0.9, 0.95, 0.99</td></tr><tr><td style='text-align: center;'>ASVB (full episodes) size</td><td style='text-align: center;'>1000, 2500, 5000</td></tr><tr><td style='text-align: center;'>Batch size</td><td style='text-align: center;'>32, 64, 128</td></tr><tr><td style='text-align: center;'>Sampling trajectory size</td><td style='text-align: center;'>10, 25, 50</td></tr><tr><td style='text-align: center;'>Polyak averaging  $ \omega $</td><td style='text-align: center;'>0.995, 0.9998</td></tr><tr><td style='text-align: center;'>N° hidden layers</td><td style='text-align: center;'>2, 3</td></tr><tr><td style='text-align: center;'>Hidden layers size</td><td style='text-align: center;'>64, 128</td></tr><tr><td style='text-align: center;'>Mix embed. size</td><td style='text-align: center;'>32, 64</td></tr><tr><td style='text-align: center;'>Hypernet embed. size</td><td style='text-align: center;'>32, 64</td></tr><tr><td style='text-align: center;'>N° hypernet layers</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>N° Advantage hypernet layers</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Advantage hypernet embed. size</td><td style='text-align: center;'>32, 64</td></tr></table>

Table 5 lists the hyper-parameters considered in our experiments. When a parameter differs from the algorithm variations and environments, we indicate the values with a separator. Shared parameters between all the algorithms are indicated once.

## E ENVIRONMENTAL IMPACT

Despite each training run being “relatively” computationally inexpensive due to the use of CPUs, the experiments of our evaluation led to cumulative environmental impacts due to computations that run on computer clusters for an extended time. Nonetheless, it is crucial to foster sample efficiency (i.e., reducing the training time for the agents, hence the computational resources used to train them) to reduce the environmental footprint of such learning systems. In this direction, our work considers designing macro-action methods that significantly improve the sample efficiency of the learning algorithms (i.e., the number of simulation steps required to learn a policy), as shown by previous research on the topic (Xiao et al., 2020a;b).

Our experiments were conducted using a private infrastructure with a carbon efficiency of  $ \approx0.275\frac{kgCO_{2}eq}{kWh} $ , requiring a cumulative  $ \approx360 $  hours of computation. Total emissions are estimated

<div style="text-align: center;">Table 5: Hyper-parameters used in our experiments (considering all the algorithm variations).</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>AVF-VDN</td><td style='text-align: center;'>AVF-QMIX</td><td style='text-align: center;'>AVF-QPLEX</td></tr><tr><td style='text-align: center;'>Learning rate</td><td style='text-align: center;'>5e-4 — 2.5e-4</td><td style='text-align: center;'>5e-4 — 2.5e-4 — 2.5e-5</td><td style='text-align: center;'>5e-5 — 2.5e-4 — 2.5e-5</td></tr><tr><td style='text-align: center;'>$ \gamma $</td><td style='text-align: center;'></td><td style='text-align: center;'>0.9</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>ASCB size</td><td style='text-align: center;'></td><td style='text-align: center;'>2500</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>Batch size</td><td style='text-align: center;'></td><td style='text-align: center;'>32 — 64</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>Sampling traj. size</td><td style='text-align: center;'></td><td style='text-align: center;'>10 — 25</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>$ \omega $</td><td style='text-align: center;'></td><td style='text-align: center;'>0.995</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>N° hidden layers</td><td style='text-align: center;'></td><td style='text-align: center;'>2</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>Hidden layers size</td><td style='text-align: center;'></td><td style='text-align: center;'>64</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>Mix embed. size</td><td style='text-align: center;'>-</td><td style='text-align: center;'>32</td><td style='text-align: center;'>32</td></tr><tr><td style='text-align: center;'>Hypernet embed. size</td><td style='text-align: center;'>-</td><td style='text-align: center;'>32</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>N° hypernet. layers</td><td style='text-align: center;'>-</td><td style='text-align: center;'>2</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>N° Adv. hypernet layers</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Adv. hypernet embed. size</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>32</td></tr></table>

to be  $ \approx 10.39kgCO_{2}eq $  using the Machine Learning Impact calculator, and we purchased offsets for this amount through Freedom.

## F DOMAIN DESCRIPTION

### F.1 Box Pushing (BP)

In this collaborative task, two agents have to work together to push a big box to a goal area at the top of a grid world to obtain a higher credit than pushing the small box on each own. The small box is movable with a single agent, while the big one requires two agents to push it simultaneously.

The state space consists of each agent's position and orientation, as well as the location of each box. Agents have a set of primitive actions, including moving forward, turning left or right, and staying in place. The available macro-actions are Go-to-Small-Box(i) and Go-to-Big-Box that navigates the agent to a predefined waypoint (red) under the corresponding box and terminates with a pose facing it; and a Push macro-action that makes the agent move forward and terminate when the robot hits the world boundary or the big box. Each agent observation is very limited in both the primitive and macro level, which is the state of the front cell: empty, teammate, boundary, small box, or big box.

The team receives a terminal reward of +300 for pushing the big box to the goal area or +20 for pushing one small box to the goal area. If any agent hits the world’s boundary or pushes the big box on its own, a penalty of -10 is issued. An episode terminates when any box is moved to the goal area or reaches the maximum horizon, 100 time steps. In our work, we consider the variant of this task in terms of the grid world size as shown in Fig. 9.

The original work of Xiao et al. (2020a) also released a primitive action version of the BP task. In the primitive action version, each agent has four actions: move forward, turn left, turn right, and stay. The small box moves forward one grid cell when any robot faces it and executes the move forward action.

### F.2 WAREHOUSE TOOL DELIVERY (WTD)

Warehouse Tool Delivery scenarios vary in the number of agents, humans, and the speed at which they work. In each scenario, the humans assemble an item with four work phases. Each phase requires several primitive time steps and a specific tool. We assume that the human already holds the tool for the first phase, and the rest must be found and delivered in a particular order by a team of robots to finish the subsequent work phases. The objective of the robot team is to assist the humans in completing their tasks as quickly as possible by finding and delivering the correct tools in the proper order and timely fashion without making the humans wait.

The environmental space is continuous, and the global state includes 1) each mobile robot's 2D position; 2) the execution status of the manipulator robot's macro-action in terms of the rest of primitive time steps to terminate; 3) the work phase of each human with its completed percentage; and 4) each tool's position.

Mobile robots have three navigation macro-actions: 1) Go-W(i) moves the robot to the corresponding workshop and locates at the red spot in the end; 2) Go-TR leads the robot to the red waypoint in the middle of the tool room; 3) Get-Tool navigates the robot the pre-allocated waypoint beside the manipulator and wait there, which will not terminate until either receiving a tool or waiting there for 10 time steps. Mobile robots move at a fixed velocity and are only allowed to receive tools from the manipulator rather than the human. There are three applicable macro-actions for the manipulator robot: 1) Search-Tool(i) takes 6 time steps to find a particular tool and place it in a staging area when there are less than two tools there; otherwise, it freezes the robot for the same amount of time.; 2) Pass-to-M(i) takes 4 time steps to pick up the first found tool from the staging area and pass it to a mobile robot; 3) Wait-M consumes 1 time step to wait for a mobile robot.

Each mobile robot is always aware of its location and the type of tool carried by itself. Meanwhile, it is also allowed to observe the number of tools in the staging area or a human's current work phase when it is at the tool room or the corresponding workshop, respectively. The macro-observation of the manipulator robot is limited to the type of tools present in the staging area and the identity of the mobile robot waiting at the adjacent waypoints.

Rewards for this domain are structured such that the team earns a reward of +100 when they deliver a correct tool to a human on time. However, if the delivery is delayed, an additional penalty of -20 is imposed. Moreover, the team incurs a penalty of -10 if the manipulator robot attempts to pass a tool to a mobile robot that is not adjacent, and a penalty of -1 happens every time step.

We consider four variations of WTD shown in Fig. 8: a) WTD-S, involves one human and two mobile robots; b) WTD-D, involves two humans and two mobile robots; c) WTD-T, involves three humans and two mobile robots. d) WTD-F, involves four humans and three mobile robots. The human working speeds under different scenarios are listed in Table 6.

<div style="text-align: center;">Table 6: The number of time steps each human takes on each working phase in scenarios.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Scenarios</td><td style='text-align: center;'>WTD-S</td><td style='text-align: center;'>WTD-D</td><td style='text-align: center;'>WTD-T</td><td style='text-align: center;'>WTD-F</td></tr><tr><td style='text-align: center;'>Human-0</td><td style='text-align: center;'>[20, 20, 20, 20]</td><td style='text-align: center;'>[27, 20, 20, 20]</td><td style='text-align: center;'>[38, 38, 38, 38]</td><td style='text-align: center;'>[40, 40, 40, 40]</td></tr><tr><td style='text-align: center;'>Human-1</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>[27, 20, 20, 20]</td><td style='text-align: center;'>[38, 38, 38, 38]</td><td style='text-align: center;'>[40, 40, 40, 40]</td></tr><tr><td style='text-align: center;'>Human-2</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>[27, 27, 27, 27]</td><td style='text-align: center;'>[40, 40, 40, 40]</td></tr><tr><td style='text-align: center;'>Human-3</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>[40, 40, 40, 40]</td></tr></table>

Each episode stops when all humans have obtained the correct tools for all work phases or when the maximum time steps (150) are reached.

<div style="text-align: center;"><img src="imgs/img_in_image_box_339_1115_604_1383.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">(a) BP-10</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_611_1115_876_1381.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">(b) BP-30</div>


<div style="text-align: center;">Figure 7: Overview of the considered box pushing task variations.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_237_162_453_318.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(a) WTD-S</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_465_162_688_323.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(b) WTD-D</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_699_164_976_320.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;">(c)WTD-T</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_414_362_737_545.jpg" alt="Image" width="26%" /></div>


<div style="text-align: center;">(d) WTD-F</div>


<div style="text-align: center;">Figure 8: Overview of the considered warehouse tool delivery task variations.</div>


### F.3 CAPTURE TARGET (CT)

In this domain, there are 10 agents represented by blue circles, assigned with the task of capturing a randomly moving target indicated by a red cross (as shown in Fig. 22). Each agent's macro-observation captures the same information as its primitive one, including the agent's position (being always observable) and the target's position (being partially observable with a flickering probability of 0.3). The applicable primitive-actions include moving up, down, left, right, and stay. The macro-action set consists of Move-to-T, directs the agent to move towards the target with an updated target position according to the latest primitive observation, and Stay lasts a single time step. The horizon of this task is 60 time steps, and a terminal reward of +1 is given only when all agents capture the target simultaneously by being in the same cell.

<div style="text-align: center;"><img src="imgs/img_in_image_box_479_905_739_1164.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">Figure 9: Overview of the considered capture target task.</div>


## G Missing Plots from Section 5

For a clearer visualization of the results in Table 1, Figure 10 shows the normalized average return at convergence for all our algorithm variations and environments.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_214_274_1000_1266.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 10: Normalized average return for all our algorithm variations. Tasks have different characteristics affecting the performance of the different update schemes.</div>


In the following, we report all the training curves for the proposed algorithms, omitting the “AVF” prefix for simplicity.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_210_234_1007_762.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 11: Avg. return over training for AVF-VDN- $ \{D0, D1, D2\} $  using the macro-state.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_219_838_1009_1368.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 12: Avg. return over training for AVF-QMIX- $ \{D0, D1, D2\} $  using the macro-state.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_209_161_1009_693.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 13: Avg. return over training for AVF-QMIX–{D0, D1, D2}-MS using the joint macro-state.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_212_755_1012_1282.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 14: Avg. return over training for AVF-QPLEX- $ \{D0, D1, D2\} $  using the macro-state.</div>


<div style="text-align: center;">Moreover, Figure 16 shows the training curves for previous macro-action baselines (Xiao et al., 2022; 2020a; Xu et al., 2023).</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_211_269_1014_797.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 15: Avg. return over training for AVF-QPLEX-MS- $ \{D0, D1, D2\} $  using the joint macrostate.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_213_1088_1009_1308.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 16: Avg. return over training for Dec-MADDQN, Cen-MADDQN, Mac-IAICC, HAVEN.</div>