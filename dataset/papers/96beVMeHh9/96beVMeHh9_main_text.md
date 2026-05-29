# CAUSAL IDENTIFICATION FOR COMPLEX FUNCTIONAL LONGITUDINAL STUDIES

Andrew Ying

Irvine, CA 92606, USA

aying9339@gmail.com

## ABSTRACT

Real-time monitoring in modern medical research introduces functional longitudinal data, characterized by continuous-time measurements of outcomes, treatments, and confounders. This complexity leads to uncountably infinite treatment-confounder feedbacks and infinite-dimensional data, which traditional causal inference methodologies cannot handle. Inspired by the coarsened data framework, we adopt stochastic process theory, measure theory, and net convergence to propose a nonparametric causal identification framework. This framework generalizes classical g-computation, inverse probability weighting, and doubly robust formulas, accommodating time-varying outcomes subject to mortality and censoring for functional longitudinal data. We examine our framework through Monte Carlo simulations. Our approach addresses significant gaps in current methodologies, providing a solution for functional longitudinal data and paving the way for future estimation work in this domain.

## 1 INTRODUCTION

The advent of real-time monitoring technologies in healthcare has led to the continuous-time measurement of outcomes, treatments, and confounders, which we term "functional longitudinal data." Here by "functional" we mean the first-generation data described in Wang et al. (2016), or termed curve data (Gasser et al., 1984; Rice & Silverman, 1991; Gasser & Kneip, 1995), operating over time. For example, the Medical Information Mart for Intensive Care IV (MIMIC-IV) (Johnson et al., 2023) is a freely accessible electronic health record (EHR) database that records ICU care data, including physiological measurements, laboratory values, medication administration, and clinical events. Another example is Continuous Glucose Monitoring (CGM) (Rodbard, 2016; Klonoff et al., 2017), an increasingly adopted technology for insulin-requiring patients that provides insights into glycemic fluctuations. CGM offers a real-time, high-resolution stream of data, capturing the intricate fluctuations in interstitial fluid glucose levels every few minutes.

These examples illustrate the recent prevalence of functional longitudinal data, highlighting the necessity of a causal framework, as understanding treatment effects is of paramount interest in these settings. However, there is a great lack of investigation of causal inference at the intersection of longitudinal data and functional data. Even identifying causal parameters of interest through observed data becomes highly nontrivial in this setting, due to the issue of uncountably infinite treatment-confounder feedbacks (Hernán & Robins, 2020) within functional longitudinal data. Treatment-confounder feedbacks occur when treatments taken over time influence variables (confounders) that in turn affect future treatments. For example, in a medical study, a patient's current medication (treatment) could affect their future health (a confounder), and their health might determine which medications they receive later. This back-and-forth interaction over time creates a cycle that is difficult to disentangle when analyzing causal relationships. In functional longitudinal data, this feedback becomes even more complex because both treatments and confounders are recorded as continuous functions over time rather than at discrete time points.

Moreover, functional longitudinal data, modeled as infinite-dimensional, continuous-time stochastic processes, demand a measure-theoretic foundation to ensure mathematical rigor. This requirement introduces complexities far beyond the scope of classical causal inference, which typically assumes finite-dimensional data and more elementary statistical tools. Apart from the mathematical rigor,

from a statistical level, traditional approaches for handling functional data often rely on parametric or semi-parametric modeling assumptions, such as smoothness or sparsity, to facilitate analysis and reduce dimensionality. However, these assumptions are typically made for mathematical convenience rather than being grounded in prior knowledge. As a result, inferences drawn from such models may reflect the assumptions as much as, or more than, the data itself.

To bridge this gap, we aim to propose a novel identification framework for functional longitudinal data with time-varying outcomes subject to mortality and censoring, who enjoys the nonparametric property, making it more flexible and adaptable to various datasets.

We first define a causal quantity representing the mean of counterfactual outcomes under an idealized randomized world. To connect the observed data distribution to this idealized world, inspired by the coarsened data framework (Heitjan & Rubin, 1991) and through the application of continuous-time stochastic process theory and measure theory, we upgrade classical causal assumptions to accommodate functional longitudinal data nonparametrically. These together resolve the issue of uncountably infinite treatment-confounder feedbacks (Hernán & Robins, 2020) for functional longitudinal data. We generalize the well-known g-computation formula, inverse probability weighting formula, and double robust formula. We examine our identification framework through Monte Carlo simulations.

The paper is organized as follows. In Section 2 we present a literature review of related work. In Section 3, we define the notation and parameters of interest. Then we propose identification assumptions and generalize the well-known g-computation, inverse probability weighting, and double robust formulas (Hernán & Robins, 2020). Additionally, we prove that our identification is nonparametric. We conduct Monte Carlo simulations to examine our framework in Section 4. Section 5 discusses future directions. While this paper builds a population-level framework with numerical results, it does not explore estimation or associated inference, which is beyond the scope of this study and left for future research.

## 2 RELATED WORK

Causal Inference for Non-Functional Longitudinal Studies. Current causal frameworks for longitudinal studies fall into two main categories: "regular longitudinal studies," where time advances in fixed intervals (Greenland & Robins, 1986; Robins, 1986), and "irregular longitudinal studies," where events occur at random but discrete time points (Lok, 2008; Røysland, 2011; Rytgaard et al., 2022). "Regular longitudinal studies" are straightforward but limited to structured designs, while "irregular longitudinal studies," such as those by Rytgaard et al. (2022), accommodate random visit times by modeling treatment and confounder processes as counting processes. These approaches assume finite treatment-confounder feedbacks and finite-dimensional data, relying on stepwise paths and joint densities for causal identification.

However, modern medical studies often generate functional longitudinal data through continuous monitoring of treatments and confounders, as seen in intensive care settings (Johnson et al., 2016; 2018) and wearable devices for chronic disease management (Mastrototaro, 2000; Klonoff, 2005; Rodbard, 2016). Existing frameworks, designed for discrete-time or stepwise processes, are insufficient for such infinite-dimensional data, highlighting the need for new causal inference tools that accommodate the complexities of functional longitudinal data.

Causal Inference for Functional Data. Existing research on causal inference has examined functional data within observational studies, as highlighted in works by (Miao et al., 2020; Zhang et al., 2021; Tan et al., 2022). These studies share a similar data format with our analysis. However, our approach distinguishes itself by focusing on the time-dependent nature of longitudinal studies, where data evolve continuously over time. In contrast, the cited works primarily address "point exposure," which looks at the impact of a single treatment or covariates measured at beginning of a study, without accounting for how treatments or covariates may change and interact over a longer period.

Existing Work for Functional Longitudinal Data. The only exceptions that investigated causal inference for functional longitudinal data are Ying (2024a) and Sun & Crawford (2022). However, Ying (2024a) only investigated a single outcome, measured at the end of some medical studies, neither proving the nonparametric property nor conducting any numerical investigation. On the other hand, Sun & Crawford (2022) imposed stochastic differential equations with stringent parametric

assumptions. This situation highlights a significant gap in methodological advancements within the field. A related study (Ying, 2024b) explores a more general framework, building upon the methods and insights presented here. However, it does not include numerical examples or practical verifications, one of our contributions.

## 3 PROPOSED METHOD

### 3.1 PREPARATION

Consider a longitudinal study spanning from time 0 to  $ \infty $ :

•  $ A(t) $  and  $ L(t) $  are two stochastic processes denoting the treatment administered and the measured confounders, respectively, at any given time t. At any time,  $ A(t) $  and  $ L(t) $  could be binary, categorical, continuous, or even functional. We denote  $ \bar{A}(t) = \{A(s) : 0 \leq s \leq t\} $  and  $ \bar{L}(t) = \{L(s) : 0 \leq s \leq t\} $ , with  $ \bar{A} $  and  $ \bar{L} $  representing the collections of treatments and confounders over the entire study.

• We are interested in an outcome of interest  $ Y(t) $ , as a subset of  $ L(t) $ , that is,  $ Y(t) \subset L(t) $ . This notation was chosen purely for simplicity. We are not assuming  $ Y(t) $  must affect treatment assignment but instead allow this dependency to exist or not. This flexibility is critical as in many cases (e.g., disease progression), outcomes can influence treatment adjustments, and therefore acting as a confounder as well.

• Let $T$ be a time-to-event endpoint, for instance, death, and $C$ be the right censoring time. Define $X = \min(T, C)$ as the censored event time and $\Delta = \mathbb{1}(T \leq C)$ the event indicator. Therefore when $\Delta = 1$, $X = T$ and when $\Delta = 0$, $X = C$. We also define $N(t) = \mathbb{1}(X \leq t)$ as the counting processes of $X$.

• Write the counterfactual time-to-event endpoint  $ T_{\bar{a}} $  and counterfactual covariates  $ L_{\bar{a}}(t) $ , for any  $ \bar{a} \in A $ , where A encompasses all possible values of  $ \bar{a} $ . Therefore we have  $ X_{\bar{a}} = \min(T_{\bar{a}}, C) $  and  $ \Delta_{\bar{a}} = \mathbb{1}(T_{\bar{a}} < C) $ . We assume that the future cannot affect the past, that is,  $ \mathbb{1}(T_{\bar{a}} \geq t) = \mathbb{1}(T_{\bar{a}}' \geq t) $  and  $ L_{\bar{a}}(t) = L_{\bar{a}'}(t) $  whenever  $ \bar{a}(t) = \bar{a}'(t) $ . We also write  $ T_{A} = \{T_{\bar{a}} : \bar{a} \in A\} $  and  $ \bar{L}_{A} = \{\bar{L}_{\bar{a}} : \bar{a} \in A\} $ .

• The full data are  $ \{\bar{A}, C, T_{A}, \bar{L}_{A}\} $  and the observed data are  $ \{\bar{A}, X, \Delta, \bar{L}\} $ . Note that on the observed data level,  $ A(t) $  and  $ L(t) $  are not observed for  $ t \leq X $  or defined for  $ t \leq T $ . For easier notation in this paper, we offset  $ A(t) = A(X) $  and  $ L(t) = L(X) $  for observed data whenever t > X. In this way, the stochastic processes  $ A(t) $  and  $ L(t) $  are well defined at any t > 0.

• Define  $ \mathcal{F}_{t}=\sigma(\{A(s),L(s),\mathbb{1}(X\leq s),\mathbb{1}(X\leq s)\Delta:\forall s\leq t\}) $  as a filtration of information observed up to time t. Also we write  $ \mathcal{F}_{t-}=\sigma(\cup_{0\leq s<t}\mathcal{F}_{t}) $  and  $ \mathcal{G}_{t}=\sigma(\{\mathcal{F}_{t-},A(t)\}) $ . We define  $ G_{\infty+}=F_{\infty} $ . We write  $ F_{0-} $  and  $ G_{0-} $  as the trivial sigma algebra for convenience. Note that X is a stopping time with respect to  $ F_{t} $ , with  $ F_{\infty}=F_{X}=\sigma(\{\bar{A},X,\Delta,\bar{L}\}) $ .

• We use  $ \mathbb{P}(\mathrm{d}x\mathrm{d}\delta\mathrm{d}\bar{a}\mathrm{d}\bar{l}) $  (Bhattacharya & Waymire, 2007; Durrett, 2019; Gill & Robins, 2001) to represent the measure on the path space induced by the stochastic processes. Note that this is not a density function. $ ^{1} $  We use E as the corresponding expectation.

In the context of MIMIC-III,  $ A(t) $  could represent antibiotics usage at time t, and  $ L(t) $  may include a range of clinical measurements, such as severity of illness scores, vital signs, laboratory values, blood gas values, urine output, weight, height, age, gender, service type, total fluid intake, and total fluid output at time t. The outcome  $ Y(t) $  might measure illness progression influenced by antibiotics,

such as changes in severity scores over time. T could represent the time to discharge or mortality, with C as the time the patient is censored, such as at the end of data collection. Counterfactual outcomes like  $ T_{\bar{a}} $  might represent the time to recovery under a specific antibiotic regimen  $ \bar{a} $ , and  $ L_{\bar{a}}(t) $  could represent the trajectory of severity scores under that treatment.

Similarly, in the context of CGM,  $ A(t) $  represents insulin dosage at time t, and  $ L(t) $  includes glucose levels and immediate behavioral changes such as diet, medications, and physical activity at time t. The outcome  $ Y(t) $  represents the glucose levels monitored in real time in response to insulin adjustments. T could represent the time to a severe glucose event, with C as the time the patient stops CGM usage. Counterfactual outcomes like  $ T_{\bar{a}} $  represent the time to stable glucose control under a specific insulin dosing regime  $ \bar{a} $ , and  $ L_{\bar{a}}(t) $  captures the counterfactual glucose trajectory.

We are interested in learning a marginal mean of transformed potential outcomes including a time-to-event outcome and an outcome process under a user-specified treatment regime in the absence of censoring,

 $$ \int_{\mathcal{A}}\mathbb{E}(\nu(T_{\bar{a}},\bar{Y}_{\bar{a}}))\mathbb{G}(\mathbf{d}\bar{a}), $$ 

where  $ \nu $  is some user-specified function and G is a priori defined (signed) measure on A, representing a stochastic treatment regime. Here stochastic treatment regimes do not prescribe a specific treatment value but instead define the probability of receiving each possible treatment. In other words, it assigns treatments randomly according to a specified probability distribution. Stochastic treatment regimes offer a flexible approach for modeling treatments that are either continuous or challenging to precisely quantify. Unlike deterministic regimes, where treatment decisions are fixed, stochastic regimes introduce variability, enabling a broader range of real-world applications. This approach is particularly beneficial in scenarios where treatments are not strictly prescribed but instead follow probabilistic guidelines or are influenced by patient behavior or external factors. Examples of  $ \nu(\cdot) $ :

• $\nu(T_{\bar{a}}, \bar{Y}_{\bar{a}}) = \mathbb{1}(T_{\bar{a}} > t)$, for some time $t > 0$, identifies the effect of $\bar{a}$ on the survival probability. For instance, in MIMIC-III, this could represent the probability of a patient surviving beyond time $t$ under a specific antibiotic regimen $\bar{a}$. Alternatively, $\nu(T_{\bar{a}}, \bar{Y}_{\bar{a}}) = \min(T_{\bar{a}}, \tau)$ represents the restricted mean survival time.

•  $ \nu(T_{\bar{a}},\bar{Y}_{\bar{a}})=Y_{\bar{a}}(\tau) $  is the outcome measured at time  $ \tau $ , for some  $ \tau>0 $ . In CGM, it could correspond to the glucose level at time  $ \tau $  under a specific insulin dosing strategy  $ \bar{a} $ . Alternatively,  $ \nu(T_{\bar{a}},\bar{Y}_{\bar{a}})=Y_{\bar{a}}(T_{\bar{a}}) $  represents the outcome measured at the time-to-event  $ T_{\bar{a}} $ . In MIMIC-III, this might capture the severity of illness or lactate level at the time of recovery or death. Finally,  $ \nu(T_{\bar{a}},\bar{Y}_{\bar{a}})=\int_{0}^{\tau}w(t)Y_{\bar{a}}(t)dt/\tau $  represents the weighted averaged outcome over  $ [0,\tau] $ , where  $ w(t) $  is a user-specified weight function.

We assume  $ \mathbb{E}(\nu(T_{\bar{a}},\bar{Y}_{\bar{a}})) $  is integrable against G. This exploration encompasses marginal means under static treatment regimes, as discussed in various literature (Rytgaard et al., 2022; Cain et al., 2010; Young et al., 2011; Hernán & Robins, 2020). This quantity can be seen as the mean of counterfactual outcomes under an idealized randomized world, where  $ \bar{a} $  is randomized to follow a stochastic treatment regime G. Examples of G:

• $\mathbb{G}=\mathbb{1}(\bar{A}=\bar{a})$ representing the averaged treatment outcome under a specific regime is of interest. $\mathbb{G}=\mathbb{1}(\bar{A}=\bar{a})-\mathbb{1}(\bar{A}=\bar{a}')$ representing the averaged treatment effect of specific regime $\bar{a}$ versus another $\bar{a}'.$

• For treatments like physical activity, which is inherently variable and challenging to quantify precisely, can be modeled using stochastic regimes. For example, rather than prescribing a strict regimen of 30 minutes of exercise daily, a stochastic regime might increase the likelihood of patients engaging in activity based on encouragements or incentives. In both cases, G can be specified as a distribution instead of delta masses.

### 3.2 IDENTIFICATION ASSUMPTIONS

We have defined the parameter of interest (1). Intuitively if treating treatment process  $ \bar{A} $  as a selection process (Heitjan & Rubin, 1991), (1) is the mean of  $ \nu(T_{\bar{a}},\bar{Y}_{\bar{a}}) $  when  $ \bar{A} $  were to follow G and there is no censoring. To create such a pseudo-population, note that for any sequences of partitions

 $ \{\Delta_{K}[0,\infty]\}_{K=1}^{\infty} $ , where we have a partition  $ \Delta_{K}[0,\infty] $  over  $ [0,\infty] $  is a finite sequence of  $ K+1 $  numbers of the form  $ 0=t_{0}<\cdots<t_{K}=\infty $ , we loosely have the following decomposition

 $$ \mathbb{P}(\mathbf{d}x\mathbf{d}\delta\mathbf{d}\bar{a}\mathbf{d}\bar{l}) $$ 

 $$ =\prod_{j=0}^{K-1}F_{T}(t_{j+1}|\mathcal{F}_{t_{j}})^{\Delta(N(t_{j+1})-N(t_{j}))}(1-F_{T}(t_{j+1}|\mathcal{F}_{t_{j}}))^{(1-\Delta)N(t_{j+1})} $$ 

 $$ F_{C}(t_{j+1}|\mathcal{F}_{t_{j}})^{(1-\Delta)(N(t_{j+1})-N(t_{j}))}(1-F_{C}(t_{j+1}|\mathcal{F}_{t_{j}}))^{\Delta N(t_{j+1})} $$ 

 $$ \mathbb{P}(\bar{\mathbf{d l}}(t_{j+1}))|\mathcal{F}_{t_{j}})\mathbb{P}(\bar{\mathbf{d a}}(t_{j+1})|\mathcal{F}_{t_{j}})., $$ 

where we temporarily write  $ F_{T} $  and  $ F_{C} $  as the distribution functions of T and C. We intervene treatment distribution at each time  $ t_{j} $  to approximate the pseudo-population where  $ \bar{A} $  were to follow G as:

 $$ \mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\bar{a}\mathbf{d}\bar{l}) $$ 

 $$ =\prod_{j=0}^{i}F_{T}(t_{j+1}|\mathcal{F}_{t_{j}})^{\Delta(N(t_{j+1})-N(t_{j}))}[1-(1-\Delta)N(t_{j+1})] $$ 

 $$ \mathbb{P}(\mathbf{d}\bar{l}(t_{j+1}))|\mathcal{F}_{t_{j}})\mathbb{G}(\mathbf{d}\bar{a}(t_{j+1})|\bar{a}(t_{j})). $$ 

Here informally, one might understand this intervention as we replace the censoring distribution (4) and treatment distribution (5) between  $ (t_{j}, t_{j+1}] $  by no censoring as in (7) and targeted treatment distribution G as in (8). For readers unfamiliar with intervention-based causal inference language, we refer to Rytgaard et al. (2022, Definitions 1 & 2). A more formal and mathematically rigorous decompositions are given in Section A the appendix.

To eliminate confounder bias, we need to make sure there is no unmeasured confounders. We adapt the commonly known “coarsening at random” (Heitjan & Rubin, 1991) assumption into:

Assumption 1 (Full conditional randomization). The treatment assignment is independent of the all potential outcomes and covariates given history, in the sense that there exists a bounded function  $ \varepsilon(t,\eta)>0 $  with  $ \int_{0}^{\infty}\varepsilon(t,\eta)dt\to0 $  as  $ \eta\to0 $ , such that for any  $ t\in[0,\infty] $ ,  $ \eta>0 $ ,

 $$ \sup_{\bar{a}\in\mathcal{A}}\mathbb{E}(\|\mathbb{P}(\mathbf{d}t_{\bar{a}}\mathbf{d}\bar{l}_{\bar{a}}|\bar{A}(t+\eta),\mathcal{F}_{t})-\mathbb{P}(\mathbf{d}t_{\bar{a}}\mathbf{d}\bar{l}_{\bar{a}}|\mathcal{F}_{t})\|_{\mathrm{TV}})<\varepsilon(t,\eta), $$ 

where  $ \|\cdot\|_{TV} $  is the total variation norm over the path space’s signed measure space.

This assumption claims that, the treatment distribution, or equally, the probability of coarsening, in a small period of time around t, only depends on the observed data up to time t and independent of further part of counterfactuals. This assumption says in an approximating sense that there is no common cause between treatment decision between time  $ [t, t + \eta] $  and all future counterfactual confounders. Intuitively and unofficially, one might see this as saying  $ \mathbb{P}(\mathrm{dt}_{\bar{a}}\mathrm{d}\bar{l}_{\bar{a}}|\bar{A}(t + \eta), \mathcal{F}_{t}) \approx \mathbb{P}(\mathrm{dt}_{\bar{a}}\mathrm{d}\bar{l}_{\bar{a}}|\mathcal{F}_{t}) $ , or approximately,  $ (T_{\bar{a}}, \bar{L}_{\bar{a}}) \perp \bar{A}(t + \eta)|\mathcal{F}_{t} $ .

We also need an assumption over the censoring mechanism to eliminate the censoring bias. We consider the well-known conditionally independent censoring assumption (Tsiatis, 2006; Andersen et al., 2012). Define the full data censoring time hazard function as

 $$ \lambda_{C}(t|T,\bar{A},\bar{L})=\lim_{dt\to0}\mathbb{P}(C\leq t+\mathbf{d}t|C>t,T,\bar{A},\bar{L})/dt. $$ 

The following assumption requires that the full data censoring time hazard at time t only depends on the observed data up to time t.

Assumption 2 (Conditional independent censoring). The censoring mechanism is said to be conditionally independent if

 $$ \lambda_{C}(t|T,\bar{A},\bar{L})=\lim_{dt\to0}\mathbb{P}(C\leq t+\mathbf{d}t|C>t,T>t,\bar{A}(t),\bar{L}(t))\mathbb{1}(T>t)/dt. $$ 

Note that in order to overcome the continuous-time issue, here we impose Assumption 1 over an infinitesimal period of time. This type of idea is also adopted in Assumption 2. Note that how Assumption 2 is given on the intensity process whereas Assumption 1 is on the conditioning event. This is because one does not have intensity process for a general stochastic process.

With Assumptions 1 and 2, we are able to show that whenever  $ |\Delta_{K}[0,\infty]|\to0 $ ,  $ P_{\Delta_{K}[0,\infty],G} $  approximates a pseudo-measure where treatment distribution are intervened by uncountable times into following G, where the mesh  $ |\Delta_{K}[0,\infty]| $  of a partition  $ \Delta_{K}[0,\infty] $  is defined as  $ \max[\max_{i=0,\ldots,K-1}(t_{j+1}-t_{j}),1/t_{K-1}] $ , representing the maximum gap length of the partition:

Proposition 1 (Intervenable). Under Assumptions 1 and 2, the measures  $ P_{\Delta_{K}[0,\infty],G} $  converges to the same (signed) measure  $ \mathbb{P}_{\mathbb{G}} := \mathbb{P}(\mathrm{d}x_{\bar{a}}\mathrm{d}l_{\bar{a}})\mathbb{G}(\mathrm{d}\bar{a})\delta_{\bar{a}} $  in the total variation norm on the path space as the meshes shrink to zero, regardless of the choices of partitions, that is,

 $$ \|\mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\mathbf{d}\bar{a}\mathbf{d}\bar{l})-\mathbb{P}(\mathbf{d}x_{\bar{a}}\mathbf{d}\bar{l}_{\bar{a}})\mathbb{G}(\mathbf{d}\bar{a})\delta_{\bar{a}}\|_{\mathrm{T V}}\to0, $$ 

whenever |∆K[0,∞]| → 0.

We refer  $ P_{G} $  as the target distribution. This proposition has helped us to use the intervened observed data distribution  $ \mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathrm{d}x\mathrm{d}\delta\bar{d}\bar{d}l) $  to identify the target counterfactuals distribution  $ \mathbb{P}(\mathrm{d}x_{\bar{a}}\mathrm{d}\bar{l}_{\bar{a}})\mathbb{G}(\mathrm{d}\bar{a})\delta_{\bar{a}} $  in an asymptotic sense. The following assumption links the observed variable with the counterfactuals.

Assumption 3 (Full consistency). For any t,

 $$ T=T_{\bar{A}},L(t)=L_{\bar{A}}(t). $$ 

The full consistency assumption links the observed outcome and the potential outcome via the treatment actually received. It says that if an individual receives the treatment  $ \bar{A} = \bar{a} $ , then his/her observed outcome Y matches  $ Y_{\bar{a}} $ .

The following assumption ensures that the observed data can identify the target distribution. Assumption 4 (Positivity).



 $$ \mathbb{P}_{\mathbb{G}}\ll\mathbb{P}. $$ 

With the above assumptions, we are able to generalize the well-known identification formulas: g-computation, inverse probability weighting, and double robust formulas, into functional longitudinal data. Note that our assumptions can be weaker but chosen for ease to interpret.

### 3.3 IDENTIFICATION FORMULAS

Below we show how we generalize the well-known g-computation, inverse probability weighting, and double robust formulas for “functional longitudinal data.” For readers unfamiliar with the concepts, we refer to Hernán & Robins (2020). We also prepare a review for these formulas in discrete-time longitudinal data in Section B in the appendix.

Definition 1 (G-computation process). Under Assumptions 1 and 2, define

 $$ H_{\mathbb{G}}(t)=\mathbb{E}_{\mathbb{G}}[\nu(X,\bar{Y})|\mathcal{G}_{t}], $$ 

as a projection process, which is apparently a  $ P_{G} $ -martingale. We call  $ H_{\mathbb{G}}(t) $  the g-computation process. Note that

 $$ H_{\mathbb{G}}(\infty)=\nu(X,\bar{Y}),\ H_{\mathbb{G}}(0-)=\mathbb{E}_{\mathbb{G}}[\nu(X,\bar{Y})]. $$ 

The g-computation process intuitively serves as a consecutive adjustment of the target  $ \nu(X,\bar{Y}) $  from  $ \infty $  to 0. It represents a mix of original conditional distributions of covariate process together with the intervened treatment process G, from end of study to the beginning. Following this adjustment to the beginning of study, we have:

Theorem 1 (G-computation formula). Under Assumptions 1, 2, 3, and 4, (1) is identified via a g-computation formula as

 $$ \int_{\mathcal{A}}\mathbb{E}(\nu(T_{\bar{a}},\bar{Y}_{\bar{a}}))\mathbb{G}(\mathbf{d}\bar{a})=H_{\mathbb{G}}(0-). $$ 

Definition 2 (Inverse probability weighting process). Under Assumptions 1 and 2, define

 $$ Q_{\mathbb{G}}(t)=\mathbb{E}\left(\frac{\mathbf{d}\mathbb{P}_{\mathbb{G}}}{\mathbf{d}\mathbb{P}}\bigg|\mathcal{G}_{t}\right), $$ 

as the Radon-Nikodym derivative at any time t, which is apparently a P-martingale. We call  $ Q_{\mathbb{G}}(t) $  the inverse probability weighting process. Note that

 $$ Q_{\mathbb{G}}(\infty)=\mathbb{E}\left(\frac{\mathbf{d}\mathbb{P}_{\mathbb{G}}}{\mathbf{d}\mathbb{P}}\bigg|\mathcal{G}_{\infty}\right)=\frac{\mathbf{d}\mathbb{P}_{\mathbb{G}}}{\mathbf{d}\mathbb{P}},\ Q_{\mathbb{G}}(0-)=1. $$ 

The IPW process intuitively serves as a continuous adjustment of the treatment process  $ \bar{A} $  from 0 to  $ \infty $ , using which as weights one may create a pseudo population as if the whole process were to follow  $ P_{G} $ . It reweights the observed data distribution P into  $ P_{G} $  from the beginning of the study to the end. Following this reweighting throughout the longitudinal study, we have:

Theorem 2 (Inverse probability weighting formula). Under Assumptions 1, 2, 3, and 4, (1) is identified via an inverse probability weighting formula as

 $$ \int_{\mathcal{A}}\mathbb{E}(\nu(T_{\bar{a}},\bar{Y}_{\bar{a}}))\mathbb{G}(\mathbf{d}\bar{a})=\mathbb{E}\left[Q_{\mathbb{G}}(\infty)\nu(X,\bar{Y})\right]. $$ 

For any two  $ G_{t} $ -adapted processes  $ H(t) $  and  $ Q(t) $ , and a partition  $ \Delta_{K}[0,\infty] $ , we write

 $$ \Xi_{\Delta_{K}[0,\infty]}(H,Q)=\sum_{j=0}^{K}Q(t_{j})\left\{\int H(t_{j+1})\mathbb{G}(\mathbf{d}\bar{a}(t_{j+1})|\bar{A}(t_{j}))-H(t_{j})\right\}+\int H(0)\mathbb{G}(\mathbf{d}\bar{a}(0)). $$ 

We also define $\Xi(H,Q)$ as the limit of $\Xi_{\Delta_{K}[0,\infty]}(H,Q)$ in probability whenever it exists. We have:

Theorem 3 (Doubly robust formula). Under Assumptions 1, 2, 3, and 4, for any $\mathcal{G}_{t}$-adapted processes $H(t)$ and $Q(t)$ at the law where $\Xi(H,Q)$, as the limit of $\Xi_{\Delta_{K}[0,\infty]}(H,Q)$ in probability, exists and

 $$ \lim_{|\Delta_{K}[0,\infty]|\rightarrow0}\mathbb{E}(\Xi_{\Delta_{K}[0,\infty)}(H,Q])=\mathbb{E}(\Xi(H,Q)), $$ 

we have

 $$ \int_{\mathcal{A}}\mathbb{E}(\nu(T_{\bar{a}},\bar{Y}_{\bar{a}}))\mathbb{G}(\mathbf{d}\bar{a})=\mathbb{E}(\Xi(H,Q)), $$ 

provided that either  $ H = H_{G} $  or  $ Q = Q_{G} $ 

As one can see, the doubly robust formula provides extra protection against possible misspecification on either the g-computation process or the IPW process.

### 3.4 NO RESTRICTIONS ON THE OBSERVED DATA DISTRIBUTION: A NONPARAMETRIC FRAMEWORK

For functional data, where the complexity of continuous, infinite-dimensional outcomes makes it even harder to justify any specific model, relying on parametric assumptions becomes especially unrealistic. In this subsection, we demonstrate that our identification framework imposes no restrictions on the observed data. Our framework deliberately separates modeling assumptions from identification, focusing purely on structural assumptions necessary for causal inference. This ensures that the framework extracts information only from the data, avoiding the risk of introducing unwarranted or misleading conclusions based on arbitrary assumptions. This property is advantageous for researchers and practitioners because nonparametric frameworks are flexible and require minimal assumptions, making them robust and adaptable to diverse datasets. This aligns with the recent assumption-lean efforts in the causal inference community (Vansteelandt & Dukes, 2022; Vansteelandt et al., 2024).

We demonstrate this by proving that, for any given observed data distribution, we can identify a sequence of full data distributions—each satisfying Assumptions 1, 2, 3, and 4—such that their corresponding distributions on the observed data closely approximate the initial observed data distribution. That is, we write the set of all observed data distributions as P and its subset satisfying Assumptions 1, 2, 3, and 4 as M, then we show that M is a dense subset of P in the total variation norm.

Up to now, we have used P to represent both the distribution on the sample space and the path space. In this subsection, we use P to denote the distribution on the observed data  $ (\bar{A}, X, \Delta, \bar{L}) $  and  $ P^{F} $  to denote the distribution on the full data  $ (\bar{A}, C, T_{\mathcal{A}}, \bar{L}_{\mathcal{A}}) $ . We have

Theorem 4. When the path space consists of all piece-wise continuous processes, for any measure P over the observed data  $ (\bar{A}, X, \Delta, \bar{L}) $ , there exists a sequence of measures  $ P_{n}^{F} $  over the full data  $ (\bar{A}, C, T_{A}, \bar{L}_{A}) $  satisfying Assumptions 1, 2, 3, 4, whose inductions on the observed data converges to P in the total variation norm.

Technically, we have not achieved full nonparametric paradigm. However, we deem that the regularity condition “the path space is piece-wise continuous processes” is general enough for practical considerations. For example, both multivariate counting processes and continuous processes like Brownian process satisfy this regularity condition. It is noteworthy that achieving this “almost nonparametric” nature is the best one can hope for. This realization was confirmed in (Gill et al., 1997, Section 9) for “coarsening at random” assumption, even though our framework exhibits certain distinctions.

## 4 EXPERIMENT RESULT

In this section, we employ Monte Carlo simulations to empirically assess how the identification works. We decide to evaluate the performance of the g-computation formula only, for two reasons:

1. The g-computation formula is the only one that can be easily approximated through raw simulated data, whereas inverse probability weighting (and hence the doubly robust formula) cannot be directly approximated without estimation or computation. In fact, in causal inference with longitudinal data, the true causal effects are often not analytically computable. Instead, they are approximated numerically using methods like the g-computation formula through sampling like we outline below, with very large sample sizes, a standard practice for benchmarking estimator performance;

2. On the population level, the values of the three formulas are all the same, equaling (1). Therefore, approximating g-computation formula is sufficient for our purposes.

To that end, we need to go through 4 steps:

1. Come up with a reasonable data generating process;

2. Compute the parameter of interest (1) (or equivalently, the left-hand side of g-computation formula in Theorem 1) according to this data generating process;

3. Simulate according to this data generating process;

4. Approximate the right-hand side of g-computation formula in Theorem 1 using the simulated data.

Step 1: To sharp the focus and ease the computation, we consider a simple setting where there is no mortality or censoring  $ (T = C \equiv \infty) $ , or other measured confounding process, except for the outcome process itself. A more complicated scenario including mortality and censoring, and other confounding process, is considered in Section D in the appendix. We take glucose levels as the outcome and insulin levels as the treatment. Both glucose and insulin levels exhibit smooth, continuous changes over time. Gaussian processes are particularly well-suited for modeling such smooth and continuous temporal processes. For  $ t \in [0, 1] $ , consider a potential outcome process  $ Y_{a}(t) $  capturing potential logarithm of glucose levels, following a Gaussian process with mean process as

 $$ \mathbb{E}(Y_{\bar{a}}(t))=-a(t), $$ 

and covariance process as

 $$ \mathrm{C o v}[Y_{\bar{a}}(t),Y_{\bar{a}}(s)]=e^{-3|t-s|},\;\forall t,s\in[0,1]. $$ 

This ensures the joint dependence among  $ Y_{\bar{a}}(t) $  and negative treatment effect of logarithm of insulin level  $ \bar{a} $ . For instance,  $ Y_{\bar{a}}(t) $  can be log of blood glucose level. Define  $ \nu(T_{\bar{a}}, \bar{Y}_{\bar{a}}) $  as the integral of  $ \bar{Y}_{\bar{a}} $  over time  $ t \in [0, 1] $ , that is,

 $$ \nu(T_{\bar{a}},\bar{Y}_{\bar{a}})=\int_{0}^{1}Y_{\bar{a}}(t)d t. $$ 

Suppose the targeted treatment regime G is a Gaussian measure with mean process t - 0.5 and jointly independent normal variables at any time points. That is, the intervened A follows a Gaussian process with a mean process

 $$ \mathbb{E}(A(t))=t-0.5, $$ 

and covariance process

 $$ \mathrm{Cov}[A(t),A(s)]=e^{-3|t-s|},\;\forall t,s\in[0,1], $$ 

representing an increase of insulin level, possibly due to some insulin intake.

Step : Then we can show that (1) (or equivalently, the left-hand side of g-computation formula in Theorem 1) equals zero, that is,

 $$ \int\mathbb{E}(\nu(T_{\bar{a}},\bar{Y}_{\bar{a}}))\mathbb{G}(\mathbf{d}\bar{a})=\int\mathbb{E}\left[\int_{0}^{1}Y_{\bar{a}}(t)d t\right]\mathbb{G}(\mathbf{d}\bar{a})=\int\int_{0}^{1}\mathbb{E}(Y_{\bar{a}}(t))d t\mathbb{G}(\bar{a}) $$ 

 $$ =\int\left[\int_{0}^{1}a(t)dt\right]\mathbb{G}(\mathbf{d}\bar{a})=\int_{0}^{1}\int a(t)\mathbb{G}(\bar{a}(t))dt=\int_{0}^{1}(t-0.5)dt=0. $$ 

Step 3: In practice, we observe a stochastic process at finite points. We consider evenly splitting  $ t \in [0,1] $  into a grid of size  $ K+1 $ :  $ \Delta_{K}[0,1] = \{t_{0} = 0, t_{1} = 1/K, \cdots, t_{K-1} = (K-1)/K, t_{K} = 1\} $ , and for  $ 1 \leq i \leq n $ , according to G specified in Step 1, we simulate i.i.d. samples  $ A_{i}(t) $  according to G specified in Step 1 at  $ \Delta_{K}[0,1] $  as

 $$ \begin{pmatrix}A_{i}(t_{0})\\A_{i}(t_{1})\\\cdots\\A_{i}(t_{K-1})\\A_{i}(t_{K})\end{pmatrix}\sim\mathcal{N}\left[\begin{pmatrix}t_{0}-0.5\\t_{1}-0.5\\\cdots\\t_{K-1}-0.5\\t_{K}-0.5\end{pmatrix},\quad\begin{pmatrix}1&e^{-3|t_{1}-t_{0}|}&\cdots&e^{-3|t_{K-1}-t_{0}|}&e^{-3|t_{K}-t_{0}|}\\e^{-3|t_{1}-t_{0}|}&1&\cdots&e^{-3|t_{K-1}-t_{1}|}&e^{-3|t_{K}-t_{1}|}\\\cdots&\cdots&\cdots&\cdots&\cdots\\e^{-3|t_{K-1}-t_{0}|}&e^{-3|t_{K-1}-t_{1}|}&\cdots&1&e^{-3|t_{K}-t_{K-1}|}\\e^{-3|t_{K}-t_{0}|}&e^{-3|t_{K}-t_{1}|}&\cdots&e^{-3|t_{K}-t_{K-1}|}&1\end{pmatrix}\right]. $$ 

By according to the distribution of  $ Y_{\bar{a}}(t) $  specified in Step 1 and consistency, we generate  $ Y_{i}(t) $  at  $ \Delta_{K}[0,1] $  as

 $$ \begin{aligned}\begin{pmatrix}Y_{i}(t_{0})\\Y_{i}(t_{1})\\\cdots\\Y_{i}(t_{K-1})\\Y_{i}(t_{K})\end{pmatrix}\sim\mathcal{N}\left[\begin{pmatrix}A_{i}(t_{0})\\A_{i}(t_{1})\\\cdots\\A_{i}(t_{K-1})\\A_{i}(t_{K})\end{pmatrix},\quad\begin{pmatrix}1&e^{-|t_{1}-t_{0}|}&\cdots&e^{-|t_{K-1}-t_{0}|}&e^{-|t_{K}-t_{0}|}\\e^{-|t_{1}-t_{0}|}&1&\cdots&e^{-|t_{K-1}-t_{1}|}&e^{-|t_{K}-t_{1}|}\\\cdots&\cdots&\cdots&\cdots&\cdots\\e^{-|t_{K-1}-t_{0}|}&e^{-|t_{K-1}-t_{1}|}&\cdots&1&e^{-|t_{K}-t_{K-1}|}\\e^{-|t_{K}-t_{0}|}&e^{-|t_{K}-t_{1}|}&\cdots&e^{-|t_{K}-t_{K-1}|}&1\end{pmatrix}\right].\end{aligned} $$ 

Step 4: The integral of  $ Y_{i}(t_{k}) $  over [0,1] is  $ \sum_{k=0}^{K} Y_{i}(t_{k}) / (K + 1) $ . The approximate of the right-hand side of g-computation formula is  $ \sum_{i=1}^{n} \sum_{k=0}^{K} Y_{i}(t_{k}) / (K + 1) / n $ .

We vary the grid sizes  $ (K = 10, 50, 250) $  to examine how a denser grid improves the approximation. This approach simulates the scenario where the mesh  $ |\Delta_{K}[0,1]] $  is shrunk to zero. Additionally, we vary the sample sizes  $ (n = 100, 500, 2500) $  to explore how larger samples enhance the approximation, leveraging the law of large numbers to better approximate the right-hand side of the g-computation formula. We repeat the process R = 10,000 times. The resulting 10,000 approximations of  $ \sum_{i=1}^{n}\sum_{k=0}^{K}Y_{i}(t_{k})/(K+1)/n $  are presented in boxplots in Figure 1, where we append biases.

The simulation results demonstrate that the g-computation formula can adequately approximate (1) even with moderate sample and grid sizes. Increasing the sample size while keeping the grid size fixed enhances the accuracy and reduces the variance of the approximation. In contrast, increasing the grid size while keeping the sample size fixed does not consistently improve accuracy or reduce variance. However, simultaneously increasing both the sample and grid sizes significantly improves accuracy and reduces variance in the approximation.

## 5 CONCLUSION

In this work, we proposed on a novel theoretical framework for causal inference under functional longitudinal studies. We introduced three methodological paradigms for causal identification: the g-computation formula, inverse probability weighting formula, and doubly robust formula. This framework, noted for nonparametric foundation, substantiates and expands upon the estimand-based causal framework introduced by Ying (2024a). It incorporates considerations for time-varying outcomes and addresses complexities such as death and right censoring, marking a significant advancement in the analysis of functional longitudinal data and enhancing the toolkit for causal inference in this area.

Our focus is on the underlying curve data (Wang et al., 2016). At the population level, our framework abstracts away the sparsity or regularity of sample-level observations. In future work, we plan to extend our framework to sample-level data, where factors such as sparsity or irregularity could

<div style="text-align: center;"><img src="imgs/img_in_chart_box_319_165_904_616.jpg" alt="Image" width="47%" /></div>


<div style="text-align: center;">Figure 1: Simulation Results of using g-computation formula by varying grid sizes in K = 10, 50, 250 and sample sizes in n = 100, 500, 2500, for R = 10000 repeats. We plot boxplots and give biases.</div>


influence the consistency of estimators. For example, investigating how the number of observed time points  $ p_{n} $  scales with the sample size n in densely observed data could provide valuable insights.

There are significant theoretical and methodological opportunities, given the limited investigation on functional longitudinal data, for the machine learning, functional data analysis and causal inference communities. To list a few, first, adapting our framework to accommodate scenarios where Assumption 1 may not hold, including contexts involving time-dependent instrumental variables and time-dependent proxies (Ying et al., 2023), warrants rigorous exploration. Following the same spirit, dependent censoring can be considered, for instance, generalizing proxy method like Ying (2024c). Second, the positivity Assumption 4 in longitudinal studies faces practical challenges due to the potential scarcity of subjects adhering to specific treatment regimes within observed populations. One might consider using semiparametric models such as marginal structural models (Robins, 1998; Røysland, 2011) and structural nested models (Robins, 1999; Lok, 2008). Other solutions include dynamic treatment regimes (Fitzmaurice et al., 2008; Young et al., 2011; Rytgaard et al., 2022) and incremental interventions (Kennedy, 2017). Third, establishing the efficiency bound for our quantity of interest by leveraging semiparametric theory, represents an engaging challenge. Fourth, partial identification using discrete-time observations is a promising direction. Finally, developing a comprehensive estimation framework remains of ultimate interest.