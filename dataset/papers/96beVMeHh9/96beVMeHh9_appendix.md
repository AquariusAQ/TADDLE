## A FORMAL FORMULAS

For any partition  $ \Delta_{K}[0,\infty] $ , we have the following decomposition

 $$ \mathbb{P}(\mathbf{d}x\mathbf{d}\delta\mathbf{d}\bar{a}\mathbf{d}\bar{l}) $$ 

 $$ =\prod_{j=0}^{K-1}\mathbb{P}[T\leq t_{j+1}|\bar{l}(t_{j+1}),\bar{a}(t_{j+1}),\mathbb{1}(t_{j}<x\leq t_{j+1},\delta=0),\mathcal{F}_{t_{j}}]^{1(t_{j}<x\leq t_{j+1},\delta=1)} $$ 

 $$ \mathbb{P}[T>t_{j+1}|\bar{l}(t_{j+1}),\bar{a}(t_{j+1}),\mathbb{1}(t_{j}<x\leq t_{j+1},\delta=0),\mathcal{F}_{t_{j}}]^{1}(x\leq t_{j+1},\delta=0) $$ 

 $$ \mathbb{P}[C\leq t_{j+1}|\bar{l}(t_{j+1}),\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}}]^{1}(t_{j}<x\leq t_{j+1},\delta=0) $$ 

 $$ \mathbb{P}[C>t_{j+1}|\bar{l}(t_{j+1}),\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}}]^{1}(x\leq t_{j+1},\delta=1) $$ 

 $$ \mathbb{P}[\bar{\mathbf{d l}}(t_{j+1})|\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}}] $$ 

 $$ \mathbb{P}[\bar{\mathbf{d a}}(t_{j+1})|\mathcal{F}_{t_{j}}]. $$ 

We intervene treatment distribution at each time  $ t_{j} $  to approximate the pseudo-population where  $ \bar{A} $  were to follow G as:

 $$ \mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\bar{a}\mathbf{d}\bar{l}) $$ 

 $$ =\prod_{j=0}^{K-1}\mathbb{P}[T\leq t_{j+1}|\bar{l}(t_{j+1}),\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}}]^{1(t_{j}<x\leq t_{j+1},\delta=1)} $$ 

 $$ \left[1-\mathbb{1}(x\leq t_{j+1},\delta=0)\right] $$ 

 $$ \mathbb{P}[\bar{\mathbf{d l}}(t_{j+1})|\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}}] $$ 

 $$ \mathbb{G}[\mathbf{d}\bar{a}(t_{j+1})|\bar{a}(t_{j})]. $$ 

## B A REVIEW OF IDENTIFICATION FORMULAS FOR REGULAR LONGITUDINAL STUDIES

The following is a review of existing methods and rewritten by our language. Consider a longitudinal study with data collected at fixed times  $ t = 0, \ldots, K $ . To align with the notation used in this paper, let  $ \tau = K $ . Here,  $ A(t) $  and  $ L(t) $  can only change at discrete time points t = k. The associated filtrations are defined as  $ \mathcal{F}_{k} = \sigma(\bar{A}(k), \bar{L}(k)) $ ,  $ \mathcal{F}_{k-} = \sigma(\bar{A}(k-1), \bar{L}(k-1)) $ , and  $ \mathcal{G}_{k} = \sigma(\bar{A}(k), \bar{L}(k-1)) $ . For a finite set of random variables, the measure on the path space induced by P corresponds to a multivariate distribution. Denote the observed data density or probability mass function as  $ p(\bar{a}, \bar{l}) $ , and use  $ p(\cdot | \cdot) $  for conditional densities or probabilities.

The observed data likelihood can be uniquely factorized based on the temporal sequence of events:

 $$ p(\bar{a},\bar{l})=\prod_{j=0}^{K}\left[p[l(j)|\bar{a}(j),\bar{l}(j-1)]p[a(j)|\bar{a}(j-1),\bar{l}(j-1)]\right], $$ 

where ¯(−1) = ¯a(−1) = ∅ for notational convenience.

To identify (1), the following positivity assumption is imposed:

 $$ \mathrm{g}(\bar{a})\ll\prod_{j=0}^{K}p\{a(j)|\bar{a}(j-1),\bar{L}(j-1)\}, $$ 

almost surely over  $ \bar{L} $ . Alternatively, a stricter version requires:

 $$ p\{a(j)|\bar{a}(j-1),\bar{L}(j-1)\}>0, $$ 

almost surely over  $ \bar{L}(j-1) $  for any  $ \bar{a} $ . For example, in the case of a binary treatment, if  $ \mathrm{g}(\bar{A}) = \mathbb{1}(\bar{A} = \bar{0}) $  (i.e., “always under control”), the positivity assumption ensures that for any patient history  $ \bar{L} $ , the probability of remaining under control is nonzero.

Given the positivity assumption, one can substitute  $  p[a(j)|\bar{a}(j-1), \bar{l}(j-1)]  $  in (43) with  $  g\{a(j)|\bar{a}(j-1)\}  $ , leading to the target distribution:

 $$ p_{\mathbb{G}}(\bar{a},\bar{l})=\prod_{j=0}^{K}\left\{p[l(j)|\bar{a}(j),\bar{l}(j-1)]p_{\mathbb{G}}[a(j)|\bar{a}(j-1)]\right\}. $$ 

This distribution enables the identification of  $ \mathbb{E}_{\mathbb{G}}(\nu(L(K))) $ , where  $ E_{G} $  represents the expectation under  $ p_{G} $ . Additional assumptions are required to interpret  $ \mathbb{E}_{\mathbb{G}}(\nu(L(K))) $  causally:

 $$ \int_{\mathcal{A}}\mathbb{E}(\nu(L_{\bar{a}}(K)))\mathrm{g}(\bar{a})\mathrm{d}\bar{a}=\mathbb{E}_{\mathbb{G}}(\nu(L(K))). $$ 

The first assumption is consistency:

 $$ L(K)=L_{\bar{A}}(K), $$ 

almost surely. This states that the observed outcome matches the potential outcome under the received treatment regime  $ \bar{A} = \bar{a} $ .

The second is the sequential randomization assumption (SRA), also referred to as “no unmeasured confounders.” It posits that the treatment  $ A(k) $  at time k is conditionally independent of the potential outcome  $ L_{\bar{a}}(K) $  given past treatment and covariate history:

 $$ A(k)\perp L_{\bar{a}}(K)\mid\bar{A}(k-1),\bar{L}(k-1), $$ 

for any  $ \bar{a}\inA $  and  $ k=0,\ldots,K $ . This ensures exchangeability across treatment groups within strata defined by observed covariates. Note that alternative frameworks relax this assumption (Tchetgen Tchetgen et al., 2018; 2020; Ying et al., 2023), but they are beyond the scope of this discussion.

The g-computation formula (Greenland & Robins, 1986; Robins, 2000) and inverse probability weighting (IPW) (Hernán et al., 2000; 2002) are two commonly used approaches for identification. The g-computation formula identifies (47) through iterative conditional expectations:

 $$ \int_{\mathcal{A}}\mathbb{E}(\nu(L_{\bar{a}}(K)))\mathrm{g}(\bar{a})d\bar{a}=\int\nu(l(K))\prod_{j=0}^{K}\left\{p[l(j)|\bar{a}(j),\bar{l}(j-1)]\mathrm{g}\{a(j)|\bar{a}(j-1)\}\mathrm{d}l(j)\mathrm{d}a(j)\right\} $$ 

Alternatively, the IPW formula uses pseudoweights derived from the Radon-Nikodym derivative:

 $$ \int_{\mathcal{A}}\mathbb{E}(\nu(L_{\bar{a}}(K)))\mathrm{g}(\bar{a})d\bar{a}=\mathbb{E}\left\{\frac{\mathrm{g}(\bar{A})\nu(L(K))}{\prod_{j=0}^{K}p[A(j)|\bar{A}(j-1),\bar{L}(j-1)]}\right\}. $$ 

Lastly, the doubly robust (DR) formula combines both approaches (Bang & Robins, 2005; Van der Laan & Robins, 2003):

 $$ \begin{align*}\int_{\mathcal{A}}\mathbb{E}(\nu(L_{\bar{a}}(K)))\mathrm{g}(\bar{a})d\bar{a}\quad&(52)\\ =&\mathbb{E}\left(Q_{\mathbb{G}}(K)\nu(L(K))-\sum_{k=0}^{K}\left\{Q_{\mathbb{G}}(k)H_{\mathbb{G}}(k)-Q_{\mathbb{G}}(k-1)\int H_{\mathbb{G}}(k)\mathrm{g}[a(k)|\bar{A}(k-1)]\mathrm{d}a(k)\right\}\right).\end{align*} $$ 

For additional details, see Robins (1986; 1997); Hernán & Robins (2020).

## C PROOFS

### C.1 PROOF OF PROPOSITION 1

We temporarily define  $ \mathcal{F}_{/C,t}=\sigma(\{L(s),A(s),\mathbb{1}(T\leq s):\forall s\leq t\}) $  as the censoring free filtration and  $ \mathcal{F}_{\bar{a},t}=\sigma(\{L_{\bar{a}}(s),\mathbb{1}(T_{\bar{a}}\leq s):\forall s\leq t\}) $  as the counterfactual filtration:

 $$ \begin{aligned}&\left\|\mathbb{P}_{\Delta\kappa[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\mathbf{d}\bar{a}\bar{d}\bar{l})-\mathbb{P}(\mathbf{d}x_{\bar{a}}\mathbf{d}l_{\bar{a}})\mathbb{G}(\bar{\mathbf{d}}\bar{\alpha})\delta_{\bar{a}}\right\|_{\mathrm{TV}}\\&=\left\|\prod_{j=0}^{K-1}\mathbb{G}(\bar{\mathbf{d}\bar{a}}(t_{j+1})|\bar{a}(t_{j}))[1-\mathbb{1}(x\leq t_{j+1},\delta=0)]\mathbb{P}(T\leq t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}})^{1(t_{j}<x\leq t_{j+1})}\right.\\&\quad\left.\quad\mathbb{P}(T>t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}})^{1-1(t_{j}<x\leq t_{j+1})}\mathbb{P}(\bar{\mathbf{d}\bar{l}}(t_{j+1})|\bar{a}(t_{j}),\mathcal{F}_{t_{j}})\right.\\&\quad\left.\quad-\mathbb{P}(\mathbf{d}x_{\bar{a}}\mathbf{d}l_{\bar{a}})\mathbb{G}(\bar{\mathbf{d}\bar{a}})\delta_{\bar{a}}\right\|_{\mathrm{TV}}\\&\leq\left\|\prod_{j=0}^{K-1}\mathbb{G}(\bar{\mathbf{d}\bar{a}}(t_{j+1})|\bar{a}(t_{j}))[1-\mathbb{1}(x\leq t_{j+1},\delta=0)]\right.\\&\quad\left.\quad\mathbb{P}(T\leq t_{j+1}|\mathcal{F}_{/C,t_{j}})^{1(t_{j}<x\leq t_{j+1})}\mathbb{P}(T>t_{j+1}|\mathcal{F}_{/C,t_{j}})^{1-1(t_{j}<x\leq t_{j+1})}\mathbb{P}(\bar{\mathbf{d}\bar{l}}(t_{j+1})|\mathcal{F}_{/C,t_{j}})\right.\\&\quad\left.\quad-\mathbb{P}(\mathbf{d}x_{\bar{a}}\mathbf{d}l_{\bar{a}})\mathbb{G}(\bar{\mathbf{d}\bar{a}})\delta_{\bar{a}}\right\|_{\mathrm{TV}}+o(1)\\&=\left\|\prod_{j=0}^{K-1}\mathbb{G}(\bar{\mathbf{d}\bar{a}}(t_{j+1})|\bar{a}(t_{j}))\{1-\mathbb{1}(x_{\bar{a}}\leq t_{j+1},\delta_{\bar{a}}=0)\}\mathbb{P}(T_{\bar{a}}\leq t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{\bar{a},t_{j}})^{1(t_{j}<x_{\bar{a}}\leq t_{j+1},\delta_{\bar{a}}=1)}\right.\\&\quad\left.\quad\mathbb{P}(T_{\bar{a}}>t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{\bar{a},t_{j}})^{1-1(t_{j}<x_{\bar{a}}\leq t_{j+1},\delta_{\bar{a}}=1)}\mathbb{P}(\bar{\mathbf{d}\bar{l}}_{\bar{a}}(t_{j+1})|\bar{a}(t_{j+1}),\mathcal{F}_{\bar{a},t_{j}})\right.\\&\quad\left.\quad-\mathbb{P}(\mathbf{d}x_{\bar{a}}\mathbf{d}l_{\bar{a}})\mathbb{G}(\bar{\mathbf{d}\bar{a}})\delta_{\bar{a}}\right\|_{\mathrm{TV}}\\&=\left\|\prod_{j=0}^{K-1}\mathbb{G}(\bar{\mathbf{d}\bar{a}}(t_{j+1})|\bar{a}(t_{j}))[1-\mathbb{1}(x_{\bar{a}}\leq t_{j+1},\delta_{\bar{a}}=0)]\right.\\&\quad\left.\quad\{\mathbb{P}(T_{\bar{a}}\leq t_{j+1}|C>t_{j+1},\bar{a}(t_{j+1}),\mathcal{F}_{\bar{a},t_{j}})^{1(t_{j}<x_{\bar{a}}\leq t_{j+1},\delta_{\bar{a}}=1)}\right.\\&\quad\left.\mathbb{P}(T_{\bar{a}}>t_{j+1}|C>t_{j+1},\bar{a}(t_{j+1}),\mathcal{F}_{\bar{a},t_{j}})^{1-1(t_{j}<x_{\bar{a}}\leq t_{j+1},\delta_{\bar{a}}=1)}\mathbb{P}(\bar{\mathbf{d}\bar{l}}_{\bar{a}}(t_{j+1})|C>t_{j},\bar{a}(t_{j+1}),\mathcal{F}_{\bar{a},t_{j}})\right.\\&\quad\left.\quad-\mathbb{P}(T_{\bar{a}}\leq t_{j+1}|\mathcal{F}_{\bar{a},t_{j}})^{1(t_{j}<x_{\bar{a}}\leq t_{j+1})}\mathbb{P}(T_{\bar{a}}>t_{j+1}|\mathcal{F}_{\bar{a},t_{j}})^{1-1(t_{j}<x_{\bar{a}}\leq t_{j+1})}\mathbb{P}(\bar{\mathbf{d}\bar{l}}_{\bar{a}}(t_{j+1})|\mathcal{F}_{\bar{a},t_{j}})\right\|\Bigg\|_{TV}\to0\end{aligned} $$ 

### C.2 Proof of Theorem 1

Since  $ \mathbb{P}_{\mathbb{G}}(\mathrm{d}x\mathrm{d}\delta\mathrm{d}\bar{a}\mathrm{d}\bar{l}) $  is a limit of measures in total variation norm of

 $$ \mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\bar{a}\mathbf{d}\bar{l}), $$ 

whenever |∆K[t, ∞]| → 0, we have

 $$ \int f(x,\delta,\bar{a},\bar{l})\mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\mathbf{d}\bar{a}\mathbf{d}\bar{l})\to\mathbb{E}_{\mathbb{G}}[f(X,\Delta,\bar{A},\bar{L})], $$ 

for any bounded functions  $  f(x, \delta, \bar{a}, \bar{l})  $ . Therefore, we have

 $$ \begin{aligned}&\left|H_{\mathbb{G}}(0-)-\int\mathbb{E}(\nu(T_{\bar{a}},Y_{\bar{a}}))\mathbb{G}(\bar{\mathbf{d}a})\right|\\&\leq\left|\int\nu(x,\bar{y})\mathbb{P}(\mathbf{d}x\mathbf{d}\delta\bar{\mathbf{d}a}\bar{\mathbf{d}l})-\int\nu(x,\bar{y})\mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\bar{\mathbf{d}a}\bar{\mathbf{d}l})\right|\\&+\left|\int\nu(x,\bar{y})\mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\bar{\mathbf{d}a}\bar{\mathbf{d}l})-\int\mathbb{E}(\nu(T_{\bar{a}},Y_{\bar{a}}))\mathbb{G}(\bar{\mathbf{d}a})\right|\\&=\left|\int\nu(x,\bar{y})\mathbb{P}_{\Delta_{K}[0,\infty],\mathbb{G}}(\mathbf{d}x\mathbf{d}\delta\bar{\mathbf{d}a}\bar{\mathbf{d}l})-\int\mathbb{E}(\nu(T_{\bar{a}},Y_{\bar{a}}))\mathbb{G}(\bar{\mathbf{d}a})\right|+o(1)\\&=\left|\int\nu(x,\bar{y})\prod_{j=0}^{K-1}\mathbb{P}(T\leq t_{j+1}|C>t_{j+1},\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}})^{1(t_{j}<x\leq t_{j+1},\delta=1)}\right.\\&\left.\mathbb{P}(T>t_{j+1}|C>t_{j+1},\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}})^{1-1(t_{j}<x\leq t_{j+1},\delta=1)}\mathbb{P}(\bar{\mathbf{d}l}(t_{j+1})|C>t_{j+1},\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}})\right.\\&\left.[1-\mathbb{1}(x\leq t_{j+1},\delta=0)]\mathbb{G}(\bar{\mathbf{d}a}(t_{j+1})|\bar{a}(t_{j}))-\int\mathbb{E}(\nu(T_{\bar{a}},Y_{\bar{a}}))\mathbb{G}(\bar{\mathbf{d}a})\right|+o(1),\end{aligned} $$ 

where  $ o(1) $  converges to zero when  $ \Delta_{K}[0,\infty]\to0 $ . By Assumptions 1, 2, 3, and 4, the above term is less than or equal to

 $$ \begin{aligned}&\left|\int\nu(x,\bar{y})\mathbb{P}(T\leq t_{K}|\bar{a}(t_{K}),\mathcal{F}_{t_{K-1}})^{1(t_{K-1}<x\leq t_{K},\delta=1)}\right.\\&\left.\mathbb{P}(T>t_{K}|\bar{a}(t_{K}),\mathcal{F}_{t_{K-1}})^{1-1(t_{K-1}<x\leq t_{K},\delta=1)}\mathbb{P}(\bar{\mathbf{d}\bar{l}}(t_{K})|\bar{a}(t_{K}),\mathcal{F}_{t_{K-1}})\right.\\&\left.\left[1-\mathbb{1}(x\leq t_{K},\delta=0)\right]\mathbb{G}(\bar{\mathbf{d}\bar{a}}(t_{K})|\bar{a}(t_{K-1}))\prod_{j=0}^{K-1}\mathbb{P}(T\leq t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})^{1(t_{j}<x\leq t_{j+1},\delta=1)}\right.\right.\\&\left.\mathbb{P}(T>t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})^{1-1(t_{j}<x\leq t_{j+1},\delta=1)}\mathbb{P}(\bar{\mathbf{d}\bar{l}}(t_{j+1})|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})\right.\\&\left.\left[1-\mathbb{1}(x\leq t_{j+1},\delta=0)\right]\mathbb{G}(\bar{\mathbf{d}\bar{a}}(t_{j+1})|\bar{a}(t_{j}))-\int\mathbb{E}(\nu(T_{\bar{a}},Y_{\bar{a}}))\mathbb{G}(\bar{\mathbf{d}\bar{a}})\right|+o(1),\right.\end{aligned} $$ 

which by Assumptions 1, 2, 3, and 4, equals

 $$ \begin{aligned}&\left|\int\nu(x,\bar{y})\mathbb{P}(T_{\bar{a}}\leq t_{K}|\bar{a}(t_{K}),\mathcal{F}_{t_{K-1}})^{1(t_{K-1}<x_{\bar{a}}\leq t_{K},\delta_{\bar{a}}=1)}\right.\\&\left.\mathbb{P}(T_{\bar{a}}>t_{K}|\bar{a}(t_{K}),\mathcal{F}_{t_{K-1}})^{1-1(t_{K-1}<x_{\bar{a}}\leq t_{K},\delta_{\bar{a}}=1)}\mathbb{P}(\bar{\mathbf{d l}}_{(t_{K})}|\bar{a}(t_{K}),\mathcal{F}_{t_{K-1}})\right.\\&\left.[1-\mathbb{1}(x_{\bar{a}}\leq t_{K},\delta_{\bar{a}}=0)]\mathbb{G}(\bar{\mathbf{d a}}(t_{K})|\bar{a}(t_{K-1}))\prod_{j=0}^{K-1}\mathbb{P}(T\leq t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})^{1(t_{j}<x\leq t_{j+1},\delta=1)}\right.\\&\left.\mathbb{P}(T>t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})^{1-1(t_{j}<x\leq t_{j+1},\delta=1)}\mathbb{P}(\bar{\mathbf{d l}}_{(t_{j+1})}|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})\right.\\&\left.[1-\mathbb{1}(x\leq t_{j+1},\delta=0)]\mathbb{G}(\bar{\mathbf{d a}}(t_{j+1})|\bar{a}(t_{j}))-\int\mathbb{E}(\nu(T_{\bar{a}},Y_{\bar{a}}))\mathbb{G}(\bar{\mathbf{d a}})\Bigg|+o(1),\right.\end{aligned} $$ 

which by Assumptions 1, 2, 3, and 4, is less than or equal to

 $$ \begin{align*}&\left|\int\nu(x,\bar{y})\mathbb{P}(T_{\bar{a}}\leq t_{K}|\mathcal{F}_{t_{K-1}})^{1(t_{K-1}<x_{\bar{a}}\leq t_{K},\delta_{\bar{a}}=1)}\right.\\&\left.\mathbb{P}(T_{\bar{a}}>t_{K}|\mathcal{F}_{t_{K-1}})^{1-1(t_{K-1}<x_{\bar{a}}\leq t_{K},\delta_{\bar{a}}=1)}\mathbb{P}(\bar{d}\bar{l}_{a}(t_{K})|\mathcal{F}_{t_{K-1}})\right.\\&\left.\left[1-\mathbb{1}(x_{\bar{a}}\leq t_{K},\delta_{\bar{a}}=0)\right]\mathbb{G}(\bar{d}\bar{a}(t_{K})|\bar{a}(t_{K-1}))\prod_{j=0}^{K-1}\mathbb{P}(T\leq t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})^{1(t_{j}<x\leq t_{j+1},\delta=1)}\right.\\&\left.\mathbb{P}(T>t_{j+1}|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})^{1-1(t_{j}<x\leq t_{j+1},\delta=1)}\mathbb{P}(\bar{d}\bar{l}(t_{j+1})|\bar{a}(t_{j+1}),\mathcal{F}_{/C,t_{j}})\right.\\&\left.\left[1-\mathbb{1}(x\leq t_{j+1},\delta=0)\right]\mathbb{G}(\bar{d}\bar{a}(t_{j+1})|\bar{a}(t_{j}))-\int\mathbb{E}(\nu(T_{\bar{a}},Y_{\bar{a}}))\mathbb{G}(\bar{d}\bar{a})\right|+o(1).\end{align*} $$ 

By iterating the above process for  $ 0 \leq j \leq K - 2 $ , we arrive the conclusion.

### C.3 Proof of Theorem 2

The proof is immediate by noting that

 $$ \mathbb{E}\left[Q_{\mathbb{G}}(\infty)\nu(X,\bar{Y})\right]=\mathbb{E}_{\mathbb{G}}[\nu(X,\bar{Y})]=\int_{\mathcal{A}}\mathbb{E}(\nu(T_{\bar{a}},\bar{Y}_{\bar{a}}))\mathbb{G}(\mathbf{d}\bar{a}), $$ 

by Theorem 1.

### C.4 Proof of Theorem 3

We first prove the theorem when  $ H = H_{G} $ . Indeed, as the limit and expectation can interchange, we can show that

 $$ \begin{aligned}&\left|\mathbb{E}\left[\Xi_{\mathrm{out},\Delta_{K}[0,\infty]}(H_{\mathbb{G}},Q)\right]\right|\\&=\left|\mathbb{E}\left\{Q(t_{K})[\nu(X,\bar{Y})-H_{\mathbb{G}}(t_{K})]\right\}\right.\\&\left.+\sum_{j=1}^{K-1}\mathbb{E}\left(Q(t_{j})\right\}\int H_{\mathbb{G}}(t_{j+1})\mathbb{G}(\bar{\boldsymbol{d}\alpha}(t_{j+1})|\bar{A}(t_{j}))-H_{\mathbb{G}}(t_{j})\right\}\right)\Bigg|\\&=\left|0+\sum_{j=1}^{K-1}\mathbb{E}\left(Q(t_{j})\left\{\int H_{\mathbb{G}}(t_{j+1})\mathbb{G}(\bar{\boldsymbol{d}\alpha}(t_{j+1})|\bar{A}(t_{j}))-H_{\mathbb{G}}(t_{j})\right\}\right)\right|\\&\leq\sum_{j=1}^{K-1}\left|\mathbb{E}\left(Q(t_{j})\left\{\int H_{\mathbb{G}}(t_{j+1})\mathbb{G}(\bar{\boldsymbol{d}\alpha}(t_{j+1})|\bar{A}(t_{j}))-H_{\mathbb{G}}(t_{j})\right\}\right)\right|\\&\leq\sum_{j=1}^{K-1}\left|\mathbb{E}\left(Q(t_{j})\left\{\int H_{\mathbb{G}}(t_{j+1})\mathbb{G}(\bar{\boldsymbol{d}\alpha}(t_{j+1})|\bar{A}(t_{j}))-\mathbb{E}_{\mathbb{G}}\left[H_{\mathbb{G}}(t_{j+1})|\mathcal{G}_{t_{j}}\right]\right\}\right)\right|\\&\leq\sum_{j=0}^{K}\kappa\|H_{\mathbb{G}}(t_{j})Q(t_{j})\|_{1}(t_{j+1}-t_{j})^{\alpha}\\&\leq\kappa\sup_{t}\|H_{\mathbb{G}}(t)Q(t)\|_{1}\sum_{j=1}^{K-1}(t_{j+1}-t_{j})^{\alpha}\rightarrow0,\end{aligned} $$ 

when  $ \left|\Delta_{K}[0,\infty]\right|\to0 $ , where we have used the fact that  $ H_{\mathbb{G}}(t) $  is a  $ P_{G} $ -martingale and Assumptions 1, 2.

We now proceed to the case when  $ Q = Q_{G} $ . Indeed, as the limit and expectation can interchange, we can show that

 $$ \begin{aligned}&\left|\mathbb{E}\left[\Xi_{\mathsf{t r t},\Delta_{K}[0,\infty]}(H,Q_{\mathbb{G}})\right]\right|\\&=\left|\mathbb{E}\left(\sum_{j=0}^{K}\left\{Q_{\mathbb{G}}(t_{j})H(t_{j})-Q_{\mathbb{G}}(t_{j-1})\int H(t_{j})\mathbb{G}(\mathsf{d}\bar{a}(t_{j})|\bar{A}(t_{j-1}))\right\}\right)\right.\\&\left.-\mathbb{E}\left\{Q_{\mathbb{G}}(0)H(0)-\int H(0)\mathbb{G}(\mathsf{d}\bar{a}(0))\right\}\right|\\&=\left|\mathbb{E}\left(\sum_{j=0}^{K}\left\{Q_{\mathbb{G}}(t_{j})H(t_{j})-Q_{\mathbb{G}}(t_{j-1})\int H(t_{j})\mathbb{G}(\mathsf{d}\bar{a}(t_{j})|\bar{A}(t_{j-1}))\right\}\right)+0\right|\\&\leq\sum_{j=0}^{K}\left|\mathbb{E}\left\{Q_{\mathbb{G}}(t_{j})H(t_{j})-Q_{\mathbb{G}}(t_{j-1})\int H(t_{j})\mathbb{G}(\mathsf{d}\bar{a}(t_{j})|\bar{A}(t_{j-1}))\right\}\right|\\&=\sum_{j=0}^{K}\left|\mathbb{E}\left\{Q_{\mathbb{G}}(t_{j-1})\mathbb{E}_{\mathbb{G}}[H(t_{j})|\mathcal{G}_{t_{j-1}}]\right.\right.\\&\left.\left.-Q_{\mathbb{G}}(t_{j-1})\int H(t_{j})\mathbb{G}\{\mathsf{d}\bar{a}(t_{j})|\bar{A}(t_{j-1})\}\mathbb{P}(\mathsf{d}\bar{l}(t_{j})|\mathcal{G}_{t_{j-1}})\right\}\right|\\&\leq\kappa\sum_{j=0}^{K}\|H(t_{j-1})Q_{\mathbb{G}}(t_{j-1})\|_{1}(t_{j}-t_{j-1})^{\alpha}\\&\leq\kappa\sup_{t}\|H(t)Q_{\mathbb{G}}(t)\|_{1}\sum_{j=0}^{K}(t_{j}-t_{j-1})^{\alpha}\rightarrow0,\end{aligned} $$ 

when  $ |\Delta_{K}[0,\infty]|\to0 $ , where we have used the fact that  $ Q_{\mathbb{G}}(t) $  is a P-martingale and Assumptions 1, 2.

### C.5 Proof of Theorem 4

We first simplify our setting by ignoring censoring and absorbing event time T into  $ \bar{L} $  as well. This is because the conditional independent censoring assumption (Assumption 2) is known to be nonparametric. Our observed data become  $ (\bar{A},\bar{L}) $  and full data become  $ (\bar{A},\bar{L}_{A}) $ .

We first proves that Assumption 1 does not have restrictions on the observed data. We proceed with a constructive proof. For any partition  $ \Delta_{K}[0,\infty] $ , we define a measure on the full data path space. In fact, one has the knowledge on the decomposition

 $$ \begin{align*}\mathbb{P}(\bar{\mathbf{d}\bar{a}}\bar{\mathbf{d}\bar{l}})&=\prod_{j=0}^{K-1}\left\{\mathbb{P}(\bar{\mathbf{d}\bar{a}}(t_{j+1})|\mathcal{F}_{t_{j}})\mathbb{P}(\bar{\mathbf{d}\bar{l}}(t_{j+1})|\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}})\right\}\\&=\prod_{j=0}^{K-1}\left\{\mathbb{P}(\bar{\mathbf{d}\bar{a}}(t_{j+1})|\bar{a}(t_{j}),\bar{l}_{\bar{a}}(t_{j}))\mathbb{P}(\bar{\mathbf{d}\bar{l}_{\bar{a}}}(t_{j+1})|\bar{a}(t_{j+1}),\bar{l}_{\bar{a}}(t_{j}))\right\}.\end{align*} $$ 

Intuitively  $ \mathbb{P}(\mathrm{d}\bar{l}_{\bar{a}}(t_{j+1})|\bar{a}(t_{j+1}),\bar{l}_{\bar{a}}(t_{j})) $  are close to  $ \mathbb{P}(\mathrm{d}\bar{l}_{\bar{a}}(t_{j+1})|\bar{l}_{\bar{a}}(t_{j})) $ , whereas the other term  $ \mathbb{P}(\mathrm{d}\bar{a}(t_{j+1})|\bar{a}(t_{j}),\bar{l}(t_{j})) $  is close to  $ \mathbb{P}(\mathrm{d}\bar{a}(t_{j+1})|\bar{a}(t_{j}),\bar{l}_{\mathcal{A}}) $ . Therefore, one may define a measure on the full data path space by

 $$ \mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\mathbf{d}\bar{l}_{\bar{a}}):=\prod_{j=0}^{K-1}\mathbb{P}(\mathbf{d}\bar{l}_{\bar{a}}(t_{j+1})|\bar{a}(t_{j+1}),\bar{l}_{\bar{a}}(t_{j}))\\ =\prod_{j=0}^{K-1}\mathbb{P}(\mathbf{d}\bar{l}_{\bar{a}}(t_{j+1})|\bar{a}(t_{j+1}),\mathcal{F}_{t_{j}}). $$ 

Then without loss of generality one may construct  $ \mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{l}_{A}) $  by assuming joint independence among  $ \bar{l}_{A} $ . One can also define

 $$ \mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\mathbf{d}\bar{a}|\bar{l}_{\mathcal{A}}):=\prod_{j=0}^{K-1}\mathbb{P}(\mathbf{d}\bar{a}(t_{j+1})|\mathcal{F}_{t_{j}}). $$ 

Then for any sequences of partitions with the mesh going to zero, one may construct a sequence of measures and show that this sequence of measures is Cauchy by a triangular inequality and Assumption 1, following a similar logic as previous proofs. Therefore the sequence converges to a measure  $ P^{F} $ , which is independent of the choice of partitions.

Next we need to show that  $ P^{F} $  induces P on the observed data and  $ P^{F} $  satisfies Assumption 1. The first is trivial because any  $ P_{\Delta_{K}[0,\infty]}^{F} $  induces P on the observed data, then so is their limit. To prove the second, for any time t and  $ \varepsilon > 0 $ , one might smartly choose a partition  $ \Delta_{K}[0,\infty] $  with  $ P_{\Delta_{K}[0,\infty]}^{F} $  close enough to  $ P^{F} $  and  $ t, t + \eta \in \Delta_{K}[0,\infty] $ . This can be done because the convergence point is independent of the choice of partitions. We have

 $$ \begin{aligned}&\mathbb{E}^{F}(\|\mathbb{P}^{F}(\bar{\mathbf{d l}}_{\mathcal{A}}|\mathcal{F}_{t})-\mathbb{P}^{F}[\bar{\mathbf{d l}}_{\mathcal{A}}|\bar{A}(t+\eta),\mathcal{F}_{t}]\|_{\mathrm{T V}})\\&\leq\mathbb{E}^{F}(\|\mathbb{P}^{F}(\bar{\mathbf{d l}}_{\mathcal{A}}|\mathcal{F}_{t})-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d l}}_{\mathcal{A}}|\mathcal{F}_{t})\|_{\mathrm{T V}})\\&+\mathbb{E}^{F}\{\|\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d l}}_{\mathcal{A}}|\mathcal{F}_{t})-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}[\bar{\mathbf{d l}}_{\mathcal{A}}|\bar{A}(t+\eta),\mathcal{F}_{t}]\|_{\mathrm{T V}}\}\\&+\mathbb{E}^{F}\{\|\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}[\bar{\mathbf{d l}}_{\mathcal{A}}|\bar{A}(t+\eta),\mathcal{F}_{t}]-\mathbb{P}^{F}[\bar{\mathbf{d l}}_{\mathcal{A}}|\bar{A}(t+\eta),\mathcal{F}_{t}]\|_{\mathrm{T V}}\}\\&\leq2\|\mathbb{P}^{F}-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}\|\|_{\mathrm{T V}}\\&+\mathbb{E}^{F}\{\|\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d l}}_{\mathcal{A}}|\mathcal{F}_{t})-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}[\bar{\mathbf{d l}}_{\mathcal{A}}|\bar{A}(t+\eta),\mathcal{F}_{t}]\|_{\mathrm{T V}}\}\\&\leq4\|\mathbb{P}^{F}-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}\|\|_{\mathrm{T V}}\\&+\mathbb{E}_{\Delta_{K}[0,\infty]}^{F}\{\|\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d l}}_{\mathcal{A}}|\mathcal{F}_{t})-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}[\bar{\mathbf{d l}}_{\mathcal{A}}|\bar{A}(t+\eta),\mathcal{F}_{t}]\|_{\mathrm{T V}}\}.\end{aligned} $$ 

The first term can be chosen to be sufficiently small. We rewrite the second term

 $$ \begin{aligned}&\mathbb{E}_{\Delta_{K}}^{F}[0,\infty]\{\|\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d}\bar{\mathcal{I}}}|\mathcal{F}_{t})-\mathbb{P}_{\Delta_{K}}^{F}[0,\infty][\bar{\mathbf{d}\bar{\mathcal{I}}}|\bar{A}(t+\eta),\mathcal{F}_{t}]\|_{\mathrm{TV}}\}\\&=\sup_{f:\|f\|_{1}=1}\int f(\bar{l}_{\mathcal{A}})\{\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d}\bar{\mathcal{I}}}|\mathcal{F}_{t})\\&\quad-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}[\bar{\mathbf{d}\bar{\mathcal{I}}}|\bar{a}(t+\eta),\mathcal{F}_{t}]\}\mathbb{P}(\bar{\mathbf{d}\bar{a}}(t+\eta)\mathbf{d}\mathcal{F}_{t})\\&\leq\sup_{f:\|f\|_{1}=1}\sup_{g:\|g\|_{1}=1}\int f(\bar{l}_{\mathcal{A}})g(\bar{a}(t+\eta))[\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}[\bar{\mathbf{d}\bar{\mathcal{I}}}\bar{\mathbf{d}\bar{a}}(t+\eta)|\mathcal{F}_{t}\\&\quad-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d}\bar{\mathcal{I}}}|\mathcal{F}_{t})\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}\{\bar{\mathbf{d}\bar{a}}(t+\eta)|\mathcal{F}_{t}\}]\mathbb{P}(\bar{\mathbf{d}\mathcal{F}_{t}})\\&=\sup_{f:\|f\|_{1}=1}\sup_{g:\|g\|_{1}=1}\int f(\bar{l}_{\mathcal{A}})g(\bar{a}(t+\eta))\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d}\bar{\mathcal{I}}}|\mathcal{F}_{t})\\&\{\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}[\bar{\mathbf{d}\bar{a}}(t+\eta)|\bar{a}(t),\bar{l}_{\mathcal{A}}]-\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}[\bar{\mathbf{d}\bar{a}}(t+\eta)|\mathcal{F}_{t}]\}\mathbb{P}(\bar{\mathbf{d}\mathcal{F}_{t}})\\&=\sup_{f:\|f\|_{1}=1}\sup_{g:\|g\|_{1}=1}\int f(\bar{l}_{\mathcal{A}})g(\bar{a}(t+\eta))\mathbb{P}_{\Delta_{K}[0,\infty]}^{F}(\bar{\mathbf{d}\bar{\mathcal{I}}}|\mathcal{F}_{t})\\&\left[\prod_{j=0}^{K-1}\mathbb{P}(\bar{\mathbf{d}\bar{a}}(t+\eta)|\mathcal{F}_{t})-\mathbb{P}(\mathbb{1})\right]\mathbb{P}(\bar{\mathbf{d}\mathcal{F}_{t}})\\&\leq\mathbb{E}\left\|\prod_{j=0}^{K-1}\mathbb{P}(\bar{\mathbf{d}\bar{a}}(t+\eta)|\mathcal{F}_{t})-\mathbb{P}(\bar{\mathbf{d}\bar{a}}(t+\eta)|\mathcal{F}_{t})\right\|_{\mathrm{TV}}\\&\leq\varepsilon(t,\eta).\end{aligned} $$ 

Assumption 3 is irrelevant here because it is imposed on the stochastic process but not the distributions.

For Assumption 4, a necessary condition for it to hold is that P is well supported on the path space conditioning on any filtration. For this to happen, note that for any P, one can find a well-supported  $ P^{\prime} $  satisfying Assumptions 1, 2, 3, so that  $ (1-\varepsilon)\mathbb{P}+\varepsilon\mathbb{P}^{\prime} $  is well-supported and hence satisfies Assumption 4. Since addition will not break Assumptions 1, 2, 3, we have found  $ (1-\varepsilon)\mathbb{P}+\varepsilon\mathbb{P}^{\prime} $  satisfying Assumptions 1, 2, 3, and 4, and approximates P.

## D ADDITIONAL EXPERIMENT RESULT

In this section, we employ Monte Carlo simulations to empirically assess how the identification works. To that end, we need to go through 4 steps:

1. Come up with a reasonable data generating process;

2. Compute the parameter of interest (1) (or equivalently, the left-hand side of g-computation formula in Theorem 1) according to this data generating process;

3. Simulate according to this data generating process;

4. Approximate the right-hand side of g-computation formula in Theorem 1 using the simulated data.

Step 1: For  $ t \in [0,1] $ , consider a potential outcome process  $ Y_{\bar{a}}(t) $  and potential covariate  $ L_{\bar{a}}(t) $  following a Gaussian process with mean process as

 $$ \begin{pmatrix}\mathbb{E}\{Y_{\bar{a}}(t)\}\\\mathbb{E}\{L_{\bar{a}}(t)\}\end{pmatrix}=\begin{pmatrix}-a(t)\\0.5a(t)\end{pmatrix}, $$ 

and covariance process

 $$ \begin{pmatrix}\mathrm{Cov}\{Y_{\bar{a}}(t),Y_{\bar{a}}(s)\}&\mathrm{Cov}\{Y_{\bar{a}}(t),L_{\bar{a}}(s)\}\\\mathrm{Cov}\{L_{\bar{a}}(t),Y_{\bar{a}}(s)\}&\mathrm{Cov}\{L_{\bar{a}}(t),L_{\bar{a}}(s)\}\end{pmatrix}=\begin{pmatrix}e^{-3|t-s|}/2&e^{-1-6|t-s|}/2\\e^{-1-6|t-s|}/2&e^{-3|t-s|}/2\end{pmatrix}. $$ 

This ensures the joint dependence among  $ Y_{\bar{a}}(t) $  and  $ L_{\bar{a}}(t) $  and non-zero treatment effect of  $ \bar{a} $ . Generate the event time following a Cox model

 $$ \mathbb{P}(T_{\bar{a}}>t|\bar{Y}_{\bar{a}},\bar{L}_{\bar{a}})=\exp\left\{-t\exp\left[0.2a(t)-0.8Y_{\bar{a}}(t)-2L_{\bar{a}}(t)\right]\right\}. $$ 

and an independent censoring process

 $$ \mathbb{P}(C>t)=\begin{cases}\exp(-t),&\text{when}t\leq1\\ 0,&\text{when}t>1,\end{cases} $$ 

where time 1 represents an administrative censoring. Define  $ \nu(T_{\bar{a}},\bar{Y}_{\bar{a}}) $  as the integral of  $ \bar{Y}_{\bar{a}} $  over time  $ t\in[0,1] $ , that is,

 $$ \nu(T_{\bar{a}},\bar{Y}_{\bar{a}})=\int_{0}^{T_{\bar{a}}\wedge1}Y_{\bar{a}}(t)d t. $$ 

where  $ T_{\bar{a}} $  is capped at 1 because beyond that all subjects are censored. Suppose the targeted treatment regime G is a Gaussian measure with mean zero process and jointly independent normal variables at any time points. That is, the intervened A follows a Gaussian process with a mean process

 $$ \mathbb{E}(A(t))=0, $$ 

and covariance process

 $$ \mathrm{Cov}\{A(t),A(s)\}=e^{-3|t-s|}/2,\;\forall t,s\in[0,1]. $$ 

Step : Then we can show that (1) (or equivalently, the left-hand side of g-computation formula in Theorem 1) equals zero, that is,

 $$ \int\mathbb{E}(\nu(T_{\bar{a}},\bar{Y}_{\bar{a}}))\mathbb{G}(\mathbf{d}\bar{a})=\int\mathbb{E}\left[\int_{0}^{T_{\bar{a}}\wedge1}Y_{\bar{a}}(t)d t\right]\mathbb{G}(\bar{a}) $$ 

 $$ =\int\mathbb{E}\left[\int_{0}^{\mathbb{E}(T_{\bar{0}})\wedge1}Y_{\bar{a}}(t)d t\right]\mathbb{G}(\bar{a}) $$ 

 $$ =\int\int_{0}^{\mathbb{E}(T_{\bar{0}})\wedge1}\mathbb{E}(\bar{Y}_{\bar{a}}(t))d t\mathbb{G}(\bar{a})=\int\int_{0}^{\mathbb{E}(T_{\bar{0}})}a(t)d t\mathbb{G}(\bar{a}) $$ 

 $$ =\int_{0}^{\mathbb{E}(T_{\bar{0}})\wedge1}\int a(t)\mathbb{G}(\bar{a}(t))d t=\int_{0}^{1}0d t=0. $$ 

We used the fact that, through our data generating process, the treatment does not have effect on survival time.

Step 3: In practice, we observe a stochastic process at finite points. We consider evenly splitting  $ t \in [0,1] $  into a grid of size  $ K+1 $ :  $ \Delta_{K}[0,1] = \{t_{0} = 0, t_{1} = 1/K, \cdots, t_{K-1} = (K-1)/K, t_{K} = 1\} $ , and for  $ 1 \leq i \leq n $ , according to G specified in Step 1, we simulate i.i.d. samples  $ A_{i}(t) $  according to G specified in Step 1 at  $ \Delta_{K}[0,1] $  as

 $$ \begin{pmatrix}A_{i}(t_{0})\\A_{i}(t_{1})\\\cdots\\A_{i}(t_{K-1})\\A_{i}(t_{K})\end{pmatrix}\sim\mathcal{N}\left[\begin{pmatrix}t_{0}-0.5\\t_{1}-0.5\\\cdots\\t_{K-1}-0.5\\t_{K}-0.5\end{pmatrix},\quad\begin{pmatrix}1&e^{-3|t_{1}-t_{0}|}&\cdots&e^{-3|t_{K-1}-t_{0}|}&e^{-3|t_{K}-t_{0}|}\\e^{-|t_{1}-t_{0}|}&1&\cdots&e^{-3|t_{K-1}-t_{1}|}&e^{-3|t_{K}-t_{1}|}\\\cdots&\cdots&\cdots&\cdots&\cdots\\e^{-3|t_{K-1}-t_{0}|}&e^{-3|t_{K-1}-t_{1}|}&\cdots&1&e^{-3|t_{K}-t_{K-1}|}\\e^{-3|t_{K}-t_{0}|}&e^{-|t_{K}-t_{1}|}&\cdots&e^{-3|t_{K}-t_{K-1}|}&1\end{pmatrix}\right]. $$ 

By according to the distribution of  $ Y_{\bar{a}}(t) $  specified in Step 1 and consistency, we generate  $ Y_{i}(t) $  and  $ L_{i}(t) $  following a similar manner. Generate the event time following a Cox model

 $$ \mathbb{P}(T_{i}>t|\bar{A}_{i},\bar{Y}_{i},\bar{L}_{i})=\exp\left\{-t\exp\left[0.2A(t)-0.8Y_{i}(t)-2L_{i}(t)\right]\right\}. $$ 

and an independent censoring process

 $$ \mathbb{P}(C_{i}>t)=\begin{cases}\exp(-t),&\text{when}t\leq1\\ 0,&\text{when}t>1,\end{cases} $$ 

Step 4: The integral of  $ Y_{i}(t_{k}) $  over [0,1] is  $ \sum_{0<k<T_{i}} Y_{i}(t_{k})/(K+1) $ . The approximate of the right-hand side of g-computation formula is  $ \sum_{i=1}^{n} \sum_{0<k<T_{i}} Y_{i}(t_{k})/(K+1)/n $ .

We vary the grid sizes  $ (K=10,50,250) $  to examine how a denser grid improves the approximation. This approach simulates the scenario where the mesh  $ |\Delta_{K}[0,1]] $  is shrunk to zero. Additionally, we vary the sample sizes  $ (n=100,500,2500) $  to explore how larger samples enhance the approximation, leveraging the law of large numbers to better approximate the right-hand side of the g-computation formula. We repeat the process R=10,000 times. The resulting 10,000 approximations of  $ \sum_{i=1}^{n}\sum_{k=0}^{K}Y_{i}(t_{k})/(K+1)/n $  are presented in boxplots in Figure D, where we append biases.

The simulation results demonstrate that the g-computation formula can adequately approximate (1) even with moderate sample and grid sizes. The biases are larger than that in Section 4 possibly because we are focusing on a more difficult task. This time, however, increasing the sample size while keeping the grid size fixed does not enhance the accuracy. Increasing the grid size while keeping the sample size fixed does improve accuracy or reduce variance in this case. Again, simultaneously increasing both the sample and grid sizes significantly improves accuracy and reduces variance in the approximation.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_266_501_954_1022.jpg" alt="Image" width="56%" /></div>


<div style="text-align: center;">Figure 2: Simulation Results of using g-computation formula by varying grid sizes in K = 10, 50, 250 and sample sizes in n = 100, 500, 2500, for R = 10000 repeats. We plot boxplots and give biases.</div>