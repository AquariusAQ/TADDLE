## A BACKGROUND ON KERNELS AND MMD

In this section, we provide a brief overview of kernel methods, leading towards the maximum mean discrepancy. For further background, we refer the reader to (Muandet et al., 2017).

Kernels and Feature Maps Suppose X is a topological space on which we wish to study either functions  $ f : X \to R $  or measures  $ \mu \in \mathcal{P}(\mathcal{X}) $ . A kcommon way to consider such objects is by using a feature map

 $$ \Phi:\mathcal{X}\rightarrow\mathcal{H} $$ 

into some Hilbert space H. Heuristically, we can consider

• functions on X via linear functionals  $ \langle\ell,\Phi(\cdot)\rangle_{\mathcal{H}}:\mathcal{X}\to\mathbb{R} $  where  $ \ell\inH $ , and

• measures on X by considering the kernel mean embedding (which we also denote by  $ \Phi $ ), defined by

 $$ \Phi:\mathcal{P}(\mathcal{X})\to\mathcal{H},\quad\Phi(\mu)=\int_{\mathcal{X}}\Phi(x)d\mu(x). $$ 

Given this feature map, we can define a positive-definite kernel  $ k: X \times X \to R $  defined by

 $$ k(x,y):=\langle\Phi(x),\Phi(y)\rangle_{\mathcal{H}}. $$ 

Reproducing Kernel Hilbert Spaces. In fact, we can also go in the other direction and start with a continuous positive definite kernel  $ k : X \times X \to R $ , and obtain a feature map from X into a Hilbert space of functions. In particular, we define H to be the completion of the linear span of functions  $ \{k(x,\cdot) : X \to \mathbb{R} : x \in \mathcal{X}\} $ , equipped with the inner product

 $$ \langle k(x,\cdot),k(y,\cdot)\rangle:=k(x,y). $$ 

By the Moore-Aronszajn theorem (Aronszajn, 1950), H is a Hilbert space with the reproducing kernel property: for any  $ f \in H $  and  $ x \in X $ , we have

 $$ \langle f,k(x,\cdot)\rangle=f(x). $$ 

Thus, H is a reproducing kernel Hilbert space (RKHS). Note that this is a Hilbert space of functions,  $ \mathcal{H} \subset C(\mathcal{X}, \mathbb{R}) $ . Then, we can define a feature map  $ \Phi : X \to H $  by

 $$ \Phi(x):=k(x,\cdot). $$ 

Universality and Characteristicness We wish to consider feature maps (or kernels) which satisfy additional properties such that they can approximate functions and characterize measures. Let  $ \mathcal{F}\subset C(\mathcal{X},\mathbb{R}) $  be a topological vector space, and suppose M is a space of measures on X. A feature map  $ \Phi:X\toH $  (with associated kernel  $ k:X\timesX\toR $ ), where  $ H\subsetF $  is (Simon-Gabriel & Schölkopf, 2018)

• universal with respect to F if H is dense in F (we can approximate functions in F using functions in H); and

• characteristic with respect to M if the kernel mean embedding in Equation (18) is injective.

Maximum Mean Discrepancy Given a feature map  $ \Phi : X \to H $  (with kernel k) characteristic to the space of probability measures  $ \mathcal{P}(\mathcal{X}) $ , we can use the Hilbert space norm to define a metric on this space of measures. In fact, this is equivalent to the notion of maximum mean discrepancy (MMD) from statistics. In particular, given a function class  $ \mathcal{F} \subset C(\mathcal{X}, \mathbb{R}) $ , we define the MMD with respect to F by

 $$ \mathbf{M M D}_{\mathcal{F}}(\mu,\nu):=\sup_{f\in\mathcal{F}}\left(\mathbb{E}_{x\sim\mu}[f(x)]-\mathbb{E}_{y\sim\nu}[f(y)]\right). $$ 

Now, by (Gretton et al., 2012, Lemma 4), if we choose F to be the unit ball of the RKHS H (with respect to a characteristic kernel k), the MMD with respect to F is exactly the Hilbert space norm,

 $$ \mathbf{M M D}_{\mathcal{F}}(\mu,\nu)=\left\|\Phi(\mu)-\Phi(\nu)\right\|_{\mathcal{H}}=:\mathbf{M M D}_{k}(\mu,\nu), $$ 

where the right hand side is how we define  $ MMD_{k} $  in Equation (11).

## B CHARACTERISTIC kernels on  $ \Omega $ 

In this appendix, we provide a detailed discussion of the proof of Theorem 1.

We begin by characterizing some of the elements in the RKHS HΩ.

Lemma 1. The RKHS  $ H_{\Omega} $  satisfies

 $$ \mathcal{H}_{\Omega}\supset\{g:\Omega\to\mathbb{R}:g(b,\ell)=\ell\cdot f(b,\ell),f\in\mathcal{H}\}. $$ 

Proof. By Moore-Aronszajn (Aronszajn, 1950), an element of  $ g \in H_{\Omega} $  is defined by a convergent series

 $$ g(b,\ell)=\sum_{i=1}^{\infty}c_{i}k_{\Omega}((b_{i},\ell_{i}),(b,\ell))=\ell\sum_{i=1}^{\infty}c_{i}\ell_{i}k((b_{i},\ell_{i}),(b,\ell))=\ell\cdot f(b,\ell), $$ 

where  $ f(b,\ell) = \sum_{i=1}^{\infty} c_{i} \ell_{i} k((b_{i},\ell_{i}),(b,\ell)) $ . Note that if the coefficient series for f given by  $ \sum_{i=1}^{\infty} |c_{i}| \ell_{i} k((b_{i},\ell_{i}),(b_{i},\ell_{i})) $  is convergent, then the coefficient series for g given by

 $$ \sum_{i=1}^{\infty}|c_{i}|k_{\Omega}((b_{i},\ell_{i}),(b_{i},\ell_{i}))=\sum_{i=1}^{\infty}|c_{i}|\ell_{i}^{2}k((b_{i},\ell_{i}),(b_{i},\ell_{i}))\leq T\sum_{i=1}^{\infty}|c_{i}|\ell_{i} k((b_{i},\ell_{i}),(b_{i},\ell_{i})) $$ 

is also convergent, since  $ \ell_{i} \leq T $ .

Universality and Characteristicness. Our first main result concerns the transfer of universal and characteristic properties from k to  $ k_{\Omega} $ . First, we define the space of linear-growth continuous functions on  $ \Omega_{T} $  by

 $$ C_{\mathrm{l i n}}(\Omega_{T}):=\{\ell\cdot f(b,\ell)\in C(\Omega_{T}):f(b,\ell)\in C([0,T]^{2})\}, $$ 

where we equip it with the norm defined on  $ g = \ell \cdot f $  by

 $$ \|g\|_{\mathrm{l i n}}:=|f|_{\infty}. $$ 

Note that for all  $  g \in C_{\operatorname{lin}}(\Omega)  $  have the property that  $  g(*) = 0  $ , and  $  (C_{\operatorname{lin}}(\Omega_T), \|\cdot\|_{\operatorname{lin}})  $  and  $  (C([0,T]^2), |\cdot|_{\infty})  $  are isometric Banach spaces.

Theorem 4. Let $k:[0,T]^{2}\times[0,T]^{2}\to\mathbb{R}$ be a kernel on $[0,T]^{2}$ universal to $C([0,T]^{2})$. Then, $k_{\Omega}:\Omega\times\Omega\to\mathbb{R}$ is universal with respect to $C_{lin}(\Omega_{T})$.

Proof. Let  $  g \in C_{\operatorname{lin}}(\Omega_T)  $  and suppose  $ g = \ell \cdot f $  for some  $  f \in C([0, T]^2)  $ . Because k is universal, there exists  $ f_n \in H $  such that  $ |f_n - f|_{\infty} \to 0 $ . Then, by Lemma 1,  $ g_n = \ell \cdot f_n \in H_\Omega $ , and furthermore, by the definition of  $ \|\cdot\|_{\operatorname{lin}} $  in Equation (29), we have  $ \|g_n - g\|_{\operatorname{lin}} \to 0 $ . Thus,  $ k_\Omega $  is universal with respect to  $  C_{\operatorname{lin}}(\Omega_T)  $ .  $ \square $ 

The main result we wish to obtain is characteristicness with respect to measures on  $ \Omega $ . We begin by applying the duality theorem of (Simon-Gabriel & Schölkopf, 2018, Theorem 6) which immediately implies characteristicness with respect to the topological dual of  $ C_{\mathrm{lin}}(\Omega) $ , which we equip with the weak- $ ^{*} $  topology with respect to  $ C_{\mathrm{lin}}(\Omega) $ .

Corollary 1. Let $k:[0,T]^{2}\times[0,T]^{2}\to\mathbb{R}$ be a kernel on $[0,T]^{2}$ universal to $C([0,T]^{2})$. Then, $k_{\Omega}$ is characteristic with respect to $C_{lin}(\Omega)^{*}$.

Our next task is to show that our desired measures are contained in this dual space.

Theorem 5. Let  $ q:[0,T]^{2}\to\Omega $  denote the quotient map, and let  $ s:[0,T]^{2}\to[0,\infty) $  be defined by  $ s(b,\ell)=\ell $  for  $ \ell>0 $  and  $ s(b,0)=0 $ . Define

 $$ \mathcal{M}_{l i n}(\Omega):=\left\{q_{*}\mu\in\mathcal{M}(\Omega):\mu\in\mathcal{M}([0,T]^{2}),\mu(\{\ell=0\})=0,\left|\int_{[0,T]^{2}}\ell d\mu\right|<\infty\right\}, $$ 

then  $ \mathcal{M}_{lin}(\Omega) \subset C_{lin}(\Omega)^{*} $ 

Proof. Let  $ q_{*}\mu \in \mathcal{M}_{\mathrm{lin}}(\Omega) $ . Then, for  $ g = \ell \cdot f \in C_{\mathrm{lin}}(\Omega) $ , we have

 $$ \left|\int_{\Omega}\ell\cdot f(b,\ell)q_{*}d\nu\right|=\left|\int_{[0,T]^{2}}\ell\cdot f(b,\ell)d\mu\right|\leq\|g\|_{\mathrm{l i n}}\left|\int_{[0,T]^{2}}\ell d\mu\right|, $$ 

where we use  $ \mu(\{\ell=0\})=0 $  in the first equality. Thus,  $ q_{*}\mu $  is a bounded (hence continuous) linear functional.

As we wish to use this kernel to study PPMs, we are interested in probability measures on  $ \Omega $ . However, the elements in the dual only contain measures which are trivial on  $ * \in \Omega $  (whereas PPMs may have nontrivial mass on  $ * $ ), and thus we first define a different representation of probability measures. In particular, we define

 $$ \mathcal{P}_{\mathrm{l i n}}(\Omega):=\{\nu\in\mathcal{M}_{\mathrm{l i n}}(\Omega):|\nu|\leq1\}=\{\nu\in\mathcal{M}(\Omega):\nu(\{*\})=0,|\nu|\leq1\}\subset\mathcal{M}_{\mathrm{l i n}}(\Omega). $$ 

Note that the equality holds since the moment condition in Equation (30) is immediately satisfied since the measures are finite with bounded support. With the following two lemmas, we show that this coincides with  $ \mathcal{P}(\Omega) $ .

Lemma 2. The space  $ C_{\mathrm{lin}}(\Omega) $  is dense in  $ C_{0}(\Omega) $  equipped with the uniform topology.

Proof. Let  $  g \in C_{0}(\Omega)  $ , and since g is uniformly continuous (since  $ \Omega $  is compact) and  $  g(*) = 0  $ , for every  $ \epsilon > 0 $ , there exists some  $ \ell_{\epsilon} $  such that  $  g(b, \ell) < \epsilon  $  whenever  $ \ell < \ell_{\epsilon} $ . Now, define  $  f_{n} \in C([0, T]^{2})  $  by

 $$ f_{n}(b,\ell)=\frac{g(b,\ell)}{\ell}\quad\mathrm{f o r}\quad\ell\geq\ell_{1/n}\quad\mathrm{a n d}\quad f_{n}(b,\ell)=\frac{g(b,\ell_{1/n})}{\ell_{1/n}}\quad\mathrm{f o r}\quad\ell<\ell_{1/n}. $$ 

Then, define  $ g_{n}(b,\ell)=\ell\cdot f(b,\ell) $ , where  $ g_{n}(b,\ell)=g(b,\ell) $  whenever  $ \ell\geq\ell_{1/n} $ . When  $ \ell<\ell_{1/n} $ , we have

 $$ |g_{n}(b,\ell)-g(b,\ell)|=|g_{n}(b,\ell_{1/n}-g(b,\ell)|\leq\frac{2}{n}, $$ 

so $g_{n}$ converges uniformly to $g$.

Lemma 3. There exists a homeomorphism

 $$ \psi:\mathcal{P}(\Omega)\to\mathcal{P}_{l i n}(\Omega) $$ 

where  $ \mathcal{P}(\Omega) $  is equipped with the weak- $ ^{*} $  topology with respect to  $ C(\Omega) $  and  $ \mathcal{P}_{lin}(\Omega) $  is equipped with the weak- $ ^{*} $  topology with respect to  $ C_{lin}(\Omega) $ .

Proof. We define  $ \psi $  and its inverse by

 $$ \psi(\mu):=\mu-\mu(*)\delta_{*}\quad\mathrm{a n d}\quad\psi^{-1}(\nu)=\nu+(1-\nu(\Omega))\delta_{*}, $$ 

where $\delta_{*}$ denotes the Dirac measure on $\ast\in\Omega$. By definition of $\mathcal{P}_{\mathrm{lin}}(\Omega)$ in Equation (32), this map is a bijection. It remains to show that $\mu_{n}\to\mu$ in $\mathcal{P}(\Omega)$ if and only if $\psi(\mu_{n})\to\psi(\mu)$ in $\mathcal{P}_{\mathrm{lin}}(\Omega)$.

Note that for any  $  f \in C(\Omega)  $  and  $  \mu \in \mathcal{P}(\Omega)  $ , we can decompose the integral as

 $$ \mu(f)=\int_{\Omega}f d\mu=\int_{\Omega-\{*\}}f d\mu+f(*)\mu(\{*\}). $$ 

Then, for  $ f \in C_{\mathrm{lin}}(\Omega) $ , we have

 $$ \mu(f)=\int_{\Omega-\{*\}}f d\mu=\int_{\Omega-\{*\}}f d\psi(\mu) $$ 

since  $ f(*)=0 $  and  $ \mu=\psi(\mu) $  on  $ \Omega-\{*\} $ . Thus, if  $ \mu_{n}\to\mu $  in  $ \mathcal{P}(\Omega) $  then  $ \psi(\mu_{n})\to\psi(\mu) $  in  $ \mathcal{P}_{\mathrm{lin}}(\Omega) $ .

Next, suppose that  $ \nu_{n}\to\nu $  in  $ \mathcal{P}_{\mathrm{lin}}(\Omega) $ . Now, we note that  $ \mathcal{P}_{\mathrm{lin}}(\Omega)\subset\mathcal{M}(\Omega) $ , and thus are also continuous functionals on  $ C(\Omega) $  with respect to the uniform topology. By Lemma 2, we have  $ \nu_{n}(f)\to\nu(f) $  for all  $ f\in C_{0}(\Omega) $ . However, this also implies it holds for all  $ f\in C(\Omega) $ , since

we obtain functions in  $ C(\Omega) $  by adding a constant to a function in  $ C_{0}(\Omega) $  and  $ \nu(\{*\}) = 0 $  for all  $ \nu \in \mathcal{P}_{\mathrm{lin}}(\Omega) $ . This implies that

 $$ \int_{\Omega-\{*\}}f d\psi^{-1}(\nu_{n})\rightarrow\int_{\Omega-\{*\}}f d\psi^{-1}(\nu) $$ 

since $\nu=\psi^{-1}(\nu)$ on $\Omega-\{*\}$ for all $\nu\in\mathcal{P}_{\mathrm{lin}}(\Omega)$. Furthermore, this implies that $\nu_{n}(\Omega)\to\nu(\Omega)$, and thus by definition of $\psi^{-1}$ in Equation (36), we have $\psi^{-1}(\nu_{n})\to\psi^{-1}(\nu)$ in $\mathcal{P}(\Omega)$.

This result allows us to work with  $ \mathcal{P}(\Omega) $  and  $ \mathcal{P}_{\mathrm{lin}}(\Omega) $  interchangeably. In particular, note that  $ \Phi_{\Omega}(*) = 0 \in \mathcal{H}_{\Omega} $ . This implies that for any  $ \mu \in \mathcal{P}(\Omega) $ , we have

 $$ \Phi_{\Omega}(\mu)=\Phi_{\Omega}(\psi(\mu)). $$ 

Proof of Theorem 1. By Corollary 1,  $ k_{\Omega} $  is characteristic with respect to  $ \mathcal{M}_{\mathrm{lin}}(\Omega) $ , and  $ \mathcal{P}_{\mathrm{lin}}(\Omega) \subset \mathcal{M}_{\mathrm{lin}}(\Omega) $  by Theorem 5. Next, by Lemma 3, we can apply the identity Equation (40) to conclude that  $ k_{\Omega} $  is characteristic with respect to  $ \mathcal{P}(\Omega) $ .

## C METRIZING WEAK TOPOLOGY

In this appendix, we provide a detailed discussion of the proof of Theorem 2.

Maximum Mean Discrepancy (MMD). Because  $ k_{\Omega} $  is a characteristic kernel on  $ \Omega $ , the kernel mean embedding, defined (by an abuse of notation) on  $ \nu = q_{*} \mu \in \mathcal{M}_{\operatorname{lin}}(\Omega) $  by

 $$ \Phi_{\Omega}:\mathcal{M}_{\mathrm{l i n}}(\Omega)\to\mathcal{H}_{\Omega},\quad\Phi(\nu):=\mathbb{E}_{z\sim\nu}[\Phi_{\Omega}(z)], $$ 

is injective. This induces a metric on  $ \mathcal{M}_{\mathrm{lin}}(\Omega) $ , called the maximum mean discrepancy (MMD),

 $$ \mathbf{M M D}_{k}\big(\nu_{1},\nu_{2}\big):=\big\|\Phi\big(\nu_{1}\big)-\Phi\big(\nu_{2}\big)\big\|_{\mathcal{H}_{\Omega}}. $$ 

Our next result shows that the MMD metrizes the weak- $ ^{*} $  topology on  $ \mathcal{P}_{\mathrm{lin}}(\Omega) $ , where we can directly apply (Sriperumbudur, 2016, Theorem 3.2). See also (Simon-Gabriel et al., 2023) for related results.

Theorem 6. The MMD metric metrizes the weak topology on  $ \mathcal{P}(\Omega) $ . In other words, given measures  $ \nu_{n}, \nu \in \mathcal{P}(\Omega) $ , we have  $ MMD_{k}(\nu_{n}, \nu) \to 0 $  if and only if

 $$ \left|\nu_{n}(g)-\nu(g)\right|\rightarrow0 $$ 

for all g ∈ C(Ω).

Proof. This follows from (Sriperumbudur, 2016, Theorem 3.2) as  $ \Omega $  is a compact Polish space and  $ k_{\Omega} $  is a continuous bounded kernel.

Corollary 2. The p-Wasserstein metric  $ W_{p} $  and the MMD metric  $ MMD_{k} $  induce the same topology on  $ \mathcal{P}(\Omega) $ .

Proof. This is a direct consequence of the above, since the p-Wasserstein distance also metrizes the weak topology on  $ \mathcal{P}(\Omega) $  (Villani, 2009, Theorem 6.9). □

## D PPM REGULARIZER HAS CONTINUOUS GRADIENTS

We say that a function  $ f : R^{n} \to R^{m} $  is a  $ C^{1} $  function if all first derivatives of f are continuous. We will begin with a more general statement which will immediately imply Theorem 3. Fix  $ \mu \in \mathcal{P}_{c}(\mathbb{R}^{N}) $  to be a measure, and let  $ h_{\theta} : R^{N} \to R^{L} $  be a mapping from  $ R^{N} $  to the latent space  $ R^{L} $ . Suppose that the mapping  $ h_{\theta} $  is parametrized by  $ \theta \in R^{P} $ , and define  $ H : R^{P} \times R^{N} \to R^{L} $  by  $ H(\theta, x) = h_{\theta}(x) $ . We aim to show that  $ H : R^{P} \to H_{\Omega} $ , defined by

 $$ \mathfrak{H}(\theta):=\Phi_{\Omega}(\mathbf{P P M}_{q}(h_{\theta}(\mu))) $$ 

is a smooth function. In particular, the pipeline for computing this feature map is

 $$ \mathcal{P}_{c}(\mathbb{R}^{N})\xrightarrow{(h_{\theta})_{*}}\mathcal{P}_{c}(\mathbb{R}^{L})\xrightarrow{(-)^{\otimes(2k+2)}}\mathcal{P}_{c}((\mathbb{R}^{L})^{2k+2})\xrightarrow{(\mathrm{P H}_{k})_{*}}\mathcal{P}(\Omega)\xrightarrow{\Phi}\mathcal{H}, $$ 

Let  $  F : \mathbb{R}^{P} \to \mathcal{P} \left( (\mathbb{R}^{L})^{2k+2} \right)  $  be the map from the parameter space to the product measure in latent space. We assume that the resulting product measure has a  $ C^{1} $  density, which is true if H is  $ C^{1} $ , and  $ \mu $  has a  $ C^{1} $  density. Then, F has the form

 $$ F(\theta)=f(\theta,x)dx $$ 

where  $  f : \mathbb{R}^{p} \times (\mathbb{R}^{L})^{2k+2} \to \mathbb{R}  $  is a  $ C^{r} $  function.

Theorem 7. Let  $ \mu\in\mathcal{P}(\mathbb{R}^{N}) $  be a probability measure with a  $ C^{1} $  density, suppose  $ H:R^{P}\timesR^{N}\toR^{L} $  is a  $ C^{1} $  function. Then  $ \mathfrak{H}:R^{P}\toH_{\Omega} $  is a  $ C^{1} $  function (where derivatives are Fréchet derivatives).

Proof. Expanding out the definition of H, we have

 $$ \Phi_{\Omega}(\mathbf{P P M}_{q}(h_{\theta}(\mu)))=\int_{\Omega}\Phi_{\Omega}(z)d(\mathbf{P H}_{k})_{*}F(\theta)(z) $$ 

 $$ =\int_{(\mathbb{R}^{L})^{2k+2}}\Phi_{\Omega}\circ\mathbf{P H}_{k}(x)d F(\theta)(x) $$ 

 $$ =\int_{(\mathbb{R}^{L})^{2k+2}}\Phi_{\Omega}\circ\mathbf{P H}_{k}(x)f(\theta,x)d x, $$ 

where we use the definition of the pushforward in the second line, and the definition of the density of F in Equation (46). Now, this integral is a Bochner integral since it is valued in a Hilbert space, and we can still differentiate under the integral (by Hille's theorem; see (Dieudonne, 1969, Paragraph 8.11.2)). Then, we have

 $$ \frac{\partial}{\partial\theta_{i}}\Phi_{\Omega}(\mathbf{P P M}_{q}(h_{\theta}(\mu)))=\int_{(\mathbb{R}^{L})^{2k+2}}\Phi\circ\mathbf{P H}_{k}(x)\frac{\partial f(\theta,x)}{\partial\theta_{i}}d x, $$ 

which is continuous since f is  $ C^{1} $ 

Proof of Theorem 3. By direct application of Theorem 7, both

 $$ \boldsymbol{\theta}\mapsto\Phi_{\Omega}\circ\mathrm{PPM}_{q}\circ d_{\boldsymbol{\theta}}(\mu)\quad\mathrm{a n d}\quad(\boldsymbol{\theta},\boldsymbol{\omega})\mapsto\Phi_{\Omega}\circ\mathrm{PPM}_{q}\circ d_{\boldsymbol{\theta}}\circ g_{\boldsymbol{\omega}}(\nu) $$ 

are  $ C^{1} $  functions. Then, since the norm is continuously differentiable away from the origin, we obtain the desired result.

## E SHAPE MATCHING

### E.1 IMPLEMENTATION DETAILS FOR SHAPE MATCHING

Shape Matching Experiment. The task is to match the "shape" of two point clouds by optimizing a loss function of the form  $ L + T $  in ambient space. Throughout the experiment, we use gradient descent with momentum as the optimization algorithm. The value of the momentum parameter is 0.9 and the step size is 0.05. Two simple reference shapes in  $ R^{2} $  are chosen for visualization purposes. They are a unit circle (left), and the union of two intersecting unit circles (right), shown in Figure 2. A noisy point cloud with 512 points is first initialized with a normally distributed at the origin with a standard deviation of 0.3, we call training shape. The reference shapes are also sampled as a point cloud wiht 512 points. The reference shape is fixed and the training shape changes towards the reference, guided by some loss function. We test for the effectiveness of four loss functions, they are Cramer, MMD, Cramer + PPM-Reg and MMD + PPM-Reg. The MMD metric using an RBF kernel (with width  $ \sigma = 0.1 $ ). Throughout the experiment, the hyperparameter of PPM-Reg is fixed as  $ \lambda = 1 $ ,  $ \lambda_{0} = 1 $ ,  $ \lambda_{1} = 6000 $ ,  $ \sigma = 0.1 $  and s = 2000. For Cramer + PPM-Reg, the weight of the cramer loss is 1.6. For MMD + PPM-Reg, the weight of the MMD loss is 5. In Figure 4, we show the shapes after 16000 training steps. Even at this stage, we observe that there are several "leftover" points. This is partially due to the choice of the underlying loss functions, and we observe that in (b), (d) and (h), the trained shapes using PPM-Reg largely capture the topological features of the reference shape.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_220_172_401_353.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_417_171_600_353.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_615_172_797_353.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(c)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_812_172_995_353.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(d)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_219_398_401_580.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(e)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_417_398_600_581.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(f)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_614_399_797_581.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(g)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_812_399_994_581.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(h)</div>


<div style="text-align: center;">Figure 4: Plots of the shape matching experiment at convergence after 16000 steps. (a) and (e) use only the Cramer loss. (b) and (f) use Cramer + PPM-Reg. (c) and (g) use only the MMD loss. (d) and (h) use MMD + PPM-Reg.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_233_735_608_1015.jpg" alt="Image" width="30%" /></div>


<div style="text-align: center;">(a) Unit circle</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_616_738_989_1014.jpg" alt="Image" width="30%" /></div>


<div style="text-align: center;">(b) Union of two intersecting unit circles</div>


<div style="text-align: center;">Figure 5: Illustrative example of the PPM-Reg in shape matching using MMD with increasing handicap contain. Plotting 2-Wasserstein distance up to 1-dimensional persistent homology between the reference shape and training shape  $ (PD_{dist}) $  over optimization steps. The reference shape of (a) unit circle and (b) union of two intersecting unit circles.  $ c_{\delta} $  indicate the strength of the handicap (detail provided in Appendix E.2). Showing that the ability to match topological features as the handicap contains increase.</div>


Computational Comparison. Results are computed on Nvidia Geforce RTX 3060 with Intel Core i7-10700.

### E.2 IMPERFECT CONVERGENCE

This section demonstrates the efficacy of PPM-Reg when the primary loss function L is optimized imperfectly. As discussed in Section 1, most GAN training algorithms converge to a local saddle point. In simpler latent-space matching tasks, L is often optimized alongside with other tasks. To summarize, converging to an imperfect L value is a common occurrence in machine learning.

To restrict the solution of the shape matching problem, we impose a penalty term that prevents the centroid of the reference shapes and the training shape from being smaller than a user-defined value  $ c_{\delta} $ . The penalty term  $ f_{p} $  is given by

 $$ f_{p}=\frac{\lambda_{p}}{\beta}l n(1+e^{\beta(c_{\delta}-\|c_{t r a i n}-c_{r e f}\|_{2})}), $$ 

where  $ c_{train} $  and  $ c_{ref} $  refers to the centroid of the training shape and reference shapes respectively. The term  $ \lambda_{p} $  is the penalty strength,  $ \beta $  is a tuning parameter. For the MMD case,  $ V = L + f_{p} $ . For the MMD + PPM-Reg case,  $ V = L + \lambda T + f_{p} $ .

Implementation Details. We largely follow the setup on Section 6.2 and Appendix E.1, but make a few minor changes. We reduced the step size to 0.01 and performed 16,000 gradient steps. The hyperparameter of PPM-Reg is fixed as  $ \lambda = 1 $ ,  $ \lambda_{0} = 0.3 $ ,  $ \lambda_{1} = 6000 $ ,  $ \sigma = 0.1 $  and s = 2000. For the penalty function  $ f_{p} $ , we set  $ \beta = 80 $ . The  $ \lambda_{p} $  value remains the same when we vary  $ c_{\delta} $  and compare against adding PPM-Reg. The  $ \lambda_{p} $  value is determined such that  $ \|c_{train} - c_{ref}\|_{2} $  converges to  $ c_{\delta} $  when  $ c_{\delta} = 0.04 $ .

Evaluation Metrics. We consider the case where  $ c_{\delta} \in \{0, 0.04, 0.12\} $ . For reference shape normalized to  $ [-1, 1] $ , having a  $ c_{\delta} = 0.12 $  is very small. We compare the main loss with the addition of PPM-Reg and track the 2-Wasserstein distance up-to the 1-dimensional persistence diagrams along gradient step.

Result. Figure 5 shows the change in PD distance as  $ c_{\delta} $  increases with the circle Figure 5(a) and union of two circles Figure 5(b). Compared with only using MMD as main loss, adding PPM-Reg consistently converges to a smaller PD distance regardless of the  $ c_{\delta} $  value. In contrast, when only using MMD, the PD distance increases as  $ c_{\delta} $  value increases. Figure 5 illustrates the benefit of explicitly comparing the difference between topological features with PPM-Reg compared with implicitly considering topological features with only MMD when the optimization problem may not converge to near zero.

## F UNCONDITIONAL IMAGE GENERATION

### F.1 IMPLEMENTATION DETAILS FOR UNCONDITIONAL IMAGE GENERATION

This section fills in the details of the unconditional image generation experiment in Section 6.3.


<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">$ \lambda $</td><td colspan="3">$ \sigma=0.05 $</td><td colspan="3">$ \sigma=0.5 $</td><td colspan="3">$ \sigma=1.0 $</td></tr><tr><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>FD $ _{\text{Dinov2}} $</td><td style='text-align: center;'>WD $ _{\text{latent}} $</td><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>FD $ _{\text{Dinov2}} $</td><td style='text-align: center;'>WD $ _{\text{latent}} $</td><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>FD $ _{\text{Dinov2}} $</td><td style='text-align: center;'>WD $ _{\text{latent}} $</td></tr><tr><td style='text-align: center;'>1.0</td><td style='text-align: center;'>0.74</td><td style='text-align: center;'>945.27</td><td style='text-align: center;'>0.6330</td><td style='text-align: center;'>0.68</td><td style='text-align: center;'>846.33</td><td style='text-align: center;'>0.6228</td><td style='text-align: center;'>0.56</td><td style='text-align: center;'>780.68</td><td style='text-align: center;'>0.6080</td></tr><tr><td style='text-align: center;'>5.0</td><td style='text-align: center;'>0.73</td><td style='text-align: center;'>928.60</td><td style='text-align: center;'>0.6310</td><td style='text-align: center;'>0.72</td><td style='text-align: center;'>884.40</td><td style='text-align: center;'>0.6282</td><td style='text-align: center;'>0.74</td><td style='text-align: center;'>894.77</td><td style='text-align: center;'>0.6308</td></tr><tr><td style='text-align: center;'>10.0</td><td style='text-align: center;'>0.74</td><td style='text-align: center;'>880.68</td><td style='text-align: center;'>0.6332</td><td style='text-align: center;'>0.77</td><td style='text-align: center;'>922.58</td><td style='text-align: center;'>0.6379</td><td style='text-align: center;'>0.71</td><td style='text-align: center;'>923.50</td><td style='text-align: center;'>0.6274</td></tr></table>

<div style="text-align: center;">Table 4: Ablation study on AnimeFace dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">$ \lambda $</td><td colspan="3">$ \sigma=0.05 $</td><td colspan="3">$ \sigma=0.5 $</td><td colspan="3">$ \sigma=1.0 $</td></tr><tr><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>FD $ _{\text{Dinov2}} $</td><td style='text-align: center;'>WD $ _{\text{latent}} $</td><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>FD $ _{\text{Dinov2}} $</td><td style='text-align: center;'>WD $ _{\text{latent}} $</td><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>FD $ _{\text{Dinov2}} $</td><td style='text-align: center;'>WD $ _{\text{latent}} $</td></tr><tr><td style='text-align: center;'>1.0</td><td style='text-align: center;'>0.87</td><td style='text-align: center;'>737.09</td><td style='text-align: center;'>0.6945</td><td style='text-align: center;'>0.65</td><td style='text-align: center;'>704.74</td><td style='text-align: center;'>0.6744</td><td style='text-align: center;'>0.81</td><td style='text-align: center;'>745.39</td><td style='text-align: center;'>0.6886</td></tr><tr><td style='text-align: center;'>5.0</td><td style='text-align: center;'>0.72</td><td style='text-align: center;'>733.66</td><td style='text-align: center;'>0.6815</td><td style='text-align: center;'>0.58</td><td style='text-align: center;'>700.73</td><td style='text-align: center;'>0.6666</td><td style='text-align: center;'>0.68</td><td style='text-align: center;'>695.36</td><td style='text-align: center;'>0.6768</td></tr><tr><td style='text-align: center;'>10.0</td><td style='text-align: center;'>0.61</td><td style='text-align: center;'>690.01</td><td style='text-align: center;'>0.6691</td><td style='text-align: center;'>0.69</td><td style='text-align: center;'>683.35</td><td style='text-align: center;'>0.6781</td><td style='text-align: center;'>0.71</td><td style='text-align: center;'>719.39</td><td style='text-align: center;'>0.6795</td></tr></table>

<div style="text-align: center;">Table 5: Ablation study on CelebA dataset.</div>


Implementation Details. The network architectures used in the unconditional image generation in Section 6.3 are shown in Figure 8. As illustrated in Figure 8,  $ g_{\omega} $  is a ResNet based CNN that takes 128-dimensional noise vector as input.  $ d_{\theta} $  is a CNN that outputs a 128-dimensional latent vector.

We compare the basic Cramer (Bellemare et al., 2017) value function V = L, with the use of PPMReg V = L + T. For the addition of PPM-Reg case, we fix  $ \lambda_{0} = 0.001 $ ,  $ \lambda_{1} = 0.6 $ , and s = 1024.  $ \lambda = \{1.0, 5.0, 10.0\} $  and  $ \sigma = \{0.05, 0.1, 0.5\} $  are the tuning parameter. In both cases, the standard Adam optimizer with learning rate  $ 1 \times 10^{-4} $  is used to train the network. For  $ g_{\omega} $ ,  $ \beta_{1} = 0.0 $  and  $ \beta_{2} = 99 $ . For  $ d_{\theta} $ ,  $ \beta_{1} = 0.5 $  and  $ \beta_{2} = 0.99 $ . The batch size is 192. For the CelebA dataset, the GAN training run for 5440 epochs. For AnimeFace data set, the GAN training run for 7000 epochs.

During training, we compute CMMD for every 160 epochs. We report the CMMD (Jayasumana et al., 2024) and  $ WD_{latent} $  and  $ FD_{Dinov2} $  (Stein et al., 2024) with the smallest CMMD value across training epochs. Those quantitative metrics are computed by sampling 10K images from the data set and generating 10k images from the network.

Ablation Study. There are two primary parameters in our topological regularizer. The parameter  $ \lambda $  controls the strength of the regularizer, while  $ \sigma $  controls the width of the RBF kernel in defining the MMD for PPMs. We provide an ablation study to show how the evaluation metrics change as we vary these parameters in Table 5 for AnimeFace and Table 4 for CelebA.

### F.2 FURTHER UNCONDITIONAL IMAGE GENERATION EXPERIMENT

<div style="text-align: center;">This section introduces supplementary unconditional image generation experiments with an increase in image resolution as well as additional datasets in conjunction with appropriate network architectures.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="3">CelebA</td><td colspan="3">LSUN Kitchen</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>FD $ _{\text{Dinov2}} $</td><td style='text-align: center;'>WD $ _{\text{latent}} $</td><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>FD $ _{\text{Dinov2}} $</td><td style='text-align: center;'>WD $ _{\text{latent}} $</td></tr><tr><td style='text-align: center;'>Cramer (Bellemare et al., 2017)</td><td style='text-align: center;'>0.52</td><td style='text-align: center;'>902.09</td><td style='text-align: center;'>0.7335</td><td style='text-align: center;'>1.56</td><td style='text-align: center;'>1592.06</td><td style='text-align: center;'>0.7690</td></tr><tr><td style='text-align: center;'>Cramer + PPM-Reg</td><td style='text-align: center;'>0.46</td><td style='text-align: center;'>826.04</td><td style='text-align: center;'>0.7296</td><td style='text-align: center;'>1.31</td><td style='text-align: center;'>1381.22</td><td style='text-align: center;'>0.7502</td></tr></table>

<div style="text-align: center;">Table 6: Quantitative evaluation on  $ 64 \times 64 $  image generation, values are reported at the epoch with the smallest CMMD.</div>


Implementation Details. Similar to Section 6.3,  $ g_{\omega} $  is a ResNet based CNN that takes 128-dimensional noise vector as input.  $ d_{\theta} $  is a CNN that outputs a 128-dimensional latent vector. The major difference is instead of just generating  $ 32 \times 32 $  images as in Section 6.3, we test our PPM-Reg with higher resolution. Specifically, we consider the  $ 64 \times 64 $  image generation task. To accommodate the increase in modeling complexity, we introduce a new network architecture shown in Figure 9.

To avoid confusion, we will write out our implementation details. We are comparing Cramer (Belle-mare et al., 2017) and the addition of PPM-Reg. For the addition of PPM-Reg case, we only fix  $ \lambda_{0}=0.001 $  and s=1024. The value of  $ \lambda_{1} $  changes as the training epoch runs. Specifically, we adapt cosine annealing (Loshchilov & Hutter, 2016) on  $ \lambda_{1} $ , the  $ \lambda_{1} $  value at epoch t term  $ \lambda_{1}^{t} $  is given by

 $$ \lambda_{1}^{t}=\begin{cases}\lambda_{1}^{min}+\frac{1}{2}(\lambda_{1}^{max}-\lambda_{1}^{min})(1+\cos(\frac{t}{t_{end}}\pi)),&if t\leq t_{end}\\ \lambda_{1}^{min},&if t>t_{end},\end{cases} $$ 

where  $ \lambda_{1}^{min} $ ,  $ \lambda_{1}^{max} $  and  $ t_{end} $  are user-defined variable.  $ \lambda_{1}^{min} $  and  $ \lambda_{1}^{max} $  are the range of the  $ \lambda_{1} $ ,  $ t_{end} $  is the ending epoch for the cosine annealing. We fix  $ \lambda_{1}^{min}=0.1 $ ,  $ \sigma=0.5 $  for both dataset. For CelebA,  $ \lambda=1 $ ,  $ \lambda_{1}^{max}=1 $  and  $ t_{end}=1920 $ . For LSUN Kitchen,  $ \lambda=10 $ ,  $ \lambda_{1}^{max}=0.8 $ , and  $ t_{end}=260 $ . The standard Adam optimizer with learning rate  $ 1\times10^{-4} $  is used to train the network. For  $ g_{\omega} $ ,  $ \beta_{1}=0.0 $  and  $ \beta_{2}=99 $ . For  $ d_{\theta} $ ,  $ \beta_{1}=0.5 $  and  $ \beta_{2}=0.99 $ . The batch size is 192. For the CelebA dataset, the GAN training is run for 2560 epochs, while for the LSUN Kitchen data set, the GAN training is run for 300 epochs.

Dataset and Evaluation Metrics. We add a new dataset LSUN Kitchen (Yu et al., 2015) and also use CelebA (Liu et al., 2015) at a higher resolution. Images are centered and resized to  $ 64 \times 64 $ .

<div style="text-align: center;"><img src="imgs/img_in_chart_box_230_163_484_353.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_488_164_736_354.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_738_166_991_353.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_230_383_484_579.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">(c)</div>


<div style="text-align: center;">(d)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_488_388_737_579.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">(e)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_740_387_991_578.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">(f)</div>


<div style="text-align: center;">Figure 6: CMMD (a,e),  $ FD_{Dinov2} $  (b,f) and  $ WD_{latent} $  (c,g) versus training epochs for the CelebA (a-c) and LSUN Kitchen (d-f) dataset for the  $ 64 \times 64 $  image generation. 10K samples are randomly generated to compute the distances, and moving averages with a window of 5 are used to smooth the values. For CelebA, distances are recorded every 160 epochs. For LSUN Kitchen, distances are recorded every 20 epochs.</div>


During training, we compute CMMD for every 160 epochs for CelebA and 20 epochs for LSUN Kitchen. We report the CMMD (Jayasumana et al., 2024) and  $ WD_{latent} $  and  $ FD_{Dinov2} $  (Stein et al., 2024) with the smallest CMMD value across training epochs. For justification for using the chosen quantitative evaluation metrics, readers can refer to Section 6.3. Those quantitative metrics are computed by sampling 10K images from the data set and generating 10k images from the network.

Result. Figure 6 tracks the three quantitative metrics during training and Table 6 reports the three metrics with the smallest CMMD. While the  $ FD_{Dinov2} $  metric has comparable values in Figure 6(b), at the point of the smallest CMMD, Cramer + PPM-Reg has a smaller  $ FD_{Dinov2} $  for CelebA, shown in Table 6. Our quantitative results in Table 6 reinforce the fact that using PPM-Reg improves the generative ability of GANs. Compared to CelebA dataset, using PPM-Reg in the LSUN Kitchen dataset results in a more significant improvement. We conjecture that since LSUN Kitchen is a much larger dataset (2,212,277 training samples) that contains diverse images (i.e., different color tones, layouts), its underlying lower dimensional submanifold has more complex topological features compared with CelebA. The significant improvement gives evidence that PPM-Reg is useful in discovering more complex topological structures in latent space.

## G SEMI-SUPERVISED LEARNING

### G.1 IMPLEMENTATION DETAILS FOR SEMI-SUPERVISED LEARNING

This section fills in the details of the semi-supervised learning experiment in Section 6.4.

Network Architecture and Implementation Details. The network architectures used in the semi-supervised learning experiment in Section 6.4 are shown in Figure 10. The input noise vector of  $ g_{\omega} $  has dimension of 64. We set the dimension of the output latent vector of  $ d_{\theta} $  as 64.

Specifically,  $ g_{\omega} $  and  $ d_{\theta} $  are trained with the GAN framework for 4,000 epochs using the standard Adam optimizer with learning rate  $ 1 \times 10^{-4} $ . For  $ g_{\omega} $ , we set  $ \beta_{1} = 0.0 $  and  $ \beta_{2} = 99 $ ; for  $ d_{\theta} $ , we set  $ \beta_{1} = 0.5 $  and  $ \beta_{2} = 0.99 $ . The batch size is 192. For Cramer + PPM-Reg, we fix  $ \lambda_{0} = 1 $ ,  $ \lambda_{1} = 90 $ .

<div style="text-align: center;"><img src="imgs/img_in_image_box_343_159_554_369.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_667_159_879_369.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;">Figure 7: Generated images from SSL experiment from trained  $ g_{\omega} $  for MNIST using (a) Cramer Distance and (b) PPM-Reg.</div>


<div style="text-align: center;">and $s=1024$. $\lambda=\{0.025,0.05,0.1\}$ and $\sigma=\{0.05,0.1,0.5\}$ are the tuning parameter. After the GAN framework completes the training of $g_{\omega}$ and $d_{\theta}$, the weights of $d_{\theta}$ are frozen. Using $d_{\theta}$ as feature extraction, the output of $d_{\theta}$ is fed into the classifier. The classifier $c_{\gamma}$ is train with the standard Adam optimizer with learning rate $1\times10^{-4}$ where $\beta_{1}=0.1$ and $\beta_{2}=0.99$ for 1000 epochs.</div>


<div style="text-align: center;">"Baseline" connects  $ d_{\theta} $  and  $ c_{\gamma} $  and trains as a classifier without first pertaining  $ d_{\theta} $  with GANs. The standard Adam optimizer with learning rate  $ 1 \times 10^{-4} $  where  $ \beta_{1} = 0.1 $  and  $ \beta_{2} = 0.99 $  are used to train the "Baseline" network for 4,000 epochs.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="3">Fashion-MNIST (200 labels)</td><td colspan="3">Fashion-MNIST (400 labels)</td></tr><tr><td style='text-align: center;'>$ \lambda $</td><td style='text-align: center;'>$ \sigma=0.05 $</td><td style='text-align: center;'>$ \sigma=0.1 $</td><td style='text-align: center;'>$ \sigma=0.5 $</td><td style='text-align: center;'>$ \sigma=0.05 $</td><td style='text-align: center;'>$ \sigma=0.1 $</td><td style='text-align: center;'>$ \sigma=0.5 $</td></tr><tr><td style='text-align: center;'>0.1</td><td style='text-align: center;'>75.39  $ \pm $  1.12</td><td style='text-align: center;'>74.96  $ \pm $  0.96</td><td style='text-align: center;'>75.24  $ \pm $  1.20</td><td style='text-align: center;'>75.39  $ \pm $  1.12</td><td style='text-align: center;'>74.96  $ \pm $  0.96</td><td style='text-align: center;'>75.24  $ \pm $  1.20</td></tr><tr><td style='text-align: center;'>0.05</td><td style='text-align: center;'>76.35  $ \pm $  1.09</td><td style='text-align: center;'>76.25  $ \pm $  1.30</td><td style='text-align: center;'>76.81  $ \pm $  0.88</td><td style='text-align: center;'>76.35  $ \pm $  1.09</td><td style='text-align: center;'>76.25  $ \pm $  1.30</td><td style='text-align: center;'>76.81  $ \pm $  0.88</td></tr><tr><td style='text-align: center;'>0.025</td><td style='text-align: center;'>76.42  $ \pm $  0.85</td><td style='text-align: center;'>76.77  $ \pm $  1.29</td><td style='text-align: center;'>76.84  $ \pm $  1.23</td><td style='text-align: center;'>76.42  $ \pm $  0.85</td><td style='text-align: center;'>76.77  $ \pm $  1.29</td><td style='text-align: center;'>76.84  $ \pm $  1.23</td></tr></table>

<div style="text-align: center;">Table 7: Ablation study on Fashion-MNIST dataset with 200/400 labels. Test-set classification accuracy (\%) is shown averaged over 10 runs.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">$ \lambda $</td><td colspan="3">Kuzushiji-MNIST (200 labels)</td><td colspan="3">Kuzushiji-MNIST (400 labels)</td></tr><tr><td style='text-align: center;'>$ \sigma=0.05 $</td><td style='text-align: center;'>$ \sigma=0.1 $</td><td style='text-align: center;'>$ \sigma=0.5 $</td><td style='text-align: center;'>$ \sigma=0.05 $</td><td style='text-align: center;'>$ \sigma=0.1 $</td><td style='text-align: center;'>$ \sigma=0.5 $</td></tr><tr><td style='text-align: center;'>0.1</td><td style='text-align: center;'>73.70 \pm 2.58</td><td style='text-align: center;'>70.84 \pm 2.56</td><td style='text-align: center;'>70.16 \pm 1.96</td><td style='text-align: center;'>78.35 \pm 1.81</td><td style='text-align: center;'>76.18 \pm 1.87</td><td style='text-align: center;'>76.67 \pm 1.31</td></tr><tr><td style='text-align: center;'>0.05</td><td style='text-align: center;'>74.35 \pm 2.45</td><td style='text-align: center;'>73.65 \pm 1.70</td><td style='text-align: center;'>73.97 \pm 1.24</td><td style='text-align: center;'>78.99 \pm 1.41</td><td style='text-align: center;'>79.92 \pm 1.88</td><td style='text-align: center;'>78.77 \pm 1.48</td></tr><tr><td style='text-align: center;'>0.025</td><td style='text-align: center;'>74.47 \pm 1.65</td><td style='text-align: center;'>75.78 \pm 1.99</td><td style='text-align: center;'>75.41 \pm 2.83</td><td style='text-align: center;'>79.40 \pm 1.65</td><td style='text-align: center;'>79.33 \pm 1.69</td><td style='text-align: center;'>80.04 \pm 1.37</td></tr></table>

<div style="text-align: center;">Table 8: Ablation study on Kuzushiji-MNIST with 200/400 labels. Test-set classification accuracy (\%) is shown averaged over 10 runs.</div>


Ablation Study. We show how the classification accuracy varies with respect to the topological regularization strength  $ \lambda $  and the RBF width  $ \sigma $  in Table 7 for Fashion-MNIST, Table 8 for KuzushijIMNIST, and Table 9 for MNIST.

### G.2 FURTHER SEMI-SUPERVISED LEARNING EXPERIMENT

This section introduces supplementary semi-supervised learning experiments with additional datasets in conjunction with appropriate network architectures.

Network Architecture and Implementation Details. Similar to the setup in Section 6.4,  $ g_{\omega} $  is a deconvolutional network with 64 dimension noise vector as input.  $ d_{\theta} $  is a CNN with a 64 dimension latent vector as output. The additional SVHM dataset is more intricate than the MNIST variants evaluated in Section 6.4, as it is a color image dataset with higher resolution. To tackle the increase in


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="3">MNIST (200 labels)</td><td colspan="3">MNIST (400 labels)</td></tr><tr><td style='text-align: center;'>$ \lambda $</td><td style='text-align: center;'>$ \sigma=0.05 $</td><td style='text-align: center;'>$ \sigma=0.1 $</td><td style='text-align: center;'>$ \sigma=0.5 $</td><td style='text-align: center;'>$ \sigma=0.05 $</td><td style='text-align: center;'>$ \sigma=0.1 $</td><td style='text-align: center;'>$ \sigma=0.5 $</td></tr><tr><td style='text-align: center;'>0.1</td><td style='text-align: center;'>96.34 \pm 0.29</td><td style='text-align: center;'>96.32 \pm 0.20</td><td style='text-align: center;'>96.18 \pm 0.19</td><td style='text-align: center;'>97.19 \pm 0.28</td><td style='text-align: center;'>97.06 \pm 0.28</td><td style='text-align: center;'>97.07 \pm 0.26</td></tr><tr><td style='text-align: center;'>0.05</td><td style='text-align: center;'>96.61 \pm 0.37</td><td style='text-align: center;'>96.62 \pm 0.39</td><td style='text-align: center;'>96.22 \pm 0.59</td><td style='text-align: center;'>97.29 \pm 0.20</td><td style='text-align: center;'>97.33 \pm 0.21</td><td style='text-align: center;'>97.16 \pm 0.25</td></tr><tr><td style='text-align: center;'>0.025</td><td style='text-align: center;'>96.13 \pm 0.42</td><td style='text-align: center;'>95.97 \pm 0.37</td><td style='text-align: center;'>95.91 \pm 0.55</td><td style='text-align: center;'>97.04 \pm 0.27</td><td style='text-align: center;'>96.92 \pm 0.20</td><td style='text-align: center;'>97.2 \pm 0.25</td></tr></table>

<div style="text-align: center;">Table 9: Ablation study on MNIST with 200/400 labels. Test-set classification accuracy (\%) is shown averaged over 10 runs.</div>


input dimension and the complexity of modeling the dataset, we introduce a new network architecture shown in Figure 11.

The training setup largely follows the previous experiment; the major difference is the number of training epochs. Specifically,  $ g_{\omega} $  and  $ d_{\theta} $  are trained with the GAN framework for 1,200 epochs. The standard Adam optimizer with learning rate  $ 1 \times 10^{-4} $  is used to train the network. For  $ g_{\omega} $ ,  $ \beta_{1} = 0.0 $  and  $ \beta_{2} = 99 $ . For  $ d_{\theta} $ ,  $ \beta_{1} = 0.5 $  and  $ \beta_{2} = 0.99 $ . The batch size is 192. For Cramer + PPM-Reg, we fix  $ \lambda_{0} = 1 $ ,  $ \lambda_{1} = 90 $  and s = 1024.  $ \lambda = \{0.025, 0.05, 0.1\} $  and  $ \sigma = \{0.05, 0.1, 0.5\} $  are the tuning parameters. After the GAN framework completes the training of  $ g_{\omega} $  and  $ d_{\theta} $ , the weights of  $ d_{\theta} $  are frozen. Using  $ d_{\theta} $  as feature extraction, the output of  $ d_{\theta} $  is fed into the classifier. The classifier  $ c_{\gamma} $  is trained with the standard Adam optimizer with learning rate  $ 1 \times 10^{-4} $  where  $ \beta_{1} = 0.1 $  and  $ \beta_{2} = 0.99 $  for 1000 epochs. The standard Adam optimizer with learning rate  $ 1 \times 10^{-4} $  where  $ \beta_{1} = 0.1 $  and  $ \beta_{2} = 0.99 $  are used to train the "Baseline" network for 4,000 epochs.

<div style="text-align: center;">Dataset and Evaluation Metrics. We compare the SSL performance with the dataset SVHN. In this experiment, 400 and 600 labels are randomly sampled from the data set. Because of the random nature involved in selecting a few labels, we conducted the experiments ten times and provided the statistics of the highest test-set accuracy achieved.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Number of labels</td><td colspan="2">SVHN</td></tr><tr><td style='text-align: center;'>400</td><td style='text-align: center;'>600</td></tr><tr><td style='text-align: center;'>Baseline</td><td style='text-align: center;'>44.68  $ \pm $  2.44</td><td style='text-align: center;'>53.68  $ \pm $  4.15</td></tr><tr><td style='text-align: center;'>Cramer</td><td style='text-align: center;'>38.12  $ \pm $  1.75</td><td style='text-align: center;'>43.23  $ \pm $  1.14</td></tr><tr><td style='text-align: center;'>Cramer + PPM-Reg</td><td style='text-align: center;'>57.20  $ \pm $  1.51</td><td style='text-align: center;'>61.39  $ \pm $  0.66</td></tr></table>

<div style="text-align: center;">Table 10: Test-set classification accuracy (\%) on SVHN with 400 and 600 labeled examples. The average and the error bar are computed over 10 runs.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="3">SVHN (400 labels)</td><td colspan="3">SVHN (600 labels)</td></tr><tr><td style='text-align: center;'>$ \lambda $</td><td style='text-align: center;'>$ \sigma=0.05 $</td><td style='text-align: center;'>$ \sigma=0.1 $</td><td style='text-align: center;'>$ \sigma=0.5 $</td><td style='text-align: center;'>$ \sigma=0.05 $</td><td style='text-align: center;'>$ \sigma=0.1 $</td><td style='text-align: center;'>$ \sigma=0.5 $</td></tr><tr><td style='text-align: center;'>0.1</td><td style='text-align: center;'>55.66  $ \pm $  1.78</td><td style='text-align: center;'>57.20  $ \pm $  1.51</td><td style='text-align: center;'>54.84  $ \pm $  1.21</td><td style='text-align: center;'>59.48  $ \pm $  1.26</td><td style='text-align: center;'>61.39  $ \pm $  0.66</td><td style='text-align: center;'>59.13  $ \pm $  0.95</td></tr><tr><td style='text-align: center;'>0.05</td><td style='text-align: center;'>47.56  $ \pm $  1.67</td><td style='text-align: center;'>56.24  $ \pm $  1.25</td><td style='text-align: center;'>52.41  $ \pm $  1.49</td><td style='text-align: center;'>50.73  $ \pm $  1.06</td><td style='text-align: center;'>60.08  $ \pm $  0.84</td><td style='text-align: center;'>60.91  $ \pm $  1.19</td></tr><tr><td style='text-align: center;'>0.025</td><td style='text-align: center;'>55.96  $ \pm $  1.03</td><td style='text-align: center;'>56.96  $ \pm $  1.92</td><td style='text-align: center;'>55.35  $ \pm $  1.06</td><td style='text-align: center;'>59.13  $ \pm $  0.95</td><td style='text-align: center;'>56.45  $ \pm $  1.41</td><td style='text-align: center;'>58.97  $ \pm $  1.00</td></tr></table>

<div style="text-align: center;">Table 11: Ablation study on SVHN with 400/600 labels. Test-set classification accuracy (\%) is shown averaged over 10 runs.</div>


Result. Table 10 shows the test classification accuracy. An ablation study is provided in Table 11. SVHN contains 72,657 training samples and 400 and 600 labels constitute only 0.55% and 0.82% of the original datasets, respectively. The results follow the same characteristic in Section 6.4, only using Cramer does not improve the classification accuracy in SSL. In contrast, the use of PPM-Reg results in a notable improvement in classification accuracy. Specifically, using PPM-Reg with 400 labels yields a 12.52% increase in accuracy compared to the Baseline. The consistent outcome with more complex datasets and network architecture reinforces our claim that PPM-Reg can help learn a more informative latent encoding thereby improving SSL performance.

## H NETWORK ARCHITECTURE

<div style="text-align: center;"><img src="imgs/img_in_image_box_311_218_635_1311.jpg" alt="Image" width="26%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_728_276_909_1294.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;">Figure 8: Network architecture of generator (a), discriminator (b) for  $ 32 \times 32 $  unconditional image generation.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_292_157_629_1399.jpg" alt="Image" width="27%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_725_172_927_1394.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;">Figure 9: Network architecture of generator (a), discriminator (b) for  $ 64 \times 64 $  unconditional image generation.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_337_372_489_1296.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_539_239_685_1299.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_735_550_884_1293.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;">(c)</div>


<div style="text-align: center;">Figure 10: Network architecture of generator (a), discriminator (b) and classi\textsubscript{a}ter (c) for semi-supervised learning for MNIST variants.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_338_379_488_1282.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_539_251_684_1281.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_735_560_883_1279.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;">(c)</div>


<div style="text-align: center;">Figure 11: Network architecture of generator (a), discriminator (b) and classi\textsubscript{acter} (c) for semi-supervised learning for SVHN dataset.</div>