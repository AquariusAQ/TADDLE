# TOWARDS SCALABLE TOPOLOGICAL REGULARIZERS

Hiu-Tung Wong $ ^{1,\dagger} $ , Darrick Lee $ ^{2,\dagger} $ , Hong Yan $ ^{1,3} $ 

 $ ^{1} $ Centre for Intelligent Multidimensional Data Analysis, Science and Technology Park, Hong Kong  

 $ ^{2} $ School of Mathematics, University of Edinburgh, UK

 $ ^{3} $ Department of Electrical Engineering, City University of Hong Kong, Kowloon, Hong Kong hiutang@innocimda.com, darrick.lee@ed.ac.uk, h.yan@cityu.edu.hk

## ABSTRACT

Latent space matching, which consists of matching distributions of features in latent space, is a crucial component for tasks such as adversarial attacks and defenses, domain adaptation, and generative modelling. Metrics for probability measures, such as Wasserstein and maximum mean discrepancy, are commonly used to quantify the differences between such distributions. However, these are often costly to compute, or do not appropriately take the geometric and topological features of the distributions into consideration. Persistent homology is a tool from topological data analysis which quantifies the multi-scale topological structure of point clouds, and has recently been used as a topological regularizer in learning tasks. However, computation costs preclude larger scale computations, and discontinuities in the gradient lead to unstable training behavior such as in adversarial tasks. We propose the use of principal persistence measures, based on computing the persistent homology of a large number of small subsamples, as a topological regularizer. We provide a parallelized GPU implementation of this regularizer, and prove that gradients are continuous for smooth densities. Furthermore, we demonstrate the efficacy of this regularizer on shape matching, image generation, and semi-supervised learning tasks, opening the door towards a scalable regularizer for topological features.

## 1 INTRODUCTION

Latent space matching is a fundamental task in deep learning. Quantifying differences in latent representations and optimizing the network accordingly enables applications such as adversarial attack and defenses (Yu et al., 2021; Madaan et al., 2020; Lin et al., 2020), domain adaptation (Sun et al., 2016; Sun & Saenko, 2016; Long et al., 2017; Xu et al., 2019) and few-shot learning (Schonfeld et al., 2019; Xu et al., 2022; Mondal et al., 2023). Unsupervised training frameworks such as Generative Adversarial Networks (GANs) (Goodfellow et al., 2014) are fundamentally built on this concept, which is the primary framework considered throughout this article.

Topological Features of Latent Representations. The manifold hypothesis states that real-world high dimensional data sets are often concentrated about lower dimensional submanifolds. Recent work has empirically verified that image datasets such as CIFAR-10 and ImageNet satisfy a union of manifolds hypothesis (Brown et al., 2022), where the intrinsic dimension of connected components may be different. In the context of GANs, correctly learning the geometric and topological properties of these lower-dimensional structures enable meaningful interpolation in data space by traversing network latent spaces. Such properties are crucial to generalization ability (Zhou et al., 2020; Wang et al., 2021b) and generation quality (Zhu et al., 2023; Katsumata et al., 2024). Furthermore, topological metrics based on persistent homology provide highly effective evaluation metrics for GANs (Zhou et al., 2021; Khrulkov & Oseledets, 2018; Barannikov et al., 2021; Charlier et al., 2019).

Standard approaches to latent space matching use metrics on probability measures such as the Wasserstein distance and maximum mean discrepancy (MMD) metrics, which implicitly take topological features into consideration. In particular, the measures (and thus all topological properties) become

equivalent when the distance is trivial. However, GANs are not guaranteed to reach a global minimum, and often converge to local saddle points (Berard et al., 2019; Liang & Stokes, 2019). When measures are a finite distance apart, their topological properties may be distinct. Motivated by the above work, which demonstrates that topological similarity between real and generated distributions is a critical component of GAN performance, we propose the use of a topological regularizer which explicitly measures the difference between topological features at non-equilibrium states.

Persistent Homology. Persistent homology (PH) is a tool which summarizes the multi-scale topological features of a dataset in an object called a persistence diagram. Such topological summaries have been applied in machine learning tasks (Hensel et al., 2021), such as image segmentation (Hu et al., 2019; Clough et al., 2022; Shit et al., 2021; Waibel et al., 2022), and graph learning (Horn et al., 2021; Ballester & Rieck, 2024). The standard way to quantify the topological differences between datasets is to compute the Wasserstein distance between their persistence diagrams. However, there are two difficulties in directly applying persistence-based methods in adversarial deep learning tasks.

1. Scalability. Persistent homology of a large point cloud is prohibitively expensive to compute $ ^{1} $ . Even worse, the persistent homology algorithm is highly nontrivial to parallelize. Modern PH packages are either pure CPU implementations (Bauer, 2021; Pérez et al., 2021) or use a CPU-GPU hybrid algorithm (Zhang et al., 2020).

2. Smoothness. Persistent homology is differentiable almost everywhere, which allows us to compute backpropagate through PH layers; in fact, stochastic subgradient descent is provably convergent with respect to persistence-based functions (Carriere et al., 2021). However, in adversarial tasks, where the loss function is constantly changing, discontinuities in the gradient leads to highly unstable training dynamics (Wiatrak et al., 2019).

Contributions. We address these two issues by modifying the two central parts of the classical persistence pipeline: the topological summary itself, as well as the metric used to compare them.

• Topological Summary: Principal Persistence Measures. To reduce the computational cost, we compute the persistent homology of many small batches of subsamples in parallel. By choosing a specific number of points depending on the homology dimension, the persistence computation significantly simplifies, and we obtain an object called the principal persistence measure (PPM) (Gómez & Mémo, 2024). We provide a pure GPU implementation of the PPM, which enables a scalable methodology to incorporate topological features in larger-scale ML tasks. Moreover, subsampling results in a smoother features (Solomon et al., 2021), resulting in more stable training behavior.

• Topological Metric: Maximum Mean Discrepancy for PPMs. The Wasserstein distance is the primary metric used to compare PPMs (Gómez & Mémoli, 2024). In practice, one often uses entropic regularization to lower the computational cost (Cuturi, 2013; Lacombe et al., 2018). Despite this, it is still computationally expensive, and we use maximum mean discrepancy (MMD) metrics to compare PPMs. This coincides with the persistence weighted kernels introduced in (Kusano et al., 2016) for persistence diagrams. Our main theoretical results deal with establishing this metric in PPM framework.

– Theorem 1 builds characteristic kernels for PPMs from kernels on  $ R^{2} $ 

– Theorem 2 shows that these MMD metrics induce the same topology as Wasserstein.

– Theorem 3 shows that gradients with respect to this metric are continuous.

Theorem 1 and Theorem 2 adapt results from (Kusano et al., 2016; Divol & Lacombe, 2021) to the setting of PPMs, while to the authors' knowledge, Theorem 3 is novel.

These theoretical results imply that we can use PPM-Reg as an alternative to computationally expensive Wasserstein (or Sinkhorn) metrics, which produces a stable gradient for training deeper networks. In particular, the proposed methods allow us to incorporate topological features into large-scale machine learning tasks in a stable manner (Papamarkou et al., 2024, Section 4.2). We

demonstrate this empirically in Section 6, where we provide extensive experiments to demonstrate the efficacy of PPM-Reg in the GAN framework.

Related Work. The application of persistent homology in machine learning has been enabled by theoretical studies into the differentiability properties of PH (Carriere et al., 2021; Leygonie et al., 2022), which have also been extended to the multiparameter setting (Scoccola et al., 2024). However, large-scale computation of PH remains a challenge, though recent work has considered computational strategies for optimization problems (Nigmetov & Morozov, 2024; Luo & Nelson, 2024). Our MMD metric for PPMs is also related to work on kernels for persistence diagrams Kusano et al. (2016) and linear representations of persistence diagrams Divol & Lacombe (2021); Divol & Polonik (2019).

Subsampling methods for PH of metric measure spaces was introduced in (Blumberg et al., 2014), and used to approximate PH for point clouds (Chazal et al., 2015; Cao & Monod, 2022; Stolz, 2023). Furthermore, distributed approaches for computing the true PH of point clouds have been proposed in (Yoon & Ghrist, 2020; Torras-Casas, 2023) via spectral sequence methods. More recently, (Solomon et al., 2021) used subsampling methods for topological function optimization, motivated by the same issues of computational cost and instability of gradients (Bendich et al., 2020), and (Solomon et al., 2022) showed that such distributed persistence methods interpolate between geometric and topological features based on the number of subsamples. The starting point of this article is (Gómez & Mémoi, 2024), which introduces principal persistence measures.

## 2 LATENT SPACE MATCHING IN GENERATIVE ADVERSARIAL NETWORKS

Our primary consideration is the latent space matching in generative adversarial networks (GANs). A GAN is an unsupervised training framework consisting of a generator  $ g_{\omega}: R^{N} \to R^{M} $ , a discriminator  $ d_{\theta}: R^{M} \to R^{L} $  and a value function  $ \mathcal{V}: \mathcal{P}(\mathbb{R}^{L}) \times \mathcal{P}(\mathbb{R}^{L}) \to \mathbb{R} $  (Goodfellow et al., 2014). Consider a set of training data, such as a collection of images, which we view as a probability measure  $ \mu $  on  $ R^{M} $ , the data space. The generator  $ g_{\omega} $  is parameterized by  $ \omega \in R^{G} $ , and its goal is to map a given noise measure  $ \nu $  on  $ R^{N} $  (the noise space) to  $ R^{M} $  such that  $ g_{\omega}(\nu) $  can be interpreted as novel examples of  $ \mu $ . The discriminator  $ d_{\theta} $ , parametrized by  $ \theta \in R^{D} $ , performs dimensionality reduction, sending the data space to the latent space  $ R^{L} $ . Finally, the value function is used to quantify the difference between the real data  $ \mu $  and the generated data  $ g_{\omega}(\nu) $  by  $ \mathcal{V}(d_{\theta}(\mu), d_{\theta}(g_{\omega}(\nu))) $ .

The generator is optimized such that it minimizes the value function, while the goal of the discriminator is to maximize it. Training algorithms (Goodfellow et al., 2014; Arjovsky et al., 2017; Gulrajani et al., 2017) have been proposed to find an equilibrium of the minimax problem, given by

 $$ \operatorname*{m a x}_{\theta}\operatorname*{m i n}_{\omega}\mathcal{V}\left(d_{\theta}(\mu),d_{\theta}(g_{\omega}(\nu))\right). $$ 

In practice, parameters in  $ d_{\theta} $  and  $ g_{\omega} $  are updated alternatively. Common value functions used are metrics between probability distributions such as the Wasserstein distance, which has better theoretical properties in solving the minimax problem with gradient descent (Arjovsky et al., 2017), and the Cramer distance (Bellemare et al., 2017), which has unbiased gradients with mini-batch training. However, these metrics do not explicitly take topological features of the distributions into consideration. This motivates the use of a topological regularizer, which explicitly accounts for topological features in a non-equilibrium state. In particular, we consider value functions of the form

 $$ \mathcal{V}=\mathcal{L}+\lambda\mathcal{T}, $$ 

where L is the main loss function,  $ \lambda > 0 $  is a hyperparameter, T is our proposed topological regularizer which will be introduced in the following sections.

## 3 PRINCIPAL PERSISTENCE MEASURES

We provide a streamlined exposition of the notion of principal persistence measures (Gómez & Mémoli, 2024). As there already exists several excellent references for persistent homology, we refer the reader to (Edelsbrunner & Harer, 2010; Dey & Wang, 2022; Hensel et al., 2021) for further background. Furthermore, we highlight the fact that we only consider persistent homology for simple point clouds, with an explicit definition in Equation (4). For a topological space X, we use  $ \mathcal{P}(\mathcal{X}) $  (resp.  $ \mathcal{P}_{c}(\mathcal{X}) $ ) to denote the Borel probability measure (resp. with compact support) on X.

Throughout this article, we consider persistent homology of point clouds with the Vietoris-Rips filtration.

<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_217_238_364_382.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_373_229_801_387.jpg" alt="Image" width="34%" /></div>


<div style="text-align: center;">(d)</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_215_394_365_556.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;">(c)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_811_239_1005_392.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_372_394_804_555.jpg" alt="Image" width="35%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_809_395_1007_557.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">Figure 1: An illustration of PH and PPMs. (a) An example point cloud X. (b) Snapshots of the Vietoris-Rips filtration  $ X_{\epsilon} $  of X at various  $ \epsilon $ . Edges are added between  $ x_{i} $  and  $ x_{j} $  when  $ d(x_{i}, x_{j}) > \epsilon $  and higher simplices are added when all pairwise distances are greater than  $ \epsilon $ . (c) The dimension 1 persistence diagram of X in birth-lifetime coordinates. The one point with large lifetime represents the fact that there is a hole in the dataset which persists through multiple scales. (d) An example of a subsampling (in red) of  $ 4 = 2q + 2 $  points when q = 1. (e) Snapshots of the Vietoris-Rips filtration of the subsample, where the distances of the bold lines are  $ t_{b} $  and  $ t_{d} $ . (f) The dimension 1 principal persistence measure of X, where the point given by the example subsample is shown in red.</div>


Persistent Homology. Let  $ X = \{x_{i}\}_{i=1}^{N} $ , where  $ x_{i} \in R^{n} $ . Persistent homology (Edelsbrunner & Harer, 2010) of dimension  $ q \in N $  builds a multi-scale topological summary of X in three steps:

1. Construct a sequence of topological spaces  $ X_{\epsilon} $  representing the point cloud at a scale parameter  $ \epsilon > 0 $ , equipped with inclusion maps  $ X_{\epsilon} \hookrightarrow X_{\epsilon'} $  for  $ \epsilon < \epsilon' $  (see Figure 1(b)).

2. Compute the dimension q homology of  $ X_{\epsilon} $  to obtain topological properties at each scale.

3. Track the birth, b, and lifetime,  $ \ell $ , of topological features across scales by using the induced maps  $ H_{q}(X_{\epsilon}) \to H_{q}(X_{\epsilon^{\prime}}) $ , and summarize this information as a multi-set  $ \mathrm{PH}_{q}(X) = \{(b_{i}, \ell_{i})\}_{i=1}^{r} $  called a persistence diagram $ ^{2} $  (see Figure 1(c)).

The points (b, ℓ) ∈ PHq(X) in a persistence diagram are valued in the quotient of the half plane

 $$ \Omega:=\{(b,\ell)\in\mathbb{R}^{2}:\ell\geq0\}/\{\ell=0\}, $$ 

as topological features  $ (b,\ell)\in\mathrm{PH}_{q}(X) $  where points with trivial lifetime  $ \ell=0 $  are equivalent to the feature not existing. We view this as a pointed quotient metric space  $ (\Omega,d,*) $ , where d is the quotient of the Euclidean metric on  $ R^{2} $  and  $ ^{*} $  represents the collapsed point  $ \{\ell=0\} $ .

Persistent Homology of Small Point Clouds. It is shown in (Gómez & Mémoli, 2024, Theorem 4.4) that  $ PH_{q} $  of a point cloud S with exactly  $ 2q + 2 $  points has at most a single topological feature, and can be explicitly computed as follows. Given a point  $ x \in S $ , let  $ x^{(1)}, x^{(2)} \in S $  denote the points such that  $ d(x, x^{(1)}) \geq d(x, x^{(2)}) \geq d(x, a) $  for all  $ a \in S - \{x^{(1)}, x^{(2)}\} $ . Then,

 $$ \mathbf{P}\mathbf{H}_{q}(S)=\{(t_{b},t_{d}-t_{b})\},\quad t_{b}:=\max_{x\in S}d(x,x^{(2)}),\quad t_{d}:=\min_{x\in S}d(x,x^{(1)}) $$ 

whenever  $ t_{d} \geq t_{b} $ , and  $ \mathrm{PH}_{q}(S) = \{\ast\} $  otherwise (see Figure 1(e)). As we will exclusively consider  $ PH_{q} $  of  $ 2q + 2 $  points, we will consider this as a map  $ \mathrm{PH}_{q} : (\mathbb{R}^{n})^{2q+2} \to \Omega $ . We emphasize that Equation (4) is a significant simplification of the full persistent homology computation (Otter et al., 2017), and is the key to parallelized computations discussed in Section 6.1.

Principal Persistence Measures. Principal persistence measures (PPMs) of dimension q (Gómez & Mémoli, 2024) contains  $ PH_{q} $  of all subsamples S of a point cloud X with exactly  $ |S| = 2q + 2 $  points. More formally, we will consider the more general setting of probability measures on  $ R^{n} $  with compact support rather than point clouds $ ^{3} $  on  $ R^{n} $ . Then, the PPM of dimension q is defined as

 $$ \mathbf{P P M}_{q}:\mathcal{P}_{c}(\mathbb{R}^{n})\to\mathcal{P}(\Omega),\quad\mathbf{P P M}_{q}(\mu):=(\mathbf{P H}_{q})_{*}\mu^{\otimes(2q+2)} $$ 

where  $ \mu^{\otimes n} $  is the product measure on  $ (\mathbb{R}^{n})^{2q+2} $ , and  $ (\mathrm{PH}_{q})_{*} $  is the pushforward map. In other words, we take  $ 2q + 2 $  i.i.d. samples from  $ \mu $  and compute  $ PH_{q} $  on each collection to obtain a probability measure on  $ \Omega $  (see Figure 1(f)).

Metrics and Stability. Let  $ p \geq 1 $ , and let  $ W_{p} $  denote the p-Wasserstein metric on  $ R^{n} $  and  $ \Omega $ . A key property shown in (Gómez & Mémoi, 2024, Theorem 3.8, Theorem 4.11) is that PPMs are stable:

 $$ W_{p}(\mathrm{PPM}_{q}(\mu),\mathrm{PPM}_{q}(\nu))\leq C_{q}W_{p}(\mu,\nu),\quad\mathrm{for all}\quad\mu,\nu\in\mathcal{P}_{c}(\mathbb{R}^{n}) $$ 

where  $ C_{q} > 0 $  is a constant which depends on q. In particular, p-Wasserstein metrics on  $ \Omega $  for PPMs is the analogue of the partial p-Wasserstein distance for persistence diagrams.

## 4 MAXIMUM MEAN DISCREPANCY FOR PPMs

In order to further reduce the computational cost and obtain smoothness properties, we will use maximum mean discrepancy (MMD) metrics to compare PPMs. The kernels defined here adapted from the persistence weighted kernels of Kusano et al. (2016). We assume basic familiarity with kernels and refer the reader to Appendix A for background.

Bounded PPMs and Notation. Throughout this section, we work with bounded PPMs valued in

 $$ \Omega_{T}:=\{(b,\ell)\in[0,T]^{2}:\ell\geq0\}/\{\ell=0\} $$ 

for some T > 0. We continue to denote the collapsed point by *. Note that for  $ \mu \in \mathcal{P}_{c}(\mathbb{R}^{n}) $ , where the support of  $ \mu $  has diameter T, we have  $ \mathrm{PPM}_{k}(\mu) \in \mathcal{P}(\Omega_{T}) $ . In order to simplify notation, we use  $ \Omega = \Omega_{T} $  throughout this section. We use the notation  $ z = (b, \ell) $  for elements in both  $ [0, T]^{2} $  and  $ \Omega $ .

Kernels on  $ \Omega $ . Following the construction in Kusano et al. (2016), we introduce a procedure to turn a kernel k on  $ [0, T]^{2} $  into a kernel on  $ \Omega $ . Suppose  $ k : [0, T]^{2} \times [0, T]^{2} \to R $  is a kernel, where H is its reproducing kernel Hilbert space (RKHS), and let  $ \Phi : [0, T]^{2} \to H $  be the associated feature map given by  $ \Phi(z) = k(z, \cdot) $ . We define a feature map  $ \Phi_{\Omega} : \Omega \to H $  into the same RKHS by

 $$ \Phi_{\Omega}(z)=\ell\cdot\Phi(z)=\ell\cdot k(z,\cdot)\ when\ell>0\quad and\quad\Phi_{\Omega}(*)=0. $$ 

Then, for  $ z_{1}, z_{2} \in \Omega - \{*\} $ , the associated kernel  $ k_{\Omega}: \Omega \times \Omega \to R $ , satisfies

 $$ k_{\Omega}(z_{1},z_{2}):=\langle\Phi_{\Omega}(z_{1}),\Phi_{\Omega}(z_{2})\rangle_{\mathcal{H}}=\ell_{1}\cdot\ell_{2}\cdot k(z_{1},z_{2}) $$ 

by the reproducing kernel property of H, and  $  k_{\Omega}(*,z) = k_{\Omega}(z,*) = 0  $ . Note that  $ k_{\Omega} $  is continuous on  $ \Omega \times \Omega $ . We denote the RKHS of  $ k_{\Omega} $  by  $ H_{\Omega} $ , where we have an embedding  $ H_{\Omega} \hookrightarrow H $  by definition.

Characteristic Kernels on  $ \Omega $ . Recall that a kernel  $ k: X \times X \to R $  (with associated feature map  $ \Phi: X \to H $ ) is characteristic with respect to probability measures  $ \mathcal{P}(\mathcal{X}) $  if the kernel mean embedding, also denoted by  $ \Phi: \mathcal{P}(\mathcal{X}) \to \mathcal{H} $ ,

 $$ \Phi(\mu):=\mathbb{E}_{x\sim\mu}[\Phi(x)], $$ 

is injective. The following result shows that if we start with a characteristic kernel on  $ [0, T]^{2} $ , the above procedure yields a characteristic kernel on  $ \Omega $ . This can be shown using similar methods as (Kusano et al., 2016, Section 3.1) in the current setting, but we provide an independent proof of a slightly stronger statement in Theorem 5 of Appendix B.

Theorem 1. Let  $ k:[0,T]^{2}\times[0,T]^{2}\toR $  be a kernel which is universal with respect to  $ C([0,T]^{2}) $  (or equivalently, characteristic with respect to  $ \mathcal{P}([0,T]^{2}) $ ). Then,  $ k_{\Omega}:\Omega\times\Omega\toR $  is characteristic with respect to  $ \mathcal{P}(\Omega) $ .

MMD for Principal Persistence Measures. A characteristic kernel  $ k_{\Omega} $  on  $ \Omega $  induces a metric on  $ \mathcal{P}(\Omega) $  via the norm, called the maximum mean discrepancy (MMD),

 $$ \mathbf{M M D}_{k}\big(\nu_{1},\nu_{2}\big):=\big\|\Phi\big(\nu_{1}\big)-\Phi\big(\nu_{2}\big)\big\|_{\mathcal{H}_{\Omega}}. $$ 

Let  $ \nu_{1}=\frac{1}{N}\left(\sum_{i=1}^{n}\delta_{x_{i}}+(N-n)\delta_{*}\right) $  and  $ \nu_{2}=\frac{1}{M}\left(\sum_{j=1}^{m}\delta_{y_{j}}+(M-m)\delta_{*}\right) $  be discrete measures in  $ \mathcal{P}(\Omega) $ , with n and m nontrivial points  $ x_{i}, y_{j} \in \Omega - \{*\} $  respectively. The MMD is given by

 $$ \mathbf{M M D}_{k}^{2}(\nu_{1},\nu_{2})=\frac{1}{N^{2}}\sum_{i,j=1}^{n}k_{\Omega}(x_{i},x_{j})-\frac{2}{N M}\sum_{i=1}^{n}\sum_{j=1}^{m}k_{\Omega}(x_{i},y_{j})+\frac{1}{M^{2}}\sum_{i,j=1}^{m}k_{\Omega}(y_{i},y_{j}). $$ 

The normalization is with respect to the total (including *) numbers of points in  $ \nu_{1} $  and  $ \nu_{2} $ , but we only compute kernels between nontrivial points since  $ k_{\Omega}(*,z)=k_{\Omega}(z,*)=0 $ . This enables the use of computable MMD metrics for PPMs. While the stability property in Equation (6) may no longer hold, MMD metrics yield the same topology on the space of probability measures (see also (Kusano et al., 2016, Theorem 3.2) in the finite setting). While a related result in a different context is given in (Divol & Lacombe, 2021, Proposition 5.1), we provide an independent proof in Appendix C.

Theorem 2. Let k be a characteristic kernel on  $ \Omega $ . The p-Wasserstein metric  $ W_{p} $  and the MMD metric  $ MMD_{k} $  induce the same topology on  $ \mathcal{P}(\Omega) $ .

Remark 1. By viewing persistence diagrams as measures (Divol & Lacombe, 2021; Giusti & Lee, 2023; Bubenik & Elchesen, 2022), persistence diagrams can be viewed as elements in  $ \mathcal{M}_{lin}(\Omega) $ , defined in Equation (30). This includes persistence diagrams with possibly infinite cardinality (with finite total persistence). While in Kusano et al. (2016), analogous kernels are defined for finite persistence diagrams, our results hold for persistence measures in  $ \mathcal{M}_{lin}(\Omega) $ .

## 5 TOPOLOGICAL REGULARIZATION WITH PPMs

In this section, we introduce our proposed topological regularizer based on computing the PPM of probability measures and comparing the PPMs using MMD. Let  $ k_{\Omega} $  be a characteristic kernel as defined in the previous section. Returning to the notation of Section 2, we define our dimension q topological regularizer, PPM-Reg, on the latent space  $ R^{L} $  by  $ \mathcal{T}_{q} : \mathcal{P}_{c}(\mathbb{R}^{L}) \times \mathcal{P}_{c}(\mathbb{R}^{L}) \to \mathbb{R} $  by

 $$ \mathcal{T}_{q}(\mu,\nu):=\mathbf{M M D}_{k_{\Omega}}(\mathbf{P P M}_{q}(\mu),\mathbf{P P M}_{q}(\nu))=\left\|\Phi_{\Omega}(\mathbf{P P M}_{q}(\mu))-\Phi_{\Omega}(\mathbf{P P M}_{q}(\nu))\right\|_{\mathcal{H}_{\Omega}}. $$ 

When we apply this in the GAN setting, we consider  $ \mathcal{T}_{q}\left(d_{\boldsymbol{\theta}}(\boldsymbol{\mu}),d_{\boldsymbol{\theta}}(g_{\boldsymbol{\omega}}(\boldsymbol{\nu}))\right) $ , where  $ \nu\in\mathcal{P}_{c}(\mathbb{R}^{N}) $  is the noise measure, and  $ \mu\in\mathcal{P}_{c}(\mathbb{R}^{M}) $  is the data measure. Let  $ \mathfrak{T}_{q}:\mathbb{R}^{G}\times\mathbb{R}^{D}\toR $  be

 $$ \mathfrak{T}_{q}(\omega,\pmb{\theta}):=\mathcal{T}_{q}\left(d_{\pmb{\theta}}(\mu),d_{\pmb{\theta}}(g_{\pmb{\omega}}(\nu))\right). $$ 

Our main result of this section is to show that  $ T_{q} $  is smooth with respect to  $ \omega $  and  $ \theta $  given sufficient smoothness conditions on the underlying measures and the discriminator and generator. Recall that a function  $ f : R^{n} \to R^{m} $  is a  $ C^{1} $  function if all first derivatives of f are continuous. The following is our main theoretical result, proved in Appendix D.

Theorem 3. Let  $ k_{\Omega} $  be a characteristic kernel. Suppose  $ \mu\in\mathcal{P}_{c}(\mathbb{R}^{M}) $  and  $ \nu\in\mathcal{P}_{c}(\mathbb{R}^{N}) $  have  $ C^{1} $  densities. Suppose the joint functions  $ G:\mathbb{R}^{G}\times\mathbb{R}^{N}\to\mathbb{R}^{M} $  defined by  $ G(\pmb{\omega},x)=g_{\pmb{\omega}}(x) $  and  $ D:\mathbb{R}^{D}\times\mathbb{R}^{M}\to\mathbb{R}^{L} $  defined by  $ D(\pmb{\theta},y)=d_{\pmb{\theta}}(y) $  be  $ C^{1} $  functions. Then,  $ T_{q} $  is a  $ C^{1} $  function wherever the PPM is not the trivial measure at the origin.

## 6 EXPERIMENTS AND RESULTS

We provide empirical experiments which demonstrates the efficacy of PPM-Reg as a topological regularizer. First, in Section 6.2, we provide an expository shape matching experiment to illustrate the behavior of PPM-Reg, and provide computational comparisons. Next, in Section 6.3, we apply PPM-Reg to a GAN-based generative modelling problem, consistently improving the generative quality of GANs. Finally, in Section 6.4, we consider a GAN-based semi-supervised learning problem, which demonstrates the effectiveness of PPM-Reg in improving the discriminative ability of GANs. Due to space limitations, we have placed implementation details and additional experiments for each of the three settings in Appendix E, Appendix F, and Appendix G. $ ^{4} $ 

### 6.1 COMPUTATIONAL SETUP AND IMPLEMENTATION OVERVIEW

Cramer Distance. The Wasserstein and Cramer distance are both probability metrics that are sensitive to the geometry of the change in distribution (Bellemare et al., 2017). Moreover, the Cramer distance does not depend on hyperparameters which simplifies our comparison. We primarily use the Cramer metric as our main loss function L. Following the definition of (Bellemare et al., 2017), for  $ \mu, \nu \in \mathcal{P}(\mathbb{R}^{d}) $ . The Cramer Distance  $ \mathcal{E}(\mu, \nu) $  is defined as

 $$ \mathcal{E}(\mu,\nu):=\mathbb{E}_{x\sim\mu}\big[\mathcal{D}(x)\big]-\mathbb{E}_{y\sim\nu}\big[\mathcal{D}(y)\big],\quad\mathcal{D}(z):=\mathbb{E}_{y^{\prime}\sim\nu}\big[\|z-y^{\prime}\|_{2}\big]-\mathbb{E}_{x^{\prime}\sim\mu}\big[\|z-x^{\prime}\|_{2}\big] $$ 

where  $ x, x^{\prime} $  (resp.  $ y, y^{\prime} $ ) are independent random variables with law  $ \mu $  (resp.  $ \nu $ ). We do not take the gradient estimation in (Bellemare et al., 2017, Appendix C.3) as we obtain sufficient samples.

Implementation of PPM. For all experiments, we use s subsamples from  $ \mu^{\otimes(2q+2)} $  to approximate the PPM (using the same number of subsamples for dimension 0 and 1). The persistent homology of each subsample is computed using Equation (4) in parallel on the GPU. Throughout these experiments, our base kernel is the radial basis function (RBF) kernel  $ k_{\mathrm{RBF}}(z_{1}, z_{2}) = \exp\left(-\|z_{1} - z_{2}\|^{2}/2\sigma\right) $  where the width  $ \sigma > 0 $  is a hyperparameter. Thus, the induced kernel  $ k_{\Omega} : \Omega \times \Omega \to R $  from Equation (9) evaluated on  $ z_{i} = (b_{i}, \ell_{i}) \in \Omega $  is

 $$ k_{\Omega}(z_{1},z_{2})=\ell_{1}\cdot\ell_{2}\exp\left(-\|z_{1}-z_{2}\|^{2}/2\sigma\right). $$ 

We use Equation (12) to compute the MMD metric between PPMs. Furthermore, we use a weighted combination of dimension 0 and 1 PPM in our topological regularizer, such that

 $$ \mathcal{T}=\lambda_{0}\mathcal{T}_{0}+\lambda_{1}\mathcal{T}_{1}, $$ 

where the weights  $ \lambda_{0},\lambda_{1}>0 $  are hyperparameters. We will call this PPM-Reg.

### 6.2 SHAPE MATCHING

<div style="text-align: center;"><img src="imgs/img_in_chart_box_728_787_915_958.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_957_417_1108.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_415_954_622_1107.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_620_954_817_1108.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_815_956_1009_1108.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">Figure 2: Visual example of PPM-Reg in a shape matching experiment using Cramer or MMD as the main loss function. 1st row: Plots of a reference point cloud (in blue) and the initial condition of a random point cloud (in orange). 2nd Row: Plots of 2-Wasserstein distance between 1-dimensional persistent homology between the reference shape and training shape over optimization steps.</div>


Our task is to optimize the individual points of a point cloud to match the “shape” of a reference point cloud using a loss function of the form  $ L + T $ , which operates directly on the ambient space of the point clouds. We choose L to either be the Cramer distance (Bellemare et al., 2017) or an MMD metric using an RBF kernel (with width  $ \sigma = 0.1 $ ). Our aim in this expository experiment is twofold.

1. Demonstrate the ability of PPM-Reg to regularize topological features in point clouds by comparing the true (non-subsampled) persistence diagrams of the trained and reference shapes. Our focus is on showing that this occurs near the beginning of the optimization, since in GAN settings, the regularization is important away from global minima (see Introduction).

2. Show the computational efficiency of PPM-Reg, which enables its use in later experiments.

Shape Matching Experiment. Our main results are summarized in Figure 2. We choose two reference shapes in  $ R^{2} $  for visualization purposes: a circle, and the union of two intersecting circles. In the second row of Figure 2, we plot the 2-Wasserstein distance between the 1-dimensional full (non-subsampled) persistence diagrams between the fixed reference and the trained point cloud, as a function of optimization steps. In each case, we see that adding PPM-Reg significantly reduces the topological distance. An interesting feature of each of these plots is that there is an initial spike in the PD distance when using PPM-Reg. Empirically, this is due to the fact that the point cloud must first move through a regime with trivial topological structure before PPM-Reg can faithfully match the topology of the reference. The training behavior is best understood by observing the dynamics of the optimization, and we provide animations of these experiments in the supplementary material.

Computational Comparisons. The efficiency of the PPM-Reg is derived from two major components: the parallelizable PPM and the iterative-free MMD. Table 1 empirically shows the computational benefit of each component as the size of the point cloud and the number of subsamples s are varied. We use Cramer as the main loss, and consider the computational cost of using PPM-Reg, W-PPM-Reg and PD-Reg. PD-Reg computes the 2-Wasserstein distance between dimension 0 and 1 full persistent homology with Vietoris-Rips filtration. W-PPM-Reg computes the 2-Wasserstein distance between PPMs of dimension 0 and 1. We use the torch-topological package (Lab) to compute persistent homology and Wasserstein distances.

Our aim is to compare the real-world usage of these methods using the circle experiment. Thus, PPM-Reg computations are performed on a GPU, while PD-Reg and W-PPM-Reg use hybrid CPU-GPU methods. The computational cost of PD-Reg grows exponentially with respect to the size of the point cloud. While the computational cost of W-PPM-Reg is sublinear with respect to the size of the point cloud, the cost is exponential with respect to the number of subsamples s. Remarkably, due to parallelization, PPM-Reg is sublinear with respect to the number of subsamples s and is nearly constant as the size of the point cloud increases. In the following experiments, we find that s = 1024 and s = 2048 performs well in practice. In summary, using MMD mediates the drawback of the increased number of features extracted by PPM, resulting in our significantly faster PPM-Reg. With parallelization, our pure PyTorch implementation of PPM-Reg outperforms highly optimized low-level CPU implementations used in torch-topological.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="4">Cramer + PPM-Reg</td><td colspan="4">Cramer + W-PPM-Reg</td><td rowspan="2">Cramer + PD-Reg</td></tr><tr><td style='text-align: center;'>No. points</td><td style='text-align: center;'>s = 512</td><td style='text-align: center;'>s = 1024</td><td style='text-align: center;'>s = 2048</td><td style='text-align: center;'>s = 512</td><td style='text-align: center;'>s = 1024</td><td style='text-align: center;'>s = 2048</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>128</td><td style='text-align: center;'>0.55  $ \pm $  0.005</td><td style='text-align: center;'>0.61  $ \pm $  0.007</td><td style='text-align: center;'>0.98  $ \pm $  0.004</td><td style='text-align: center;'>3.06  $ \pm $  0.033</td><td style='text-align: center;'>13.28  $ \pm $  0.198</td><td style='text-align: center;'>73.00  $ \pm $  3.323</td><td style='text-align: center;'>1.99  $ \pm $  0.048</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>256</td><td style='text-align: center;'>0.56  $ \pm $  0.008</td><td style='text-align: center;'>0.61  $ \pm $  0.005</td><td style='text-align: center;'>0.99  $ \pm $  0.005</td><td style='text-align: center;'>3.25  $ \pm $  0.083</td><td style='text-align: center;'>14.04  $ \pm $  0.187</td><td style='text-align: center;'>80.11  $ \pm $  2.545</td><td style='text-align: center;'>10.43  $ \pm $  0.075</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>512</td><td style='text-align: center;'>0.56  $ \pm $  0.010</td><td style='text-align: center;'>0.61  $ \pm $  0.014</td><td style='text-align: center;'>0.98  $ \pm $  0.006</td><td style='text-align: center;'>3.43  $ \pm $  0.111</td><td style='text-align: center;'>16.43  $ \pm $  0.458</td><td style='text-align: center;'>91.29  $ \pm $  3.081</td><td style='text-align: center;'>107.11  $ \pm $  2.837</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>1024</td><td style='text-align: center;'>0.57  $ \pm $  0.005</td><td style='text-align: center;'>0.61  $ \pm $  0.005</td><td style='text-align: center;'>1.00  $ \pm $  0.007</td><td style='text-align: center;'>3.90  $ \pm $  0.092</td><td style='text-align: center;'>19.45  $ \pm $  0.468</td><td style='text-align: center;'>121.76  $ \pm $  3.424</td><td style='text-align: center;'>655.58  $ \pm $  11.823</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr></table>

<div style="text-align: center;">Table 1: Running time of 100 gradient steps (in seconds) in matching circle form randomly initialize gaussian in  $ R^{2} $  with GPU computation enable. The averages are computed over 10 runs.</div>


Imperfect Convergence. In Appendix E.2, we observe the same trends in additional modified experiments, which prevent the centroid of the trained shape from converging to the centroid of the reference. This is done to mimic the GAN setting where training algorithms often converge to saddle points rather than global minima (Berard et al., 2019; Liang & Stokes, 2019).

### 6.3 UNCONDITIONAL IMAGE GENERATION

Next, we consider the use of PPM-Reg in an unconditional image generation task, which is the standard benchmark to evaluate GANs (Goodfellow et al., 2014; Arjovsky et al., 2017).

Network Architecture and Implementation Details. We use a ResNet based CNN as the generator  $ g_{\omega} $ , which takes a 128-dimensional noise vector as input. We use a CNN as the discriminator  $ d_{\theta} $  and the output of the network is a 128-dimensional latent vector. We compare the Cramer value function V = L, with the use of PPM-Reg  $ V = L + T $ . As our network differs from (Bellemare et al., 2017), we retrain both regularized and unregularized networks for a fair comparison.

3. WD $ _{latent} $ : 2-Wasserstein distance of CLIP embeddings (Radford et al., 2021)

Dataset and Evaluation Metrics. We consider the CelebA (Liu et al., 2015) and AnimeFace (Churchill & Chao, 2019) datasets. Images are centered and resized to  $ 32 \times 32 $ . While the Frechét Inception Distance (FID) (Heusel et al., 2017) is a popular metric to evaluate the distance between generated and real images, recent empirical work has thoroughly investigated several drawbacks of FID (Horak et al., 2021; Stein et al., 2024; Jayasumana et al., 2024). Instead, we adopt three metrics:

1. CMMD (Jayasumana et al., 2024): MMD of CLIP embeddings (Radford et al., 2021).

2.  $ FD_{Dinov2} $  (Stein et al., 2024): Frechét Distance of Dinov2 (Oquab et al., 2024) embeddings

<div style="text-align: center;">In Table 2, these are computed by sampling / generating 10K images from the data set / network. We report CMMD,  $ FD_{Dinov2} $  and  $ WD_{latent} $  using the epoch with the smallest CMMD.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="3">AnimeFace</td><td colspan="3">CelebA</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>$ FD_{\text{Dinov2}} $</td><td style='text-align: center;'>$ WD_{\text{latent}} $</td><td style='text-align: center;'>CMMD</td><td style='text-align: center;'>$ FD_{\text{Dinov2}} $</td><td style='text-align: center;'>$ WD_{\text{latent}} $</td></tr><tr><td style='text-align: center;'>Cramer (Bellemare et al., 2017)</td><td style='text-align: center;'>0.73</td><td style='text-align: center;'>953.99</td><td style='text-align: center;'>0.6294</td><td style='text-align: center;'>0.72</td><td style='text-align: center;'>722.86</td><td style='text-align: center;'>0.6795</td></tr><tr><td style='text-align: center;'>Cramer + PPM-Reg</td><td style='text-align: center;'>0.56</td><td style='text-align: center;'>780.68</td><td style='text-align: center;'>0.6080</td><td style='text-align: center;'>0.58</td><td style='text-align: center;'>700.73</td><td style='text-align: center;'>0.6666</td></tr></table>

<div style="text-align: center;">Table 2: Quantitative evaluation on  $ 32 \times 32 $  image generation, values are reported at the epoch with the smallest CMMD.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_253_700_514_885.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_515_704_741_886.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_742_702_967_886.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_251_883_511_1054.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_513_882_737_1054.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_740_884_963_1055.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">Figure 3: CMMD (a,d),  $ FD_{Dinov2} $  (b,e) and  $ WD_{latent} $  (c,f) versus training epochs for the AnimeFace (a-c) and CelebA (d-f) dataset. 10K samples are randomly generated to compute distances; moving averages with a window of 5 are used to smooth the values. Distances recorded every 160 epochs.</div>


Results. Figure 3 tracks these three metrics during training. Figure 3 (a,b,d,e) shows that using PPM-Reg improves image generation quality for both AnimeFace and CelebA. The Wasserstein distance can better detect geometric information in embedding space. Tracking  $ WD_{latent} $  in Figure 3 (c,f) shows that adding PPM-Reg provides more information and helps discover geometric structures in the latent space in an unsupervised way. This reinforces work that shows that persistence-based methods are able to effectively measure image generation quality (Zhou et al., 2021; Khrulkov & Oseledets, 2018; Barannikov et al., 2021; Charlier et al., 2019). As training progresses, improved Cramer loss does not always lead to improved evaluation metrics (Figure 3 (a,c)). Our reported results use CMMD as an early stopping criterion which is prohibitively expensive to compute in practice. In contrast, the evaluation metrics consistently decrease with respect to training time, and this implies that may not need to compute additional metrics for early stopping. In Appendix F.2, we consider larger  $ (64 \times 64) $  image generation experiments with the CelebA and LSUN Kitchen datasets, and find similarly improved results, demonstrating the efficacy of PPM-Reg in larger-scale experiments.

### 6.4 SEMI-SUPERVISED LEARNING

Semi-supervised learning (SSL) methods use unlabeled data alongside a small amount of labeled data to train a classification network (Yang et al., 2022). SSL often assumes that classification problems are supported on low-dimensional manifolds, which allows a network to learn the classification problem with limited labels (Niyogi, 2013). In practice, knowledge of the low-dimensional manifold can be learned by encoding the unlabeled data to latent representations. With few labeled data points, a simple classifier is trained using those latent representations (Wang et al., 2021a; Decourt & Duong, 2020; Truong et al., 2019; Das et al., 2021). Here, we demonstrate PPM-Reg can help encode more informative latent representations, and significantly reduce classification error in SSL.

Network Architecture and Implementation Details. We use a deconvolutional network as  $ g_{\omega} $ , which takes a 64 dimension noise vector as input. We use a CNN as  $ d_{\theta} $  and the output of the network is a 64 dimension latent vector. We use an MLP as a classifier parameterized by  $ \gamma $ , termed  $ c_{\gamma} $ . We first learn the latent representations using a Cramer GAN (Bellemare et al., 2017) framework with all available data, and compare it against the addition of PPM-Reg. After training the GAN, the discriminator  $ d_{\theta} $  is frozen and its output is used as the features to train the classifier  $ c_{\gamma} $  with the subset of training samples. For a comparison without latent representations learning, we consider a "Baseline", where  $ d_{\theta} $  and  $ c_{\gamma} $  are trained together as a classifier (without the generative part).

<div style="text-align: center;">Dataset and Evaluation Metrics. We compare the SSL performance with Fashion-MNIST (Xiao, 2017), Kuzushiji-MNIST (Clanuwat et al., 2018) and MNIST. In these experiments, 200 and 400 labels are randomly sampled from the data set. Due to the inherent randomness in sampling few labels, experiments are repeated ten times and the statistics of the best test-set accuracy are reported.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="2">Fashion-MNIST</td><td colspan="2">Kuzushiji-MNIST</td><td colspan="2">MNIST</td></tr><tr><td style='text-align: center;'>Number of labels</td><td style='text-align: center;'>200</td><td style='text-align: center;'>400</td><td style='text-align: center;'>200</td><td style='text-align: center;'>400</td><td style='text-align: center;'>200</td><td style='text-align: center;'>400</td></tr><tr><td style='text-align: center;'>Baseline</td><td style='text-align: center;'>67.18  $ \pm $  0.95</td><td style='text-align: center;'>71.00  $ \pm $  0.83</td><td style='text-align: center;'>48.40  $ \pm $  1.79</td><td style='text-align: center;'>55.10  $ \pm $  1.55</td><td style='text-align: center;'>80.52  $ \pm $  1.49</td><td style='text-align: center;'>86.39  $ \pm $  1.14</td></tr><tr><td style='text-align: center;'>Cramer</td><td style='text-align: center;'>62.70  $ \pm $  1.25</td><td style='text-align: center;'>68.58  $ \pm $  1.08</td><td style='text-align: center;'>47.77  $ \pm $  1.40</td><td style='text-align: center;'>56.06  $ \pm $  1.88</td><td style='text-align: center;'>71.10  $ \pm $  1.52</td><td style='text-align: center;'>78.26  $ \pm $  1.30</td></tr><tr><td style='text-align: center;'>Cramer + PPM-Reg</td><td style='text-align: center;'>76.84  $ \pm $  1.23</td><td style='text-align: center;'>80.59  $ \pm $  0.69</td><td style='text-align: center;'>75.78  $ \pm $  1.99</td><td style='text-align: center;'>79.33  $ \pm $  1.69</td><td style='text-align: center;'>96.62  $ \pm $  0.39</td><td style='text-align: center;'>97.33  $ \pm $  0.21</td></tr></table>

<div style="text-align: center;">Table 3: Test-set classification accuracy (\%) on Fashion-MNIST, Kuzushiji-MNIST and MNIST with 200 and 400 labeled examples. The average and the error bar are computed over 10 runs.</div>


Result. Table 3 shows the test classification accuracy, where we use only 0.33% (200) and 0.66% (400) of the total number of labels. Compared with Baseline, only using Cramer does not significantly improve the classification accuracy in SSL. Note that while the Cramer GAN (without PPM-Reg) has reasonable generative ability, shown in Figure 7, this does not imply strong discriminator performance in SSL. Remarkably, using PPM-Reg significantly improves the classification accuracy. For example, compared with the Baseline, Kuzushiji-MNIST has gain 27.38% improvement with 200 labels. Notably, with the latent representations learned with PPM-Reg, we can get a good accuracy using only 0.66% of the labels. This section demonstrates that discovering topological structures in latent space is not only useful in generative tasks, but can be leveraged to massively improve classification accuracy when very few labels are available. In Appendix G.2, we observe similar performance gains in additional experiments on the SVHN dataset.

## 7 CONCLUSION

In this article, we propose a novel method for stable and scalable topological regularization based on the subsampling principle of PPMs, opening up the possibility of detecting topological information in larger-scale machine learning problems. We introduced a theoretical framework for using kernel methods and MMD metrics for PPMs, and demonstrated the efficacy of this methodology in a variety of experimental settings. This work suggests several directions for future study. From the theoretical and computational perspective, can we develop parallelizable approximate computations in more general settings? From an applied perspective, how can we leverage approximate topological summaries in further machine learning tasks such as classification or regression?

## ACKNOWLEDGMENTS

This work was supported by the Hong Kong Innovation and Technology Commission (InnoHK Project CIMDA).