# DECOMPOSITION OF ONE-LAYER NEURAL NETWORKS VIA THE INFINITE SUM OF REPRODUCING KERNEL BANACH SPACES

Anonymous authors

Paper under double-blind review

## ABSTRACT

In this paper, we define the sum of RKBSs using the characterization theorem of RKBSs and show that the sum of RKBSs is compatible with the direct sum of feature spaces. Moreover, we decompose the integral RKBS  $ \mathcal{F}_{\sigma}(\mathcal{X},\Omega) $  into the sum of p-norm RKBSs  $ \{\mathcal{L}_{\sigma}^{1}(\mu_{i})\}_{i\in I} $ . Finally, we provide some applications to enhance the structural understanding of the integral RKBS class.

## 1 INTRODUCTION

To analyze the performance of neural networks, the hypothesis space represented by (infinite width) neural networks has been studied. Based on the concept of variation spaces (Kurková & Sanguineti, 2001; Mhaskar, 2004), Bach (2017) defined the  $ F_{1} $  spaces as an integral representation of the neural networks using the total variation norm. In subsequent, E & Stephan (2022) defined the Barron spaces employing the path norm and showed that the  $ F_{1} $  spaces and the Barron spaces can be isometrically isomorphic when using the Rectified Linear Unit (RELU) activation function.

The concept of Reproducing Kernel Banach Spaces (RKBSs) is a generalization of the Reproducing Kernel Hilbert Spaces (RKHSs), similar to how Banach spaces extend Hilbert spaces (Zhang et al., 2009). Relating to neural networks, Bartolucci et al. (2023) defined a class of integral RKBSs which are variants of the  $ F_{1} $  spaces. They defined a class of integral RKBSs through the characterization theorem of the RKBS introduced by Combettes et al. (2018) which describe an RKBS using a feature space and its associated feature map.

In this study, our primary focus is on a class of integral RKBSs. We aim to decompose this function space and identify its fundamental building blocks. Decomposing a function space entails preserving both its algebraic operations and topological properties. Since we are dealing with RKBS, we additionally need to ensure that the decomposition preserves the property that evaluation functionals remain continuous (see Definition 3.2). Considering the case of RKHS, there exists a sum of RKHSs that naturally extends the space in a canonical manner, resulting in an RKHS (Aronszajn, 1950). Using this approach, we aim to define the (potentially infinite) sum of RKBSs and investigate its relationship with the feature spaces.

The main questions of this paper are the following:

(1) Finding a natural definition for the sum of RKBSs that is compatible with the usual direct sum of Banach spaces.

(2) How can we decompose a class of integral RKBSs using the sum defined in question (1)?

To answer these questions, we define the sum of RKBSs, see Proposition 3.7, and show that the direct sum of the feature (Banach) spaces is compatible with the sum of RKBSs (Proposition 4.2). Roughly speaking, it is well-known that the space of the Radon measures can be decompose as the vast  $ l^{1} $  direct sum of  $ L^{1} $  spaces. As an analogue of the fact described above, we decompose the class of integral RKBSs using the sum of p-norm RKBSs (Theorem 4.4).

### 1.1 RELATED WORK

Before the era of neural networks, one of the main topics in machine learning was the kernel method, exemplified by concepts such as Reproducing Kernel Hilbert Spaces (RKHSs) and Support Vector Machines (Aronszajn, 1950; Steinwart & Christmann, 2008; Berlinet & Thomas-Agnan, 2011). Machine learning models that use RKHS as their hypothesis space are guaranteed the existence of a solution through the Representer Theorem. Additionally, an algorithm for explicitly finding this solution is clearly presented (Smola & Schölkopf, 1998; Shalev-Shwartz & Ben David, 2014). This characteristic significantly reduces the gap between theoretical understanding and practical application. One standard method for extending RKHS is through their sum, which plays a crucial role in enhancing the approximation ability of machine learning models. For instance, approaches like the multiple kernel algorithm demonstrate the utility of such extensions in effectively capturing diverse features of data (Yamanishi et al., 2004; Gönen & Alpaydın, 2011).

However, RKHS-based learning algorithms exhibit certain limitations due to their inner product structure. To address these challenges, the concept of Reproducing Kernel Banach Spaces (RKBSs) was introduced. Numerous studies have explored its theoretical foundations and applications (Zhang et al., 2009; Song et al., 2013; Fasshauer et al., 2015; Lin et al., 2022). Meanwhile, early theoretical research on neural networks primarily focused on approximation properties (Cybenko, 1989; Hornik et al., 1990; Barron, 1993). This line of inquiry led to further investigations into the hypothesis spaces of infinitely wide neural networks, culminating in the introduction of concepts such as Barron spaces and variation spaces (Bach, 2017; E et al., 2022; E & Stephan, 2022; Siegel & Xu, 2023).

Recent studies have attempted to analyze the hypothesis spaces of neural networks within the RKBS framework. For this purpose, the concept of integral RKBS has been introduced, which enables the proof of the Representer Theorem for one-layer neural networks (Bartolucci et al., 2023). However, unlike RKHS-based models, neural networks lack a clear algorithm for finding the solutions guaranteed by the Representer Theorem. In this study, we propose a method to decompose the hypothesis space of one-layer neural networks while preserving the RKBS structure. This approach enables a bottom-up exploration of the hypothesis space of one-layer neural networks, with the goal of contributing to the development of explicit algorithms for solutions guaranteed by the Representer Theorem in neural network settings.

### 1.2 ORGANIZATION

This paper is organized as follows. In Section 2, we briefly review the definitions and basic facts of the functional analysis. In Section 3, following Bartolucci et al. (2023); Spek et al. (2022), we introduce the definition of RKBSs and related function subclasses, namely a class of integral RKBSs and a class of p-norm RKBSs. We present some basic properties of these function classes, particularly focusing on the comparison between integral RKBSs and spaces of continuous functions (Proposition 3.5). Moreover, we define the sum of RKBSs, which is a modified version of Example 3.13 in Combettes et al. (2018) and the theorem in 353p of Aronszajn (1950), by using the characterization theorem of an RKBS. In Section 4, we state the main result of this article. We provide the compatibility between the sum of RKBSs and the direct sum of feature (Banach) spaces. Furthermore, using the compatibility (Proposition 4.2), we obtain that a class of integral RKBSs can be decomposed into the sum of p-norm RKBSs (Theorem 4.4). In Section 5, we provide direct applications of Theorem 4.4, showing how the size of the RKBS  $ \mathcal{F}_{\sigma}(\mathcal{X},\Omega) $  compares to the finite sum of p-norm RKHSs.

## 2 PRELIMINARIES AND NOTATIONS

In this paper, we denote $I$ as a non-empty index set and the set $\{1,\ldots,n\}$ is denoted by $[n]$. We consistently use $p$ and $q$ as conjugate indices, where $p$ satisfies $1 \leq p < \infty$. The data space is represented as $\mathcal{X}$, and the parameter space as $\Omega$. For convenience, we assume that $\mathcal{X}$ and $\Omega$ are compact subsets of $\mathbb{R}^{d}$ and $\mathbb{R}^{D}$ for some $d, D \in \mathbb{N}$, respectively. Furthermore, we use the notation $\cong$ to denote isomorphisms between vector spaces and the notation $\cong_{\mathcal{B}}$ to denote isometric isomorphisms between Banach spaces.

### 2.1 DIRECT SUM OF NORMED VECTOR SPACES

Let  $ \left\{a_{i}\right\}_{i\in I} $  be a family of elements in a Hausdorff commutative topological group (HCTG) H. Define F as the collection of all finite subsets of I, and order F by inclusion. Then F becomes a directed set. For each  $ F\inF $ , define  $ a_{F}:=\sum_{i\in F}a_{i} $ . Since F is a finite set,  $ a_{F} $  would be well-defined. Thus,  $ (a_{F})_{F\in\mathcal{F}} $  is a net in H. The family  $ \{a_{i}\}_{i\in I} $  is said to be summable if the net  $ (a_{F})_{F\in\mathcal{F}} $  converges. In this case, the limit is called the sum of the family  $ \{a_{i}\}_{i\in I} $ , and we denote it by  $ H\sum_{i\in I}a_{i} $ . When we consider sums in the norm topology of R, we use the term  $ \sum_{i\in I}a_{i} $  instead of  $ \sum_{i\in I}a_{i} $ . The contents related to the summable family in HCTG and R can be found in III §5 and IV §7 of Bourbaki (1971) respectively.

For a given index set  $ I \neq \emptyset $ , let  $ \{X_{i}\}_{i \in I} $  be a family of sets indexed by I. Then the direct product of the sets in  $ \{X_{i}\}_{i \in I} $  is defined by  $ \prod_{i \in I} X_{i} := \{\mathbf{x} : I \to \bigcup_{i \in I} X_{i} : \mathbf{x}(i) \in X_{i} \text{ for all } i \in I\} $ . When we assume that  $ X_{i} \neq \emptyset $  for all  $ i \in I $ , by the axiom of choice,  $ \prod_{i \in I} X_{i} $  is the non-empty set. In this case, for  $ j \in I $ , we can define  $ p_{j} : \prod_{i \in I} X_{i} \to X_{j} $  by  $ p_{j}(\mathbf{x}) = \mathbf{x}(j) $  for  $ x \in \prod_{i \in I} X_{i} $ . And we call  $ p_{j} $  is the j-th canonical projection. By abuse of notation, for any  $ x \in \prod_{i \in I} X_{i} $ , we denote x by  $ (x_{i})_{i \in I} $  which means  $ \mathbf{x}(j) = x_{j} \in X_{j} $  for all  $ j \in I $ . When  $ \{X_{i}\}_{i \in I} $  is a collection of R-vector spaces, the direct product of  $ \{X_{i}\}_{i \in I} $  is the vector space  $ \prod_{i \in I} X_{i} $  with componentwise addition and scalar multiplication. In this case, the canonical projections are linear maps. Furthermore, if  $ \{X_{i}\}_{i \in I} $  are topological spaces, then we can define the direct product of  $ \{X_{i}\}_{i \in I} $  by giving a topology on  $ \prod_{i \in I} X_{i} $ , called the product topology. Under this situation, the canonical projections are continuous maps.

Let $\{X_{i}: i \in I\}$ be a collection of normed vector spaces indexed by $I$. Then we can define the direct sum of the normed vector spaces $\{X_{i}: i \in I\}$ as follows:

Definition 2.1 (The direct sum of normed vector spaces (Conway, 1997)). For  $ 1 \leq p < \infty $ , we define

 $$ \bigoplus_{i\in I}^{p}X_{i}:=\left\{\mathbf{x}\in\prod_{i\in I}X_{i}:\left[\sum_{i\in I}\|\mathbf{x}(i)\|_{X_{i}}^{p}\right]^{\frac{1}{p}}<\infty\right\} $$ 

as a normed vector space equipped with the norm  $ \|\mathbf{x}\|_{\bigoplus_{i\in I}^{p}X_{i}}=\left[\sum_{i\in I}\|\mathbf{x}(i)\|_{X_{i}}^{p}\right]^{\frac{1}{p}} $ . For  $ p=\infty $ , we define

 $$ \bigoplus_{i\in I}^{\infty}X_{i}:=\left\{\mathbf{x}\in\prod_{i\in I}X_{i}:\sup_{i\in I}\|\mathbf{x}(i)\|_{X_{i}}<\infty\right\} $$ 

as a normed vector space equipped with the norm  $ \|\mathbf{x}\|_{\bigoplus_{i\in I}^{\infty}X_{i}}=\sup_{i\in I}\|\mathbf{x}(i)\|_{X_{i}} $ 

In particular, if each  $ X_{i} $  is a Banach space, then the direct sum of  $ \{X_{i}\}_{i\in I} $  is a Banach space. Let p and q be conjugate indices with  $ 1\leq p<\infty $ . We can obtain the following relationship between the duality and the direct sum (see III §5 Exercise 4 in Conway (1997)):

 $$ \Phi:\bigoplus_{i\in I}^{q}\left(X_{i}^{*}\right)\rightarrow\left(\bigoplus_{i\in I}^{p}X_{i}\right)^{*}\mathrm{a s}\Phi\left(\left(g_{i}\right)_{i\in I}\right)\left(f_{i}\right)_{i\in I}=\sum_{i\in I}\left\langle g_{i},f_{i}\right\rangle $$ 

for  $ (g_{i})_{i\in I}\in\bigoplus_{i\in I}^{q}(X_{i}^{*}) $  and  $ (f_{i})_{i\in I}\in\bigoplus_{i\in I}^{p}X_{i} $ . Then  $ \Phi $  is well-defined and it is an isometric isomorphism.

### 2.2 REVIEW OF MEASURE THEORY

Let K be a compact metric space. Then we know that the Borel and Baire  $ \sigma $ -algebra over K is coincide and every Borel measure on K is Radon measure (see Proposition 6.3.4 and Theorem 7.1.7 in Bogachev & Ruas (2007)). Let  $ C(K) $  be the Banach space consisting of continuous real-valued functions defined on K, equipped with the supremum norm. We denote by  $ \mathcal{M}(K) $  the Banach space of (signed) Borel measures defined on K, endowed with the total variation norm. Additionally, the set of positive measures in  $ \mathcal{M}(K) $  is denoted by  $ \mathcal{M}(K)^{+} $ , and the set of probability measures

in $\mathcal{M}(K)$ is denoted by $P(K)$. We know that by the Riesz Representation Theorem, there is an isometric isomorphism

 $$ \Lambda:\mathcal{M}(K)\to C(K)^{*}\mathrm{d e f i n e d b y}\Lambda(\mu)(f):=\int_{K}f d\mu\quad\mathrm{f o r}\mu\in\mathcal{M}(K)\mathrm{a n d}f\in C(K). $$ 

Let a measure space $(K,\Sigma,\mu)$ be given. Then, for $1\leq p\leq\infty$, we can define the Banach space $L^{p}(\mu)$ consisting of equivalence class of $p$-th power integrable functions with norm $\begin{cases}\|f\|_{p}=\left(\int_{K}|f|d\mu\right)^{1/p}<\infty,&if 1\leq p<\infty\\\|f\|_{\infty}=\operatorname{ess sup}|f|<\infty,&if p=\infty\end{cases}$. When $p$ and $q$ are conjugate indices with $1<p<\infty$, there is an isometric isomorphism.

 $$ \Xi:L^{q}(\mu)\to L^{p}(\mu)^{*}\mathrm{d e f i n e d b y}\Xi(g)(f):=\int_{K}f g d\mu\quad\mathrm{f o r}g\in L^{q}(\mu)\mathrm{a n d}f\in L^{p}(\mu). $$ 

It is also true for the case of  $ p = 1, q = \infty $  if the measure space  $ (K, \Sigma, \mu) $  is indeed  $ \sigma $ -finite. We use the notation  $ L^{p}(K, \mu) $  instead of  $ L^{p}(\mu) $  if there is a need to distinguish the domain space K.

A family F of measures in  $ \mathcal{M}(K)^{+} $  is called singular if  $ \mu \perp \nu $  whenever  $ \mu, \nu \in F $  and  $ \mu \neq \nu $  (see Definition 4.2.4 and Definition 4.6.1 in Dales et al. (2016)). Let S be a nonempty subset of  $ \mathcal{M}(K)^{+} $ . Then, by Zorn's lemma, there exists a maximal element in the set  $ \{A : A \subset S, A $  is a singular family in  $ \mathcal{M}(K)^{+}\} $ . This maximal element is called a maximal singular family in S. Let  $ \{\mu_{i}\}_{i \in I} $  be a maximal singular family in  $ P(K) $ . Then, there exists an isometric isomorphism

 $$ \Theta:\bigoplus_{i\in I}^{1}L^{1}(\mu_{i})\rightarrow\mathcal{M}(K)\mathrm{d e f i n e d b y}\Theta\left((f_{i})_{i\in I}\right)=\mathcal{M}(K)\sum_{i\in I}\rho_{i} $$ 

for  $ (f_{i})_{i\in I}\in\bigoplus_{i\in I}^{1}L^{1}(\mu_{i}) $ , where  $ \rho_{i}(B)=\int_{B}f_{i}d\mu_{i} $  for all  $ i\in I $  and Borel set B in K. (see Theorem 4.6.6 in Dales et al. (2016) and Proposition 4.3.8 in Albiac & Kalton (2016)). We use the notation  $ \Phi,\Lambda,\Xi $  and  $ \Theta $  liberally in situations that are isometrically isomorphic, as described above.

## 3 REPRODUCING KERNEL BANACH SPACES

### 3.1 DEFINITION OF RKBS

When we consider  $ R^{X} = \prod_{x \in X} R_{x} $ , where  $ R_{x} $  is just a copy of R for each  $ x \in X $ , there is a natural topological structure called the product topology. Equivalently, it is the initial topology with respect to the family of canonical projections  $ \{p_{x} : R^{X} \to R_{x}\}_{x \in X} $ . Since this topology is compatible with the vector space structure of  $ R^{X} $ ,  $ R^{X} $  becomes a Hausdorff topological vector space (HTVS). Thus, we may consider a summable family  $ (a_{i})_{i \in I} $  in  $ R^{X} $  and denote its sum in  $ R^{X} $  by  $ R^{X} \sum_{i \in I} a_{i} $  if it exists.

Let V be a linear subspace of  $ R^{X} $ . Then a topology on V induced by the product topology of  $ R^{X} $  again gives V the structure of a HTVS. Additionally, due to the transitivity of the initial topology, the subspace topology on V coincides with the initial topology induced by the family of restrictions  $ \{p_{x}|_{V}:V\toR_{x}\}_{x\inX} $ . We denote such a HTVS as  $ (V,\{p_{x}|_{V}\}_{x\in\mathcal{X}}) $ . (Relating reference can be found in Narici & Beckenstein (2010); Bogachev & Smolyanov (2017); Bourbaki (1971)). To distinguish between an index set I and the data set X, we use the term for the case of the latter as follows:

Definition 3.1. Let V be a linear subspace of  $ R^{X} $ . For each  $ x \in X $ , we use the term evaluation functional at  $ x \in X $  on V to refer to the restriction of the canonical projection  $ p_{x}|_{V}: V \to R_{x} $ , denoting it as  $ ev_{x} $ . Specifically, the function  $ ev_{x}: V \to R $  is a linear functional defined by  $ ev_{x}(f) = f(x) $  for all  $ f \in V $ .

Now we define a reproducing kernel Banach space on X as follows:

Definition 3.2 (Definition of reproducing kernel Banach space (Bartolucci et al., 2023; Lin et al., 2022)). For a given set X, a reproducing kernel Banach space (RKBS) B on X is a Banach space B of functions  $ f : X \to R $  such that

1. as a vector space, B is a linear subspace of  $ R^{\chi} $ 

2. for all  $ x \in X $ , there is a constant  $ C_{x} \geq 0 $  such that for all  $ f \in B $ ,  $ |f(x)| \leq C_{x} \|f\|_{\mathcal{B}} $ .

According to the definition, all evaluation functionals on B are continuous. In other words, we have that for all  $ x \in X $ ,  $ ev_{x} \in B^{*} $ . Therefore, the norm topology of an RKBS  $ (\mathcal{B}, \|\cdot\|_{\mathcal{B}}) $  is finer than the HTVS  $ (\mathcal{B}, \{ev_{x}\}_{x \in \mathcal{X}}) $ . Let  $ (\mathcal{B}, \|\cdot\|_{1}) $  and  $ (\mathcal{B}, \|\cdot\|_{2}) $  be two RKBSs on the same linear subspace B of  $ R^{X} $ . Then by the Closed Graph Theorem, two norms  $ \| \cdot \|_{1} $  and  $ \| \cdot \|_{2} $  on the linear space B is equivalent (see I §3 Exercise 2 in Bourbaki (1953) and Corollary IV $ _{1} $  in Aronszajn (1950)) $ ^{1} $ . In other words, when we have a function space B, we can give an unique RKBS structure on B up to equivalence of norms.

We will consider these RKBSs as hypothesis spaces in machine learning. The reason for using RKBS is as follows: When defining a hypothesis space (or function space) in machine learning, we consider completeness and pointwise convergence as the minimal assumptions required for the properties of the function space (see Chapter 1 in Berlinet & Thomas-Agnan (2011)).

### 3.2 CHARACTERIZATION OF RKBSs

Before we state the characterization theorem of RKBSs, we introduce a method that induces a mathematical structure from a pre-existing structure. Let $V$ be a normed vector space over $\mathbb{R}$ equipped with the norm $\|\cdot\|_{V}$, and let $W$ be a vector space over $\mathbb{R}$. If there is a vector space isomorphism $T:V\to W$, then $\|T^{-1}(\cdot)\|_{V}:W\to\mathbb{R}$ defines a norm on $W$. Furthermore, when we consider $W$ as a normed vector space equipped with the norm $\|T^{-1}(\cdot)\|_{V}$, the linear isomorphism $T:(V,\|\cdot\|_{V})\to(W,\|T^{-1}(\cdot)\|_{V})$ becomes an isometric isomorphism (This is referred to as the transport of structure).

Let V and W be vector spaces. If $T:V\to W$ is a linear map, then there exists an unique linear map $\hat{T}:V/\ker T\to W$ such that $\hat{T}\circ\pi=T$, where $\pi:V\to V/\ker T$ defined by $\pi(v)=[v]$ for $v\in V$. Throughout this paper, we use the notation $\hat{T}$ to denote the induced linear map described above in similar situations. We now state the characterization theorem of RKBSs introduced by Combettes et al. (2018).

Theorem 3.3 (Characterization of RKBSs (Bartolucci et al., 2023; Combettes et al., 2018)). A linear subspace $\mathcal{B}$ of $\mathbb{R}^{\mathcal{X}}$ is an RKBS on $\mathcal{X}$ if and only if there exists a Banach space $\Psi$ and a map $\psi : \mathcal{X} \to \Psi^{*}$ such that $\mathcal{B} = \operatorname{im}(A) = \{f : \exists \nu \in \Psi \text{ s.t.} A(\nu) = f\}$ with the norm $\|f\|_{\mathcal{B}} = \inf_{\nu \in A^{-1}(f)} \|\nu\|_{\Psi}$,

where $A:\Psi\to\mathbb{R}^{\mathcal{X}}$ is a linear map defined by $(A\nu)(x):=\langle\psi(x),\nu\rangle$ for $x\in\mathcal{X}$ and $\nu\in\Psi$.

Note that the linear map $A$ is the linear transformation induced from the family of the linear maps $\{ \psi(x):\Psi\to\mathbb{R}_{x}\}_{x\in X}$ by the universal property of the direct product of the vector spaces $\{ \mathbb{R}_{x}\}_{x\in\mathcal{X}}$. We briefly review the proof provided in Bartolucci et al. (2023). In the necessity part of the proof, it is shown that $\ker A$ is closed in $\Psi$ by the following equations:



 $$ \ker(A)=\{\nu\in\Psi:\psi(x)(\nu)=0for all x\in\mathcal{X}\}=\bigcap_{x\in X}\ker\psi(x). $$ 

Thus, $\Psi/\ker A$ can be a Banach space with the quotient norm. Consider the linear map $\hat{A}:\Psi/\ker A\to\mathbb{R}^{\mathcal{X}}$ such that $A=\hat{A}\circ\pi$. Since $\hat{A}:\Psi/\ker A\cong\mathrm{im}(A)$ is an isomorphism of vector spaces, by the transport of the structure, $\mathcal{B}=\mathrm{im}(A)$ becomes a Banach space with the norm:

 $$ \|f\|_{\mathcal{B}}=\|\hat{A}^{-1}(f)\|_{\Psi/\ker(A)}=\inf_{\nu\in\pi^{-1}(\hat{A}^{-1}(f))}\|\nu\|_{\Psi}=\inf_{\nu\in A^{-1}(f)}\|\nu\|_{\Psi} $$ 

The evaluation functionals are continuous as follows: for any $f \in \mathcal{B}$ and $\nu \in A^{-1}(f)$, we have

$|f(x)| = |A\nu(x)| \leq \|\psi(x)\|_{\Psi^*} \|\nu\|_{\Psi}$. Thus, we can deduce that for all $x \in \mathcal{X}$,

 $$ \|e v_{x}(f)\|_{\mathbb{R}}=|f(x)|\leq\|\psi(x)\|_{\Psi^{*}}\inf_{\nu\in A^{-1}(f)}\|\nu\|_{\Psi}=\|\psi(x)\|_{\Psi^{*}}\|f\|_{\mathcal{B}}. $$ 

From now on, for a given RKBS B, we consider a corresponding space  $ \Psi $ , a map  $ \psi $  and an induced linear map A. In this situation, by abuse of notation, we may say that an RKBS triple  $ \mathcal{B} = (\Psi, \psi, A) $  is given. Each component of the triple  $ (\Psi, \psi, A) $  has a specific name. Specifically, we refer to  $ \Psi $  as a feature space,  $ \psi $  as a feature map, and A as an RKBS map in order.

### 3.3 ONE-LAYER NEURAL NETWORKS

In this subsection, we assume that  $ \Omega_{1}\subsetR^{d} $  and  $ \Omega_{2}\subsetR $  are compact, and let  $ \Omega=\Omega_{1}\times\Omega_{2} $ . Consider a continuous nonlinear function  $ g:R\toR $ . The prediction function represented by a one-layer neural network with a one-dimensional target can be expressed as follows:

 $$ f(x)=\sum_{i=1}^{m}\eta_{i}g(x\cdot\theta_{i}-b_{i}), $$ 

where  $ x \in X $ ,  $ \theta_{i} \in \Omega_{1} $ ,  $ b_{i} \in \Omega_{2} $  and  $ \eta_{i} \in R $  for  $ i = 1, \ldots, m $ . For convention, and with some abuse of notation, we define a continuous function  $ \sigma : X \times \Omega \to R $  by  $ \sigma(x, w) = g(x \cdot \theta - b) $  where  $ w = (\theta, b) $ . This gives the following simplified representation:  $ f(x) = \sum_{i=1}^{m} \eta_{i} \sigma(x, w_{i}) $ . Using measure-theoretic notation, we can have an integral representation of the equation 3.3:  $ f(x) = \int_{\Omega} \sigma(x, w) d\left(\sum_{i=1}^{m} \eta_{i} \delta_{w_{i}}\right) $  where  $ \delta_{w_{i}} $  is the Dirac measure at  $ w_{i} $ . When considering the limit as  $ m \to \infty $  in equation 3.3, we obtain the following:

 $$ \int_{\Omega}\sigma(x,w)d\left(\sum_{i=1}^{m}\eta_{i}\delta_{w_{i}}\right)\rightarrow\int_{\Omega}\sigma(x,w)d\mu(w), $$ 

for some  $ \mu\in\mathcal{M}(\Omega) $ . A more detailed explanation can be found in Chapter 9 of Bach (2024). In the following subsection, we will define the hypothesis space of one-layer neural networks in a more abstract way using this relaxed expression.

### 3.4 INTEGRAL RKBS AND P-NORM RKBS

Directly using the characterization theorem, we can define the hypothesis spaces that are considered to represent one-layer neural networks. Until section 4, we consider a fixed element  $ \sigma $  in  $ C(\mathcal{X} \times \Omega) $ , where X is a compact subset of  $ R^{d} $  and  $ \Omega $  is a compact subset of  $ R^{D} $  for some  $ d, D \in N $ .

Let V and W be a real normed vector spaces. If we denote  $ V^{**} $  be the bidual space of V, then there is a linear isometric embedding  $ \iota:V\to V^{**} $ , called the canonical embedding of V in  $ V^{**} $ , defined by  $ \iota(v)(v^{*})=v^{*}(v) $  for  $ v\in V $  and  $ v^{*}\in V^{*} $ . For a given bounded linear operator  $ T:V\to W $ , the dual operator of T is the linear operator  $ T^{*}:W^{*}\to V^{*} $  defined by  $ T^{*}(w^{*}):=w^{*}\circ T $  for  $ w^{*}\in W^{*} $ .

Definition 3.4 (A class of integral RKBSs, associated with the function  $ \sigma $  (Bartolucci et al., 2023; Spek et al., 2022)). Let  $ \mathcal{M}(\Omega) $  be a feature space. Consider a feature map  $ \psi: X \to \mathcal{M}(\Omega)^{*}(\cong_{\mathcal{B}} C(\Omega)^{**}) $  defined by  $ \psi(x) = \Lambda^{*}(\iota(\sigma(x,\cdot))) $  for all  $ x \in X $ , where  $ \iota: C(\Omega) \to C(\Omega)^{**} $  is the canonical embedding of  $ C(\Omega) $  in  $ C(\Omega)^{**} $  and  $ \Lambda^{*} $  is the dual operator of  $ \Lambda: \mathcal{M}(\Omega) \to C(\Omega)^{*} $ , which is defined in equation 2.2. Then there is a linear map  $ A: \mathcal{M}(\Omega) \to \mathbb{R}^{\mathcal{X}} $  defined by  $ (A\mu)(x) = \langle\psi(x),\mu\rangle = \int_{\Omega}\sigma(x,w)d\mu(w) $  for  $ x \in X $  and  $ \mu \in \mathcal{M}(\Omega) $ . An integral RKBS  $ \mathcal{F}_{\sigma}(\mathcal{X},\Omega) $ , associated with the function  $ \sigma $  is defined by the Banach space

 $$ \mathcal{F}_{\sigma}(\mathcal{X},\Omega):=\left\{f\in\mathbb{R}^{\mathcal{X}}:\exists\mu\in\mathcal{M}(\Omega)\;s.t.\;\forall x\in\mathcal{X},f(x)=\int_{\Omega}\sigma(x,w)d\mu(w)\right\}, $$ 

equipped with the norm  $ \|f\|_{\mathcal{F}_{\sigma}(X,\Omega)}=\inf_{\mu\in A^{-1}(f)}\|\mu\|_{\mathcal{M}(\Omega)} $ 

In the above Definition 3.4, consider the linear map  $ A : \mathcal{M}(\Omega) \to \mathbb{R}^{\mathcal{X}} $ . We deduce that, by the Dominated Convergence Theorem,  $ \operatorname{im}(A) $  is a linear subspace of  $ C(\mathcal{X}) $  (see Theorem 2.27 in Folland (1999)). Furthermore, from the inequality  $ \|A\mu\|_{C(\mathcal{X})} \leq \sup_{x \in \mathcal{X}, w \in \Omega} |\sigma(x, w)| \|\mu\| $ , we can see that the map  $ A : \mathcal{M}(\Omega) \to C(\mathcal{X}) $  is indeed a bounded operator. Recently, Steinwart showed that when X is an uncountable compact metric space, there is no RKHS H on X such that  $ C(\mathcal{X}) \subset \mathcal{H} $  (Steinwart, 2024). We can obtain a similar result for the class of integral RKBS as well.

Proposition 3.5. The bounded operator  $  A : \mathcal{M}(\Omega) \to C(\mathcal{X})  $  defined by

 $$ (A\mu)(x)=\int_{\Omega}\sigma(x,w)d\mu(w) $$ 

for  $ x \in X $  and  $ \mu \in \mathcal{M}(\Omega) $  is compact.

Using the proposition above, it follows that if  $ \operatorname{im}(A) $  is closed in  $ C(\mathcal{X}) $ , then  $ \operatorname{im}(A) $  has finite dimension. Thus, when X is an infinite compact metric space, we deduce that  $ \mathcal{F}_{\sigma}(\mathcal{X},\Omega)\subsetneq\mathcal{C}(\mathcal{X}) $  and in general,  $ \mathcal{F}_{\sigma}(\mathcal{X},\Omega) $  cannot be a Banach space if it equipped with the supremum norm.

Definition 3.6 (A class of p-Norm RKBS, associated with the function  $ \sigma $  (Spek et al., 2022)). Let  $ \pi \in P(\Omega) $  be given. Let p and q be conjugate indices such that  $ 1 \leq p < \infty $ . Take a feature space  $ \Psi $  as  $ L^{p}(\pi) $  and choose a feature map  $ \psi : X \to (L^{p}(\pi))^{*} $  defined by  $ \psi(x) = \Xi(\sigma(x,\cdot)) $  for  $ x \in X $ , where  $ \Xi : L^{q}(\pi) \to (L^{p}(\pi))^{*} $  is the isometric isomorphism defined in equation 2.3. Then, there is a linear map  $ A : L^{p}(\pi) \to R^{X} $  defined by  $ (Ah)(x) = \langle \psi(x), h \rangle $  for  $ x \in X $  and  $ h \in L^{p}(\pi) $ . We define a p-Norm RKBS  $ \mathcal{L}_{\sigma}^{p}(\pi) $ , associated with the function  $ \sigma $  by the Banach space

 $$ \mathcal{L}_{\sigma}^{p}(\pi):=\left\{f\in\mathbb{R}^{\mathcal{X}}:\exists h\in L^{p}(\pi)\;s.t.\;\forall x\in\mathcal{X},f(x)=\int_{\Omega}\sigma(x,w)h(w)d\pi(w)\right\}, $$ 

equipped with the norm  $ \|f\|_{\mathcal{L}_{\sigma}^{p}(\pi)}=\inf_{h\in A^{-1}(f)}\|h\|_{L^{p}(\pi)} $ 

When we consider the p = 2 case, we obtain the RKHS  $ \mathcal{L}_{\sigma}^{2}(\pi) $ . This space corresponds to  $ F_{2} $  as described in Bach (2017). The kernel of  $ \mathcal{L}_{\sigma}^{2}(\pi) $  is given by  $ k(x, y) = \int_{\Omega} \sigma(x, w) \sigma(y, w) d\pi(w) $  for  $ (x, y) \in \mathcal{X} \times \mathcal{X} $ . Furthermore,  $ \mathcal{L}_{\sigma}^{2}(\pi) $  is embedded in  $ \mathcal{L}_{\sigma}^{1}(\pi) $  (that is,  $ \mathcal{L}_{\sigma}^{2}(\pi) \subset \mathcal{L}_{\sigma}^{1}(\pi) $ ) and for all  $ f \in \mathcal{L}_{\sigma}^{2}(\pi) $ ,  $ \|f\|_{\mathcal{L}_{\sigma}^{1}(\pi)} \leq \|f\|_{\mathcal{L}_{\sigma}^{2}(\pi)} $ . As an analogue to the case of  $ L^{p} $  space, we sometimes use the notation  $ \mathcal{L}_{\sigma}^{p}(\Omega, \pi) $  instead of  $ \mathcal{L}_{\sigma}^{p}(\pi) $  to avoid confusion.

### 3.5 INFINITE SUM OF REPRODUCING KERNEL BANACH SPACES

Let an RKBS B be given. If we consider an evaluation functional on B evaluating at  $ x \in X $  by  $ ev_{x} : B \to R $ , as discussed earlier, then we have that for all  $ x \in X $ ,  $ ev_{x} \in B^{*} $ . Thus, if we assume that a collection of RKBSs  $ \{B_{i}\}_{i \in I} $  is given and denote  $ ev_{x}^{i} $  as the evaluation functional on  $ B_{i} $  evaluating at  $ x \in X $ , then for all  $ i \in I $  and  $ x \in X $ ,  $ ev_{x}^{i} \in B_{i}^{*} $ . Now, we define the sum of RKBSs as follows, modifying Example 3.13 in Combettes et al. (2018) and the theorem on page 353 of Aronszajn (1950):

Proposition 3.7 (Infinite sum of reproducing kernel Banach spaces). Let p and q be conjugate indices with  $ 1 \leq p < \infty $ . Let  $ \{B_{i}\}_{i \in I} $  be a collection of RKBSs on X. Suppose that for all  $ x \in X $ ,  $ (ev_{x}^{i})_{i \in I} \in \bigoplus_{i \in I}^{q} B_{i}^{*} $ . Let  $ \bigoplus_{i \in I}^{p} B_{i} $  be a feature space and define a feature map  $ s : X \to \bigoplus_{i \in I}^{p} B_{i} $  by  $ \mathbf{s}(x) = \Phi((ev_{x}^{i})_{i \in I}) $  for  $ x \in X $ , where  $ \Phi : \bigoplus_{i \in I}^{q} B_{i}^{*} \to \bigoplus_{i \in I}^{p} B_{i} $  is the isometric isomorphism defined in equation 2.1. Then there is a linear map  $ S : \bigoplus_{i \in I}^{p} B_{i} \to R^{X} $  defined by  $ (\mathcal{S}(f_{i})_{i \in I}) (x) = \langle \mathbf{s}(x), (f_{i})_{i \in I} \rangle $  for  $ (f_{i})_{i \in I} \in \bigoplus_{i \in I}^{p} B_{i} $  and  $ x \in X $ . By the Theorem 3.3, we can define an RKBS  $ \mathcal{B} = \operatorname{Im}(\mathcal{S}) = \{\mathbb{R}^{X} \sum_{i \in I} f_{i} : (f_{i})_{i \in I} \in \bigoplus_{i \in I}^{p} B_{i}\} $  equipped with the norm  $ \|f\|_{\mathcal{B}} = \operatorname{inf}_{(f_{i})_{i \in I} \in \mathcal{S}^{-1}(f)} \|(f_{i})_{i \in I}\|_{\bigoplus_{i \in I}^{p} B_{i}} = \inf_{f =_{\mathbb{R}^{X}} \sum_{i \in I} f_{i}} \|(f_{i})_{i \in I}\|_{\bigoplus_{i \in I}^{p} B_{i}} $ .

Note that the property of net in the initial topology implies that  $ f =_{R^{X}} \sum_{i \in I} f_{i} $  in  $ (\mathbb{R}^{\mathcal{X}}, \{p_{x}\}_{x \in \mathcal{X}}) $  is equivalent to  $ f(x) = \sum_{i \in I} f_{i}(x) $  for all  $ x \in X $ . Thus, we have that

 $$ \begin{align*}\mathcal{B}&=\left\{f\in\mathbb{R}^{\mathcal{X}}:\exists(f_{i})_{i\in I}\in\bigoplus_{i\in I}^{p}\mathcal{B}_{i}\text{s.t.}\forall x\in\mathcal{X},f(x)=\sum_{i\in I}\left\langle e v_{x}^{i},f_{i}\right\rangle\right\}\\&=\left\{\mathbb{R}^{x}\sum_{i\in I}f_{i}:(f_{i})_{i\in I}\in\bigoplus_{i\in I}^{p}\mathcal{B}_{i}\right\}.\end{align*} $$ 

From now on, we denote B mentioned in the Proposition 3.7 by  $ \sum_{i\in I}^{p}B_{i} $  and call it the sum of the family of RKBSs  $ \{B_{i}\}_{i\in I} $ . In particular, for the case of p=1, we denote B as  $ \sum_{i\in I}B_{i} $ . In the Proposition 3.7, we intentionally use the notation s for the feature map and S for the RKBS map to emphasize that they are used to represent the sum of RKBSs. Moreover, we denote the RKBS triple of the sum of RKBSs by  $ \sum_{i\in I}^{p}\mathcal{B}_{i}=(\bigoplus_{i\in I}^{p}\mathcal{B}_{i},\mathbf{s},\mathcal{S}) $  using s and S.

Remark 3.8. Let a family of RKBS triples $\{\mathcal{B}_{i}=(\Psi_{i},\psi_{i},A_{i})\}_{i\in I}$ be given. From the equation 3.2, we know that $\|ev_{x}^{i}\|_{B_{i}^{*}}\leq\|\psi_{i}(x)\|_{\Psi_{i}^{*}}$ for all $x\in\mathcal{X}$ and $i\in I$. Thus, instead of assuming $(ev_{x}^{i})_{i\in I}\in\bigoplus_{i\in I}^{q}\mathcal{B}_{i}^{*}$ for all $x\in\mathcal{X}$, it suffices to assume that $(\psi_{i}(x))_{i\in I}\in\bigoplus_{i\in I}^{q}\Psi_{i}^{*}$ for all $x\in\mathcal{X}$.

## 4 MAIN RESULTS

### 4.1 COMPATIBILITY BETWEEN THE SUM OF RKBSS AND THE DIRECT SUM OF FEATURE SPACES

In this section, we present the compatibility between the sum of RKBSs and the direct sum of feature spaces. Before stating our main proposition, we prove the following lemma, which says that the restriction to the direct sum of Banach spaces of the product of (quotient, isometry) maps preserves their properties.

Lemma 4.1. Suppose  $ 1 \leq p < \infty $ , and let a family of Banach spaces  $ \{X_{i}\}_{i \in I} $  be given.

1. Suppose that for each  $ i \in I $ ,  $ D_i $  is a closed linear subspace of  $ X_i $ , and let  $ \pi_i : X_i \to X_i / D_i $  be the projection map defined by  $ \pi_i(x_i) := [x_i] $  for  $ x_i \in X_i $ . Then, the map  $ (\pi_i)_{i \in I} : \bigoplus_{i \in I}^p X_i \to \bigoplus_{i \in I}^p X_i / D_i $  defined by  $ (\pi_i)_{i \in I} ((x_i)_{i \in I}) = (\pi_i(x_i))_{i \in I} $  for  $ (x_i)_{i \in I} \in \bigoplus_{i \in I}^p X_i $  is a surjective bounded linear operator.

2. Assume there is another family of Banach spaces  $ \{Y_{i}\}_{i\in I} $ . If for each  $ i\in I $ , there is an isometric isomorphism  $ \phi_{i}:X_{i}\to Y_{i} $ , then the map  $ (\phi_{i})_{i\in I}:\bigoplus_{i\in I}^{p}X_{i}\to\bigoplus_{i\in I}^{p}Y_{i} $  defined by  $ (\phi_{i})_{i\in I}\left((x_{i})_{i\in I}\right)=(\phi_{i}(x_{i}))_{i\in I} $  for  $ (x_{i})_{i\in I}\in\bigoplus_{i\in I}^{p}X_{i} $  is an isometric isomorphism.

The following proposition is one of the main result of this paper. It states that when we have a family of RKBSs, there is an RKBS induced by the direct sum of feature spaces, which is isometrically isomorphic to the sum of the given family of RKBSs. Conversely, when we have an RKBS induced by the direct sum of feature spaces, there is a collection of RKBSs such that their sum is isometrically isomorphic to the given RKBS.

Proposition 4.2 (Compatibility). Let  $ I \neq \emptyset $  be an index set. Let p and q be conjugate indices, where p satisfies  $ 1 \leq p < \infty $ .

1. Suppose a family of RKBS triples  $ \{\mathcal{B}_{i} = (\Psi_{i}, \psi_{i}, A_{i})\}_{i \in I} $  is given and  $ (ev_{x}^{i})_{i \in I} \in \bigoplus_{i \in I}^{q} \mathcal{B}_{i}^{*} $  for all  $ x \in X $ . Then, there is an RKBS triple  $ \mathcal{B} = (\bigoplus_{i \in I}^{p} \Psi_{i}, \psi, A) $  such that  $ \mathcal{B} \cong \sum_{i \in I}^{p} \mathcal{B}_{i} $ .

2. For an RKBS triple  $ \mathcal{B} = (\bigoplus_{i \in I}^{p} \Psi_{i}, \psi, A) $ , there is a family of reproducing kernel Banach spaces  $ \{\mathcal{B}_{i \in I}\}_{i \in I} $  such that  $ \mathcal{B} \cong \sum_{i \in I}^{p} \mathcal{B}_{i} $ .

<div style="text-align: center;"><img src="imgs/img_in_image_box_353_1008_867_1157.jpg" alt="Image" width="41%" /></div>


<div style="text-align: center;">Figure 1: Commutative diagram for the compatibility</div>


The diagram above intuitively illustrates the result we aim to demonstrate in Proposition 4.2. Detailed information about each map can be found in Appendix A.4.

### 4.2 DECOMPOSITION OF ONE-LAYER NEURAL NETWORKS

The following lemma shows that if there is an isometrically isomorphic feature space, then we can construct an isometrically isomorphic RKBS.

Lemma 4.3. Let  $ \Psi_{1} $  be a Banach space and let  $ \mathcal{B}_{2} = (\Psi_{2}, \psi_{2}, A_{2}) $  be an RKBS triple. If there is an isomeric isomorphism  $ \xi : \Psi_{1} \to \Psi_{2} $ , then there is an RKBS triple  $ \mathcal{B}_{1} = (\Psi_{1}, \psi_{1}, A_{1}) $  such that  $ B_{1} \cong_{B} B_{2} $ .

Now, we introduce our main theorem. It states that the integral RKBS  $ \mathcal{F}_{\sigma}(\mathcal{X},\Omega) $  defined in the Definition 3.4 can be decomposed into the sum of a family of p-norm RKBSs  $ \{\mathcal{L}_{\sigma}^{1}(\mu_{i})\}_{i\in I} $  defined in the Definition 3.6, where  $ \{\mu_{i}\}_{i\in I} $  is a maximal singular family in  $ P(\Omega) $ .

Theorem 4.4. Let $\{\mu_{i}\}_{i\in I}$ be a maximal singular family in $P(\Omega)$. Then, we have the following:

 $$ \mathcal{F}_{\sigma}(\mathcal{X},\Omega)\cong\underset{\mathcal{B}}{\overset{\simeq}{\sum}}\sum_{i\in I}\mathcal{L}_{\sigma}^{1}(\mu_{i}). $$ 

Remark 4.5. In the proof of the Theorem 4.4, we can see that the following set equality holds:

 $$ \mathcal{F}_{\sigma}(\mathcal{X},\Omega)=\sum_{i\in I}\mathcal{L}_{\sigma}^{1}(\mu_{i}). $$ 

Furthermore, since  $ \Omega $  is a compact metric space in our setting,  $ \mathcal{L}_{\sigma}^{1}(\mu_{i}) $  is a separable RKBS for all  $ i \in I $ . Thus, we decompose the integral RKBS  $ \mathcal{F}_{\sigma}(\mathcal{X}, \Omega) $  into infinitely many separable RKBSs.

## 5 APPLICATION

Let $\{ \mu_{i} \}_{i \in [n]}$ be any finite family in $P(\Omega)$. Since for each $i \in [n]$, $k_{i}(x, y) = \int_{\Omega} \sigma(x, w) \sigma(y, w) d\mu_{i}$ for $(x, y) \in \mathcal{X} \times \mathcal{X}$ is the reproducing kernel of $\mathcal{L}_{\sigma}^{2}(\mu_{i})$, the sum kernel $k(x, y) = \sum_{i=1}^{n} k_{i}(x, y)$ is the reproducing kernel of $\sum_{i \in [n]} \mathcal{L}_{\sigma}^{2}(\mu_{i})$ (The notation $\sum_{i \in [n]}^{2}$ refers to the case where we defined it in Proposition 3.7 with $p = 2$ and $I = [n]$. In this setting, we can guarantee that $\sum_{i \in [n]} \mathcal{L}_{\sigma}^{2}(\mu_{i})$ is an RKHS. The following proposition shows that when we consider the finite singular family in $P(\Omega)$, the RKHS $\sum_{i \in [n]} \mathcal{L}_{\sigma}^{2}(\mu_{i})$ contained in the RKBS $\mathcal{F}_{\sigma}(\mathcal{X}, \Omega)$ with the same associated function $\sigma$.

Proposition 5.1. For any finite singular family  $ \{\mu_{i}\}_{i\in[n]} $  in  $ P(\Omega) $ , we have

 $$ \sum_{i\in[n]}^{2}\mathcal{L}_{\sigma}^{2}(\mu_{i})\subset\mathcal{F}_{\sigma}(\mathcal{X},\Omega). $$ 

Let a family of continuous functions  $ \{\sigma_{i}:X\times\Omega\toR\}_{i=1}^{n} $  be given. By the Tietze extension theorem and the pasting lemma, there is a continuous function  $ \sigma:X\times\Omega\times[0,1]\toR $  which is an extension of the continuous function  $ \tilde{\sigma}:X\times\Omega\times\left\{\frac{1}{n},\ldots,\frac{n-1}{n},1\right\}\toR $  defined by  $ \tilde{\sigma}(x,w,\frac{i}{n})=\sigma_{i}(x,w) $  for all  $ i=1,\ldots,n $ ,  $ x\inX $  and  $ w\in\Omega $ . In the following proposition, we show that the finite sum of p-norm RKBSs associated with different functions is contained in the integral RKBS associated with a suitable function when considering a larger parameter space. This means that the class of integral RKBSs is quite large due to its flexibility in choosing the dimension of the parameter space.

Proposition 5.2. Let a family of continuous functions  $ \{\sigma_{i}:X\times\Omega\toR\}_{i=1}^{n} $  be given. Let  $ \{\pi_{i}\}_{i=1}^{n} $  be a collection of probability measures in  $ \Omega $ . Then, we have

 $$ \sum_{i\in[n]}^{2}\mathcal{L}_{\sigma_{i}}^{2}(\Omega,\pi_{i})\subset\mathcal{F}_{\sigma}(\mathcal{X},\Omega\times[0,1]). $$ 

Remark 5.3. For the purpose of a realistic application, we consider the case where p = 2 in this section, but the results can also be generalized to the case where  $ 1 \leq p < \infty $ . Note that from the Corollary 13 in Spek et al. (2022), it is known that  $ \mathcal{F}_{\sigma}(\mathcal{X}, \Omega) = \bigcup_{\pi \in P(\Omega)} \mathcal{L}_{\sigma}^{p}(\pi) $ . Thus, the Proposition 5.1 can be obtained without needing to consider the infinite sum of RKBSs. However, using this approach allows for a more systematic exploration.

## 6 CONCLUSION AND FUTURE WORK

We showed that there is a compatibility property between the direct sum of feature spaces and the sum of RKBSs. By using this, we can decompose a class of integral RKBS  $ \mathcal{F}_{\sigma}(\mathcal{X},\Omega) $  into the sum of p-norm RKBSs  $ \{\mathcal{L}_{\sigma}^{1}(\mu_{i})\}_{i\in I} $ . The advantage of this analytical method is that it allows for a

more structural understanding of the RKBS class through an appropriate decomposition approach. In Section 5, we partially explained these advantages by comparing the integral RKBS class to the previously known sum of RKHSs. Additionally, through these insights, we expect that it would be helpful in designing multiple kernel learning algorithms for the RKBS class. To ensure the feasibility of learning, we need to consider the Representer Theorem, which is discussed in paper Bartolucci et al. (2023) for the integral RKBS class. If the most generalized form of the Representer Theorem presented by Unser & Aziznejad (2022) can be extended to the infinite case, it seems likely that this would enable the recovery of the results obtained in Bartolucci et al. (2023) within the context of our findings.