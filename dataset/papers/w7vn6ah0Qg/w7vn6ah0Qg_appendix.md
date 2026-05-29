## APPENDIX

## A ALGORITHM

Algorithm 1 KokerNet for time series forecasting.

Input:  $ X_T $  with T time points.

Output:  $ K_s $ ,  $ K_{ns} $ ,  $ K_{dis} $ ,  $ g_\Theta(\cdot) $ ,  $ \Theta = \{\omega_1, \cdots, \omega_M\} $ ,  $ \Phi_{de} $ ,  $ \Psi_{de} $ .

1: Calculating  $ S_v \leftarrow p\bar{p} $  based on Eq (7) and Eq (9).

2: Dividing  $ X_T $  into  $ X_s $  and  $ X_{ns} $  based on  $ S_v $ , and

3: repeat

4: For  $ X_s = [X_{s}^{back}, X_{s}^{fore}] $ :

5: Compute the distribution  $ \mathcal{N}(\mu_s, \delta_s^2) \leftarrow X_{s}^{back} $ ;

6: Compute  $ Z_{s}^{back} \leftarrow g(X_{s}^{back}) $ , and forecast  $ Z_{s}^{fore} \leftarrow K_s Z_{s}^{back} $  with  $ K_s $ ;

7: Decode  $ Z_{s}^{fore} $  with the decoder  $ \Phi_{de} $ ,  $ \hat{X}_{s}^{fore} \leftarrow \Phi_{de}(Z_{s}^{fore}) $ ;

8: Compute the distribution  $ \mathcal{N}(\hat{\mu}_s, \delta_s^2) \leftarrow \hat{X}_{s}^{fore} $ ;

9: Compute the loss  $ L_{dis}^s $ ;

10:

11: For  $ X_{ns} = [x_1, \cdots, x_Q] $ :

12: Compute the distribution  $ \{N_{ns}^1, N_{ns}^2, \ldots, N_{ns}^Q\} \leftarrow X_{ns} = [x_1, \cdots, x_Q] $ ;

13:  $ z_i \leftarrow g(x_i) $ ,  $ i = 1, \ldots, Q $ ;

14:  $ Z_{back} \leftarrow [z_1, \ldots, z_{Q-1}] $ ,  $ Z_{fore} \leftarrow [z_2, \ldots, z_Q] $ ;

15:  $ \mathcal{K}_{ns} \leftarrow (Z_{back})(\mathbf{Z}_{fore})^\top $ ;

16:  $ \hat{z}_i \leftarrow K_{ns} z_{i-1} $ 

17: Decode  $ \hat{x}_i \leftarrow \Psi(\hat{z}_i) $ ,  $ i = 2, \ldots, Q $ ;

18: Compute the distribution  $ \mathcal{N}(\hat{\mu}_i, \hat{\delta}_i^2) \leftarrow \hat{x}_i $ ,  $ i = 2, \ldots, Q $ ;

19: Compute the loss  $ L_{ns}^{align} \leftarrow L(\mathcal{N}(\hat{\mu}_i, \hat{\delta}_i^2), \mathcal{N}(\hat{\mu}_s^i, \hat{\delta}_{ns}^{2,i})) $ ,  $ i = 2, \ldots, Q $ .

20:

21: While for the distribution do:

22:  $ g_{dis}[\mu_{ns}^i, \delta_{ns}^{2,i}] \leftarrow \mathcal{N}(\mu_{ns}^i, \delta_{ns}^{2,i}) $ ,  $ i = 1, \ldots, Q $ ;

23: Similar to the process of  $ X_{ns} $ ;

24: Decode  $ [\hat{\mu}_s^i, \hat{\delta}_{ns}^{2,i}] \leftarrow \Upsilon_{de}(\mathcal{K}_{dis}[\mu_{ns}^i, \delta_{ns}^{2,i-1}]) $ ,  $ i = 2, \ldots, Q $ ;

25: Compute the loss  $ L_{dis}^{ns} \leftarrow \mathcal{L}(\mathcal{N}(\mu_{ns}^i, \delta_{ns}^{2,i}), \mathcal{N}(\hat{\mu}_s^i, \delta_{ns}^{2,i})) $ ,  $ i = 2, \ldots, Q $ .

26:

27: Compute the total loss  $ L_{KokerNet} \leftarrow L_{fore}^s + L_{dis}^s + L_{fore}^{ns} + L_{dis}^{align} + L_{dis}^{rec} $ .

28: Update

29: until Convergence

## B RELATED WORKS IN NON-STATIONARY TIME SERIES FORECASTING

Transformer-based deep models (Zhou et al., 2021; Wu et al., 2021; Zhang & Yan, 2022; Zhou et al., 2022; Liu et al., 2021) have achieved great success in forecasting time series with seasonality and trend. However, most of these models are difficult to deal with the non-stationary time series, characterized by the intrinsic change of distribution over time. Recently, several approaches to non-stationary time series forecasting have been developed (Passalis et al., 2019; Kim et al., 2021; Liu et al., 2022). These approaches can be roughly categorized into two aspects. One is the stationarization method, where the focus is on processing the non-stationary time series into stationary ones before performing the forecasting task. Adaptive Norm (Ogasawara et al., 2010) applies z-score normalization for each series fragment by global statistics of a sampled set. DAIN (Passalis et al., 2019) employs a nonlinear neural network to adaptively stationarize time series according to the observed training distribution. RevIN (Kim et al., 2021) introduces a two-stage instance normalization, which transforms model input and output respectively to reduce the discrepancy of each series. Non-stationary Transformer (Liu et al., 2022) utilizes series stationarization to attenuate time series non-stationarity and de-stationary attention to re-incorporate non-stationary information of raw series. The other category is decomposition methods, which divides the non-stationary time series into

time-invariant and time-variant parts (Wang et al., 2023; Liu et al., 2023). The time-invariant part is used to characterize the shared global dynamics, while the time-variant part is used to describe the localized dynamics. KNF (Wang et al., 2023) and Koopa (Liu et al., 2023) cope with the nonstationary time series by introducing both the global and local Koopman operators to explore the time-invariant and time-variant dynamics, respectively.

## C THE ARCHITECTURE OF KOOPMAN OPERATOR LEARNING AND DISTRIBUTION CONSTRAINT

<div style="text-align: center;"><img src="imgs/img_in_image_box_234_438_982_857.jpg" alt="Image" width="61%" /></div>


<div style="text-align: center;">Figure 4: The architecture of Koopman operator learning and distribution constraint. We first decompose the time series  $ X_{T} $  into stationary and non-stationary components based on the designed index  $ S_{v} $ . For the stationary input  $ X_{s} $ , a global Koopman operator  $ K_{s} $  is learned with the stationary distribution constraint. For the non-stationary input  $ X_{ns} $ , a local Koopman operator  $ K_{ns} $  is learned with the non-stationary distribution constraint, which is explored by the historical distribution via the distribution Koopman operator  $ K_{dis} $ .</div>


## D THE PROOF OF THEORETICAL RESULTS

#### D.1 THE DERIVATION PROCEDURE OF g IN EQ. (3)

Lemma D.1. (Bochner's Theorem) A continuous kernel  $  k(\mathbf{x}, \mathbf{x}^{\prime}) = k(\mathbf{x} - \mathbf{x}^{\prime})  $  on  $ R^{d} $  is positive definite if and only if  $  k(\tau)  $ ,  $ \tau = x - x^{\prime} $  is the Fourier transform of a non-negative measure. Such that:

 $$ \begin{align*}k(\pmb{\tau})=&\int_{\mathbb{R}^{d}}s(\lambda)e^{i\lambda\pmb{\tau}}d\lambda,\\s(\lambda)=&\int_{\mathbb{R}^{d}}k(\pmb{\tau})e^{-i\lambda\pmb{\tau}}d\pmb{\tau}.\end{align*} $$ 

Bochner’s Theorem ensures that its inverse Fourier Transform is a probability measure, which means that  $ s(w) $  can be considered as a probability density function.

 $$ \begin{align*}k(\boldsymbol{x}-\boldsymbol{x}^{\prime})=&\int_{\mathbb{R}^{d}}s(w)e^{i\boldsymbol{w}\tau}d\boldsymbol{w}\\=&\mathbb{E}_{\boldsymbol{w}\sim\mathbb{S}}[e^{i\boldsymbol{w}\tau}]\\=&\mathbb{E}_{\boldsymbol{w}\sim\mathbb{S}}[\cos\boldsymbol{w}(\boldsymbol{x}-\boldsymbol{x}^{\prime})+i\sin\boldsymbol{w}(\boldsymbol{x}-\boldsymbol{x}^{\prime})]\\=&\mathbb{E}_{\boldsymbol{w}\sim\mathbb{S}}[\cos\boldsymbol{w}(\boldsymbol{x}-\boldsymbol{x}^{\prime})]\\=&2\mathbb{E}_{\varphi\sim[-\pi,\pi]}[\cos\left(\boldsymbol{w}\boldsymbol{x}+\varphi\right)\cos\left(\boldsymbol{w}\boldsymbol{x}^{\prime}+\varphi\right)]\\\approx&\frac{2}{M}\sum_{m=1}^{M}\langle g(\boldsymbol{x}),g(\boldsymbol{x}^{\prime})\rangle,\end{align*} $$ 

where  $  g(\mathbf{x}) = \sqrt{\frac{2}{M}} [\cos(\mathbf{w}_1 \mathbf{x} + b_1), \cos(\mathbf{w}_2 \mathbf{x} + b_2), \ldots, \cos(\mathbf{w}_M \mathbf{x} + b_M)]^{\top}  $ , M is the sampling number.

### D.2 THE PROOF OF THEOREM 1

Definition D.1. We say that a matrix A is a  $ \Delta $ -spectral approximation of another matrix B, if  $ (1-\Delta)B\preceq A\preceq(1+\Delta)B $ .

Lemma D.2. Let B be a fixed  $ d_{1} \times d_{2} $  matrix. Construct a  $ d_{1} \times d_{2} $  random matrix A that satisfies

 $$ \mathrm{E}[\boldsymbol{A}]=\boldsymbol{B}\quad and\quad||\boldsymbol{A}||_{2}\leq s. $$ 

Let  $ V_{1} $  and  $ V_{2} $  be semidefinite upper bounds for the expected squares:

 $$ \mathrm{E}[A A^{*}]\preceq V_{1}\quad a n d\quad\mathrm{E}[A^{*}A]\preceq V_{2}. $$ 

Define the quantities

 $$ v=m a x(\|\boldsymbol{V}_{1}\|_{2},\|\boldsymbol{V}_{2}\|_{2})\quad a n d\quad r=(t r(\boldsymbol{V}_{1})+t r(\boldsymbol{V}_{2}))/v. $$ 

Form the matrix sampling estimator

 $$ \bar{A}_{n}=\frac{1}{n}\sum_{k=1}^{n}A_{k}, $$ 

where each  $ A_{k} $  is an independent copy of A. Then, for all  $ t \geq \sqrt{\frac{v}{n}} + \frac{2s}{3n} $ ,

 $$ Pr(||\bar{\boldsymbol{A}}_{n})-\boldsymbol{B}||_{2}\geq t)\leq4r e x p(\frac{-n t^{2}/2}{v+2s t/3}). $$ 

Theorem D.1. Sample w according to the spectral density function  $  p(\mathbf{w})  $  and set  $  \mathbf{Z} = g(\mathbf{X}_{T})  $ . When the sampling number  $ M \geq \frac{2\delta(3\sqrt{n}+2\Delta)}{3\Delta^{2}} \ln \frac{8\sqrt{n}}{\rho} $ , with the probability of at least  $ 1 - \rho $ ,  $ ZZ^{\top} $  is the  $ \Delta $ -spectral approximation of  $  \mathbf{K} = \langle f(\mathbf{X}_{T}), f(\mathbf{X}_{T}) \rangle_{\mathcal{H}}  $ .

Proof. Since  $ k(\cdot,\cdot) $  is a positive definite (PD) kernel function, the corresponding kernel matrix K is a PD matrix. So, the kernel matrix K has its inverse form  $ K^{-1} $ , and it can be conducted the eigendecomposition as  $ K = Q^{\top} \Lambda Q = Q^{\top} \Sigma^{2} Q $ .  $ \Sigma $  is a diagonal matrix and the elements are the square root of the eigenvalues of the kernel matrix K.

Let  $ K = Q^{\top} \Sigma^{2} Q $  be an eigendecomposition of K, the  $ \Delta $ -spectral approximation can be written as:

 $$ (1-\Delta)\boldsymbol{K}\preceq\boldsymbol{Z}\boldsymbol{Z}^{\top}\preceq(1+\Delta)\boldsymbol{K}. $$ 

Simplifying eq. (C.8) and multiplying by  $ \Sigma^{-1}Q $  on the left and  $ Q^{\top}\Sigma^{-1} $  on the right, we have

 $$ -\Delta\Sigma^{-1}Q Q^{\top}\Sigma^{2}Q Q^{\top}\Sigma^{-1}\preceq\Sigma^{-1}Q\mathbf{Z}\mathbf{Z}^{\top}Q^{\top}\Sigma^{-1}-\Sigma^{-1}Q\mathbf{K}Q^{\top}\Sigma^{-1}\preceq\Delta\Sigma^{-1}Q Q^{\top}\Sigma^{2}Q Q^{\top}\Sigma^{-1} $$ 

and it suffices to show that:

 $$ ||\Sigma^{-1}Q\mathbf{Z}\mathbf{Z}^{\top}Q^{\top}\Sigma^{-1}-\Sigma^{-1}Q\mathbf{K}Q^{\top}\Sigma^{-1}||_{2}\leq\Delta, $$ 

holds with a probability of at least  $ 1 - \rho $ .

Let

 $$ \begin{array}{r}{\mathbf{Y}_{m}=\Sigma^{-1}Q\mathbf{Z}_{m}\mathbf{Z}_{m}^{\top}Q^{\top}\Sigma^{-1},}\end{array} $$ 

we have

 $$ \mathrm{E}[\boldsymbol{Y}_{m}]=\Sigma^{-1}Q\boldsymbol{Z}\boldsymbol{K}\Sigma^{-1},\frac{1}{M}\sum_{m=1}^{M}=\Sigma^{-1}Q\boldsymbol{Z}\boldsymbol{Z}^{\top}Q^{\top}\Sigma^{-1}. $$ 

Next, we bound the norm of  $ Y_{m} $  and the stable rank  $ E[Y_{m}^{2}] $ . Since  $ Y_{m} $  is always a rank one matrix we have

 $$ \begin{align*}||\boldsymbol{Y}_{m}||_{2}=&||\boldsymbol{\Sigma}^{-1}Q\boldsymbol{Z}_{m}\boldsymbol{Z}_{m}^{\top}Q^{\top}\boldsymbol{\Sigma}^{-1}||_{2}\\=&tr(\boldsymbol{\Sigma}^{-1}Q\boldsymbol{Z}_{m}\boldsymbol{Z}_{m}^{\top}Q^{\top}\boldsymbol{\Sigma}^{-1})\\=&tr(\boldsymbol{Z}_{m}^{\top}Q^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}^{-1}Q\boldsymbol{Z}_{m})\\=&\boldsymbol{Z}_{m}^{\top}Q^{\top}\boldsymbol{\Sigma}^{-2}Q\boldsymbol{Z}_{m}\\=&\boldsymbol{Z}_{m}^{\top}\boldsymbol{K}^{-1}\boldsymbol{Z}_{m}=\boldsymbol{\delta},\end{align*} $$ 

and

 $$ \begin{align*}\boldsymbol{Y}_{m}^{2}=&\boldsymbol{\Sigma}^{-1}Q\boldsymbol{Z}_{m}\boldsymbol{Z}_{m}^{\top}Q^{\top}\boldsymbol{\Sigma}^{-1}\boldsymbol{\Sigma}^{-1}Q\boldsymbol{Z}_{m}\boldsymbol{Z}_{m}^{\top}Q^{\top}\boldsymbol{\Sigma}^{-1}\\=&\boldsymbol{\Sigma}^{-1}Q\boldsymbol{Z}_{m}\boldsymbol{Z}_{m}^{\top}\boldsymbol{K}^{-1}\boldsymbol{Z}_{m}\boldsymbol{Z}_{m}^{\top}Q^{\top}\boldsymbol{\Sigma}^{-1}\\=&\delta\boldsymbol{\Sigma}^{-1}Q\boldsymbol{Z}_{m}\boldsymbol{Z}_{m}^{\top}Q^{\top}\boldsymbol{\Sigma}^{-1}\\=&\delta\boldsymbol{Y}_{m}.\end{align*} $$ 

We calculate  $ E[Y_{m}^{2}] $  as:

 $$ \begin{aligned}\mathrm{E}[\boldsymbol{Y}_{m}^{2}]=&\mathrm{E}[\delta\boldsymbol{Y}_{m}]\\=&\delta\boldsymbol{\Sigma}^{-1}Q\boldsymbol{K}Q^{\top}\boldsymbol{\Sigma}^{-1}\\=&\delta\boldsymbol{I}_{n}.\end{aligned} $$ 

According to Lemma D.2, we have

 $$ Pr(||\frac{1}{M}\sum_{m=1}^{M}\mathbf{Y}_{m}-\Sigma^{-1}Q\mathbf{K}Q^{\top}\Sigma^{-1}||_{2}\geq\Delta)\leq8\sqrt{n}\mathbf{e x p}(\frac{M\Delta^{2}/2}{\delta\sqrt{n}+2\delta\Delta/3}). $$ 

Therefore,  $ ZZ^{\top} $  is the  $ \Delta $ -spectral approximation of K with the probability of at least  $ 1 - \rho $  with the sampling number  $ M \geq \frac{2\delta(3\sqrt{n}+2\Delta)}{3\Delta^{2}}\ln\frac{8\sqrt{n}}{\rho} $ .

### D.3 THE PROOF OF THEOREM 2

For the Koopman operator  $ K^{t} $ , an eigenfunction  $  f \in L^{2}(\mu)  $  corresponding to that eigenvalue satisfies:

 $$ \mathcal{K}^{t}f=e^{i\omega t}f. $$ 

where  $ \omega $  is a real eigenfrequency.

Theorem D.2. For every eigenfrequency $\omega\in R$ of the Koopman operator $\bar{\mathcal{K}}$, Let $g$ be the measurement function on the finite trajectory $\{x_{1},x_{2},\ldots,x_{T}\}$. Then,

(i) When  $ T \geq \sqrt{\frac{2}{M}\frac{3\omega_{max}}{\epsilon}} $ ,  $ g_{\omega} $  can approximate any Koopman eigenfunction with  $ \epsilon $  accuracy, for  $ \epsilon > 0 $ .

(ii)  $ \lim_{T\to\infty}g_{\omega} $  is an eigenfunction of the Koopman operator K.

Proof. We define

 $$ g_{\omega}(\boldsymbol{x})=\frac{1}{T}\int_{\tau=0}^{T}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau, $$ 

based on  $ g \in \bar{H} $ . Let the Koopman operator  $ K^{t} $  acts on  $  g_{\omega}(\boldsymbol{x})  $ , such that

 $$ \begin{align*}\mathcal{K}^{t}g_{\omega}(\boldsymbol{x})=&\frac{1}{T}\int_{\tau=0}^{T}\mathcal{K}^{t}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau\\=&\frac{1}{T}\int_{\tau=0}^{T}g(\boldsymbol{F}^{\tau+t}(\boldsymbol{x}))e^{-i\omega\tau}d\tau\\=&\frac{1}{T}\int_{\tau=0}^{T}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega(\tau-t)}d\tau\\=&e^{i\omega t}\frac{1}{T}\int_{\tau=t}^{T}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau.\end{align*} $$ 

Therefore, we have

 $$ \begin{align*}|\mathcal{K}^{t} g_{\omega}(\boldsymbol{x})-e^{i\omega t}g_{\omega}(\boldsymbol{x})|=&|e^{i\omega t}\frac{1}{T}\int_{\tau=t}^{T} g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau-e^{i\omega t}\frac{1}{T}\int_{\tau=0}^{T} g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau|\\=&\frac{1}{T}|e^{i\omega t}\int_{\tau=0}^{t} g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau|\\\leq&\sqrt{\frac{2}{MT^{2}}}|e^{i\omega t}\int_{\tau=0}^{t} e^{-i\omega\tau}d\tau|\\=&\sqrt{\frac{2}{MT^{2}}}|e^{i\omega t}(-i\omega(e^{-i\omega t}-1))|\\=&\sqrt{\frac{2}{MT^{2}}}|-i\omega+i\omega e^{i\omega t}|\\\leq&\sqrt{\frac{2}{MT^{2}}}\Big[|i\omega|+|i\omega\cos\omega t|+|i\omega\sin\omega t|\Big]\\\leq&\sqrt{\frac{2}{MT^{2}}}3\omega_{\max}\leq\epsilon.\end{align*} $$ 

So, when  $ T \geq \sqrt{\frac{2}{M}} \frac{3\omega_{\max}}{\epsilon} $ ,  $ g_{\omega}(\boldsymbol{x}) $  can approximate any Koopman eigenfunction with  $ \epsilon $  accuracy, for any  $ \epsilon > 0 $ 

(ii) When  $ T \to \infty $ ,  $  g_{\omega}(\boldsymbol{x})  $  can be defined as:



 $$ g_{\omega}(\boldsymbol{x})=\lim_{T\rightarrow\infty}\frac{1}{T}\int_{\tau=0}^{T}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau. $$ 

Let the Koopman operator  $ K^{t} $  acts on  $ g_{\omega}(x) $ , such that:

 $$ \begin{align*}\mathcal{K}^{t}g_{\omega}(\boldsymbol{x})=&\lim_{T\rightarrow\infty}\frac{1}{T}\int_{\tau=0}^{T}\mathcal{K}^{t}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau\\=&\lim_{T\rightarrow\infty}\frac{1}{T}\int_{\tau=0}^{T}g(\boldsymbol{F}^{\tau+t}(\boldsymbol{x}))e^{-i\omega\tau}d\tau\\=&\lim_{T\rightarrow\infty}\frac{1}{T}\int_{\tau=0}^{T}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega(\tau-t)}d\tau\\=&e^{i\omega t}\lim_{T\rightarrow\infty}\frac{1}{T}\int_{\tau=t}^{T}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega\tau}d\tau\\=&e^{i\omega t}\Big[\lim_{T\rightarrow\infty}\frac{1}{T}\int_{\tau=0}^{T}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega(\tau-t)}d\tau-\alpha\Big]\\=&e^{i\omega t}\lim_{T\rightarrow\infty}\frac{1}{T}\int_{\tau=0}^{T}g(\boldsymbol{F}^{\tau}(\boldsymbol{x}))e^{-i\omega(\tau-t)}d\tau\\=&e^{i\omega t}g_{\omega}(\boldsymbol{x}).\end{align*} $$ 

Therefore, when  $ T \rightarrow \infty $ ,  $ g_{\omega} $  is an eigenfunction of the Koopman operator  $ K^{t} $ 

## E THE DETAILED PROCESS FOR TIME SERIES DECOMPOSITION

For the given time series, we first remove its global trends and seasonal effects. Such operations would cause the residual of the time series to tend to be stochastic fluctuation, which is the main attribution of the non-stationarity of real-world time series. Then, the detrended and deseasonalized residuals are divided into J segments  $ X_{T}^{r} = [x_{1}^{r}, x_{2}^{r}, \ldots, x_{J}^{r}] $  and the index  $ S_{v} $  is calculated based on the Kolmogorov-Smirnov test to determine the proportion of the stationary and non-stationary components. After that, we perform the Fourier transform in the original segments  $ X_{T} = [x_{1}, x_{2}, \ldots, x_{J}] $  to calculate the frequency spectrum and sort all frequencies by the number of occurrences. Finally, the top  $ \alpha $  percent of the frequency spectrum are considered as the components of the stationary, while the remaining are considered as the components of the non-stationary. The disentanglement in the given time series  $ X_{T} $  is mathematically formulated as follows:

 $$ \begin{aligned}\boldsymbol{X}_{\mathrm{s}}=&\mathrm{FT}^{-1}(S_{\alpha},\mathrm{FT}(\boldsymbol{X}_{T})),\\\boldsymbol{X}_{\mathrm{ns}}=&\mathrm{FT}^{-1}(S-S_{\alpha},\mathrm{FT}(\boldsymbol{X}_{T})),\\\boldsymbol{X}_{T}=&\boldsymbol{X}_{\mathrm{s}}+\boldsymbol{X}_{\mathrm{ns}}\end{aligned} $$ 

where  $ X_{s} $ ,  $ X_{ns} $  are time-invariant and time-variant components respectively. S is the set of frequency spectrum.  $ S_{\alpha} $  is the set of global shared frequency spectrum. FT $ ^{-1} $  denotes the inverse of FT.

In our work, we do not focus on the specific design for the detrending and deseasonalizing. Therefore, we conduct it by the commonly used Pytorch code with the additive model of the seasonal_decompose function. The additive model deems that the time series X consists of three components, including trend (i.e., the global trends)  $ X_{trend} $ , seasonal  $ X_{trend} $ , and residual  $ X_{r} $ .  $ X = X_{trend} + X_{trend} + X_{r} $ . The seasonal_decompose function directly returns the components, trend, seasonal, and residual in the code. More precisely, the flow of the seasonal_decompose function mainly includes four steps: (1) Determine the seasonal cycle (i.e., period) of the data. The period denotes the length of the season; (2) Compute the trend components. The seasonal component is the remaining cyclical pattern after removing the trend component; (3) Compute the trend components; (4) compute the residual by  $ X_{r} = X - X_{trend} - X_{trend} $ .

## F ADDITIONAL EXPERIMENTAL RESULTS

### F.1 ADDITIONAL RESULTS ON MULTIVARIATE TIME SERIES FORECASTING

Due to the limited pages, we list additional multivariate time series forecasting results. The results on ETTh1, ETTm1, and ETTm2 datasets are reported in Table 5. As shown in Table 5, our KokerNet still achieves competitive performance compared with state-of-the-art deep forecasting models.

<div style="text-align: center;">Table 5: Multivariate time series forecasting results with different forecasting lengths  $ H \in \{48, 96, 144, 192\} $  under the lookback length T = 2H on ETTh1, ETTm1, and ETTm2 datasets. The best results are highlighted in bold and the suboptimal results are highlighted in underline. (All the results of the compared methods are replications based on the publicly available code.)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Models</td><td colspan="2">KokerNet</td><td colspan="2">Ns.Transformer</td><td colspan="2">Autoformer</td><td colspan="2">Koopa</td><td colspan="2">iTransformer</td><td colspan="2">KNF</td><td colspan="2">Crossformer</td></tr><tr><td style='text-align: center;'>Metric</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MAE</td></tr><tr><td rowspan="4">ETThl</td><td style='text-align: center;'>48</td><td style='text-align: center;'>0.3366</td><td style='text-align: center;'>0.3779</td><td style='text-align: center;'>0.5152</td><td style='text-align: center;'>0.4784</td><td style='text-align: center;'>0.4722</td><td style='text-align: center;'>0.4595</td><td style='text-align: center;'>0.3455</td><td style='text-align: center;'>0.3843</td><td style='text-align: center;'>0.3442</td><td style='text-align: center;'>0.3800</td><td style='text-align: center;'>0.8760</td><td style='text-align: center;'>0.7090</td><td style='text-align: center;'>0.3545</td><td style='text-align: center;'>0.3989</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>0.4003</td><td style='text-align: center;'>0.4177</td><td style='text-align: center;'>0.5436</td><td style='text-align: center;'>0.5064</td><td style='text-align: center;'>0.5003</td><td style='text-align: center;'>0.4746</td><td style='text-align: center;'>0.3871</td><td style='text-align: center;'>0.4058</td><td style='text-align: center;'>0.3991</td><td style='text-align: center;'>0.4150</td><td style='text-align: center;'>0.9750</td><td style='text-align: center;'>0.7440</td><td style='text-align: center;'>0.4082</td><td style='text-align: center;'>0.4258</td></tr><tr><td style='text-align: center;'>144</td><td style='text-align: center;'>0.4068</td><td style='text-align: center;'>0.4205</td><td style='text-align: center;'>0.5473</td><td style='text-align: center;'>0.5064</td><td style='text-align: center;'>0.4670</td><td style='text-align: center;'>0.4666</td><td style='text-align: center;'>0.4298</td><td style='text-align: center;'>0.4289</td><td style='text-align: center;'>0.4165</td><td style='text-align: center;'>0.4244</td><td style='text-align: center;'>0.8010</td><td style='text-align: center;'>0.6620</td><td style='text-align: center;'>0.5002</td><td style='text-align: center;'>0.4955</td></tr><tr><td style='text-align: center;'>192</td><td style='text-align: center;'>0.4226</td><td style='text-align: center;'>0.4352</td><td style='text-align: center;'>0.6211</td><td style='text-align: center;'>0.5287</td><td style='text-align: center;'>0.5060</td><td style='text-align: center;'>0.4802</td><td style='text-align: center;'>0.4401</td><td style='text-align: center;'>0.4357</td><td style='text-align: center;'>0.4427</td><td style='text-align: center;'>0.4503</td><td style='text-align: center;'>0.9410</td><td style='text-align: center;'>0.7440</td><td style='text-align: center;'>0.5782</td><td style='text-align: center;'>0.5645</td></tr><tr><td rowspan="4">ETTm1</td><td style='text-align: center;'>48</td><td style='text-align: center;'>0.2942</td><td style='text-align: center;'>0.3452</td><td style='text-align: center;'>0.4084</td><td style='text-align: center;'>0.4003</td><td style='text-align: center;'>0.8157</td><td style='text-align: center;'>0.5999</td><td style='text-align: center;'>0.2863</td><td style='text-align: center;'>0.3361</td><td style='text-align: center;'>0.3162</td><td style='text-align: center;'>0.3565</td><td style='text-align: center;'>1.0260</td><td style='text-align: center;'>0.7920</td><td style='text-align: center;'>0.3128</td><td style='text-align: center;'>0.3654</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>0.2960</td><td style='text-align: center;'>0.3452</td><td style='text-align: center;'>0.4419</td><td style='text-align: center;'>0.4429</td><td style='text-align: center;'>0.5762</td><td style='text-align: center;'>0.5124</td><td style='text-align: center;'>0.3264</td><td style='text-align: center;'>0.3648</td><td style='text-align: center;'>0.3019</td><td style='text-align: center;'>0.3483</td><td style='text-align: center;'>0.9570</td><td style='text-align: center;'>0.7820</td><td style='text-align: center;'>0.3235</td><td style='text-align: center;'>0.3670</td></tr><tr><td style='text-align: center;'>144</td><td style='text-align: center;'>0.3163</td><td style='text-align: center;'>0.3612</td><td style='text-align: center;'>0.5081</td><td style='text-align: center;'>0.4459</td><td style='text-align: center;'>0.7313</td><td style='text-align: center;'>0.5649</td><td style='text-align: center;'>0.3546</td><td style='text-align: center;'>0.3798</td><td style='text-align: center;'>0.3216</td><td style='text-align: center;'>0.3637</td><td style='text-align: center;'>0.9210</td><td style='text-align: center;'>0.7600</td><td style='text-align: center;'>0.3667</td><td style='text-align: center;'>0.4019</td></tr><tr><td style='text-align: center;'>192</td><td style='text-align: center;'>0.3301</td><td style='text-align: center;'>0.3719</td><td style='text-align: center;'>0.5379</td><td style='text-align: center;'>0.4661</td><td style='text-align: center;'>0.6689</td><td style='text-align: center;'>0.5421</td><td style='text-align: center;'>0.3683</td><td style='text-align: center;'>0.3875</td><td style='text-align: center;'>0.3378</td><td style='text-align: center;'>0.3760</td><td style='text-align: center;'>0.8960</td><td style='text-align: center;'>0.7310</td><td style='text-align: center;'>0.3820</td><td style='text-align: center;'>0.4213</td></tr><tr><td rowspan="4">ETTm2</td><td style='text-align: center;'>48</td><td style='text-align: center;'>0.1396</td><td style='text-align: center;'>0.2315</td><td style='text-align: center;'>0.1726</td><td style='text-align: center;'>0.2603</td><td style='text-align: center;'>0.1919</td><td style='text-align: center;'>0.2941</td><td style='text-align: center;'>0.1403</td><td style='text-align: center;'>0.2326</td><td style='text-align: center;'>0.1415</td><td style='text-align: center;'>0.2361</td><td style='text-align: center;'>0.6210</td><td style='text-align: center;'>0.6230</td><td style='text-align: center;'>0.1860</td><td style='text-align: center;'>0.2938</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>0.1739</td><td style='text-align: center;'>0.2576</td><td style='text-align: center;'>0.2414</td><td style='text-align: center;'>0.3092</td><td style='text-align: center;'>0.2852</td><td style='text-align: center;'>0.3530</td><td style='text-align: center;'>0.1804</td><td style='text-align: center;'>0.2614</td><td style='text-align: center;'>0.1850</td><td style='text-align: center;'>0.2736</td><td style='text-align: center;'>1.5350</td><td style='text-align: center;'>1.0120</td><td style='text-align: center;'>0.3818</td><td style='text-align: center;'>0.436</td></tr><tr><td style='text-align: center;'>144</td><td style='text-align: center;'>0.2111</td><td style='text-align: center;'>0.2871</td><td style='text-align: center;'>0.3705</td><td style='text-align: center;'>0.3827</td><td style='text-align: center;'>0.2749</td><td style='text-align: center;'>0.3453</td><td style='text-align: center;'>0.2155</td><td style='text-align: center;'>0.2859</td><td style='text-align: center;'>0.2190</td><td style='text-align: center;'>0.2971</td><td style='text-align: center;'>1.3370</td><td style='text-align: center;'>0.8760</td><td style='text-align: center;'>0.4135</td><td style='text-align: center;'>0.4796</td></tr><tr><td style='text-align: center;'>192</td><td style='text-align: center;'>0.2301</td><td style='text-align: center;'>0.3033</td><td style='text-align: center;'>0.3237</td><td style='text-align: center;'>0.3540</td><td style='text-align: center;'>0.3039</td><td style='text-align: center;'>0.3633</td><td style='text-align: center;'>0.2401</td><td style='text-align: center;'>0.3009</td><td style='text-align: center;'>0.2393</td><td style='text-align: center;'>0.3098</td><td style='text-align: center;'>1.3550</td><td style='text-align: center;'>0.9080</td><td style='text-align: center;'>0.6551</td><td style='text-align: center;'>0.6130</td></tr></table>

### F.2 THE INFLUENCE OF THE PROPORTION OF THE STATIONARY COMPONENTS AND SEGMENTATION LENGTH

To explore the influence of the proportion of the stationary components on the results and the influence of segmentation length on the results, we also include additional experiments. In Figure 5, we report the MSE results on Traffic, ETTh1, ETTm1, and ETTm2, which is not reported in the main paper due to the limited pages. From Figure 5, we can observe that the MSE tends to decrease as the proportion of stationary components increases.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_369_402_551.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_416_369_601_552.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_617_369_801_552.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_816_370_1000_552.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">Figure 5: The influence of the proportion of the stationary components on the results.</div>


In order to provide a more intuitive comparison of the influence of the stationary components, we analyze the result on the M4 dataset again. Concretely, we first show the forecastability, Lyapunov exponents, trend, and seasonality of the M4 dataset with different frequencies in Table 6.  $ \downarrow $  indicates the smaller the value, the higher the non-stationarity, while  $ \uparrow $  indicates the higher the value, the higher the non-stationarity. Then, we report the forecasting results on the M4 dataset (i.e., the univariate time series forecasting results in the main text) in Table 7. From Table 6 and Table 7, we can observe that our KokerNet exhibits enhanced performance in scenarios where seasonality is less pronounced and forecastability is heightened. By contrast, our KokerNet tends to perform mediocre in scenarios where seasonality is more pronounced. In particular, our KokerNet achieves a 12.71% reduction in sMAPE for Yearly data and a 10.31% reduction for Daily data. This demonstrates that our proposal can perform better on datasets and yield superior results on datasets with higher non-stationarity.

<div style="text-align: center;">Table 6: The forecastability, Lyapunov exponents, trend, and seasonality of the M4 dataset with different frequencies.  $ \downarrow $  indicates the smaller the value, the higher the non-stationarity, while  $ \uparrow $  indicates the higher the value, the higher the non-stationarity. The results are cited from Wang et al. (2023).</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Forecastability ( $ \downarrow $ )</td><td style='text-align: center;'>LEs ( $ \uparrow $ )</td><td style='text-align: center;'>Trend ( $ \downarrow $ )</td><td style='text-align: center;'>Seasonality ( $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>Yearly</td><td style='text-align: center;'>0.58</td><td style='text-align: center;'>0.004</td><td style='text-align: center;'>4.32</td><td style='text-align: center;'>0.00%</td></tr><tr><td style='text-align: center;'>Quarterly</td><td style='text-align: center;'>0.47</td><td style='text-align: center;'>0.003</td><td style='text-align: center;'>1.06</td><td style='text-align: center;'>84.51%</td></tr><tr><td style='text-align: center;'>Monthly</td><td style='text-align: center;'>0.44</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>0.48</td><td style='text-align: center;'>66.34%</td></tr><tr><td style='text-align: center;'>Weekly</td><td style='text-align: center;'>0.43</td><td style='text-align: center;'>0.013</td><td style='text-align: center;'>0.13</td><td style='text-align: center;'>0.00%</td></tr><tr><td style='text-align: center;'>Daily</td><td style='text-align: center;'>0.44</td><td style='text-align: center;'>0.020</td><td style='text-align: center;'>0.05</td><td style='text-align: center;'>0.00%</td></tr><tr><td style='text-align: center;'>Hourly</td><td style='text-align: center;'>0.46</td><td style='text-align: center;'>0.003</td><td style='text-align: center;'>0.02</td><td style='text-align: center;'>99.76%</td></tr></table>

<div style="text-align: center;">In Figure 6, we report the MSE result with different forecasting length  $ H = \{48, 96, 144, 192\} $  on ECL. From Figure 6, we can observe that the time series tends to become more stationary as the length of the time series increases, and the proportion of stationary components has a greater impact on the results.</div>


### F.3 TIME SERIES DECOMPOSITION

As previously discussed, the real-world time series commonly contains both time-invariant and time-variant patterns, corresponding to the stationary and non-stationary components. Therefore, we decompose the time series into these two components to evade information loss and the introduction of unnecessary disturbances caused by the single-component assumption. To evaluate the effectiveness of the time series decomposition step, we perform an experiment with three cases, including

<div style="text-align: center;">Table 7: Univariate time series forecasting results with different frequencies on M4 dataset. The best results are highlighted in bold. (All the results of the compared methods are replications based on the publicly available code.)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td colspan="3">KokerNet</td><td colspan="3">PatchTST</td><td colspan="3">DLinear</td><td colspan="3">Koopa</td></tr><tr><td style='text-align: center;'>sMAPE</td><td style='text-align: center;'>MAPE</td><td style='text-align: center;'>MASE</td><td style='text-align: center;'>sMAPE</td><td style='text-align: center;'>MAPE</td><td style='text-align: center;'>MASE</td><td style='text-align: center;'>sMAPE</td><td style='text-align: center;'>MAPE</td><td style='text-align: center;'>MASE</td><td style='text-align: center;'>sMAPE</td><td style='text-align: center;'>MAPE</td><td style='text-align: center;'>MASE</td></tr><tr><td style='text-align: center;'>Yearly</td><td style='text-align: center;'>13.454</td><td style='text-align: center;'>16.571</td><td style='text-align: center;'>3.033</td><td style='text-align: center;'>16.668</td><td style='text-align: center;'>23.302</td><td style='text-align: center;'>3.729</td><td style='text-align: center;'>15.413</td><td style='text-align: center;'>18.467</td><td style='text-align: center;'>3.696</td><td style='text-align: center;'>14.707</td><td style='text-align: center;'>19.417</td><td style='text-align: center;'>3.275</td></tr><tr><td style='text-align: center;'>Quarterly</td><td style='text-align: center;'>10.213</td><td style='text-align: center;'>11.779</td><td style='text-align: center;'>1.192</td><td style='text-align: center;'>12.606</td><td style='text-align: center;'>15.118</td><td style='text-align: center;'>1.628</td><td style='text-align: center;'>10.546</td><td style='text-align: center;'>12.288</td><td style='text-align: center;'>1.242</td><td style='text-align: center;'>10.775</td><td style='text-align: center;'>12.823</td><td style='text-align: center;'>1.287</td></tr><tr><td style='text-align: center;'>Monthly</td><td style='text-align: center;'>12.780</td><td style='text-align: center;'>14.874</td><td style='text-align: center;'>0.940</td><td style='text-align: center;'>15.859</td><td style='text-align: center;'>19.902</td><td style='text-align: center;'>1.273</td><td style='text-align: center;'>13.233</td><td style='text-align: center;'>15.750</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>16.127</td><td style='text-align: center;'>19.378</td><td style='text-align: center;'>1.270</td></tr><tr><td style='text-align: center;'>Weekly</td><td style='text-align: center;'>11.157</td><td style='text-align: center;'>10.309</td><td style='text-align: center;'>3.396</td><td style='text-align: center;'>11.551</td><td style='text-align: center;'>11.234</td><td style='text-align: center;'>4.465</td><td style='text-align: center;'>11.168</td><td style='text-align: center;'>12.003</td><td style='text-align: center;'>5.936</td><td style='text-align: center;'>10.221</td><td style='text-align: center;'>9.542</td><td style='text-align: center;'>3.135</td></tr><tr><td style='text-align: center;'>Daily</td><td style='text-align: center;'>3.035</td><td style='text-align: center;'>4.387</td><td style='text-align: center;'>3.251</td><td style='text-align: center;'>3.576</td><td style='text-align: center;'>5.590</td><td style='text-align: center;'>3.894</td><td style='text-align: center;'>3.384</td><td style='text-align: center;'>5.165</td><td style='text-align: center;'>3.685</td><td style='text-align: center;'>3.395</td><td style='text-align: center;'>4.886</td><td style='text-align: center;'>3.682</td></tr><tr><td style='text-align: center;'>Hourly</td><td style='text-align: center;'>18.013</td><td style='text-align: center;'>23.685</td><td style='text-align: center;'>3.094</td><td style='text-align: center;'>34.211</td><td style='text-align: center;'>118.404</td><td style='text-align: center;'>10.752</td><td style='text-align: center;'>17.223</td><td style='text-align: center;'>23.482</td><td style='text-align: center;'>2.702</td><td style='text-align: center;'>18.171</td><td style='text-align: center;'>23.683</td><td style='text-align: center;'>2.808</td></tr><tr><td style='text-align: center;'>Others</td><td style='text-align: center;'>4.858</td><td style='text-align: center;'>6.410</td><td style='text-align: center;'>3.248</td><td style='text-align: center;'>6.685</td><td style='text-align: center;'>15.336</td><td style='text-align: center;'>4.503</td><td style='text-align: center;'>5.089</td><td style='text-align: center;'>7.173</td><td style='text-align: center;'>3.765</td><td style='text-align: center;'>5.109</td><td style='text-align: center;'>6.777</td><td style='text-align: center;'>3.570</td></tr><tr><td style='text-align: center;'>Average</td><td style='text-align: center;'>12.408</td><td style='text-align: center;'>14.739</td><td style='text-align: center;'>1.623</td><td style='text-align: center;'>15.474</td><td style='text-align: center;'>20.841</td><td style='text-align: center;'>2.535</td><td style='text-align: center;'>13.269</td><td style='text-align: center;'>16.089</td><td style='text-align: center;'>2.196</td><td style='text-align: center;'>14.476</td><td style='text-align: center;'>17.862</td><td style='text-align: center;'>2.207</td></tr></table>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_221_423_405_606.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_418_423_604_606.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_619_423_804_607.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_818_423_1001_607.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">Figure 6: The influence of the segmentation length on the results.</div>


 $ K_{s}, K_{ns} $ , and  $ Pre_{def} $ .  $ K_{s} $  denotes the entire time series is stationary, and only the global Koopman operator is learned for the time series.  $ K_{ns} $  denotes the entire time series is non-stationary, and only the local Koopman operator is learned. Specifically, in the  $ Pre_{def} $  case, we set the candidate range of the proportions as  $ [10\%, 20\%, \ldots, 90\%] $  and select the most optimal result, where 10% represents that the stationary component account for 10%, while the non-stationary component account for 90%.

In the main text, we just report the results of two datasets. To verify the generalization effectiveness of the time series decomposition step, we also include the additional results on the remaining datasets of the main text. The results are shown in Table 8. We can observe that the results are consistent with our conclusions in the main text (i.e., time series decomposition is important for the non-stationary data).

<div style="text-align: center;">Table 8: The results with the single component. Here,  $ K_{s} $  denotes only the global shared Koopman operator included in our model,  $ K_{ns} $  denotes only local Koopman operator included in our model, and  $ Pre_{def} $  denotes the best result for different proportion of the stationary component under the decomposition case. The best results are highlighted in bold.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td rowspan="2">Dataset Metric</td><td colspan="2">ETTh1</td><td colspan="2">ETTm1</td><td colspan="2">ETTm2</td><td colspan="2">Traffic</td><td colspan="2">Weather</td><td colspan="2">Exchange</td><td colspan="2">ECL</td></tr><tr><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>MSE</td><td style='text-align: center;'>MAE</td></tr><tr><td rowspan="4">$ \mathcal{K}_{s} $</td><td style='text-align: center;'>48</td><td style='text-align: center;'>0.3368</td><td style='text-align: center;'>0.3793</td><td style='text-align: center;'>0.2899</td><td style='text-align: center;'>0.3393</td><td style='text-align: center;'>0.1385</td><td style='text-align: center;'>0.2304</td><td style='text-align: center;'>0.4475</td><td style='text-align: center;'>0.2984</td><td style='text-align: center;'>0.1389</td><td style='text-align: center;'>0.1772</td><td style='text-align: center;'>0.0427</td><td style='text-align: center;'>0.1425</td><td style='text-align: center;'>0.1570</td><td style='text-align: center;'>0.2437</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>0.3892</td><td style='text-align: center;'>0.4118</td><td style='text-align: center;'>0.2969</td><td style='text-align: center;'>0.3461</td><td style='text-align: center;'>0.1750</td><td style='text-align: center;'>0.2591</td><td style='text-align: center;'>0.4073</td><td style='text-align: center;'>0.2827</td><td style='text-align: center;'>0.1631</td><td style='text-align: center;'>0.2091</td><td style='text-align: center;'>0.0861</td><td style='text-align: center;'>0.2071</td><td style='text-align: center;'>0.1365</td><td style='text-align: center;'>0.2291</td></tr><tr><td style='text-align: center;'>144</td><td style='text-align: center;'>0.4112</td><td style='text-align: center;'>0.4242</td><td style='text-align: center;'>0.3169</td><td style='text-align: center;'>0.3627</td><td style='text-align: center;'>0.2073</td><td style='text-align: center;'>0.2869</td><td style='text-align: center;'>0.4123</td><td style='text-align: center;'>0.2892</td><td style='text-align: center;'>0.1813</td><td style='text-align: center;'>0.2292</td><td style='text-align: center;'>0.1360</td><td style='text-align: center;'>0.2656</td><td style='text-align: center;'>0.1453</td><td style='text-align: center;'>0.2373</td></tr><tr><td style='text-align: center;'>192</td><td style='text-align: center;'>0.4822</td><td style='text-align: center;'>0.4557</td><td style='text-align: center;'>0.316</td><td style='text-align: center;'>0.3726</td><td style='text-align: center;'>0.251</td><td style='text-align: center;'>0.3073</td><td style='text-align: center;'>0.4198</td><td style='text-align: center;'>0.2947</td><td style='text-align: center;'>0.2033</td><td style='text-align: center;'>0.2516</td><td style='text-align: center;'>0.2069</td><td style='text-align: center;'>0.2690</td><td style='text-align: center;'>0.192</td><td style='text-align: center;'>0.2441</td></tr><tr><td rowspan="4">$ \mathcal{K}_{ns} $</td><td style='text-align: center;'>48</td><td style='text-align: center;'>0.6882</td><td style='text-align: center;'>0.5521</td><td style='text-align: center;'>0.6813</td><td style='text-align: center;'>0.5412</td><td style='text-align: center;'>0.1938</td><td style='text-align: center;'>0.2867</td><td style='text-align: center;'>0.3573</td><td style='text-align: center;'>0.7819</td><td style='text-align: center;'>0.1917</td><td style='text-align: center;'>0.2580</td><td style='text-align: center;'>0.1071</td><td style='text-align: center;'>0.1868</td><td style='text-align: center;'>0.8349</td><td style='text-align: center;'>0.7630</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>0.7035</td><td style='text-align: center;'>0.5634</td><td style='text-align: center;'>0.6021</td><td style='text-align: center;'>0.5160</td><td style='text-align: center;'>0.2210</td><td style='text-align: center;'>0.3076</td><td style='text-align: center;'>0.3794</td><td style='text-align: center;'>0.7969</td><td style='text-align: center;'>0.2338</td><td style='text-align: center;'>0.2859</td><td style='text-align: center;'>0.1633</td><td style='text-align: center;'>0.2895</td><td style='text-align: center;'>0.8326</td><td style='text-align: center;'>0.7575</td></tr><tr><td style='text-align: center;'>144</td><td style='text-align: center;'>0.7106</td><td style='text-align: center;'>0.5783</td><td style='text-align: center;'>0.6468</td><td style='text-align: center;'>0.5378</td><td style='text-align: center;'>0.2539</td><td style='text-align: center;'>0.3290</td><td style='text-align: center;'>0.3981</td><td style='text-align: center;'>0.8058</td><td style='text-align: center;'>0.2623</td><td style='text-align: center;'>0.3092</td><td style='text-align: center;'>0.2712</td><td style='text-align: center;'>0.3823</td><td style='text-align: center;'>0.8395</td><td style='text-align: center;'>0.7596</td></tr><tr><td style='text-align: center;'>192</td><td style='text-align: center;'>0.7239</td><td style='text-align: center;'>0.5868</td><td style='text-align: center;'>0.6052</td><td style='text-align: center;'>0.5257</td><td style='text-align: center;'>0.2859</td><td style='text-align: center;'>0.3519</td><td style='text-align: center;'>0.4088</td><td style='text-align: center;'>0.8058</td><td style='text-align: center;'>0.2862</td><td style='text-align: center;'>0.3274</td><td style='text-align: center;'>0.3666</td><td style='text-align: center;'>0.4497</td><td style='text-align: center;'>0.8479</td><td style='text-align: center;'>0.7627</td></tr><tr><td rowspan="4">Predef</td><td style='text-align: center;'>48</td><td style='text-align: center;'>0.3363</td><td style='text-align: center;'>0.3792</td><td style='text-align: center;'>0.2916</td><td style='text-align: center;'>0.3392</td><td style='text-align: center;'>0.1383</td><td style='text-align: center;'>0.2299</td><td style='text-align: center;'>0.4458</td><td style='text-align: center;'>0.2986</td><td style='text-align: center;'>0.1398</td><td style='text-align: center;'>0.1777</td><td style='text-align: center;'>0.0452</td><td style='text-align: center;'>0.1469</td><td style='text-align: center;'>0.1564</td><td style='text-align: center;'>0.2432</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>0.4003</td><td style='text-align: center;'>0.4177</td><td style='text-align: center;'>0.2958</td><td style='text-align: center;'>0.3451</td><td style='text-align: center;'>0.1739</td><td style='text-align: center;'>0.2576</td><td style='text-align: center;'>0.4065</td><td style='text-align: center;'>0.2816</td><td style='text-align: center;'>0.1657</td><td style='text-align: center;'>0.2125</td><td style='text-align: center;'>0.0862</td><td style='text-align: center;'>0.2074</td><td style='text-align: center;'>0.1363</td><td style='text-align: center;'>0.2285</td></tr><tr><td style='text-align: center;'>144</td><td style='text-align: center;'>0.4064</td><td style='text-align: center;'>0.4202</td><td style='text-align: center;'>0.3163</td><td style='text-align: center;'>0.3612</td><td style='text-align: center;'>0.2109</td><td style='text-align: center;'>0.2870</td><td style='text-align: center;'>0.4089</td><td style='text-align: center;'>0.2863</td><td style='text-align: center;'>0.1819</td><td style='text-align: center;'>0.2292</td><td style='text-align: center;'>0.1389</td><td style='text-align: center;'>0.2693</td><td style='text-align: center;'>0.1446</td><td style='text-align: center;'>0.2367</td></tr><tr><td style='text-align: center;'>192</td><td style='text-align: center;'>0.4225</td><td style='text-align: center;'>0.4351</td><td style='text-align: center;'>0.3301</td><td style='text-align: center;'>0.3719</td><td style='text-align: center;'>0.2300</td><td style='text-align: center;'>0.3034</td><td style='text-align: center;'>0.4159</td><td style='text-align: center;'>0.2914</td><td style='text-align: center;'>0.2020</td><td style='text-align: center;'>0.2495</td><td style='text-align: center;'>0.2034</td><td style='text-align: center;'>0.3261</td><td style='text-align: center;'>0.1511</td><td style='text-align: center;'>0.2442</td></tr></table>

### F.4 MODEL EFFICIENCY COMPARISON

<div style="text-align: center;"><img src="imgs/img_in_chart_box_273_234_976_546.jpg" alt="Image" width="57%" /></div>


<div style="text-align: center;">Figure 7: Model efficiency comparison on ETTh1 with H = 144. Training time and memory footprint are recorded with the same batch size (32) and official code configuration.</div>


## REFERENCES

Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi, and Jaegul Choo. Reversible instance normalization for accurate time-series forecasting against distribution shift. In Proceedings of the International Conference on Learning Representations, 2021.

Shizhan Liu, Hang Yu, Cong Liao, Jianguo Li, Weiyao Lin, Alex X Liu, and Schahram Dustdar. Pyraformer: Low-complexity pyramidal attention for long-range time series modeling and forecasting. In Proceedings of the International Conference on Learning Representations, 2021.

Yong Liu, Haixu Wu, Jianmin Wang, and Mingsheng Long. Non-stationary transformers: Exploring the stationarity in time series forecasting. In Advances in Neural Information Processing Systems, pp. 9881–9893, 2022.

Yong Liu, Chenyu Li, Jianmin Wang, and Mingsheng Long. Koopa: Learning non-stationary time series dynamics with koopman predictors. In Advances in Neural Information Processing Systems, volume 36, pp. 12271–12290, 2023.

Eduardo Ogasawara, Leonardo C Martinez, Daniel De Oliveira, Geraldo Zimbrão, Gisele L Pappa, and Marta Mattoso. Adaptive normalization: A novel data normalization approach for nonstationary time series. In Proceedings of the International Joint Conference on Neural Networks, pp. 1–8, 2010.

Nikolaos Passalis, Anastasios Tefas, Juho Kanniainen, Moncef Gabbouj, and Alexandros Iosifidis. Deep adaptive input normalization for time series forecasting. IEEE Transactions on Neural Networks and Learning Systems, 31(9):3760–3765, 2019.

Rui Wang, Yihe Dong, Sercan Ö. Arik, and Rose Yu. Koopman neural operator forecaster for time-series with temporal distributional shifts. In Proceeding of the International Conference on Learning Representations, 2023.

Haixu Wu, Jiehui Xu, Jianmin Wang, and Mingsheng Long. Autoformer: Decomposition transformers with auto-correlation for long-term series forecasting. In Advances in Neural Information Processing Systems, pp. 22419–22430, 2021.

Yunhao Zhang and Junchi Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In Proceedings of the International Conference on Learning Representations, 2022.

Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong, and Wancai Zhang. Informer: Beyond efficient transformer for long sequence time-series forecasting. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 35, pp. 11106–11115, 2021.

Tian Zhou, Ziqing Ma, Qingsong Wen, Xue Wang, Liang Sun, and Rong Jin. Fedformer: Frequency enhanced decomposed transformer for long-term series forecasting. In Proceedings of the International Conference on Machine Learning, pp. 27268–27286, 2022.