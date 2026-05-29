## A BACKGROUND ON TRADITIONAL PERSISTENCE

Persistent homology (PH) is a mathematical machinery which allows us to quantify shape of an object along various dimension(s). By shape here, we broadly understand object properties that are preserved under continuous transformations, i.e. the ones that do not change "holes" in the object. Such transformations include, for example, folding, bending, and twisting; while "holes" can be connected components (0-dim topological features), loops (1-dim topological features), voids (2-dim topological features), and their higher-dimensional counterparts. PH monitors evolution of these topological features as we monotonically change certain user-defined parameters. The idea is to select some suitable monotonic sequence of scales  $ \alpha_{1} < \alpha_{2} < \ldots < \alpha_{m} $ , and then to study graph G not as a single object but as a parametrized sequence of nested graphs  $ G_{\alpha_{1}} \subseteq G_{\alpha_{2}} \subseteq \ldots \subseteq G_{\alpha_{m}} = G $ , recording which holes appear (born) and disappear (die) throughout this filtration. Holes that persist over the filtration (i.e., with longer lifespans) are likelier to contain some essential latent information about the structural organization of G, while holes with shorter lifespans are often referred to as topological noise. To systematize the process of selecting and recording holes, we build an abstract simplicial complex  $ \mathcal{K}(\mathcal{G}_{\alpha_{i}}) $  on each  $ \mathcal{G}_{\alpha_{i}} $ , which results in a filtration of complexes  $ \mathcal{K}(\mathcal{G}_{\alpha_{1}}) \subseteq \mathcal{K}(\mathcal{G}_{\alpha_{m}}) \subseteq \ldots \subseteq \mathcal{K}(\mathcal{G}_{\alpha_{m}}) $  and counting the associated simplices in H. As an example of such scale parameter  $ \alpha $ , we can choose edge weight and then use an abstract simplicial complex  $ \mathcal{K}(\mathcal{G}_{\alpha}) = \{\tilde{\mathcal{G}} \subseteq \mathcal{G}|diam(\tilde{\mathcal{G}}) \leq \alpha\} $ , that is, we keep only nodes (and the associated induced subgraphs) with a shortest weighted path of at most  $ \alpha $ . Such  $ \mathcal{K} $  is called a Vietoris-Rips complex and the induced nested sequence is called a power filtration. (For an overview of various filtration on graphs see Adams et al. (2017); Bauer (2019); Hofer et al. (2020).) As such, this traditional PH framework considers a sequence of linear maps  $ \mathcal{K}(\mathcal{G}_{\alpha_{i}}) \hookrightarrow \mathcal{K}(\mathcal{G}_{\alpha_{i+1}}) $  along the same direction, thereby assuming that each simplicial complex needs to be a subset of the following one, allowing us only to add new simplices to the preceding complex. This limits applicability of PH to time-dependent and dynamic graphs, where the goal is to extract time-aware topological signatures that persist over time.

### B PROOF OF PROPOSITION 5.2

Proof. By definition of ZS, we get

 $$ ||Z S-Z S^{\prime}||_{\infty}=\max_{1\leq k\leq m}\sum_{l=1}^{n}\left|Z F C_{\alpha_{k}}(\Delta t_{l})-Z F C_{\alpha_{k}}^{\prime}(\Delta t_{l})\right|, $$ 

where

 $$ Z F C_{\alpha_{k}}(\Delta t_{l})=\sum_{j=1}^{\mathcal{M}}\omega_{l}\kappa_{l}^{\alpha_{k}}(t_{b_{j}},t_{d_{j}})_{\alpha_{k}}, $$ 

corresponding to zigzag persistence diagram  $ PDz_{\alpha_{k}} $  and  $ ZFC'_{\alpha_{k}}(\Delta t_{l}) $  is its counterpart corresponding to the perturbed  $ PDz_{\alpha_{k}}' $ .

From Proposition 3.2 of Chen et al. (2022), we have

 $$ \left|ZFC_{\alpha_{k}}(\Delta t_{l})-ZFC^{\prime}_{\alpha_{k}}(\Delta t_{l})\right|\leq w_{l}L_{l}\mathcal{W}_{1}\big(\mathbf{PDz}_{\alpha_{k}},\mathbf{PDz}_{\alpha_{k}}\big)<w_{l}L_{l}\epsilon_{k}. $$ 

Hence.

 $$ \begin{align*}\|Z S-Z S^{\prime}\|_{\infty}\leq\max_{1\leq k\leq m}\sum_{l=1}^{n}w_{l} L_{l}\mathcal{W}_{1}\big(\mathbf{PDz}_{\alpha_{k}},\mathbf{PDz}_{\alpha_{k}}\big)\\\leq L\max_{1\leq k\leq m}\mathcal{W}_{1}\big(\mathbf{PDz}_{\alpha_{k}},\mathbf{PDz}_{\alpha_{k}}\big)\Bigg(\sum_{l=1}^{n}\omega_{l}\Bigg)\\\leq L\max_{1\leq k\leq m}\mathcal{W}_{1}\big(\mathbf{PDz}_{\alpha_{k}},\mathbf{PDz}_{\alpha_{k}}\big)\leq L\epsilon,\end{align*} $$ 

where  $ L = \max 1 \leq l \leq nL_{l} $  and  $ \epsilon = \max_{1 \leq k \leq m} \epsilon_{k} $ .

## C BASELINES AND EXPERIMENTAL SETUP

Baselines. We use the following popular models for spatio-temporal graph forecasting as baselines: (i) 6 probabilistic methods: (1) Latent Ordinary Differential Equations (ODE) Rubanova et al. (2019), (2) DeepAR (which is a forecasting method based on autoregressive recurrent neural networks) Salinas et al. (2020), (3) Conditional Score-based Diffusion models for Imputation (CSDI) Tashiro et al. (2021), (4) TimeGrad (which is an autoregressive model for multivariate probabilistic time series forecasting) Rasul et al. (2021), (5) MC Dropout (which uses the MC Dropout Gal et al. (2017) for probabilistic spatio-temporal forecasting) Gal et al. (2017); Wu et al. (2021), and (6) DiffSTG (which is a non-autoregressive framework) Wen et al. (2023); and (ii) 6 GNN-based models: (1) Spatio-Temporal Graph Convolutional Networks (STGCN) Yu et al. (2018), Diffusion Convolutional Recurrent Neural Network (DCRNN) Li et al. (2018), Adaptive Graph Convolutional Recurrent Network (AGCRN) Bai et al. (2020), Spectral Temporal Graph Neural Network (StemGNN) Cao et al. (2020), STID Shao et al. (2022), and Fourier Graph Neural Network (FourierGNN) Yi et al. (2024b). For graph classification, we compare our ZS-DM with 15 state-of-the-art (SOTA) baselines including: (1) Graphlet Kernel (GL) Shervashidze et al. (2009), (2) Weisfeiler-Lehman Sub-tree Kernel (WL) Shervashidze et al. (2011), (3) Deep Graph Kernels (DGK) Yanardag & Vishwanathan (2015), (4) node2vec Grover & Leskovec (2016), (5) sub2vec Adhikari et al. (2018), (6) graph2vec Narayanan et al. (2017), (7) InfoGraph Sun et al. (2019), (8) Graph Contrastive Learning (GraphCL) You et al. (2020), (9) Adversarial-Graph Contrastive Learning (AD-GCL) Suresh et al. (2021), (10) Rationale-aware Graph Contrastive Learning (RGCL) Li et al. (2022), (11) Graph Contrastive Learning scheme with Topology Augmentation guided by the Graph Spectrum (GCL-TAGS) Lin et al. (2022), (12) Masked Graph Autoencoder (GraphMAE) Hou et al. (2022), (13) CW Networks (CWN) Bodnar et al. (2021), (14) Topological Graph Neural Networks (TOGL) Horn et al. (2022), and (15) Directional Diffusion Models (DDM) Yang et al. (2024).

## D ROBUSTNESS AND SENSITIVITY ANALYSIS

Robustness to Noise and Varying Sizes of Training Data We also perform experiments on reducing the training set from 90% to 80% and 70%. We find that the ZS-DM gains are higher for lower training sizes, i.e. on average for MUTAG and BZR 0.22% for 90% training size (Table 2 main body), 1.58% for 80% training size (Table 9 below), and 1.94% for 70% training size (Table 10 below). Furthermore, variability of ZS-DM tends to be noticeably lower than runner up SOTAs, i.e., up to 1.5-2 times lower, suggesting that the ZP idea may be helpful for uncertainty quantification. More

<div style="text-align: center;">Table 9: Performance comparison of robustness study with 80% training set.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>MUTAG</td><td style='text-align: center;'>BZR</td></tr><tr><td style='text-align: center;'>AD-GCL</td><td style='text-align: center;'>82.16±2.09</td><td style='text-align: center;'>80.43±1.55</td></tr><tr><td style='text-align: center;'>RGCL</td><td style='text-align: center;'>82.77±2.34</td><td style='text-align: center;'>81.35±2.00</td></tr><tr><td style='text-align: center;'>DDM</td><td style='text-align: center;'>83.38±2.85</td><td style='text-align: center;'>81.21±1.85</td></tr><tr><td style='text-align: center;'>ZS-DM (ours)</td><td style='text-align: center;'>84.59±1.47</td><td style='text-align: center;'>82.78±1.29</td></tr></table>

<div style="text-align: center;">Table 10: Performance comparison of robustness study with 70% training set.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>MUTAG</td><td style='text-align: center;'>BZR</td></tr><tr><td style='text-align: center;'>AD-GCL</td><td style='text-align: center;'>79.50±2.23</td><td style='text-align: center;'>74.28±2.22</td></tr><tr><td style='text-align: center;'>RGCL</td><td style='text-align: center;'>80.35±1.68</td><td style='text-align: center;'>75.45±1.85</td></tr><tr><td style='text-align: center;'>DDM</td><td style='text-align: center;'>80.78±2.07</td><td style='text-align: center;'>76.92±2.88</td></tr><tr><td style='text-align: center;'>ZS-DM (ours)</td><td style='text-align: center;'>82.23±1.57</td><td style='text-align: center;'>78.58±1.04</td></tr></table>

generally, these findings suggest that the ZS approach is capable of capturing some fine-grained latent information on the graph structure that the more conventional approaches cannot, but the role of such finer-grained information diminishes as the sample size increases.

<div style="text-align: center;">Table 11: Sensitivity of graph classification with different choices of scales.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Data</td><td style='text-align: center;'>degree</td><td style='text-align: center;'>betweenness</td><td style='text-align: center;'>closeness</td></tr><tr><td style='text-align: center;'>MUTAG</td><td style='text-align: center;'>91.68±0.34 $ ^{*} $</td><td style='text-align: center;'>90.70±0.83</td><td style='text-align: center;'>88.40±0.37</td></tr><tr><td style='text-align: center;'>BZR</td><td style='text-align: center;'>86.20±0.12 $ ^{*} $</td><td style='text-align: center;'>83.35±1.93</td><td style='text-align: center;'>82.52±0.82</td></tr></table>

<div style="text-align: center;">Table 12: Performance comparison among ZS features with different dimensions.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Data</td><td style='text-align: center;'>0-dim</td><td style='text-align: center;'>1-dim</td><td style='text-align: center;'>0- &amp; 1-dim</td></tr><tr><td style='text-align: center;'>MUTAG</td><td style='text-align: center;'>91.68±0.34*</td><td style='text-align: center;'>88.91±0.47</td><td style='text-align: center;'>90.17±0.12</td></tr><tr><td style='text-align: center;'>BZR</td><td style='text-align: center;'>86.20±0.12*</td><td style='text-align: center;'>84.34±0.90</td><td style='text-align: center;'>85.30±0.36</td></tr></table>

Sensitivity Analysis to the Choice of Filtration Scales. Table 11 shows the sensitivity of graph classification results to the choice of scales, i.e., using node-degree, node-betweenness, and node-closeness scores. In general, we find that the performance for more homogenous graphs is less sensitive to filtrations. For sparser and more heterogeneous graphs, degree-based or power filtrations are often the preferred choices. We have also conducted additional experiments to compare ZS and traditional persistence. From Table 7, we observe that our diffusion model with ZS always outperforms with highly statistically significant gains the diffusion model with traditional persistence on both MUTAG and BZR datasets. Additionally, in experiments of graph classification (see Table 12), we can incorporate 0-dimensional feature, 1-dimension feature, or both 0- and 1-dimensional features into the model. However, we found that our ZS-DM model with 0-dimensional features always outperforms other scenarios. This phenomenon can be potentially explained by stronger signal yielded by 0-dimensional features and much higher representation of 0-dimensional features.

## E ADDITIONAL DETAILS OF ZIGZAG SPAGHETTI AND EXPERIMENTS

Zigzag spaghetti has two main advantages: first, it allows for capturing essential topological characteristics of a dynamic object over time at all resolution scales simultaneously, and, second, it is easily tractable and computationally efficient (relative to other time-aware topological summaries). To justify the second point, we present the running time comparison among ZS, ZPI, and ZFC for graph classification on MUTAG. The obtained results suggest that ZS-DM achieves the best performance and also delivers competitive running time (in seconds):

<div style="text-align: center;">Table 13: Running time and performance comparison.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Data</td><td style='text-align: center;'>ZS-DM (ours)</td><td style='text-align: center;'>ZPI-DM</td><td style='text-align: center;'>ZFC-DM</td></tr><tr><td style='text-align: center;'>MUTAG</td><td style='text-align: center;'>0.21 sec (91.68 $ \pm $ 0.34)</td><td style='text-align: center;'>0.37 sec (90.52 $ \pm $ 0.63)</td><td style='text-align: center;'>0.18 sec (90.73 $ \pm $ 0.59)</td></tr></table>

<div style="text-align: center;"><img src="imgs/img_in_image_box_289_272_951_664.jpg" alt="Image" width="54%" /></div>


<div style="text-align: center;">Figure 2: Visualization of zigzag persistence, zigzag persistence diagram and zigzag filtration curves.</div>


In addition, the complexity of zigzag persistence on graphs can be improved for a subclass of graphs satisfying certain assumptions on the size of the unions in the filtration, i.e. of the size of the union of all graphs in the filtration is  $ \Omega m^{\epsilon} $  for any fixed  $ \epsilon $ , i.e.,  $ 0 < \epsilon < 1 $  Dey & Hou (2023). However, this is not true for a general class of graphs, and some other more drastic approaches are needed to improve scalability of zigzag persistence on denser graphs. One such potential direction is to compute zigzag persistence based only on landmarks rather than on all nodes and then to use Dowker or witness complexes. This approach has not been yet investigated in computational topology and its theoretical guarantees are yet unknown. We believe though that it is a very promising approach that can fundamentally shift the scalability problem for ZP.

Furthermore, based on Qin et al. (2023), we incorporate its sparse denoising network (which contains a graph transformer architecture with a sparse attention mechanism) into our zigzag spaghetti approach and have conducted additional experiments on MUTAG and BZR datasets. We call the zigzag spaghetti model with sparse denoising network  $ ZS-DM_{new} $ . From the Table 14, we observe that  $ ZS-DM_{new} $  always outperforms the previous version (i.e.,  $ ZS-DM_{old} $ ) on both MUTAG and BZR datasets. Thanks very much for this excellent suggestion for further improvement of the ZS approach.

From the Figure 2 (c), we observe that although overall there are similarities in curves for all scales, we find that, starting from time 3, the time evolving graphs deliver consistency in terms of topological characteristics of order 1, while there is a notable variability among topological characteristics over the time period from time point 0 to time point 3. Furthermore, another notable topological feature is observed over the time period 1-2, but its lifespan is relatively short.

<div style="text-align: center;">Table 14: ZS-DM with sparse denoising network.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>MUTAG</td><td style='text-align: center;'>BZR</td></tr><tr><td style='text-align: center;'>ZS-DM_{old}</td><td style='text-align: center;'>91.68±0.34</td><td style='text-align: center;'>86.20±0.12</td></tr><tr><td style='text-align: center;'>ZS-DM_{new}</td><td style='text-align: center;'>92.15±0.32</td><td style='text-align: center;'>86.39±0.10</td></tr></table>