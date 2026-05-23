## APPENDIX OVERVIEW

In the following, we will provide supplementary materials to better illustrate our methods and experiments.

• Section.A provides the algorithm detail of the mixture-of-expert model

• Section.B provides theoretical details

• Section.C provides more experiment results.

• Section.D provides the information of computing infrastructure for all experiments.

- Section.E tests the scalability and computational efficiency of our method using varying sample sizes for different dataset.

- Section.F explores the impact of spatial and temporal grid resolutions on both model performance and computational costs.

- Section.G provides the experiment results for the impact of features on the human decision-making process.

## A ALGORITHM DETAILS

### A.1 SPATIAL-TEMPORAL EMBEDDING

We adopt a spatial-temporal embedding method akin to that described in (Alexey, 2020; Aksan et al., 2021). Further details can be found in Fig.1. Initially, the region is segmented into distinct blocks. To encode block order, we introduce sinusoidal positional encoding for area position embedding. Subsequently, linear embedding is utilized for spatial information, typically latitude and longitude. Temporal information is encoded using sinusoidal positional encoding as described in (Zuo et al., 2020). In the context of decision-making, relevant static features can be encoded using one-hot semantic encoding, while dynamic features are encoded linearly. The embeddings for spatial, temporal, and relevant feature information are then combined via element-wise addition to generate the comprehensive embedding.

First, we divide the area into disjoint blocks. To inject a notation of block ordering we add sinusoidal positional encoding for these area position embedding. Then considering the spatial information which are usually latitude and longitude, we apply linear embedding. For encoding temporal information, we also adopt sinusoidal positional encoding a (Zuo et al., 2020). Considering other relevant features related to the decision-making process, we can apply one-hot semantic encoding for static feature and linear encoding for dynamic features. Finally, the embedding for spatial information, temporal information, and relevant feature information then directly element-wise addition together to obtain the overall embedding.

This approach is similar to the use of positional embeddings and feature embeddings in attention mechanisms, where initial embeddings are transformed through linear projections to capture more nuanced information. By combining the base embeddings A and B with these flexible projections, our model can more accurately represent and adapt to diverse preference patterns and social influences, enriching the overall decision-making framework.

### A.2 OVERALL ALGORITHM

## B THEORETICAL DETAILS

### B.1 DETAILS OF  $ \alpha $ -ENTMAX

 $$ \alpha-entmax\ (z):=\underset{\boldsymbol{p}\in\Delta^{M}}{\argmax}\quad\boldsymbol{p}^{\top}\boldsymbol{z}+H_{\alpha}^{\mathrm{T}}(\boldsymbol{p}), $$ 

where  $ \triangle^{M}:=\left\{p\inR^{M}:\sum_{i}p_{i}=1\right\} $  is the probability simplex, and, for  $ \alpha\geq1 $ ,  $ H_{\alpha}^{T} $  is the Tsallis continuous family of entropies (Tsallis, 1988):

 $$ \mathrm{H}_{\alpha}^{\mathrm{T}}(\boldsymbol{p}):=\begin{cases}\frac{1}{\alpha(\alpha-1)}\sum_{j}\left(p_{j}-p_{j}^{\alpha}\right),&\alpha\neq1\\ -\sum_{j}p_{j}\log p_{j},&\alpha=1\end{cases} $$ 

Algorithm 1 Learning the Model Parameters for the Mixture-of-Experts Model

1: Input: Observed data  $ \{y_{i,m}\}_{i=1}^{N} $ , initial parameters  $ \theta = [\pi, A, B, \{[\alpha^{h}, \tau^{h}, W_{h}^{a}, W_{h}^{b}, U_{h}]\}_{h \in [H]}] $ 

2: Output: Optimized model parameters  $ \theta^{*} $ 

3: Initialization: Initialize  $ \theta $  randomly or heuristically.

4: Description: A and B serve as shared feature embeddings that encode the positional and contextual information necessary for understanding the preference distribution in generating the events.

5: for each expert  $ h \in [H] $  do

6: Compute gating function:

 $$  g^{h} = \mathbf{g}_{\alpha^{h}, \tau^{h}} \left( A^{h} \left( B^{h} \right)^{\top} \mathbf{1} \right),  $$ 

where  $ A^{h} = AW_{a}^{h} $  and  $ B^{h} = BW_{b}^{h} $ .

7: end for

8: for each event  $ i \in [N] $  and each pair  $ m \in [M] $  do

9: Compute probability:

 $$  P_{m} = \sum_{h=1}^{H} \pi^{h} \frac{g_{m}^{h} \exp \left( U_{m}^{h} \right)}{\sum_{m' = 1}^{M} g_{m'}^{h} \exp \left( U_{m'}^{h} \right)}.  $$ 

10: end for

11: Optimize: Maximize the likelihood function:

 $$  \mathcal{L}(\theta) = \prod_{i=1}^{N} \prod_{m=1}^{M} \left( P_{i,m} \right)^{y_{i,m}}  $$ 

to update  $ \theta $  using gradient descent or a similar optimization method.

12: Output: Optimized parameters  $ \theta^{*} $ .

This family contains the well-known Shannon and Gini entropies, corresponding to the cases  $ \alpha = 1 $  and  $ \alpha = 2 $ , respectively.

### B.2 Proof of Theorem 1

Lemma 1.  $ \alpha $ -entmax $ _{m}(z) $  is 1-Lipschitz continuous w.r.t.  $ l_{2} $  norm.

Proof. Using the variational representation of  $ \alpha $ -entmax,

 $$ \alpha-entmax(\boldsymbol{z})=\underset{\boldsymbol{p}\in\Delta^{M}}{\argmax}\quad\boldsymbol{p}^{\top}\boldsymbol{z}+H_{\alpha}^{\mathrm{T}}(\boldsymbol{p}) $$ 

By the Envelope theorem, its subgradient belongs to  $ \Delta^{M} $ , and hence is bounded by  $ \max_{p\in\Delta^{M}}||p||_{2}=1 $ . As a result,  $ \alpha $ -entmax is 1-Lipschitz continuous in  $ l_{2} $ -norm. ☐

### Lemma 2. Define a function

 $$ f_{m}(\boldsymbol{z},\boldsymbol{u}):=\frac{\alpha-entmax_{m}(\boldsymbol{z})\exp(\boldsymbol{u}_{m})}{\sum_{m^{\prime}=1}^{M}\alpha-entmax_{m^{\prime}}(\boldsymbol{z})\exp(\boldsymbol{u}_{m^{\prime}})}. $$ 

Then  $ f_{m} $  is L-Lipschitz, where  $ L := \frac{1}{l} $ . l is the minimum value of positive  $ \alpha $ -entmax $ _{m}(\mathbf{z}) $ .

Proof. Note that  $ \alpha\text{-entmax}(z) $  is a sparse gating function, with a set of  $ [M_{+}]\subset[M] $  non-zero items. We consider  $ \alpha\text{-entmax}(z)>0 $  and  $ \alpha\text{-entmax}_{m}(z)=0 $  separately.

(i) For m such that  $ \alpha\text{-entmax}_{m}(\mathbf{z}) > 0 $ ,

 $ f_{m}(\mathbf{z}, \mathbf{u}) $  can be written as  $ \text{softmax}_{m}(\log(\alpha\text{-entmax}(\mathbf{z})) + \mathbf{u}) $ . By lemma 1, we can derive the Lipschitz constant of  $ \log(\alpha\text{-entmax}_{m}(\mathbf{z})) $ .

 $$ \nabla log(\alpha-entmax_{m}(\boldsymbol{z}))=\frac{\nabla\alpha-entmax_{m}(\boldsymbol{z})}{\alpha-entmax_{m}(\boldsymbol{z})}\leq\frac{\nabla\alpha-entmax_{m}(\boldsymbol{z})}{\min(\alpha-entmax_{m}(\boldsymbol{z}))} $$ 

Let's denote  $ \min(\alpha\text{-}entmax_{m}(z)) $  as l. Then  $ \log(\alpha\text{-}entmax_{m}(z)) $  is  $ \frac{1}{l} $ -Lipschitz. Adding U does not change the Lipschitz constant of a function. Since  $ softmax(\cdot) $  is a 1-Lipschitz function, the composite function  $ softmax_{m}(\log(\alpha\text{-}entmax(z)) + u) $  is also  $ \frac{1}{l} $ -Lipschitz.

(ii) For m such that  $ \alpha\text{-entmax}_{m}(z)=0 $ ,

we consider another  $ \alpha\text{-entmax}_{m}(z') $ . If  $ \alpha\text{-entmax}_{m}(z') = 0 $ ,  $ \frac{1}{l} $ -Lipschitz also holds. If  $ \alpha\text{-entmax}_{m}(z') > 0 $ , by Mean Value Theorem, there exists  $ \alpha\text{-entmax}_{m}(z'') > 0 $ ,  $ z'' \in [z, z'] $ , where previous analysis holds.

With the above definition of  $ f_{m} $ , for a sample  $ (A^{i}, B^{i}) $ , the choice probability of option m for the latent class h can be written as

 $$ \sigma_{m}\left(A^{i},B^{i};W_{A}^{h},W_{B}^{h},U^{h}\right)=f_{m}\big(\{A_{j}^{n}W_{A}^{h}(B^{i}W_{B}^{h})^{\top}\mathbf{1}\}_{j=1,\ldots,M},U^{h}\big), $$ 

where  $ A_{i}^{n} $  denotes the j-th row of the matrix  $ A^{i} $ 

Let us derive a bound for the empirical Rademacher complexity. Since a linear functional of the probability distribution  $ \pi $  attains its supremum at the point mass, the above expectation equals

 $$ \mathbb{E}_{\epsilon}\left[\sup_{\|W_{A}W_{B}^{\top}\|_{F}\leq C_{W},\|U\|_{F}\leq C_{U}}\frac{1}{N}\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}f_{m}\big(A^{i}W_{A}(B^{i}W_{B})^{\top}\mathbf{1},U\big)\right], $$ 

where  $ M_{n} $  be the subset of [M] for which  $ f_{m} \neq 0 $ . Since  $ f_{m} $  is L-Lipschitz, using the vector contraction lemma (Maurer, 2016), it holds that

 $$ \Re_{n}(\mathcal{W})\leq\sqrt{2}L\cdot\mathbb{E}_{\epsilon}\left[\sup_{\boldsymbol{w}\in\mathcal{W}}\frac{1}{N}\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\left(\epsilon_{nm j}^{W}A_{j}^{n}W_{A}(B^{i}W_{B})^{\top}\mathbf{1}+\epsilon_{nm j}^{U}U_{j}\right)\right]. $$ 

By the additive separability of the Rademacher complexity, the expectation above equals

 $$ \mathbb{E}_{\epsilon}\left[\sup_{\|W_{A}W_{B}^{\top}\|_{F}\leq C_{W}}\frac{1}{N}\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\epsilon_{nmj}^{W}A_{j}^{n}W_{A}(B^{i}W_{B})^{\top}\mathbf{1}\right]+\mathbb{E}_{\epsilon}\left[\sup_{\|U\|_{2}\leq C_{U}}\frac{1}{N}\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\epsilon_{nmj}^{U}U_{j}^{h}\right]. $$ 

For the first term, we rewrite it as

 $$ \sup_{\|W_{A}W_{B}^{\top}\|_{F}\leq C_{W}}\left\langle W_{A}W_{B}^{\top},\frac{1}{N}\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\epsilon_{nmj}^{W}(B^{i})^{\top}\mathbf{1}A_{j}^{n}\right\rangle. $$ 

which, by Cauchy-Schwarz inequality, is upper bounded by

 $$ \frac{C_{W}}{N}\left\|\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\epsilon_{nmj}^{W}(B^{i})^{\top}\mathbf{1}A_{j}^{n}\right\|_{F}. $$ 

By Jensen’s inequality,

 $$ \begin{align*}\mathbb{E}_{\epsilon}\left[\left\|\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\epsilon_{nmj}^{W}(B^{i})^\top\mathbf{1}A_{j}^{n}\right\|_{F}\right]&\leq\sqrt{\mathbb{E}_{\epsilon}\left[\left\|\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\epsilon_{nmj}^{W}(B^{i})^\top\mathbf{1}A_{j}^{n}\right\|_{F}^{2}\right]}\\&\leq\sqrt{\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\|(B^{i})^\top\mathbf{1}A_{j}^{n}\|_{F}^{2}}\\&\leq\sqrt{\kappa\sum_{n=1}^{N}\sum_{j=1}^{M}\|(B^{i})^\top\mathbf{1}A_{j}^{n}\|_{F}^{2}}\end{align*} $$ 

Observe that

 $$ \sum_{j=1}^{M}\|(\boldsymbol{B}^{i})^{\top}\mathbf{1}\boldsymbol{A}_{j}^{n}\|_{F}^{2}=\|(\boldsymbol{B}^{i})^{\top}\mathbf{1}\|_{2}^{2}\cdot\sum_{j=1}^{M}\|\boldsymbol{A}_{j}^{n}\|_{2}^{2}\leq\|\boldsymbol{B}^{i}\|_{F}\cdot\|\boldsymbol{A}^{i}\|_{F}. $$ 

Hence, the first term of (13) is bounded by

 $$ \frac{\sqrt{\kappa}\nu^{2}C_{W}}{\sqrt{N}}. $$ 

Similarly, we can show that the second term of (13) is bounded by

 $$ \begin{align*}\mathbb{E}_{\epsilon}\left[\sup_{\|U\|_{2}\leq C_{U}}\frac{1}{N}\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\sum_{j=1}^{M}\epsilon_{nmj}^{U}U_{j}\right]&\leq\frac{C_{U}}{N}\mathbb{E}_{\epsilon}\left[\left\|\sum_{n=1}^{N}\sum_{m\in\mathcal{M}_{n}}\epsilon_{nm.}^{U}\right\|_{2}\right]\\&\leq\frac{\sqrt{\kappa M}C_{U}}{\sqrt{N}}.\end{align*} $$ 

## C MORE EXPERIMENT RESULTS

### C.1 LEARNED UTILITY FUNCTION ON NYC CRIME DATASET

The results are shown in Fig.5. Seeing from the value, the learn utility function play an important role in adjusting the choices based on social characteristics. The patterns of U scores are similar with the mixture patterns adjusted by utility scores.

<div style="text-align: center;"><img src="imgs/img_in_image_box_215_161_1003_654.jpg" alt="Image" width="64%" /></div>


#### Figure 5: Learned U function for each expert in the experiment January 1, 2024, in New York City. C.2 HOW TO EXPLAIN RESULTS OF LGCP

We fit our model using a new objective function based on the least squared error between estimated probability and the probability from the LGCP. This approach allows us to interpret the expert patterns learned by our model to explain the already fitted LGCP model.

We conduct non-negative matrix factorization on the learned intensity matrix  $ \Lambda\inR^{T\times S} $  from LGCP.

 $$ \Lambda\approx WB $$ 

where  $ W \in R^{T \times H} $  is the weight matrix, and  $ B \in H^{T \times S} $  is the basis matrix. S is the number of spatial grids, and T is the number of temporal intervals. H is the number of mixture, which is set to be same as our model.

### C.3 FIGURES

We fit our model on two other real-world spatial-temporal datasets. Fig.6 and Fig.7 are the results on Chicago Crime dataset. Fig.8 and Fig.9 are the results on Shanghai mobike renting dataset.

## D COMPUTING INFRASTRUCTURE

All synthetic data experiments, as well as the real-world data experiments, including the comparison experiments with baselines, are performed on Ubuntu 20.04.3 LTS system with Intel(R) Xeon(R) Gold 6248R CPU @ 3.00GHz, 227 Gigabyte memory.

## E Scalability

Across all experiments, as the dataset sample size increases, both evaluation metrics, aRMSE and MAPE, decrease for the prediction task. Taking the NYC dataset as an example, with an increase in data size from the current 732 samples to 5561 samples, aRMSE decreases to 2.06, and MAPE decreases to 100.72. The training time required for model convergence remains within acceptable limits on the current computing infrastructure. For the NYC dataset with 5561 samples, the model converges in only 3.8970 hours. Even for the large-scale Mobike dataset with 8786 samples, our model converges and achieves good inference and prediction results after training for approximately 7.1056 hours.

<div style="text-align: center;"><img src="imgs/img_in_image_box_211_175_1006_473.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 6: Comparison of the actual crime frequency and the modeled probability on July 5, 2024, in Chicago.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_214_608_1005_999.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 7: Mixing coefficient  $ \pi^{h} $  (Left bar plot) and mixture pattern adjusted by utility score  $ (g^{h}\exp(U^{h})) $  for different experts (Right heatmaps) on July 5, 2024, in Chicago. The selection of the number of experts is based on empirical experiments.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_212_1157_1007_1351.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 8: Comparison of the actual mobike renting frequency and the modeled probability on August 7, 2016, in Shanghai.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_214_178_1008_538.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 9: Mixing coefficient  $ \pi^{h} $  (Left bar plot) and mixture pattern adjusted by utility score  $ (g^{h}\exp(U^{h})) $  for different experts (Right heatmaps) on August 7, 2016, in Shanghai. The selection of the number of experts is based on empirical experiments.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_696_1008_1174.jpg" alt="Image" width="64%" /></div>


Figure 10: Scalability and the computation time cost of our method. For each dataset, we vary the sample size from small to large scale. Specifically, for NYC Crime dataset, we extract the records from 2024-01-01 – 2024-01-01: 732 samples, 2024-01-01 – 2024-01-03: 1840 samples, 2024-01-01 – 2024-01-05: 2985 samples, 2024-01-01 – 2024-01-07: 4016 samples, 2024-01-01 – 2024-01-10: 5561 samples. For Chicago Crime dataset, we extract the records from 2024-07-05 – 2024-07-05: 861 samples, 2024-07-05 – 2024-07-07: 2434 samples, 2024-07-05 – 2024-07-08: 3207 samples, 2024-07-05 – 2024-07-10: 4578 samples, 2024-07-05 – 2024-07-11: 5321 samples. For Shanghai Mobike dataset, we extract the records from 2016-08-02 – 2016-08-02: 1457 samples, 2016-08-02 – 2016-08-03: 3347 samples, 2016-08-02 – 2016-08-04: 5054 samples, 2016-08-02 – 2016-08-05: 6602 samples, 2016-08-02 – 2016-08-06: 8786 samples.

## F IMPACT OF SPATIAL AND TEMPORAL RESOLUTIONS

<div style="text-align: center;"><img src="imgs/img_in_chart_box_213_213_476_416.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_479_215_742_416.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_747_216_1008_417.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">Figure 11: Impact of spatial and temporal resolution on model performance and computational efficiency. We vary the resolution for 5 cases: Case-1: 2 (12 hour) time grids, 25 (5 * 5) region blocks, Case-2: 4 (6 hour) time grids, 100 (10 * 10) region blocks, Case-3: 6 (4 hour) time grids, 225 (15 * 15) region blocks, Case-4: 8 (3 hour) time grids, 400 (20 * 20) region blocks, Case-5: 12 (2 hour) time grids, 625 (25 * 25) region blocks. And we also vary the sample size for 732, 2985, and 5561 samples.</div>


We have explored the impact of spatial and temporal grid resolutions on both model performance and computational costs. We take NYC Crime dataset for an example and the results are presented in Fig.11. With finer resolutions for the time grid and region blocks, the model is expected to capture event patterns more accurately and with greater granularity in time and location. However, our experimental results indicate that increasing the fine-grained spatial and temporal resolution does not significantly enhance the model performance. For instance, when comparing Case-5 with Case-2 using a dataset of 2985 samples, despite Case-5 having finer resolution, the aRMSE only decreases from 2.29 to 2.15, and the MAPE decreases from 104.43 to 101.83. This could be attributed to the overly detailed partitioning of time and space, leading to insufficient instances of events at each time-location pair, thereby impacting the model's effectiveness. Further validation of this observation is evident when varying the sample size within the same case. For Case-2, increasing the sample size from 732 to 5561 results in a more significant improvement in model performance, with the aRMSE decreasing from 2.34 to 2.06 and the MAPE decreasing from 107.94 to 100.72. This underscores the substantial impact of increasing dataset size on model effectiveness. Hence, the results presented in our paper reflect a trade-off in selecting resolution based on balancing model performance and the level of detail in capturing time-location pair patterns. In the revised version of our paper, we will incorporate these experiments to demonstrate the generality of our approach.

## G THE IMPACT OF FEATURES ON THE HUMAN DECISION-MAKING PROCESS

In our experiment, we consider the severity of the crime, suspect race, and suspect gender as key categorical features. The utility score for each time-location pair is determined through regression analysis involving these features, with the coefficients reflecting the magnitude of impact on the utility score. For each expert, the utility score patterns are different, reflecting different preference patterns. We take NYC Crime dataset with original 732 samples (temporal and spatial resolutions are 4 time slots and  $ 10 \times 10 $  area blocks) as an example and the results are shown in Fig.12, Fig.13, and Fig.14.

To illustrate the “human decision process” in a more clear way, we take an example for the top-5 crime event time location-pair for different races of expert-1 with largest utility score. Distinct patterns emerge based on the time and location of crime events across various racial groups. Black suspects tend to engage in criminal activities during the early morning or late night hours, while White Hispanic suspects are less active in criminal activities during the early morning hours. The timing of criminal activities among suspects of other races is more varied. At the regional level, the concentration areas for criminal activities among suspects of different races vary significantly. By incorporating social norms or individual information of suspects into the utility function, our approach better captures the role of individual differences and human decision-making processes in engaging in criminal activities.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_209_1004_584.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 12: Impact of severity of the crime on utility function.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_765_1003_1339.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 13: Impact of suspect race on utility function.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_220_1004_610.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 14: Impact of suspect gender on utility function.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Race</td><td style='text-align: center;'>W</td><td style='text-align: center;'>AI/AN</td><td style='text-align: center;'>WH</td></tr><tr><td style='text-align: center;'>Top-1</td><td style='text-align: center;'>0h-6h, (40.577, -73.845)</td><td style='text-align: center;'>12h-18h, (40.808, -73.895)</td><td style='text-align: center;'>12h-18h, (40.770, -73.794)</td></tr><tr><td style='text-align: center;'>Top-2</td><td style='text-align: center;'>18h-24h, (40.616, -74.045)</td><td style='text-align: center;'>0h-6h, (40.693, -73.995)</td><td style='text-align: center;'>18h-24h, (40.847, -73.845)</td></tr><tr><td style='text-align: center;'>Top-3</td><td style='text-align: center;'>0h-6h, (40.693, -73.845)</td><td style='text-align: center;'>0h-6h, (40.693, -73.995)</td><td style='text-align: center;'>6h-12h, (40.693, -73.945)</td></tr><tr><td style='text-align: center;'>Top-4</td><td style='text-align: center;'>12h-18h, (40.654, -73.945)</td><td style='text-align: center;'>18h-24h, (40.770, -73.895)</td><td style='text-align: center;'>6h-12h, (40.847, -73.845)</td></tr><tr><td style='text-align: center;'>Top-5</td><td style='text-align: center;'>18h-24h, (40.654, -73.895)</td><td style='text-align: center;'>12h-18h, (40.885, -73.895)</td><td style='text-align: center;'>12h-18h, (40.847, -73.845)</td></tr></table>

<div style="text-align: center;">Table 3: Top-5 crime event time location-pair for different races of expert-1 with largest utility score on NYC Crime Dataset. We use abbreviations to denote different races, where W: White, AI/AN: American Indian/Alaskan Native, and WH: White Hispanic. The time-location pairs are recorded as: time, (Lat., Lon.)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Race</td><td style='text-align: center;'>B</td><td style='text-align: center;'>BH</td><td style='text-align: center;'>A/PI</td></tr><tr><td style='text-align: center;'>Top-1</td><td style='text-align: center;'>0h-6h, (40.731, -73.995)</td><td style='text-align: center;'>18h-24h, (40.693, -73.744)</td><td style='text-align: center;'>0h-6h, (40.538, -74.196)</td></tr><tr><td style='text-align: center;'>Top-2</td><td style='text-align: center;'>0h-6h, (40.731, -73.895)</td><td style='text-align: center;'>0h-6h, (40.616, -74.146)</td><td style='text-align: center;'>0h-6h, (40.808, -73.845)</td></tr><tr><td style='text-align: center;'>Top-3</td><td style='text-align: center;'>0h-6h, (40.847, -73.895)</td><td style='text-align: center;'>0h-6h, (40.770, -73.995)</td><td style='text-align: center;'>18h-24h, (40.616, -73.795)</td></tr><tr><td style='text-align: center;'>Top-4</td><td style='text-align: center;'>18h-24h, (40.731, -73.995)</td><td style='text-align: center;'>6h-12h, (40.808, -73.845)</td><td style='text-align: center;'>12h-18h, (40.577, -73.995)</td></tr><tr><td style='text-align: center;'>Top-5</td><td style='text-align: center;'>0h-6h, (40.808, -73.845)</td><td style='text-align: center;'>18h-24h, (40.770, -73.945)</td><td style='text-align: center;'>12h-18h, (40.731, -73.945)</td></tr></table>

<div style="text-align: center;">Table 4: Top-5 crime event time location-pair for different races of expert-1 with largest utility score on NYC Crime Dataset. We use abbreviations to denote different races, where B: Black, BH: Black Hispanic, and A/PI: Asian/Pacific Islander. The time-location pairs are recorded as: time, (Lat., Lon.)</div>