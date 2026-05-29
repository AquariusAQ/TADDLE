

# EFFICIENT TIME SERIES FORECASTING VIA HYPER-COMPLEX MODELS AND FREQUENCY AGGREGATION

Anonymous authors

Paper under double-blind review

## ABSTRACT

Time-series forecasting is a long-standing challenge in statistics and machine learning, with one of the key difficulties being the ability to process sequences with long-range dependencies. A recent line of work has addressed this by applying the short-time Fourier transform (STFT), which partitions sequences into multiple subsequences and applies a Fourier transform to each separately. We propose the Frequency Information Aggregation (FIA-Net), a model that can utilize two backbone architectures: the Window-Mixing MLP (WM-MLP), which aggregates adjacent window information in the frequency domain, and the Hyper-Complex MLP (HC-MLP), which treats the set of STFT windows as hyper-complex (HC) valued vectors. And employ HC algebra to efficiently combine information from all STFT windows altogether. Furthermore, due to the nature of HC operations, the HC-MLP uses up to three times fewer parameters than the equivalent standard window aggregate-gation method. We evaluate the FIA-Net on various time-series benchmarks and show that the proposed methodologies outperform existing state-of-the-art meth-ods in terms of both accuracy and efficiency. Our code is publicly available on https://anonymous.4open.science/r/research-1803/.

## 1 INTRODUCTION

Time series forecasting (TSF) is a long-standing challenge that plays a key role in various domains, such as energy management Rajagukguk et al. (2020), traffic prediction Chen & Chen (2019), and financial analysis Sezer et al. (2020). With the development of deep learning, myriad neural network (NN) architectures have been proposed and have gradually improved the accuracy on the TSF problem. Two key architectures that have been used for TSF are recurrent NNs (RNNs) Zhang & Man (1998); Graves (2012); Chung et al. (2014) and transformers Vaswani et al. (2018); Zhou et al. (2022a); Wu et al. (2021); Zhang & Yan (2023), each of which aims to capture long-term dependencies through a different functional feature extraction procedure. While both methods were proven useful, RNNs struggled with long-term dependencies Pascanu et al. (2013) or non-stationary data patterns. While transformer architectures may overlook important temporal information due to permutation invariance Kim et al. (2024), they require many parameters and may suffer from long runtime. Additional NN-based approaches for TSF consider graph NNs (GNNs) Wu et al. (2020) and decomposition models Oreshkin et al. (2019).

Recent advancements have demonstrated promising results in processing and extracting features from the frequency domain Yi et al. (2023a). Techniques leveraging frequency-based transformations have been applied in various contexts, ranging from computational efficiency improvements Wu et al. (2021) to seasonal-trend decomposition Zhou et al. (2022a). To better process the frequency domain data, Yi et al. (2023b) developed a complex-valued MLP, which demonstrated superior capability in capturing both temporal and cross-channel dependencies. To better handle nonstationarities in the data, Shen et al. (2024); Tu et al. (2024); Zeng et al. (2023b) substituted the standard FFT with the Short-Time Fourier Transform (STFT) Gabor (1946), which divides the sequence into separate windows and transforms each window individually into the frequency domain. While showing better suitability for nonstationary time series data, the STFT yields a set of windows, each of which represents exclusive information about the sequence. However, in practice, adjacent windows are highly correlated, albeit processed separately by current STFT-based models.

To incorporate the overlooked shared information, we propose the FIA-Net, a novel TSF model designed to handle long-term dependencies in the data by aggregating information from subsets of STFT windows. The FIA-Net has an MLP backbone that processes the STFT windows in the frequency domain. We propose two novel MLP architectures. The first is termed window-mixing MLP (WM-MLP), which mixes each STFT window with its neighboring bands. The second is the HC-MLP. The HC-MLP leverages HC algebra to efficiently combine information from all STFTs together. By using HC algebra, the FIA-Net is implemented with three times fewer parameters than the equivalent WM-MLP.

The main contributions of this paper are as follows:

• We construct the FIA-Net with the WM-MLP backbone. The resulting TSF model captures inter-window dependencies in the frequency domain and benefits from a forward pass complexity of  $ O(L \log L/p) $  operations, where L is the lookback window length and p is the number of STFT windows.

• We propose a novel HC-MLP backbone that expands the receptive field of the WM-MLP while requiring a fraction of the total parameters.

• To reduce the model size and complexity, we filter the STFT windows, leaving only the top-M frequency components. We show that accuracy is maintained even when M is significantly smaller than the total number of components.

• We provide an array of experiments that demonstrate the performance of the model and its efficiency. We show that the FIA-Net improves upon existing models' accuracy by up to 20%.

• We provide an ablation study, in which we explore the effect of operating over the complex plane and compare the performance of the two considered MLP backbones.

## 2 RELATED WORK

Time-Series Forecasting The first notable works on TSF utilize classical statistical linear models such as ARIMA. Box & Jenkins (1968); Box & Pierce (1970) which consider series decomposition. These were then generalized to a non-linear setting in Watson (1993). To overcome the limitations posed by the classical models, deep learning was incorporated, where initially, sequential deep learning was performed using RNN-based models. Two key RNN models are long-short term memory networks Graves (2012), which introduce a sophisticated gating mechanism, and the DeepAR model Zhang & Man (1998), which connects the RNN model with AR modeling. While RNNs have demonstrated expressive power for sequential modeling, they often suffer from low efficiency and high runtimes in both the forward and backward passes Pascanu et al. (2013). To address these limitations, two popular architectural advancements emerged: transformers and GNNs. Transformer-based approaches such as Informer Zhou et al. (2023), Reformer Kitaev et al. (2020), and PatchTST Nie et al. (2023) leverage the attention mechanism to effectively capture temporal dependencies while introducing innovative methods to reduce the complexity of attention operations.

In contrast, GNNs have been applied to better model dependencies among time series variables by representing them as nodes in a graph. This approach is particularly effective for capturing spatiotemporal patterns. For instance, AGCRN Bai et al. (2020) proposed an adaptive graph convolution mechanism that dynamically adjusts graph structures based on inter-series relationships. Similarly, MTGNN Wu et al. (2020) integrates graph convolutions with temporal convolutional layers to jointly learn spatial and temporal dependencies. However, GNNs were not specifically developed to improve upon RNNs but rather to address unique challenges in spatio-temporal modeling.

Frequency Domain Models for Time Series Forecasting A recent line of work attempts to solve the TFS problem in the frequency domain Yi et al. (2023a), with the purpose of revealing patterns that may be hidden in the time domain. The FEDformer Zhou et al. (2022a) uses a Fourier-based framework to separate trend and seasonal components by leveraging the Fourier Transform on subsequences, allowing it to isolate periodic patterns more effectively. ETSformer Woo et al. (2022) combines exponential smoothing and applies attention in the frequency domain to enhance seasonality modeling by capturing both short- and long-term dependencies. In FiLM Zhou et al. (2022b), Fourier projections are used to reduce noise and emphasize relevant features. Additionally, SFM

<div style="text-align: center;"><img src="imgs/img_in_image_box_222_161_997_490.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure 1: Window Mixing mechanism. An input X is transformed into a set of p STFT windows, which are transformed to the frequency domain and then fed into the WM-MLP, which aggregates adjacent windows. The WM-MLP outputs are then transformed back to the time domain via a real STFT, from which the prediction (red) is obtained.</div>


Zhang et al. (2017) and StemGNN Cao et al. (2020) utilize frequency decomposition and Graph Fourier Transforms to handle complex temporal dependencies in multivariate time series. FRETs Yi et al. (2023b) extends this approach by proposing frequency-domain MLPs to learn complex relationships between the real and imaginary components of the FFT. FREQTSF Shen et al. (2024) uses STFT with attention mechanisms to capture temporal patterns across overlapping time windows. While frequency models, and specifically the recent use of STFT, have shown significant improvements in TFS performance, each STFT window is often processed separately, ignoring the strong correlations between adjacent windows.

Hyper-complex Numbers HC numbers extend the complex number system to higher dimensions Hamilton (1844). Base-4 HC numbers have been widely used in computer graphics to model 3D rotations Parcollet et al. (2016). Base-8 HC numbers have been explored in image classification and compression Parcollet et al. (2016); Luo et al. (2010), developing an HC network that showed favorable performance on popular datasets. The merit of HC numbers to extract relevant information in time-series was explored in Saoud & Al-Marzouqi (2020), in which an HC-net was used to analyze brainwave data, and in Kycia & Niemczynowicz (2024), which explored HC-network for financial data. In this work, we explore the utility of HC architectures for the efficient processing of STFT windows in the frequency domain.

## 3 PROPOSED MODEL : FIA-NET

In this section, we describe FIA-Net, a TSF model that leverages shared information between STFT windows. We begin by discussing the existing gap in current frequency domain TSF methods, followed by a brief introduction to frequency domain MLPs Yi et al. (2022). We then outline the FIA-Net components, presenting the novel complex MLP backbone, discussing a simple frequency compression step that reduces the MLP input dimension, and outline the complete model.

Motivation Even though most real-world time-series data is non-stationary, it may adhere to a piecewise stationary structure, as observed in speech signals? and financial data Fryzlewicz & Cho (2014). This local stationarity allows us to partition the series into stationary correlated STFT subsequences that can be transformed in the frequency domain. The correlation between the STFT sequences has been efficiently utilized in recent works, even though, as we later show, it affects the downstream model accuracy in the task of time prediction.



<div style="text-align: center;"><img src="imgs/img_in_image_box_800_1262_979_1402.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">Figure 2: FD-MLP architecture.</div>


## Frequency Domain MLPs

As we handle complex-valued data, we adopt the frequency domain MLP (FD-MLP) unit from Yi et al. (2023b). The FD-MLP generalizes the simple neuron to operate with complex-valued weights and biases. Incorporating complex MLPs has been shown to improve the model performance as it aligns better with the geometrical structure induced by the complex plane. The FD-MLP unit is visualized in Figure 2. In Section 4, we will discuss the expansion of the FD-MLP for hyper-complex numbers.

### 3.1 ADJACENT INFORMATION AGGREGATION

Consider a sequence  $ X = \{x_{1}, \ldots, x_{L}\} \in R^{D \times L} $  where  $ x_{i} \in R^{D} $ , L is the sequence length, which we refer to as the lookback size, and D is the latent space dimension. Our objective is to predict the next T elements of the sequence  $ \hat{X} = \{\hat{x}_{L+1}, \ldots, \hat{x}_{L+T}\} \in R^{D \times T} $ , where T is a predetermined prediction horizon. We are interested in processing X in the frequency domain. We utilize the STFT, which partitions X into p windows and applies the FFT separately to each window. In addition, we exploit the real-valued inputs to perform a Real STFT, which results in half the frequency coefficients. The STFT for the i-th window is defined as:

 $$ \mathsf{S T F T}\{X\}(\omega,\tau_{i})=\sum_{t=1}^{L}x_{t}w(t-\tau_{i})e^{-j\omega t}, $$ 

Where,  $  w(t - \tau_{i})  $  is the window function centered at the location of the i-th window ( $ i \in \{1, \ldots, p\} $ ),  $ \omega $  represents the angular frequency, and j satisfies  $ j^{2} = -1 $ . Each window is defined by its center  $ \tau_{i} $  and has a size of  $ \frac{N_{FFT}}{2} + 1 $ . The output of the STFT consists of p windows, each producing a spectrum of length  $ \frac{N_{FFT}}{2} + 1 $ .

We propose the window mixing MLP (WM-MLP), which adapts the FD-MLP to properly aggregate neighboring STFT windows to incorporate shared information. Given a set of complex transformed windows  $ \{C_{1},\ldots,C_{p}\} $ , the WM-MLP operates on the ith window  $ C_{i}^{in} $  as follows:

 $$ C_{i}^{\mathsf{o u t}}=\sigma\left(C_{i}^{\mathsf{i n}}W_{i\rightarrow i}+C_{i-1}^{\mathsf{i n}}\overline{{W}}_{(i-1)\rightarrow i}+C_{i+1}^{\mathsf{i n}}\overline{{W}}_{(i+1)\rightarrow i}+B_{i}\right) $$ 

where $\sigma(\cdot)$ is an activation function, $(W_{(i-1)\to i}, W_{i\to i}, W_{(i+1)\to i})_{i=1}^{p}$ are the WM-MLP weight matrices with $C_{j}$ being a matrix of zeros for $j\notin\{1,\ldots,p\}$, and $(B_{i})_{i=1}^{p}$ are the WM-MLP bias vectors, and $\overline{W}$ is the elementwise complex conjugate of $W$. The outputs of the WM-MLP are transformed back to the time domain using the element-wise inverse STFT, which is given by:

 $$ \mathsf{i}\mathsf{S T F T}\{X^{F}(w,\tau_{i})\}(t)=\sum_{\omega}X^{F}(\omega,\tau)e^{j\omega t}w(t-\tau_{i}) $$ 

The STFT, WM-MLP operation, and inverse transform are depicted by Figure 1. In highly nonstationary data, energy transition between adjacent windows can be sharp. To that end, we introduce a minor overlap between adjacent windows of  $ N_{FFT} - \frac{L - N_{FFT}}{p-1} $ , which implicitly adjusts their statistics prior to processing by the TSF model by increasing the inter-window correlations.

### 3.2 IMPLEMENTATION DETAILS AND COMPLETE SYSTEM

Selective Frequency Compression To reduce the input dimensionality to the WM-MLP, we compress each transformed window  $ C_{i} \in C^{N_{FFT} \times D} $  along the frequency axis. Specifically, we select the top M frequency components based on their real and imaginary values across each dimension and denote the compressed window with  $ C_{i}^{M} $ . Then,  $ (C_{1}^{M}, \ldots, C_{p}^{M}) $  is fed into the WM-MLP layer. The top-M procedure is given by

 $$ C_{i}^{M}=\underset{j=1,\ldots,M}{\mathrm{T o p-M}}\left|C_{i,j}\right|\mathbb{C} $$ 

where  $ C_{i,j} $  is the jth component of  $ C_{i} $  and  $ |z|_{\mathbb{C}} $  is the magnitude of  $ z \in C $ . Additionally, we store the top component indices of equation 4 in a list  $ \mathcal{I}(i) $ , which encodes the band from which the information came. To transform the WM-MLP output  $ C_{i}^{out} $  back to the time domain, we perform a

<div style="text-align: center;"><img src="imgs/img_in_image_box_236_173_980_438.jpg" alt="Image" width="60%" /></div>


<div style="text-align: center;">Figure 3: FIA-Net Model: The input, denoted X, is first fed into the embedding layer, resulting in  $ X_{E} $ , which is transformed to the frequency domain via the STFT. We then extract the top-M components of each STFT window and feed the compressed windows through the WM-MLP. The MLP outputs are then passed through position-aware zero padding, whose outputs are transformed back to the time domain and summed with  $ X_{E} $  via skip connection. The model output  $ \hat{X} $  is then given by applying a linear transformation.</div>


position-aware zero padding, which adds  $ N_{FFT} - M $  zeros while placing the nonzero components in their original indices, which correspond to the original frequency bands, i.e.,

 $$ C_{i,j}^{\mathtt{p a d d e d}}=\left\{\begin{aligned}&C_{i,j}^{\mathtt{o u t}},&j\in\mathcal{I}(i)\\ &0,&\text{else.}\end{aligned}\right. $$ 

In Section 5, we demonstrate that, in addition to improving computational efficiency, this frequency compression procedure enhances the performance of downstream TSF tasks. The selection of top-M components allows us to reduce the model's complexity while maintaining the most relevant frequency information.

Complete Model The complete FIA-Net, as shown in Figure 3, operates as follows: Given an input  $ X \in R^{B \times L \times D} $ , the dimension of X is expanded through a learned embedding layer, resulting in  $ X_{E} \in R^{B \times L \times D \times E} $ . This expanded representation is then fed into an STFT block that uses the real input to perform R - STFT on  $ X_{E} $ . The transformed signal is passed through the SM block, whose output is further processed by the WM-MLP. The WM-MLP outputs are subsequently padded and transformed back to the temporal axis, where they are integrated with  $ X_{E} $  via a skip connection and resized to the desired output sequence shape using a two-layer MLP decomposition.

Model Complexity The forward pass complexity of the WM-MLP is primarily determined by the STFT complexity, which is  $ O(L \log \left(\frac{L}{p}\right)) $ . This represents a significant reduction in complexity compared to transformer-based methods, which employ intricate mechanisms to reduce their  $ O(L^{2}) $  attention complexity to  $ O(L \log L) $ . Additionally, the application of top-M frequency selection further optimizes the forward pass in the frequency domain, reducing both computational demands and the corresponding MLP size. A detailed analysis of these complexities is provided in Table 11.

## 4 WINDOW AGGREGATION VIA HYPER-COMPLEX MODELS

Even though the WM-MLP backbone integrates valuable information that benefits the FIA-Net's accuracy, information is not only shared between two adjacent STFT windows. In fact, the stronger the dependencies on the long-term past, the more information is shared between two distant windows on the frequency axis. Ideally, we would like to aggregate information between all p STFT windows. Unfortunately, a straightforward extension of the WM-MLP requires  $ O(p^{2}) $  weight matrices, which may impair the training procedure and increase model complexity. To address that, we interpret the set of windows as an HC vector and propose an HC-based MLP that efficiently processes the set of STFT windows. We begin with a short introduction on HC-algebras, followed by the construction of the proposed MLP backbone for the FIA-Net.

<div style="text-align: center;"><img src="imgs/img_in_image_box_376_159_845_510.jpg" alt="Image" width="38%" /></div>


<div style="text-align: center;">Figure 4: HC-MLP operating on  $ C^{\mathrm{in}} = (C_{1}^{\mathrm{in}}, C_{2}^{\mathrm{in}}, C_{3}^{\mathrm{in}}, C_{4}^{\mathrm{in}}) $ , implementing the HC multiplication (equation 6). Each output unit is the sum of the corresponding inner blocks of the same color, where a  $ \oplus $  symbol denotes complex addition and a  $ \otimes $  denotes complex multiplication. A red outline denotes minus multiplication, and a blue input arrow denotes complex conjugation.</div>


### 4.1 HYPER-COMPLEX NUMBERS

HC numbers generalize the complex field by introducing additional dimensions while maintaining algebraic properties. HC number systems are defined by a parameter q that determines the number of components in the number system. Complex numbers can thus be viewed as an HC number with q = 2, and an HC number of base q can be represented with p = q/2 complex numbers. In what follows, we focus on HC numbers with p = 4, termed Octonions $\mathbb{O}$, whose elements are denoted $o = (\alpha_{1}, \alpha_{2}, \alpha_{3}, \alpha_{4}) \in \mathbb{O}$, with $\alpha_{i} \in \mathbb{C}$ for $i = 1, \ldots, 4$. Additional discussion on $p \neq 4$ is given in Appendix C.

The addition of two Octonions,  $ o_{1} = (\alpha_{1}, \ldots, \alpha_{4}) $  and  $ o_{2} = (\beta_{1}, \ldots, \beta_{4}) $ , is given by their componentwise sum, while their multiplication follows the Cayley-Dickson construction Khmelnytskaya & Shapiro (2021). The product  $ o_{3} = o_{1} \cdot o_{2} = (\gamma_{1}, \gamma_{2}, \gamma_{3}, \gamma_{4}) $  is given by:

 $$ \begin{aligned}\gamma_{1}&=\alpha_{1}\beta_{1}-\alpha_{2}\overline{\beta_{2}}-\alpha_{3}\overline{\beta_{3}}-\alpha_{4}\overline{\beta_{4}}\\\gamma_{2}&=\alpha_{2}\overline{\beta}_{1}+\alpha_{1}\beta_{2}+\alpha_{3}\overline{\beta_{4}}-\alpha_{4}\overline{\beta_{3}}\\\gamma_{3}&=\alpha_{3}\overline{\beta}_{1}+\alpha_{4}\overline{\beta}_{2}+\alpha_{1}\beta_{3}-\alpha_{2}\overline{\beta}_{4}\\\gamma_{4}&=\alpha_{4}\overline{\beta}_{1}+\alpha_{2}\overline{\beta}_{3}+\alpha_{1}\beta_{4}-\alpha_{3}\overline{\beta}_{2}\end{aligned} $$ 

Hyper-complex numbers exhibit additional properties such as closed-form expressions for norm calculations and norm preservation for specific bases. For completeness, we provide additional information on HC-numbers in Appendix C, where the proposed MLP is presented under specific bases.

### 4.2 HYPER-COMPLEX MLP

The longer the range of temporal dependencies in the data, the more shared information there is between gathered windows. In such cases, the WM-MLP, which incorporates short-term information in the frequency domain, might fail to capture long-term dependencies. To that end, our goal is to increase the extent to which information is shared across the STFT windows. To derive a parameter-efficient solution, we incorporate HC algebra into the frequency domain learning procedure.

Assume that we are given p = 4 complex-valued STFT windows  $ (C_{i}^{\mathrm{in}} \in \mathbb{C}^{B \times M \times E})_{i=1}^{4} $ , where the second axis is the transformed frequency domain after top-M frequency component selection. We treat the set of windows as a single Octonion tensor  $ (C_{1}^{\mathrm{in}}, C_{2}^{\mathrm{in}}, C_{3}^{\mathrm{in}}, C_{4}^{\mathrm{in}}) \in \mathbb{C}^{B \times M \times E} $  and feed it through an HC-valued MLP, whose output is  $ C^{\mathrm{out}} = \sigma(C^{\mathrm{in}} \cdot W + B) $ . For  $ C^{out} = $ 

 $ (C_{1}^{\text{out}}, C_{2}^{\text{out}}, C_{3}^{\text{out}}, C_{4}^{\text{out}}) $ , it is given by:

 $$ \begin{aligned}&C_{1}^{\mathrm{out}}=\sigma(C_{1}^{\mathrm{in}}W_{1}-C_{2}^{\mathrm{in}}\overline{W}_{2}-C_{3}^{\mathrm{in}}\overline{W}_{3}-C_{4}^{\mathrm{in}}\overline{W}_{4}+B_{1}),\\&C_{2}^{\mathrm{out}}=\sigma(C_{2}^{\mathrm{in}}\overline{W}_{1}+C_{1}^{\mathrm{in}}W_{2}-C_{4}^{\mathrm{in}}\overline{W}_{3}+C_{3}^{\mathrm{in}}\overline{W}_{4}+B_{2}),\\&C_{3}^{\mathrm{out}}=\sigma(C_{3}^{\mathrm{in}}W_{1}+C_{1}^{\mathrm{in}}\overline{W}_{3}-C_{2}^{\mathrm{in}}\overline{W}_{4}+C_{4}^{\mathrm{in}}\overline{W}_{2}+B_{3}),\\&C_{4}^{\mathrm{out}}=\sigma(C_{4}^{\mathrm{in}}\overline{W}_{1}+C_{1}^{\mathrm{in}}W_{4}-C_{3}^{\mathrm{in}}\overline{W}_{2}+C_{2}^{\mathrm{in}}\overline{W}_{3}+B_{4}).\\ \end{aligned} $$ 

where  $  W = (W_{1}, \ldots, W_{4}) \in \mathbb{O}^{E \times E}, B = (B_{1}, \ldots, B_{4}) \in \mathbb{O}^{E \times 1}  $  are the HC-MLP weights and bias, respectively, and  $ \sigma $  is a standard activation function, e.g., ReLU. We stress that, as considered in the complex MLP from Yi et al. (2023b), the HC-MLP is implemented with real-valued operations, which allows it to plug into every existing automatic differentiation scheme over standard GPUs. The HC-MLP unit is depicted in Figure 4.

The WM-MLP demonstrates distinct advantages depending on the prediction horizon. For shorter prediction lengths, it achieves better performance by effectively leveraging all available information from adjacent and nearby windows. In contrast, for longer horizons, where only closer temporal information remains relevant, the WM-MLP's ability to aggregate adjusted windows proves to be more effective. This behavior is clearly demonstrated in Section 5.2. Moreover, the HC perspective offers a significant advantage in terms of parameter efficiency. It allows for an implementation with only p weight matrices, whereas the corresponding WM-MLP would require 3p - 2 weight matrices (and even  $ p^{2} $  weight matrices for a generalization of the WM-MLP), all while preserving performance. This reduction in parameters becomes increasingly dramatic as p > 4, as further detailed in Appendix C.

## 5 RESULTS AND DISCUSSION

### 5.1 EXPERIMENTAL SETTING

Datasets Following Zhou et al. (2022a); Yi et al. (2023b), we consider the following representative real-world datasets: 1) WTH (Weather), 2) Exchange (Finance), 3) Traffic, 4) ECL (Electricity), 5) ETTh1 (Electricity transformer temperature hourly), and 6) ETTm1 (Electricity transformer temperature minutely). The train/validation/test split is 70%, 15%, and 15%, respectively.

Baselines In this research, we followed the TSF SoTA baselines: 1) FedFormer Zhou et al. (2022a), 2) Reformer Kitaev et al. (2020), 3) FreTS Yi et al. (2023b), 4) PatchTST Nie et al. (2023), 5) Informer Zhou et al. (2023), 6) Autoformer Wu et al. (2021) and 7) LSTF-Linear Zeng et al. (2023a).

Experiments setup All experiments were conducted using PyTorch Paszke et al. (2019) on a single RTX 3090, utilizing mean squared error (MSE) loss and the Adam optimizer Kingma (2014). We established an initial learning rate of  $ 10^{-3} $  with an exponential decay scheduler. Hyperparameters were optimized individually for each dataset (see Appendix B.3 for specific details). We report performance metrics under both root mean squared error (RMSE) and mean absolute error (MAE). Additional information on the Normalization B.5, datasets B.1, and baseline models B.2 can be found in the appendix.

### 5.2 MAIN RESULTS

Table 1 compares the FIA-Net performance under both the WM-MLP and the HC-MLP backbones with the SoTA baselines. It is evident that the FIA-Net consistently outperforms the baselines on most considered values of prediction horizon T, with an average improvement of 5.4% in MAE and 3.8% in RMSE over SoTA models. We note that the performance of the HC-MLP-based network, which is implemented with significantly fewer parameters, achieves comparable results with the corresponding WM-MLP and attains the best results over several settings. We can deduce that the HC-MLP is more suitable for shorter-term prediction, while the WM-MLP backbone is more suitable for longer ranges.

The WM-MLP backbone results reported in Table 1 consider an optimization with respect to p, the number of windows, while the HC-MLP considers a fixed size of p = 4 windows. Thus, for a more

<div style="text-align: center;">Table 1: Forecasting performance comparison across datasets and prediction horizons using RMSE and MAE. Lower values indicate better performance. Bold denotes the best results, and underlined indicates the second-best.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td style='text-align: center;'>|</td><td colspan="4">Weather</td><td colspan="4">Exchange</td><td colspan="4">Traffic</td><td colspan="4">Electricity</td><td colspan="4">ETThI</td><td colspan="4">ETTmI</td></tr><tr><td style='text-align: center;'>Metric</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td rowspan="2">HC-MLP (Ours)</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.069</td><td style='text-align: center;'>0.079</td><td style='text-align: center;'>0.090</td><td style='text-align: center;'>0.098</td><td style='text-align: center;'>0.050</td><td style='text-align: center;'>0.062</td><td style='text-align: center;'>0.078</td><td style='text-align: center;'>0.112</td><td style='text-align: center;'>0.032</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.036</td><td style='text-align: center;'>0.070</td><td style='text-align: center;'>0.068</td><td style='text-align: center;'>0.071</td><td style='text-align: center;'>0.077</td><td style='text-align: center;'>0.083</td><td style='text-align: center;'>0.085</td><td style='text-align: center;'>0.094</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>0.074</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.026</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.830</td><td style='text-align: center;'>0.839</td><td style='text-align: center;'>0.043</td><td style='text-align: center;'>0.054</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.061</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>0.017</td><td style='text-align: center;'>0.017</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>0.040</td><td style='text-align: center;'>0.041</td><td style='text-align: center;'>0.044</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.068</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.056</td><td style='text-align: center;'>0.069</td><td style='text-align: center;'>0.062</td></tr><tr><td rowspan="2">WM-MLP (Ours)</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.071</td><td style='text-align: center;'>0.081</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>0.048</td><td style='text-align: center;'>0.060</td><td style='text-align: center;'>0.076</td><td style='text-align: center;'>0.107</td><td style='text-align: center;'>0.033</td><td style='text-align: center;'>0.033</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>0.036</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>0.068</td><td style='text-align: center;'>0.070</td><td style='text-align: center;'>0.076</td><td style='text-align: center;'>0.084</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>0.076</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.094</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.031</td><td style='text-align: center;'>0.041</td><td style='text-align: center;'>0.045</td><td style='text-align: center;'>0.053</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>0.047</td><td style='text-align: center;'>0.058</td><td style='text-align: center;'>0.086</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>0.041</td><td style='text-align: center;'>0.044</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>0.066</td><td style='text-align: center;'>0.071</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.058</td><td style='text-align: center;'>0.064</td></tr><tr><td rowspan="2">FreTS</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.071</td><td style='text-align: center;'>0.081</td><td style='text-align: center;'>0.090</td><td style='text-align: center;'>0.099</td><td style='text-align: center;'>0.051</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.110</td><td style='text-align: center;'>0.036</td><td style='text-align: center;'>0.038</td><td style='text-align: center;'>0.038</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.072</td><td style='text-align: center;'>0.079</td><td style='text-align: center;'>0.087</td><td style='text-align: center;'>0.091</td><td style='text-align: center;'>0.096</td><td style='text-align: center;'>0.108</td><td style='text-align: center;'>0.077</td><td style='text-align: center;'>0.083</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.096</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.032</td><td style='text-align: center;'>0.040</td><td style='text-align: center;'>0.046</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.037</td><td style='text-align: center;'>0.050</td><td style='text-align: center;'>0.062</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>0.020</td><td style='text-align: center;'>0.019</td><td style='text-align: center;'>0.020</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>0.040</td><td style='text-align: center;'>0.046</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>0.061</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.07</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>0.062</td><td style='text-align: center;'>0.069</td></tr><tr><td rowspan="2">PatchTST</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.074</td><td style='text-align: center;'>0.084</td><td style='text-align: center;'>0.094</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>0.074</td><td style='text-align: center;'>0.093</td><td style='text-align: center;'>0.166</td><td style='text-align: center;'>0.032</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>0.040</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>0.066</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>0.081</td><td style='text-align: center;'>0.091</td><td style='text-align: center;'>0.094</td><td style='text-align: center;'>0.099</td><td style='text-align: center;'>0.113</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.085</td><td style='text-align: center;'>0.091</td><td style='text-align: center;'>0.097</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>0.042</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.056</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.071</td><td style='text-align: center;'>0.132</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>0.020</td><td style='text-align: center;'>0.021</td><td style='text-align: center;'>0.031</td><td style='text-align: center;'>0.032</td><td style='text-align: center;'>0.043</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.069</td><td style='text-align: center;'>0.073</td><td style='text-align: center;'>0.087</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.059</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.070</td></tr><tr><td rowspan="2">LTSF-Linear</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.081</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.098</td><td style='text-align: center;'>0.106</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>0.069</td><td style='text-align: center;'>0.085</td><td style='text-align: center;'>0.116</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>0.042</td><td style='text-align: center;'>0.040</td><td style='text-align: center;'>0.041</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.070</td><td style='text-align: center;'>0.071</td><td style='text-align: center;'>0.080</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.094</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>0.108</td><td style='text-align: center;'>0.080</td><td style='text-align: center;'>0.087</td><td style='text-align: center;'>0.093</td><td style='text-align: center;'>0.099</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.040</td><td style='text-align: center;'>0.048</td><td style='text-align: center;'>0.056</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.038</td><td style='text-align: center;'>0.053</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.092</td><td style='text-align: center;'>0.020</td><td style='text-align: center;'>0.022</td><td style='text-align: center;'>0.020</td><td style='text-align: center;'>0.021</td><td style='text-align: center;'>0.045</td><td style='text-align: center;'>0.043</td><td style='text-align: center;'>0.044</td><td style='text-align: center;'>0.054</td><td style='text-align: center;'>0.063</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>0.070</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.060</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.072</td></tr><tr><td rowspan="2">FEDformer</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>0.092</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>0.109</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.105</td><td style='text-align: center;'>0.183</td><td style='text-align: center;'>0.036</td><td style='text-align: center;'>0.042</td><td style='text-align: center;'>0.042</td><td style='text-align: center;'>0.042</td><td style='text-align: center;'>0.072</td><td style='text-align: center;'>0.072</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.077</td><td style='text-align: center;'>0.096</td><td style='text-align: center;'>0.100</td><td style='text-align: center;'>0.105</td><td style='text-align: center;'>0.116</td><td style='text-align: center;'>0.087</td><td style='text-align: center;'>0.093</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>0.108</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.050</td><td style='text-align: center;'>0.051</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.050</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.080</td><td style='text-align: center;'>0.151</td><td style='text-align: center;'>0.022</td><td style='text-align: center;'>0.023</td><td style='text-align: center;'>0.022</td><td style='text-align: center;'>0.022</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.051</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.072</td><td style='text-align: center;'>0.076</td><td style='text-align: center;'>0.080</td><td style='text-align: center;'>0.090</td><td style='text-align: center;'>0.063</td><td style='text-align: center;'>0.068</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.081</td></tr><tr><td rowspan="2">Autoformer</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.104</td><td style='text-align: center;'>0.103</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>0.110</td><td style='text-align: center;'>0.066</td><td style='text-align: center;'>0.083</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>0.181</td><td style='text-align: center;'>0.042</td><td style='text-align: center;'>0.050</td><td style='text-align: center;'>0.053</td><td style='text-align: center;'>0.050</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.099</td><td style='text-align: center;'>0.115</td><td style='text-align: center;'>0.119</td><td style='text-align: center;'>0.105</td><td style='text-align: center;'>0.114</td><td style='text-align: center;'>0.119</td><td style='text-align: center;'>0.136</td><td style='text-align: center;'>0.109</td><td style='text-align: center;'>0.112</td><td style='text-align: center;'>0.125</td><td style='text-align: center;'>0.126</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.061</td><td style='text-align: center;'>0.059</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.050</td><td style='text-align: center;'>0.063</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.150</td><td style='text-align: center;'>0.026</td><td style='text-align: center;'>0.033</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.051</td><td style='text-align: center;'>0.051</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>0.116</td><td style='text-align: center;'>0.079</td><td style='text-align: center;'>0.086</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>0.081</td><td style='text-align: center;'>0.083</td><td style='text-align: center;'>0.091</td><td style='text-align: center;'>0.093</td></tr><tr><td rowspan="2">Informer</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.139</td><td style='text-align: center;'>0.134</td><td style='text-align: center;'>0.115</td><td style='text-align: center;'>0.132</td><td style='text-align: center;'>0.084</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>0.127</td><td style='text-align: center;'>0.170</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>0.047</td><td style='text-align: center;'>0.053</td><td style='text-align: center;'>0.054</td><td style='text-align: center;'>0.124</td><td style='text-align: center;'>0.138</td><td style='text-align: center;'>0.144</td><td style='text-align: center;'>0.148</td><td style='text-align: center;'>0.121</td><td style='text-align: center;'>0.137</td><td style='text-align: center;'>0.145</td><td style='text-align: center;'>0.157</td><td style='text-align: center;'>0.096</td><td style='text-align: center;'>0.107</td><td style='text-align: center;'>0.119</td><td style='text-align: center;'>0.149</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>0.132</td><td style='text-align: center;'>0.066</td><td style='text-align: center;'>0.068</td><td style='text-align: center;'>0.093</td><td style='text-align: center;'>0.117</td><td style='text-align: center;'>0.023</td><td style='text-align: center;'>0.030</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.094</td><td style='text-align: center;'>0.105</td><td style='text-align: center;'>0.112</td><td style='text-align: center;'>0.116</td><td style='text-align: center;'>0.093</td><td style='text-align: center;'>0.103</td><td style='text-align: center;'>0.112</td><td style='text-align: center;'>0.125</td><td style='text-align: center;'>0.070</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.090</td><td style='text-align: center;'>0.115</td></tr><tr><td rowspan="2">Reformer</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.152</td><td style='text-align: center;'>0.201</td><td style='text-align: center;'>0.203</td><td style='text-align: center;'>0.228</td><td style='text-align: center;'>0.146</td><td style='text-align: center;'>0.169</td><td style='text-align: center;'>0.189</td><td style='text-align: center;'>0.201</td><td style='text-align: center;'>0.053</td><td style='text-align: center;'>0.054</td><td style='text-align: center;'>0.053</td><td style='text-align: center;'>0.054</td><td style='text-align: center;'>0.125</td><td style='text-align: center;'>0.138</td><td style='text-align: center;'>0.144</td><td style='text-align: center;'>0.148</td><td style='text-align: center;'>0.143</td><td style='text-align: center;'>0.148</td><td style='text-align: center;'>0.155</td><td style='text-align: center;'>0.155</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.108</td><td style='text-align: center;'>0.128</td><td style='text-align: center;'>0.163</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.108</td><td style='text-align: center;'>0.147</td><td style='text-align: center;'>0.154</td><td style='text-align: center;'>0.173</td><td style='text-align: center;'>0.126</td><td style='text-align: center;'>0.147</td><td style='text-align: center;'>0.157</td><td style='text-align: center;'>0.166</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.095</td><td style='text-align: center;'>0.121</td><td style='text-align: center;'>0.122</td><td style='text-align: center;'>0.120</td><td style='text-align: center;'>0.113</td><td style='text-align: center;'>0.120</td><td style='text-align: center;'>0.124</td><td style='text-align: center;'>0.126</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.081</td><td style='text-align: center;'>0.100</td><td style='text-align: center;'>0.132</td></tr></table>

<div style="text-align: center;">suitable comparison, Table 2 shows a comparison of the FIA-Net performance under both backbones with p = 4. We note that when p is similar for both models, the FIA-Net attains similar results under both backbones, while the HC-MLP requires significantly fewer parameters. Consequently, when the number of windows allows for an HC-MLP version (e.g.,  $ p = 2^{\ell} $  as we further explain in Appendix C), an HC-MLP backbone is preferable.</div>


<div style="text-align: center;">Table 2: Performance comparison between WM-MLP and HC-MLP with a fixed number of STFT windows  $ (p = 4) $ . Results demonstrate that HC-MLP achieves comparable accuracy while significantly reducing model parameters, making it preferable for efficient implementations.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td rowspan="2">Metric</td><td colspan="4">Traffic</td><td colspan="4">ETTh1</td><td colspan="4">ETTm1</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td></tr><tr><td rowspan="2">WM-MLP ( $ p = 4 $ )</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.033</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.036</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>0.094</td><td style='text-align: center;'>0.100</td><td style='text-align: center;'>0.103</td><td style='text-align: center;'>0.074</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.096</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>0.017</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>0.058</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.068</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.056</td><td style='text-align: center;'>0.060</td><td style='text-align: center;'>0.067</td></tr><tr><td rowspan="2">HC-MLP</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.032</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>0.036</td><td style='text-align: center;'>0.083</td><td style='text-align: center;'>0.085</td><td style='text-align: center;'>0.094</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>0.072</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.096</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>0.017</td><td style='text-align: center;'>0.017</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.068</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>0.056</td><td style='text-align: center;'>0.060</td><td style='text-align: center;'>0.067</td></tr></table>

### 5.3 ABLATION STUDIES

We consider three ablation studies that best demonstrate the key aspects of the proposed work. We focus on the effect of frequency selection, the size of the lookback window, and the omission of real/imaginary components in the training procedure. We show that, in various cases, the total amount of parameters can be decreased by up to 60%. Due to space limitations, the results are demonstrated on a single dataset, while a full discussion and additional results are given in Appendix D.4.

#### 5.3.1 FREQUENCY DIMENSION COMPRESSION

We study the effect of the parameter M in the top-M frequency component selection process on the ETTh dataset. As seen in figure 5, even though the model performance varies over different datasets and forecasting horizon sizes, in most cases, M = 4 attains the best accuracy. Furthermore, note that taking  $ M < M_{max} = \frac{N_{FFT}}{2} + 1 $  improves the model's results. We conjecture that considering fewer frequency components decreases the NN class complexity, which potentially simplifies the optimization procedure landscape while preserving most of the information contained within the signal. We expand upon this discussion and provide additional results in the Appendix D.1.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_771_1220_1002_1396.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">Figure 5: Accuracy vs. M</div>


#### 5.3.2 EFFECT OF LOOKBACK WINDOW SIZE

In this section, we evaluate the impact of varying lookback window sizes  $ L \in \{24, 48, 96, 192, 288, 480, 576, 720\} $  for different prediction lengths  $ T \in \{96, 192, 336, 720\} $ . As shown in Figure 6, the dotted line represents the RMSE, while the solid line represents the MAE. The model's performance initially improves as L increases, as expected, since a longer lookback provides more contextual information. However, many models exhibit parabolic behavior, where performance deteriorates after a certain point due to overfitting to noise or unrealistic patterns in the data. In contrast, our model maintains stable performance and effectively avoids overfitting, demonstrating its robustness to changes in lookback window size. Additional experiments can be found in Appendix D.2.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_770_205_1003_381.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">Figure 6: Accuracy vs. L.</div>


#### 5.3.3 REDUNDANCY OF COMPLEX REPRESENTATION

We study the effect of the real and imaginary components on prediction quality. We fix the hyperparameters E = 128, p = 13,  $ N_{FFT} = 16 $ ,  $ M = M_{max} $ , and compare several scenarios, such that each scenario considers the masking of a different component, either in the data, the parameters, or both. The masking occurs in both training and inference. As seen in Table 3, the elimination of either the real or imaginary components in the data does not significantly affect the downstream accuracy, which may hint at redundancy in the learning procedure. Furthermore, this redundancy is maintained when we consider the intersection omission of the real/imaginary parts of both the data and the MLP weights. This phenomenon can be explained through the Kramers-Kronig relation (KKR) Kronig (1926); Kramers (1927), which provides a representation of the real component of an analytic complex-valued function in terms of its complex components and vice versa. Roughly speaking, for a complex-valued function  $  c(\omega) = \mathsf{Re}\{c\}(\omega) + \mathsf{i} \mathsf{Im}\{c\}(\omega)  $ , the KKR are given by

 $$ \mathsf{Re}\{c\}(\omega)=\frac{1}{\pi}\int_{-\infty}^{\infty}\frac{\mathsf{Im}\{c\}(\sigma)}{\omega-\sigma}d\sigma,\quad\mathsf{Im}\{c\}(\omega)=-\frac{1}{\pi}\int_{-\infty}^{\infty}\frac{\mathsf{Re}\{c\}(\sigma)}{\omega-\sigma}d\sigma. $$ 

Thus, we conjecture that masking one component forces the other to recover both in the learning procedure by implicitly approximating the KRR. We therefore believe that a sophisticated system design that considers a KRR-based architecture may lead to the sufficiency of a single component in the forecasting task but leaves a complete study of that subject to future work. This phenomenon is further explored in Appendix 8.


<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Dataset</td><td style='text-align: center;'>I/O</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/336</td><td colspan="2">96/720</td></tr><tr><td style='text-align: center;'>Hidden Part</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>RMSE</td></tr><tr><td rowspan="7">ETTm1</td><td style='text-align: center;'>$ X^{\text{Real}} $</td><td style='text-align: center;'>0.0522</td><td style='text-align: center;'>0.0797</td><td style='text-align: center;'>0.0560</td><td style='text-align: center;'>0.0850</td><td style='text-align: center;'>0.0597</td><td style='text-align: center;'>0.0888</td><td style='text-align: center;'>0.0658</td><td style='text-align: center;'>0.0958</td></tr><tr><td style='text-align: center;'>$ X^{\text{Imag}} $</td><td style='text-align: center;'>0.0521</td><td style='text-align: center;'>0.0792</td><td style='text-align: center;'>0.0562</td><td style='text-align: center;'>0.0844</td><td style='text-align: center;'>0.0592</td><td style='text-align: center;'>0.0879</td><td style='text-align: center;'>0.0684</td><td style='text-align: center;'>0.0976</td></tr><tr><td style='text-align: center;'>$ W^{\text{Real}} $</td><td style='text-align: center;'>0.0522</td><td style='text-align: center;'>0.0791</td><td style='text-align: center;'>0.0557</td><td style='text-align: center;'>0.0843</td><td style='text-align: center;'>0.0588</td><td style='text-align: center;'>0.0875</td><td style='text-align: center;'>0.0669</td><td style='text-align: center;'>0.0964</td></tr><tr><td style='text-align: center;'>$ W^{\text{Imag}} $</td><td style='text-align: center;'>0.0526</td><td style='text-align: center;'>0.0801</td><td style='text-align: center;'>0.0560</td><td style='text-align: center;'>0.0849</td><td style='text-align: center;'>0.0596</td><td style='text-align: center;'>0.0888</td><td style='text-align: center;'>0.0651</td><td style='text-align: center;'>0.0953</td></tr><tr><td style='text-align: center;'>$ W^{\text{Imag}}, X^{\text{Imag}} $</td><td style='text-align: center;'>0.0523</td><td style='text-align: center;'>0.0798</td><td style='text-align: center;'>0.0560</td><td style='text-align: center;'>0.0849</td><td style='text-align: center;'>0.0592</td><td style='text-align: center;'>0.0884</td><td style='text-align: center;'>0.0644</td><td style='text-align: center;'>0.0947</td></tr><tr><td style='text-align: center;'>$ W^{\text{Real}}, X^{\text{Real}} $</td><td style='text-align: center;'>0.0522</td><td style='text-align: center;'>0.0791</td><td style='text-align: center;'>0.0557</td><td style='text-align: center;'>0.0843</td><td style='text-align: center;'>0.0588</td><td style='text-align: center;'>0.0887</td><td style='text-align: center;'>0.0669</td><td style='text-align: center;'>0.0930</td></tr><tr><td style='text-align: center;'>$ \emptyset $</td><td style='text-align: center;'>0.0522</td><td style='text-align: center;'>0.0791</td><td style='text-align: center;'>0.0565</td><td style='text-align: center;'>0.0848</td><td style='text-align: center;'>0.0592</td><td style='text-align: center;'>0.0878</td><td style='text-align: center;'>0.0685</td><td style='text-align: center;'>0.0975</td></tr></table>

<div style="text-align: center;">Table 3: Performance comparison on ETTm1 for  $ I/O = 96 \times \{96, 192, 336, 720\} $  with various modes.  $ X^{Real}/X^{Imag} $  hide the real/imaginary parts of the input, while  $ W^{Real}/W^{Imag} $  zero out the corresponding weights. Completely ignoring both components is denoted as  $ (W^{\mathrm{Imag}}, X^{\mathrm{Imag}}) $  or  $ (W^{\mathrm{Real}}, X^{\mathrm{Real}}) $ .</div>


## 6 CONCLUSION

This paper presents FIA-Net, a new model for long-term time series forecasting using STFT window aggregation in the frequency domain and HC MLPs. The proposed methodology shows superior performance over existing SoTA on standard benchmark datasets. We show that treating the set of

STFT windows as a single HC tensor, which is processed by a novel HC-MLP, significantly reduces the total amount of parameters, with no degradation in the TSF accuracy. We study various schemes to increase model efficiency by, for example, choosing the top-M magnitude frequency components. Experimental results show that the omission of one of the complex representation components does not induce notable segregation in performance, which may be explained by the KKR. For future work, we aim to leverage the KKR equations to propose a forecasting model that only considers the real component in the complex representation while operating over the complex plane. Additionally, we plan to further investigate the relationship between the number of adjacent STFT windows in the WM-MLP backbone and the statistical properties of the datasets.

## REFERENCES

L. Bai, L. Yao, C. Li, X. Wang, and C. Wang. Adaptive graph convolutional recurrent network for traffic forecasting. In Advances in Neural Information Processing Systems (NeurIPS), 2020.

G. E. P. Box and G. M. Jenkins. Some recent advances in forecasting and control. Journal of the Royal Statistical Society: Series C (Applied Statistics), 17(2):91–109, 1968.

G. E. P. Box and D. A. Pierce. Distribution of residual autocorrelations in autoregressive-integrated moving average time series models. Journal of the American Statistical Association, 65:1509–1526, 1970.

L. Cao, K. Yi, L. Hu, Q. Zhang, N. Cao, and Z. Niu. Stemgnn: Graph neural networks for multivariate time series forecasting. In Proceedings of the 37th International Conference on Machine Learning (ICML), 2020.

X. Chen and R. Chen. A review on traffic prediction methods for intelligent transportation system in smart cities. In 2019 12th International Congress on Image and Signal Processing, BioMedical Engineering and Informatics (CISP-BMEI), pp. 1–5. IEEE, 2019.

J. Chung, C. Gulcehre, K. Cho, and Y. Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

P. Fryzlewicz and H. Cho. Multiple-change-point detection for auto-regressive conditional heteroscedastic processes. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 76(5):903–924, 2014. doi: 10.1111/rssb.12058. URL https://academic.oup.com/jrss/article/76/5/903/1745094.

D. Gabor. Theory of communication. part 1: The analysis of information. Journal of the Institution of Electrical Engineers-Part III: Radio and Communication Engineering, 93(26):429–441, 1946.

A. Graves. Supervised sequence labelling with recurrent neural networks. Studies in Computational Intelligence, 385:37–45, 2012.

W. R. Hamilton. On quaternions; or on a new system of imaginaries in algebra. Proceedings of the Royal Irish Academy, 1844.

I. L. Kantor and A. S. Solodovnikov. Hypercomplex numbers: An elementary introduction to algebras. 1989. URL https://archive.org/details/hypercomplexnumb0000kant.

K. V. Khmelnytskaya and M. Shapiro. Function theories in cayley-dickson algebras and number theory. Complex Analysis and Operator Theory, 15(2):1–40, 2021. doi: 10.1007/s11785-020-01082-6. URL https://www.researchgate.net/publication/349897693_Function_Theories_in_Cayley-Dickson_Algebra_and_Number_Theory.

D. Kim, J. Park, J. Lee, and H. Kim. Are self-attentions effective for time series forecasting? arXiv preprint arXiv:2405.16877, 2024.

D. P. Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

N. Kitaev, Ł. Kaiser, and A. Levskaya. Reformer: The efficient transformer. In International Conference on Learning Representations (ICLR), 2020. arXiv preprint arXiv:2001.04451.

H. A. Kramers. La diffusion de la lumière par les atomes. In Atti del Congresso Internazionale dei Fisici, Como, volume 2, pp. 545–557, 1927.

R. de L. Kronig. On the theory of dispersion of x-rays. Journal of the Optical Society of America, 12(6):547–557, 1926. doi: 10.1364/JOSA.12.000547.

R. Kycia and A. Niemczynowicz. Hypercomplex neural network in time series forecasting of stock data. arXiv preprint arXiv:2401.04632, 2024.

L. Luo, H. Feng, and L. Ding. Color image compression based on quaternion neural network principal component analysis. In 2010 International Conference on Multimedia Technology, pp. 1–4. IEEE, 2010.

Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam. A time series is worth 64 words: Long-term forecasting with transformers. In International Conference on Learning Representations (ICLR), 2023.

B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio. N-beats: Neural basis expansion analysis for interpretable time series forecasting. arXiv preprint arXiv:1905.10437, 2019.

T. Parcollet, M. Morchid, P.-M. Bousquet, R. Dufour, and G. Linarès. Quaternion convolutional neural networks for image classification and compression. In Proceedings of the IEEE Spoken Language Technology Workshop (SLT), pp. 362–368. IEEE, 2016.

R. Pascanu, T. Mikolov, and Y. Bengio. On the difficulty of training recurrent neural networks. arXiv preprint arXiv:1211.5063v2, 2013. Available at: https://arxiv.org/abs/1211.5063.

A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems (NeurIPS), 32, 2019.

R. A. Rajagukguk, R. A. Ramadhan, and H.-J. Lee. A review on deep learning models for forecasting time series data of solar irradiance and photovoltaic power. Energies, 13(24):6623, 2020.

L. S. Saoud and H. Al-Marzouqi. Metacognitive sedenion-valued neural network and its learning algorithm. IEEE Access, 8:144823–144836, 2020. doi: 10.1109/ACCESS.2020.3014690.

O. B. Sezer, M. U. Gudelek, and A. M. Ozbayoglu. Financial time series forecasting with deep learning: A systematic literature review: 2005–2019. Applied Soft Computing, 90:106181, 2020.

R. Shen, L. Liu, B. Wang, Y. Guan, Y. Yang, and J. Jiang. Freqtsf: Time series forecasting via simulating frequency kramer-kronig relations. arXiv preprint arXiv:2407.21275, 2024.

Fei-Fan Tu, Dong-Jie Liu, Zhi-Wei Yan, Xiao-Bo Jin, and Guang-Gang Geng. Stft-tcan: A tcn-attention based multivariate time series anomaly detection architecture with time-frequency analysis for cyber-industrial systems. Computers & Security, 144:103961, 2024. doi: 10.1016/j.cose.2024.103961.

A. Vaswani, S. Bengio, E. Brevdo, F. Chollet, A. N. Gomez, S. Gowes, L. Jones, Ł. Kaiser, N. Kalchbrenner, N. Parmar, R. Sepassi, N. Shazeer, and J. Uszkoreit. Tensor2 tensor for neural machine translation. CoRR, abs/1803.07416, 2018. URL http://arxiv.org/abs/1803.07416.

M. W. Watson. Vector autoregressions and cointegration. Technical report, 1993.

S. Woo, S. Lee, J. Kim, S. Kim, H. Kim, and W. Jang. Etsformer: Exponential smoothing transformer for time-series forecasting. In Proceedings of the 38th International Conference on Machine Learning (ICML), 2022.

H. Wu, J. Xu, J. Wang, and M. Long. Autoformer: Decomposition transformers with AutoCorrelation for long-term series forecasting. In Advances in Neural Information Processing Systems (NeurIPS), 2021.

Z. Wu, S. Pan, G. Long, J. Jiang, X. Chang, and C. Zhang. Connecting the dots: Multivariate time series forecasting with graph neural networks. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD), pp. 753–763, 2020.

K. Yi, Q. Zhang, L. Hu, N. Cao, and Z. Niu. Cost: A contrastive framework for self-supervised time series representation learning. In Advances in Neural Information Processing Systems (NeurIPS), 2022.

K. Yi, Q. Zhang, L. Cao, S. Wang, G. Long, L. Hu, H. He, Z. Niu, W. Fan, and H. Xiong. A survey on deep learning based time series analysis with frequency transformation. Journal of the ACM, 37(4):111, 2023a. doi: 10.1145/374000.374001.

K. Yi, Q. Zhang, W. Fan, S. Wang, P. Wang, H. He, N. An, D. Lian, L. Cao, and Z. Niu. Frequency-domain MLPs are more effective learners in time series forecasting. In Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS), 2023b.

Ailing Zeng, Muxi Chen, Lei Zhang, and Qiang Xu. Are transformers effective for time series forecasting? 2023a.

Zhen Zeng, Rachneet Kaur, Suchetha Siddagangappa, Tucker Balch, and Manuela Veloso. From pixels to predictions: Spectrogram and vision transformer for better time series forecasting. In Proceedings of the 4th ACM International Conference on AI in Finance (ICAIF '23), Brooklyn, NY, USA, 2023b. ACM. doi: 10.1145/3604237.3626905.

J. Zhang and K.-F. Man. Time series prediction using rnn in multi-dimension embedding phase space. In Proceedings of the IEEE International Conference on Systems, Man, and Cybernetics (SMC), pp. 1868–1873. IEEE, 1998.

L. Zhang, C. C. Aggarwal, and G.-J. Qi. Stock price prediction via discovering multi-frequency trading patterns. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), pp. 2141–2149, 2017.

Y. Zhang and J. Yan. Crossformer: Transformer utilizing cross-dimension dependency for multivariate time series forecasting. In The Eleventh International Conference on Learning Representations (ICLR), 2023.

H. Zhou, J. Li, S. Zhang, S. Zhang, M. Yan, and H. Xiong. Expanding the prediction capacity in long sequence time-series forecasting. Artificial Intelligence, 318:103886, 2023.

T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, and R. Jin. FEDformer: Frequency enhanced decomposed transformer for long-term series forecasting. In Proc. 39th International Conference on Machine Learning (ICML 2022), 2022a.

T. Zhou, Z. Ma, Q. Wen, X. Wang, L. Sun, T. Yao, W. Yin, and R. Jin. Film: Frequency improved legendre memory model for long-term time series forecasting. arXiv preprint arXiv:2205.08897, 2022b.

## APPENDIX FOR "Efficient Time Series Forecasting via Hyper-Complex Models and Frequency Aggregation"

## A Notations & Symbols

### A.1 NOTATION

We provide a detailed table of the involved notation in this paper:


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Symbol</td><td style='text-align: center;'>Description</td></tr><tr><td style='text-align: center;'>B</td><td style='text-align: center;'>Batch size.</td></tr><tr><td style='text-align: center;'>L</td><td style='text-align: center;'>Lookback window size.</td></tr><tr><td style='text-align: center;'>D</td><td style='text-align: center;'>Number of features for each time step.</td></tr><tr><td style='text-align: center;'>T</td><td style='text-align: center;'>Length of the prediction horizon.</td></tr><tr><td style='text-align: center;'>E</td><td style='text-align: center;'>Embedding size.</td></tr><tr><td style='text-align: center;'>M</td><td style='text-align: center;'>Number of frequencies to select from all the frequencies using the top M magnitudes.</td></tr><tr><td style='text-align: center;'>X</td><td style='text-align: center;'>Multivariate time series with a lookback window of size L at timestamps  $ t $ .</td></tr><tr><td style='text-align: center;'>X_{t}</td><td style='text-align: center;'>Multivariate values of  $ D $  distinct series at timestamp  $ t $ .</td></tr><tr><td style='text-align: center;'>X_{t,i}</td><td style='text-align: center;'>The value of the  $ i $ -th feature of the distinct series at timestamp  $ t $ .</td></tr><tr><td style='text-align: center;'>$ \bar{X} $</td><td style='text-align: center;'>Ground truth target values.</td></tr><tr><td style='text-align: center;'>$ \sigma $</td><td style='text-align: center;'>Activation function</td></tr><tr><td style='text-align: center;'>P</td><td style='text-align: center;'>Number of windows in the STFT.</td></tr><tr><td style='text-align: center;'>N_{FFT}</td><td style='text-align: center;'>Number of frequency bins in each window of the STFT.</td></tr><tr><td style='text-align: center;'>$ \omega $</td><td style='text-align: center;'>Window function for the STFT.</td></tr><tr><td style='text-align: center;'>X_{E}</td><td style='text-align: center;'>X after traversing through the embedding layer.</td></tr><tr><td style='text-align: center;'>X_{Rec}</td><td style='text-align: center;'>The reconstructed X after the frequency alteration.</td></tr><tr><td style='text-align: center;'>c_{i}^{t}</td><td style='text-align: center;'>The  $ i $ -th window of the input in the time domain.</td></tr><tr><td style='text-align: center;'>C_{i}</td><td style='text-align: center;'>The  $ i $ -th window of the STFT containing  $ N_{FFT} $  frequency bins.</td></tr><tr><td style='text-align: center;'>C_{i}^{in}</td><td style='text-align: center;'>The  $ i $ -th window of the STFT, retaining the top  $ M $  frequency components based on magnitude.</td></tr><tr><td style='text-align: center;'>C_{i}^{out}</td><td style='text-align: center;'>The  $ i $ -th window of the STFT after the WM-MLP/WHC has been applied.</td></tr><tr><td style='text-align: center;'>W_{i\rightarrow j}</td><td style='text-align: center;'>The weights that capture the frequency energy shift between window  $ i $  and  $ j $ , defined as  $ W_{i\rightarrow j} = W_{i\rightarrow j}^{Real} + j W_{i\rightarrow j}^{Img} $ , where  $ W_{i\rightarrow j} \in \mathbb{C}^{E \times E} $ .</td></tr><tr><td style='text-align: center;'>B_{i\rightarrow j}</td><td style='text-align: center;'>The basis that captures the frequency energy shift between windows  $ i $  and  $ j $ , defined as  $ B_{i\rightarrow j} = B_{i\rightarrow j}^{Real} + j B_{i\rightarrow j}^{Img} $ , where  $ B_{i\rightarrow j} \in \mathbb{C}^{E} $ .</td></tr></table>

Table 4: Table of Symbols and Descriptions

### A.2 DIMENSIONS

The following table summarizes the dimensions of the data tensor in every step of the FIA-Net.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Symbol</td><td style='text-align: center;'>Dimension</td></tr><tr><td style='text-align: center;'>X</td><td style='text-align: center;'>$ R^{B \times L \times D} $</td></tr><tr><td style='text-align: center;'>$ X_{E} $</td><td style='text-align: center;'>$ R^{B \times L \times D \times E} $</td></tr><tr><td style='text-align: center;'>$ C_{i} $</td><td style='text-align: center;'>$ \mathbb{C}^{B \times N_{FFT} \times D \times E} $</td></tr><tr><td style='text-align: center;'>$ C_{i}^{M} $</td><td style='text-align: center;'>$ \mathbb{C}^{B \times M \times D \times E} $</td></tr><tr><td style='text-align: center;'>$ C_{i}^{in/out} $</td><td style='text-align: center;'>$ \mathbb{C}^{B \times M \times D \times E} $</td></tr><tr><td style='text-align: center;'>$ X_{Rec} $</td><td style='text-align: center;'>$ R^{B \times L \times D \times E} $</td></tr><tr><td style='text-align: center;'>$ \hat{X} $</td><td style='text-align: center;'>$ R^{B \times T \times D} $</td></tr></table>

<div style="text-align: center;">Table 5: Table of Symbols and Dimension</div>


## B ADDITIONAL EXPERIMENTAL DETAILS

### B.1 DATASET DESCRIPTIONS

In our experiments, we utilized thirteen real-world datasets to assess the effectiveness of models for long-term TSF. Below, we provide the details of these datasets, categorized by their forecasting horizon.

• Exchange: This dataset includes daily exchange rates for eight countries (Australia, Britain, Canada, Switzerland, China, Japan, New Zealand, and Singapore) from 1990 to 2016.

• Weather: This dataset gathers 21 meteorological indicators, including humidity and air temperature, from the Weather Station of the Max Planck Biogeochemistry Institute in Germany in 2020. The data is collected every 10 minutes.

• Traffic: For long-term forecasting, this dataset includes hourly traffic data from 862 freeway lanes in San Francisco, with data collected since January 1, 2015.

• Electricity: For long-term forecasting, this dataset covers electricity consumption data from 321 clients, with records starting from January 1, 2011, and a sampling interval of 15 minutes.

• ETT: This dataset is sourced from two electric transformers, labeled ETTh1 and ETTm1, with two different resolutions: 15 minutes and 1 hour. These are used as benchmarks for long-term forecasting.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Datasets</td><td style='text-align: center;'>Weather</td><td style='text-align: center;'>Traffic</td><td style='text-align: center;'>Electricity</td><td style='text-align: center;'>ETTh1</td><td style='text-align: center;'>ETTm1</td><td style='text-align: center;'>Exchange Rates</td></tr><tr><td style='text-align: center;'>Features</td><td style='text-align: center;'>21</td><td style='text-align: center;'>862</td><td style='text-align: center;'>321</td><td style='text-align: center;'>7</td><td style='text-align: center;'>7</td><td style='text-align: center;'>8</td></tr><tr><td style='text-align: center;'>Timesteps</td><td style='text-align: center;'>52696</td><td style='text-align: center;'>17544</td><td style='text-align: center;'>26304</td><td style='text-align: center;'>17420</td><td style='text-align: center;'>69680</td><td style='text-align: center;'>7588</td></tr><tr><td style='text-align: center;'>Frequency</td><td style='text-align: center;'>10m</td><td style='text-align: center;'>1h</td><td style='text-align: center;'>1h</td><td style='text-align: center;'>1h</td><td style='text-align: center;'>15m</td><td style='text-align: center;'>1d</td></tr><tr><td style='text-align: center;'>Lookback Window</td><td style='text-align: center;'>96</td><td style='text-align: center;'>48</td><td style='text-align: center;'>96</td><td style='text-align: center;'>96</td><td style='text-align: center;'>96</td><td style='text-align: center;'>96</td></tr><tr><td style='text-align: center;'>Prediction Length</td><td style='text-align: center;'>96, 192, 336, 720</td><td style='text-align: center;'>96, 192, 336, 720</td><td style='text-align: center;'>96, 192, 336, 720</td><td style='text-align: center;'>96, 192, 336, 720</td><td style='text-align: center;'>96, 192, 336, 720</td><td style='text-align: center;'>96, 192, 336, 720</td></tr></table>

<div style="text-align: center;">Table 6: Long Term Datasets Parameters</div>


### B.2 BASELINES

We employ a selection of SoTA representative models for our comparative analysis, focusing on Transformer-based architectures and other popular models. The models included are as follows:

• Informer: Informer enhances the efficiency of self-attention mechanisms to effectively capture dependencies across variables. The source code was obtained from GitHub, and we utilized the default configuration with a dropout rate of 0.05, two encoder layers, one decoder layer, a learning rate of 0.0001, and the Adam optimizer.

• Reformer: Reformer combines the power of Transformers with efficient memory and computation management, especially for long sequences. The source code was sourced from GitHub, and we employed the recommended configuration for our experiments.

• Autoformer: Autoformer introduces a decomposition block embedded within the model to progressively aggregate long-term trends from intermediate predictions. The source code was accessed from GitHub, and we followed the recommended settings for all experiments.

• FEDformer: FEDformer introduces an attention mechanism based on low-rank approximation in the frequency domain combined with a mixture of expert decomposition to handle distribution shifts. The source code was retrieved from GitHub. We utilized the Frequency Enhanced Block (FEB-f) and selected the random mode with 64 as the experimental configuration.

• LTSF-Linear: LTSF-Linear is a minimalist model employing simple one-layer linear models to learn temporal relationships in time series data. We used it as our baseline for long-term forecasting, downloading the source code from GitHub, and adhered to the default experimental settings.

• PatchTST: PatchTST is a Transformer-based model designed for TSF, introducing patching and a channel-independent structure to enhance model performance. The source code was obtained from GitHub, and we used the recommended settings for all experiments.

• FreTS: FRETS is a sophisticated model tailored for efficient TSF by exploiting a frequency domain approach. The implementation is available on GitHub, and we utilized the default configuration as recommended by the authors. In our work, FRETS serves as the foundational model. We address its limitations, particularly its handling of non-stationary data, while adapting its strengths, such as its complex frequency learner. To fully grasp the contributions of this paper, we recommend reviewing FRETS in detail first.

### B.3 IMPLEMENTATION DETAILS

Table 7 lists the hyperparameter values used in the FIA-Net implementation. Both WM-MLP and HC-MLP backbones are implemented with the same hyperparameter values, except for p, the number of STFT windows.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>DataSets</td><td style='text-align: center;'>Weather</td><td style='text-align: center;'>Traffic</td><td style='text-align: center;'>Electricity</td><td style='text-align: center;'>ETTh1</td><td style='text-align: center;'>ETTm1</td><td style='text-align: center;'>Exchange rate</td></tr><tr><td style='text-align: center;'>Batch Size</td><td style='text-align: center;'>16</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td><td style='text-align: center;'>8</td><td style='text-align: center;'>8</td><td style='text-align: center;'>8</td></tr><tr><td style='text-align: center;'>Embed Size</td><td style='text-align: center;'>128</td><td style='text-align: center;'>32</td><td style='text-align: center;'>64</td><td style='text-align: center;'>128</td><td style='text-align: center;'>128</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>Hidden Size</td><td style='text-align: center;'>256</td><td style='text-align: center;'>256</td><td style='text-align: center;'>256</td><td style='text-align: center;'>256</td><td style='text-align: center;'>256</td><td style='text-align: center;'>256</td></tr><tr><td style='text-align: center;'>NFF</td><td style='text-align: center;'>16</td><td style='text-align: center;'>32</td><td style='text-align: center;'>32</td><td style='text-align: center;'>6</td><td style='text-align: center;'>48</td><td style='text-align: center;'>32</td></tr><tr><td style='text-align: center;'>STFT Windows</td><td style='text-align: center;'>7</td><td style='text-align: center;'>13</td><td style='text-align: center;'>13</td><td style='text-align: center;'>33</td><td style='text-align: center;'>4</td><td style='text-align: center;'>13</td></tr><tr><td style='text-align: center;'>S-M</td><td style='text-align: center;'>10</td><td style='text-align: center;'>$ M_{\text{max}} $</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td><td style='text-align: center;'>$ M_{\text{max}} $</td></tr><tr><td style='text-align: center;'>Epoch</td><td style='text-align: center;'>10</td><td style='text-align: center;'>10</td><td style='text-align: center;'>10</td><td style='text-align: center;'>10</td><td style='text-align: center;'>10</td><td style='text-align: center;'>10</td></tr></table>

<div style="text-align: center;">Table 7: Hyperparameter Settings for Long-Term Datasets for the WM-MLP and HC-MLP</div>


### B.4 EVALUATION METRICS

In this study, we use the Mean Squared Error (MSE) as the loss function during training. However, for evaluation, we report both the Mean Absolute Error (MAE) and the Root Mean Squared Error (RMSE).

which are defined as follows:

 $$ \mathbf{MSE}=\frac{1}{n}\sum_{i=1}^{n}(Y_{i}-\hat{Y}_{i})^{2},\quad\mathbf{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(Y_{i}-\hat{Y}_{i})^{2}},\quad\mathbf{MAE}=\frac{1}{n}\sum_{i=1}^{n}|Y_{i}-\hat{Y}_{i}| $$ 

Where:

•  $ Y_{i} $  represents the true target values.

•  $ \hat{Y}_{i} $  represents the predicted values,

• n is the total number of samples.

### B.5 NORMALIZATION METHODS

In this study, similar to the FRETS model Yi et al. (2023b), we apply min-max normalization to standardize the input data to the range between 0 and 1. This method helps in ensuring that all features contribute equally to the model and prevents any specific feature from dominating due to differences in scale. The formula for min-max normalization is given by:

 $$ X_{Norm}=\frac{X-X_{\min}}{X_{\max}-X_{\min}} $$ 

By normalizing the data, we ensure that all input features are within the same range, which can improve model convergence and performance.

## C ADDITIONAL INFORMATION ON HC NUMBERS AND MODELS

In this section we extend the discussion on HC numbers, considering additional values of p beyond p = 4. We couple the presentation with the construction of the corresponding HC-MLP in the considered base. Recall that the base of a HC number, i.e., the number of its components is given by b = 2p. While hyper-complex number can be defined for any value of b, most research has been performed on b that is given by a power of 2, as the resulting structure of the (algebraic) field. The addition of two HC numbers is simply given by the component-wise summation. In what follows, we focus on HC multiplication and additional properties. For more information on the HC number system,, we refer the reader to Kantor & Solodovnikov (1989).

### C.1 BASE 2 - COMPLEX NUMBERS

When b = 2, the resulting field is the complex plane C. We describe C for completeness of presentation. Given two complex numbers  $ C_{1} = \alpha_{1} + j\alpha_{2} $  and  $ C_{2} = \beta_{1} + j\beta_{2} $ , where  $ \alpha_{1}, \alpha_{2}, \beta_{1}, \beta_{2} $  are real numbers, their complex multiplication is defined as:

 $$ C_{1}\cdot C_{2}=(\alpha_{1}\beta_{1}-\alpha_{2}\beta_{2})+j(\alpha_{1}\beta_{2}+\alpha_{2}\beta_{1}) $$ 

The norm of a complex number is given by:

 $$ |C_{1}|_{\mathbb{C}}=\sqrt{\alpha_{1}^{2}+\alpha_{2}^{2}}, $$ 

which is preserved under multiplication, i.e.,

 $$ |C_{1}\cdot C_{2}|_{\mathbb{C}}=|C_{1}|_{\mathbb{C}}\cdot|C_{2}|_{\mathbb{C}}. $$ 

Since the STFT with a single window (p = 1) is equivalent to the standard FFT, applying our method for hyper-complex number MLP results in the following equation:

 $$ C_{in}=FFT(X) $$ 

 $$ C^{\mathrm{out}}=\sigma(C_{\mathrm{Real}}^{\mathrm{in}}\cdot W_{\mathrm{1,Real}}-C_{\mathrm{Imag}}^{\mathrm{in}}\cdot W_{\mathrm{1,Imag}}+B_{\mathrm{1,Real}})+\sigma(j(C_{\mathrm{Real}}^{\mathrm{in}}\cdot W_{\mathrm{1,Imag}}+C_{\mathrm{Imag}}^{\mathrm{in}}\cdot W_{\mathrm{1,Real}}+B_{\mathrm{1,Imag}}) $$ 

Here,  $ W_{i} \in C^{E \times E} $  denotes the layer weights,  $ B \in C^{E} $  represents the bias term, and the multiplication occurs across the embedding dimension. Note that for b = 2 the HV formulation boils down to the one from Yi et al. (2023b). Thus, the HC-MLP can be considered as an HC generalization of the FD-MLP, which allows for efficient window aggregation.

### C.2 BASE 4 - QUATERNIONS

Denote the field of Quaternions with $\tilde{\mathbb{Q}}$. We represent Quatenions with a couple of Complex number, i.e., for $H_{1}, H_{2} \in \tilde{\mathbb{Q}}$, $H_{1} = (\alpha_{1}, \alpha_{2})$ and $H_{2} = (\beta_{1}, \beta_{2})$, their multiplication is defined as

 $$ H_{1}\cdot H_{2}=\left(\alpha_{1}\beta_{1}-\overline{\alpha_{2}}\beta_{2},\quad\alpha_{2}\overline{\beta_{1}}+\alpha_{1}\beta_{2}\right) $$ 

The norm of a quaternion is given by:

 $$ |q|_{\tilde{\mathbb{Q}}}=\sqrt{|\alpha_{1}|_{\mathbb{C}}^{2}+|\alpha_{2}|_{\mathbb{C}}^{2}} $$ 

The norm is preserved under multiplication, meaning:

 $$ |q_{1}\cdot q_{2}|_{\tilde{\mathbb{Q}}}=|q_{1}|_{\tilde{\mathbb{Q}}}\cdot|q_{2}|_{\tilde{\mathbb{Q}}} $$ 

For our model, the corresponding HC-MLP (which we denote QuatMLP) operating on  $ C^{\mathrm{in}} = (C_{1}^{\mathrm{in}}, C_{2}^{\mathrm{in}}) \in \tilde{\mathbb{Q}} $ , is given by,

 $$ C^{\mathsf{out}}=\mathsf{QuatMLP}(C^{\mathsf{in}})=\sigma(C^{\mathsf{in}}\cdot W+B) $$ 

where:

 $$ C_{1}^{\mathsf{out}}=\sigma(C_{1}\cdot W_{1}-\overline{{C_{2}}}\cdot W_{2}+B_{1}),\quad C_{2}^{\mathsf{out}}=\sigma(C_{2}\cdot\overline{{W_{1}}}+C_{1}\cdot W_{2}+B_{2}). $$ 

Here,  $ W_{i} \in C^{E \times E} $ , i = 1, 2 denote the layer weights,  $ B \in C^{E} $  represents the bias term, and the multiplication involves complex MLP operations across the embedding dimension.

### C.3 BASE 16 - SEDENIONS

Elements on the Sedenions field, denoted SS, are denoted with 8-tuples of complex numbers. Given two sedenions represented by complex numbers  $ S_{1}, S_{2} \in SS $ ,  $ S_{1} = (\alpha_{1}, \alpha_{2}, \ldots, \alpha_{8}) $  and  $ S_{2} = (\beta_{1}, \beta_{2}, \ldots, \beta_{8}) $ , their multiplication is given by

 $$ S_{1}\cdot S_{2}=\begin{pmatrix}\alpha_{1}\beta_{1}-\alpha_{2}\overline{\beta_{2}}-\alpha_{3}\overline{\beta_{3}}-\alpha_{4}\overline{\beta_{4}}-\alpha_{5}\overline{\beta_{5}}-\alpha_{6}\overline{\beta_{6}}-\alpha_{7}\overline{\beta_{7}}-\alpha_{8}\overline{\beta_{8}}\\\alpha_{1}\beta_{2}+\alpha_{2}\beta_{1}+\alpha_{3}\overline{\beta_{4}}-\alpha_{4}\overline{\beta_{3}}+\alpha_{5}\overline{\beta_{6}}-\alpha_{6}\overline{\beta_{5}}+\alpha_{7}\overline{\beta_{8}}-\alpha_{8}\overline{\beta_{7}}\\\alpha_{1}\beta_{3}-\alpha_{2}\overline{\beta_{4}}+\alpha_{3}\beta_{1}+\alpha_{4}\beta_{2}+\alpha_{5}\overline{\beta_{7}}-\alpha_{6}\overline{\beta_{8}}-\alpha_{7}\overline{\beta_{5}}+\alpha_{8}\overline{\beta_{6}}\\\alpha_{1}\beta_{4}+\alpha_{2}\beta_{3}-\alpha_{3}\beta_{2}+\alpha_{4}\beta_{1}+\alpha_{5}\overline{\beta_{8}}+\alpha_{6}\overline{\beta_{7}}-\alpha_{7}\overline{\beta_{6}}-\alpha_{8}\overline{\beta_{5}}\\\alpha_{1}\beta_{5}-\alpha_{2}\overline{\beta_{6}}-\alpha_{3}\overline{\beta_{7}}-\alpha_{4}\overline{\beta_{8}}+\alpha_{5}\beta_{1}+\alpha_{6}\beta_{2}+\alpha_{7}\beta_{3}+\alpha_{8}\beta_{4}\\\alpha_{1}\beta_{6}+\alpha_{2}\beta_{5}-\alpha_{3}\overline{\beta_{8}}+\alpha_{4}\overline{\beta_{7}}-\alpha_{5}\beta_{2}+\alpha_{6}\beta_{1}-\alpha_{7}\beta_{4}+\alpha_{8}\beta_{3}\\\alpha_{1}\beta_{7}+\alpha_{2}\beta_{8}+\alpha_{3}\beta_{5}-\alpha_{4}\beta_{6}-\alpha_{5}\beta_{3}+\alpha_{6}\beta_{4}+\alpha_{7}\beta_{1}-\alpha_{8}\beta_{2}\\\alpha_{1}\beta_{8}-\alpha_{2}\beta_{7}+\alpha_{3}\beta_{6}+\alpha_{4}\beta_{5}-\alpha_{5}\beta_{4}-\alpha_{6}\beta_{3}+\alpha_{7}\beta_{2}+\alpha_{8}\beta_{1}\end{pmatrix} $$ 

where each component follows the rules of Complex multiplication. The norm of a sedenion is given by:

 $$ |S|_{S S}=\sqrt{\sum_{j=1}^{8}|\alpha_{j}|_{\mathbb{C}}^{2}} $$ 

Unlike nase 2, 4 and 8, Sedenions do not preserve the norm under addition and multiplication.

The base-16 HC-MLP, denoted SedMLP, operating on an input  $ C^{in} $  from the STFT with multiple windows  $ C^{in} = (C_{j}^{in})_{j=1}^{8} $ , is given by

 $$ C^{\mathsf{out}}=\mathsf{SedMLP}(C^{\mathsf{in}})=\sigma(C^{\mathsf{in}}\cdot W+B) $$ 

where:

 $$ C_{1}^{\mathsf{o u t}}=\sigma\left(C_{1}^{\mathsf{i n}}W_{1}-C_{2}^{\mathsf{i n}}\overline{W_{2}}-C_{3}^{\mathsf{i n}}\overline{W_{3}}-C_{4}^{\mathsf{i n}}\overline{W_{4}}-C_{5}^{\mathsf{i n}}\overline{W_{5}}-C_{6}^{\mathsf{i n}}\overline{W_{6}}-C_{7}^{\mathsf{i n}}\overline{W_{7}}-C_{8}^{\mathsf{i n}}\overline{W_{8}}+B_{1}\right) $$ 

 $$ C_{2}^{\mathsf{o u t}}=\sigma\left(C_{1}^{\mathsf{i n}}W_{2}+C_{2}^{\mathsf{i n}}W_{1}+C_{3}^{\mathsf{i n}}\overline{W_{4}}-C_{4}^{\mathsf{i n}}\overline{W_{3}}+C_{5}^{\mathsf{i n}}\overline{W_{6}}-C_{6}^{\mathsf{i n}}\overline{W_{5}}+C_{7}^{\mathsf{i n}}\overline{W_{8}}-C_{8}^{\mathsf{i n}}\overline{W_{7}}+B_{2}\right) $$ 

 $$ C_{3}^{\mathrm{out}}=\sigma\left(C_{1}^{\mathrm{in}}W_{3}-C_{2}^{\mathrm{in}}\overline{W_{4}}+C_{3}^{\mathrm{in}}W_{1}+C_{4}^{\mathrm{in}}W_{2}+C_{5}^{\mathrm{in}}\overline{W_{7}}-C_{6}^{\mathrm{in}}\overline{W_{8}}-C_{7}^{\mathrm{in}}\overline{W_{5}}+C_{8}^{\mathrm{in}}\overline{W_{6}}+B_{3}\right) $$ 

 $$ C_{4}^{\mathsf{out}}=\sigma\left(C_{1}^{\mathsf{in}}W_{4}+C_{2}^{\mathsf{in}}W_{3}-C_{3}^{\mathsf{in}}W_{2}+C_{4}^{\mathsf{in}}W_{1}+C_{5}^{\mathsf{in}}\overline{W_{8}}+C_{6}^{\mathsf{in}}\overline{W_{7}}-C_{7}^{\mathsf{in}}\overline{W_{6}}-C_{8}^{\mathsf{in}}\overline{W_{5}}+B_{4}\right) $$ 

 $$ C_{5}^{\mathsf{out}}=\sigma\left(C_{1}^{\mathsf{in}}W_{5}-C_{2}^{\mathsf{in}}\overline{W_{6}}-C_{3}^{\mathsf{in}}\overline{W_{7}}-C_{4}^{\mathsf{in}}\overline{W_{8}}+C_{5}^{\mathsf{in}}W_{1}+C_{6}^{\mathsf{in}}W_{2}+C_{7}^{\mathsf{in}}W_{3}+C_{8}^{\mathsf{in}}W_{4}+B_{5}\right) $$ 

 $$ C_{6}^{\mathsf{out}}=\sigma\left(C_{1}^{\mathsf{in}}W_{6}+C_{2}^{\mathsf{in}}W_{5}-C_{3}^{\mathsf{in}}\overline{W_{8}}+C_{4}^{\mathsf{in}}\overline{W_{7}}-C_{5}^{\mathsf{in}}W_{2}+C_{6}^{\mathsf{in}}W_{1}-C_{7}^{\mathsf{in}}W_{4}+C_{8}^{\mathsf{in}}W_{3}+B_{6}\right) $$ 

 $$ C_{7}^{\mathsf{out}}=\sigma\left(C_{1}^{\mathsf{in}}W_{7}+C_{2}^{\mathsf{in}}W_{8}+C_{3}^{\mathsf{in}}W_{5}-C_{4}^{\mathsf{in}}W_{6}-C_{5}^{\mathsf{in}}W_{3}+C_{6}^{\mathsf{in}}W_{4}+C_{7}^{\mathsf{in}}W_{1}-C_{8}^{\mathsf{in}}W_{2}+B_{7}\right) $$ 

 $$ C_{8}^{\mathsf{out}}=\sigma\left(C_{1}^{\mathsf{in}}W_{8}-C_{2}^{\mathsf{in}}W_{7}+C_{3}^{\mathsf{in}}W_{6}+C_{4}^{\mathsf{in}}W_{5}-C_{5}^{\mathsf{in}}W_{4}-C_{6}^{\mathsf{in}}W_{3}+C_{7}^{\mathsf{in}}W_{2}+C_{8}^{\mathsf{in}}W_{1}+B_{8}\right) $$ 

Here,  $ W_{i} \in C^{E \times E}, i = 1, \ldots, 8 $  denotes the layer weights,  $ B \in C^{E} $  represents the bias term, and the multiplication involves complex MLP operations across the embedding dimension.

## D ADDITIONAL ABLATION STUDIES

This section presents additional ablation studies, expanding on the findings reported in Section 5.3. We analyze the impact of FFT resolution, embedding size, and the number of STFT windows on WM-MLP performance. Additionally, we include further results for the frequency compression, sequence length, and real vs. imaginary component discussions. Furthermore, we provide a comparative analysis of various hyper-complex fields (octonions, quaternions, and sedenions) for the HC-MLP and report the corresponding results.

### D.1 PARAMETER SENSITIVITY

In this section, we conduct a parameter sweep to examine the effects of different hyperparameters on model performance. To accomplish this, we utilize two datasets: the ETTh1 dataset and the electricity dataset. Each section presents four graphs illustrating the results on the two datasets for a configuration of  $ I/O = 96 \times 96, 336 $ . Except for the specific experiment sweep, the embedding size is set to 128 for the ETTh1 dataset and 64 for the electricity dataset, with M set to 0 for all datasets.

Embed Size In this section, we evaluate the influence of embedding size on the model's performance. We conducted experiments with embedding dimensions  $ E \in \{1, 2, 4, 8, 16, 32, 64, 128, 256, 512\} $ , while keeping the following parameters fixed:  $ N_{FFT} = 16 $ , B = 8, p = 13, and  $ M = M_{max} $ . We can observe that as we increase the embedding size, the loss decreases until we reach a certain point (which is dependent on the dataset). This is likely because a larger embedding size enables the model to capture more features; however, an excessively high embedding size may lead to overfitting.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_216_732_406_876.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(a) T=96 on ETTh1</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_415_731_606_876.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(b) T=336 on ETTh</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_730_804_876.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(c) T=96 on electricity</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_813_730_1005_875.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(d) T=336 on electricity</div>


<div style="text-align: center;">Figure 7: Comparison of MSE and MAE across different values of E for varying T on the ETTh1 and Electricity datasets.</div>


Amount of Windows (High Dim) In this section, we evaluate the influence of the number of windows  $ (p) $  on the model's performance. We conducted experiments with different window counts  $ p \in \{3, 6, 14, 17, 25, 33\} $ , while keeping the following parameters fixed: B = 8,  $ M = M_{max} $ , and the overlap between windows is 50%.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_1137_405_1282.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(a) T=96 on ETTh1</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_415_1137_607_1282.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(b) T=336 on ETTh1</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_616_1138_806_1282.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(c) T=96 on electricity</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_815_1138_1004_1282.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(d) T=336 on electricity</div>


<div style="text-align: center;">Figure 8: Comparison of MSE and MAE across different values of p for varying T on the ETTh1 and Electricity datasets.</div>


FFT Resolution (NFFT) In this section, we evaluate the influence of the FFT resolution  $ (N_{\mathrm{FFT}}) $  on the model's performance. We conducted experiments with different  $ N_{FFT} \in $ 

 $ \{6,8,12,16,24,32,48\} $ , while keeping the following parameters fixed: p = 25, B = 8,  $ M = M_{max} $ .

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_230_407_376.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(a) T=96 on ETTh1</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_416_230_605_376.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(b) T=336 on ETTh1</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_614_230_805_376.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(c) T=96 on electricity</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_814_230_1006_376.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(d) T=336 on electricity</div>


Figure 9: Comparison of MSE and MAE across different values of  $ N_{FFT} $  for varying T on the ETTh1 and Electricity datasets.

Frequency Choose Max (M) In this section, we provide additional results for various datasets and prediction lengths T regarding the discussion on frequency compression 5.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_579_406_723.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(a) T=96 on ETTh1</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_415_579_606_724.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(b) T=336 on ETTh1</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_578_805_724.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(c) T=96 on electricity</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_813_578_1006_724.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(d) T=336 on electricity</div>


Figure 10: Comparison of MSE and MAE across different values of M for various T on the ETTh1 and Electricity datasets.

### D.2 DIFFERENT LOOKBACK WINDOW

In this section, we present additional results for various lookback windows on the ETTh1 and ETTm1 datasets.

 $$ \begin{aligned}\xrightarrow{\quad}\quad&\text{T}=96(MAE)\quad&\xrightarrow{\quad}\quad&\text{T}=336(MAE)\\\xrightarrow{\quad}\quad&\text{T}=192(MAE)\quad&\xrightarrow{\quad}\quad&\text{T}=720(MAE)\end{aligned} $$ 

<div style="text-align: center;"><img src="imgs/img_in_chart_box_255_1033_600_1297.jpg" alt="Image" width="28%" /></div>


<div style="text-align: center;">(a) ETTh1 Dataset</div>


 $$ \begin{aligned}\bullet\bullet\bullet\quad&T=96(RMSE)\quad&\cdots\bullet\cdots\quad&T=336(RMSE)\\\bullet\bullet\bullet\quad&T=192(RMSE)\quad&\cdots\bullet\cdots\quad&T=720(RMSE)\end{aligned} $$ 

<div style="text-align: center;"><img src="imgs/img_in_chart_box_619_1034_965_1296.jpg" alt="Image" width="28%" /></div>


<div style="text-align: center;">(b) ETTm1 Dataset</div>


<div style="text-align: center;">Figure 11: MAE and RMSE in relation to the Lookback Window L for varying prediction lengths  $ T \in \{96, 192, 336, 720\} $  for the ETTh1 and ETTm1 datasets.</div>


### D.3 REAL VS IMAGINARY COMPONENTS

<div style="text-align: center;">This section provides additional information regarding the real versus imaginary experiment discussed in Section 5.3.3.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Dataset</td><td style='text-align: center;'>I/O</td><td colspan="2">96/96</td><td colspan="2">96/192</td><td colspan="2">96/336</td><td colspan="2">96/720</td></tr><tr><td style='text-align: center;'>Hidden Part</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>MAE</td><td style='text-align: center;'>RMSE</td></tr><tr><td rowspan="7">ETTm1</td><td style='text-align: center;'>$ X^{\text{Real}} $</td><td style='text-align: center;'>0.0522</td><td style='text-align: center;'>0.0797</td><td style='text-align: center;'>0.0560</td><td style='text-align: center;'>0.0850</td><td style='text-align: center;'>0.0597</td><td style='text-align: center;'>0.0888</td><td style='text-align: center;'>0.0658</td><td style='text-align: center;'>0.0958</td></tr><tr><td style='text-align: center;'>$ X^{\text{Imag}} $</td><td style='text-align: center;'>0.0521</td><td style='text-align: center;'>0.0792</td><td style='text-align: center;'>0.0562</td><td style='text-align: center;'>0.0844</td><td style='text-align: center;'>0.0592</td><td style='text-align: center;'>0.0879</td><td style='text-align: center;'>0.0684</td><td style='text-align: center;'>0.0976</td></tr><tr><td style='text-align: center;'>$ W^{\text{Real}} $</td><td style='text-align: center;'>0.0522</td><td style='text-align: center;'>0.0791</td><td style='text-align: center;'>0.0557</td><td style='text-align: center;'>0.0843</td><td style='text-align: center;'>0.0588</td><td style='text-align: center;'>0.0875</td><td style='text-align: center;'>0.0669</td><td style='text-align: center;'>0.0964</td></tr><tr><td style='text-align: center;'>$ W^{\text{Imag}} $</td><td style='text-align: center;'>0.0526</td><td style='text-align: center;'>0.0801</td><td style='text-align: center;'>0.0560</td><td style='text-align: center;'>0.0849</td><td style='text-align: center;'>0.0596</td><td style='text-align: center;'>0.0888</td><td style='text-align: center;'>0.0651</td><td style='text-align: center;'>0.0953</td></tr><tr><td style='text-align: center;'>$ W^{\text{Imag}}, X^{\text{Imag}} $</td><td style='text-align: center;'>0.0523</td><td style='text-align: center;'>0.0798</td><td style='text-align: center;'>0.0560</td><td style='text-align: center;'>0.0849</td><td style='text-align: center;'>0.0592</td><td style='text-align: center;'>0.0884</td><td style='text-align: center;'>0.0644</td><td style='text-align: center;'>0.0947</td></tr><tr><td style='text-align: center;'>$ W^{\text{Real}}, X^{\text{Real}} $</td><td style='text-align: center;'>0.0522</td><td style='text-align: center;'>0.0791</td><td style='text-align: center;'>0.0557</td><td style='text-align: center;'>0.0843</td><td style='text-align: center;'>0.0588</td><td style='text-align: center;'>0.0887</td><td style='text-align: center;'>0.0669</td><td style='text-align: center;'>0.0930</td></tr><tr><td style='text-align: center;'>Normal</td><td style='text-align: center;'>0.0522</td><td style='text-align: center;'>0.0791</td><td style='text-align: center;'>0.0565</td><td style='text-align: center;'>0.0848</td><td style='text-align: center;'>0.0592</td><td style='text-align: center;'>0.0878</td><td style='text-align: center;'>0.0685</td><td style='text-align: center;'>0.0975</td></tr><tr><td rowspan="7">ETTh1</td><td style='text-align: center;'>$ X^{\text{Real}} $</td><td style='text-align: center;'>0.0584</td><td style='text-align: center;'>0.0877</td><td style='text-align: center;'>0.0638</td><td style='text-align: center;'>0.0944</td><td style='text-align: center;'>0.0684</td><td style='text-align: center;'>0.0997</td><td style='text-align: center;'>0.0767</td><td style='text-align: center;'>0.1047</td></tr><tr><td style='text-align: center;'>$ X^{\text{Imag}} $</td><td style='text-align: center;'>0.0582</td><td style='text-align: center;'>0.0879</td><td style='text-align: center;'>0.0634</td><td style='text-align: center;'>0.0943</td><td style='text-align: center;'>0.0679</td><td style='text-align: center;'>0.0997</td><td style='text-align: center;'>0.0756</td><td style='text-align: center;'>0.1041</td></tr><tr><td style='text-align: center;'>$ W^{\text{Real}} $</td><td style='text-align: center;'>0.0586</td><td style='text-align: center;'>0.0880</td><td style='text-align: center;'>0.0644</td><td style='text-align: center;'>0.0948</td><td style='text-align: center;'>0.0685</td><td style='text-align: center;'>0.0998</td><td style='text-align: center;'>0.0759</td><td style='text-align: center;'>0.1039</td></tr><tr><td style='text-align: center;'>$ W^{\text{Imag}} $</td><td style='text-align: center;'>0.0584</td><td style='text-align: center;'>0.0880</td><td style='text-align: center;'>0.0646</td><td style='text-align: center;'>0.0951</td><td style='text-align: center;'>0.0694</td><td style='text-align: center;'>0.1008</td><td style='text-align: center;'>0.0781</td><td style='text-align: center;'>0.1065</td></tr><tr><td style='text-align: center;'>$ W^{\text{Imag}}, X^{\text{Imag}} $</td><td style='text-align: center;'>0.0586</td><td style='text-align: center;'>0.0880</td><td style='text-align: center;'>0.0644</td><td style='text-align: center;'>0.0947</td><td style='text-align: center;'>0.0685</td><td style='text-align: center;'>0.0998</td><td style='text-align: center;'>0.0759</td><td style='text-align: center;'>0.1040</td></tr><tr><td style='text-align: center;'>$ W^{\text{Real}}, X^{\text{Real}} $</td><td style='text-align: center;'>0.0587</td><td style='text-align: center;'>0.0882</td><td style='text-align: center;'>0.0642</td><td style='text-align: center;'>0.0948</td><td style='text-align: center;'>0.0690</td><td style='text-align: center;'>0.1005</td><td style='text-align: center;'>0.0765</td><td style='text-align: center;'>0.1050</td></tr><tr><td style='text-align: center;'>Normal</td><td style='text-align: center;'>0.0586</td><td style='text-align: center;'>0.0878</td><td style='text-align: center;'>0.0639</td><td style='text-align: center;'>0.0945</td><td style='text-align: center;'>0.0684</td><td style='text-align: center;'>0.0998</td><td style='text-align: center;'>0.0765</td><td style='text-align: center;'>0.1043</td></tr></table>

Table 8: Performance comparison on the ETTm1, ETTh1, and Electricity datasets for  $ I/O = 96 \times \{96, 192, 336, 720\} $  with different modes.  $ X^{Real} $  and  $ X^{Imag} $  refer to hiding the real and imaginary parts of the input, respectively.  $ W^{Real} $  and  $ W^{Imag} $  denote zeroing the real and imaginary weights, respectively. The cases where both the real and imaginary components are completely ignored (i.e., both weights and inputs are zeroed) are represented by  $ W^{Imag} $ ,  $ X^{Imag} $  and  $ W^{Real} $ ,  $ X^{Real} $ . MAE and RMSE are reported, where lower values indicate better performance.

### D.4 HC-MLP EXPERIMENTAL RESULTS WITH FOR VARIOUS VALUES OF p

<div style="text-align: center;">In this section, we present additional results on the HC-MLP for various bases. Specifically, we provide results for the Quaternion base  $ (p = 2, QuatMLP) $ , Octonion base  $ (p = 4, OctMLP) $ , and Sedenion base  $ (p = 8, SedMLP) $ . Additionally, we include results for a model that aggregates all windows without using hyper-complex numbers, referred to as BasicMLP. Further details about its implementation can be found in B.3.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td rowspan="2">Metric</td><td colspan="4">Traffic</td><td colspan="4">ETTh1</td><td colspan="4">ETTm1</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td></tr><tr><td rowspan="2">SedenionMLP ( $ p = 8 $ )</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.0340</td><td style='text-align: center;'>0.0346</td><td style='text-align: center;'>0.0351</td><td style='text-align: center;'>0.0363</td><td style='text-align: center;'>0.0896</td><td style='text-align: center;'>0.0948</td><td style='text-align: center;'>0.0999</td><td style='text-align: center;'>0.1047</td><td style='text-align: center;'>0.0814</td><td style='text-align: center;'>0.0857</td><td style='text-align: center;'>0.0894</td><td style='text-align: center;'>0.0977</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.0168</td><td style='text-align: center;'>0.0169</td><td style='text-align: center;'>0.0173</td><td style='text-align: center;'>0.0186</td><td style='text-align: center;'>0.0598</td><td style='text-align: center;'>0.0640</td><td style='text-align: center;'>0.0685</td><td style='text-align: center;'>0.0767</td><td style='text-align: center;'>0.0542</td><td style='text-align: center;'>0.0573</td><td style='text-align: center;'>0.0609</td><td style='text-align: center;'>0.0682</td></tr><tr><td rowspan="2">OctontionMLP ( $ p = 4 $ )</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.0335</td><td style='text-align: center;'>0.0343</td><td style='text-align: center;'>0.0349</td><td style='text-align: center;'>0.0361</td><td style='text-align: center;'>0.0834</td><td style='text-align: center;'>0.0874</td><td style='text-align: center;'>0.0941</td><td style='text-align: center;'>0.1017</td><td style='text-align: center;'>0.0739</td><td style='text-align: center;'>0.0831</td><td style='text-align: center;'>0.0888</td><td style='text-align: center;'>0.0967</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.0166</td><td style='text-align: center;'>0.0167</td><td style='text-align: center;'>0.0172</td><td style='text-align: center;'>0.0185</td><td style='text-align: center;'>0.0579</td><td style='text-align: center;'>0.0635</td><td style='text-align: center;'>0.0676</td><td style='text-align: center;'>0.0759</td><td style='text-align: center;'>0.0496</td><td style='text-align: center;'>0.0556</td><td style='text-align: center;'>0.0603</td><td style='text-align: center;'>0.0673</td></tr><tr><td rowspan="2">QuaternionMLP ( $ p = 2 $ )</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.0335</td><td style='text-align: center;'>0.0343</td><td style='text-align: center;'>0.0350</td><td style='text-align: center;'>0.0362</td><td style='text-align: center;'>0.0874</td><td style='text-align: center;'>0.0938</td><td style='text-align: center;'>0.0997</td><td style='text-align: center;'>0.1059</td><td style='text-align: center;'>0.0796</td><td style='text-align: center;'>0.0847</td><td style='text-align: center;'>0.0887</td><td style='text-align: center;'>0.0974</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.0165</td><td style='text-align: center;'>0.0167</td><td style='text-align: center;'>0.0172</td><td style='text-align: center;'>0.0184</td><td style='text-align: center;'>0.0580</td><td style='text-align: center;'>0.0633</td><td style='text-align: center;'>0.0687</td><td style='text-align: center;'>0.0783</td><td style='text-align: center;'>0.0526</td><td style='text-align: center;'>0.0564</td><td style='text-align: center;'>0.0603</td><td style='text-align: center;'>0.0678</td></tr><tr><td rowspan="2">BasicMLP</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.0372</td><td style='text-align: center;'>0.0391</td><td style='text-align: center;'>0.0384</td><td style='text-align: center;'>0.0415</td><td style='text-align: center;'>0.0962</td><td style='text-align: center;'>0.1025</td><td style='text-align: center;'>0.1061</td><td style='text-align: center;'>0.1187</td><td style='text-align: center;'>0.0832</td><td style='text-align: center;'>0.0903</td><td style='text-align: center;'>0.0967</td><td style='text-align: center;'>0.1066</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.0180</td><td style='text-align: center;'>0.0195</td><td style='text-align: center;'>0.0201</td><td style='text-align: center;'>0.0217</td><td style='text-align: center;'>0.0650</td><td style='text-align: center;'>0.0714</td><td style='text-align: center;'>0.0761</td><td style='text-align: center;'>0.0886</td><td style='text-align: center;'>0.0546</td><td style='text-align: center;'>0.0595</td><td style='text-align: center;'>0.0649</td><td style='text-align: center;'>0.0753</td></tr></table>

<div style="text-align: center;">Table 9: Comparison of different hypercomplex structures on the ETT and Traffic datasets. QuadMLP (2 windows), OctMLP (4 windows), and SedMLP (8 windows) represent hypercomplex models of increasing dimensionality, while BasicMLP is a non-hypercomplex linear model aggregating window information. Performance is reported using MSE and RMSE metrics, where lower values indicate better accuracy.</div>


### D.5 EXTENDED NEIGHBORHOOD AGGREGATION IN WM-MLP

In this section, we present additional results on the WM-MLP with extended neighborhood aggregation. Specifically, we provide results for varying neighborhood sizes, where the model incorporates information not only from directly adjacent windows but also from second-order and third-order neighbors. The experiments were conducted on the ETTm1 and ETTh1 datasets with prediction lengths of 96, 192, 336, and 720.

For the two-neighbor case, the output  $ C_{i}^{out} $  is computed as:

 $$ \begin{aligned}C_{i}^{\mathrm{out}}=\sigma\Big(C_{i}^{\mathrm{in}}W_{i\rightarrow i}&+C_{i-1}^{\mathrm{in}}W_{(i-1)\rightarrow i}+C_{i+1}^{\mathrm{in}}W_{(i+1)\rightarrow i}\\&+C_{i-2}^{\mathrm{in}}W_{(i-2)\rightarrow i}+C_{i+2}^{\mathrm{in}}W_{(i+2)\rightarrow i}+B_{i}\Big).\end{aligned} $$ 

For the three-neighbor case, the output  $ C_{i}^{out} $  is computed as:

 $$ \begin{aligned}C_{i}^{\mathrm{out}}=\sigma\Big(C_{i}^{\mathrm{in}}W_{i\rightarrow i}&+C_{i-1}^{\mathrm{in}}W_{(i-1)\rightarrow i}+C_{i+1}^{\mathrm{in}}W_{(i+1)\rightarrow i}\\&+C_{i-2}^{\mathrm{in}}W_{(i-2)\rightarrow i}+C_{i+2}^{\mathrm{in}}W_{(i+2)\rightarrow i}\\&+C_{i-3}^{\mathrm{in}}W_{(i-3)\rightarrow i}+C_{i+3}^{\mathrm{in}}W_{(i+3)\rightarrow i}+B_{i}\Big).\end{aligned} $$ 


<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td rowspan="2">Metric</td><td colspan="4">ETTh1</td><td colspan="4">ETTm1</td></tr><tr><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td><td style='text-align: center;'>96</td><td style='text-align: center;'>192</td><td style='text-align: center;'>336</td><td style='text-align: center;'>720</td></tr><tr><td rowspan="2">WM-MLP (1 Neighbor)</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.084</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>0.076</td><td style='text-align: center;'>0.082</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>0.094</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>0.064</td><td style='text-align: center;'>0.068</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.058</td><td style='text-align: center;'>0.064</td></tr><tr><td rowspan="2">WM-MLP (2 Neighbors)</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.095</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>0.106</td><td style='text-align: center;'>0.120</td><td style='text-align: center;'>0.084</td><td style='text-align: center;'>0.091</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>0.104</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.071</td><td style='text-align: center;'>0.076</td><td style='text-align: center;'>0.090</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.060</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.073</td></tr><tr><td rowspan="2">WM-MLP (3 Neighbors)</td><td style='text-align: center;'>RMSE</td><td style='text-align: center;'>0.095</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>0.109</td><td style='text-align: center;'>0.120</td><td style='text-align: center;'>0.084</td><td style='text-align: center;'>0.091</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>0.010</td></tr><tr><td style='text-align: center;'>MAE</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.071</td><td style='text-align: center;'>0.078</td><td style='text-align: center;'>0.090</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>0.060</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>0.073</td></tr></table>

Table 10: Performance comparison of WM-MLP with varying numbers of neighbors (1, 2, and 3) on the ETTh1 and ETTm1 datasets for prediction lengths of 96, 192, 336, and 720. Metrics include RMSE and MAE. Results for WM-MLP with one neighbor are derived from the baseline values reported in the original paper.

### D.6 COMPLEXITY ANALYSIS

We conducted an asymptotic analysis of modern models to compare their training time, memory usage, and testing steps. The results are summarized in Table 11. The comparison highlights the computational efficiency of the WM-MLP and HC-MLP models relative to other state-of-the-art approaches. Specifically, both models demonstrate competitive performance with logarithmic complexity in training time and memory, and a constant number of testing steps.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'>Training Time</td><td style='text-align: center;'>Training Memory</td><td style='text-align: center;'>Testing Steps</td></tr><tr><td style='text-align: center;'>WM-MLP</td><td style='text-align: center;'>$ \mathcal{O}(L \log \frac{L}{p}) $</td><td style='text-align: center;'>$ \mathcal{O}(L) $</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>HC-MLP</td><td style='text-align: center;'>$ \mathcal{O}(L \log \frac{L}{p} + p^{2}) $</td><td style='text-align: center;'>$ \frac{1}{3} \mathcal{O}(L) $</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>FreTS</td><td style='text-align: center;'>$ \mathcal{O}(L \log L) $</td><td style='text-align: center;'>$ \mathcal{O}(L) $</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>PatchTST</td><td style='text-align: center;'>$ \mathcal{O}(L/S) $</td><td style='text-align: center;'>$ \mathcal{O}(L/S) $</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>LTSF-Linear</td><td style='text-align: center;'>$ \mathcal{O}(L) $</td><td style='text-align: center;'>$ \mathcal{O}(L) $</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>FEDformer</td><td style='text-align: center;'>$ \mathcal{O}(L) $</td><td style='text-align: center;'>$ \mathcal{O}(L) $</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>Autoformer</td><td style='text-align: center;'>$ \mathcal{O}(L \log L) $</td><td style='text-align: center;'>$ \mathcal{O}(L \log L) $</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>Informer</td><td style='text-align: center;'>$ \mathcal{O}(L \log L) $</td><td style='text-align: center;'>$ \mathcal{O}(L \log L) $</td><td style='text-align: center;'>1</td></tr><tr><td style='text-align: center;'>Transformer</td><td style='text-align: center;'>$ \mathcal{O}(L^{2}) $</td><td style='text-align: center;'>$ \mathcal{O}(L^{2}) $</td><td style='text-align: center;'>L</td></tr><tr><td style='text-align: center;'>Reformer</td><td style='text-align: center;'>$ \mathcal{O}(L \log L) $</td><td style='text-align: center;'>$ \mathcal{O}(L \log L) $</td><td style='text-align: center;'>1</td></tr></table>

<div style="text-align: center;">Table 11: Comparison of models in terms of asymptotic complexity for training time, memory usage, and testing steps as a function of the lookback window length  $ (L) $ . Here, S denotes the patch size used in PatchTST, and p represents the number of windows in the STFT transformation.</div>


### D.7 VISUALIZATIONS

<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_768_598_1000.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_621_768_1002_1001.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(a) Traffic I/O = 96/96</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_1061_601_1295.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(b) Traffic I/O = 96/192</div>


<div style="text-align: center;">(c) Traffic I/O = 96/336</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_621_1063_1003_1294.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(d) Traffic I/O = 96/720</div>


<div style="text-align: center;">Figure 12: Ground Truth vs. Predictions for Different I/O Settings (Traffic Dataset).</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_510_598_742.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(a) Electricity I/O = 96/96</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_621_510_1002_742.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(b) Electricity I/O = 96/192</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_805_600_1038.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(c) Electricity I/O = 96/336</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_621_806_1002_1037.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">(d) Electricity I/O = 96/720</div>


<div style="text-align: center;">Figure 13: Ground Truth vs. Predictions for Different I/O Settings (Electricity Dataset).</div>
