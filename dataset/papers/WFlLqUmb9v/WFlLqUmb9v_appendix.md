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