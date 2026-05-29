## A MORE RELATED WORKS

Prior works have approached OOD detection from various perspectives and with different data assumptions, e.g., with or without access to training labels, batches of test data, or single test data points in a streaming fashion, and with or without knowledge and inductive bias of the data. In the following, we give an overview organized by different data assumptions with a focus on where our method fits.

The first assumption is whether the method has access to training labels. There has been extensive work on classifier-based methods that assume access to training labels (Hendrycks & Gimpel, 2016; Frosst et al., 2019; Sastry & Oore, 2020; Bahri et al., 2021; Papernot & McDaniel, 2018; Osawa et al., 2019; Guénais et al., 2020; Lakshminarayanan et al., 2016; Pearce et al., 2020). Within this category, there are different assumptions as well, such as access to a pretrained network or knowledge of OOD test examples. See Table 1 of Sastry & Oore (2020) for a summary of such methods.

When we do not assume access to the training labels, the problem becomes more general and also harder. Under this category, some methods assume access to a batch of test data where either all the data points are OOD or not (Nalisnick et al., 2019). A more general setting does not assume OOD data would come in batches. Under this setup, there are methods that implicitly assume prior knowledge of the data, such as the input complexity method (Serrà et al., 2019), where the use of image compressors implicitly assumes an image-like structure, or the likelihood ratio method (Ren et al., 2019), where a noisy background model is trained with the assumption of a background-object structure.

## B INTERPRETATION OF LIKELIHOOD CANCELLATION

Recall VAEs' likelihood estimation (parameterized by  $ \theta $ ):

 $$ \log p_{\theta}(\mathbf{x})\approx\log\left[\frac{p_{\theta}\left(\mathbf{x}\mid\mathbf{z}\right)p\left(\mathbf{z}\right)}{q_{\phi}\left(\mathbf{z}\mid\mathbf{x}\right)}\right], $$ 

The decoder  $ p_{\theta}(\mathbf{x}\mid\mathbf{z})^{\prime} $  reconstruction focuses on the pixel textures, while encoder  $ q_{\phi}(\mathbf{z}\mid\mathbf{x})^{\prime} $  samples evaluated at the prior,  $ p(\mathbf{z}) $ , describe semantics. Consider  $ x_{OOD} $ , whose lower level features are similar to ID data, but is semantically different. We can imagine  $ p_{\theta}(\mathbf{x}\mid\mathbf{z}) $  is large while  $ p(\mathbf{z}) $  is small. However, (Havtorn et al., 2021) demonstrates  $ p_{\theta}(\mathbf{x}) $  is dominated by lower level information. Even if  $ p(\mathbf{z}) $  wants to reveal  $ x_{OOD} $ 's OOD nature, we cannot decipher it through  $ p_{\theta}(\mathbf{x}_{OOD}) $ . The converse:  $ p_{\theta}(\mathbf{x}\mid\mathbf{z}) $  can flag  $ x_{OOD} $  when the reconstruction error is big. But if  $ p(\mathbf{z}) $  is unusually high compared to typical  $ x_{ID} $ ,  $ p_{\theta}(\mathbf{x}) $  may appear less OOD.

## C SUFFICIENT STATISTICS AND WHERE TO FIND THEM

Though in the case of the Gaussian parameterized VAE decoder and encoder, it is easy to find the corresponding minimal sufficient statistics, the same might not be true for more complicated distributions. Here we briefly overview the Fisher-Neyman factorization perspective on the sufficiency principle which can help find sufficient statistics in more complicated distributions. A sufficiency statistics is also characterized by Fisher-Neyman factorization theorem (Wasserman, 2006):  $ T(\mathbf{x}) $  is a sufficient statistics for  $ p(\mathbf{x}|\psi) $  parameterized by  $ \psi $  if and only if:

 $$ \ell(\psi|\mathbf{x})=p(\mathbf{x}|\psi)=f(\mathbf{x})g_{\psi}(T(\mathbf{x})) $$ 

i.e. the density  $  p(\mathbf{x}|\psi)  $  can be factored into a product such that f, does not depend on  $ \psi $  and g that does depend on  $ \psi $  but who depends on x only through  $  T(\mathbf{x})  $ . For example, if we perform inference by maximum likelihood:

 $$ \psi_{\mathrm{M L E}}=\arg\max_{\psi}\ell(\psi|\mathbf{x})=\arg\max_{\psi}f(\mathbf{x})g_{\psi}(T(\mathbf{x}))=\arg\max_{\psi}g_{\psi}(T) $$ 

T is sufficient for MLE procedure, because  $ \psi_{MLE} $  only requires T.

The sufficiency principle states that, if $T(\mathbf{x})$ is a sufficient statistic for the likelihood function $p(\psi|\mathbf{x})$, then any inference about $\psi$ should depend on $T(\mathbf{x})$ only.

## D EXPERIMENTAL DETAILS

### D.1 VAE ARCHITECTURE AND TRAINING

For the architecture and the training of our VAEs, we followed Xiao et al. (2020). In addition, we have trained VAEs of varying latent dimensions,  $ \{1, 2, 5, 10, 100, 1000, 2000, 3096, 5000, 10000\} $ , and instead of training for 200 epochs and taking the resulting model checkpoint, we took the checkpoint that had the best validation loss. For LPath-1M, we conducted experiments on VAEs with all latent dimensions and for LPath-2M, we paired one high-dimensional VAE from the group  $ \{3096, 5000, 10000\} $  and one low-dimensional VAE from the group  $ \{1, 2, 5\} $ .

In addition to Gaussian VAEs as mentioned in Section D.1.3, we also empirically experimented with a categorical decoder, in the sense the decoder output is between the discrete pixel ranges, as in Xiao et al. (2020). Strictly speaking, this no longer satisfies the Gaussian distribution anymore, which may in turn violate our sufficient statistics perspective. However, we still experimented with it to test whether LPath principles can be interpreted as a heuristic to inspire methods that approximate sufficient statistics that can work reasonably well, and we observed that categorical decoders work similarly with Gaussian decoders.

#### D.1.1 DIMENSIONALITY TRADE-OFF

In this section, we discuss heuristics for training VAEs in the context of OOD detection, focusing on the trade-offs involved in selecting the latent dimension.

Balancing the Trade-off in Latent Dimension A single VAE encounters a trade-off when selecting the latent dimension for effective OOD detection:

• Higher Latent Dimension Benefits the Encoder: Increasing the latent dimension enhances the encoder's ability  $ q_{\phi} $  to discriminate between in-distribution (ID) and OOD data. A higher-dimensional latent space allows the encoder to map ID and OOD data to more distinct regions, reducing overlap and improving separability. This increased capacity enables the encoder to capture complex features of the data, improving its discriminative power.

• Lower Latent Dimension Benefits the Decoder: Decreasing the latent dimension enhances the decoder's ability  $ p_{\theta} $  to identify OOD data through reconstruction errors. A lower-dimensional latent space constrains the decoder, making it less capable of accurately reconstructing OOD data that it hasn't seen during training. This constraint leads to larger reconstruction errors  $ u(\mathbf{x}) = \|\mathbf{x} - \widehat{\mathbf{x}}\|_2 $  for OOD samples, providing a useful signal for detection.

This trade-off poses a challenge: adjusting the latent dimension to favor one component (encoder or decoder) may compromise the performance of the other. Increasing the latent dimension benefits the encoder but may reduce the decoder's effectiveness in generating meaningful reconstruction errors. Conversely, decreasing the latent dimension enhances the decoder's ability to produce larger reconstruction errors for OOD data but may impair the encoder's discriminative capacity.

Implications for VAE Design When designing a single VAE for OOD detection, it's essential to consider this trade-off:

• For the Encoder: Aim for a higher latent dimension to improve the separation between ID and OOD data in the latent space.

• For the Decoder: Consider a lower latent dimension to increase reconstruction errors for OOD data, enhancing detection based on reconstruction discrepancies.

However, finding an optimal latent dimension that satisfies both requirements within a single VAE can be challenging. Adjusting the latent dimension to favor one aspect inherently affects the other, leading to suboptimal performance in at least one component.

Two VAEs Face No Such Trade-off To overcome this trade-off inherent in a single VAE, we propose using two VAEs with different latent dimensions, as discussed in the next section. By pairing a high-dimensional VAE with a low-dimensional one, we can leverage the strengths of both models without being constrained by the conflicting requirements of a single latent dimension.

#### D.1.2 Pairing VAEs: Leveraging Dual Latent Dimensions

Two VAEs Overcome the Trade-off To resolve the trade-off in latent dimension selection, we propose training two VAEs with different latent dimensions:

1. High-Dimensional VAE: This VAE has an overparameterized (large) latent dimension. Its encoder  $ q_{\phi} $  is capable of capturing complex features and provides informative statistics such as  $ v(\mathbf{x}) $  and  $ w(\mathbf{x}) $  that help discriminate between ID and OOD data.

2. Low-Dimensional VAE: This VAE has an underparameterized (small) latent dimension. Its decoder  $ p_{\theta} $  is constrained, leading to higher reconstruction errors  $ u(\mathbf{x}) $  for OOD data due to its limited capacity to represent unfamiliar inputs.

By combining the strengths of both VAEs, we can effectively detect OOD data. The high-dimensional VAE's encoder excels at distinguishing ID from OOD data in the latent space, while the low-dimensional VAE's decoder amplifies reconstruction errors for OOD samples.

Implementation Details In practice, we extract the following statistics:

• From the High-Dimensional VAE:

 $$ v(\mathbf{x})=\|\mu_{\mathbf{z}}(\mathbf{x})\|_{2}, $$ 

 $$ w(\mathbf{x})=\|\mathbf{\sigma  _{z}}(\mathbf{x})\|_{2}, $$ 

where  $ \mu_{\mathbf{z}}(\mathbf{x}) $  and  $ \sigma_{\mathbf{z}}(\mathbf{x}) $  are the encoder's mean and standard deviation in the latent space.

• From the Low-Dimensional VAE:

 $$ u(\mathbf{x})=\|\mathbf{x}-\widehat{\mathbf{x}}\|_{2}, $$ 

 $$ s(\mathbf{x})=\|\sigma_{\mathbf{x}}(\mu_{\mathbf{z}}(\mathbf{x}))\|_{2}, $$ 

where  $ \widehat{x} $  is the reconstructed input, and  $ \sigma_{\mathbf{x}}(\mu_{\mathbf{z}}(\mathbf{x})) $  is the decoder's standard deviation.

By integrating these statistics, we create a comprehensive feature set for OOD detection that leverages both the encoder's discriminative ability and the decoder's reconstruction error signal.

Empirical Results This approach has led to improvements in challenging OOD detection scenarios. For instance, when training on CIFAR-10 as the in-distribution dataset and using CIFAR-100, vertically flipped (VFlip), and horizontally flipped (HFlip) images as OOD datasets, our method achieved state-of-the-art results.

Remarkably, this was accomplished even though both VAEs, when considered individually, might have limitations:

• The Overparameterized VAE (high latent dimension) may overfit the training data, potentially reducing its generalization to unseen data.

• The Underparameterized VAE (low latent dimension) may struggle to reconstruct even some ID data accurately due to its limited capacity.

However, by combining their complementary strengths, we surpassed the performance of larger model architectures specifically designed for image data (see Table 1).

Pairing two VAEs with different latent dimensions allows us to capitalize on the advantages of both high and low-dimensional latent spaces without being constrained by the trade-offs inherent in a single model. This strategy provides a practical and effective solution for improving OOD detection performance, demonstrating that sometimes “it takes two to transcend.”

#### D.1.3 CONSTANT DECODER COVARIANCE

In typical VAE learning, the decoder's variance is fixed Dai et al., so it cannot be used as an inferential parameter. We initially treated the decoder as an isotropic Gaussian with a learnable scalar covariance matrix  $ \sigma_{\mathbf{x}}(\mathbf{z})^{2}I $ , where I is the identity matrix and  $ \sigma_{\mathbf{x}}(\mathbf{z})^{2} $  is a learnable scalar. We later observed that the scalar  $ \sigma_{\mathbf{x}}(\mathbf{z}) $  always converges to a small value and remains fixed for any ID or OOD data. And given that in typical VAE learning, the decoder's variance is fixed Dai et al.. We decided to use a fixed scalar as well and exclude this term from our algorithm.

This reduces the minimal sufficient statistics for encoder and decoder pair:

 $$ \left(\mu_{\mathbf{z}}(\mathbf{x}),\sigma_{\mathbf{z}}(\mathbf{x}),\mu_{\mathbf{x}}(\mathbf{z}),\sigma_{\mathbf{x}}(\mathbf{z})\right)\longrightarrow\left(\mu_{\mathbf{z}}(\mathbf{x}),\sigma_{\mathbf{z}}(\mathbf{x}),\mu_{\mathbf{x}}(\mathbf{z})\right) $$ 

#### D.1.4 TRAINING OBJECTIVE MODIFICATION FOR STRONGER CONCENTRATION

Inspired by the well known concentration of Gaussian probability measures, to encourage stronger concentration of the latent code around the spherical shell with radius  $ \sqrt{m} $  for better OOD detection, we propose the following modifications to standard VAEs' loss functions:

We replace the initial KL divergence by:

 $$ \mathcal{D}^{\mathrm{t y p i c a l}}[Q_{\phi}(\mathbf{z}\mid\mu_{\mathbf{z}}(\mathbf{x}),\sigma(\mathbf{x}))\|P(\mathbf{z})] $$ 

 $$ \begin{aligned}=&\mathcal{D}^{typical}[\mathcal{N}(\mu_{\mathbf{z}}(\mathbf{x}),\sigma_{\mathbf{z}}(\mathbf{x}))\|\mathcal{N}(0,I)]\end{aligned} $$ 

 $$ \begin{aligned}=&\frac{1}{2}\left(\mathrm{tr}(\sigma_{\mathbf{z}}(\mathbf{x}))+\left|(\mu_{\mathbf{z}}(\mathbf{x}))^{\top}(\mu_{\mathbf{z}}(\mathbf{x}))-m\right|-m-\log\det(\sigma_{\mathbf{z}}(\mathbf{x}))\right)\end{aligned} $$ 

where m is the latent dimension.

In training, we also use Maximum Mean Discrepancy (MMD) Gretton et al. (2012) as a discriminator since we are not dealing with complex distribution but Gaussian. The MMD is computed with Gaussian kernel. This extra modification is because the above magnitude regularization does not take distribution into account.

The final objective:

 $$ \begin{array}{r}{\mathbb{E}_{\mathbf{x}\sim P_{\mathrm{I D}}}\mathbb{E}_{\mathbf{z}\sim Q_{\phi}}\mathbb{E}_{\mathbf{n}\sim\mathcal{N}}[\log P_{\theta}(\mathbf{x}\mid\mathbf{z})]-\mathcal{D}^{\mathrm{t y p i c a l}}[Q_{\phi}(\mathbf{z}\mid\boldsymbol{\mu}_{\mathbf{z}}(\mathbf{x}),\sigma(\mathbf{x}))\|P(\mathbf{z})]-\operatorname{M M D}(\mathbf{n},\boldsymbol{\mu}_{\mathbf{z}}(\mathbf{x}))}\end{array} $$ 

The idea is that for  $ P_{ID} $ , we encourage the latent codes to concentrate around the prior's typical sets. That way,  $ P_{OOD} $  may deviate further from  $ P_{ID} $  in a controllable manner. In experiments, we tried the combinations of the metric regularizer,  $ D^{typical} $ , and the distribution regularizer, MMD. This leads to two other objectives:

 $$ \mathbb{E}_{\mathbf{x}\sim P_{\mathrm{I D}}}\mathbb{E}_{\mathbf{z}\sim Q_{\phi}}[\log P_{\theta}(\mathbf{x}\mid\mathbf{z})]-\mathcal{D}^{\mathrm{t y p i c a l}}[Q_{\phi}(\mathbf{z}\mid\boldsymbol{\mu}_{\mathbf{z}}(\mathbf{x}),\sigma(\mathbf{x}))\|P(\mathbf{z})] $$ 

 $$ \mathbb{E}_{\mathbf{x}\sim P_{\mathrm{I D}}}\mathbb{E}_{\mathbf{z}\sim Q_{\phi}}\mathbb{E}_{\mathbf{n}\sim\mathcal{N}}[\log P_{\theta}(\mathbf{x}\mid\mathbf{z})]-\mathcal{D}[Q_{\phi}(\mathbf{z}\mid\mu_{\mathbf{z}}(\mathbf{x}),\sigma(\mathbf{x}))\|P(\mathbf{z})]-\mathbf{M M D}(\mathbf{n},\mu_{\mathbf{z}}(\mathbf{x})) $$ 

where D is the standard KL divergence.

But we did not observe a significant difference in the final AUROC different variations. We still include those attempted modifications for future work.

### D.2 FEATURE PROCESSING TO BOOST COPOD PERFORMANCES

Like most statistical algorithms, COPOD/MD is not scale invariant, and may prefer more dependency structures closer to the linear ones. When we plot the distributions of  $ u(\mathbf{x}) $  and  $ v(\mathbf{x}) $ , we find that they exhibit extreme skewness. To make COPOD's statistical estimation easier, we process them by quantile transform. That is, for ID data, we map the tuple of statistics' marginal distributions to  $ \mathcal{N}(0,1) $ . To ease the low dimensional empirical copula, we also de-correlate the joint distribution of  $ (u(\mathbf{x}), v(\mathbf{x})), w(\mathbf{x})) $ . We do so using Kessy et al. (2018)'s de-correlation method, similar to Morningstar et al. (2021).

### D.3 WIDTH AND HEIGHT OF A VECTOR INSTEAD OF ITS  $ l^{2} $  NORM TO EXTRACT COMPLEMENTARY INFORMATION

In our visual inspection, we find that the distribution of the scalar components of  $ (u(\mathbf{x}), v(\mathbf{x}), w(\mathbf{x})) $  can be rather uneven. For example, the visible space reconstruction  $ x - \hat{x} $  error can be mostly low for many pixels, but very high at certain locations. These information can be washed away by the  $ l^{2} $  norm. Instead, we propose to track both  $ l^{p} $  norm and  $ l^{q} $  norm for small p and large q.

For small p,  $ l^{p} $  measures the width of a vector, while  $ l^{q} $  measures the height of a vector for big q. To get a sense of how they capture complementary information, we can borrow intuition from  $ l^{p} \approx l^{0} $ , for small p and  $ l^{q} \approx l^{\infty} $ , for large q.  $ \|x\|_{0} $  counts the number of nonzero entries, while  $ \|x\|_{\infty} $  measures the height of x. For x with continuous values, however,  $ l^{0} $  norm is not useful because it always returns the dimension of x, while  $ l^{\infty} $  norm just measures the maximum component.

Extreme measures help screen extreme data. We therefore use  $ l^{p} $  norm and  $ l^{q} $  norm as a continuous relaxation to capture this idea:  $ l^{p} $  norm will “count” the number of components in x that are unusually small, and  $ l^{q} $  norm “measures” the average height of the few biggest components. These can be more discriminative against OOD than  $ l^{2} $  norm alone, due to the extreme (proxy for OOD) conditions they measure. We observe some minor improvements, detailed in Table 2’s ablation study.


<table border=1 style='margin: auto; width: max-content;'><tr><td colspan="2">ID: CIFAR10</td><td colspan="3">OOD</td></tr><tr><td style='text-align: center;'>OOD Dataset</td><td style='text-align: center;'>SVHN</td><td style='text-align: center;'>CIFAR100</td><td style='text-align: center;'>Hflip</td><td style='text-align: center;'>Vflip</td></tr><tr><td style='text-align: center;'>$ l^{2} $  norm</td><td style='text-align: center;'>0.96</td><td style='text-align: center;'>0.60</td><td style='text-align: center;'>0.53</td><td style='text-align: center;'>0.61</td></tr><tr><td style='text-align: center;'>$ (l^{p}, l^{q}) $</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.62</td><td style='text-align: center;'>0.53</td><td style='text-align: center;'>0.61</td></tr></table>

<div style="text-align: center;">Table 2: Comparing the AUC of  $ l^{2} $  norm versus our  $ (l^{p}, l^{q}) $  measures.</div>


## E ABLATION STUDIES

### E.1 INDIVIDUAL STATISTICS

To empirically validate how  $ (u(\mathbf{x}), v(\mathbf{x}), w(\mathbf{x})) $  complement each other, we use individual component alone in first stage and fit the second stage COPOD as usual. We notice significant drops in performances. We fit COPOD on individual statistics  $ u(\mathbf{x}) $ ,  $ v(\mathbf{x}) $ ,  $ w(\mathbf{x}) $  and show the results in Table 3. We can see that our original combination in Table 1 is better overall.

### E.2 MD

To test the efficacy of  $ (u(\mathbf{x}), v(\mathbf{x}), w(\mathbf{x})) $  without COPOD, we replace COPOD by a popular algorithm in OOD detection, the MD algorithm Lee et al. (2018) and report such scores in Table 1. The scores are comparable to COPOD, suggesting  $ (u(\mathbf{x}), v(\mathbf{x}), w(\mathbf{x})) $  is the primary contributor to our performances.

### E.3 LATENT DIMENSIONS

One hypothesis on the relationship between latent code dimension and OOD detection performance is that lowering dimension incentivizes high level semantics learning, and higher level feature learning


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="4">OOD Dataset</td></tr><tr><td style='text-align: center;'>Statistic</td><td style='text-align: center;'>SVHN</td><td style='text-align: center;'>CIFAR100</td><td style='text-align: center;'>Hflip</td><td style='text-align: center;'>Vflip</td></tr><tr><td style='text-align: center;'>$ u(\mathbf{x}) $</td><td style='text-align: center;'>0.96</td><td style='text-align: center;'>0.59</td><td style='text-align: center;'>0.54</td><td style='text-align: center;'>0.59</td></tr><tr><td style='text-align: center;'>$ v(\mathbf{x}) $</td><td style='text-align: center;'>0.94</td><td style='text-align: center;'>0.56</td><td style='text-align: center;'>0.54</td><td style='text-align: center;'>0.59</td></tr><tr><td style='text-align: center;'>$ w(\mathbf{x}) $</td><td style='text-align: center;'>0.93</td><td style='text-align: center;'>0.58</td><td style='text-align: center;'>0.54</td><td style='text-align: center;'>0.61</td></tr><tr><td style='text-align: center;'>$ v(\mathbf{x}) \&amp; w(\mathbf{x}) $</td><td style='text-align: center;'>0.94</td><td style='text-align: center;'>0.58</td><td style='text-align: center;'>0.54</td><td style='text-align: center;'>0.60</td></tr><tr><td style='text-align: center;'>$ u(\mathbf{x}) \&amp; v(\mathbf{x}) $</td><td style='text-align: center;'>0.97</td><td style='text-align: center;'>0.61</td><td style='text-align: center;'>0.53</td><td style='text-align: center;'>0.61</td></tr><tr><td style='text-align: center;'>$ u(\mathbf{x}) \&amp; w(\mathbf{x}) $</td><td style='text-align: center;'>0.98</td><td style='text-align: center;'>0.61</td><td style='text-align: center;'>0.54</td><td style='text-align: center;'>0.61</td></tr></table>

<div style="text-align: center;">Table 3: COPOD on individual statistics. ID dataset is CIFAR10.</div>


<div style="text-align: center;">can help discriminate OOD v.s. ID. We conducted experiments on the below latent dimensions and report their AUC based on  $ v(\mathbf{x}) $  (norm of the latent code) in Table 4</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Latent dimension</td><td style='text-align: center;'>1</td><td style='text-align: center;'>2</td><td style='text-align: center;'>5</td><td style='text-align: center;'>10</td><td style='text-align: center;'>100</td><td style='text-align: center;'>1000</td><td style='text-align: center;'>3096</td><td style='text-align: center;'>5000</td></tr><tr><td style='text-align: center;'>$ v(\mathbf{x}) $  AUC</td><td style='text-align: center;'>0.39</td><td style='text-align: center;'>0.63</td><td style='text-align: center;'>0.52</td><td style='text-align: center;'>0.45</td><td style='text-align: center;'>0.22</td><td style='text-align: center;'>0.65</td><td style='text-align: center;'>0.76</td><td style='text-align: center;'>0.59</td></tr></table>

<div style="text-align: center;">Table 4: Lower latent code dimension doesn’t help to discriminate in practice. Clearly, lowering the dimension isn’t sufficient to increase OOD performances.</div>