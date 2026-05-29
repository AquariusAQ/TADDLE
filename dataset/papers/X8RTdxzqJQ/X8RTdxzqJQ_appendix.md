## A ALGORITHM

We present the complete algorithm for SSL-C2ST in Algorithm 1.

Algorithm 1 paradigm of testing with SSL-C2ST

Input:  $ S_{P} $ ,  $ S_{Q} $ , S, significance level  $ \alpha $ , latent feature vector size H, an autoencoder  $ f_{\alpha} $  consist of a featurizer  $ \phi $  and a decoder  $ \phi^{-1} $  parameterized by  $ \theta_{\phi} $ , a binary classifier g concatenated after featurizer parameterized by  $ \theta_{g} $ , SSL-C2ST model  $ f = g \circ \phi $ , learning rate  $ \eta_{\phi} $ ,  $ \eta_{g} $ , MSE loss function  $ L_{MSE} $ , CE loss function  $ L_{CE} $ , total epoch for representation learning  $ T_{rl} $ , total epoch for training classifier  $ T_{cl} $ .

1: Derive the unlabelled data  $ S_{\text{unl}} = \text{shuffle}(S_{P} \cup S_{Q}) $ 

# Phase 1: train the Featurizer  $ \phi $  from  $ f_{\alpha} $  on  $ S_{\text{unl}} $ 

for  $ t = 1, 2, \ldots, T_{rl} $  do

2:  $ X_{t} \leftarrow \text{minibatch from } S_{\text{unl}} $ ;

3:  $ X_{t}' \leftarrow f_{\alpha}(X_{t}) $ ;

4:  $ \theta_{\phi} \leftarrow \theta_{\phi} - \eta_{\phi} \nabla_{\text{Adam}} \mathcal{L}_{\text{MSE}}(X_{t}, X_{t}') $  based on Eq. (3);

end for

# Phase 2: train a Classifier f (consist of  $ \phi $  and g) on  $ S^{\text{tr}} = (S_{P}^{\text{tr}}, \mathbf{0}) \cup (S_{Q}^{\text{tr}}, \mathbf{1}) $ 

for  $ t = 1, 2, \ldots, T_{cl} $  do

5:  $ (X_{t}, l_{t}) \leftarrow \text{minibatch from } S^{\text{tr}} $ ;

6:  $ \hat{l}_{t} \leftarrow g \circ \phi(X_{t}) $ ;

7:  $ \theta_{g} \leftarrow \theta_{g} - \eta_{g} \nabla_{\text{Adam}} \mathcal{L}_{\text{CE}}(\hat{l}_{t}, l_{t}) $  based on Eq. (4);

end for

# Phase 3: permutation test with f on  $ S^{\text{te}} = S_{P}^{\text{te}} \cup S_{Q}^{\text{te}} $ 

8:  $ est \leftarrow \hat{t}(S_{P}^{\text{te}}, S_{Q}^{\text{te}}, f) $  based on Eq. (1);

for  $ i = 1, 2, \ldots, n_{perm} $  do

9: Shuffle  $ S^{\text{te}} $  into X and Y;

10:  $ perm_{i} \leftarrow \hat{t}_{\text{M}}(X, Y; f) $ 

end for

Output:  $ \mathbb{I}\left[\frac{1}{n_{perm}}\sum_{i=1}^{n_{perm}}\mathbb{I}(est < perm_{i}) \leq \alpha\right] $ 

## B OVERVIEW OF MAJOR CATEGORIES OF SOTA SSL METHODS

Building on the SSL assumptions, we will recap how contemporary SOTA SSL methods incorporate these principles and assumptions, setting the stage for an analysis of their applicability to the specific challenges presented by our problem setting.

Transductive vs Inductive learning. Classification tasks within machine learning can typically be categorized within two distinct problem settings: transductive and inductive learning (Chapelle et al., 2006). Transductive learning is concerned with predicting the labels of the specific unlabeled data that was present during the training process, emphasizing a tailored fit to this data. Inductive learning, on the other hand, focuses on the generalization of the learned classifier to new, unseen data. In learnable two-sample testing, the goal is to test whether the given two samples are drawn from same distributions. To make it, we firstly split samples into labelled set and unlabelled set, then find out that whether it is possible to learn a classifier that can distinguish two samples from the mixed unlabelled set. It becomes apparent that applying SSL methodologies to the two-sample testing problem inherently requires a transductive learning approach. This conceptual groundwork necessitates a detailed examination of current SSL methods to identify their foundational assumptions and evaluate their performance in two-sample test scenarios.

Major categories. Currently, we identify that there are five main categories of SOTA SSL methods: consistency regularisation, pseudo-labelling, graph-based, generative models and hybrid (often a combination of consistency regularisation and pseudo-labelling) (Yang et al., 2023). We will succinctly explicate how they work, and how they are applied for our downstream two-sample testing tasks in the experiments of various levels of HDGM.

• Consistency Regularisation: Based on the manifold assumption or the smoothness assumption, the consistency regularisation methods apply consistency constraints to the final loss function, where the intuition is that if the data follows the smoothness assumption or manifold assumption, even though we construct some perturbations in the inputs, it will not influence the output of classification (Xie et al., 2020).

• Pseudo-Labelling: Pseudo-labelling uses its own predictions to generate labels for unlabeled data, which are then used to further train the model. It relies on the assumptions that model's high-confidence predictions are accurate. This assumption is based on the cluster assumption for the validity and efficacy of propagating labels to unlabelled data based on model predictions (Lee et al., 2013).

• Graph-Based: Graph-based methods will construct a similarity graph based on the raw dataset, where each node represents a data instance, and weighted-edge represents the similarity between two data instances. Based on the smoothness assumption, the label information can be propagated from labelled nodes to unlabelled nodes, if two nodes are closely connected in the constructed graph (Song et al., 2021b).

• Generative Models: Generative methods learn to model the underlying distribution of both labelled data and unlabelled data, using this learned representation to generate new data points and infer missing labels. Based on the manifold assumption, the generative models aim to learn the underlying low-dimensional manifold and generate data points that adhere to the same manifold, used for further model training (Kingma & Welling, 2013).

• Hybrid: Hybrid methods are just combination of multiple methods, such as consistency regularisation, pseudo-labelling, and sometimes generative approaches. These models typically rely on the smoothness assumption and cluster assumption, in order to infer the labels of unlabelled data (Sohn et al., 2020).

## C EXPERIMENTAL DETAILS

### C.1 OVERVIEW OF DATASETS

High-Dimensional Gaussian mixtures. The high dimensional Gaussian mixtures (HDGM) benchmark is a synthetic dataset that is composed of multiple Gaussian distributions, each representing a cluster, which is proposed by Liu et al. (2020). In our experiments, we are considering bimodal Gaussian mixtures, which means the number of clusters remains 2 irrelevant to the dimension of the multivariate Gaussian distributions. In Section 4, we consider there are three levels of HDGM, which are HDGM-Easy, HDGM-Medium and HDGM-Hard in order to specify that most SOTA SSL methods are not suitable for two-sample testing problems. In other places rather than Section 4, we regard HDGM as HDGM-Hard. Under  $ H_{0} $ , P and Q are the same, which is denoted as HDGM-S; and under  $ H_{1} $ , we slightly modify a mild covariance  $ \pm0.5 $  between first two dimensions in the covariance matrix of Q and other setups are the same as HDGM-S, which is referred to as HDGM-D. Thus, HDGM-S and HDGM-D are both noted by hard-level HDGM. The details of how to synthesize P and Q to derive HDGM-Easy, HDGM-Medium, HDGM-Hard, HDGM-S and HDGM-D are described in Table 3. We regard n as the number of samples drawn from each cluster in each distribution and N as the number of total samples drawn from both P and Q, where  $ N = n \times c \times 2 $ . We conduct two experiments on HDGM-D, increasing the N from N = 1000 to N = 10000 when keeping the dimension d remain the same. One experiment is a low-dimensional HDGM-D with d = 2 and another is a high-dimensional HDGM-D with d = 10. Moreover, we conduct both low-dimensional and high-dimensional HDGM-S to show that the type-I error is controlled. The result is shown in Figure 5, which will be analyzed in the section 6.

MNIST vs MNIST-Fake. The MNIST datasets is a collection of 70,000 grayscale images of handwritten digits, ranging from 0 to 9, divided into a training set of 60,000 images and a test of 10,000 images (LeCun et al., 1998). The MNIST-Fake is the set of 10,000 images generated by a pretrained deep convolutional generative adversarial network (DCGAN) (Radford et al., 2016). The MNIST benchmark (MNIST vs MNIST-Fake) is also proposed by Liu et al. (2020), aiming to test the performance of testing methods in the image space. Under  $ H_{0} $ , we draw samples both from the MNIST-Fake. Under  $ H_{1} $ , we compare the samples from real MNIST, P, and samples from MNIST-Fake, Q. We regard N as the number of samples each drawn from P and Q, where we

increase N from N = 200 to N = 1000. The result of the average test power of all methods is displayed in the Table 2. All methods are tested with a reasonable type-I error rate.

ImageNet vs ImageNet-Fake. The ImageNet dataset is a comprehensive collection of over 14 million labeled high-resolution images belonging to roughly 22,000 categories (Deng et al., 2009). The ImageNet-Fake dataset comprises 10,000 high-quality images generated using the advanced StyleGAN-XL model, a state-of-the-art generative adversarial network designed for large and diverse datasets (Sauer et al., 2022). This benchmark (ImageNet vs ImageNet-Fake) extends the framework established by Liu et al. (2020) to a more complex and diverse image domain, testing the robustness of two-sample testing methods at a larger scale. Under the null hypothesis  $ H_{0} $ , samples are drawn from ImageNet-Fake, while under the alternative hypothesis  $ H_{1} $ , we compare samples from the real ImageNet dataset, P, with those from ImageNet-Fake, Q. We vary the number of samples drawn from each, P and Q, from N = 200 to N = 1000 to examine the scalability of the test methods. The outcomes in terms of average test power across various methodologies are summarized in Table 2, with all tests maintaining a reasonable type-I error rate.

### C.2 IMPLEMENTATION DETAILS OF C2ST AND SSL-C2ST

• C2ST: a C2ST uses statistic in Eq. (1) to measure the difference of two samples. Rather than 3 phases described in the Algorithm 1, C2ST-based methods is purely supervised with only 2 phases. Implementation of C2ST paradigm is to only take Phase 2 and Phase 3 from Algorithm 1. Most of the implementation details are referenced from Lopez-Paz & Oquab (2018a) and Liu et al. (2020). The splitting portion of training and testing is always half to half, and the model architecture is the same for C2ST and SSL-C2ST, where first few layers are feature extractor and followed by a classification layer. Moreover, in the first step of Phase 3, we do not utilize the softmax probability of the first value of the logits returned by the classifier to calculate the statistic of two samples, we apply Eq. (1) which directly derive the mean of the classification prediction accuracy of two samples.

• SSL-C2ST: a SSL version of C2ST. Most of the implementation details are described in the Algorithm 1, except we replace the way of calculating a statistic from Eq. (2) to Eq. (1).

In C2ST, we have a classifier f consisting of a randomly initialized feature extractor  $ \phi_{\theta}(x) $  followed by a logistic regression layer with parameters w and b, where

 $$ f(x)=\phi_{\theta}(x)\times\boldsymbol{w}+\boldsymbol{b}. $$ 

As the f is a binary classifier,  $  f(x) = [z_{0}, z_{1}]  $  and  $ \text{softmax}(f(x)) = [p_{0}, p_{1}] $ , where  $ p_{0} + p_{1} = 1 $ . All parameters  $ \theta $ , w and b are updated through the supervised learning on the training set, which aims to minimize the occurrence of incorrect predictions. Then, use the empirical probability of the correct predictions on an unseen testing set to measure the difference between two samples.

However, in SSL-C2ST, we have g consisting of a feature extractor  $ \phi_{a}(x) $  trained on  $ S_{P}^{tr} \cup S_{P}^{te} \cup S_{Q}^{tr} \cup S_{Q}^{te} $  without labels via unsupervised learning and a logistic regression layer for subsequent supervised training purpose. In the unsupervised learning step, we use  $ \phi_{a}(x) $  to extract a latent feature vector z from the input x, and then use a decoder  $ \phi_{a}^{-1}(x) $  to reconstruct z to a reconstructed  $ x' $ . We update the parameters of  $ \phi_{a} $  by minimizing the difference between the reconstructed input  $ x' $  and the original input x. After the unsupervised training procedure, we add a classification layer after  $ \phi_{a} $  to form a classifier g, and train the classification layer in the same way as the C2ST.

### C.3 DETAILS OF SSL-C2ST-M AND OTHER MMD BASED METHODS

We first introduce SSL-C2ST-M and compare the following state-of-the-art testing methods on two benchmark datasets:

• SSL-C2ST-M: An advanced SSL-C2ST-based method. Rather than using the prediction labels (0 or 1) to measure the test accuracy, we utilize MMD to calculate the differences between output features extracted from the SSL-C2ST. The output features could be the output of the hidden layer or the logits output of the classifier trained by the SSL-C2ST, as we discuss in Section 5.1.

• C2ST-M: a C2ST-based method that is the same as C2ST, except it uses the statistic in Eq. (2) to measure the absolute mean differences between the probability of the logits of two samples, as we discuss in the Section 2. In Liu et al. (2020), this method is also called C2ST-L where L refers to logit.

- MMD-D: MMD with a deep kernel (Liu et al., 2020); a state-of-the-art testing method in the supervised paradigm. MMD-D learns a deep kernel by directly maximizing the test power of MMD, leading to an increase in test power on the testing set.

- MMD-FUSE: a state-of-the-art testing method in the unsupervised paradigm. It fuses several MMD statistics based on the simple kernel of different combinations of hyperparameters into a new powerful statistic, then conducts a permutation test based on the fused statistic (Biggs et al., 2023).

#### C.3.1 IMPLEMENTATION DETAILS OF SSL-C2ST-M

In the implementation of SSL-C2ST-M, the classifier is trained with the same way as how we do in the SSL-C2ST. However, SSL-C2ST-M is more flexible in the procedures of testing. For datasets whose input vector size is small in SSL-C2ST, such as HDGM, we use the absolute value of differences between the mean of  $ p_{0} $  of samples from P and that of samples from Q. It measures the mean probability that samples will be classified label 0 by SSL-C2ST. For image datasets that have large input vector size, such as MNIST, we use the hidden-layer output of the classifier trained by the SSL-C2ST, whose input vector size is 100, to compute the MMD between the features extracted from two samples. For high-dimensional image datasets, the latent vector with a larger size can contain more useful information to measure the difference between two extracted features.

### C.4 DETAILS OF HDGM DATASETS

Table 3 displays the details of how HDGM datasets are generated (Liu et al., 2020). Different levels of HDGM datasets are first proposed in this paper, in order to show why SOTA SSL methods cannot be directly applied in the two-sample testing problem. The level of HDGM is differed from whether the data points are highly overlapping or whether the clusters within the same distribution are isolated. For the HDGM-Easy,  $ \Delta_{\mu}=10 $  and  $ \Delta_{q}=5 $ . For the HDGM-Medium,  $ \Delta_{\mu}=10 $  and  $ \Delta_{q}=0 $ . For the HDGM-Hard,  $ \Delta_{\mu}=0.5 $  and  $ \Delta_{q}=0 $ .

Table 3: Details of how to synthesize P and Q in the experiments. Let c = 2 be the number of the clusters in each distribution, d > 2 be the dimension of multivariate normal distribution of each cluster.  $ (\boldsymbol{\mu}_{1}, \ldots, \boldsymbol{\mu}_{c}) $  is a set of d-dimensional mean vector  $ \mu_{i} $  that specifies that mean of each dimension in the distribution, where  $ \mu_{1} = 0_{d} $ ,  $ \mu_{i} = \mu_{i-1} + \Delta_{\mu} \times 1_{d} $ .  $ I_{d} $  is the  $ d \times d $  identity matrix,  $ \Delta_{\mu} $  is the cluster mean difference within the same distribution, and  $ \Delta_{q} $  is the mean difference between

 $$ \Delta_{1}=0.5,\Delta_{2}=-0.5 $$ 

 $$ \mathbf{\Sigma}_{i}=\begin{pmatrix}1&\Delta_{i}&\mathbf{0}_{d-2}\\\Delta_{i}&1&\mathbf{0}_{d-2}\\\mathbf{0}_{d-2}^{T}&\mathbf{0}_{d-2}^{T}&I_{d-2}\end{pmatrix} $$ 


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Datasets</td><td style='text-align: center;'>$ \mathbb{P} $</td><td style='text-align: center;'>$ \mathbb{Q} $</td></tr><tr><td style='text-align: center;'>HDGM-S</td><td style='text-align: center;'>$ \sum_{i=1}^{c}\mathcal{N}(\boldsymbol{\mu}_{i},I_{d}) $</td><td style='text-align: center;'>$ \sum_{i=1}^{c}\mathcal{N}(\boldsymbol{\mu}_{i},I_{d}) $</td></tr><tr><td style='text-align: center;'>HDGM-D</td><td style='text-align: center;'>$ \sum_{i=1}^{c}\mathcal{N}(\boldsymbol{\mu}_{i},I_{d}) $</td><td style='text-align: center;'>$ \sum_{i=1}^{c}\mathcal{N}(\boldsymbol{\mu}_{i}+\Delta_{q},\boldsymbol{\Sigma}_{i}) $</td></tr></table>

### C.5 DETAILS OF COMPUTING RESOURCES

The experiments of the work are conducted on three platforms. One platform is a Nvidia-4090 GPU PC with Pytorch framework. The second platform is a High-performance Computer cluster with lots of Nvidia-A100 GPU with Pytorch framework. The last platform is a Nvidia-4090 GPU Window Subsystem for Linux with Jax framework. The memory of three platforms are all over 16 GB. The storage of disk of three platforms are all over 512 GB.

### C.6 A DISCUSSION ABOUT SUPERVISED SEQUENTIAL TWO-SAMPLE TESTING AND APPLICABILITY OF SSL-C2ST

Supervised sequential two-sample testing represents another approach to utilizing testing data (Pandeva et al., 2022). In this framework, a classifier is trained to determine whether two samples from a single batch originate from the same distribution. Initially, batches are split and fed sequentially into the classifier as testing data. Batches that do not reject the null hypothesis are concatenated with previous batches and used as training data for the classifier, continuing until all batches are exhausted or a single batch rejects the null hypothesis. The sequential nature of the test emerges from the use of e-values, which are updated as more data becomes available, allowing for a dynamic assessment of the testing hypothesis. However, this method should not be directly compared to our method due to different problem settings and designs. Firstly, in sequential two-sample testing, data are split into several batches and tests are conducted on single, small batches. Conversely, in other supervised two-sample testing approaches, data are only split into two halves, creating a trade-off between the number of training and testing samples.

Furthermore, the design of our SSL-C2ST method is compatible with any other supervised two-sample testing framework, including sequential two-sample testing. As long as a proportion of data is used for testing, we can remove the labels from this testing data and concatenate it into the training data. This allows us to learn IRs through representation learning, followed by the original supervised two-sample testing framework.

### C.7 EXPERIMENT RESULT OF SEQUENTIAL TWO-SAMPLE TESTING

In this part, we will display the result of supervised sequential two-sample test that proposed by Pandeva et al. (2022) on the HDGM-Hard dataset, and compared the result with original C2ST and SSL-C2ST in our problem setting. We can find that even though this method can have a small increase on the test power over the original C2ST method, but have a large decrease to our method. The number of batches we choose is five, if we choose the number of batches to two, it is exactly similar as C2ST; if we choose the number of batches to a large number like ten, the test power will drop down, since the test data size will be too small. Thus, we decide five as the number of batches, and C2ST-Sequential(5) in the Table 4 represent the supervised sequential two-sample testing with the number of batches equal to five.

<div style="text-align: center;">Table 4: Experiment results of test power of sequential two-sample testing with Batch5 over original C2ST and our propose SSL-C2ST on HDGM-hard dataset. N is the total size of two samples inputed in 100 trials.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'>N=4000</td><td style='text-align: center;'>N=6000</td><td style='text-align: center;'>N=8000</td><td style='text-align: center;'>Avg.</td></tr><tr><td style='text-align: center;'>C2ST-Sequential (5)</td><td style='text-align: center;'>0.32</td><td style='text-align: center;'>0.57</td><td style='text-align: center;'>0.79</td><td style='text-align: center;'>0.56</td></tr><tr><td style='text-align: center;'>C2ST</td><td style='text-align: center;'>0.29</td><td style='text-align: center;'>0.49</td><td style='text-align: center;'>0.78</td><td style='text-align: center;'>0.52</td></tr><tr><td style='text-align: center;'>SSL-C2ST</td><td style='text-align: center;'>0.50</td><td style='text-align: center;'>0.81</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.77</td></tr></table>

### C.8 FUTURE WORK

Autoencoder is the basic representation learning algorithm we introduce to enhance our SSL-C2ST, we can also replace it to more advanced representation algorithms, such as semi-supervised variational autoencoder (VAE) (Kingma et al., 2014),  $ \beta $ -VAE (Higgins et al., 2016), or other autoencoder-based representation learning algorithms (Tschannen et al., 2018b).

### C.9 REPRODUCIBILITY

All the reproducible code can be found in the anonymous link.

## D THEORETICAL ANALYSIS

#### D.1 PROOF OF THEOREM 5.1

Proof. Let  $ f' \in C_{\phi}: X \to \{0,1\} $  be the SSL-C2ST classifier model which has the same model architecture as C2ST. Recall from Eq. (1), the accuracy of  $ f' $  on the testing set  $ S_{te} $  is

 $$ \hat{t}=\frac{1}{n_{\mathrm{t e}}}\sum_{(z_{k},l_{k})\in\mathcal{S}_{\mathrm{t e}}}\mathbb{I}\left[f^{\prime}(z_{k})=l_{k}\right], $$ 

where  $ n_{te} = |S_{te}| $ , then we have that

 $$ \Pr\left(\mathbb{I}\left[f^{\prime}\left(z\right)=l\right]=\tau\right)=\begin{cases}p&if\tau=1,\\1-p&if\tau=0.\end{cases} $$ 

Lemma D.1. Under null hypothesis  $ H_{0}: P = Q $ , samples  $ S_{P} $  and  $ S_{Q} $  follows the same distribution, so  $ n_{te}\hat{t} $  is the sum of identically distributed Bernoulli random variables with a probability of random-guessing  $ p_{H_{0}} = \frac{1}{2} $ , which follows a Binomial( $ n_{te}, p_{H_{0}} $ ). For a large  $ n_{te} $  and using the central limit theorem,  $ \hat{t} $  will converge to a  $ \mathcal{N}\left(\frac{1}{2}, \frac{1}{4n_{te}}\right) $ .

Lemma D.2. Under  $ H_{1}: P \neq Q $ ,  $ n_{te}\hat{t} $  is the sum of Bernoulli random variables that may not be identically distributed. In that way,  $ n_{te}\hat{t} $  follows a Poisson Binomial distribution, which can be approximated by a Binomial  $ (n_{\mathrm{te}}\bar{p}, n_{\mathrm{te}}\bar{p}(1-\bar{p})) $ , where  $ \bar{p} = n^{-1} \sum_{k=1}^{m+n} p_{k} $  (Ehm, 1991). For a large  $ n_{te} $ , a central limit theorem holds that  $ \hat{t} $  will converge to a  $ \mathcal{N}\left(\bar{p}, \frac{\bar{p}(1-\bar{p})}{n_{\mathrm{te}}}\right) $ . Let  $ \bar{p} = 1 - \epsilon(\mathbb{P}, \mathbb{Q}; f') $ , where  $ \epsilon(\mathbb{P}, \mathbb{Q}; f') \in \left(0, \frac{1}{2}\right) $  represent the inability of  $ f' $  on distinguishing between P and Q, then  $ \hat{t} \sim \mathcal{N}\left(1 - \epsilon, n_{\mathrm{te}}^{-1} (\epsilon - \epsilon^2)\right) $ .

Thus, the Type-II error is defined as the probability of failing to reject  $ H_{0} $ , while  $ H_{1} $  is actually true. This occurs when the test statistic  $ \hat{t} $ , which follows the distribution under  $ H_{1} $ , does not exceed the critical threshold determined by the null distribution  $ H_{0} $  at a specified significance level  $ \alpha $ . According to Lemma D.1, the threshold value  $ t_{\alpha} $  can be calculated as

 $$ t_{\alpha}=\mu+z_{\alpha}\times\sigma=\frac{1}{2}+\Phi^{-1}(1-\alpha)\times\frac{1}{\sqrt{4n_{\mathrm{te}}}}, $$ 

combined with Lemma D.2, so the Type-II error is

 $$ \begin{align*}\beta=\mathrm{Pr}_{T\sim\mathcal{N}\left(1-\epsilon,n_{\mathrm{te}}^{-1}(\epsilon-\epsilon^{2})\right)}\left(T<t_{\alpha}\right)&=\mathrm{Pr}_{T\sim\mathcal{N}\left(1-\epsilon,n_{\mathrm{te}}^{-1}(\epsilon-\epsilon^{2})\right)}\left(T<\frac{1}{2}+\frac{\Phi^{-1}(1-\alpha)}{\sqrt{4n_{\mathrm{te}}}}\right)\\&=\mathrm{Pr}_{T^{\prime}\sim\mathcal{N}\left(0,n_{\mathrm{te}}^{-1}(\epsilon-\epsilon^{2})\right)}\left(T^{\prime}<\frac{\Phi^{-1}(1-\alpha)}{\sqrt{4n_{\mathrm{te}}}}+\epsilon-\frac{1}{2}\right)\\&=\mathrm{Pr}_{Z\sim\mathcal{N}(0,1)}\left(Z<\sqrt{\frac{n_{\mathrm{te}}}{\epsilon-\epsilon^{2}}}\left(\frac{\Phi^{-1}(1-\alpha)}{\sqrt{4n_{\mathrm{te}}}}+\epsilon-\frac{1}{2}\right)\right)\\&=\Phi\left(\sqrt{\frac{n_{\mathrm{te}}}{\epsilon-\epsilon^{2}}}\left(\frac{\Phi^{-1}(1-\alpha)}{\sqrt{4n_{\mathrm{te}}}}+\epsilon-\frac{1}{2}\right)\right)\\&=\Phi\left(\frac{\Phi^{-1}(1-\alpha)/2+\left(\epsilon-\frac{1}{2}\right)\sqrt{n_{\mathrm{te}}}}{\sqrt{\epsilon-\epsilon^{2}}}\right).\end{align*} $$ 

Thus, the test power is

 $$ \pi(\alpha,n_{\mathrm{te}},\epsilon)=1-\beta=1-\Phi\left(\frac{\Phi^{-1}(1-\alpha)/2+\left(\epsilon-\frac{1}{2}\right)\sqrt{n_{\mathrm{te}}}}{\sqrt{\epsilon-\epsilon^{2}}}\right)=\Phi\left(\frac{\left(\frac{1}{2}-\epsilon\right)\sqrt{n_{\mathrm{te}}}-\Phi^{-1}(1-\alpha)/2}{\sqrt{\epsilon-\epsilon^{2}}}\right). $$ 

As we know  $ \Phi^{-1}(1-\alpha)/2 $  is a constant, for a reasonably fixed large  $ n_{te} $ , if we are trying to maximizing the test power, we are actually maximizing the first term of numerator, which is

 $$ \mathcal{J}(\mathbb{P},\mathbb{Q};f^{\prime})=\max_{\epsilon}\frac{\left(\frac{1}{2}-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})\right)}{\sqrt{\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})^{2}}},\text{where}\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})\in\left(0,\frac{1}{2}\right) $$ 

This is equivalent to

 $$ \begin{aligned}\mathcal{J}(\mathbb{P},\mathbb{Q};f^{\prime})&=\min_{\epsilon}\frac{\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})}{\sqrt{\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})^{2}}}\\&=\min_{\epsilon}\frac{\sqrt{\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})}}{\sqrt{1-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})}}\\&=\min_{\epsilon}\frac{\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})}{1-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})}\end{aligned} $$ 

The proof of equivalence can be found at the end of the proof. Since  $ \epsilon(\mathbb{P},\mathbb{Q};f^{\prime})\in\left(0,\frac{1}{2}\right) $ , it is clear to see that directly minimizing the  $ \epsilon(\mathbb{P},\mathbb{Q};f^{\prime}) $  will optimize the objectives of maximizing the test power.

Moreover, we will show that the Type-I error is also controlled, which is the probability of reject  $ H_{0} $ , while  $ H_{0} $  is true:

 $$ \begin{align*}\Pr_{T\sim\mathcal{N}\left(\frac{1}{2},\frac{1}{4n_{\mathrm{te}}}\right)}\left(T>t_{\alpha}\right)&=Pr_{T\sim\mathcal{N}\left(\frac{1}{2},(4n_{\mathrm{te}})^{-1}\right)}\left(T>\frac{1}{2}+\frac{\Phi^{-1}(1-\alpha)}{\sqrt{4n_{\mathrm{te}}}}\right)\\&=\Pr_{T^{\prime}\sim\mathcal{N}(0,(4n_{\mathrm{te}})^{-1})}\left(T^{\prime}>\frac{\Phi^{-1}(1-\alpha)}{\sqrt{4n_{\mathrm{te}}}}\right)\\&=\Pr_{Z\sim\mathcal{N}(0,1)}\left(Z>\Phi^{-1}(1-\alpha)\right)\\&=1-\Pr_{Z\sim\mathcal{N}(0,1)}\left(Z<\Phi^{-1}(1-\alpha)\right)\\&=1-\Phi\left(\Phi^{-1}(1-\alpha)\right)\\&=\alpha,\end{align*} $$ 

## Proof of equivalence

If we define  $ f(\epsilon) = \frac{1/2 - \epsilon}{\sqrt{\epsilon - \epsilon^{2}}} $ ,  $ g(\epsilon) = \frac{\epsilon}{\sqrt{\epsilon - \epsilon^{2}}} $  and  $ D(\epsilon) = \sqrt{\epsilon - \epsilon^{2}} $ , where  $ \epsilon \in (0, \frac{1}{2}) $ . The equation  $ \max_{\epsilon} f(\epsilon) = \max_{\epsilon} \left( \frac{1/2}{D(\epsilon)} - g(\epsilon) \right) $  holds. It is clear to find that  $ f(\epsilon) $  and  $ \frac{1/2}{D(\epsilon)} $  are monotonically decreasing over the domain of  $ \epsilon $ . Thus, only if  $ g(\epsilon) $  is monotonically increasing over the domain of  $ \epsilon $ , the equation  $ \max_{\epsilon} \left( \frac{1/2}{D(\epsilon)} - g(\epsilon) \right) = \max_{\epsilon} \left( \frac{1/2}{D(\epsilon)} \right) - \min_{\epsilon} \left( g(\epsilon) \right) $  holds. Firstly, let us calculate the derivative of  $ D(\epsilon) = \left( \epsilon - \epsilon^{2} \right)^{1/2} $  w.r.t  $ \epsilon $ .

 $$ \begin{aligned}D^{\prime}(\epsilon)&=\frac{1}{2(\epsilon-\epsilon^{2})^{1/2}}\cdot(1-\epsilon+(-\epsilon))\\&=\frac{1-2\epsilon}{2D(\epsilon)},\end{aligned} $$ 

then, we take the derivative of  $ g(\epsilon)=\frac{\epsilon}{D(\epsilon)} $  w.r.t  $ \epsilon $ ,

 $$ \begin{aligned}g^{\prime}(\epsilon)&=\frac{1\cdot D(\epsilon)-\epsilon\cdot D^{\prime}(\epsilon)}{D(\epsilon)^{2}}=\frac{1}{D(\epsilon)^{2}}\cdot\frac{2D(\epsilon)^{2}-\epsilon(1-2\epsilon)}{2D(\epsilon)}\\&=\frac{1}{\epsilon(1-\epsilon)}\cdot\frac{2(\epsilon-\epsilon^{2})-\epsilon(1-2\epsilon)}{2D(\epsilon)}\\&=\frac{\epsilon}{\epsilon(1-\epsilon)\cdot2\sqrt{\epsilon(1-\epsilon)}}=\frac{1}{2(1-\epsilon)\sqrt{\epsilon(1-\epsilon)}}.\end{aligned} $$ 

We can find that over the domain of  $ \epsilon\in(0,\frac{1}{2}) $ ,  $ g'(\epsilon)>0 $ , which concludes the proof. The reason why deriving the objective to be equivalent to  $ \min_{\epsilon}\frac{\epsilon}{\sqrt{\epsilon-\epsilon^{2}}} $  is we can simplify it to  $ \min_{\epsilon}\sqrt{\frac{\epsilon}{(1-\epsilon)}}=\min_{\epsilon}\frac{\epsilon}{(1-\epsilon)} $ , where  $ \epsilon\in(0,\frac{1}{2}) $ . In that way, it is quite straightforward to understand how minimizing  $ \epsilon $  can help to improve test power.

#### D.2 Proof of Theorem 5.3

Let  $ \epsilon(\mathbb{P},\mathbb{Q};f)\in\left(0,\frac{1}{2}\right) $  be the inability of f to distinguish between distribution P and Q. Then we define the  $ \operatorname{err}_{\operatorname{te}}(f)=2\epsilon(\mathbb{P},\mathbb{Q};f)\in(0,1) $  to be the error rate of f on distribution P and Q.

Theorem D.3. (Boucheron et al., 2000) Suppose function space $\mathcal{C}:\{f|f:\mathcal{X}\to\{0,1\}\}$ has finite VC-dimension for $V\geq1$. For any sample $S$, any function $f$, we have

 $$ \Pr\left[\sup_{f\in\mathcal{C}}|err_{\mathrm{te}}(f)-\widehat{err}_{\mathrm{te}}(f)|\geq\Delta\right]\leq8\mathcal{C}[2m_{\mathrm{l}},\mathcal{S}]e^{-m\Delta^{2}/8}. $$ 

So for any  $ \Delta,\delta>0 $ , if we draw from S a sample satisfying

 $$ m_{\mathrm{l}}\geq\frac{8}{\Delta}\left(\ln(\mathcal{C}[m_{\mathrm{l}},\mathcal{S}])+\ln\left(\frac{8}{\delta}\right)\right), $$ 

then, with probability at least  $ 1 - \delta $ , all functions f satisfy  $ \left| \operatorname{err}_{\operatorname{te}}(f) - \widehat{\operatorname{err}}_{\operatorname{te}}(f) \right| \leq \Delta $ .

Proof. The given unlabelled sample size implies that with probability  $ 1 - \delta/2 $ , all  $ f' \in C $  have

 $$ |\widehat{e r r}_{\mathrm{u n l}}(\phi_{f^{\prime}})-e r r_{\mathrm{u n l}}(\phi_{f^{\prime}})|\leq\sqrt{\frac{\ln\left(\frac{4s}{\delta}\right)}{2m_{\mathrm{u}}}}\leq\Delta, $$ 

which also implies that

 $$ \widehat{err}_{\mathrm{unl}}(\phi_{f^{\prime}*})\leq err_{\mathrm{unl}}(\phi_{f^{\prime}})+\sqrt{\frac{\ln\left(\frac{4s}{\delta}\right)}{2m_{\mathrm{u}}}}\leq\xi+\sqrt{\frac{\ln\left(\frac{4s}{\delta}\right)}{2m_{\mathrm{u}}}}\leq\xi+\Delta. $$ 

Using the standard VC bounds (e.g., Theorem D.3), the labelled sample size  $ m_{1} $  implies that with probability at least  $ 1 - \delta/4 $ , all  $ f' \in \mathcal{C}_{\phi, \mathcal{S}, \chi}(\xi + 2\Delta) $  have  $ |\text{err}_{\text{te}}(f) - \widehat{\text{err}}_{\text{te}}(f)| \leq \Delta $ . Then, by Hoeffding bounds, with probability at least  $ 1 - \delta/4 $  we have

 $$ \widehat{err}_{\mathrm{te}}(f^{\prime*})\leq err_{\mathrm{te}}(f^{\prime*})+\sqrt{\log(4/\delta)/2m_{\mathrm{l}}}\leq err_{\mathrm{te}}(f^{\prime*})+\Delta. $$ 

Therefore, with probability at least  $ 1 - \delta $ , the  $ f' \in C_{\phi} $  that optimizes  $ \widehat{err}_{\mathrm{te}}(f') $  subject to  $ \widehat{err}_{\mathrm{unl}}(\phi_{f'}) \leq \xi + \Delta $  has

 $$ \widehat{err}_{\mathrm{te}}(f^{\prime})\leq err_{\mathrm{te}}(f^{\prime*})+\sqrt{\frac{\ln\left(\frac{4s}{\delta}\right)}{2m_{\mathrm{u}}}}+\sqrt{\log(4/\delta)/2m_{\mathrm{l}}}\leq err_{\mathrm{te}}(f^{\prime*})+\sqrt{\frac{\ln\left(\frac{4s}{\delta}\right)}{2m_{\mathrm{u}}}}+\Delta. $$ 

Moreover, since we have  $ \widehat{err}_{\mathrm{te}}(f')=\operatorname{Pr}_{(z_i,l_i)\sim\mathcal{S}}\left[f'(z_i)\neq l_i\right]\in(0,1) $  which is proportional to the empirical inability  $ \hat{\epsilon}(S_P,S_Q;f')\in\left(0,\frac{1}{2}\right) $ . Thus, we can conclude the following inequality

 $$ 2\hat{\epsilon}(S_{P},S_{Q};f^{\prime})\leq e r r_{\mathrm{t e}}(f^{\prime*})+\Delta+\sqrt{\frac{\ln\left(\frac{4s}{\delta}\right)}{2m_{\mathrm{u}}}}, $$ 

since errte(f1*) = 2ε(P, Q; f1*)

 $$ \hat{\epsilon}(S_{P},S_{Q};f^{\prime})\leq\epsilon(\mathbb{P},\mathbb{Q};f^{\prime*})+\frac{\Delta}{2}+\sqrt{\frac{\ln\left(\frac{4s}{\delta}\right)}{8m_{\mathrm{u}}}}, $$ 

which concludes the proof.