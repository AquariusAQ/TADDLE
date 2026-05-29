# REVISIT NON-PARAMETERIC TWO-SAMPLE TESTING AS A SEMI-SUPERVISED LEARNING PROBLEM

Anonymous authors

Paper under double-blind review

## ABSTRACT

Learning effective data representations is crucial in answering if two samples X and Y are from the same distribution (a.k.a. the non-parametric two-sample testing problem), which can be categorized into: i) learning discriminative representations (DRs) that distinguish between two samples in a supervised-learning paradigm, and ii) learning inherent representations (IRs) focusing on data's inherent features in an unsupervised-learning paradigm. However, both paradigms have issues: learning DRs reduces the data points available for the two-sample testing phase, and learning purely IRs misses discriminative cues. To mitigate both issues, we propose a novel perspective to consider non-parametric two-sample testing as a semi-supervised learning (SSL) problem, introducing the SSL-based Classifier Two-Sample Test (SSL-C2ST) framework. While a straightforward implementation of SSL-C2ST might directly use existing state-of-the-art (SOTA) SSL methods to train a classifier with labeled data (with sample indexes X or Y) and unlabeled data (the remaining ones in the two samples), conventional two-sample testing data often exhibits substantial overlap between samples and violates SSL methods' assumptions, resulting in low test power. Therefore, we propose a two-step approach: first, learn IRs using all data, then fine-tune IRs with only labelled data to learn DRs, which can both utilize information from whole dataset and adapt the discriminative power to the given data. Extensive experiments and theoretical analysis demonstrate that SSL-C2ST outperforms traditional C2ST by effectively leveraging unlabeled data. We also offer a stronger empirically designed test achieving the SOTA performance in many two-sample testing datasets.

## 1 INTRODUCTION

Two-sample tests aim to solve the problem of “Whether two samples are drawn from the same distribution?”. Classical two-sample tests, including t-tests which test the empirical mean differences between two samples, often need to assume that samples are drawn from specific distributions (e.g., Gaussian distributions with the same variance). To alleviate the strict assumptions, non-parametric two-sample tests are proposed to solve the problem only based on observed data (Gretton et al., 2012a; B; Heller, 2016; Székely & Rizzo, 2013; Jitkrittum et al., 2016; Chen & Friedman, 2017; Ghoshdastidar et al., 2017; Lopez-Paz & Oquab, 2018b; Ramdas et al., 2017; Sutherland et al., 2017; Gao et al., 2018; Ghoshdastidar & von Luxemburg, 2018; Lerasle et al., 2019; Liu et al., 2020; Kirchler et al., 2020; Kübler et al., 2020; Cheng & Xie, 2021; Kübler et al., 2022; Kübler et al., 2022; Liu et al., 2021; Deka & Sutherland, 2023; Bonnier et al., 2023).

For example, the Kolmogorov-Smirnov (K-S) test is designed to compare the cumulative distribution functions derived from two samples, but it can only be effective in extremely low-dimensional data (Kolmogorov, 1933; Smirnov, 1948). The maximum mean discrepancy (MMD) test adopts the kernel mean embedding of distribution and uses it to measure the discrepancy between two distributions Gretton et al. (2012a). The statistics used in these non-parametric two-sample tests are also widely adopted in many other fields, such as domain adaptation, causal discovery, generative modeling, adversarial learning, and more (Gong et al., 2016; Bińkowski et al., 2018; Stojanov et al., 2019; Cano & Krawczyk, 2020; Oneto et al., 2020; Gao et al., 2021; Fang et al., 2021b; Zhong et al., 2021; Fang et al., 2021a; Song et al., 2021a; Tahmasbi et al., 2021; Taskesen et al., 2021; Bergamin et al., 2022).

To improve the test power of non-parametric two-sample tests in practical applications, recent studies have shown that learning good data representations is crucial before performing two-sample testing (Kirchler et al., 2020; Liu et al., 2020; 2021; Gao et al., 2021; Bergamin et al., 2022). For example, Kirchler et al. (2020) directly use a pre-trained feature extractor to extract features of two samples and find it is useful to increase the test power during the testing. Meanwhile, Liu et al. (2020) propose a unified learning paradigm to learn deep-net representations of data via maximizing the test power of MMD and show that the learned representations can help capture the difference between two samples. Recently, Biggs et al. (2023) point out that, after discarding the sample information (namely, we do not know which sample the data belongs to), learning representations from whole samples will not influence the type I error of permutation-based testing methods, which further justifies the correctness of learning good representations for testing.

## Two learning paradigms and their

issues. There are two main data representation learning paradigms in the two-sample testing field: 1) the supervised paradigm; and 2) the unsupervised paradigm (see Figure 1). In paradigm 1), we first split samples into training and testing sets, then learn a representation extractor to obtain two samples' discriminative representations (DRs) (Sutherland et al., 2017; Lopez-Paz & Oquab, 2018b; Liu et al., 2020; 2021). In paradigm 2), we can learn a representation extractor based on data from the whole samples after discarding the sample information (Biggs et al., 2023). For example, unsupervised learning can be used to learn inherent representations (IRs) of samples (Biggs et al., 2023).

Figure 1: Visualisation of two learning paradigms. Blue color represents data with sample index X, red color represents data with sample index Y, and transparent represents data without sample index information. The square represents original input samples, the circle represents the inherent representations (IRs) learned from unsupervised model, and the triangle represents the discriminative representations (DRs) learned from supervised model.

Although the supervised paradigm has obtained success in many fields (Gao et al., 2021; Bergamin et al., 2022), we have to use part of samples to train a good classifier (Lopez-Paz & Oquab, 2018b) or a kernel function (Liu et al., 2020), which will cause fewer samples are used in the final testing procedure. Namely, the procedure of splitting samples into training and testing sets will naturally lower the test power. There has to be a trade-off between the extra power provided by the learned functions/kernels and the sacrificed power due to the sample-splitting procedure. For example, Biggs et al. (2023) recently reveal that combining several pre-defined kernels on the whole samples can provide higher test power compared to deep-kernel MMD test (Liu et al., 2020) on some datasets, indicating that, in some cases, the sacrificed power might be higher than the extra power provided by learned functions or kernels.





<div style="text-align: center;"><img src="imgs/img_in_image_box_530_445_1007_663.jpg" alt="Image" width="38%" /></div>


In the unsupervised paradigm, researchers try to develop testing methods that do not need the data-splitting procedure. To avoid sacrificing power from the data-splitting procedure, Kübler et al. (2020) propose a new testing method based on the linear-time estimator of MMD and the selective inference framework. Because Kübler et al. (2020) use a linear-time estimator of MMD, there is a test-power reduction compared to the U-statistic or V-statistic of MMD. Then, Schrab et al. (2023) and Biggs et al. (2023) propose new ways to combine several kernels in a given candidate set and perform the two-sample testing directly on the whole samples. Empirical experiments support that their newly proposed statistic, MMD-FUSE, enjoy even higher test power than the most effective method in the first paradigm given a good candidate set. However, there is still an open question in this paradigm: can we always expect a relatively good kernel in the candidate set for any given two samples?

Motivation. Based on the development of the two-sample testing methods reviewed above, it can be seen that both the supervised paradigm and unsupervised paradigm have their own issues. For the supervised paradigm, we have to use a relatively large amount of data to ensure that we can learn a good function or kernel, resulting in a possibly higher sacrificed power. For the unsupervised paradigm, a good candidate set is key but we do not have supervision to find such a candidate set.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_165_474_381.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">(a) HDGM-Easy</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_476_165_740_381.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">(b) HDGM-Medium</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_747_165_1005_380.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">(c) HDGM-Hard</div>


<div style="text-align: center;">Figure 2: Visualisation of first two dimensions of samples for different levels of the HDGM dataset whose dimension is 10. For the HDGM-Easy and HDGM-Medium, the cluster mean difference  $ \Delta_{\mu} $  within the same distribution is 10, while for the HDGM-Hard,  $ \Delta_{\mu} $  is 0.5. For the HDGM-Easy, the distribution mean difference  $ \Delta_{q} $  between P and Q is 5, while for HDGM-Medium and HDGM-Hard,  $ \Delta_{q} $  is 0. Other setting of how to generate HDGM dataset is described in Appendix C.4</div>


Thus, to obtain a better two-sample testing method, we might aim to reduce the requirement of a large training set in the supervised paradigm or to provide supervision to find a good kernel in the unsupervised paradigm. The first aim is quite similar to the advantage that semi-supervised learning (SSL) can bring in classification: Given many unlabeled data, SSL methods can help us obtain a good classifier even the training set is small (Balcan & Blum, 2010).

Our contributions. In this paper, we revisit the non-parametric two-sample testing as an SSL problem and propose an SSL-based classifier two-sample test (SSL-C2ST). SSL-C2ST extends the state-of-the-art (SOTA) two-sample testing method C2ST by incorporating SSL methods.

In our experiments, we firstly implemented several SOTA SSL methods, depending on different SSL frameworks, such as consistency regularization (CR) (Xie et al., 2020), pseudo labeling (PL) (Lee et al., 2013), generative models (GM) (Kingma & Welling, 2013), and hybrid methods (HB) (Sohn et al., 2020), within the C2ST framework. However, the result was not satisfactory (see Table 1), because two-sample testing fundamentally differs from typical classification tasks; it is a problem of distinguishing between two distributions (or saying two samples) rather than an instance-level classification. Furthermore, the high degree of overlap between the two samples in testing dataset challenges the basic assumptions of these SSL methods, such as HDGM in Figure 2b and Figure 2c, where two distributions are largely overlapped, violating the assumptions of many SSL methods.

This violation of assumptions leads SSL methods to have low test power in two-sample testing. Our method, SSL-C2ST, is implemented in two phases. At first, we learn the IRs from whole dataset in an unsupervised autoencoder-based representation learning (Tschannen et al., 2018a). Following this, we apply the C2ST framework, not by training an randomly initialized classifier, but by fine-tuning the pre-trained encoder with an added classification layer in order to learn the DRs.

We provide the first theoretical analysis to show that, with a high probability, involving a larger testing set (without sample information) in the training process will lead to a higher lower bound of the test power of SSL-C2ST, verifying the effectiveness of SSL-C2ST in theory. Besides, the empirical test power on three benchmark datasets also shows that SSL-C2ST clearly outperforms C2ST.

On the empirical side, we are also interested in the data representations extracted by the trained classifier in SSL-C2ST. We perform MMD tests (with a linear kernel) on the different-level data representations of the testing set, called SSL-C2ST-M. These tests clearly outperform the corresponding baselines empirically. Notably, SSL-C2ST-M outperforms the MMD with the deep kernel (MMD-D (Liu et al., 2020)) and MMD-FUSE (Biggs et al., 2023) on the MNIST and ImageNet dataset.

Impact of our study in the field. The success of SSL-C2ST-(M) might provide evidence that SSL-based testing methods have the potential to overcome the key issues of both paradigms. For the supervised paradigm, SSL-based testing methods can leverage the useful information in the testing set (without sample information), thus we can expect either to use a smaller training set (sacrificing less power) or to learn a better function/kernel (more extra power) with the help of the useful information in the testing set. For the unsupervised paradigm, SSL-based testing methods might provide some supervision to guide the learning process of unsupervised learning or to form a better candidate set that contains a function/kernel that can help distinguish between two samples better.

<div style="text-align: center;">Table 1: Result of C2ST test power on HDGM-Easy, HDGM-Medium and HDGM-Hard (d=10), on different total size of two samples N inputed in 100 trials. Compared to other application of SOTA SSL methods on C2ST, where C2ST-CR, C2ST-PL, C2ST-GM, and C2ST-HB represent that we learn the classifier of C2ST using consistency-regularisation, pseudo-labelling, generative-model, and hybrid SSL frameworks, respectively, and SSL-C2ST is our method. $ ^{2} $ </div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td colspan="3">HDGM-Easy</td><td colspan="3">HDGM-Medium</td><td colspan="3">HDGM-Hard</td></tr><tr><td style='text-align: center;'>N=60</td><td style='text-align: center;'>N=80</td><td style='text-align: center;'>N=100</td><td style='text-align: center;'>N=2000</td><td style='text-align: center;'>N=3000</td><td style='text-align: center;'>N=4000</td><td style='text-align: center;'>N=4000</td><td style='text-align: center;'>N=6000</td><td style='text-align: center;'>N=8000</td></tr><tr><td style='text-align: center;'>C2ST</td><td style='text-align: center;'>0.64</td><td style='text-align: center;'>0.91</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.44</td><td style='text-align: center;'>0.82</td><td style='text-align: center;'>0.97</td><td style='text-align: center;'>0.29</td><td style='text-align: center;'>0.49</td><td style='text-align: center;'>0.78</td></tr><tr><td style='text-align: center;'>C2ST-CR</td><td style='text-align: center;'>0.65</td><td style='text-align: center;'>0.92</td><td style='text-align: center;'>1.00</td><td style='text-align: center;'>0.40</td><td style='text-align: center;'>0.84</td><td style='text-align: center;'>0.97</td><td style='text-align: center;'>0.32</td><td style='text-align: center;'>0.42</td><td style='text-align: center;'>0.75</td></tr><tr><td style='text-align: center;'>C2ST-PL</td><td style='text-align: center;'>0.72</td><td style='text-align: center;'>0.96</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.40</td><td style='text-align: center;'>0.76</td><td style='text-align: center;'>0.93</td><td style='text-align: center;'>0.36</td><td style='text-align: center;'>0.45</td><td style='text-align: center;'>0.77</td></tr><tr><td style='text-align: center;'>C2ST-GM</td><td style='text-align: center;'>0.64</td><td style='text-align: center;'>0.92</td><td style='text-align: center;'>1.00</td><td style='text-align: center;'>0.43</td><td style='text-align: center;'>0.85</td><td style='text-align: center;'>0.97</td><td style='text-align: center;'>0.22</td><td style='text-align: center;'>0.40</td><td style='text-align: center;'>0.72</td></tr><tr><td style='text-align: center;'>C2ST-HB</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>1.00</td><td style='text-align: center;'>1.00</td><td style='text-align: center;'>0.25</td><td style='text-align: center;'>0.43</td><td style='text-align: center;'>0.58</td><td style='text-align: center;'>0.28</td><td style='text-align: center;'>0.43</td><td style='text-align: center;'>0.65</td></tr><tr><td style='text-align: center;'>SSL-C2ST</td><td style='text-align: center;'>0.97</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>1.00</td><td style='text-align: center;'>0.58</td><td style='text-align: center;'>0.97</td><td style='text-align: center;'>1.00</td><td style='text-align: center;'>0.50</td><td style='text-align: center;'>0.81</td><td style='text-align: center;'>0.99</td></tr></table>

## 2 PRELIMINARY

Two-sample Test. Two-sample test is one of the statistical hypothesis tests that aims to assess whether two independent identically distributed i.i.d. samples, denoted by  $ S_{P} = \{x_{i}\}_{i=1}^{n} \sim P^{n} $  and  $ S_{Q} = \{y_{j}\}_{j=1}^{m} \sim Q^{m} $ , where  $ x_{i}, y_{j} \in X $ , are drawn from the same distribution (Lehmann & Romano, 2005). In two-sample testing, the null hypothesis  $ H_{0} $  refers to two samples sourcing from the same distribution, which stands for P = Q. The alternative hypothesis  $ H_{1} $  indicates that two samples are drawn from different distributions, meaning  $ P \neq Q $ . Whether we should accept or reject  $ H_{0} $  depends on the test statistic  $ \hat{t} $ , which represents the differences between two samples.

Classifier Two-sample Test (C2ST). The idea of C2ST is to use a supervised classification algorithm to distinguish between the two samples. If the classifier performs significantly better than random guessing, it suggests that two samples come from different distributions (Lopez-Paz & Oquab, 2018b):

Problem Setting. In our problem setting, we assume the total number of two samples are fixed and given, and we are trying to distinguish whether these two given samples are from the same distribution or not. No more extra data is provided for testing data and the test data is known, so it can be regarded as a transductive learning problem. Thus, the C2ST is conducted in the following steps:

Firstly, construct the dataset  $ \mathcal{S}=\{(x_{i},0)|x_{i}\in S_{P}\}_{i=1}^{n}\cup\{(y_{j},1)\in S_{Q}\}_{j=1}^{m}:=\{(z_{k},l_{k})\}_{k=1}^{m+n}\sim\mathcal{D} $ , where m=n; then shuffle and split S into  $ S_{tr} $  and  $ S_{te} $ , where  $ S=S_{tr}\cup S_{te} $ . Let  $ f^{*}:X\to\{0,1\} $  be a binary classifier that is trained on  $ S_{tr} $  from a concept class C and  $ p_{k}=p(l_{k}=1|z_{k}) $ , where

 $$ f^{*}=\underset{f\in\mathcal{C}}{\arg\min}\sum_{\left(z_{k},l_{k}\right)\in\mathcal{S}}-\left[l_{k}\log p_{k}+\left(1-l_{k}\right)\log\left(1-p_{k}\right)\right], $$ 

and  $ f^{*}(z_{k}) $  be the estimate of the conditional probability distribution  $ \mathbb{I}\left(p(l_{k}=1|z_{k})>\frac{1}{2}\right) $ , the statistic or the accuracy of the classifier  $ f^{*} $  on  $ S_{te} $  can be written as:

 $$ \hat{t}=\frac{1}{n_{\mathrm{t e}}}\sum_{(z_{k},l_{k})\in\mathcal{S}_{\mathrm{t e}}}\mathbb{I}\left[f^{*}(z_{k})=l_{k}\right], $$ 

where  $ n_{te} = |S_{te}| $  and I is the indicator function. Finally, we compute the p-value to determine if the test statistic is significantly greater than the random guessing accuracy, utilizing the approximate null distribution of C2ST outlined in Appendix D.1 and the permutation test discussed next.

Testing with  $ \hat{t} $ . According to the standard central limit theorem (Serfling, 2009), the test statistic  $ \hat{t} $  in Eq. (1) converges to normal distributions under both the null or alternative hypothesis. Although it is viable for us to derive the threshold  $ t_{\alpha} $  of the null hypothesis distribution and perform a traditional Z-Test, it is simpler and faster to instead implement a permutation test (Sutherland et al., 2017). We will permute and randomly assign samples to new  $ S_{P}^{te'} $  and  $ S_{Q}^{te'} $  for n times. Under  $ H_{0} $ , the samples from P and Q should be interchangeable, implying that the test statistic  $ \hat{t} $  should exhibit minimal variation between its value based on the original sequence of samples and its computation from several randomly permuted sequences. Thus, if the original test statistic is large enough than most of the statistic derived from the randomly permuted sequences, we can conclude that we reject  $ H_{0} $ .

C2ST-based MMD (C2ST-M). Moreover, we can also consider using a trained classifier in C2ST to extract representations of two samples, and then regard representations of two samples as the new two samples. For these new two samples, we can use MMD (with a linear kernel) to compute the difference between two samples. Let  $ S_{P}^{te} $  and  $ S_{Q}^{te} $  be the splitting samples of  $ S_{P} $  and  $ S_{Q} $  in the testing set  $ S_{te} $  and  $ n_{x}^{te} $  and  $ n_{y}^{te} $  be the sample size of  $ S_{P}^{te} $  and  $ S_{Q}^{te} $ . In general, the statistic used in C2ST-M is

 $$ \hat{t}_{M}=\left\|\frac{1}{n_{x}^{\mathrm{te}}}\sum_{x_{i}\in S_{P}^{\mathrm{te}}}h(x_{i})-\frac{1}{n_{y}^{\mathrm{te}}}\sum_{y_{i}\in S_{Q}^{\mathrm{te}}}h(y_{i})\right\|_{2}^{2}, $$ 

where h is the feature extractor (could be the model's output, i.e., logit), or the model's hidden-layer output, and  $ \|\cdot\|_{2} $  is the L2 norm. When h is logits, C2ST-M is known as C2ST-L in (Liu et al., 2020).

## 3 REVISIT NON-PARAMETER TWO-SAMPLE TEST AS A SEMI-SUPERVISED LEARNING PROBLEM

This section presents two research questions that we will address in the paper. As both existing two-sample testing paradigms have their own limitations, our first research question comes out

### 3.1 IS IT POSSIBLE TO BOTH ELIMINATE THE SIDE-EFFECT OF DATA SPLITTING AND OBTAIN THE HIGH DISCRIMINATIVE POWER?

Except the supervised paradigm and unsupervised paradigm, the semi-supervised one is another well-known paradigm. According to the definition of SSL, SSL can leverage the information  $ P(x) $  from unlabeled data to help the inference of  $ P(y|x) $  (Chapelle et al., 2006). If the unlabeled data degrades prediction accuracy by misguiding the inference (e.g., due to violating the assumptions of SSL techniques), then that cannot be classified as effective SSL method. As we attempt to utilize the information from the unlabeled testing data to increase the test power of the supervised two-sample testing methods, SSL techniques seem to be reliable to solve that research question. However, since we are the first to frame two-sample testing as a SSL problem, we have to be responsible to evaluate whether current SSL techniques can be directly applied on the supervised two-sample testing methods.

### 3.2 CAN SOTA SSL TECHNIQUES BE SUCCESSFULLY APPLIED ON SUPERVISED TWO-SAMPLE TESTING METHODS?

This question is worthy to investigate, since in the definition of SSL, the consequence of failure in applying SSL techniques is highlighted, which can lead to a worse performance than the original supervised method. Thus, we will firstly conduct motivation experiments to directly apply the main SOTA SSL techniques on the SOTA supervised two-sample testing method C2ST to examine the fitness of SSL assumptions on the two-sample testing data. If it fails, we will propose a viable method that can utilize the information from the unlabeled testing data, which can pave the way for the further advanced techniques to be applied.

## 4 CAN WE DIRECTLY APPLY SSL METHODS IN TWO-SAMPLE TESTING?

In this section, we will discuss the key assumptions of traditional SSL methods. Then, we will analyze whether we can directly apply those methods in our two-sample testing scenarios.

Assumptions of SSL methods. In principle, incorporating unsupervised information from unlabeled data has the potential to enhance the predictions made by purely supervised learning models. However, the efficacy of SSL is often relied on some assumptions below (Chapelle et al., 2006).

• Smoothness assumption: If points  $ x_{1} $  and  $ x_{2} $  are close, then so should be their labels  $ y_{1} $ ,  $ y_{2} $ .

• Cluster assumption: If points are in the same cluster, they are likely to be of the same class.

• Manifold assumption: The (high-dimensional) data lie (roughly) on a low-dimensional manifold.



Based on those assumptions, there are five representative SSL frameworks (Yang et al., 2023): consistency-regularisation (Xie et al., 2020), pseudo-labelling (Lee et al., 2013), graph-based (Song et al., 2021b), generative-models (Kingma & Welling, 2013) and hybrid (Sohn et al., 2020) SSL methods. The details of SSL methods are demonstrated in Appendix B.

Testing data might not satisfy the assumptions made by many SSL methods. In the traditional two-sample testing problem settings, there are normally overlapping between two samples. As we can see in Figure 2b and Figure 2c, for the HDGM-Medium and HDGM-Hard datasets, there are high-overlapping areas between two distributions. This will highly violate the first two assumptions of SSL mentioned above. For the smoothness assumption, our dataset will have exactly the same data point in two samples, but allocated with different labels, this will notably influence the SSL methods that are based on such assumptions. For cluster assumptions, we can see in HDGM-Medium, although there are two obvious clusters, they are not the same labels within the same cluster.

Empirical result for validity of SOTA SSL methods on two-sample testing. The empirical results, presented in Table 1 $ ^{3} $ , show that the application of SOTA SSL methods on C2ST not only underperforms our proposed method but also often yields poorer results compared to the original C2ST on HDGM-Medium and HDGM-Hard datasets, which are the common overlapping distribution data in the context of two-sample testing. This underperformance can be attributed to the fundamental nature of the two-sample testing problem, which is distinct from typical classification tasks. In two-sample testing, the two input samples should not inherently possess class labels. During training, we manually assign labels to facilitate distinction by the classifier, whereas in testing, we consider the two samples holistically rather than focusing on individual instance accuracy. Furthermore, standard SSL methods, which primarily enhance classification through data augmentation based on smoothness assumptions or infer pseudo labels based on clustering assumptions, aim to generate high-confidence training data. However, in two-sample testing, these approaches are flawed; data augmentation may alter the samples' distributions, and pseudo label inference often proves inaccurate. These discrepancies lead to the ineffectiveness of these SSL methods in two-sample testing contexts. Therefore, we propose a two-sample test through a two-phase SSL approach, shown below.

## 5 How to Utilize Unlabelled Data Increasing Test Power?

In this section, we introduce the structure design and the algorithm of our SSL-C2ST, and then we offer theoretical analysis to validate the effectiveness of our method.

### 5.1 OUR PROPOSAL: SSL-C2ST

As the two-sample testing problem violates the native assumptions of SOTA SSL methods, we propose a pipeline that follows the definition of SSL, which utilizes the unlabelled samples and labelled samples in two phases. The first phase is an unsupervised auto-encoder-based (AE-based) representation learning, which learns a feature extractor that captures the inherent features for both samples. The next phase is the same as the C2ST pipeline, where the feature encoder in the model is not randomly initialized, but extracted from the representation learning in the previous phase. The ablated part of this approach compared to the C2ST is the AE-based representation learning, so its effectiveness will be aligned with AE-based representation learning, which relies on the manifold assumption, and that is particularly well-suited for two-sample testing scenarios. The paradigm of SSL-C2ST is shown in Figure 3, consisting of three steps: 1) learning IRs; 2) learning DRs; and 3) performing two-sample testing.

Since SSL-C2ST has the same classifier architecture as C2ST but with different training objectives, we need to decompose the classifier model f into two parts: a feature extractor  $ \phi \in F: X \to R^{k} $  that used to learn IRs and followed by a classifier  $ g \in G: R^{k} \to \{0, 1\} $  that used to learn DRs. We denote by  $ \phi_{f} $  and  $ g_{f} $  the feature extractor and the classifier of a specified model f. Then, let  $ f' \in C_{\phi}: X \to \{0, 1\} $  be the SSL-C2ST classifier model, where  $ C_{\phi} = \{f'| f' = g \circ \phi, g \in G\} \subseteq C $  and  $ C = \bigcup_{\phi \in F} C_{\phi} $ . Given two available samples  $ S_{P} $  and  $ S_{Q} $ , and construct a dataset S referred to the problem setting in Section 2.

<div style="text-align: center;"><img src="imgs/img_in_image_box_262_162_966_476.jpg" alt="Image" width="57%" /></div>


<div style="text-align: center;">Figure 3: Overview of the SSL-C2ST paradigm compared to original C2ST paradigm. Firstly, an encoder was learned from unsupervised AutoEncoder-Based representation learning on whole data. Secondly, fine-tune the learned encoder followed by the supervised learning in C2ST. At last, perform the permutation test based on the model and the output statistics.</div>


Learning IRs. The first step is to train a representation learning model on the whole unlabelled dataset  $ S_{unl} $ , using the mean squared error (MSE) as the loss function to compare the differences between input and reconstructed output. Specifically, we aim to learn a function  $ \phi^{*} $  such that

 $$ \phi^{*}=\arg\min_{\phi}\frac{1}{\left|\mathcal{S}_{\mathrm{unl}}\right|}\sum_{z_{i}\sim\mathcal{S}_{\mathrm{unl}}}\left\|\psi(\phi(z_{i}))-z_{i}\right\|_{2}^{2}, $$ 

where  $ \psi:R^{k}\toX $  is the decoder, and  $ \phi^{*}(z_{i}) $  is called the IR of  $ z_{i} $ 

Learning DRs. Then, utilize the featurizer from the representation learning model and concatenate with a classification layer to form a binary classifier model. The combined model is fine-tuned on  $ S_{tr} $ , applying a cross-entropy (CE) loss function to compare the output of SSL-C2ST with the label of samples. Specifically, we aim to learn a function  $ g^{*} $  such that  $ \operatorname*{Pr}_{(z_{i},l_{i})\sim\mathcal{S}}\left[g\circ\phi^{*}(z_{i})\neq l_{i}\right] $  can be minimized on S, which can be implemented by the following surrogate objective.

 $$ g^{*}=\arg\min_{g}\mathcal{L}(g\circ\phi^{*})=\arg\min_{g}\frac{1}{|\mathcal{S}_{\mathrm{tr}}|}\sum_{(z_{i},l_{i})\sim\mathcal{S}_{\mathrm{tr}}}-[l_{i}\log p_{i}+(1-l_{i})(1-\log p_{i})], $$ 

where  $ p_{i} $  is defined in the problem setting in Section 2. In this paper,  $ g^{*} $  is a neural network consisting of multiple layers, so  $ g^{*} $  can be expressed by  $ g^{*} = h^{*} \circ h_{rep}^{*} $  where  $ h_{rep}^{*} \in \{h_{rep} : R^{k} \to R^{d_{rep}}\} $  and  $ h^{*} \in \{h : R^{d_{rep}} \to \{0,1\}\} $ . Normally, we call  $ h^{*} $  as a classification head and  $ h_{rep}^{*} $  as a representation function. Thus, a DR of  $ z_{i} $  is  $ h_{\mathrm{rep}}^{*} \circ \phi^{*}(z_{i}) $ .

Testing. In the end, compute the test statistic in Eq. (1) (by setting  $ f^{\prime*} $  as  $ g^{*} \circ \phi^{*} $ ) based on the original sequence of samples and the r times permuted samples, reject  $ H_{0} $  if original statistic is larger than the threshold derived from permuted statistics.

Overall algorithm. The procedure of how to implement SSL-C2ST is summarised in Algorithm 1 of the Appendix A. Note that, in Eq. (4), we can also consider continuing optimizing  $ \phi^{*} $ , which can increase the complexity of trainable classifiers, if the data is very complex.

### 5.2 THEORETICAL ANALYSIS OF SSL-C2ST

In this section, we discuss what the approximated power of our SSL-C2ST test is and how the size of unlabelled data  $ m_{u} $  helps to improve test power.

Test Power. Test power is the probability that a test will correctly reject  $ H_{0} $ , when  $ H_{1} $  holds. It represents the ability of the test to detect the difference between P and Q, so analyzing this power is essential for evaluating the performance of one two-sample testing method.

Theorem 5.1. (Lopez-Paz & Oquab, 2018b) Let $f^{\prime}\in\mathcal{C}_{\phi}:\mathcal{X}\to\{0,1\}$ be the SSL-C2ST classifier model. Let $H_{0}:t=\frac{1}{2}$ and $H_{1}:t=1-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})$, where $t$ is the test accuracy and $\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})=\operatorname*{Pr}_{(z_{i},l_{i})\sim\mathcal{D}}\left[f^{\prime}(z_{i})\neq l_{i}\right]/2\in\left(0,\frac{1}{2}\right)$ represents the inability of $f^{\prime}$ to distinguish between $\mathbb{P}$ and $\mathbb{Q}$. The test power of $\hat{t}$ is:

 $$ \mathrm{P r}_{H_{1}}\left(\hat{t}_{H_{0}}>t_{\alpha}\right)=\Phi\left(\frac{\left(\frac{1}{2}-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})\right)\sqrt{n_{\mathrm{t e}}}-\Phi^{-1}(1-\alpha)/2}{\sqrt{\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})^{2}}}\right), $$ 

where  $ \alpha\in(0,1) $  is the significance level,  $ t_{\alpha} $  is the  $ (1-\alpha) $  quantile and  $ \Phi $  is the CDF of standard normal distribution. The Type-I error of  $ \hat{t} $  is also controlled no more than  $ \alpha $ , which ensures that the test will not always reject  $ H_{0} $ , when  $ H_{0} $  is true.

Understand SSL-C2ST via Theorem 5.1. In hypothesis testing, our primary aim is to maximize test power while maintaining control over the Type-I error rate. While we know that via Theorem 5.1,  $ \Phi^{-1}(1-\alpha)/2 $  is a constant, for a reasonably fixed large  $ n_{te} $ , the first term  $ \left(\frac{1}{2}-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})\right) $  in the numerator dominates the test power. In fact, to ensure that the model can achieve the optimal test power on a fixed test dataset, it is equivalent to minimize

 $$ \mathcal{J}\left(\mathbb{P},\mathbb{Q};f^{\prime}\right):=\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})\Big/(1-\epsilon(\mathbb{P},\mathbb{Q};f^{\prime})), $$ 

where we estimate it with

 $$ \hat{\mathcal{J}}\left(S_{P},S_{Q};f^{\prime}\right):=\hat{\epsilon}(S_{P},S_{Q};f^{\prime})\Big/(1-\hat{\epsilon}(S_{P},S_{Q};f^{\prime})),\mathrm{\ and\ }\hat{\epsilon}(S_{P},S_{Q};f^{\prime})\in\left(0,\frac{1}{2}\right) $$ 

where  $ \hat{\epsilon}(S_{P}, S_{Q}; f') = \frac{1}{2} \widehat{\text{err}}(f') = \frac{1}{2 |\mathcal{S}|} \sum_{(x_{i}, l_{i}) \sim \mathcal{S}} \mathbb{I}[f'(x_{i}) \neq l_{i}] $ . The proof of above can be found in Appendix D.1. From Eq. (7), we can find that if we learn a classifier  $ f' $  from Eq. (4) that has a smaller  $ \hat{\epsilon}(S_{P}, S_{Q}; f') $ , we can minimize the  $ \hat{J} $ , leading to maximizing the test power. Thus, we will analyze how the use of unlabelled data and the size of unlabelled data  $ m_{u} $  helps to learn a classifier model  $ f' $  that have a smaller  $ \hat{\epsilon}(S_{P}, S_{Q}; f') $  in the semi-supervised learning.

We first give a definition of compatibility, an important measurement when analyzing SSL methods.

Definition 5.2 (Compatibility). The compatibility of classifier model f is defined as  $ \chi:C\timesX\to[0,1] $ , and  $ \chi(f,\mathcal{S})=\mathbb{E}_{x\sim\mathcal{S}}[\chi(f,x)] $  estimates how “compatible” the feature extractor of f is with fixed dataset S. Thus, the incompatibility of f with S is  $ 1-\chi(f,\mathcal{S}) $ . We can also call it unlabelled error rate  $ err_{\mathrm{unl}}(\phi_{f}) $ , where  $ err_{\mathrm{unl}}(\phi_{f})=1-\chi(f,\mathcal{S}) $ . Thus, given value  $ \xi $ , we define  $ \mathcal{C}_{\mathcal{S},\chi}(\xi)=\{f\in\mathcal{C}:\text{err}_{\mathrm{unl}}(\phi_{f})\leq\xi\} $ .

Then, the following theorem shows our main theoretical result, based on the compatibility.

Theorem 5.3. Let  $ f'^* = \arg\min_{f' \in \mathcal{C}_\phi} [\epsilon(\mathbb{P}, \mathbb{Q}; f') | err_{unl}(\phi_{f'}) \leq \xi] $ . The following holds with probability at least  $ 1 - \delta $ , for any arbitrarily small  $ \Delta_{m_u, m_1} > 0 $ ,

 $$ \hat{\epsilon}(S_{P},S_{Q};f^{\prime})\leq\epsilon(\mathbb{P},\mathbb{Q};f^{\prime}{}^{*})+\frac{\Delta_{m_{\mathrm{u}},m_{\mathrm{l}}}}{2}+\sqrt{\frac{\ln\left(\frac{4}{\delta}\right)}{8m_{\mathrm{u}}}}, $$ 

with the unlabelled sample size

 $$ m_{\mathrm{u}}=\mathcal{O}\left(\Delta^{-2}\log\Delta^{-1}\max\left[VCdim\left(\mathcal{C}_{\phi}\right),VCdim\left(\chi(\mathcal{C}_{\phi})\right)\right]+\Delta^{-2}\log(2/\delta)\right), $$ 

and the labelled sample size

 $$ m_{\mathrm{l}}=\frac{8}{\Delta^{2}}\left[\log\left(2\mathcal{C}_{\phi,\mathcal{S},\chi}(\xi+2\Delta)\left[2m_{\mathrm{l}},\mathcal{S}\right]\right)+\log(4/\delta)\right]. $$ 

Here, $\chi(\mathcal{C}_{\phi})=\{\chi_{f^{\prime}}:f^{\prime}\in\mathcal{C}_{\phi}\}$ is assumed to have a finite VC dimension, $\chi_{f^{\prime}}(\cdot)=\chi(f^{\prime},\cdot)$, and $\mathcal{C}_{\phi,\mathcal{S},\chi}(\xi+2\Delta)[2m_{1},\mathcal{S}]$ is the expected split number for $2m_{1}$ points drawn from $\mathcal{S}$ using functions in $\mathcal{C}_{\phi}\cap\mathcal{C}_{\mathcal{S},\chi}(\xi+2\Delta)$.

The proof of Theorem 5.3 is presented in the Appendix D.2. Theorem 5.3 indicates that when we are training model  $ f' $ , the increment in the size of unlabelled data  $ m_{u} $  can reduce the upper bound

<div style="text-align: center;"><img src="imgs/img_in_chart_box_291_162_500_337.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(a) Power vs. N; MNIST</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_505_161_713_337.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">(b) Power vs. N; HDGM; d=2</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_717_162_926_337.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(c) Power vs. N; HDGM; d=10</div>


<div style="text-align: center;">Figure 4: Test power of SSL-C2ST and C2ST. Barplot to show how SSL-C2ST outperforms C2ST in the MNIST dataset (a), HDGM-D when d=2 (b) and HDGM-D when d=10 (c).</div>


of the empirical error rate of  $ f^{\prime} $ . Thus, the upper bound of the empirical inability  $ \hat{\epsilon}(S_{P}, S_{Q}; f^{\prime}) $  will decrease as well, leading to a direct increase on the lower bound of the approximate test power as shown in Eq. (7). In other words, this theorem ensures the effectiveness of unlabelled data in the improvement of SSL-C2ST test power.

## 6 EXPERIMENTS

In this section, we will analyze the experiment result of SSL-C2ST on two commonly used benchmark datasets and one advanced ImageNet dataset. Also, we will discuss the empirical extensions based on SSL-C2ST (i.e., SSL-C2ST-M), and present the comparative analysis of experimental results across three benchmarks. The overview information of three datasets and experimental implementation details of proposed method and other two-sample testing methods can be found in Appendix C.

Datasets. We conducted experiments on five different datasets to thoroughly evaluate our methods. To assess the performance of current SSL methods applied to two-sample testing, we utilized three synthetic datasets: HDGM-Easy, HDGM-Medium, and HDGM-Hard. Moreover, we conduct the experiments of our proposed methods against other SOTA two-sample testing methods to evaluate the effectiveness of semi-supervised paradigm on three datasets: MNIST, ImageNet, and HDGM. Detailed descriptions of these datasets are provided in Appendix C.1.

Baselines. We evaluate the performance of our proposed methods SSL-C2ST and SSL-C2ST-M against several SOTA baseline methods in two-sample testing, specifically C2ST, C2ST-M, MMD-D, and MMD-FUSE. These baselines serve as competitive references to highlight the improvements achieved by our approach. For comprehensive details on each baseline method, including their implementations and parameter settings, please refer to Appendix C.2 and C.3.

Ablation Study: Verification of SSL-C2ST over C2ST. We first verify the effectiveness of SSL-C2ST via comparing SSL-C2ST and C2ST, which provides empirical evidence for Theorem 5.3. The implementation details of SSL-C2ST and C2ST can be found in Appendix C.2.

The visualized result of how our SSL-C2ST outperforms C2ST is displayed in Figure 4. In both dataset MNIST and HDGM-Hard, we can see that the test power of SSL-C2ST is higher than that of C2ST no matter how many numbers of two samples are drawn from the distribution. Although the differences between two methods are little when N is small, the test power of SSL-C2ST has a huge gap over C2ST when N is large enough and converges to 1 with a relative smaller N compare to C2ST. This also verifies our theoretical analysis that our model is more likely to have large improvement of test power if the number of unlabelled samples in the semi-supervised learning is sufficiently large.

Compared to C2ST, SSL-C2ST learns a compact and potentially more informative representation of the whole data, which makes efficient use of the unlabelled test data. This can not only discover underlying patterns or features that might not directly related to the labels but to the data distribution itself, but also provide a regularizing effect to prevent the model being more likely to overfit the training data. Such featurizer in the SSL-C2ST can result in a better generalization from the learned representations and improve the classifier's performance on the testing set predictions.

Test-power Results of SSL-C2ST-M. After we validate the effectiveness of SSL-C2ST, we will also introduce an advanced empirical testing method based on our SSL-C2ST: SSL-C2ST-M and how they

<div style="text-align: center;">(a) Power vs. N; d = 10</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_164_1004_356.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">(b) Level vs. N; d = 10</div>


<div style="text-align: center;">(c) Power vs. N; d = 2</div>


<div style="text-align: center;">(d) Level vs. N; d = 2</div>


<div style="text-align: center;">Figure 5: Results on HDGM-D and HDGM-S for  $ \alpha = 0.05 $ . Left: average test power (a) and average type-I error (b) when increasing total two sample size N from N = 1000 to N = 10000, keeping d = 10 in 100 trials. Right: average test power (c) and average type-I error (d) when increasing N from N = 1000 to N = 10000, keeping d = 2 in 100 trials.</div>


<div style="text-align: center;">Table 2: MNIST and ImageNet ( $ \alpha = 0.05 $ ). Average test power for comparing M real MNIST images to M DCGAN-generated MNIST images, and Average test power for comparing M real ImageNet images to M StyleGAN-XL-generated ImageNet images. $ ^{5} $ </div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td colspan="6">MNIST</td><td colspan="6">ImageNet</td></tr><tr><td style='text-align: center;'>M=200</td><td style='text-align: center;'>M=400</td><td style='text-align: center;'>M=600</td><td style='text-align: center;'>M=800</td><td style='text-align: center;'>M=1000</td><td style='text-align: center;'>Avg.</td><td style='text-align: center;'>M=200</td><td style='text-align: center;'>M=400</td><td style='text-align: center;'>M=600</td><td style='text-align: center;'>M=800</td><td style='text-align: center;'>M=1000</td><td style='text-align: center;'>Avg.</td></tr><tr><td style='text-align: center;'>C2ST</td><td style='text-align: center;'>0.180</td><td style='text-align: center;'>0.720</td><td style='text-align: center;'>0.980</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.776</td><td style='text-align: center;'>0.150</td><td style='text-align: center;'>0.300</td><td style='text-align: center;'>0.350</td><td style='text-align: center;'>0.600</td><td style='text-align: center;'>0.850</td><td style='text-align: center;'>0.450</td></tr><tr><td style='text-align: center;'>C2ST-M</td><td style='text-align: center;'>0.250</td><td style='text-align: center;'>0.730</td><td style='text-align: center;'>0.990</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.794</td><td style='text-align: center;'>0.150</td><td style='text-align: center;'>0.350</td><td style='text-align: center;'>0.450</td><td style='text-align: center;'>0.700</td><td style='text-align: center;'>0.850</td><td style='text-align: center;'>0.500</td></tr><tr><td style='text-align: center;'>MMD-D</td><td style='text-align: center;'>0.290</td><td style='text-align: center;'>0.996</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.857</td><td style='text-align: center;'>0.210</td><td style='text-align: center;'>0.400</td><td style='text-align: center;'>0.570</td><td style='text-align: center;'>0.780</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.592</td></tr><tr><td style='text-align: center;'>MMD-FUSE</td><td style='text-align: center;'>0.320</td><td style='text-align: center;'>0.870</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.838</td><td style='text-align: center;'>0.230</td><td style='text-align: center;'>0.450</td><td style='text-align: center;'>0.610</td><td style='text-align: center;'>0.790</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.616</td></tr><tr><td style='text-align: center;'>SSL-C2ST</td><td style='text-align: center;'>0.260</td><td style='text-align: center;'>0.950</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.842</td><td style='text-align: center;'>0.200</td><td style='text-align: center;'>0.400</td><td style='text-align: center;'>0.500</td><td style='text-align: center;'>0.650</td><td style='text-align: center;'>0.950</td><td style='text-align: center;'>0.540</td></tr><tr><td style='text-align: center;'>SSL-C2ST-M</td><td style='text-align: center;'>0.491</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.895</td><td style='text-align: center;'>0.400</td><td style='text-align: center;'>0.500</td><td style='text-align: center;'>0.650</td><td style='text-align: center;'>0.750</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>0.660</td></tr></table>

outperform state-of-the-art testing methods from the supervised paradigm (MMD with deep kernel) and the unsupervised paradigm (MMD-FUSE). The implementation details of these MMD-based methods can be found in Appendix C.3.

The overall result of all testing methods for the HDGM dataset is shown in Figure 5. We can see that SSL-C2ST-M method has the highest test power in both the 2-dimensional HDGM-D and 10-dimensional HDGM-D, no matter how we choose N, while all type-I errors are reasonably controlled around  $ \alpha = 0.05 $ . For MNIST and ImageNet datasets, the results of all methods are shown in Table 2, although SSL-C2ST-M does not outperform MMD-D and MMD-FUSE in MNIST when N = 400 and in ImageNet when N = 600, it has a clear increase in the test power when N is small, leading to a powerful average test power performance across two image datasets.

Discussion of Sequential Two-sample Testing. Moreover, sequential two-sample testing methods also utilize information from the test data but is a different problem setting from ours. We provide detailed descriptions of sequential two-sample testing in Appendix C.6, along with experimental results C.7 demonstrating that our methods outperform these approaches within our setting. Additionally, we discuss how our proposed paradigm can be applied to other supervised two-sample testing methods, potentially enhancing their performance in two-sample testing.

## 7 CONCLUSION

Non-parametric two-sample testing is an important problem in both statistics and machine learning fields. This paper presents a new paradigm, semi-supervised learning-based classifier two-sample test (SSL-C2ST), to learn better data representations for addressing this problem and gives a theoretical analysis of why the proposed paradigm can have a higher test power compared to two representative paradigms in the field. In the end, an advanced empirical testing method, SSL-C2ST with MMD (SSL-C2ST-M), is presented in the experiments and shows superior performance compared to previous testing methods. Both theoretical analysis and empirical evidence show that the proposed new paradigm might be a cure for key issues of the existing two paradigms in the two-sample testing field, paving a new road to revisit and address the two-sample testing problem.