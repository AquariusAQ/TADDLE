## A APPENDIX

### A.1 COMBINATION WITH OTHER OUTPUT-BASED LOSS

In our main paper, we utilize our feature separation loss based on OE method (Hendrycks et al., 2018) since it is the most classical approach. In this section, we also combine our feature separation loss with another output-based loss to demonstrate our wide availability. We adopt Energy-OE approach (Liu et al., 2020) as our basic loss, which is a commonly-used output-based loss. The mathematical formula is as follows:

 $$ \min_{f}L_{CE}+\lambda L_{energy} $$ 

 $$ L_{e n e r g y}=\mathbb{E}_{(x_{\mathrm{i n}},y)\sim D_{\mathrm{i n}}}(\operatorname*{m a x}(0,E(x_{\mathrm{i n}})-m_{\mathrm{i n}}))^{2} $$ 

 $$ +\mathbb{E}_{x_{\mathrm{out}}\sim D_{\mathrm{out}}^{\mathrm{aux}}}(\max(0,m_{\mathrm{out}}-E(x_{\mathrm{out}})))^{2} $$ 

where  $ E(x) = -T \cdot \log \sum_{i} C e^{f_{i}(x)/T} $  is the energy score function, and  $ \lambda $ ,  $ m_{in} $ , and  $ m_{out} $  are hyperparameters. We adopt the recommended setting in Energy-OE (Liu et al., 2020) to set the parameters and finetune our model. Combining our feature separation loss with the basic method, we obtain our training objective loss as follows:

 $$ \min_{f}L_{CE}+\lambda L_{energy}+\alpha L_{Clu}+\beta L_{Sep} $$ 

In our experiments, we still set  $ \alpha = 1.0 $  and  $ \beta = 1.0 $  for consistency with previous setting. The results are shown in Table 9, where our method significantly reduces the FPR95 by 6.35% on CIFAR100 benchmark compared to the basic Energy-OE approach, convincingly demonstrating our wide availability and effectiveness.

<div style="text-align: center;">Table 9: Combination with Energy-OE loss on CIFAR10 and CIFAR100 benchmarks. The best result is in bold.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td colspan="2">SVHN</td><td colspan="2">LSUN</td><td colspan="2">iSUN</td><td colspan="2">Textures</td><td colspan="2">Places365</td><td colspan="2">Average</td></tr><tr><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td></tr><tr><td colspan="13">CIFAR-10</td></tr><tr><td style='text-align: center;'>Energy-OE Liu et al. (2020)</td><td style='text-align: center;'>0.75</td><td style='text-align: center;'>99.50</td><td style='text-align: center;'>0.90</td><td style='text-align: center;'>98.98</td><td style='text-align: center;'>1.50</td><td style='text-align: center;'>99.22</td><td style='text-align: center;'>2.75</td><td style='text-align: center;'>98.92</td><td style='text-align: center;'>9.05</td><td style='text-align: center;'>97.33</td><td style='text-align: center;'>2.99</td><td style='text-align: center;'>98.79</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>1.20</td><td style='text-align: center;'>99.33</td><td style='text-align: center;'>0.65</td><td style='text-align: center;'>99.14</td><td style='text-align: center;'>1.75</td><td style='text-align: center;'>99.33</td><td style='text-align: center;'>2.20</td><td style='text-align: center;'>99.08</td><td style='text-align: center;'>7.55</td><td style='text-align: center;'>97.88</td><td style='text-align: center;'>2.67</td><td style='text-align: center;'>98.95</td></tr><tr><td colspan="13">CIFAR-100</td></tr><tr><td style='text-align: center;'>Energy-OE Liu et al. (2020)</td><td style='text-align: center;'>17.75</td><td style='text-align: center;'>96.94</td><td style='text-align: center;'>34.00</td><td style='text-align: center;'>94.82</td><td style='text-align: center;'>60.75</td><td style='text-align: center;'>87.32</td><td style='text-align: center;'>45.70</td><td style='text-align: center;'>90.09</td><td style='text-align: center;'>53.50</td><td style='text-align: center;'>89.08</td><td style='text-align: center;'>42.34</td><td style='text-align: center;'>91.65</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>11.05</td><td style='text-align: center;'>97.65</td><td style='text-align: center;'>21.35</td><td style='text-align: center;'>96.40</td><td style='text-align: center;'>52.95</td><td style='text-align: center;'>88.63</td><td style='text-align: center;'>42.65</td><td style='text-align: center;'>91.14</td><td style='text-align: center;'>51.95</td><td style='text-align: center;'>88.81</td><td style='text-align: center;'>35.99</td><td style='text-align: center;'>92.53</td></tr></table>

### A.2 Detailed Results on Different Architectures

We report the detailed results with ResNet18 and DenseNet121 architectures on CIFAR10 and CIFAR100 benchmarks in Table 10.

<div style="text-align: center;">Table 10: Detailed Results with ResNet18 and DenseNet121 architectures. The best result is in bold.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">SVHN</td><td colspan="2">LSUN</td><td colspan="2">iSUN</td><td colspan="2">Textures</td><td colspan="2">Places365</td><td colspan="2">Average</td></tr><tr><td style='text-align: center;'>FPR95↓</td><td style='text-align: center;'>AUROC↑</td><td style='text-align: center;'>FPR95↓</td><td style='text-align: center;'>AUROC↑</td><td style='text-align: center;'>FPR95↓</td><td style='text-align: center;'>AUROC↑</td><td style='text-align: center;'>FPR95↓</td><td style='text-align: center;'>AUROC↑</td><td style='text-align: center;'>FPR95↓</td><td style='text-align: center;'>AUROC↑</td><td style='text-align: center;'>FPR95↓</td><td style='text-align: center;'>AUROC↑</td></tr><tr><td colspan="14">CIFAR-10</td></tr><tr><td rowspan="3">ResNet18</td><td style='text-align: center;'>OE</td><td style='text-align: center;'>3.55</td><td style='text-align: center;'>97.46</td><td style='text-align: center;'>4.35</td><td style='text-align: center;'>98.00</td><td style='text-align: center;'>4.20</td><td style='text-align: center;'>97.58</td><td style='text-align: center;'>7.95</td><td style='text-align: center;'>97.47</td><td style='text-align: center;'>11.70</td><td style='text-align: center;'>96.25</td><td style='text-align: center;'>6.35</td><td style='text-align: center;'>97.35</td></tr><tr><td style='text-align: center;'>DAL</td><td style='text-align: center;'>0.75</td><td style='text-align: center;'>99.38</td><td style='text-align: center;'>1.70</td><td style='text-align: center;'>98.94</td><td style='text-align: center;'>2.10</td><td style='text-align: center;'>98.05</td><td style='text-align: center;'>4.35</td><td style='text-align: center;'>98.20</td><td style='text-align: center;'>9.15</td><td style='text-align: center;'>96.45</td><td style='text-align: center;'>3.61</td><td style='text-align: center;'>98.20</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>0.50</td><td style='text-align: center;'>99.60</td><td style='text-align: center;'>2.10</td><td style='text-align: center;'>99.19</td><td style='text-align: center;'>3.10</td><td style='text-align: center;'>99.02</td><td style='text-align: center;'>3.15</td><td style='text-align: center;'>98.71</td><td style='text-align: center;'>8.75</td><td style='text-align: center;'>97.26</td><td style='text-align: center;'>3.52</td><td style='text-align: center;'>98.75</td></tr><tr><td rowspan="3">DenseNet121</td><td style='text-align: center;'>OE</td><td style='text-align: center;'>4.40</td><td style='text-align: center;'>98.51</td><td style='text-align: center;'>4.40</td><td style='text-align: center;'>98.68</td><td style='text-align: center;'>22.80</td><td style='text-align: center;'>96.19</td><td style='text-align: center;'>5.55</td><td style='text-align: center;'>98.52</td><td style='text-align: center;'>16.80</td><td style='text-align: center;'>95.78</td><td style='text-align: center;'>10.79</td><td style='text-align: center;'>97.54</td></tr><tr><td style='text-align: center;'>DAL</td><td style='text-align: center;'>3.10</td><td style='text-align: center;'>98.65</td><td style='text-align: center;'>3.10</td><td style='text-align: center;'>99.05</td><td style='text-align: center;'>21.40</td><td style='text-align: center;'>96.56</td><td style='text-align: center;'>5.55</td><td style='text-align: center;'>98.46</td><td style='text-align: center;'>15.60</td><td style='text-align: center;'>95.85</td><td style='text-align: center;'>9.75</td><td style='text-align: center;'>97.71</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>4.45</td><td style='text-align: center;'>98.73</td><td style='text-align: center;'>4.45</td><td style='text-align: center;'>98.86</td><td style='text-align: center;'>13.90</td><td style='text-align: center;'>97.41</td><td style='text-align: center;'>4.90</td><td style='text-align: center;'>98.76</td><td style='text-align: center;'>16.80</td><td style='text-align: center;'>95.94</td><td style='text-align: center;'>8.90</td><td style='text-align: center;'>97.94</td></tr><tr><td colspan="14">CIFAR-100</td></tr><tr><td rowspan="3">ResNet18</td><td style='text-align: center;'>OE</td><td style='text-align: center;'>55.75</td><td style='text-align: center;'>92.95</td><td style='text-align: center;'>35.45</td><td style='text-align: center;'>93.96</td><td style='text-align: center;'>70.10</td><td style='text-align: center;'>87.37</td><td style='text-align: center;'>65.00</td><td style='text-align: center;'>88.38</td><td style='text-align: center;'>58.50</td><td style='text-align: center;'>88.29</td><td style='text-align: center;'>56.96</td><td style='text-align: center;'>90.19</td></tr><tr><td style='text-align: center;'>DAL</td><td style='text-align: center;'>51.55</td><td style='text-align: center;'>93.40</td><td style='text-align: center;'>32.35</td><td style='text-align: center;'>94.65</td><td style='text-align: center;'>69.75</td><td style='text-align: center;'>88.90</td><td style='text-align: center;'>63.50</td><td style='text-align: center;'>89.52</td><td style='text-align: center;'>57.30</td><td style='text-align: center;'>88.31</td><td style='text-align: center;'>54.89</td><td style='text-align: center;'>90.95</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>36.70</td><td style='text-align: center;'>93.29</td><td style='text-align: center;'>34.90</td><td style='text-align: center;'>94.12</td><td style='text-align: center;'>66.00</td><td style='text-align: center;'>88.68</td><td style='text-align: center;'>57.35</td><td style='text-align: center;'>90.01</td><td style='text-align: center;'>56.75</td><td style='text-align: center;'>88.39</td><td style='text-align: center;'>50.34</td><td style='text-align: center;'>90.90</td></tr><tr><td rowspan="3">DenseNet121</td><td style='text-align: center;'>OE</td><td style='text-align: center;'>59.40</td><td style='text-align: center;'>90.35</td><td style='text-align: center;'>48.70</td><td style='text-align: center;'>90.15</td><td style='text-align: center;'>70.65</td><td style='text-align: center;'>83.28</td><td style='text-align: center;'>66.90</td><td style='text-align: center;'>85.36</td><td style='text-align: center;'>64.75</td><td style='text-align: center;'>84.65</td><td style='text-align: center;'>62.08</td><td style='text-align: center;'>86.76</td></tr><tr><td style='text-align: center;'>DAL</td><td style='text-align: center;'>47.00</td><td style='text-align: center;'>92.81</td><td style='text-align: center;'>60.05</td><td style='text-align: center;'>87.88</td><td style='text-align: center;'>55.20</td><td style='text-align: center;'>88.65</td><td style='text-align: center;'>65.40</td><td style='text-align: center;'>86.90</td><td style='text-align: center;'>78.60</td><td style='text-align: center;'>82.07</td><td style='text-align: center;'>61.25</td><td style='text-align: center;'>87.66</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>67.40</td><td style='text-align: center;'>90.49</td><td style='text-align: center;'>37.15</td><td style='text-align: center;'>92.77</td><td style='text-align: center;'>65.15</td><td style='text-align: center;'>85.68</td><td style='text-align: center;'>63.00</td><td style='text-align: center;'>87.13</td><td style='text-align: center;'>62.95</td><td style='text-align: center;'>86.18</td><td style='text-align: center;'>59.13</td><td style='text-align: center;'>88.45</td></tr></table>

### A.3 HARD OOD DETECTION

In addition to testing on the regular OOD datasets, we further consider three hard OOD datasets proposed in Tack et al. (2020), which are considered more difficult to distinguish from ID samples.

Following the same setting in (Tack et al., 2020; Sun et al., 2022b; Wang et al., 2024), we evaluate our detection performance on LSUN-Fix (Yu et al., 2015), ImageNet-Resize (Deng et al., 2009b) and CIFAR100 (Krizhevsky et al., 2009a) with CIFAR10 as the ID dataset. Specific results are shown in Table 11. As we can see, our method shows comparable performance with DAL (Wang et al., 2024) over three hard OOD datasets, outperforming the baseline OE method by 2.55% on FPR95 on the ImageNet-Resize dataset.

<div style="text-align: center;">Table 11: Hard OOD detection on CIFAR10 benchmark.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3">Method</td><td colspan="7">Near-OOD Dataset</td><td style='text-align: center;'></td></tr><tr><td colspan="2">LSUN-Fix</td><td colspan="2">ImageNet-Resize</td><td colspan="2">CIFAR-100</td><td colspan="2">Tiny-ImageNet</td></tr><tr><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td></tr><tr><td colspan="9">With contrastive learning</td></tr><tr><td style='text-align: center;'>CSI*</td><td style='text-align: center;'>39.79</td><td style='text-align: center;'>93.63</td><td style='text-align: center;'>37.47</td><td style='text-align: center;'>93.93</td><td style='text-align: center;'>45.64</td><td style='text-align: center;'>87.64</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>CIDER</td><td style='text-align: center;'>8.98</td><td style='text-align: center;'>98.56</td><td style='text-align: center;'>43.45</td><td style='text-align: center;'>93.82</td><td style='text-align: center;'>55.84</td><td style='text-align: center;'>90.0</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>KNN+*</td><td style='text-align: center;'>24.88</td><td style='text-align: center;'>95.75</td><td style='text-align: center;'>30.52</td><td style='text-align: center;'>94.85</td><td style='text-align: center;'>40.00</td><td style='text-align: center;'>89.11</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td colspan="9">With auxiliary OOD data</td></tr><tr><td style='text-align: center;'>OE</td><td style='text-align: center;'>1.00</td><td style='text-align: center;'>99.53</td><td style='text-align: center;'>7.20</td><td style='text-align: center;'>98.48</td><td style='text-align: center;'>25.05</td><td style='text-align: center;'>94.86</td><td style='text-align: center;'>19.55</td><td style='text-align: center;'>91.49</td></tr><tr><td style='text-align: center;'>DAL</td><td style='text-align: center;'>0.65</td><td style='text-align: center;'>99.59</td><td style='text-align: center;'>3.75</td><td style='text-align: center;'>98.63</td><td style='text-align: center;'>26.00</td><td style='text-align: center;'>94.35</td><td style='text-align: center;'>20.75</td><td style='text-align: center;'>92.18</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>0.75</td><td style='text-align: center;'>99.07</td><td style='text-align: center;'>4.65</td><td style='text-align: center;'>98.42</td><td style='text-align: center;'>24.60</td><td style='text-align: center;'>94.69</td><td style='text-align: center;'>17.65</td><td style='text-align: center;'>92.48</td></tr></table>

### A.4 GENERALIZATION TO CLASS-IMBALANCED DATASETS

Since our method is based on the assumption of the NC property, we recognize that the phenomenon may not always hold in real-world scenarios. Therefore, in this section, we evaluate our performance on imbalanced data distribution to explore the generalizability of our approach.

On imbalanced data distribution, the NC property declines to “Minority Collapse” Zhang et al. (2024a), where the classifiers for minority classes are squeezed into one direction. This phenomenon destroys the geometric structure of NC, but does not break the low-dimensional property of features of ID data. Following the settings in Cao et al. (2019); Zhu et al. (2022a), we create an imbalanced version of CIFAR10 dataset, denote as CIFAR10-LT. Specifically, we consider a long-tailed imbalance with ratio (denote the ratio between sample sizes of the most frequent and least frequent class). Based on above imbalanced data, we firstly pretrain a WideResNet-40-2 model on CIFAR10-LT, and then finetune the model using auxiliary OOD datasets. The training setting is the same as our previous experiments. The final detection performance of our method and OE is shown in Table 12, which validates our effectiveness under imbalanced data condition.

<div style="text-align: center;">Table 12: Results on imbalanced CIFAR10 data. We report the FPR95/AUROC.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td colspan="2">SVHN</td><td colspan="2">LSUN</td><td colspan="2">iSUN</td><td colspan="2">Textures</td><td colspan="2">Places365</td><td colspan="2">Average</td></tr><tr><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td></tr><tr><td style='text-align: center;'>OE Hendrycks et al. (2018)</td><td style='text-align: center;'>15.15</td><td style='text-align: center;'>96.56</td><td style='text-align: center;'>12.00</td><td style='text-align: center;'>97.25</td><td style='text-align: center;'>19.70</td><td style='text-align: center;'>96.63</td><td style='text-align: center;'>18.05</td><td style='text-align: center;'>96.15</td><td style='text-align: center;'>29.75</td><td style='text-align: center;'>93.63</td><td style='text-align: center;'>18.93</td><td style='text-align: center;'>96.04</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>14.05</td><td style='text-align: center;'>96.38</td><td style='text-align: center;'>12.05</td><td style='text-align: center;'>96.93</td><td style='text-align: center;'>13.55</td><td style='text-align: center;'>97.38</td><td style='text-align: center;'>24.05</td><td style='text-align: center;'>95.63</td><td style='text-align: center;'>24.95</td><td style='text-align: center;'>94.07</td><td style='text-align: center;'>17.73</td><td style='text-align: center;'>96.08</td></tr></table>

### A.5 TRAINING STRATEGY CHOICE AND IMPACT

In this section, we discuss the choice of our training strategy and its impact on our performance. Our training process is consisted of two stage: Stage I-training with cross-entropy loss to ensure the occurrence of NC phenomenon; Stage II-training with all losses including cross-entropy, outlier exposure, and our proposed separation and clustering losses to enlarge the discrepancy between ID and OOD data. When a well-trained model is used as the initial parameter, Stage I is not necessary. In practice, for the sake of an unified and simple framework, we can always firstly train with CE loss, and when the accuracy remains unchanged, switch to Stage II. But in our experiment, to align with experimental settings of other methods, we fixed the training epoch at 5, which is a relatively small epoch, thus we directly use Stage II on the ImageNet benchmark.

To further investigate the impact of training strategy on our performance, we conduct experiments

on CIFAR10 benchmark (where the initial model parameter is half-trained), with different training epochs of Stage I. Specifically, denote the training epoch of Stage I as "Ep-CE", training epoch of Stage II as "Ep-All", we choose different Ep-CE ranging from [0, 10, 20, 25, 30, 40, 50] with a fixed Ep-All equalling to 25. When Ep-CE equals to zero, it corresponds to directly training the model using Stage II. Experiment results are shown in Table 13, where our method is not sensitive to the strategy choice, but CE loss fine-tuning does enhance performance for half-trained models.

<div style="text-align: center;">Table 13: Influence of training strategy on CIFAR10 benchmark. We report the average FPR95/AUROC across five OOD datasets.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Ep-CE + Ep-All</td><td style='text-align: center;'>0+50</td><td style='text-align: center;'>10+25</td><td style='text-align: center;'>20+25</td><td style='text-align: center;'>25+25</td><td style='text-align: center;'>30+25</td><td style='text-align: center;'>40+25</td><td style='text-align: center;'>50+25</td></tr><tr><td style='text-align: center;'>FPR95/AUROC</td><td style='text-align: center;'>2.53/98.53</td><td style='text-align: center;'>2.56/99.06</td><td style='text-align: center;'>2.56/99.09</td><td style='text-align: center;'>2.49/98.93</td><td style='text-align: center;'>2.46/99.16</td><td style='text-align: center;'>2.01/99.11</td><td style='text-align: center;'>2.53/98.97</td></tr><tr><td style='text-align: center;'>ID Acc</td><td style='text-align: center;'>94.43</td><td style='text-align: center;'>95.33</td><td style='text-align: center;'>95.52</td><td style='text-align: center;'>95.53</td><td style='text-align: center;'>95.56</td><td style='text-align: center;'>95.64</td><td style='text-align: center;'>95.35</td></tr></table>

### A.6 FEATURE DISTANCE ANCHORING ON CLASS MEAN

Since our training loss explicitly optimizes the cosine similarity between FC weights and features, the distance metric we used in Table 7 is aligned with our objective in some degree. Therefore, for a more fair comparison, we evaluate the feature distance based on the class mean vectors instead of FC weights in this section. The result is shown in Table 14. The absolute value is different from results calculated based on model weights, but in relative comparison, our method still shows better feature separation between ID and OOD data.

<div style="text-align: center;">Table 14: Feature Distance based on class mean vectors. we report the average Euclidean distance and cosine similarity on the whole dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td colspan="3">Euclidean Distance</td><td colspan="3">Cosine Similarity</td></tr><tr><td style='text-align: center;'>ID</td><td style='text-align: center;'>OOD</td><td style='text-align: center;'>Diff $ \uparrow $</td><td style='text-align: center;'>ID</td><td style='text-align: center;'>OOD</td><td style='text-align: center;'>Diff $ \uparrow $</td></tr><tr><td style='text-align: center;'>Vanilla</td><td style='text-align: center;'>2.31</td><td style='text-align: center;'>3.08</td><td style='text-align: center;'>0.77</td><td style='text-align: center;'>0.94</td><td style='text-align: center;'>0.81</td><td style='text-align: center;'>0.13</td></tr><tr><td style='text-align: center;'>OE</td><td style='text-align: center;'>1.46</td><td style='text-align: center;'>4.91</td><td style='text-align: center;'>3.45</td><td style='text-align: center;'>0.94</td><td style='text-align: center;'>0.49</td><td style='text-align: center;'>0.45</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>1.47</td><td style='text-align: center;'>6.18</td><td style='text-align: center;'>4.71</td><td style='text-align: center;'>0.98</td><td style='text-align: center;'>0.31</td><td style='text-align: center;'>0.67</td></tr></table>

### A.7 Fluctuation in Detection Performance

We have observed considerable fluctuations in the performance of the DAL method (Wang et al., 2024) under repeated experiments with identical settings. Therefore, we evaluate the mean and variance of performance after repeating the same experiment five times. The results, shown in Table 15, indicate that our approach exhibits greater stability, particularly on the CIFAR100 benchmark. Notably, even the OE method shows the FPR95 variance of 1.137% on CIFAR100, whereas our method maintains a variance of only 0.027%.

<div style="text-align: center;">Table 15: Fluctuation in detection performance of different methods.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td></tr><tr><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td></tr><tr><td style='text-align: center;'>OE Hendrycks et al. (2018)</td><td style='text-align: center;'>3.22 $ \pm $ 0.0017</td><td style='text-align: center;'>99.07 $ \pm $ 0.0014</td><td style='text-align: center;'>36.29 $ \pm $ 1.137</td><td style='text-align: center;'>92.31 $ \pm $ 0.013</td></tr><tr><td style='text-align: center;'>DAL Wang et al. (2024)</td><td style='text-align: center;'>2.87 $ \pm $ 0.0234</td><td style='text-align: center;'>98.82 $ \pm $ 0.0027</td><td style='text-align: center;'>30.44 $ \pm $ 2.216</td><td style='text-align: center;'>93.07 $ \pm $ 0.075</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>2.49 $ \pm $ 0.0007</td><td style='text-align: center;'>98.92 $ \pm $ 0.0033</td><td style='text-align: center;'>29.50 $ \pm $ 0.027</td><td style='text-align: center;'>93.98 $ \pm $ 0.014</td></tr></table>

### A.8 DIFFERENT SCORE FUNCTIONS

Since we use our proposed score function in Eq 6 to detect OOD samples while other auxiliary OOD data based methods only employ MSP score (Hendrycks & Gimpel, 2016), in this part, we also evaluate our model using MSP score for a fair comparison. The results in Table 16 demonstrate

that our approach also achieves commendable performance under the MSP score, with only a slight decline compared to using the proposed score function.

<div style="text-align: center;">Table 16: Performance of adopting different score functions in our method.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td rowspan="2">Score Function</td><td colspan="2">CIFAR10</td><td colspan="2">CIFAR100</td></tr><tr><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td><td style='text-align: center;'>FPR95 $ \downarrow $</td><td style='text-align: center;'>AUROC $ \uparrow $</td></tr><tr><td rowspan="2">Ours</td><td style='text-align: center;'>MSP</td><td style='text-align: center;'>2.77</td><td style='text-align: center;'>98.76</td><td style='text-align: center;'>29.96</td><td style='text-align: center;'>93.27</td></tr><tr><td style='text-align: center;'>Eq. 6</td><td style='text-align: center;'>2.49</td><td style='text-align: center;'>98.93</td><td style='text-align: center;'>29.47</td><td style='text-align: center;'>94.00</td></tr></table>

### A.9 VISUALIZATION

In Figure 2, we visualize the random two-class of samples of CIFAR10 dataset and the test unseen OOD samples of SVHN dataset. The low-dimensional visualization is calculated by linear projection into the subspace spanned by  $ w_{1} $ ,  $ w_{2} $  and principal components of OOD features, where  $ w_{i} $  is the corresponding model parameters of the last fully connected layer. Specifically, denote features as  $ z \in R^{n \times d} $ , the reduction matrix as  $ M \in R^{d \times 3} $ , where n is the sample number and d is the feature dimension, then the coordinate in 3D space is computed as:  $ z_{3D} = zM $ ,  $ z_{3D} \in R^{n \times 3} $ . Using the above linear projection operation, we obtain the coordinate for both ID and OOD samples, then visualize them in 2D and 3D space. In the following, we respectively choose two classes of ID samples and the test unseen OOD data to visualize their features in 2D and 3D space. In these figures, we can discover that, for all samples of the ten classes in CIFAR10, ID features rarely distribute on redundant dimensions, in contrast, OOD features almost locate on redundant dimensions while little component on model weight dimensions.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_240_908_481_1092.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(a) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_492_910_730_1092.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(b) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_741_909_983_1093.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(c) Our model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_248_1146_485_1376.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(d) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_503_1149_733_1378.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(e) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_754_1149_983_1379.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(f) Our model</div>


<div style="text-align: center;">Figure 3: ID sample: Class-0 and Class-1, OOD sample: SVHN</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_238_174_481_356.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(a) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_487_173_731_355.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(b) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_739_176_983_356.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(c) Our model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_247_410_484_641.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(d) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_504_412_733_639.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(e) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_746_419_984_643.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(f) Our model</div>


<div style="text-align: center;">Figure 4: ID sample: Class-2 and Class-3, OOD sample: SVHN</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_238_913_481_1095.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(a) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_489_912_730_1096.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(b) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_739_914_983_1096.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(c) Our model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_253_1149_483_1379.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(d) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_501_1148_734_1377.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(e) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_750_1152_984_1377.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(f) Our model</div>


<div style="text-align: center;">Figure 5: ID sample: Class-4 and Class-5, OOD sample: SVHN</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_238_175_482_356.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(a) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_490_175_731_355.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(b) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_739_174_983_356.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(c) Our model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_246_408_482_643.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(d) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_504_411_733_643.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(e) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_746_424_985_643.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(f) Our model</div>


<div style="text-align: center;">Figure 6: ID sample: Class-6 and Class-7, OOD sample: SVHN</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_238_783_480_967.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(a) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_489_785_732_966.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(b) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_738_785_982_967.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(c) Our model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_251_1021_483_1249.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(d) Vanilla model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_503_1025_733_1249.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(e) OE-trained model</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_753_1027_984_1250.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(f) Our model</div>


<div style="text-align: center;">Figure 7: ID sample: Class-8 and Class-9, OOD sample: SVHN</div>