## A APPENDIX

### A.1 OBTAINING CONFORMAL CONFIDENCE SETS WITH INCREASING COMBINATION FUNCTIONS

As discussed in Remark 2.3 the results of Sections 2.2 and 2.3 can be generalized to a wider class of combination functions.

Definition A.1. We define a suitable combination function to be a function  $ C : \mathcal{P}(\mathcal{V}) \times \mathcal{X} \to \mathbb{R} $  which is increasing in the sense that for all sets  $ A \subseteq V $  and each  $ v \in A $ ,  $ C(v, X) \leq C(\mathcal{A}, X) $  for all  $ X \in X $ .

The maximum is a suitable combination function since  $ X(v) = \max_{v \in \{v\}} X(v) \leq \max_{v \in \mathcal{A}} X(v) $ . As such this framework directly generalizes the results of the main text.

We can construct generalized marginal confidence sets as follows.

Theorem A.2. (Marginal inner set) Under Assumptions 1 and 2, given  $ \alpha_{1}\in(0,1) $ , define

 $$ \lambda_{I}(\alpha_{1})=\inf\left\{\lambda:\frac{1}{n}\sum_{i=1}^{n}1\left[C(\{v\in\mathcal{V}:Y_{i}(v)=1\},f_{I}(s(X_{i})))\leq\lambda\right]\geq\frac{\lceil(1-\alpha_{1})(n+1)\rceil}{n}\right\}, $$ 

for a suitable combination function C, and define  $ I(X) = \{v \in \mathcal{V} : C(v, f_{I}(s(X))) > \lambda_{I}(\alpha_{1})\} $ . Then,

 $$ \mathbb{P}\left(I(X_{n+1})\subseteq\{v\in\mathcal{V}:Y_{n+1}(v)=1\}\right)\geq1-\alpha_{1}. $$ 

The proof follows that of Theorem 2.1. The key observation is that for any suitable combination function C, given  $ \lambda \in R $ ,  $ A \subseteq V $  and  $ X \in X $ ,  $ C(A, X) \leq \lambda $  implies that  $ C(v, X) \leq \lambda $ . This is the relevant property of the maximum which we used for the results in the main text. For the outer set we similarly have the following.

Theorem A.3. (Marginal outer set) Under Assumptions 1 and 2, given  $ \alpha_{2} \in (0,1) $ , define

 $$ \lambda_{O}(\alpha_{2})=\inf\left\{\lambda:\frac{1}{n}\sum_{i=1}^{n}1\left[C(\{v\in\mathcal{V}:Y_{i}(v)=0\},-f_{O}(s(X_{i})))\leq\lambda\right]\geq\frac{\lceil(1-\alpha_{2})(n+1)\rceil}{n}\right\}. $$ 

for a suitable combination function C, and let  $ O(X) = \{v \in \mathcal{V} : C(v, -f_O(s(X))) \leq \lambda_O(\alpha_2)\} $ . Then,

 $$ \mathbb{P}\left(\{v\in\mathcal{V}:Y_{n+1}(v)=1\}\subseteq O(X_{n+1})\right)\geq1-\alpha_{2}. $$ 

Joint results can be analogously obtained.

### A.2 Obtaining Confidence Sets from Risk Control

We can alternatively establish Theorems 2.1 and A.2 using an argument from risk control (Angelopoulos et al., 2024). In particular, given an image pair  $ (X, Y) $  and  $ \lambda \in R $ , let

 $$ I_{\lambda}(X)=\{v\in\mathcal{V}:f_{I}(s(X),v)>\lambda\}. $$ 

Define a loss function,  $ L : \mathcal{P}(\mathcal{V}) \times \mathcal{Y} \to \mathbb{R} $  which sends  $ (X, Y) $  to

 $$ L(I_{\lambda}(X),Y)=1\left[I_{\lambda}(X)\not\subseteq\{v\in\mathcal{V}:Y(v)=1\}\right]. $$ 

For  $ i=1,\ldots,n+1 $ , let  $ L_{i}(\lambda)=L(I_{\lambda}(X_{i}),Y_{i}) $ . Arguing as in the proof of Theorem 2.1 it follows that  $ L_{i}(\lambda)=1[\tau_{i}>\lambda] $ . Then applying Theorem 1 of Angelopoulos et al. (2024) it follows that

 $$ \mathbb{E}\left[L_{n+1}(\hat{\lambda})\right]\leq\alpha_{1}, $$ 

where  $ \hat{\lambda}=\inf\left\{\lambda:\frac{1}{n}\sum_{i=1}^{n}L_{i}(\lambda)\leq\alpha_{1}-\frac{1-\alpha_{1}}{n}\right\} $ . Arguing as in Appendix A of (Angelopoulos et al., 2024) it follows that

 $$ \hat{\lambda}=\inf\left\{\lambda:\frac{1}{n}\sum_{i=1}^{n}1\left[\tau_{i}\leq\lambda\right]\geq\frac{\lceil(1-\alpha_{1})(n+1)\rceil}{n}\right\}=\lambda_{I}(\alpha_{1}), $$ 

and so  $ I(X) = I_{\hat{\lambda}}(X) $ . As such

 $$ \mathbb{P}\left(I(X_{n+1})\subseteq\{v\in\mathcal{V}:Y_{n+1}(v)=1\}\right)=1-\mathbb{E}\left\lceil L_{n+1}(\hat{\lambda})\right\rceil\geq1-\alpha_{1}, $$ 

and we recover the desired result. Arguing similarly it is possible to establish a proof of Theorem 2.2.

### A.3 CHARACTERIZING THE RELATIONSHIP BETWEEN HAUSSDRUFF DISTANCE AND THE DISTANCE TRANSFORMED SCORES

In this section we provide a proof of Theorem 2.8 and state the analogous result for the inner confidence sets obtained using the distance transformation.

Proof. Consider the outer confidence sets obtained using the distance transformed scores. Then given  $ 1 \leq i \leq n $  such that  $ H(\hat{M}(X_{i}), Y_{i}) \leq k $ , we have

 $$ Y_{i}=\left(Y_{i}\cap\hat{M}(X_{i})\right)\cup\left(Y_{i}\cap\hat{M}(X_{i})^{C}\right) $$ 

where the union is disjoint. The distance transformed scores  $ d_{\rho}(\hat{M}(X_{i}),v) $  are positive for  $ v \in \hat{M}(X_{i}) $  and negative for  $ v \notin \hat{M}(X_{i}) $ . As such

 $$ \begin{align*}\gamma_{i}&=\max_{v\in\mathcal{V}:Y_{i}(v)=1}-f_{O}(s(X_{i}),v)=\max_{v\in\mathcal{V}:Y_{i}(v)=1}-d_{\rho}(\hat{M}(X_{i}),v)\\&=\max_{v\in\mathcal{V}:Y_{i}(v)=1}\min_{e\in E(\hat{M}(X_{i}))}\rho(v,e)\leq H_{\rho}(\hat{M}(X_{i}),Y_{i})\leq k.\end{align*} $$ 

Since this holds for all i on a set $J$ which has $\frac{|J|}{n} > 1 - \alpha_{2}$, it follows that $\lambda_{O}(\alpha_{2}) \leq k$. Arguing similarly in the opposite direction it follows that for any new observation $X_{n+1}$ we have that

 $$ H_{\rho}(\hat{M}(X_{n+1}),O(X_{n+1}))\leq k. $$ 

Finally if  $ H_{\rho}(\hat{M}(X_{n+1}),Y_{n+1})\leq k $ , then it follows that  $ H_{\rho}(O(X_{n+1}),Y_{n+1})\leq2k $  by the triangle inequality.

A similar result can be established for the inner confidence sets via an analogous proof. We state this formally as follows.

Theorem A.4. For each  $ v \in V $ , let  $ f_{I}(s(X), v) = d_{\rho}(\hat{M}(X), v) $  and define  $ I(X) $  as in Section 2.2. Suppose that  $ H_{\rho}(\hat{M}(X_{i}), Y_{i}) \leq k $ , some  $ k \in R $ , for all  $ i \in J $ , for some  $ J \subseteq \{1, \ldots, n\} $  such that  $ \frac{|J|}{n} > 1 - \alpha_{1} $ . Then  $ H_{\rho}(\hat{M}(X_{n+1}), I(X_{n+1})) \leq k $ . In particular if  $ H_{\rho}(\hat{M}(X_{n+1}), Y_{n+1}) \leq k $ , then  $ H_{\rho}(I(X_{n+1}), Y_{n+1}) \leq 2k $ .

### A.4 Deriving Confidence Sets from Bounding Boxes

We can use our results in order to provide valid inference for bounding boxes via an adaptation of the approach of Andéol et al. (2023). In particular given  $ Z \in Y $ , let  $ B_{I,\operatorname*{max}}(Z) $  be the largest box which can be contained within the set  $ \{v \in \mathcal{V} : Z(v) = 1\} $  and let  $ B_{O,\operatorname*{min}}(Z) $  be the smallest box which contains the set  $ \{v \in \mathcal{V} : Z(v) = 1\} $ . Given  $ Y \in Y $ , let  $ cc(Y) \subseteq \mathcal{P}(\mathcal{V}) $  denote the set of connected components of the set  $ \{v \in \mathcal{V} : Y(v) = 1\} $  for a given connectivity criterion (which we take to be 4 in our examples), and note that these components can themselves be identified as elements of Y. Define

 $$ B_{I}(Y)=\cup_{c\in cc(Y)}B_{I,\max}(c)\ \mathrm{and}\ B_{O}(Y)=\cup_{c\in cc(Y)}B_{O,\min}(c) $$ 

to be the unions of the largest inner and smallest outer boxes of the connected components of the image Y, respectively. Then define

 $$ \hat{B}_{I}(s(X))=\cup_{c\in cc(\hat{M}(X))}B_{I,\max}(c)and\hat{B}_{O}(s(X))=\cup_{c\in cc(\hat{M}(X))}B_{O,\min}(c) $$ 

to be the unions of the largest inner and smallest outer boxes of the connected components of the predicted mask  $ \hat{M}(X) $ , respectively. Note that this is well-defined as  $ \hat{M}(X) $  is a function of  $ s(X) $ .

For the remainder of this section we shall assume that  $ V \subset R^{2} $ , this is not strictly necessary but will help to simplify notation. Given  $ u, v \in V $ , write  $ u = (u_{1}, u_{2}) $  and  $ v = (v_{1}, v_{2}) $  and let  $ \rho(u, v) = \max(|u_{1} - v_{1}|, |u_{2} - v_{2}|) $  be the chessboard metric.

Definition A.5. (Bounding box scores) For each  $ X \in X $  and  $ v \in V $ , let

 $$ b_{I}(s(X),v)=d_{\rho}(\hat{B}_{I}(s(X)),v)\mathrm{a n d}b_{O}(s(X),v)=d_{\rho}(\hat{B}_{O}(s(X)),v) $$ 

be the distance transformed scores based on the chessboard distance to the predicted inner and outer box collections  $ \hat{B}_{I}(s(X)) $  and  $ \hat{B}_{O}(s(X)) $ , respectively. We also define a combination of these  $ b_{M} $ , primarily for the purposes of plotting in Figure 2, as follows. Let  $ b_{M}(s(X),v)=b_{O}(s(X),v) $  for each  $ v\notin\hat{B}_{O}(s(X)) $  and let  $ b_{M}(s(X),v)=\max(b_{I}(s(X),v),0) $  for  $ v\in\hat{B}_{O}(s(X)) $ . We shall write  $ b_{I}(s(X))\in\mathcal{X} $  to denote the image which has  $ b_{I}(s(X))(v)=b_{I}(s(X),v) $  and similarly for  $ b_{O}(s(X)) $  and  $ b_{M}(s(X)) $ .

Now consider the sequences of image pairs  $ (X_{i}, B_{i}^{I})_{i=1}^{n} $  and  $ (X_{i}, B_{i}^{O})_{i=1}^{n} $ . These both satisfy exchangeability and so, applying Theorems A.2 and A.3, we obtain the following bounding box validity results.

Corollary A.6. (Marginal inner bounding boxes) Suppose Assumption 1 holds and that  $ (X_{i}, Y_{i})_{i=1}^{n+1} $  is independent of the functions s and  $ b_{I} $ . Given  $ \alpha_{1} \in (0,1) $ , define

 $$ \lambda_{I}(\alpha_{1})=\inf\left\{\lambda:\frac{1}{n}\sum_{i=1}^{n}1\left[C(B_{i}^{I},b_{I}(s(X_{i})))\leq\lambda\right]\geq\frac{\lceil(1-\alpha_{1})(n+1)\rceil}{n}\right\}, $$ 

for a suitable combination function C, and define  $ I(X) = \{v \in \mathcal{V} : C(v, b_{I}(s(X))) > \lambda_{I}(\alpha_{1})\} $ . Then,

 $$ \mathbb{P}\left(I(X_{n+1})\subseteq B_{n+1}^{I}\subseteq\{v\in\mathcal{V}:Y_{n+1}(v)=1\}\right)\geq1-\alpha_{1}. $$ 

Corollary A.7. (Marginal outer bounding boxes) Suppose Assumption 1 holds and that  $ (X_{i}, Y_{i})_{i=1}^{n+1} $  is independent of the functions s and  $ b_{O} $ . Given  $ \alpha_{2} \in (0,1) $ , define

 $$ \lambda_{O}(\alpha_{2})=\inf\left\{\lambda:\frac{1}{n}\sum_{i=1}^{n}1\left[C(B_{i}^{O},-b_{O}(s(X_{i})))\leq\lambda\right]\geq\frac{\lceil(1-\alpha_{2})(n+1)\rceil}{n}\right\}. $$ 

for a suitable combination function C, and let  $ O(X) = \{v \in \mathcal{V} : C(v, -b_{O}(s(X))) \leq \lambda_{O}(\alpha_{2})\} $ . Then,

 $$ \mathbb{P}\left(\{v\in\mathcal{V}:Y_{n+1}(v)=1\}\subseteq B_{n+1}^{O}\subseteq O(X_{n+1})\right)\geq1-\alpha_{2}. $$ 

Joint results can be obtained in a similar manner to those in Section 2.3.

### A.5 WRITING THE TEST TIME STEPS AS AN ALGORITHM

In order to clarify what is done at test time we include the following algorithm which demonstrates this for the polyps data application.

Algorithm 1 Test time application of the methods (for the polyps data application)

Require: Inner and outer alpha levels $\alpha_{1}$ and $\alpha_{2}$ and thresholds $\lambda_{I}(\alpha_{1})$ and $\lambda_{O}(\alpha_{2})$ obtained as in equations (3) and (5) from the calibration dataset with $f_{I}$ the identity and $f_{O}$ the distance transformed scores. A test time observation $X_{n+1}$ and a score function $s(X_{n+1})$ obtained by an image segmenting model and a distance metric $\rho$.

1: Compute the predicted mask as $\hat{M}(X_{n+1})=\{v\in\mathcal{V}:s(X_{n+1},v)>0\}$

2: Calculate the set of points $E(\hat{M}(X_{n+1}))$ on the boundary of the predicted mask $\hat{M}(X_{n+1})$ using the marching squares algorithm.

3: Compute $d_{\rho}(\hat{M}(X_{n+1}),v)=\operatorname{sign}(\hat{M}(X_{n+1}),v)\operatorname{min}\{\rho(v,e):e\in E(\hat{M}(X_{n+1}))\}$, i.e. the distance transformed scores, for each $v\in\mathcal{V}$.

4: Let $I(X_{n+1})=\{v\in\mathcal{V}:s(X,v)>\lambda_{I}(\alpha_{1})\}$.

5: Let $O(X_{n+1})=\{v\in\mathcal{V}:-d_{\rho}(\hat{M}(X_{n+1}),v)\leq\lambda_{O}(\alpha_{2})\}$.

6: return $I(X_{n+1})$ and $O(X_{n+1})$.

For the brain imaging application,  $ \lambda_{I}(\alpha_{1}) $  and  $ \lambda_{O}(\alpha_{2}) $  are instead computed from the calibration dataset with both  $ f_{I} $  and  $ f_{O} $  being the distance transformed scores. Then line 4 of the algorithm at test time is replaced by: Let  $ I(X_{n+1}) = \{v \in \mathcal{V} : d_{\rho}(\hat{M}(X_{n+1}), v) > \lambda_{I}(\alpha_{1})\} $ . For the teeth segmentation problem the inner scores are analogously replaced but with scores smoothed using an isotropic Gaussian kernel with FWHM 2 pixels.

### A.6 ADDITIONAL SETTINGS FOR POLYPS SEGMENTATION

Here we plot additional settings and examples for the polyps data application. The version of the method which uses the distance transformation to create outer confidence sets and the untransformed logit scores to create inner confidence sets will be referred to as combo.

#### A.6.1 ADDITIONAL EXAMPLES FROM THE LEARNING DATASET

<div style="text-align: center;"><img src="imgs/img_in_image_box_250_333_976_1371.jpg" alt="Image" width="59%" /></div>


<div style="text-align: center;">Figure A8: Additional examples from the learning dataset. The layout of these figures is the same as for Figure 2.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_251_255_973_1289.jpg" alt="Image" width="58%" /></div>


<div style="text-align: center;">Figure A9: Futher examples from the learning dataset. The layout of these figures is the same as for Figure 2.</div>


<div style="text-align: center;">A.6.2 VALIDITION FIGURES FOR THE ORIGINAL AND BOUNDING BOX SCORES</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_214_207_988_1033.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure A10: Conformal confidence sets for the polyps data examples from Figure 3 for alternative scores. In each set of panels the confidence obtained from using the logit scores are shown in the middle row and those obtained from the bounding box scores are shown in the bottom row. As observed on the learning dataset the outer sets obtained when using the logit scores are very large and uninformative.</div>


BB scores Logit scores Combo Image BB scores Logit scores Combo Image

#### A.6.3 ADDITIONAL VALIDITION FIGURES

<div style="text-align: center;"><img src="imgs/img_in_image_box_214_205_989_1295.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure A11: Additional validation examples. In each example, after the original images, the rows are (from top to bottom) the combination of the original and distance transformed scores, then the logit scores and finally the bounding box scores. The interpretation of the results is the same as for Figure 3.</div>


<div style="text-align: center;">A.6.4 CONFIDENCE SETS FOR THE BOUNDING BOXES</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_215_207_986_728.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">Figure A12: Conformal confidence sets for the boundary boxes themselves using the approach introduced in Section A.4. The ground truth outer bounding boxes are shown in yellow.</div>


#### A.6.5 JOINT 90% CONFIDENCE REGIONS

<div style="text-align: center;"><img src="imgs/img_in_image_box_215_896_986_1156.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">Figure A13: Joint 90% conformal confidence sets obtained using Corollary 2.5, with  $ \alpha_{1}=0.02 $  and  $ \alpha_{2}=0.08 $ , for the polyps images in Figure 3.</div>


<div style="text-align: center;">A.6.6 MARGINAL 80 % CONFIDENCE REGIONS</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_215_207_986_467.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">Figure A14: Marginal 80% conformal confidence sets obtained for the polyps images in Figure 3.</div>


#### A.6.7 MARGINAL 95 % CONFIDENCE REGIONS

<div style="text-align: center;"><img src="imgs/img_in_image_box_215_614_986_875.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">Figure A15: Marginal 95% conformal confidence sets obtained using for the polyps images in Figure 3. These sets are also joint 90% confidence sets with equally weighted  $ \alpha_{1} = \alpha_{2} = 0.05 $ . The influence of the weighting scheme can therefore be examined by comparing to Figure A13.</div>


#### A.6.8 HISTOGRAMS OF THE COVERAGE

<div style="text-align: center;"><img src="imgs/img_in_chart_box_220_205_1001_567.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure A16: Histograms of the coverage rates obtained across each of the validation resamples for 90% inner and outer marginal confidence sets. We plot the results for the logit scores, distance transformed scores (DT) and boundary box scores (BB) from left to right. The bounding box scores are discontinuous which is the cause of the discreteness of the rightmost histograms.</div>


#### A.6.9 COMPARING THE PROPORTION

<div style="text-align: center;"><img src="imgs/img_in_chart_box_223_760_599_1050.jpg" alt="Image" width="30%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_632_761_998_1051.jpg" alt="Image" width="29%" /></div>


<div style="text-align: center;">Figure A17: Measuring the proportion of the entire image which is under/over covered by the respective confidence sets. Left: proportion of the image which lies within the true mask but outside of the inner set. Right: proportion of the image which lies within the confidence set but outside of the true mask. For both a lower proportion corresponds to increased precision.</div>


### A.7 ADDITIONAL SETTINGS FOR BRAIN MASK SEGMENTATION

#### A.7.1 COMPARING ORIGINAL AND DISTANCE TRANSFORMED SCORES ON THE LEARNING DATASET

<div style="text-align: center;"><img src="imgs/img_in_image_box_264_271_959_718.jpg" alt="Image" width="56%" /></div>


Figure A18: Learning the best transformation for brain mask segmentation. First row: original images from different subjects. Second row: confidence sets provided by calibrating the distance transformed scores on the learning dataset. Third row: confidence sets produced using the logit scores on the learning dataset. Using the logit scores produced uniformative confidence sets. Instead the distance transformation is a big improvement.

<div style="text-align: center;">A.7.2 Comparing SMOOTH TRANSFORMED SCORES ON THE LEARNING DATASET</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_264_208_960_1100.jpg" alt="Image" width="56%" /></div>


<div style="text-align: center;">Figure A19: Inner and outer sets computed by comparing smooth score transformations on the learning dataset. Scores were smoothed using an isotropic Gaussian kernel with full width at half maximum (FWHM) taking values in  $ \{5, 10, 15, 20, 25\} $  mm. The resulting inner and outer sets based on increasing levels of applied smoothness are shown from top to bottom. The performance appears to peak at 20mm.</div>


#### A.7.3 COMPARING ORIGINAL AND DISTANCE TRANSFORMED SCORES ON THE TEST DATASET

<div style="text-align: center;"><img src="imgs/img_in_image_box_264_230_959_824.jpg" alt="Image" width="56%" /></div>


Figure A20: Inner and outer confidence sets for brain mask segmentation using the distance transformed and logit scores. Top row: brain images for each subject. 2nd row: the inner and outer confidence sets produced by calibrating the distance transformed scores on the calibration dataset. 3rd row: the inner and outer confidence sets produced by calibrating the logit scores on the calibration dataset. 4th row: the inner and outer confidence sets produced by calibrating the smoothed scores (smoothed with an isotropic Gaussian kernel of 20mm - chosen for comparison here because it performed the best on the learning dataset out of the smoothing levels considered). As for the learning dataset the logit scores perform very poorly and are not able to separate the background from the segmented masks with confidence. Instead the distance transformed scores do a very good job at segmenting the mask. Indeed they do slightly better on the calibrated dataset than on the learning dataset. This occurs as the learning dataset is relatively small and does not capture the full picture. The smooth scores improve on the logit scores but do not provide as tight bounds as the distance transformed scores for neither inner nor outer sets.

#### A.7.4 COMPUTING THE COVERAGE FOR THE BRAIN IMAGING DATA

In order to study the coverage rate of the methods in the context of the brain imaging application we perform a similar validation to that described in Section 3.3 for polyps segmentation. To do so we divide the 474 subjects left, after excluding the learning dataset, into 300 calibration and 174 test images. We do this 1000 times, randomly sampling the sets of 300 and 174 images respectively and measuring the coverage in each run. We average the coverage over the 1000 runs and display the results in Figure A21. Note that unlike the box scores considered in Section 3.3, the smooth scores are not discrete so do not display discreteness issues at lower levels of coverage.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_291_378_611_710.jpg" alt="Image" width="26%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_378_932_712.jpg" alt="Image" width="26%" /></div>


<div style="text-align: center;">Figure A21: Coverage levels of the inner and outer sets averaged over 1000 validations for the original, distance transformed (DT) and smoothed scores (smoothed with a full width at half maximum of 20mm). The nominal rate is achieved in all settings considered.</div>


### A.8 ADDITIONAL SETTINGS FOR TEETH SEGMENTATION

#### A.8.1 COMPARING ORIGINAL AND DISTANCE TRANSFORMED SCORES ON THE LEARNING DATASET

<div style="text-align: center;"><img src="imgs/img_in_image_box_215_270_984_1028.jpg" alt="Image" width="62%" /></div>


Figure A22: Inner and outer confidence sets for brain mask segmentation calibrated and plotted on the learning dataset. 12 images are plotted in two sets of 6, for each set the rows are as follows. First row: original images. Second row: results of distance transformed scores - providing tight outer sets but uninformative inner sets. Third row: logit scores providing looser outer sets but more informative inner sets though these can be improved by smoothing see Figure A23.

<div style="text-align: center;">A.8.2 Comparing SMOOTH TRANSFORMED SCORES ON THE LEARNING DATASET</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_216_207_984_1210.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">Figure A23: Inner and outer sets computed by comparing smooth score transformations on the learning dataset. Scores were smoothed using an isotropic Gaussian kernel with full width at half maximum (FWHM) taking values in  $ \{2, 4, 8\} $  pixels. For each set of 6 images the resulting inner and outer sets based on increasing levels of applied smoothness are shown from top to bottom. A FWHM of 2 pixels is the best for the inner set and indeed performs better than the logit scores shown in Figure A22. Instead an increased level of smoothness is better for the outer set, attaining comparable performance to the distance transformed scores though with some additional blobs.</div>


#### A.8.3 Comparing to the Logit Scores on the test dataset

<div style="text-align: center;"><img src="imgs/img_in_image_box_216_208_983_453.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">Figure A24: Inner and outer confidence sets for brain mask segmentation performed on the test set computed by calibrating and predicting using the logit scores. The performance is less good than the score transformations optimized on the learning dataset and shown in the main text and has been included here for reference. The outer sets are larger and less precise than those based on the distance transformation. The inner sets are good but not quite as good as the ones based on a small amount of smoothing.</div>


#### A.8.4 COMPUTING THE COVERAGE FOR THE TEETH DATASET

In order to study the coverage rate of the methods in the context of the brain imaging application we perform a similar validation to that described in Sections 3.3 and A.7.4. To do so we divide the 198 subjects, into two subsets of size 99. We do this 1000 times, randomly sampling the subsets of 99 images respectively, calibrating on the first subset and measuring the coverage on the second in each run. We average the coverage over the 1000 runs and display the results in Figure A25.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_291_836_611_1168.jpg" alt="Image" width="26%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_837_930_1168.jpg" alt="Image" width="25%" /></div>


<div style="text-align: center;">Figure A25: Coverage levels of the inner and outer sets averaged over 1000 validations for the original, distance transformed (DT) and smoothed scores (smoothed with a full width at half maximum of 2 pixels). The nominal rate is achieved in all settings considered.</div>


### A.9 COMPARING PERFORMANCE METRICS FOR EACH SEGMENTATION MODEL

In the table we display performance metrics for each model, computed over the validation set used in each data application.

<div style="text-align: center;">Table 1: Performance Metrics over the validation set</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>Application</td><td style='text-align: center;'>Average Dice Score</td><td style='text-align: center;'>Average Precision</td><td style='text-align: center;'>Average Recall</td></tr><tr><td style='text-align: center;'>PraNet</td><td style='text-align: center;'>Polyps</td><td style='text-align: center;'>0.869</td><td style='text-align: center;'>0.877</td><td style='text-align: center;'>0.881</td></tr><tr><td style='text-align: center;'>HDBET</td><td style='text-align: center;'>Brain imaging</td><td style='text-align: center;'>0.976</td><td style='text-align: center;'>0.961</td><td style='text-align: center;'>0.992</td></tr><tr><td style='text-align: center;'>U-Net based GAN</td><td style='text-align: center;'>Teeth</td><td style='text-align: center;'>0.933</td><td style='text-align: center;'>0.935</td><td style='text-align: center;'>0.932</td></tr></table>

### A.10 RELATIONSHIP WITH MULTIPLE TESTING ERROR RATES

## FAMILY-WISE ERROR RATE (FWER)

In traditional multiple hypothesis testing Family-Wise Error Rate (FWER) is the probability of making at least one false discovery across a set of considered hypotheses. Given a multiple testing problem in which m hypotheses are tested and a multiple testing algorithm M, let  $ V(\mathcal{M}) $  denote the resulting set of false discoveries,  $ R(\mathcal{M}) $  the set of rejected hypotheses and T be the set of true rejections. Then the FWER is defined as:

 $$ FWER(\mathcal{M}):=\mathbb{P}(|V(\mathcal{M})|\geq1). $$ 

Then, given  $ \alpha > 0 $ , if we can guarantee that  $ FWER \leq \alpha $  then it follows that  $ R(\mathcal{M}) \subseteq T $  with probability at least  $ 1 - \alpha $ . This statement is thus analogous to the coverage guarantees which we provide in Theorems 2.1 and 2.2 in the sense that a probabilistic guarantee on the inclusion probability is provided.

## FALSE DISCOVERY RATES AND PROPORTIONS

Instead the false discovery proportion in the multiple testing setting for an algorithm M is given by:

 $$ F D P(\mathcal{M}):=\frac{\left|V(\mathcal{M})\right|}{\left|R(\mathcal{M})\right|}\cdot\mathbf{1}_{\left|R(\mathcal{M})\right|>0} $$ 

and the False Discovery Rate is given by:

 $$ FDR(\mathcal{M})=\mathbb{E}\left[FDP\right], $$ 

where  $ \mathbf{1}_{|R(\mathcal{M})|>0} $  is an indicator function that is 1 when  $ |R(\mathcal{M})|>0 $  and 0 otherwise. Controlling the FDP in probability is thus analogous to the proportion of the true mask that lies within the discovered sets, as in Bates et al. (2021) whilst controlling the FDR is instead analogous to the risk control discussed in Angelopoulos et al. (2021) in which conformal inference is used to control the expected proportion of the true mask which is discovered.