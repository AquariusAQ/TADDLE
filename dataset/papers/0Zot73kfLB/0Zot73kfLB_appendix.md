## A APPENDIX

## The appendix includes:

• Rendering equation and preliminary for vanilla 3DGS.

• Implementation details of Auxiliary Deformation Field.

• Implementation details of translation and rotation system.

• Training and evaluation resources for the model.

• Additional incremental learning experiments for self-propelled objects.

• Additional details of datasets.

• Additional quantitative results for ablation study in the main context.

• Additional ablation studies.

• Analysis of different ways for interpolation period.

• Additional quantitative results for GoPro dataset.

• Additional quantitative & qualitative results for future frame extrapolation.

• Additional qualitative results for motion segmentation.

• Additional qualitative results for extrapolation beyond dataset time span.

### A.1 PRELIMINARY FOR VANILLA 3DGS

3D Gaussian Splatting (Kerbl et al., 2023) represents a 3D scene by a set of colored 3D Gaussian kernels. Specifically, each Gaussian kernel is parameterized by a 3D position  $ P \in R^{3} $ , an orientation represented by a quaternion r, and a scaling s. By transforming the orientation r and scaling s into the rotation matrix R and scaling matrix S, a 3D covariance matrix  $ \Sigma $  can be composed as  $ \Sigma = RSS^{T}R^{T} $ . Then the Gaussian kernel can be evaluated at any location  $ x \in R^{3} $  in the 3D space:

 $$ G(\boldsymbol{x})=e^{-\frac{1}{2}(\boldsymbol{x}-\boldsymbol{P})^{T}\boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{P})}. $$ 

Besides, each Gaussian kernel has an opacity  $ \sigma $  indicating its influence in rendering, and a color c computed from spherical harmonics (SH) for view-dependent appearance.

The rendering of Gaussian kernels on the image consists of two steps. Firstly, The Gaussian kernels are projected onto the image plane, following the differentiable rasterization pipeline proposed in (Zwicker et al., 2001). The 3D position P and covariance matrix  $ \Sigma $  of each Gaussian kernel are projected into 2D position  $ P' = JWP $  and covariance matrix  $ \Sigma' = JW\Sigma W^{T}J^{T} $  respectively, where J denotes the Jacobian of the approximated projective transformation and W denotes the transformation from the world to camera coordinates. Secondly, the color of a pixel  $ \mu $  on the image can be rendered by  $ \alpha $ -blending as follows:

 $$ \boldsymbol{C}(\boldsymbol{\mu})=\sum_{i}T_{i}\alpha_{i}\boldsymbol{c}_{i},\quad T_{i}=\prod_{j=1}^{i-1}(1-\alpha_{j}), $$ 

where  $ \alpha_{i} $  is obtained by evaluating the projection of the Gaussian kernel  $ G_{i} $  on the pixel  $ \mu $ , i.e.,  $ \alpha_{i} = \sigma_{i} e^{-\frac{1}{2} (\mu - P^{\prime})^{T} \Sigma^{\prime - 1} (\mu - P^{\prime})} $ . By adjusting the parameters of Gaussian kernels mentioned above and adaptively controlling the Gaussian density, a high-fidelity representation of a 3D scene can be obtained from multi-view images. We refer readers to (Kerbl et al., 2023) for more details.

### A.2 IMPLEMENTATION DETAILS OF AUXILIARY DEFORMATION FIELD

We leverage an existing deformation field introduced in (Yang et al., 2024) as our auxiliary deformation field. In particular, the 3D position  $ x_{0} $  of each canonical Gaussian kernel and the current timestamp t are fed into an MLP-based deformation network, denoted as  $ f_{defo} $ . The implementation of this MLP is directly adapted from (Yang et al., 2024), i.e., an MLP with 8 layers in total and 256 hidden sizes for each layer, plus a ResNet layer at layer 4. At the input layer, an 8-degree

positional embedding is applied onto the 3D position  $ x_{0} $  and a 5-degree positional embedding onto time t.

Mathematically, the deformation field  $  f_{defo}(\boldsymbol{x}, t) : \mathcal{R}^{4} \to \mathcal{R}^{10}  $  is defined as

 $$ (\delta\pmb{x},\delta\pmb{r},\delta\pmb{s})=f_{d e f o}(\pmb{x}_{0},t), $$ 

where  $ \delta x $  represents the translation of the center of Gaussian kernel,  $ \delta r $  represents the rotation for the pose of Gaussian kernel in quaternion representation, and  $ \delta s $  is the difference of Gaussian sizes. Note a Gaussian size vector is parametrized by  $ \mathbf{z} = \log(\mathbf{s}) $ , so the difference can be defined as  $ \delta \mathbf{s} = \exp(\delta \mathbf{z}) $ .

After applying the deformation field onto Gaussian kernels, we can deform the Gaussian kernels from canonical time to time $t$ as:

 $$ \boldsymbol{x}_{t}=\boldsymbol{x}_{0}+\delta\boldsymbol{x} $$ 

 $$ \boldsymbol{r}_{t}=\delta\boldsymbol{r}\circ\boldsymbol{r}_{0} $$ 

 $$ \boldsymbol{s}_{t}=\exp(\boldsymbol{z}_{0}+\delta\boldsymbol{z})=\boldsymbol{s}_{0}\odot\delta\boldsymbol{s}, $$ 

where  $ \circ $  is quaternion multiplication and  $ \odot $  is element-wise multiplication.

### A.3 IMPLEMENTATION DETAILS OF TRANSLATION AND ROTATION SYSTEM

As discussed in the main context, we learn the following two groups of physical parameters for each 3D particle P as function  $ f_{\mathit{trd}}(\boldsymbol{x}):\mathcal{R}^{3}\to\mathcal{R}^{13} $ :

 $$ \{(\boldsymbol{P}_{c},\boldsymbol{v}_{c},\boldsymbol{a}_{c}),(\boldsymbol{w}_{p},\boldsymbol{\epsilon}_{p})\}=f_{t r d}(\boldsymbol{P}), $$ 

where angular velocity is defined as  $ \omega = \|w_{p}\|_{2} $  around rotation axes direction  $ \hat{k} = w_{c}/\omega $  following right-hand rule.

This module is implemented by a simple  $ 6 \times 128 $  MLPs with 6 layers in total and 128 hidden sizes for each layer. In addition, a 5-degree positional embedding is applied onto the input 3D position x, and relie is chosen as the activation functions.

### A.4 TRAINING AND EVALUATION RESOURCES FOR THE MODEL

As the complexity of different scenes varies, the total number of Gaussians learned for each scene varies from 40k to 1.6M. In general, our training time is 1.05 times longer than DefGS (or 4DGS if built on it). For example, on the bat of Dynamic Object Dataset, DefGS/4DGS need 25 minutes, while we need 27 minutes, with a slight training cost addition. Since our additional module is a tiny MLPs, we only need 367.4kB larger storage. Our rendering speed is 0.85 times slower than DefGS (or 0.8 times slower than 4DGS if built on it). For example, on the bat of Dynamic Object Dataset, they achieve 40fps and ours 32fps. We train all our models on a single NVIDIA 3090 24G GPU.

### A.5 INCREMENTAL LEARNING FOR SELF-PROPELLED OBJECTS

We include an additional incremental learning experiment to show that our framework can easily adapt to new observations when internal forces change for self-propelled objects. We choose three self-propelled objects from the Dynamic Object Dataset for this experiment. We keep the same viewing angles in training and testing split, and incrementally train the network.

To be specific, we first feed time  $ t = 0 \sim 0.15 $  to train the network, and evaluate novel view interpolation on  $ t = 0 \sim 0.15 $ , future frame extrapolation on  $ t = 0.15 \sim 0.30 $ . Next, we include  $ t = 0.15 \sim 0.30 $  to train, and evaluate novel view interpolation on  $ t = 0 \sim 0.30 $ , future frame extrapolation on  $ t = 0.30 \sim 0.45 $ . We keep adding a time interval of 0.15 till we train from  $ t = 0 \sim 0.75 $ , and extrapolate from  $ t = 0.75 \sim 0.9 $ .

We compare our performance with DefGS. Table 5 shows quantitative results. It can be seen that DefGS suffers from overfitting the previous timestamps and results in a decrease in its interpolation ability, while our model can stably adapt to new observations and achieve excellent past and future frame predictions. This means that even though the internal forces are changing for self-propelled objects, our model can easily adapt to new observations.

<div style="text-align: center;">Table 5: Quantitative results (PSNR) of incremental learning.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Interpolation</td><td style='text-align: center;'>0.15  $ \rightarrow $  0.30</td><td style='text-align: center;'>0.30  $ \rightarrow $  0.45</td><td style='text-align: center;'>0.45  $ \rightarrow $  0.60</td><td style='text-align: center;'>0.60  $ \rightarrow $  0.75</td><td style='text-align: center;'>0.75  $ \rightarrow $  0.90</td><td style='text-align: center;'>Average</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>39.386</td><td style='text-align: center;'>38.745</td><td style='text-align: center;'>35.818</td><td style='text-align: center;'>34.531</td><td style='text-align: center;'>27.904</td><td style='text-align: center;'>35.277</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>40.032</td><td style='text-align: center;'>40.706</td><td style='text-align: center;'>41.013</td><td style='text-align: center;'>40.466</td><td style='text-align: center;'>39.971</td><td style='text-align: center;'>40.438</td></tr><tr><td style='text-align: center;'>Extrapolation</td><td style='text-align: center;'>0.15  $ \rightarrow $  0.30</td><td style='text-align: center;'>0.30  $ \rightarrow $  0.45</td><td style='text-align: center;'>0.45  $ \rightarrow $  0.60</td><td style='text-align: center;'>0.60  $ \rightarrow $  0.75</td><td style='text-align: center;'>0.75  $ \rightarrow $  0.90</td><td style='text-align: center;'>Average</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>23.438</td><td style='text-align: center;'>21.360</td><td style='text-align: center;'>19.989</td><td style='text-align: center;'>19.670</td><td style='text-align: center;'>17.629</td><td style='text-align: center;'>20.417</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>29.958</td><td style='text-align: center;'>32.260</td><td style='text-align: center;'>31.384</td><td style='text-align: center;'>29.527</td><td style='text-align: center;'>28.958</td><td style='text-align: center;'>30.417</td></tr></table>

### A.6 ADDITIONAL DETAILS OF NEW DATASETS

Dynamic Multipart dataset: This dataset comprises 4 distinct objects $ ^{1} $ , including a variety of challenging motions. Details of the 4 dynamic objects are:

• Foldingchair: A folingchair is given. This chair is composed of three parts. The whole motion is unfolding this chair, so all three parts are undergoing different rotating motions.

• Hypoerbolic Slot: This is an extremely hard case, where a stick is rotating through a hypoerbolic slot. Note that, only the stick in this hyperbolic shape is dynamic, this introduces more challenges in motion extrapolation.

- Satellite: This object is a satellite with two wing doors opening and one main door opening, all rotating in different directions.

• Stove: A home stove is given. The motion is mainly closing its top cover plate.

## GoPro Dataset: This dataset includes 4 challenging real-world dynamic scenes

- Scene #1: Box. This scene contains a drawer-like box, and a person is trying to close it. The difficulty lies in a tight combination of the moving part and the static part of the box, especially in the future.

- Scene #2: Hammer. This scene contains a hammer moving on the topside of a box. The difficulty lies in the direct contact of moving objects and static objects, which requires sharp separation of diverse motion patterns in order to keep the right static/moving states in the future.

- Scene #3: Collision. This scene contains a cube and a cup moving towards each other. The difficulty is the different directions of two motions. It is hard to keep the shapes of these two objects in the future.

• Scene #4: Wrist Rest. A person is trying to bend a wrist rest. The difficulty is that the object is deformable and the motion is thus not rigid or part-wise rigid.

### A.7 ADDITIONAL QUANTITATIVE RESULTS FOR ABLATION STUDY FOR THE MAIN CONTEXT

We first elaborate how we implement the ablation study (3). Our original design of the translation and rotation dynamics system module  $  f_{trd}(\mathbf{x})  $  is only relevant to space, but not time. To obtain the corresponding physics parameters at time t, we use Equation 4 to derive the queried physics parameters. In our ablation study (3), we aim to keep the physics parameters changing over time, thus making it more complex in theory. Particularly, we use the same network architecture of  $ f_{trd} $ , except changing the input from  $  f_{trd}(\mathbf{x})  $  to  $  f_{trd}(\mathbf{x}, t)  $ , to force the change of physics parameters.

Here we show the total results for the ablation study in Table 6, both for interpolation and extrapolation.

 $ ^{1} $ All objects are purchased from SketchFab, licensed under the SketchFab Standard License: https://sketchfab.com/licenses, and are all allowed for AI generation model usage

<div style="text-align: center;">Table 6: Quantitative results of ablation studies on Dynamic Multipart dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td rowspan="2">$ f_{defo} $</td><td rowspan="2">$ f_{trd} $</td><td colspan="3">Interpolation</td><td colspan="3">Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td></tr><tr><td style='text-align: center;'>(1)  $ \delta t $</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>35.128</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>29.441</td><td style='text-align: center;'>0.984</td><td style='text-align: center;'>0.013</td></tr><tr><td style='text-align: center;'>(1)  $ 2\delta t $</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>34.807</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>30.721</td><td style='text-align: center;'>0.986</td><td style='text-align: center;'>0.012</td></tr><tr><td style='text-align: center;'>(1)  $ 3\delta t $</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>35.223</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>30.246</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.012</td></tr><tr><td style='text-align: center;'>(2) -</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>32.266</td><td style='text-align: center;'>0.987</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>27.081</td><td style='text-align: center;'>0.981</td><td style='text-align: center;'>0.018</td></tr><tr><td style='text-align: center;'>(3)  $ 2\delta t $</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>35.225</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>29.986</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.012</td></tr></table>

### A.8 ADDITIONAL ABLATION STUDIES

Since the range of motion between two consecutive frames is different across different datasets, we conduct extensive ablations about different choices of  $ \Delta t $  on all 4 datasets, and the results are listed in Table 7. We observe that  $ 3\delta t $  works better in extrapolation on three datasets (Dynamic Object/Dynamic Indoor Scene/NVIDIA Dynamic Scenes). The basic rule to select an appropriate  $ \delta t $  is based on the motion range. If the motion changes fast, so the motion between two consecutive frames is apparent enough, then a smaller  $ \delta t $  is good enough. Otherwise, if the motion is rather slow, then a larger  $ \delta t $  is preferred.

<div style="text-align: center;">Table 7: Quantitative results of ablation studies for  $ \delta t $  on all four datasets.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3"></td><td colspan="6">Dynamic Multipart Dataset</td><td colspan="6">Dynamic Object Dataset</td></tr><tr><td colspan="6">Interpolation</td><td colspan="6">Interpolation</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td></tr><tr><td style='text-align: center;'>$ \delta t $</td><td style='text-align: center;'>35.128</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>29.441</td><td style='text-align: center;'>0.984</td><td style='text-align: center;'>0.013</td><td style='text-align: center;'>38.929</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>28.506</td><td style='text-align: center;'>0.981</td><td style='text-align: center;'>0.013</td></tr><tr><td style='text-align: center;'>2 $ \delta t $</td><td style='text-align: center;'>34.807</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>30.721</td><td style='text-align: center;'>0.986</td><td style='text-align: center;'>0.012</td><td style='text-align: center;'>38.788</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.006</td><td style='text-align: center;'>28.758</td><td style='text-align: center;'>0.982</td><td style='text-align: center;'>0.011</td></tr><tr><td style='text-align: center;'>3 $ \delta t $</td><td style='text-align: center;'>35.223</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>30.246</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.012</td><td style='text-align: center;'>38.693</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.006</td><td style='text-align: center;'>29.414</td><td style='text-align: center;'>0.983</td><td style='text-align: center;'>0.012</td></tr><tr><td colspan="7">Dynamic Indoor Scene Dataset</td><td colspan="6">NVIDIA Dynamic Scenes Dataset</td></tr><tr><td rowspan="2"></td><td colspan="6">Interpolation</td><td colspan="6">Interpolation</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td></tr><tr><td style='text-align: center;'>$ \delta t $</td><td style='text-align: center;'>32.179</td><td style='text-align: center;'>0.929</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>34.387</td><td style='text-align: center;'>0.964</td><td style='text-align: center;'>0.046</td><td style='text-align: center;'>26.823</td><td style='text-align: center;'>0.891</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>28.781</td><td style='text-align: center;'>0.934</td><td style='text-align: center;'>0.070</td></tr><tr><td style='text-align: center;'>2 $ \delta t $</td><td style='text-align: center;'>32.202</td><td style='text-align: center;'>0.928</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>34.556</td><td style='text-align: center;'>0.964</td><td style='text-align: center;'>0.046</td><td style='text-align: center;'>26.943</td><td style='text-align: center;'>0.891</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>29.388</td><td style='text-align: center;'>0.938</td><td style='text-align: center;'>0.067</td></tr><tr><td style='text-align: center;'>3 $ \delta t $</td><td style='text-align: center;'>32.296</td><td style='text-align: center;'>0.928</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>35.242</td><td style='text-align: center;'>0.967</td><td style='text-align: center;'>0.045</td><td style='text-align: center;'>27.099</td><td style='text-align: center;'>0.890</td><td style='text-align: center;'>0.103</td><td style='text-align: center;'>29.440</td><td style='text-align: center;'>0.938</td><td style='text-align: center;'>0.067</td></tr></table>

The updating scheme of our translation rotation dynamics system is chosen as a second-order relationship in Equation 4, i.e., each rigid particle can have a constant acceleration. We also evaluate the third-order scheme (acceleration of acceleration) and first-order scheme (no acceleration) on Dynamic Multipart Dataset and Dynamic Object Dataset. Table 8 shows the results. We can see that, in Dynamic Object Dataset which has several self-propelled objects whose internal forces tend to change over time, not surprisingly, the third-order variant performs better. Nevertheless, due to the inherent over-parametrization, the third-order scheme tends to learn excessive rotation information to represent simple acceleration motions, thus incurring inferior performance on the Dynamic Multipart Dataset which does not have self-propelled objects.

<div style="text-align: center;">Table 8: Quantitative results of ablation studies about 3 orders of Taylor expansion in Equation 4 on Dynamic Multipart Dataset and Dynamic Object Dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3"></td><td colspan="6">Dynamic Multipart Dataset</td><td colspan="6">Dynamic Object Dataset</td></tr><tr><td colspan="3">Interpolation</td><td colspan="3">Extrapolation</td><td colspan="3">Interpolation</td><td colspan="3">Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td></tr><tr><td style='text-align: center;'>$ 1^{st} $ -order</td><td style='text-align: center;'>34.776</td><td style='text-align: center;'>0.990</td><td style='text-align: center;'>0.013</td><td style='text-align: center;'>26.729</td><td style='text-align: center;'>0.976</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>38.892</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>28.536</td><td style='text-align: center;'>0.983</td><td style='text-align: center;'>0.012</td></tr><tr><td style='text-align: center;'>$ 2^{nd} $ -order</td><td style='text-align: center;'>34.807</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>30.721</td><td style='text-align: center;'>0.986</td><td style='text-align: center;'>0.012</td><td style='text-align: center;'>38.788</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.006</td><td style='text-align: center;'>28.758</td><td style='text-align: center;'>0.982</td><td style='text-align: center;'>0.011</td></tr><tr><td style='text-align: center;'>$ 3^{rd} $ -order</td><td style='text-align: center;'>35.268</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.012</td><td style='text-align: center;'>30.503</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.013</td><td style='text-align: center;'>39.164</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>29.378</td><td style='text-align: center;'>0.983</td><td style='text-align: center;'>0.011</td></tr></table>

### A.9 ANALYSIS OF DIFFERENT WAYS FOR INTERPOLATION PERIOD

Our method has three possible strategies for interpolation rendering: (1) directly using  $ f_{defo} $  to predict the deformation at the given time t, (2) progressively calculating the Gaussian deformation at the given time t from time 0 using the motion parameters predicted by  $ f_{trd} $ , or (3) following the steps described in Section 3.3. We choose the third one in our main experiments. Here, we evaluate the other two strategies in Table 9. We can see that, the first and the third strategies are not strictly consistent, but achieve very similar performance. However, for the second strategy, the performance clearly decreases, we hypothesize that this is due to the accumulated errors in the autoregressive process.

<div style="text-align: center;">Table 9: Quantitative results of different interpolation strategies of our method on all four datasets.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td colspan="3">Multipart</td><td colspan="3">Object</td><td colspan="3">Indoor</td><td colspan="3">NVIDIA</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td></tr><tr><td style='text-align: center;'>(1)  $ f_{defo} $</td><td style='text-align: center;'>35.040</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>38.406</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>32.569</td><td style='text-align: center;'>0.930</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>26.951</td><td style='text-align: center;'>0.891</td><td style='text-align: center;'>0.102</td></tr><tr><td style='text-align: center;'>(2)  $ f_{trd} $</td><td style='text-align: center;'>30.310</td><td style='text-align: center;'>0.984</td><td style='text-align: center;'>0.017</td><td style='text-align: center;'>33.527</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.009</td><td style='text-align: center;'>31.776</td><td style='text-align: center;'>0.926</td><td style='text-align: center;'>0.092</td><td style='text-align: center;'>25.899</td><td style='text-align: center;'>0.875</td><td style='text-align: center;'>0.118</td></tr><tr><td style='text-align: center;'>(3)  $ f_{defo} + f_{trd} $</td><td style='text-align: center;'>34.807</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>38.788</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.006</td><td style='text-align: center;'>32.202</td><td style='text-align: center;'>0.928</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>26.943</td><td style='text-align: center;'>0.891</td><td style='text-align: center;'>0.102</td></tr></table>

### A.10 ADDITIONAL QUANTITATIVE RESULTS ON DYNAMIC OBJECT DATASET

Here we show the total per-scene results on Dynamic Object Dataset in Table 10 and qualitative results in Figures 8, 9&10, both for interpolation and extrapolation.

### A.11 ADDITIONAL QUANTITATIVE RESULTS ON DYNAMIC INDOOR SCENE DATASET

Here we show the total per-scene results in Dynamic Indoor Scene Datasets in Table 11 and qualitative results in Figures 10&11, both for interpolation and extrapolation.

### A.12 ADDITIONAL QUANTITATIVE RESULTS ON NVIDIA DYNAMIC SCENE DATASET

Here we show the total per-scene results on NVIDIA Dynamic Scene Dataset in Table 12 and qualitative results in Figure 13, both for interpolation and extrapolation.

### A.13 ADDITIONAL QUANTITATIVE RESULTS ON DYNAMIC MULTIPART DATASET

Here we show the total per-scene results on our Dynamic Multipart Dataset in Table 13 and qualitative results in Figure 12, both for interpolation and extrapolation.

### A.14 ADDITIONAL QUANTITATIVE & QUALITATIVE RESULTS FOR GOPro DATASET

Here we show the total results on our GoPro Dataset in Table 14 and qualitative results in Figures 14,15,&16, both for interpolation and extrapolation.

### A.15 ADDITIONAL QUALITATIVE RESULTS FOR EXTRAPOLATION BEYOND DATASET TIME SPANS

We list some meaningful longer extrapolation results from each dataset here in Figure 21. In our dataset, the training period lasts from t = 0 to t = 0.75 and the extrapolation period lasts from t = 0.75 to t = 1.0. Here we show the qualitative results till t = 1.5, which is already twice the training period. We can see that our method can still obtain physically meaningful future frame prediction in particularly high quality.

### A.16 ADDITIONAL QUALITATIVE RESULTS FOR OBJECT/PART SEGMENTATION

Figures 5, 6 and 7 show more qualitative results for the autonomous object or part segmentation based on the learned physical parameters via the simple K-means clustering algorithm.

<div style="text-align: center;">Table 10: Per-scene quantitative results on Dynamic Object dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3">Methods</td><td colspan="5">Falling Ball</td><td colspan="5">Bat</td></tr><tr><td colspan="3">Interpolation</td><td colspan="2">Extrapolation</td><td colspan="2">Interpolation</td><td colspan="3">Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>SSIM↑</td></tr><tr><td style='text-align: center;'>T-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>14.921</td><td style='text-align: center;'>0.782</td><td style='text-align: center;'>0.326</td><td style='text-align: center;'>15.418</td><td style='text-align: center;'>0.793</td><td style='text-align: center;'>0.308</td><td style='text-align: center;'>13.070</td><td style='text-align: center;'>0.836</td><td style='text-align: center;'>0.234</td><td style='text-align: center;'>13.897</td></tr><tr><td style='text-align: center;'>D-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>15.548</td><td style='text-align: center;'>0.665</td><td style='text-align: center;'>0.435</td><td style='text-align: center;'>15.116</td><td style='text-align: center;'>0.644</td><td style='text-align: center;'>0.427</td><td style='text-align: center;'>14.087</td><td style='text-align: center;'>0.845</td><td style='text-align: center;'>0.212</td><td style='text-align: center;'>15.406</td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>35.458</td><td style='text-align: center;'>0.974</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>20.242</td><td style='text-align: center;'>0.959</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>16.080</td><td style='text-align: center;'>0.908</td><td style='text-align: center;'>0.108</td><td style='text-align: center;'>16.952</td></tr><tr><td style='text-align: center;'>T-NeRF $ _{PINN} $</td><td style='text-align: center;'>17.687</td><td style='text-align: center;'>0.775</td><td style='text-align: center;'>0.368</td><td style='text-align: center;'>17.857</td><td style='text-align: center;'>0.829</td><td style='text-align: center;'>0.265</td><td style='text-align: center;'>16.412</td><td style='text-align: center;'>0.903</td><td style='text-align: center;'>0.197</td><td style='text-align: center;'>18.983</td></tr><tr><td style='text-align: center;'>HexPlane $ _{PINN} $</td><td style='text-align: center;'>32.144</td><td style='text-align: center;'>0.965</td><td style='text-align: center;'>0.065</td><td style='text-align: center;'>20.762</td><td style='text-align: center;'>0.951</td><td style='text-align: center;'>0.081</td><td style='text-align: center;'>23.399</td><td style='text-align: center;'>0.958</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>21.144</td></tr><tr><td style='text-align: center;'>NVFi(Li et al., 2023a)</td><td style='text-align: center;'>35.826</td><td style='text-align: center;'>0.978</td><td style='text-align: center;'>0.041</td><td style='text-align: center;'>31.369</td><td style='text-align: center;'>0.978</td><td style='text-align: center;'>0.041</td><td style='text-align: center;'>23.325</td><td style='text-align: center;'>0.964</td><td style='text-align: center;'>0.046</td><td style='text-align: center;'>25.015</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>37.535</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.009</td><td style='text-align: center;'>20.442</td><td style='text-align: center;'>0.976</td><td style='text-align: center;'>0.033</td><td style='text-align: center;'>38.750</td><td style='text-align: center;'>0.997</td><td style='text-align: center;'>0.004</td><td style='text-align: center;'>17.063</td></tr><tr><td style='text-align: center;'>DefGS $ _{NVF_i} $</td><td style='text-align: center;'>38.606</td><td style='text-align: center;'>0.996</td><td style='text-align: center;'>0.010</td><td style='text-align: center;'>24.873</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.015</td><td style='text-align: center;'>38.075</td><td style='text-align: center;'>0.997</td><td style='text-align: center;'>0.004</td><td style='text-align: center;'>28.950</td></tr><tr><td style='text-align: center;'>GVFi(Ours)</td><td style='text-align: center;'>38.071</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.008</td><td style='text-align: center;'>35.949</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.004</td><td style='text-align: center;'>39.626</td><td style='text-align: center;'>0.997</td><td style='text-align: center;'>0.003</td><td style='text-align: center;'>24.352</td></tr><tr><td rowspan="3">Methods</td><td colspan="5">Fan</td><td colspan="5">Telescope</td></tr><tr><td colspan="3">Interpolation</td><td colspan="2">Extrapolation</td><td colspan="2">Interpolation</td><td colspan="3">Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>SSIM↑</td></tr><tr><td style='text-align: center;'>T-NeRFPumarola et al. (2021)</td><td style='text-align: center;'>8.001</td><td style='text-align: center;'>0.308</td><td style='text-align: center;'>0.646</td><td style='text-align: center;'>8.494</td><td style='text-align: center;'>0.392</td><td style='text-align: center;'>0.593</td><td style='text-align: center;'>13.031</td><td style='text-align: center;'>0.615</td><td style='text-align: center;'>0.472</td><td style='text-align: center;'>13.892</td></tr><tr><td style='text-align: center;'>D-NeRFPumarola et al. (2021)</td><td style='text-align: center;'>7.915</td><td style='text-align: center;'>0.262</td><td style='text-align: center;'>0.690</td><td style='text-align: center;'>8.624</td><td style='text-align: center;'>0.370</td><td style='text-align: center;'>0.623</td><td style='text-align: center;'>13.295</td><td style='text-align: center;'>0.609</td><td style='text-align: center;'>0.469</td><td style='text-align: center;'>14.967</td></tr><tr><td style='text-align: center;'>TiNeuVoxFang et al. (2022)</td><td style='text-align: center;'>24.088</td><td style='text-align: center;'>0.930</td><td style='text-align: center;'>0.104</td><td style='text-align: center;'>20.932</td><td style='text-align: center;'>0.935</td><td style='text-align: center;'>0.078</td><td style='text-align: center;'>31.666</td><td style='text-align: center;'>0.982</td><td style='text-align: center;'>0.041</td><td style='text-align: center;'>20.456</td></tr><tr><td style='text-align: center;'>T-NeRF $ _{PINN} $</td><td style='text-align: center;'>9.233</td><td style='text-align: center;'>0.541</td><td style='text-align: center;'>0.508</td><td style='text-align: center;'>9.828</td><td style='text-align: center;'>0.606</td><td style='text-align: center;'>0.443</td><td style='text-align: center;'>14.293</td><td style='text-align: center;'>0.739</td><td style='text-align: center;'>0.366</td><td style='text-align: center;'>15.752</td></tr><tr><td style='text-align: center;'>HexPlane $ _{PINN} $</td><td style='text-align: center;'>22.822</td><td style='text-align: center;'>0.921</td><td style='text-align: center;'>0.079</td><td style='text-align: center;'>19.724</td><td style='text-align: center;'>0.919</td><td style='text-align: center;'>0.080</td><td style='text-align: center;'>25.381</td><td style='text-align: center;'>0.948</td><td style='text-align: center;'>0.066</td><td style='text-align: center;'>23.165</td></tr><tr><td style='text-align: center;'>NVFiLi et al. (2023a)</td><td style='text-align: center;'>25.213</td><td style='text-align: center;'>0.948</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>27.172</td><td style='text-align: center;'>0.963</td><td style='text-align: center;'>0.037</td><td style='text-align: center;'>26.487</td><td style='text-align: center;'>0.959</td><td style='text-align: center;'>0.048</td><td style='text-align: center;'>27.101</td></tr><tr><td style='text-align: center;'>DefGSYang et al. (2024)</td><td style='text-align: center;'>35.858</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.017</td><td style='text-align: center;'>20.932</td><td style='text-align: center;'>0.948</td><td style='text-align: center;'>0.038</td><td style='text-align: center;'>37.502</td><td style='text-align: center;'>0.996</td><td style='text-align: center;'>0.003</td><td style='text-align: center;'>20.684</td></tr><tr><td style='text-align: center;'>DefGS $ _{NVF_i} $</td><td style='text-align: center;'>35.217</td><td style='text-align: center;'>0.984</td><td style='text-align: center;'>0.019</td><td style='text-align: center;'>26.648</td><td style='text-align: center;'>0.972</td><td style='text-align: center;'>0.023</td><td style='text-align: center;'>37.568</td><td style='text-align: center;'>0.996</td><td style='text-align: center;'>0.003</td><td style='text-align: center;'>34.096</td></tr><tr><td style='text-align: center;'>GVFi(Ours)</td><td style='text-align: center;'>35.577</td><td style='text-align: center;'>0.986</td><td style='text-align: center;'>0.013</td><td style='text-align: center;'>29.533</td><td style='text-align: center;'>0.979</td><td style='text-align: center;'>0.012</td><td style='text-align: center;'>40.614</td><td style='text-align: center;'>0.998</td><td style='text-align: center;'>0.002</td><td style='text-align: center;'>29.744</td></tr><tr><td rowspan="3">Methods</td><td colspan="5">Shark</td><td colspan="5">Whale</td></tr><tr><td colspan="3">Interpolation</td><td colspan="2">Extrapolation</td><td colspan="2">Interpolation</td><td colspan="3">Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>PSNR↑</td><td style='text-align: center;'>SSIM↑</td><td style='text-align: center;'>LPIPS↓</td><td style='text-align: center;'>SSIM↑</td></tr><tr><td style='text-align: center;'>T-NeRFPumarola et al. (2021)</td><td style='text-align: center;'>13.813</td><td style='text-align: center;'>0.853</td><td style='text-align: center;'>0.223</td><td style='text-align: center;'>15.325</td><td style='text-align: center;'>0.882</td><td style='text-align: center;'>0.193</td><td style='text-align: center;'>16.141</td><td style='text-align: center;'>0.860</td><td style='text-align: center;'>0.212</td><td style='text-align: center;'>15.880</td></tr><tr><td style='text-align: center;'>D-NeRFPumarola et al. (2021)</td><td style='text-align: center;'>17.727</td><td style='text-align: center;'>0.903</td><td style='text-align: center;'>0.150</td><td style='text-align: center;'>19.078</td><td style='text-align: center;'>0.936</td><td style='text-align: center;'>0.092</td><td style='text-align: center;'>16.373</td><td style='text-align: center;'>0.898</td><td style='text-align: center;'>0.154</td><td style='text-align: center;'>14.771</td></tr><tr><td style='text-align: center;'>TiNeuVoxFang et al. (2022)</td><td style='text-align: center;'>23.178</td><td style='text-align: center;'>0.971</td><td style='text-align: center;'>0.059</td><td style='text-align: center;'>19.463</td><td style='text-align: center;'>0.950</td><td style='text-align: center;'>0.050</td><td style='text-align: center;'>37.455</td><td style='text-align: center;'>0.994</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>19.624</td></tr><tr><td style='text-align: center;'>T-NeRF $ _{PINN} $</td><td style='text-align: center;'>17.315</td><td style='text-align: center;'>0.878</td><td style='text-align: center;'>0.177</td><td style='text-align: center;'>18.739</td><td style='text-align: center;'>0.921</td><td style='text-align: center;'>0.115</td><td style='text-align: center;'>16.778</td><td style='text-align: center;'>0.927</td><td style='text-align: center;'>0.141</td><td style='text-align: center;'>15.974</td></tr><tr><td style='text-align: center;'>HexPlane $ _{PINN} $</td><td style='text-align: center;'>28.874</td><td style='text-align: center;'>0.976</td><td style='text-align: center;'>0.040</td><td style='text-align: center;'>22.330</td><td style='text-align: center;'>0.961</td><td style='text-align: center;'>0.047</td><td style='text-align: center;'>29.634</td><td style='text-align: center;'>0.981</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>21.391</td></tr><tr><td style='text-align: center;'>NVFiLi et al. (2023a)</td><td style='text-align: center;'>32.072</td><td style='text-align: center;'>0.984</td><td style='text-align: center;'>0.024</td><td style='text-align: center;'>28.874</td><td style='text-align: center;'>0.982</td><td style='text-align: center;'>0.021</td><td style='text-align: center;'>31.240</td><td style='text-align: center;'>0.986</td><td style='text-align: center;'>0.025</td><td style='text-align: center;'>26.032</td></tr><tr><td style='text-align: center;'>DefGSYang et al. (2024)</td><td style='text-align: center;'>37.802</td><td style='text-align: center;'>0.994</td><td style='text-align: center;'>0.006</td><td style='text-align: center;'>19.924</td><td style='text-align: center;'>0.957</td><td style='text-align: center;'>0.034</td><td style='text-align: center;'>39.740</td><td style='text-align: center;'>0.997</td><td style='text-align: center;'>0.004</td><td style='text-align: center;'>20.048</td></tr><tr><td style='text-align: center;'>DefGS $ _{NVF_i} $</td><td style='text-align: center;'>37.327</td><td style='text-align: center;'>0.994</td><td style='text-align: center;'>0.006</td><td style='text-align: center;'>29.240</td><td style='text-align: center;'>0.987</td><td style='text-align: center;'>0.007</td><td style='text-align: center;'>37.101</td><td style='text-align: center;'>0.996</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>28.686</td></tr><tr><td style='text-align: center;'>GVFi(Ours)</td><td style='text-align: center;'>40.464</td><td style='text-align: center;'>0.997</td><td style='text-align: center;'>0.004</td><td style='text-align: center;'>26.680</td><td style='text-align: center;'>0.979</td><td style='text-align: center;'>0.009</td><td style='text-align: center;'>38.376</td><td style='text-align: center;'>0.997</td><td style='text-align: center;'>0.003</td><td style='text-align: center;'>26.288</td></tr></table>

<div style="text-align: center;">Table 11: Per-scene quantitative results on Dynamic Indoor Scene dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3">Methods</td><td colspan="5">Gnome House</td><td colspan="5">Chessboard</td><td style='text-align: center;'></td></tr><tr><td colspan="3">Interpolation</td><td colspan="3">Extrapolation</td><td colspan="3">Interpolation</td><td style='text-align: center;'>Extrapolation</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>T-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>26.094</td><td style='text-align: center;'>0.716</td><td style='text-align: center;'>0.383</td><td style='text-align: center;'>23.485</td><td style='text-align: center;'>0.643</td><td style='text-align: center;'>0.419</td><td style='text-align: center;'>25.517</td><td style='text-align: center;'>0.796</td><td style='text-align: center;'>0.294</td><td style='text-align: center;'>20.228</td><td style='text-align: center;'>0.708</td></tr><tr><td style='text-align: center;'>D-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>27.000</td><td style='text-align: center;'>0.745</td><td style='text-align: center;'>0.319</td><td style='text-align: center;'>21.714</td><td style='text-align: center;'>0.641</td><td style='text-align: center;'>0.367</td><td style='text-align: center;'>24.852</td><td style='text-align: center;'>0.774</td><td style='text-align: center;'>0.308</td><td style='text-align: center;'>19.455</td><td style='text-align: center;'>0.675</td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>30.646</td><td style='text-align: center;'>0.831</td><td style='text-align: center;'>0.253</td><td style='text-align: center;'>21.418</td><td style='text-align: center;'>0.699</td><td style='text-align: center;'>0.326</td><td style='text-align: center;'>33.001</td><td style='text-align: center;'>0.917</td><td style='text-align: center;'>0.177</td><td style='text-align: center;'>19.718</td><td style='text-align: center;'>0.765</td></tr><tr><td style='text-align: center;'>T-NeRF $ _{PINN} $</td><td style='text-align: center;'>15.008</td><td style='text-align: center;'>0.375</td><td style='text-align: center;'>0.668</td><td style='text-align: center;'>16.200</td><td style='text-align: center;'>0.409</td><td style='text-align: center;'>0.651</td><td style='text-align: center;'>16.549</td><td style='text-align: center;'>0.457</td><td style='text-align: center;'>0.621</td><td style='text-align: center;'>17.197</td><td style='text-align: center;'>0.472</td></tr><tr><td style='text-align: center;'>HexPlane $ _{PINN} $</td><td style='text-align: center;'>23.764</td><td style='text-align: center;'>0.658</td><td style='text-align: center;'>0.510</td><td style='text-align: center;'>22.867</td><td style='text-align: center;'>0.658</td><td style='text-align: center;'>0.510</td><td style='text-align: center;'>24.605</td><td style='text-align: center;'>0.778</td><td style='text-align: center;'>0.412</td><td style='text-align: center;'>21.518</td><td style='text-align: center;'>0.748</td></tr><tr><td style='text-align: center;'>NSFF(Li et al., 2021)</td><td style='text-align: center;'>31.418</td><td style='text-align: center;'>0.821</td><td style='text-align: center;'>0.294</td><td style='text-align: center;'>25.892</td><td style='text-align: center;'>0.750</td><td style='text-align: center;'>0.327</td><td style='text-align: center;'>32.514</td><td style='text-align: center;'>0.810</td><td style='text-align: center;'>0.201</td><td style='text-align: center;'>21.501</td><td style='text-align: center;'>0.805</td></tr><tr><td style='text-align: center;'>NVFi(Li et al., 2023a)</td><td style='text-align: center;'>30.667</td><td style='text-align: center;'>0.824</td><td style='text-align: center;'>0.277</td><td style='text-align: center;'>30.408</td><td style='text-align: center;'>0.826</td><td style='text-align: center;'>0.273</td><td style='text-align: center;'>30.394</td><td style='text-align: center;'>0.888</td><td style='text-align: center;'>0.215</td><td style='text-align: center;'>27.840</td><td style='text-align: center;'>0.872</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>32.041</td><td style='text-align: center;'>0.918</td><td style='text-align: center;'>0.132</td><td style='text-align: center;'>21.703</td><td style='text-align: center;'>0.775</td><td style='text-align: center;'>0.207</td><td style='text-align: center;'>27.355</td><td style='text-align: center;'>0.912</td><td style='text-align: center;'>0.147</td><td style='text-align: center;'>20.032</td><td style='text-align: center;'>0.808</td></tr><tr><td style='text-align: center;'>DefGS $ _{NVF} $</td><td style='text-align: center;'>32.881</td><td style='text-align: center;'>0.919</td><td style='text-align: center;'>0.132</td><td style='text-align: center;'>33.630</td><td style='text-align: center;'>0.953</td><td style='text-align: center;'>0.077</td><td style='text-align: center;'>26.200</td><td style='text-align: center;'>0.907</td><td style='text-align: center;'>0.156</td><td style='text-align: center;'>26.730</td><td style='text-align: center;'>0.917</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>32.698</td><td style='text-align: center;'>0.921</td><td style='text-align: center;'>0.101</td><td style='text-align: center;'>36.578</td><td style='text-align: center;'>0.962</td><td style='text-align: center;'>0.055</td><td style='text-align: center;'>35.138</td><td style='text-align: center;'>0.960</td><td style='text-align: center;'>0.060</td><td style='text-align: center;'>33.685</td><td style='text-align: center;'>0.966</td></tr><tr><td colspan="7">Factory</td><td colspan="4">Dining Table</td><td style='text-align: center;'></td></tr><tr><td rowspan="2">Methods</td><td colspan="3">Interpolation</td><td colspan="3">Extrapolation</td><td colspan="3">Interpolation</td><td style='text-align: center;'>Extrapolation</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>T-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>26.467</td><td style='text-align: center;'>0.741</td><td style='text-align: center;'>0.328</td><td style='text-align: center;'>24.276</td><td style='text-align: center;'>0.722</td><td style='text-align: center;'>0.344</td><td style='text-align: center;'>21.699</td><td style='text-align: center;'>0.716</td><td style='text-align: center;'>0.338</td><td style='text-align: center;'>20.977</td><td style='text-align: center;'>0.725</td></tr><tr><td style='text-align: center;'>D-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>28.818</td><td style='text-align: center;'>0.818</td><td style='text-align: center;'>0.252</td><td style='text-align: center;'>22.959</td><td style='text-align: center;'>0.746</td><td style='text-align: center;'>0.303</td><td style='text-align: center;'>20.851</td><td style='text-align: center;'>0.725</td><td style='text-align: center;'>0.319</td><td style='text-align: center;'>19.035</td><td style='text-align: center;'>0.705</td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>32.684</td><td style='text-align: center;'>0.909</td><td style='text-align: center;'>0.148</td><td style='text-align: center;'>22.622</td><td style='text-align: center;'>0.810</td><td style='text-align: center;'>0.229</td><td style='text-align: center;'>23.596</td><td style='text-align: center;'>0.798</td><td style='text-align: center;'>0.274</td><td style='text-align: center;'>20.357</td><td style='text-align: center;'>0.804</td></tr><tr><td style='text-align: center;'>T-NeRF $ _{PINN} $</td><td style='text-align: center;'>16.634</td><td style='text-align: center;'>0.446</td><td style='text-align: center;'>0.624</td><td style='text-align: center;'>17.546</td><td style='text-align: center;'>0.480</td><td style='text-align: center;'>0.609</td><td style='text-align: center;'>16.807</td><td style='text-align: center;'>0.486</td><td style='text-align: center;'>0.640</td><td style='text-align: center;'>18.215</td><td style='text-align: center;'>0.548</td></tr><tr><td style='text-align: center;'>HexPlane $ _{PINN} $</td><td style='text-align: center;'>27.200</td><td style='text-align: center;'>0.826</td><td style='text-align: center;'>0.283</td><td style='text-align: center;'>24.998</td><td style='text-align: center;'>0.792</td><td style='text-align: center;'>0.312</td><td style='text-align: center;'>25.291</td><td style='text-align: center;'>0.788</td><td style='text-align: center;'>0.350</td><td style='text-align: center;'>22.979</td><td style='text-align: center;'>0.771</td></tr><tr><td style='text-align: center;'>NSFF(Li et al., 2021)</td><td style='text-align: center;'>33.975</td><td style='text-align: center;'>0.919</td><td style='text-align: center;'>0.152</td><td style='text-align: center;'>26.647</td><td style='text-align: center;'>0.855</td><td style='text-align: center;'>0.196</td><td style='text-align: center;'>19.552</td><td style='text-align: center;'>0.665</td><td style='text-align: center;'>0.464</td><td style='text-align: center;'>22.612</td><td style='text-align: center;'>0.770</td></tr><tr><td style='text-align: center;'>NVFiLi et al. (2023a)</td><td style='text-align: center;'>32.460</td><td style='text-align: center;'>0.912</td><td style='text-align: center;'>0.151</td><td style='text-align: center;'>31.719</td><td style='text-align: center;'>0.908</td><td style='text-align: center;'>0.154</td><td style='text-align: center;'>29.179</td><td style='text-align: center;'>0.885</td><td style='text-align: center;'>0.199</td><td style='text-align: center;'>29.011</td><td style='text-align: center;'>0.898</td></tr><tr><td style='text-align: center;'>DefGSYang et al. (2024)</td><td style='text-align: center;'>33.629</td><td style='text-align: center;'>0.943</td><td style='text-align: center;'>0.096</td><td style='text-align: center;'>22.820</td><td style='text-align: center;'>0.839</td><td style='text-align: center;'>0.169</td><td style='text-align: center;'>27.680</td><td style='text-align: center;'>0.890</td><td style='text-align: center;'>0.145</td><td style='text-align: center;'>20.965</td><td style='text-align: center;'>0.855</td></tr><tr><td style='text-align: center;'>DeiGS $ _{NVF} $</td><td style='text-align: center;'>33.643</td><td style='text-align: center;'>0.943</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>33.049</td><td style='text-align: center;'>0.954</td><td style='text-align: center;'>0.062</td><td style='text-align: center;'>27.957</td><td style='text-align: center;'>0.891</td><td style='text-align: center;'>0.145</td><td style='text-align: center;'>30.975</td><td style='text-align: center;'>0.955</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>33.423</td><td style='text-align: center;'>0.941</td><td style='text-align: center;'>0.076</td><td style='text-align: center;'>34.906</td><td style='text-align: center;'>0.963</td><td style='text-align: center;'>0.045</td><td style='text-align: center;'>27.547</td><td style='text-align: center;'>0.891</td><td style='text-align: center;'>0.118</td><td style='text-align: center;'>33.056</td><td style='text-align: center;'>0.965</td></tr></table>

### A.17 ADDITIONAL QUALITATIVE RESULTS FOR SEGMENTATION ON DYNAMIC INDOOR SCENE DATASET

## Figures 17, 18, 19, &20 shows qualitative results for the rendered mask on Dynamic Indoor Scene dataset

<div style="text-align: center;">Table 12: Quantitative results of our method and baselines on the NVIDIA Dynamic Scene dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3"></td><td colspan="5">Truck</td><td colspan="5">Skating</td></tr><tr><td colspan="3">Interpolation</td><td colspan="2">Extrapolation</td><td colspan="2">Interpolation</td><td colspan="3">Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td></tr><tr><td style='text-align: center;'>T-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>18.673</td><td style='text-align: center;'>0.548</td><td style='text-align: center;'>0.447</td><td style='text-align: center;'>18.176</td><td style='text-align: center;'>0.567</td><td style='text-align: center;'>0.447</td><td style='text-align: center;'>27.483</td><td style='text-align: center;'>0.820</td><td style='text-align: center;'>0.263</td><td style='text-align: center;'>24.063</td></tr><tr><td style='text-align: center;'>D-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>17.660</td><td style='text-align: center;'>0.554</td><td style='text-align: center;'>0.431</td><td style='text-align: center;'>16.905</td><td style='text-align: center;'>0.544</td><td style='text-align: center;'>0.445</td><td style='text-align: center;'>27.994</td><td style='text-align: center;'>0.869</td><td style='text-align: center;'>0.187</td><td style='text-align: center;'>24.361</td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>27.230</td><td style='text-align: center;'>0.846</td><td style='text-align: center;'>0.229</td><td style='text-align: center;'>24.887</td><td style='text-align: center;'>0.848</td><td style='text-align: center;'>0.209</td><td style='text-align: center;'>29.377</td><td style='text-align: center;'>0.889</td><td style='text-align: center;'>0.202</td><td style='text-align: center;'>24.224</td></tr><tr><td style='text-align: center;'>T-NeRF $ _{PINN} $</td><td style='text-align: center;'>15.241</td><td style='text-align: center;'>0.413</td><td style='text-align: center;'>0.540</td><td style='text-align: center;'>14.959</td><td style='text-align: center;'>0.395</td><td style='text-align: center;'>0.552</td><td style='text-align: center;'>21.644</td><td style='text-align: center;'>0.780</td><td style='text-align: center;'>0.338</td><td style='text-align: center;'>20.990</td></tr><tr><td style='text-align: center;'>HexPlane $ _{PINN} $</td><td style='text-align: center;'>25.494</td><td style='text-align: center;'>0.768</td><td style='text-align: center;'>0.337</td><td style='text-align: center;'>24.991</td><td style='text-align: center;'>0.768</td><td style='text-align: center;'>0.325</td><td style='text-align: center;'>24.447</td><td style='text-align: center;'>0.867</td><td style='text-align: center;'>0.225</td><td style='text-align: center;'>23.955</td></tr><tr><td style='text-align: center;'>NVFi(Li et al., 2023a)</td><td style='text-align: center;'>27.276</td><td style='text-align: center;'>0.840</td><td style='text-align: center;'>0.235</td><td style='text-align: center;'>28.269</td><td style='text-align: center;'>0.855</td><td style='text-align: center;'>0.220</td><td style='text-align: center;'>26.999</td><td style='text-align: center;'>0.848</td><td style='text-align: center;'>0.227</td><td style='text-align: center;'>28.654</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>28.327</td><td style='text-align: center;'>0.885</td><td style='text-align: center;'>0.115</td><td style='text-align: center;'>24.947</td><td style='text-align: center;'>0.875</td><td style='text-align: center;'>0.131</td><td style='text-align: center;'>24.997</td><td style='text-align: center;'>0.900</td><td style='text-align: center;'>0.138</td><td style='text-align: center;'>23.532</td></tr><tr><td style='text-align: center;'>DefGS $ _{nvf} $</td><td style='text-align: center;'>28.169</td><td style='text-align: center;'>0.884</td><td style='text-align: center;'>0.114</td><td style='text-align: center;'>28.481</td><td style='text-align: center;'>0.922</td><td style='text-align: center;'>0.088</td><td style='text-align: center;'>25.774</td><td style='text-align: center;'>0.896</td><td style='text-align: center;'>0.141</td><td style='text-align: center;'>26.577</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>27.977</td><td style='text-align: center;'>0.880</td><td style='text-align: center;'>0.097</td><td style='text-align: center;'>29.655</td><td style='text-align: center;'>0.931</td><td style='text-align: center;'>0.063</td><td style='text-align: center;'>25.909</td><td style='text-align: center;'>0.901</td><td style='text-align: center;'>0.106</td><td style='text-align: center;'>29.120</td></tr></table>

<div style="text-align: center;">Table 13: Per-scene quantitative results on Dynamic Multipart dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3">Methods</td><td colspan="5">Folding Chair</td><td colspan="5">Hyperbolic Slot</td></tr><tr><td colspan="3">Interpolation</td><td colspan="3">Extrapolation</td><td colspan="3">Interpolation</td><td style='text-align: center;'>Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td></tr><tr><td style='text-align: center;'>T-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>10.146</td><td style='text-align: center;'>0.598</td><td style='text-align: center;'>0.537</td><td style='text-align: center;'>10.260</td><td style='text-align: center;'>0.586</td><td style='text-align: center;'>0.548</td><td style='text-align: center;'>7.437</td><td style='text-align: center;'>0.424</td><td style='text-align: center;'>0.749</td><td style='text-align: center;'>7.098</td></tr><tr><td style='text-align: center;'>D-NeRF (Pumarola et al., 2021)</td><td style='text-align: center;'>11.681</td><td style='text-align: center;'>0.717</td><td style='text-align: center;'>0.437</td><td style='text-align: center;'>13.177</td><td style='text-align: center;'>0.765</td><td style='text-align: center;'>0.357</td><td style='text-align: center;'>7.279</td><td style='text-align: center;'>0.485</td><td style='text-align: center;'>0.714</td><td style='text-align: center;'>7.547</td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>34.160</td><td style='text-align: center;'>0.984</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>13.391</td><td style='text-align: center;'>0.808</td><td style='text-align: center;'>0.199</td><td style='text-align: center;'>28.637</td><td style='text-align: center;'>0.955</td><td style='text-align: center;'>0.083</td><td style='text-align: center;'>25.436</td></tr><tr><td style='text-align: center;'>NVFi(Li et al., 2023a)</td><td style='text-align: center;'>27.748</td><td style='text-align: center;'>0.962</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>23.433</td><td style='text-align: center;'>0.940</td><td style='text-align: center;'>0.063</td><td style='text-align: center;'>25.487</td><td style='text-align: center;'>0.944</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>25.757</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>37.319</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.009</td><td style='text-align: center;'>13.682</td><td style='text-align: center;'>0.820</td><td style='text-align: center;'>0.169</td><td style='text-align: center;'>31.780</td><td style='text-align: center;'>0.983</td><td style='text-align: center;'>0.030</td><td style='text-align: center;'>25.631</td></tr><tr><td style='text-align: center;'>DefGS $ _{NVF} $</td><td style='text-align: center;'>37.269</td><td style='text-align: center;'>0.994</td><td style='text-align: center;'>0.009</td><td style='text-align: center;'>25.404</td><td style='text-align: center;'>0.962</td><td style='text-align: center;'>0.022</td><td style='text-align: center;'>32.506</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.025</td><td style='text-align: center;'>29.351</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>37.910</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>27.869</td><td style='text-align: center;'>0.978</td><td style='text-align: center;'>0.015</td><td style='text-align: center;'>31.740</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>34.185</td></tr><tr><td rowspan="3">Methods</td><td colspan="3">Satellite</td><td colspan="3">Extrapolation</td><td colspan="3">Stove</td><td style='text-align: center;'>Extrapolation</td></tr><tr><td colspan="3">Interpolation</td><td colspan="3">Extrapolation</td><td colspan="3">Interpolation</td><td style='text-align: center;'>Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td></tr><tr><td style='text-align: center;'>T-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>14.614</td><td style='text-align: center;'>0.754</td><td style='text-align: center;'>0.307</td><td style='text-align: center;'>14.468</td><td style='text-align: center;'>0.751</td><td style='text-align: center;'>0.328</td><td style='text-align: center;'>7.134</td><td style='text-align: center;'>0.490</td><td style='text-align: center;'>0.605</td><td style='text-align: center;'>8.429</td></tr><tr><td style='text-align: center;'>D-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>17.991</td><td style='text-align: center;'>0.930</td><td style='text-align: center;'>0.100</td><td style='text-align: center;'>17.252</td><td style='text-align: center;'>0.926</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>16.165</td><td style='text-align: center;'>0.856</td><td style='text-align: center;'>0.262</td><td style='text-align: center;'>15.400</td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>33.061</td><td style='text-align: center;'>0.983</td><td style='text-align: center;'>0.035</td><td style='text-align: center;'>28.627</td><td style='text-align: center;'>0.978</td><td style='text-align: center;'>0.032</td><td style='text-align: center;'>23.969</td><td style='text-align: center;'>0.943</td><td style='text-align: center;'>0.109</td><td style='text-align: center;'>15.760</td></tr><tr><td style='text-align: center;'>NVFi(Li et al., 2023a)</td><td style='text-align: center;'>29.644</td><td style='text-align: center;'>0.973</td><td style='text-align: center;'>0.029</td><td style='text-align: center;'>30.075</td><td style='text-align: center;'>0.975</td><td style='text-align: center;'>0.027</td><td style='text-align: center;'>27.186</td><td style='text-align: center;'>0.959</td><td style='text-align: center;'>0.072</td><td style='text-align: center;'>21.675</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>36.832</td><td style='text-align: center;'>0.993</td><td style='text-align: center;'>0.007</td><td style='text-align: center;'>27.622</td><td style='text-align: center;'>0.979</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>32.607</td><td style='text-align: center;'>0.989</td><td style='text-align: center;'>0.029</td><td style='text-align: center;'>15.721</td></tr><tr><td style='text-align: center;'>DefGS $ _{NVF} $</td><td style='text-align: center;'>36.640</td><td style='text-align: center;'>0.993</td><td style='text-align: center;'>0.007</td><td style='text-align: center;'>34.282</td><td style='text-align: center;'>0.990</td><td style='text-align: center;'>0.007</td><td style='text-align: center;'>21.134</td><td style='text-align: center;'>0.988</td><td style='text-align: center;'>0.029</td><td style='text-align: center;'>24.781</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>36.687</td><td style='text-align: center;'>0.994</td><td style='text-align: center;'>0.006</td><td style='text-align: center;'>31.383</td><td style='text-align: center;'>0.987</td><td style='text-align: center;'>0.009</td><td style='text-align: center;'>32.892</td><td style='text-align: center;'>0.990</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>29.446</td></tr></table>

<div style="text-align: center;">Table 14: Quantitative results for both novel view interpolation and future frame extrapolation on GoPro Dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3"></td><td colspan="5">GoPro Dataset</td><td style='text-align: center;'></td></tr><tr><td colspan="3">Interpolation</td><td colspan="2">Extrapolation</td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>15.306</td><td style='text-align: center;'>0.588</td><td style='text-align: center;'>0.516</td><td style='text-align: center;'>20.323</td><td style='text-align: center;'>0.738</td><td style='text-align: center;'>0.318</td></tr><tr><td style='text-align: center;'>NVFi(Li et al., 2023a)</td><td style='text-align: center;'>14.229</td><td style='text-align: center;'>0.568</td><td style='text-align: center;'>0.569</td><td style='text-align: center;'>19.879</td><td style='text-align: center;'>0.736</td><td style='text-align: center;'>0.415</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>20.018</td><td style='text-align: center;'>0.838</td><td style='text-align: center;'>0.167</td><td style='text-align: center;'>21.193</td><td style='text-align: center;'>0.842</td><td style='text-align: center;'>0.185</td></tr><tr><td style='text-align: center;'>DefGS $ _{nvfi} $</td><td style='text-align: center;'>20.254</td><td style='text-align: center;'>0.838</td><td style='text-align: center;'>0.167</td><td style='text-align: center;'>25.469</td><td style='text-align: center;'>0.882</td><td style='text-align: center;'>0.141</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>20.124</td><td style='text-align: center;'>0.834</td><td style='text-align: center;'>0.168</td><td style='text-align: center;'>26.276</td><td style='text-align: center;'>0.890</td><td style='text-align: center;'>0.131</td></tr></table>

<div style="text-align: center;"><img src="imgs/img_in_image_box_164_178_1012_1400.jpg" alt="Image" width="69%" /></div>


<div style="text-align: center;">Figure 5: Qualitative results for Object/Part Segmentation on Dynamic Object dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_214_187_405_378.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_416_187_608_378.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_651_200_772_368.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_845_199_978_371.jpg" alt="Image" width="10%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_253_409_367_568.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_453_409_569_569.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_654_407_769_569.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_855_407_971_569.jpg" alt="Image" width="9%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_252_610_360_785.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_456_616_561_785.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_658_642_762_786.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_859_669_963_787.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_253_816_363_989.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_457_822_561_989.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_658_846_763_989.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_859_873_964_989.jpg" alt="Image" width="8%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_232_1079_380_1126.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_433_1074_580_1128.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_635_1073_782_1128.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_836_1075_983_1128.jpg" alt="Image" width="12%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_219_1215_403_1400.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_433_1281_579_1331.jpg" alt="Image" width="11%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_621_1217_805_1399.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_820_1215_1006_1400.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">Figure 6: Qualitative results for Object/Part Segmentation on Dynamic Multipart dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_217_396_405_584.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_417_395_607_585.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_616_396_808_586.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_818_396_1007_587.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_219_605_402_788.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_420_604_603_788.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_620_604_804_788.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_821_604_1004_789.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_216_805_406_996.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_416_805_607_997.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_616_807_808_998.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_819_807_1007_997.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_219_1012_401_1196.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_418_1012_603_1196.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_619_1012_804_1197.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">Figure 7: Qualitative results for Object/Part Segmentation on Dynamic Indoor Scene dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_820_1012_1003_1196.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_206_205_1007_1331.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 8: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on Dynamic Object dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_138_201_1007_1306.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 9: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on Dynamic Object dataset.</div>


Interpolation

Extrapolation

 $$ t = 0.5 $$ 

t = 0

Interpolation

t = 1

Extrapolation

t = 0

 $$ t = 0.5 $$ 

t = 1

<div style="text-align: center;"><img src="imgs/img_in_image_box_138_216_1005_1334.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 10: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on Dynamic Object and Dynamic Indoor Scene datasets.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_138_204_1005_1330.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 11: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on Dynamic Indoor Scene dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_136_183_1018_1347.jpg" alt="Image" width="72%" /></div>


<div style="text-align: center;">Figure 12: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on Dynamic Multipart dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_138_185_996_1336.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 13: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on "Skating" scene of NVIDIA Dynamic Scene dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_139_312_997_1246.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 14: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on "Box" scene of GoPro dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_138_297_997_1246.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 15: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on "Hammer" scene of GoPro dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_139_298_997_1247.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 16: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks on "Collision" scene of GoPro dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_141_248_1000_1294.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 17: Qualitative results for object segmentation on “Chessboard” of Dynamic Indoor Scene dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_140_230_1002_1305.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 18: Qualitative results for object segmentation on “Gnome House” of Dynamic Indoor Scene dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_135_153_1011_1392.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_140_244_1002_1296.jpg" alt="Image" width="70%" /></div>


<div style="text-align: center;">Figure 20: Qualitative results for object segmentation on “Factory” of Dynamic Indoor Scene dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_200_177_1003_1403.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 21: Qualitative results of RGB view synthesis for longer extrapolation from our method.</div>