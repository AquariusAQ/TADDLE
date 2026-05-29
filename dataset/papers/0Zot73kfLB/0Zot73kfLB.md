

# GVFI: LEARNING 3D GAUSSIAN VELOCITY FIELDS FROM DYNAMIC VIDEOS

Anonymous authors

Paper under double-blind review

## ABSTRACT

In this paper, we aim to model 3D scene geometry, appearance, and physical information just from dynamic multi-view videos in the absence of any human labels. By leveraging physics-informed losses as soft constraints or integrating simple physics models into neural networks, existing works often fail to learn complex motion physics, or doing so requires additional labels such as object types or masks. In this paper, we propose a new framework named GVFi to model the motion physics of complex dynamic 3D scenes. The key novelty of our approach is that, by formulating each 3D point as a rigid particle with size and orientation in space, we choose to directly learn a translation rotation dynamics system for each particle, explicitly estimating a complete set of physical parameters to govern the particle's motion over time. Extensive experiments on three existing dynamic datasets and two newly created challenging synthetic and real-world datasets demonstrate the extraordinary performance of our method over baselines in the task of future frame extrapolation. A nice property of our framework is that multiple objects or parts can be easily segmented just by clustering the learned physical parameters. Our datasets and code will be released at https://github.com/.

## 1 INTRODUCTION

Regarding our daily dynamic 3D scenes such as falling balls, rotating fans, and folding chairs, precisely modeling their geometry, appearance, and physical properties, and further predicting their future states are crucial for emerging applications in robotics, mixed reality, and embodied AI. With the advancement of recent 3D representations such as NeRF (Mildenhall et al., 2020) and 3DGS (Kerbl et al., 2023), a plethora of works (Pumarola et al., 2021; Yang et al., 2024; Wu et al., 2024) have been proposed to model various dynamic 3D scenes, achieving excellent performance in interpolating novel views within the observed time. However, they often fail to extrapolate future frames, fundamentally because they cannot learn the underlying physics priors of complex 3D scenes.

To learn physics priors, existing methods mainly consist of two categories: 1) physics-informed neural network (PINN) based methods (Raissi et al., 2019) which integrate the governing partial differential equations (PDEs) into loss functions to drive neural networks to learn physically plausible dynamic 3D scenes such as floating smoke (Chu et al., 2022) and simple moving objects (Li et al., 2023b). Although demonstrating promising results in modeling 3D geometry and physics such as velocity and viscosity, these methods usually need boundary constraints such as accurate object/foreground masks which may not always be available in practice. In addition, adding PINN losses is not a free lunch, but significantly sacrificing the efficiency in training and accuracy at boundary regions. 2) Physics model based methods (Jonathan et al., 2020; Zhong et al., 2024; Whitney et al., 2024) which encode various physics systems into neural networks to model elastic objects, fluids, etc.. Thanks to the explicit physics priors, these methods obtain impressive results in physical properties learning and simulation. Nevertheless, they are often limited to specific types of objects, materials, or motions due to the lack of generality of encoded physics priors, thus being unable to predict future motions of complex dynamic 3D objects and scenes.

In this paper, we aim to introduce a new framework to model dynamic 3D scenes just from multiview RGB videos, without needing any additional human labels such as object types or masks, ultimately being able to predict future frames viewing from arbitrary angles. Among various physical properties of a dynamic 3D scene, following the recent work NVFi (Li et al., 2023a), we also

<div style="text-align: center;"><img src="imgs/img_in_image_box_252_159_968_298.jpg" alt="Image" width="58%" /></div>


<div style="text-align: center;">Figure 1: An illustration of the overall framework.</div>


choose to learn a velocity field as it directly governs 3D scene movement. However, to accurately learn the physical velocity from RGB videos is extremely challenging, essentially due to the lack of sufficient physics constraints from raw color pixels. This problem is even harder when multiple objects or parts are undergoing rather different motion patterns. For example, regarding two adjacent objects moving in opposite directions in 3D space, the velocity of neighboring 3D surface points at the intersection region tends to have particularly distinct patterns. This means that the latent representation of per-point dynamics in 3D space could be discrete in nature. Therefore, it is more desirable to model per-point dynamics independently, thus every point having its unique motion. For generality, we regard each 3D point in space as a rigid particle with its size and orientation. If its size is zero, the rigid particle degenerates to a point.

With this insight, for each rigid particle in space, we propose to learn an independent dynamics system that includes a complete set of physical parameters to govern its motion over time. According to the laws of classical mechanics, for a specific rigid particle traversing 3D space over time, its motion can always be regarded as a rotational movement about a rotation center which has its own translation. Given this general foundation, we choose to learn a translation rotation dynamics system for each rigid particle, allowing its future motion to be derived accordingly. Alongside learning the core dynamics, we must also model the geometry and appearance of 3D scenes. In this paper, we naturally choose the recent 3D Gaussian Splatting (3DGS) (Kerbl et al., 2023) as our scene representation, thanks to its unprecedented fidelity in reconstruction and its particle (a Gaussian kernel) based representation in nature, which shares the basic concept of our defined rigid particle.

As illustrated in Figure 1, our framework consists of two major components: 1) a 3D scene representation module to learn dynamic scene geometry and appearance at a canonical timestamp, which is implemented by a vanilla 3DGS (Kerbl et al., 2023), though other variants can be adopted as well; 2) a translation rotation dynamics system module to learn a full set of physical parameters for each input rigid particle, which is just realized by multilayer perceptrons (MLPs). Based on these system parameters, the rigid particle's velocity is then derived according to the laws of classical mechanics, without needing additional physics priors such as PINN (Raissi et al., 2019) in training.

The key to our framework is the second module which simply regards each 3D Gaussian kernel as a rigid particle and takes it as input into MLPs. Nevertheless, we empirically find that it is hard to optimize this module due to the inaccuracy and instability of Gaussian kernels regressed at early training epochs. To tackle this issue, we simply train an auxiliary deformation field in parallel with our second module using an existing work such as (Yang et al., 2024) and (Wu et al., 2024).

Different from current works for modeling dynamic scenes, including NeRF-based methods, e.g., D-NeRF(Pumarola et al., 2021)/TiNeuVox(Fang et al., 2022)/HexPlane(Cao & Johnson, 2023), and 3DGS-based methods such as DefGS (Yang et al., 2024), 4DGS (Wu et al., 2024), and E-D3DGS (Bae et al., 2024), our core novelty is the introduced translation rotation dynamics system together with its effective optimization strategy, which allows us to truly learn physical parameters, ultimately achieving future frame extrapolation. By comparison, all those existing methods fail to do so, though they perform well for past frame interpolation, as extensively verified in Tables 1&2.

Our method, named GVFi, leverages 3D Gaussians to model scene geometry and appearance, while learning velocity fields via estimating translation rotation dynamics systems. Our contributions are:

• We introduce a new framework to model motion physics of complex dynamic 3D scenes, without needing prior knowledge of object shapes, types, or masks.

• We propose to learn a translation rotation dynamics system for each 3D rigid particle, thus allowing the velocity field to be derived without needing additional physics constraints in training.

• We demonstrate superior results in future frame extrapolation on three existing datasets, and two newly collected synthetic and real-world datasets with extremely challenging dynamics.

## 2 RELATED WORKS

3D Shape Representations: Static 3D objects and scenes are traditionally represented by voxels, point clouds, meshes, etc., but they usually have limited representation capabilities due to the nature of discretization. Recently, implicit representations have been developed in the literature, including occupancy fields (OF) (Mescheder et al., 2019; Chen & Zhang, 2019), un/signed distance fields (U/SDF) (Park et al., 2019; Chibane et al., 2020), and radiance fields (NeRF) (Mildenhall et al., 2020). Although demonstrating excellent performance in novel view synthesis and shape reconstruction, they are time-consuming to render 2D images or extract 3D shapes due to the integration of their continuous coordinate-based representations. To tackle this issue, the very recent 3D Gaussian Splatting (Kerbl et al., 2023) turns to represent a 3D shape as a set of explicit Gaussian kernels with various properties, achieving real-time rendering speed. In our framework, we adopt this particle-based representation, as it is amenable to our particle-based physics learning framework.

Dynamic 3D Reconstruction: Recent advances in dynamic 3D reconstruction primarily follow the development of static 3D techniques such as SDF, NeRF, and Gaussian Splatting. To model the temporal relationship, existing works (Tretschk et al., 2021; Li et al., 2021; Barron et al., 2021; Gao et al., 2021; Tian et al., 2023; Liu et al., 2023; Cao & Johnson, 2023; Fridovich-Keil et al., 2023; Cai et al., 2022; Fang et al., 2022; Li et al., 2022; Park et al., 2021; You & Hou, 2023; Du et al., 2021; Park et al., 2023; Xian et al., 2021; Wang et al., 2021; Liu et al., 2024b) usually add the time dimension into static 3D representations to learn a motion or deformation field for rigid or deformable objects and scenes. Despite achieving excellent performance in novel view synthesis, especially when integrating 3DGS as the backbone (Wu et al., 2024; Yang et al., 2024; Li et al., 2024; Lei et al., 2024; Lin et al., 2024; Lu et al., 2024), these works can only interpolate 2D views within the observed time, instead of predicting physically meaningful future frames. Basically, this is because the commonly learned motion or deformation field does not encode physics priors in nature, but just fits the correlation between pixels. In this paper, the key difference between these works and us is that we separately learn translation rotation dynamics systems for 3D rigid particles, thus enabling us to estimate physically meaningful future frames, whereas they cannot.

3D Physics Learning: To learn various physical properties for 3D objects and scenes, the recent physics-informed neural networks (PINN) (Raissi et al., 2019; Mishra & Molinaro, 2023; Raissi et al., 2020; Hao et al., 2023; Baieri et al., 2023; Chalapathi et al., 2024; Wang et al., 2024; Zhao et al., 2024) are widely applied to convert PDEs into loss functions as soft constraints, driving neural networks to learn physically meaningful targets. However, it is often inefficient to train PINNs due to the large amount of data samples needed to regularize, and the soft constraints are usually not sufficient to obtain satisfactory results. In this paper, we do not rely on such inefficient PINN losses to incorporate physics priors to train neural networks. Another line of works (Qiao et al., 2022; Deng et al., 2023; Xue et al., 2023; Franz et al., 2023; Whitney et al., 2024) integrate explicit physics systems such as springs, graphs, etc., into the learning process to model elastic objects (Zhong et al., 2024; Zhang et al., 2024; Liu et al., 2024a), fluids (Jonathan et al., 2020; Lienen et al., 2024), etc., achieving impressive results in physics learning and simulation. In this paper, we also opt to learn physics systems. However, the core difference is that we learn a translation rotation dynamics system which is applicable to common deformation and transformation dynamics, whereas existing works often learn a spring or fluid system only applicable to elastic objects or fluids.

## 3 GVFI

Our framework mainly comprises two modules together with an auxiliary deformation field to model 3D geometry, appearance, and physics. Given dynamic multi-view RGB videos with known camera poses and intrusions, the 3D scene representation module aims to learn a set of 3D Gaussian kernels to represent the 3D scene geometry and appearance in a canonical space. The auxiliary deformation field is designed to predict the translation and distortion of each Gaussian kernel given the current training time t. For these two components, we simply follow the design of existing works (Kerbl et al., 2023; Yang et al., 2024) briefly elaborated in Section 3.1. Notably, the deformation field alone cannot extrapolate frames beyond the training time. Our core module of the translation rotation dynamics system aims to learn a set of physical parameters for each 3D rigid particle, governing its motion dynamics over time, which is detailed in Section 3.2.

### 3.1 PRELIMINARY

For the input multi-view RGB videos, T represents the greatest timestamp in training and N the total number of cameras. For training stability, we first use all frames  $ \{I_{0}^{1}\cdots I_{0}^{n}\cdots I_{0}^{N}\} $  at time t=0 to train a reasonable static 3DGS model as an initialization of the 3D scene geometry and appearance, and then use the remaining frames to jointly optimize our translation and rotation dynamics system and the auxiliary deformation field.

Canonical 3D Gaussians: Following the vanilla 3DGS (Kerbl et al., 2023), we employ a set of learnable 3D Gaussian kernels  $ G_{0} $  to represent the canonical scene geometry and appearance at t=0. Each kernel is parameterized by a 3D position  $ x_{0} $ , covariance matrix obtained from quaternion  $ r_{0} $ , scaling  $ s_{0} $ , opacity  $ \sigma $ , and color c computed from spherical harmonics (SH). Following prior works (Yang et al., 2024; Wu et al., 2024), we assume the opacity  $ \sigma $  and color c of each Gaussian will not be updated, but constantly associated with the kernel and transported over time.

Given the N images at timestamp t = 0, we either initialize all canonical 3D Gaussian kernels  $ G_{0} $  randomly or based on sparse points created by SfM (Schonberger & Frahm, 2016). To train all kernels, we exactly follow the process of 3DGS (Kerbl et al., 2023) by 1) projecting Gaussian kernels into camera space, 2) rendering the projected kernels into image space, and 3) optimizing all kernel parameters via  $ \ell_{1} $  and  $ \ell_{ssim} $  losses used in 3DGS as follows.

 $$ \underbrace{\left\{\cdots\left(\boldsymbol{x}_{0},\boldsymbol{r}_{0},\boldsymbol{s}_{0},\boldsymbol{\sigma},\boldsymbol{c}\right)\cdots\right\}}_{G_{0}}\xleftarrow[\ell_{1}+\ell_{\mathrm{s s i m}}]{\mathrm{p r o j e c t+r e n d e r}}\left\{I_{0}^{1}\cdots I_{0}^{n}\cdots I_{0}^{N}\right\} $$ 

Auxiliary Deformation Field: To aid the learning of our translation and rotation dynamics system, we leverage an existing deformation field (Yang et al., 2024), but we are also amenable to other deformable Gaussian methods such as 4DGS (Wu et al., 2024), as demonstrated in our experiments in Section 4.1. In particular, the 3D position $x_{0}$ of each canonical Gaussian kernel and the current timestamp $t$ are fed into an MLP-based deformation network, denoted as $f_{defo}$, directly predicting the corresponding position displacement $\delta x$, and the change of quaternion $\delta r$ and scaling $\delta s$ from timestamp 0 to $t$. All Gaussians $G_{t}$ at time $t$ can be easily computed as follows, where the operations $\circ$ and $\odot$ follow (Yang et al., 2024).

 $$ \underbrace{\left\{\cdot\cdot\left((\boldsymbol{x}_{t}=\boldsymbol{x}_{0}+\delta\boldsymbol{x}),(\boldsymbol{r}_{t}=\boldsymbol{r}_{0}\circ\delta\boldsymbol{r}),(\boldsymbol{s}_{t}=\boldsymbol{s}_{0}\odot\delta\boldsymbol{s}),\boldsymbol{\sigma},\boldsymbol{c}\right)\cdot\cdot\right\}}_{G_{t}},(\delta\boldsymbol{x},\delta\boldsymbol{r},\delta\boldsymbol{s})=f_{d e f o}(\boldsymbol{x}_{0},t) $$ 

All these deformed Gaussians will be projected and optimized by visual images at timestamp t in a later stage as clarified in Section 3.3, where the deformation net  $ f_{defo} $  will be optimized from scratch. All details are provided in Appendix A.1 and A.2.

### 3.2 TRANSLATION ROTATION DYNAMICS SYSTEM

This module aims to learn physical parameters that govern the motion of 3D scenes. However, the dynamics of an entire space are extremely complex. Here, we simplify this problem and formulate it into just learning per rigid particle dynamics, where we treat each (canonical or deformed) Gaussian kernel as a rigid particle with size and orientation. According to the laws of classical mechanics, for a specific rigid particle  $ P \in R^{3} $ , its motion in a 3D world coordinate system can be regarded as a rotational movement about a rotation center which has its own translation. To this end, we aim to learn the following two groups of physical parameters for each 3D rigid particle P:

• Group #1 - Rotation Center Parameters including: 1) the center's position  $ P_{c} \in R^{3} $ , 2) the center's velocity  $ v_{c} \in R^{3} $ , and 3) acceleration  $ a_{c} \in R^{3} $  in the world coordinate system.

• Group #2 - Rigid Particle Rotational Parameters including: 1) the rigid particle's rotation vector  $ w_{p} \in R^{3} $  with regard to its center  $ P_{c} $ , and 2) the rigid particle's angular acceleration  $ \epsilon_{p} $ .

As illustrated in Figure 2, our translation rotation dynamics system module, denoted as  $ f_{trd} $ , takes a rigid particle P as input, directly predicting the physical parameters of its rotation center and its own rotational information. Then, this rigid particle will be naturally driven by its learned physical parameters, forming its motion dynamics, as illustrated by the trajectory in Figure 2. This module is implemented by simple MLPs:

 $$ \{(\boldsymbol{P}_{c},\boldsymbol{v}_{c},\boldsymbol{a}_{c}),(\boldsymbol{w}_{p},\boldsymbol{\epsilon}_{p})\}=f_{t r d}(\boldsymbol{P}) $$ 

<div style="text-align: center;"><img src="imgs/img_in_image_box_212_159_1007_307.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 2: The proposed translation rotation dynamics system for a specific rigid particle. The rigid particle will be driven by its learned physical parameters over time, forming a trajectory in 3D space.</div>


Notably, for an input rigid particle P, the elegance of this module  $ f_{trd} $  is that it only needs to learn this full translation rotation dynamics system at its canonical timestamp, i.e., t = 0, and that particle's future motion will be governed by the learned dynamics system when t > 0 according to the laws of mechanics. We will now derive the rigid particle's future motion as follows.

For a specific rigid particle P, now we have its estimated translation rotation dynamics parameters  $ \{(P_{c}, \boldsymbol{v}_{c}, \boldsymbol{a}_{c}), (\boldsymbol{w}_{p}, \boldsymbol{\epsilon}_{p})\} $ . Given a future timestamp t, we now calculate its updated parameters for both the rotation center and the particle itself as follows:

 $$ \boldsymbol{P}_{c}^{t}=\boldsymbol{P}_{c}+\boldsymbol{v}_{c}t+\frac{1}{2}\boldsymbol{a}_{c}t^{2},\quad\boldsymbol{v}_{c}^{t}=\boldsymbol{v}_{c}+\boldsymbol{a}_{c}t,\quad\boldsymbol{a}_{c}^{t}=\boldsymbol{a}_{c},\quad\boldsymbol{w}_{p}^{t}=\left(\|\boldsymbol{w}_{p}\|+\epsilon_{p}t\right)\frac{\boldsymbol{w}_{p}}{\|\boldsymbol{w}_{p}\|},\quad\epsilon_{p}^{t}=\epsilon_{p} $$ 

Theoretically, our above updating scheme can be naturally extended to higher orders or reduced to lower orders with regard to future time t. Intuitively, a higher order relationship from time 0 to t is expected to capture extremely complex dynamics such as a rolling ball suddenly breaking up into pieces due to unknown explosives inside, whereas a much lower order relationship tends to only capture static or constant speed scenes, thus being oversimplified. In this paper, we opt to the above second-order scheme to update dynamics parameters from time 0 to t for two primary reasons:

• In many applications such as robot manipulation, the need for future prediction usually involves a relatively short interval, i.e.,  $ |t - 0| $  is rather small, e.g., in milliseconds. In this case, a second-order relationship is usually sufficient to achieve decent approximations. In addition, a simple sliding window based approach can be applied to continuously and incrementally predict future frames given the newest visual observations from sensors.

• In our daily life, the majority of common physical movements such as rolling balls or moving cars can be generally described by a second-order relationship. In fact, both Newton's First and Second Law of Motion can be captured. Notably, since the whole 3D scene comprises a large number of rigid particles, each particle has up to second-order dynamics, i.e., with a constant acceleration between  $ 0 \sim t $ . Therefore, the compounded dynamics for the entire 3D scene can be rather complex, including various deformations and transformations in our daily lives.

Nevertheless, it is still interesting yet non-trivial to learn much higher-order relationships and we leave it for future exploration. More implementation details of this module are in Appendix A.3.

### 3.3 TRAINING

With our translation rotation dynamics module and the auxiliary deformation field, we now discuss how to connect and train them together, such that physical parameters can be truly learned.

For two timestamps  $ t' $  and t, where t is usually sampled from the training set and  $ \Delta t = t - t' $  is predefined to be small enough, we can easily obtain Gaussians  $ G_{t'} $  from the deformation field  $ f_{defo} $ .

Having our translation rotation dynamics module  $ f_{trd} $  at hand, we naturally regard the transportation of all kernels from  $ t' $  to t is governed by the corresponding physical parameters estimated by  $ f_{trd} $  at time  $ t' $ . From Equations 4, at time  $ t' $ , the physical parameters of a rigid particle P are:

 $$ \{(\boldsymbol{P}_{c}^{t^{\prime}},\boldsymbol{v}_{c}^{t^{\prime}},\boldsymbol{a}_{c}^{t^{\prime}}),(\boldsymbol{w}_{p}^{t^{\prime}},\boldsymbol{\epsilon}_{p}^{t^{\prime}})\}\xleftarrow{t^{\prime}}\{(\boldsymbol{P}_{c},\boldsymbol{v}_{c},\boldsymbol{a}_{c}),(\boldsymbol{w}_{p},\boldsymbol{\epsilon}_{p})\}=f_{t r d}(\boldsymbol{P}) $$ 

Now we can easily compute the kernel's orientation change  $ \Delta r $  from  $ r_{t'} $  to  $ r_{t} $  as follows:

 $$ \Delta\boldsymbol{r}=(\cos\frac{\Delta\theta}{2},\sin\frac{\Delta\theta}{2}\cdot\frac{\boldsymbol{w}_{p}}{\|\boldsymbol{w}_{p}\|}),\quad where\Delta\theta=(\|\boldsymbol{w}_{p}\|+\epsilon_{p}t^{\prime})(t-t^{\prime}) $$ 

<div style="text-align: center;">Table 1: Quantitative results of all methods for both future frame extrapolation and novel view interpolation on Dynamic Object Dataset and Dynamic Indoor Scene Dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3"></td><td colspan="5">Dynamic Object Dataset</td><td colspan="5">Dynamic Indoor Scene Dataset</td></tr><tr><td colspan="5">Interpolation</td><td colspan="5">Interpolation</td></tr><tr><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td></tr><tr><td style='text-align: center;'>T-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>13.163</td><td style='text-align: center;'>0.709</td><td style='text-align: center;'>0.353</td><td style='text-align: center;'>13.818</td><td style='text-align: center;'>0.739</td><td style='text-align: center;'>0.324</td><td style='text-align: center;'>24.944</td><td style='text-align: center;'>0.742</td><td style='text-align: center;'>0.336</td><td style='text-align: center;'>22.242</td></tr><tr><td style='text-align: center;'>D-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>14.158</td><td style='text-align: center;'>0.697</td><td style='text-align: center;'>0.352</td><td style='text-align: center;'>14.660</td><td style='text-align: center;'>0.737</td><td style='text-align: center;'>0.312</td><td style='text-align: center;'>25.380</td><td style='text-align: center;'>0.766</td><td style='text-align: center;'>0.300</td><td style='text-align: center;'>20.791</td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>27.988</td><td style='text-align: center;'>0.960</td><td style='text-align: center;'>0.063</td><td style='text-align: center;'>19.612</td><td style='text-align: center;'>0.940</td><td style='text-align: center;'>0.073</td><td style='text-align: center;'>29.982</td><td style='text-align: center;'>0.864</td><td style='text-align: center;'>0.213</td><td style='text-align: center;'>21.029</td></tr><tr><td style='text-align: center;'>T-NeRF $ _{PINN} $</td><td style='text-align: center;'>15.286</td><td style='text-align: center;'>0.794</td><td style='text-align: center;'>0.293</td><td style='text-align: center;'>16.189</td><td style='text-align: center;'>0.835</td><td style='text-align: center;'>0.230</td><td style='text-align: center;'>16.250</td><td style='text-align: center;'>0.441</td><td style='text-align: center;'>0.638</td><td style='text-align: center;'>17.290</td></tr><tr><td style='text-align: center;'>HexPlane $ _{PINN} $</td><td style='text-align: center;'>27.042</td><td style='text-align: center;'>0.958</td><td style='text-align: center;'>0.057</td><td style='text-align: center;'>21.419</td><td style='text-align: center;'>0.946</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>25.215</td><td style='text-align: center;'>0.763</td><td style='text-align: center;'>0.389</td><td style='text-align: center;'>23.091</td></tr><tr><td style='text-align: center;'>NSFF(Li et al., 2021)</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>29.365</td><td style='text-align: center;'>0.829</td><td style='text-align: center;'>0.278</td><td style='text-align: center;'>24.163</td></tr><tr><td style='text-align: center;'>NVFi(Li et al., 2023a)</td><td style='text-align: center;'>29.027</td><td style='text-align: center;'>0.970</td><td style='text-align: center;'>0.039</td><td style='text-align: center;'>27.594</td><td style='text-align: center;'>0.972</td><td style='text-align: center;'>0.036</td><td style='text-align: center;'>30.675</td><td style='text-align: center;'>0.877</td><td style='text-align: center;'>0.211</td><td style='text-align: center;'>29.745</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>37.865</td><td style='text-align: center;'>0.994</td><td style='text-align: center;'>0.007</td><td style='text-align: center;'>19.849</td><td style='text-align: center;'>0.949</td><td style='text-align: center;'>0.045</td><td style='text-align: center;'>29.926</td><td style='text-align: center;'>0.916</td><td style='text-align: center;'>0.130</td><td style='text-align: center;'>21.380</td></tr><tr><td style='text-align: center;'>DefGS $ _{v0,i} $</td><td style='text-align: center;'>37.316</td><td style='text-align: center;'>0.994</td><td style='text-align: center;'>0.008</td><td style='text-align: center;'>28.749</td><td style='text-align: center;'>0.984</td><td style='text-align: center;'>0.013</td><td style='text-align: center;'>30.170</td><td style='text-align: center;'>0.915</td><td style='text-align: center;'>0.133</td><td style='text-align: center;'>31.096</td></tr><tr><td style='text-align: center;'>E-D3DGS(Bae et al., 2024)</td><td style='text-align: center;'>28.075</td><td style='text-align: center;'>0.963</td><td style='text-align: center;'>0.049</td><td style='text-align: center;'>18.526</td><td style='text-align: center;'>0.923</td><td style='text-align: center;'>0.087</td><td style='text-align: center;'>29.267</td><td style='text-align: center;'>0.874</td><td style='text-align: center;'>0.222</td><td style='text-align: center;'>20.374</td></tr><tr><td style='text-align: center;'>4DGS(Wu et al., 2024)</td><td style='text-align: center;'>37.285</td><td style='text-align: center;'>0.986</td><td style='text-align: center;'>0.020</td><td style='text-align: center;'>20.354</td><td style='text-align: center;'>0.950</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>29.381</td><td style='text-align: center;'>0.889</td><td style='text-align: center;'>0.212</td><td style='text-align: center;'>21.107</td></tr><tr><td style='text-align: center;'>GVF $ _{L4dgs} $  (Ours)</td><td style='text-align: center;'>35.961</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.021</td><td style='text-align: center;'>28.316</td><td style='text-align: center;'>0.978</td><td style='text-align: center;'>0.023</td><td style='text-align: center;'>27.932</td><td style='text-align: center;'>0.860</td><td style='text-align: center;'>0.252</td><td style='text-align: center;'>31.590</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>38.788</td><td style='text-align: center;'>0.995</td><td style='text-align: center;'>0.006</td><td style='text-align: center;'>28.758</td><td style='text-align: center;'>0.982</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>32.202</td><td style='text-align: center;'>0.928</td><td style='text-align: center;'>0.089</td><td style='text-align: center;'>34.556</td></tr></table>

<div style="text-align: center;">Table 2: Quantitative results of all methods for future frame extrapolation optionally with novel view interpolation on NVIDIA Dynamic Scene Dataset, Dynamic Multipart Dataset, and GoPro Dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="3"></td><td colspan="5">NVIDIA Dynamic Scene Dataset</td><td colspan="5">Dynamic Multipart Dataset</td><td colspan="4">GoPro Dataset</td></tr><tr><td colspan="5">Interpolation</td><td colspan="5">Interpolation</td><td colspan="4">Extrapolation</td></tr><tr><td style='text-align: center;'>PSNR[ $ \uparrow $</td><td style='text-align: center;'>SSIM[ $ \uparrow $</td><td style='text-align: center;'>LPIPS[ $ \downarrow $ ]</td><td style='text-align: center;'>PSNR[ $ \uparrow $</td><td style='text-align: center;'>SSIM[ $ \uparrow $</td><td style='text-align: center;'>LPIPS[ $ \downarrow $ ]</td><td style='text-align: center;'>PSNR[ $ \uparrow $</td><td style='text-align: center;'>SSIM[ $ \uparrow $</td><td style='text-align: center;'>LPIPS[ $ \downarrow $ ]</td><td style='text-align: center;'>PSNR[ $ \uparrow $</td><td style='text-align: center;'>SSIM[ $ \uparrow $</td><td style='text-align: center;'>LPIPS[ $ \downarrow $ ]</td><td style='text-align: center;'>PSNR[ $ \uparrow $</td><td style='text-align: center;'>SSIM[ $ \uparrow $</td></tr><tr><td style='text-align: center;'>T-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>23.078</td><td style='text-align: center;'>0.684</td><td style='text-align: center;'>0.355</td><td style='text-align: center;'>21.120</td><td style='text-align: center;'>0.707</td><td style='text-align: center;'>0.358</td><td style='text-align: center;'>9.833</td><td style='text-align: center;'>0.567</td><td style='text-align: center;'>0.550</td><td style='text-align: center;'>10.064</td><td style='text-align: center;'>0.576</td><td style='text-align: center;'>0.537</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>D-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>22.827</td><td style='text-align: center;'>0.711</td><td style='text-align: center;'>0.309</td><td style='text-align: center;'>20.633</td><td style='text-align: center;'>0.709</td><td style='text-align: center;'>0.327</td><td style='text-align: center;'>13.279</td><td style='text-align: center;'>0.747</td><td style='text-align: center;'>0.378</td><td style='text-align: center;'>13.344</td><td style='text-align: center;'>0.767</td><td style='text-align: center;'>0.340</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>TiNeuVox(Fang et al., 2022)</td><td style='text-align: center;'>28.304</td><td style='text-align: center;'>0.868</td><td style='text-align: center;'>0.216</td><td style='text-align: center;'>24.556</td><td style='text-align: center;'>0.863</td><td style='text-align: center;'>0.215</td><td style='text-align: center;'>29.957</td><td style='text-align: center;'>0.966</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>20.804</td><td style='text-align: center;'>0.923</td><td style='text-align: center;'>0.090</td><td style='text-align: center;'>20.323</td><td style='text-align: center;'>0.738</td></tr><tr><td style='text-align: center;'>T-NeRF $ _{P1NN} $</td><td style='text-align: center;'>18.443</td><td style='text-align: center;'>0.597</td><td style='text-align: center;'>0.439</td><td style='text-align: center;'>17.975</td><td style='text-align: center;'>0.605</td><td style='text-align: center;'>0.428</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>HexPlanet $ _{P1NN} $</td><td style='text-align: center;'>24.971</td><td style='text-align: center;'>0.818</td><td style='text-align: center;'>0.281</td><td style='text-align: center;'>24.473</td><td style='text-align: center;'>0.818</td><td style='text-align: center;'>0.279</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>NVFitLi et al., 2023a</td><td style='text-align: center;'>27.138</td><td style='text-align: center;'>0.844</td><td style='text-align: center;'>0.231</td><td style='text-align: center;'>28.462</td><td style='text-align: center;'>0.876</td><td style='text-align: center;'>0.214</td><td style='text-align: center;'>27.516</td><td style='text-align: center;'>0.960</td><td style='text-align: center;'>0.052</td><td style='text-align: center;'>25.235</td><td style='text-align: center;'>0.955</td><td style='text-align: center;'>0.046</td><td style='text-align: center;'>19.879</td><td style='text-align: center;'>0.736</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>26.662</td><td style='text-align: center;'>0.893</td><td style='text-align: center;'>0.127</td><td style='text-align: center;'>24.240</td><td style='text-align: center;'>0.895</td><td style='text-align: center;'>0.140</td><td style='text-align: center;'>34.635</td><td style='text-align: center;'>0.990</td><td style='text-align: center;'>0.019</td><td style='text-align: center;'>20.664</td><td style='text-align: center;'>0.930</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>21.193</td><td style='text-align: center;'>0.842</td></tr><tr><td style='text-align: center;'>DefGS $ _{n\times n} $</td><td style='text-align: center;'>26.972</td><td style='text-align: center;'>0.890</td><td style='text-align: center;'>0.128</td><td style='text-align: center;'>27.529</td><td style='text-align: center;'>0.927</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>34.637</td><td style='text-align: center;'>0.990</td><td style='text-align: center;'>0.018</td><td style='text-align: center;'>28.455</td><td style='text-align: center;'>0.972</td><td style='text-align: center;'>0.017</td><td style='text-align: center;'>25.469</td><td style='text-align: center;'>0.882</td></tr><tr><td style='text-align: center;'>E-D3DGS(Bae et al., 2024)</td><td style='text-align: center;'>20.848</td><td style='text-align: center;'>0.541</td><td style='text-align: center;'>0.532</td><td style='text-align: center;'>20.301</td><td style='text-align: center;'>0.565</td><td style='text-align: center;'>0.522</td><td style='text-align: center;'>26.180</td><td style='text-align: center;'>0.955</td><td style='text-align: center;'>0.062</td><td style='text-align: center;'>18.615</td><td style='text-align: center;'>0.904</td><td style='text-align: center;'>0.114</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>4DGS(Wu et al., 2024)</td><td style='text-align: center;'>19.411</td><td style='text-align: center;'>0.462</td><td style='text-align: center;'>0.532</td><td style='text-align: center;'>22.510</td><td style='text-align: center;'>0.703</td><td style='text-align: center;'>0.408</td><td style='text-align: center;'>37.021</td><td style='text-align: center;'>0.992</td><td style='text-align: center;'>0.014</td><td style='text-align: center;'>20.564</td><td style='text-align: center;'>0.935</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>GVF $ _{4\times 9} $  (Ours)</td><td style='text-align: center;'>18.995</td><td style='text-align: center;'>0.448</td><td style='text-align: center;'>0.544</td><td style='text-align: center;'>22.706</td><td style='text-align: center;'>0.714</td><td style='text-align: center;'>0.400</td><td style='text-align: center;'>36.542</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.015</td><td style='text-align: center;'>30.801</td><td style='text-align: center;'>0.983</td><td style='text-align: center;'>0.016</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>26.943</td><td style='text-align: center;'>0.891</td><td style='text-align: center;'>0.102</td><td style='text-align: center;'>29.388</td><td style='text-align: center;'>0.938</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>34.807</td><td style='text-align: center;'>0.991</td><td style='text-align: center;'>0.011</td><td style='text-align: center;'>30.721</td><td style='text-align: center;'>0.986</td><td style='text-align: center;'>0.012</td><td style='text-align: center;'>26.276</td><td style='text-align: center;'>0.890</td></tr></table>

Then, we compute the kernel’s position translation  $ \Delta x $  from  $ x_{t'} $  to  $ x_{t} $ , which consists of two parts: 1) the translation of its rotation center, and 2) the displacement caused by the kernel’s rotation with regard to its center. In particular, they are:

 $$ \Delta\boldsymbol{x}=\left[\boldsymbol{v}_{c}^{t^{\prime}}(t-t^{\prime})+\frac{1}{2}\boldsymbol{a}_{c}^{t^{\prime}}(t-t^{\prime})^{2}\right]+\left[(\Delta\mathbf{R}-\boldsymbol{I})(\boldsymbol{x}_{t^{\prime}}-\boldsymbol{P}_{c}^{t^{\prime}})\right] $$ 

where  $ \DeltaR $  is a  $ 3 \times 3 $  rotation matrix converted from quaternion  $ \Delta r $ . Since the rotation change  $ \Delta r $  will update both the orientation and position of a Gaussian kernel, so it is also used in Equation 7.

With the above relationships, we optimize all learnable parameters of canonical Gaussian kernels  $ G_{0} $ , the deformation field  $ f_{defo} $  and our translation rotation dynamics module  $ f_{trd} $  as follows:

• Step #1: We sample two close timestamps  $ t' $  and t, where t is a timestamp appears in training dataset and  $ t' $  can be greater or smaller than t, but  $ \Delta t = |t - t'| $  is appropriately small.

• Step #2: We get  $ \{(P_{c}^{t^{\prime}}, v_{c}^{t^{\prime}}, a_{c}^{t^{\prime}}, w_{p}^{t^{\prime}}, \epsilon_{p}^{t^{\prime}}), (P_{c}, v_{c}, a_{c}, w_{p}, \epsilon_{p})\} \leftarrow f_{trd}(P) $ , and then calculate  $ \Delta x $  and  $ \Delta r $  based on  $ (x_{t^{\prime}}, r_{t^{\prime}}, s_{t^{\prime}}, \sigma, c) \leftarrow f_{defo}(x_{0}, t^{\prime}) $ , according to Equations 6&7.

• Step #3: We obtain the kernel information at time t:  $ (\mathbf{x}_{t'} + \Delta\mathbf{x}, \mathbf{r}_{t'} \circ \Delta\mathbf{r}, \mathbf{s}_{t'}, \sigma, \mathbf{c}) $  which is transported by our physical parameters from time  $ t' $ , where  $ \circ $  represents quaternion multiplication.

• Step #4: Lastly, we render all the above kernels at time t to 2D image space following 3DGS, comparing with the training images at time t. All parameters are supervised by  $ \ell_{1} $  and  $ \ell_{ssim} $ :

 $$ \left(\boldsymbol{G}_{0},f_{d e f o},f_{t r d}\right)\longleftarrow\left(\ell_{1}+\ell_{s s i m}\right) $$ 

## 4 EXPERIMENTS

Datasets: Our method is designed to learn meaningful physical information of 3D dynamic scenes, aiming at accurately predicting future motions, instead of just fitting observed video frames. In this regard, the closest work to us is the recent NVFi (Li et al., 2023a). Following NVFi, we primarily evaluate our method on its three dynamic datasets: 1) Dynamic Object dataset. It consists of 6 dynamic objects. Each object displays a unique motion pattern belonging to either rigid or deformable movement. 2) Dynamic Indoor Scene dataset. It has 4 complex indoor scenes. Each scene has multiple objects undergoing different rigid body motions. 3) NVIDIA Dynamic Scene dataset (Yoon et al., 2020). It consists of two real-world dynamic 3D scenes.

Upon a closer look at the above three datasets, we find that their dynamics captured are relatively simple. In our daily life, the majority of objects and scenes consist of multiple parts undergoing radically different motions over time, showing extremely challenging physical patterns to learn. To further evaluate the effectiveness of our design, we collect a new synthetic dataset, named 4) Dynamic Multipart dataset, and a new real-world dataset by 20 GoPros, named 5) GoPro dataset.

Our new synthetic dataset comprises 4 objects. Each has 2 to 5 distinct motion patterns on different object parts. Following (Li et al., 2023a), for each object, we collect RGBs at 15 different viewing angles over 1 (virtual) second after normalization, where each viewing angle has 60 frames captured. We reserve the first 46 frames at randomly picked 12 viewing angles as the training split, i.e., 552 frames, while leaving the 46 frames at the remaining 3 viewing angles for testing interpolation ability, i.e., 138 frames for novel view synthesis within the training time period, and keeping the last 14 frames at all 15 viewing angles for evaluating future frame extrapolation, i.e., 210 frames.

Our new real-world dataset captures 4 dynamic scenes with 20 GoPro cameras. For each dynamic scene, we select 89 frames from each view, and resize images to be a resolution of  $ 960 \times 540 $ . We reserve the first 67 frames at 17 picked viewing angles as the training split, i.e., 1139 frames, while leaving the 67 frames at the remaining 3 viewing angles for evaluating novel view interpolation within the training time period, i.e., 201 frames. We keep the last 22 frames at all 20 viewing angles for evaluating future frame extrapolation, i.e., 440 frames in total. More details are in Appendix A.6.

Baselines: We select the following baselines: 1) NVFi (Li et al., 2023a): This is the closest work to us, but differs from us in two folds. First, NVFi relies on PINN losses to learn physics priors, but we directly learn physical parameters. Second, NVFi adopts NeRF as a backbone, being short in 3D scene geometry and appearance modeling, but our method is amenable to and adopts the powerful 3DGS in nature. 2) T-NeRF (Pumarola et al., 2021). 3) D-NeRF (Pumarola et al., 2021). 4) NSFF (Li et al., 2021). 5) TiNeuVox (Fang et al., 2022). The latter four methods are based on NeRF and designed for novel view interpolation. Therefore they are expected to be rather weak for future frame extrapolation. For a fair and extensive comparison, we also include the following two baselines. 6) DefGS (Yang et al., 2024), 7) 4DGS (Wu et al., 2024), and 8) E-D3DGS (Bae et al., 2024). These very recent deformable 3D Gaussians methods are particularly strong to model dynamic 3D scenes for novel view synthesis using 3DGS as a backbone. 9) DefGS $ _{nvfi} $ . We build this baseline by combining DefGS with the velocity field proposed by NVFi. This baseline has the powerful 3DGS as a backbone as well as the current state-of-the-art NVFi learning strategy. It is trained with exactly the same settings as our method. To demonstrate the flexibility of our framework, we also adopt 4DGS as our auxiliary deformation field, denoted as GVFi $ _{4dgs} $ .

Metrics: The standard metrics PSNR, SSIM, and LPIPS are reported for RGB view synthesis in two tasks: interpolation and future frame extrapolation.

### 4.1 MAIN RESULTS OF FUTURE FRAME EXTRAPOLATION

All methods are trained in a scene-specific fashion. Since NSFF (Li et al., 2021) is not suitable for white-background images, it is not compared on our new Dynamic Multipart dataset. When training our method, we set  $ \Delta t $  to be 2 divided by the training set frame rate. The time  $ t' $  for our auxiliary deformation field  $ f_{defo} $  is set to be 0.7 in the extrapolation task, and target time t is chosen as the frame time. The time difference  $ \Delta t $  is dynamically computed.

Our primary goal is to extrapolate meaningful future frames as a continuum of the last training observations. In our evaluation, we follow Steps #1~#4 in Section 3.3 to extrapolate future frames from the last timestamp of training frames. For benchmarking, we also compare novel view (past frame) interpolation with baselines, but this is less important to us. Particularly, we also follow Steps #1~#4 to render past frames. Though this can also be achieved by progressively querying our translation rotation dynamics module  $ f_{trd} $ , it is inferior due to accumulated errors as detailed in Appendix A.9.

Results & Analysis: Tables 1&2 compare all methods on the five datasets. It can be seen that:

• Compared with NeRF and 3DGS based dynamic scene modeling methods such as T-NeRF/ D-NeRF/ TiNeuVox/ NSFF/ DefGS/ 4DGS/ E-D3DGS, both versions of our method achieve about 10 points higher on PSNR for future frame extrapolation. This means that, without explicitly

<div style="text-align: center;"><img src="imgs/img_in_image_box_254_161_466_271.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">Dynamic Object Dataset</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_501_161_713_271.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">Dynamic Multipart Dataset</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_751_164_962_267.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">Dynamic Indoor Scene Dataset</div>


<div style="text-align: center;">Figure 3: Qualitative results of clustering translation rotation physics parameters. Gaussian kernels can be autonomously grouped into meaningful objects or parts according to their motion patterns.</div>


learning physical information like us, these dynamic methods completely fail to predict the future, highlighting the core value of our method.

• Compared with the closest and also strongest baselines NVFi/DefGS $ _{nvfi} $ , our method is still constantly better than them on all datasets for future frame extrapolation. Notably, on the Dynamic Indoor Scene dataset and our newly collected Dynamic Multipart dataset, there are much more complex motion dynamics such as different objects or parts moving in distinct directions, but our best results are constantly about 3 points higher on PSNR than them. Fundamentally, this is because both NVFi and DefGS $ _{nvfi} $  rely on PINN losses as soft constraints to incorporate physics priors, whereas we directly integrate hard physics by learning translation rotation dynamics system parameters, thus being more effective in learning dynamics.

• Lastly, our framework is indeed amenable to existing deformation fields such as DefGS and 4DGS, and both versions achieve good results for future frame extrapolation on most datasets.

Figure 4 shows qualitative results. More results of the total 5 datasets are in Appendix A.10 / A.11 / A.12 / A.13/A.14. We also report the training/test time, memory cost, etc., in Appendix A.4.

### 4.2 ANALYSIS OF DYNAMICS PARAMETERS

Our core translation rotation dynamics system module is designed to learn per rigid particle's physical parameters. Ideally, for those rigid particles undergoing the same motion pattern such as all surface points of a single rigid part, they should have the same or similar physical parameters. Given this, multi-

<div style="text-align: center;">Table 3: Quantitative results of motion segmentation results on Dynamic Indoor Scene dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>AP $ \uparrow $</td><td style='text-align: center;'>PQ $ \uparrow $</td><td style='text-align: center;'>F1 $ \uparrow $</td><td style='text-align: center;'>Pre $ \uparrow $</td><td style='text-align: center;'>Rec $ \uparrow $</td><td style='text-align: center;'>mIoU $ \uparrow $</td></tr><tr><td style='text-align: center;'>M2F(Cheng et al., 2022)</td><td style='text-align: center;'>65.37</td><td style='text-align: center;'>73.14</td><td style='text-align: center;'>78.29</td><td style='text-align: center;'>94.83</td><td style='text-align: center;'>68.88</td><td style='text-align: center;'>64.42</td></tr><tr><td style='text-align: center;'>D-NeRF(Pumarola et al., 2021)</td><td style='text-align: center;'>57.26</td><td style='text-align: center;'>46.15</td><td style='text-align: center;'>59.02</td><td style='text-align: center;'>56.55</td><td style='text-align: center;'>62.94</td><td style='text-align: center;'>46.58</td></tr><tr><td style='text-align: center;'>NVFi(Li et al., 2023a)</td><td style='text-align: center;'>91.21</td><td style='text-align: center;'>78.74</td><td style='text-align: center;'>93.75</td><td style='text-align: center;'>93.76</td><td style='text-align: center;'>93.74</td><td style='text-align: center;'>67.64</td></tr><tr><td style='text-align: center;'>DefGS(Yang et al., 2024)</td><td style='text-align: center;'>51.73</td><td style='text-align: center;'>57.60</td><td style='text-align: center;'>66.43</td><td style='text-align: center;'>63.21</td><td style='text-align: center;'>70.07</td><td style='text-align: center;'>54.46</td></tr><tr><td style='text-align: center;'>DefGS $ _{nvfi} $</td><td style='text-align: center;'>55.26</td><td style='text-align: center;'>62.75</td><td style='text-align: center;'>69.83</td><td style='text-align: center;'>69.39</td><td style='text-align: center;'>72.91</td><td style='text-align: center;'>56.82</td></tr><tr><td style='text-align: center;'>GVFi (Ours)</td><td style='text-align: center;'>95.82</td><td style='text-align: center;'>93.28</td><td style='text-align: center;'>97.90</td><td style='text-align: center;'>96.21</td><td style='text-align: center;'>99.86</td><td style='text-align: center;'>79.55</td></tr></table>

ple dynamic objects or parts with distinct motions can be automatically segmented based on the similarity of learned physical parameters. By comparison, the prior work NVFi (Li et al., 2023a) can hardly achieve this autonomous dynamic segmentation by its own design, unless an external motion grouping method is applied. To further evaluate this nice property of our method, we conduct the following steps to analyze the learned dynamics parameters.

First, after training our method on the Dynamic Object dataset, Dynamic Multipart dataset, and Dynamic Indoor Scene dataset, for each dynamic scene, we have a set of well-trained canonical 3D Gaussians, an auxiliary deformation field, and our translation rotation dynamics parameters.

Then, we use the auxiliary deformation field  $ f_{defo} $  to deform the canonical 3D Gaussians  $ G_{0} $  to time t = 0.7, which is the maximum time the deformation field can query in our training. At this timestamp, the motions of different objects and parts normally achieve a steady state.

Lastly, we query all the physical parameters at this time, i.e.,  $ \{(\mathbf{P}_{c}^{t},\mathbf{v}_{c}^{t},\mathbf{a}_{c}^{t}),(\mathbf{w}_{p}^{t},\mathbf{\epsilon}_{p}^{t})\} $ . We choose  $ (\|\mathbf{v}_{c}^{t}\|,\mathbf{v}_{c}^{t}/\|\mathbf{v}_{c}^{t}\|,\|\mathbf{w}_{p}^{t}\|,\mathbf{w}_{p}^{t}/\|\mathbf{w}_{p}^{t}\|) $  as the features to cluster Gaussian particles via a simple K-means algorithm. As shown in Figure 3, all Gaussian particles can be grouped into physically meaningful objects or parts according to their actual motion patterns. More results are in Appendix A.17.

We further quantitatively evaluate our motion grouping results on Dynamic Indoor Scene Dataset. In particular, we follow Gaussian Grouping (Ye et al., 2024) to render 2D object segmentation masks for all 30 views over 60 timestamps on all 4 scenes, i.e., 7200 images in total. We compare with D-NeRF, NVFi, DefGS and  $ DefGS_{nvfi} $ . We follow NVFi to obtain segmentation results of D-NeRF and NVFi. For the 3DGS-based baselines, we also adopt OGC (Song & Yang, 2022) to segment Gaussians based on scene flows induced from their learned deformation fields. All imple-

mentation details are in Appendix. Additionally, we include a strong image-based 2D object segmentation method, Mask2Former (Cheng et al., 2022) pre-trained by human annotations on COCO dataset (Lin et al., 2014) as a fully-supervised baseline.

As shown in Table 3, our method achieves almost perfect object segmentation results on all metrics, significantly outperforming all baselines. This shows that our learned physical parameters correctly model object physical motion patterns and can be easily leveraged to identify individual objects according to their motions, without needing any human annotations.

### 4.3 ABLATION STUDY

Our framework mainly comprises 3DGS as the backbone and our core translation rotation dynamics system module, together with an auxiliary deformation field. To verify different choices of our method, we conduct the following three groups of ablation experiments.

(1) Different choices of time difference  $ \Delta t $  in training stage: Given the time interval between two consecutive frames in the training set as  $ \delta t $ , we compare three choices of the time difference  $ \Delta t $  in training stage:  $ \{\delta t, 2\delta t, 3\delta t\} $ . We choose  $ \Delta t = 2\delta t $  in our main experiments.

(2) Removing the auxiliary deformation field  $ f_{defo} $ : In particular, we feed the Gaussian particles at time  $ t' = 0 $  directly into  $ f_{trd} $ , and use the output physical parameters to directly move particles to a target future timestamp t. Note that, here  $ \Delta t $  is meaningless.

(3) Learning time-dependent physical parameters at  $ t^{\prime} $ : Instead of using Equation 4 to derive physical parameters at time  $ t^{\prime} $ , we directly learn them by a 6-layer MLPs as:  $ f_{\mathrm{trd}}^{\prime}(\boldsymbol{P}, t^{\prime}) $ . Theoretically, such a complex function can learn higher order relationships to approximate arbitrary motions than our second-order Equation 4. Nevertheless, it would be more challenging to learn and unable to guarantee physical parameters of the same motion pattern to be consistent over time.

Results & Analysis: Table 4 shows all ablation results for future frame extrapolation on our new Dynamic Multipart dataset. It can be seen that: 1) The greatest impact is caused by the removal of the deformation field  $ f_{defo} $ . Although this deformation field itself is unable to learn physics, it significantly aids our core translation rotation dynamics system module to learn physical parameters given the motion information. 2) The choice of time difference  $ \Delta t $  is also important. Once it is as small as the interval between two consecutive frames, the performance



<div style="text-align: center;">Table 4: Quantitative results of ablation studies on Dynamic Multipart dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td colspan="3">f_{defo} f_{trd}</td><td colspan="3">Extrapolation</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'>PSNR $ \uparrow $</td><td style='text-align: center;'>SSIM $ \uparrow $</td><td style='text-align: center;'>LPIPS $ \downarrow $</td><td style='text-align: center;'></td><td style='text-align: center;'></td></tr><tr><td style='text-align: center;'>(1)  $ \delta t $</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>29.441</td><td style='text-align: center;'>0.984</td><td style='text-align: center;'>0.013</td></tr><tr><td style='text-align: center;'>(1)  $ 2\delta t $</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>30.721</td><td style='text-align: center;'>0.986</td><td style='text-align: center;'>0.012</td></tr><tr><td style='text-align: center;'>(1)  $ 3\delta t $</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>30.246</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.012</td></tr><tr><td style='text-align: center;'>(2) -</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>27.081</td><td style='text-align: center;'>0.981</td><td style='text-align: center;'>0.018</td></tr><tr><td style='text-align: center;'>(3)  $ 2\delta t $</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>29.986</td><td style='text-align: center;'>0.985</td><td style='text-align: center;'>0.012</td></tr></table>

drops apparently, because the motion in short intervals is too subtle to be distinguished. For example, a rotation may be learned as a translation. However, if  $ \Delta t $  is too large, the appearance fitting could be sacrificed, so the performance is slightly weaker. 3) If physical parameters are learned but not derived, the lack of physics consistency will influence motion learning.

Detailed ablation settings and results are in Appendix A.7. More ablations of  $ \Delta t $  on four datasets, and more ablations of using 1st-/2nd-order relationships in Equation 4 are in Appendix A.8.

## 5 CONCLUSION

In this paper, we have demonstrated that complex motion dynamics can be explicitly learned just from multi-view RGB videos without needing additional human labels such as object types and masks. This is achieved by a new generic framework that simultaneously models 3D scene geometry, appearance and physics by extending the appealing 3D Gaussian Splatting technique. In contrast to existing works which usually rely on PINN losses as soft constraints to learn physics priors, we instead directly learn a complete set of physical parameters to govern the motion pattern of each 3D rigid particle in space via our core translation rotation dynamics system module. Extensive experiments on three public dynamic datasets and a newly created dynamic multipart dataset have shown the extraordinary performance of our method in the challenging task of future frame extrapolation over all baselines. In addition, the learned physical parameters can be directly used to segment objects or parts according to the similarity of parameters.

<div style="text-align: center;"><img src="imgs/img_in_image_box_196_168_1008_1421.jpg" alt="Image" width="66%" /></div>


<div style="text-align: center;">Figure 4: Qualitative results of RGB view synthesis for interpolation and extrapolation tasks.</div>


## REFERENCES

Jeongmin Bae, Seoha Kim, Youngsik Yun, Hahyun Lee, Gun Bang, and Youngjung Uh. Per-Gaussian Embedding-Based Deformation for Deformable 3D Gaussian Splatting. ECCV, 2024.

Daniele Baieri, Stefano Esposito, Filippo Maggioli, and Emanuele Rodolà. Fluid Dynamics Network: Topology-Agnostic 4D Reconstruction via Fluid Dynamics Priors. arXiv:2303.09871, 2023.

Jonathan T Barron, Keunhong Park, Steven M Seitz, and Ricardo Martin-brualla. Nerfies: Deformable Neural Radiance Fields. ICCV, 2021.

Hongrui Cai, Wanquan Feng, Xuetao Feng, Yan Wang, and Juyong Zhang. Neural Surface Reconstruction of Dynamic Scenes with Monocular RGB-D Camera. NeurIPS, 2022.

Ang Cao and Justin Johnson. HexPlane: A Fast Representation for Dynamic Scenes. CVPR, 2023.

Nithin Chalapathi, Yiheng Du, and Aditi S Krishnapriyan. Scaling physics-informed hard constraints with mixture-of-experts. ICLR, 2024.

Zhiqin Chen and Hao Zhang. Learning Implicit Fields for Generative Shape Modeling. CVPR, 2019.

Bowen Cheng, Ishan Misra, Alexander G. Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention Mask Transformer for Universal Image Segmentation. CVPR, 2022.

Julian Chibane, Aymen Mir, and Gerard Pons-Moll. Neural Unsigned Distance Fields for Implicit Function Learning. NeurIPS, 2020.

Mengyu Chu, Lingjie Liu, Quan Zheng, Erik Franz, Hans Peter Seidel, Christian Theobalt, and Rhaleb Zayer. Physics informed neural fields for smoke reconstruction with sparse data. TOG, 2022.

Yitong Deng, Hong-Xing Yu, Jiajun Wu, and Bo Zhu. Learning Vortex Dynamics for Fluid Inference and Prediction. ICLR, 2023.

Yilun Du, Yinan Zhang, and Joshua B Tenenbaum. Neural Radiance Flow for 4D View Synthesis and Video Processing. ICCV, 2021.

Jiemin Fang, Xinggang Wang, and Matthias Nießner. Fast Dynamic Radiance Fields with Time-Aware Neural Voxels. SIGGRAPH Asia, 2022.

Erik Franz, Barbara Solenthaler, and Nils Thuerey. Learning to Estimate Single-view Volumetric Flow Motions without 3D Supervision. ICLR, 2023.

Sara Fridovich-Keil, Giacomo Meanti, Frederik Warburg, Benjamin Recht, and Angjoo Kanazawa. K-Planes: Explicit Radiance Fields in Space, Time, and Appearance. CVPR, 2023.

Chen Gao, Ayush Saraf, Johannes Kopf, and Jia-Bin Huang. Dynamic View Synthesis from Dynamic Monocular Video. ICCV, 2021.

Zhongkai Hao, Chengyang Ying, Hang Su, Jun Zhu, Jian Song, and Ze Cheng. Bi-level Physics-Informed Neural Networks for PDE Constrained Optimization Using Broyden's Hypergrandients. ICLR, 2023.

Alvaro Sanchez-Gonzalez Jonathan, Godwin Tobias, Pfaff Rex, Ying Jure, and Peter W Battaglia. Learning to Simulate Complex Physics with Graph Networks. ICML, 2020.

Bernhard Kerbl, Université Côte, Georgios Kopanas, Université Côte, Thomas Leimkühler, Maxplanck-institut Informatik, and G R Aug. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH, 2023.

Jiahui Lei, Yijia Weng, Adam Harley, Leonidas Guibas, and Kostas Daniilidis. MoSca: Dynamic Gaussian Fusion from Casual Videos via 4D Motion Scaffolds. arXiv:2405.17421, 2024.

Jinxi Li, Ziyang Song, and Bo Yang. NVFi: Neural Velocity Fields for 3D Physics Learning from Dynamic Videos. NeurIPS, 2023a.

Tianye Li, Mira Slavcheva, Michael Zollhoefer, Simon Green, Christoph Lassner, Changil Kim, Tanner Schmidt, Steven Lovegrove, Michael Goesele, and Zhaoyang Lv. Neural 3D Video Synthesis From Multi-View Video. CVPR, 2022.

Xuan Li, Yi-Ling Qiao, Peter Yichen Chen, Krishna Murthy Jatavallabhula, Ming Lin, Chenfanfu Jiang, and Chuang Gan. PAC-NeRF: Physics Augmented Continuum Neural Radiance Fields for Geometry-Agnostic System Identification. ICLR, 2023b.

Zhan Li, Zhang Chen, Zhong Li, and Yi Xu. Spacetime Gaussian Feature Splatting for Real-Time Dynamic View Synthesis. CVPR, 2024.

Zhengqi Li, Simon Niklaus, Noah Snavely, and Oliver Wang. Neural Scene Flow Fields for Space-Time View Synthesis of Dynamic Scenes. CVPR, 2021.

Marten Lienen, David Lüdke, Jan Hansen-Palmus, and Stephan Günnemann. From Zero to Turbulence: Generative Modeling for 3D Flow Simulation. ICLR, 2024.

Tsung-Yi Lin, Michael Maire, Serge Belongie, Lubomir Bourdev, Ross Girshick, James Hays, Pietro Perona, Deva Ramanan, C. Lawrence Zitnick, and Piotr Dollár. Microsoft COCO: Common Objects in Context. ECCV, 2014.

Youtian Lin, Zuozhuo Dai, Siyu Zhu, and Yao Yao. Gaussian-Flow: 4D Reconstruction with Dynamic 3D Gaussian Particle. CVPR, 2024.

Fangfu Liu, Hanyang Wang, Shunyu Yao, and Shengjun Zhang. Physics3D: Learning Physical Properties of 3D Gaussians via Video Diffusion. arXiv:2406.04338, 2024a.

Isabella Liu, Hao Su, and Xiaolong Wang. Dynamic Gaussians Mesh: Consistent Mesh Reconstruction from Monocular Videos. arXiv:2404.12379, 2024b.

Yu-Lun Liu, Chen Gao, Andreas Meuleman, Hung-Yu Tseng, Ayush Saraf, Changil Kim, Yung-Yu Chuang, Johannes Kopf, and Jia-Bin Huang. Robust Dynamic Radiance Fields. CVPR, 2023.

Zhicheng Lu, Xiang Guo, Le Hui, Min Yang, Xiao Tang, Feng Zhu, and Yuchao Dai. 3D Geometry-aware Deformable Gaussian Splatting for Dynamic View Synthesis. CVPR, 2024.

Lars Mescheder, Michael Oechsle, Michael Niemeyer, Sebastian Nowozin, and Andreas Geiger. Occupancy Networks: Learning 3D Reconstruction in Function Space. CVPR, 2019.

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis. ECCV, 2020.

Siddhartha Mishra and Roberto Molinaro. Estimates on the generalization error of physics-informed neural networks for approximating PDEs. IMA Journal of Numerical Analysis, 2023.

Jeong Joon Park, Peter Florence, Julian Straub, Richard Newcombe, and Steven Lovegrove. DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation. CVPR, 2019.

Keunhong Park, Utkarsh Sinha, Peter Hedman, Jonathan T. Barron, Sofien Bouaziz, Dan B. Goldman, Ricardo Martin-Brualla, and Steven M. Seitz. HyperNeRF: A Higher-Dimensional Representation for Topologically Varying Neural Radiance Fields. SIGGRAPH Asia, 2021.

Sungheon Park, Minjung Son, Seokhwan Jang, Young Chun Ahn, Ji-Yeon Kim, and Nahyup Kang. Temporal Interpolation Is All You Need for Dynamic Neural Radiance Fields. CVPR, 2023.

Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-NeRF: Neural Radiance Fields for Dynamic Scenes. CVPR, 2021.

Yi-Ling Qiao, Alexander Gao, and Ming C. Lin. NeuPhysics: Editable Neural Geometry and Physics from Monocular Videos. NeurIPS, 2022.

M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 2019.

Maziar Raissi, Alireza Yazdani, and George Em Karniadakis. Hidden fluid mechanics: Learning velocity and pressure fields from flow visualizations. Science, 2020.

Johannes L. Schonberger and Jan-Michael Frahm. Structure-from-Motion Revisited. CVPR, 2016.

Ziyang Song and Bo Yang. OGC: Unsupervised 3D Object Segmentation from Rigid Dynamics of Point Clouds. NeurIPS, 2022.

Fengrui Tian, Shaoyi Du, and Yueqi Duan. MonoNeRF: Learning a Generalizable Dynamic Radiance Field from Monocular Videos. ICCV, 2023.

Edgar Tretschk, Ayush Tewari, Vladislav Golyanik, Michael Zollhöfer, Christoph Lassner, and Christian Theobalt. Non-Rigid Neural Radiance Fields: Reconstruction and Novel View Synthesis of a Dynamic Scene from Monocular Video. ICCV, 2021.

Chaoyang Wang, Ben Eckart, Simon Lucey, and Orazio Gallo. Neural Trajectory Fields for Dynamic Novel View Synthesis. arXiv:2105.05994, 2021.

Yiming Wang, Siyu Tang, and Mengyu Chu. Physics-Informed Learning of Characteristic Trajectories for Smoke Reconstruction. SIGGRAPH, 2024.

William F. Whitney, Tatiana Lopez-Guevara, Tobias Pfaff, Yulia Rubanova, Thomas Kipf, Kimberly Stachenfeld, and Kelsey R. Allen. Learning 3D Particle-based Simulators from RGB-D Videos. ICLR, 2024.

Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4D Gaussian Splatting for Real-Time Dynamic Scene Rendering. CVPR, 2024.

Wenqi Xian, Jia-Bin Huang, Johannes Kopf, and Changil Kim. Space-time Neural Irradiance Fields for Free-Viewpoint Video. CVPR, 2021.

Haotian Xue, Antonio Torralba, Daniel LK Yamins, Joshua B. Tenenbaum, Yunzhu Li, and Hsiao-Yu Tung. 3D-IntPhys: Learning 3D Visual Intuitive Physics for Fluids, Rigid Bodies, and Granular Materials. NeurIPS, 2023.

Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3D Gaussians for High-Fidelity Monocular Dynamic Scene Reconstruction. CVPR, 2024.

Mingqiao Ye, Martin Danelljan, Fisher Yu, and Lei Ke. Gaussian grouping: Segment and edit anything in 3d scenes. In ECCV, 2024.

Jae Shin Yoon, Kihwan Kim, Orazio Gallo, Hyun Soo Park, and Jan Kautz. Novel View Synthesis of Dynamic Scenes with Globally Coherent Depths from a Monocular Camera. CVPR, 2020.

Meng You and Junhui Hou. Decoupling Dynamic Monocular Videos for Dynamic View Synthesis. arXiv:2304.01716, 2023.

Tianyuan Zhang, Hong-Xing Yu, Rundi Wu, Brandon Y. Feng, Changxi Zheng, Noah Snavely, Jiajun Wu, and William T. Freeman. PhysDreamer: Physics-Based Interaction with 3D Objects via Video Generation. arXiv:2404.13026, 2024.

Zhiyuan Zhao, Xueying Ding, and B. Aditya Prakash. PINNsFormer: A Transformer-Based Framework For Physics-Informed Neural Networks. ICLR, 2024.

Licheng Zhong, Hong-xing Yu, Jiajun Wu, and Yunzhu Li. Reconstruction and Simulation of Elastic Objects with Spring-Mass 3D Gaussians. ECCV, 2024.

Matthias Zwicker, Hanspeter Pfister, Jeroen van Baar, and Markus H. Gross. EWA volume splatting. IEEE Visualization, 2001.

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
