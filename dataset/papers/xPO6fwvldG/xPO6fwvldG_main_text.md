# UNIRESTORE3D: A Scalable Framework for General Shape Restoration

Yuang Wang $ ^{1} $  Yujian Zhang $ ^{1} $  Sida Peng $ ^{1} $  Xingyi He $ ^{1} $  Haoyu Guo $ ^{1} $  Yujun Shen $ ^{2} $  Hujun Bao $ ^{1} $  Xiaowei Zhou $ ^{1*} $ 

 $ ^{1} $ State Key Lab of CAD&CG, Zhejiang University  $ ^{2} $ Ant Group {wangyuang, xwzhou}@zju.edu.cn

## ABSTRACT

Shape restoration aims to recover intact 3D shapes from defective ones, such as those that are incomplete, noisy, and low-resolution. Previous works have achieved impressive results in shape restoration subtasks thanks to advanced generative models. While effective for specific shape defects, they are less applicable in real-world scenarios involving multiple defect types simultaneously. Additionally, training on limited subsets of defective shapes hinders knowledge transfer across restoration types and thus affects generalization. In this paper, we address the task of general shape restoration, which restores shapes with various types of defects through a unified model, thereby naturally improving the applicability and scalability. Our approach first standardizes the data representation across different restoration subtasks using high-resolution TSDF grids and constructs a large-scale dataset with diverse types of shape defects. Next, we design an efficient and noise-robust hierarchical shape generation model that enables effective defective shape understanding and intact shape generation. Moreover, we propose a scalable training strategy for efficient model training. The capabilities of our proposed method are demonstrated across multiple shape restoration subtasks and validated on various datasets, including Objaverse, ShapeNet, GSO, and ABO.

## 1 INTRODUCTION

Restoring complete and intact shapes from defective ones is essential for applications in virtual reality, robotics, and content generation. Defective shapes can arise from various sources, including depth sensor noise (Tölgyessy et al., 2021), ill-posed 3D reconstruction (Wu et al., 2023), intrinsic defects in geometric representations (Feng & Crane, 2024), and so on. Consequently, shape restoration encompasses various repair goals, such as completion, super-resolution, and denoising, as illustrated in Fig. 1. These diverse tasks require model capabilities such as semantic understanding of highly incomplete and noisy shapes without additional inputs, robustness to varying degrees of incompleteness and noise, and the ability to produce diverse restoration results that balance quality and fidelity — making it challenging to design a unified model for general shape restoration.

To simplify the problem, many existing works focus on specific shape restoration goals other than addressing general shape restoration. By leveraging advanced deep learning techniques, such as regression models (Dai et al., 2017; 2020; Rao et al., 2022; Huang et al., 2023) and generative models (Mittal et al., 2022; Cheng et al., 2023; Warburg et al., 2023; Chu et al., 2024; Ju et al., 2024), these methods have achieved remarkable results on specific restoration goals. However, focusing on specific goals limits the models' capabilities in two key aspects. First, the defective shapes often contain various artifacts simultaneously, such as incompleteness and noise from ill-posed 3D reconstruction. Models that handle only certain artifacts have limited performance and applicability in these scenarios. Second, models aimed at specific goals are trained on limited types of data, and their learned object priors are not shared across tasks, resulting in less training data diversity compared to general shape restoration.

In this paper, we aim to address the task of general shape restoration, which targets the repair of various forms of defective geometries with a unified model. Compared with previous works on

<div style="text-align: center;"><img src="imgs/img_in_image_box_215_165_1005_452.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 1: We propose UniRestore3D for general shape restoration. It can restore various types of defective 3D shapes under arbitrary poses, generate multiple possible restoration results, and support complex multi-object scenes.</div>


specific restoration subtasks, the general shape restoration setting has two main advantages. First, it is more user-friendly. Users can apply a single model to repair various forms of defective geometries, which is significantly more convenient than first classifying the shape defect type and then using a mixture of corresponding restoration models. Second, this task enables us to exploit more data. By training on larger and more diverse datasets, the model can learn a more generalized shape prior, which is beneficial for handling in-the-wild defective shapes. As shown in Sec. 5.4, our model has better performance than models trained on specific subtasks.

To handle the general shape restoration task, we propose a novel, scalable framework that consists of three key components: a unified shape restoration dataset, an efficient and noise-robust shape restoration model, and a scalable training strategy. Regarding training data, as illustrated in Fig. 2, we construct synthetic defective-intact shape pairs with tailored construction methods for different subtasks, which realistically simulate real-world impairment scenarios. With the availability of large-scale datasets, model design and training strategies still face several technical challenges. First, restoring an intact shape solely from a defective one relies on the model's semantic understanding of the defective shape and its ability to efficiently generate high-quality shape. We design a hierarchical latent diffusion model (H-LDM) that achieves multi-scale encoding of defective shapes and efficient generation of intact ones, as shown in Fig. 3. Besides, two technical challenges remain. First, robustness to irregular noise in defective shapes; second, to ensure high fidelity to the existing input shape details, it is necessary to use high-resolution defective shapes as model inputs. This necessitates pre-compression of the defective shapes before training the H-LDM; otherwise, reading and encoding high-resolution shapes would unacceptably slow down training. Our key insight to solve these challenges is to construct a unified representation for both defective and intact shapes. Specifically, we first learn representations of intact shapes on large-scale data. Then, we train an defective shape encoder whose encodings are explicitly aligned to their corresponding intact ones. In this way, we achieve pre-compression of defective shapes, while enhancing noise robustness, as the feature alignment step enforces defective shape denoising.

The effectiveness of our method is validated on datasets including Objaverse (Deitke et al., 2023b;a), ShapeNet (Chang et al., 2015), ABO (Collins et al., 2022), GSO (Downs et al., 2022) and ScanNet (Rao et al., 2022). Our approach achieves SOTA results in tasks including noise-free shape completion, noisy shape refinement and completion. Related ablation studies have confirmed the effectiveness of key modules in our model.

## 2 RELATED WORK

### 2.1 3D SHAPE RESTORATION

Shape restoration encompasses various subtasks, each of which has been extensively studied in previous research. As illustrated in Fig. 2, the shape to be restored may exhibit local or extensive

<div style="text-align: center;"><img src="imgs/img_in_image_box_215_158_1008_458.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 2: Dataset curation and examples of shape restoration subtasks. We create (defective, intact) shape pairs by adopting different impairment strategies for different subtasks.</div>


missing regions, and the known areas may be noisy or noise-free. Most existing approaches focus on shape restoration under specific conditions, lacking general applicability. For example, surface reconstruction methods (Kazhdan & Hoppe, 2013; Peng et al., 2020; Huang et al., 2023) aim to restore complete geometry from noisy but fairly complete point clouds. Some works (Wu et al., 2018; Cui et al., 2023) target the restoration of complete point clouds from noisy ones with extensive incompleteness. Another line of works (Mittal et al., 2022; Li et al., 2023; Cui et al., 2024) train shape generative models to complete noise-free incomplete shapes. Among these, diffusion model-based approaches rely on SDEdit (Meng et al., 2022) or blended diffusion (Lugmayr et al., 2022; Avrahami et al., 2022; 2023) mechanisms to complete the missing regions, which presents certain limitations in their use. Some works (Dai et al., 2017; Rao et al., 2022; Chu et al., 2024; Liu et al., 2024a; Galvis et al., 2024) aim to restore complete, noise-free TSDF grids from noisy and defective ones. However, they are trained on limited defect scenarios, which hinders their generality. Feng & Crane (2024) proposes a signed heat method for approximating the SDF of locally corrupted geometries, but does not leverage semantic information, thus limited to surface refinement and local restoration. Our goal is to build a conditional generative model for efficient, high-quality shape restoration that supports different types of impairment. Moreover, we emphasize the ability of accepting high-resolution defective shapes as inputs, which is necessary for high fidelity restoration.

### 2.2 3D SHAPE PRIOR

Shape restoration can be regarded as a form of shape prior for repairing defective geometries. Previous works have explored various types of shape priors and applied them to 3D reconstruction. By constraining 3D reconstruction to the latent or parameter space of shape prior models, these approaches help to avoid low-quality reconstruction results under ill-posed conditions (Zhu et al., 2018; Lin et al., 2019; Sucar et al., 2020; Liu et al., 2020; Yang et al., 2021; Sun et al., 2024). Among these studies, some object-level approaches can reconstruct complete 3D models under sparse observations but are limited to specific categories (Yang et al., 2022; Sucar et al., 2020). Part-level shape priors (Rao et al., 2022; Bokhovkin & Dai, 2023; Sun et al., 2022) are category-agnostic but lack semantic understanding and thus cannot handle large missing regions. 3D diffusion models (DMs) have also been used as shape priors for 3D reconstruction (Warburg et al., 2023; Yang et al., 2023); however, they are still limited to modeling specific categories or small missing regions. General shape restoration aims to provide a general shape prior that is not confined to specific categories. We achieve this by training a conditional shape generative model, which directly maps the defective shapes onto the manifold of plausible shapes.

### 2.3 3D SHAPE GENERATIVE MODELS

3D shape generative models can be categorized based on the primary 3D representation utilized by the network, e.g., multiple 2D planes, hierarchical 3D structures, no explicit 3D representations, and hybrid approaches. To enhance generative modeling efficiency, these methods focus on leveraging

the sparsity of 3D data. Methods based on multi-view images (Szymanowicz et al., 2023; Shi et al., 2024; Liu et al., 2024b) or tri-planes (Chan et al., 2022; Gupta et al., 2023; Shue et al., 2023) exploit the sparsity of 3D data by reducing the dimensionality of the core representation, which effectively represents individual assets. Still, they are less efficient when handling complex, large-scale scenes. Approaches that discard 3D inductive bias and use latent features (Zhang et al., 2023; 2024) struggle with scaling feature sets and controlling their spatial distribution, limiting their applicability to large scenes. Hierarchical generative models based on sparse voxel hierarchies (Zheng et al., 2023; Ren et al., 2024) explicitly leverage sparsity by generating in a coarse-to-fine manner, which amortizes the generation process across levels, supports dynamic control of geometric details, and scales more efficiently to large scenes. The hierarchical structure naturally introduces multi-level conditional signals, facilitating a nuanced understanding of conditional inputs. We adopt a hierarchical generative model as our foundation and train a conditional model for general shape restoration.

## 3 GENERAL SHAPE RESTORATION

General shape restoration aims to restore complete and clean shapes x from incomplete and noisy inputs  $ x_{c} $ , which follows the conditional distribution  $ P(\mathbf{x} \mid \mathbf{x}_{c}, \mathbf{c}) $  with optional conditions c like text. Its generality is demonstrated by its extensive support for defective inputs, encompassing regular or irregular geometric deficiencies and noise from sources such as sparse viewpoints, sensor noise, and reconstruction flaws. The main challenges include understanding various incomplete and noisy inputs, identifying regions needing restoration, accommodating shapes under different poses, and preserving the original structure and semantics while generating high-quality restoration results. Based on the level of noise and incompleteness present in the input geometry, we roughly categorize the general shape restoration into four subtasks: noise-free completion, super-resolution, noisy completion, and noisy refinement, among which previous works (Rao et al., 2022; Chu et al., 2024; Li et al., 2023) only focus on solving single specific subtasks.

Dataset creation. We represent both defective and intact shapes using the TSDF grid for its versatility. To construct numerous (defective, intact) shape pairs for model training, we employ different approaches for subtasks as shown in Fig. 2:

(1) Noise-free completion: Randomly sample partial shapes with varying levels of incompleteness from complete shapes, akin to image inpainting.

(2) Super-resolution: Subsample complete shapes to lower resolutions to create defective shapes of varying difficulty.

(3) Noisy completion: Render depth maps from sparse viewpoints, apply TSDF fusion to reconstruct incomplete shapes, and introduce varying degrees of noise into depth maps and camera poses to simulate geometric noises and missing scenarios.

(4) Noisy refinement: Similar to noisy completion but fuse input shapes from more viewpoints, resulting in complete structures with noisy surface details.

We randomly perturb model poses to enhance diversity and avoid canonical pose modeling. Since super-resolution does not conflict with the other subtasks, we randomly use different resolutions in the other three subtasks to enhance data diversity and increase task difficulty. Applying these strategies to datasets including Objaverse, ShapeNet, ABO, and GSO yields a large-scale dataset with approximately 120k object models and 800k shape pairs for training and evaluation.

## 4 METHOD

We devise a conditional shape generative model to achieve general shape restoration, which restores intact shapes from defective ones. Our model consists of three modules: a hierarchical variational autoencoder (H-VAE) for intact shapes compression, a hierarchical noise-robust encoder for processing defective shapes, and a hierarchical latent diffusion model (H-LDM) for conditional generation. We adopt a hierarchical approach to shape encoding and generation, enabling a multi-level understanding of defective geometries and efficient shape restoration. Fig. 3 presents the inference pipeline of shape restoration. Specifically, we first encode the defective input into multi-level sparse feature grids utilizing the noise-robust encoder. These feature grids serve as conditional inputs at different levels of the H-LDM, which generates sparse latent grids of intact shapes that can be further decoded to sparse TSDF grids from low to high resolution. In the following sections, we will

<div style="text-align: center;"><img src="imgs/img_in_image_box_215_163_1006_446.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 3: Inference pipeline of the proposed shape restoration model. We leverage a conditional hierarchical latent diffusion model (H-LDM) to restore complete and clean shapes from incomplete and noisy inputs. Defective input shapes are encoded into multi-level sparse features (two levels here) with a hierarchical noise-robust encoder, which acts as conditional signals of the H-LDM to generate multi-level sparse feature grids, and finally decoded to the restored shapes.</div>


first introduce the hierarchical encoding of intact and defective shapes and the proposed scalable training strategy in Sec. 4.1 and then present the conditional H-LDM in Sec. 4.2.

### 4.1 HIERARCHICAL SHAPE ENCODING

The conditional H-LDM for shape restoration relies on compact encodings of intact and defective shapes. We propose a unified hierarchical shape encoding pipeline for both the intact and defective shapes, which respectively acts as the generation targets and conditional inputs of the H-LDM. Here we first introduce the probabilistic modeling and architectural design, and then the scalable two-stage training strategy.

Probabilistic modeling. We use a cascaded VAE as in Razavi et al. (2019); Vahdat & Kautz (2020) to learn the multi-level latent representations jointly, instead of learning individual VAEs for shapes at different levels individually as in Ren et al. (2024). This design simplifies the training process and facilitates learning inter-level dependencies across multi-level latents. The H-VAE consists of a series of cascaded encoders  $ E = \{E^{1}, \ldots, E^{L}\} $  across levels and independent decoders  $ D = \{D^{1}, \ldots, D^{L}\} $  at each level, as shown in the top row of Fig. 4. Specifically, the H-VAE encoders learn the approximate posterior  $ q_{\phi}(\mathbf{z} \mid \mathbf{x}) = \prod_{i} q_{\phi}(\mathbf{z}^{i} \mid \mathbf{z}^{<i}, \mathbf{x}) $ , where each level's latent  $ z^{i} $  is built upon the  $ z^{<i} $  from previous levels. The H-VAE decoders learn the likelihood  $ p_{\phi}(\mathbf{x} \mid \mathbf{z}) = \prod_{i} p_{\phi}(\mathbf{x}^{i} \mid \mathbf{z}^{>i}) $ , where shape  $ x^{i} $  at each level is decoded not only from the corresponding latent  $ z^{i} $  but also based on the coarser levels  $ z^{>i} $ .

Patch-wise encoding. To enhance the generalization of shape encoders on both intact and impaired shapes, we employ a patch-wise encoding strategy. Previous methods (Mittal et al., 2022; Yan et al., 2022) utilize patch-wise encoding to ensure that encoders can support both complete and partial shapes without noise. We further extend the patch-wise encoder to simultaneously encode noisy and noise-free 3D shapes, reducing the risk of out-of-distribution (OOD) occurrences for noisy shape encodings. Specifically, we divide the 3D shapes into multiple non-overlapping patches and apply the aforementioned cascaded encoder to encode each patch into hierarchical sparse features. The encodings of all patches at each level are then concatenated to form the latent representations of the entire shape. The decoders at different levels are not restricted to patch-wise decoding, thereby avoiding inconsistencies at the boundaries between patches.

Scalable training strategy. Given a pre-trained H-VAE of intact shapes, we can efficiently train an unconditional H-LDM for shape generation. However, for a conditional H-LDM, we still need to train a defective shape encoder along with the H-LDM, which is extremely inefficient due to the need to load high-resolution defective shapes online. To mitigate this issue, we learn defective shape encodings in advance using a two-stage training strategy as shown in Fig. 4. Our key insight is to share an unified representation between intact and defective shapes by learning the hierarchical

<div style="text-align: center;"><img src="imgs/img_in_image_box_214_165_1006_446.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 4: Learning hierarchical shape encodings. We learn hierarchical shape encodings of intact and defective shapes in two stages. These encodings are later used to train the conditional diffusion model for shape restoration. In stage-1 (top row), we train the hierarchical VAE (H-VAE) on intact shapes to learn hierarchical latent representations. In stage-2 (bottom left), we fine-tune a noise-robust encoder for encoding defective shapes through multi-level feature alignment between defective and intact encodings. The noise-robust encoder can robustly encode and pre-denoise the noisy defective shape as illustrated by the decoded shape (bottom right).</div>


encodings of intact shapes first and then align the defective shape encodings to the intact ones. In this way, we can learn a defective shape encoder in advance and pre-compress shapes before LDM training. Moreover, the forced alignment leads to a noise-robust encoder as introduced below.

Noise-robust encoder fine-tuning. A naive approach to encoding impaired shapes is directly using the H-VAE encoders trained on intact shapes. However, given that the impaired shapes contain various types of noise that do not conform to a well-defined distribution, this approach is susceptible to the influence of OOD samples. To address this, we further perform a fine-tuning stage on the cascaded encoder of the pre-trained H-VAE and turn it into a separate noise-robust defective shape encoder  $ E_{c} $ . Specifically, as shown in the bottom row of Fig. 4, given a pair of defective and intact shapes, we encode the defective shape with  $ E_{c} $  to multi-level encodings  $ z_{c} = \{z_{c}^{1}, \ldots, z_{c}^{L}\} $ , and encode the intact shape with the frozen pre-trained H-VAE encoder E to  $ z = \{z^{1}, \ldots, z^{L}\} $ . We train the noise-robust encoder  $ E_{c} $  by minimizing the discrepancy between these two sets of encodings. This fine-tuning stage not only enhances the robustness of the encoding of defective shapes but also pre-denoises the noisy inputs as shown in the bottom right of Fig. 4, thereby reducing the training difficulty of the subsequent conditional generative model.

### 4.2 HIERARCHICAL LATENT DIFFUSION MODEL

Given the hierarchical sparse encodings of the intact and defective shapes, we train a conditional hierarchical latent diffusion model (H-LDM) as the shape restoration model, which progressively generates 3D sparse structures and corresponding geometric attributes from low to high resolution. At each level, a sparse latent grid is generated and decoded to the concrete geometry, which serves as the sparse structure of the next level at higher resolution.

Specifically, given an intact shape x and a defective shape  $ x_{c} $ , we encode them separately to multilevel sparse encodings  $ z = \{z^{1}, \ldots, z^{L}\} $  and  $ z_{c} = \{z_{c}^{1}, \ldots, z_{c}^{L}\} $  as introduced in Sec. 4.1. We aim for the conditional H-LDM to learn the following probability distribution, which generates multilevel sparse latent grids from low (i = L) to high resolution (i = 1):

 $$ p_{\theta}(\mathbf{z}\mid\mathbf{z}_{\mathbf{c}},\mathbf{c})=\prod_{i=L}^{1}p_{\theta}^{i}(\mathbf{z}^{i}\mid\mathbf{z}^{>i},\mathbf{z}_{\mathbf{c}}^{i},\mathbf{c}), $$ 

where c are optional conditions besides the defective shape  $ x_{c} $ , such as text. Notably, at each level, besides the conditional encoding  $ z_{c}^{i} $  of the defective shape, we also rely on the previously generated coarser level latent grids  $ z^{>i} $  as additional conditions to provide a global context for the generation of finer latent grids  $ z^{i} $ , which introduces more details.

We model each level's denoising process  $ p_{\theta}^{i} $  using a Sparse U-Net based denoiser, where its dependent sparse structure is obtained by decoding the sparse latent grid  $ z^{i+1} $  from the previous level through the H-VAE decoder.

The H-LDM balances generation efficiency and quality thanks to the marrying of spatial sparsity and diffusion modeling on latent spaces. Moreover, it can explicitly reason over multi-level features of defective shapes through hierarchical conditioning for better restoration.

### 4.3 TRAINING AND IMPLEMENTATION DETAILS

Our pipeline consists of three trainable modules. For the H-VAE, we train all levels jointly using the standard ELBO objective (Kingma, 2013) at each level. For the H-LDM, we adopt a continuous-time diffusion model with v-parameterization and use the simplified training objective (Ho et al., 2020) at each level. We fine-tune the noise-robust encoder with the following loss:

 $$ \mathcal{L}_{\mathrm{a l i g n}}=\sum_{i=1}^{L}\left\|\mathbb{E}_{q_{\phi_{c}}(\mathbf{z}_{\mathbf{c}}^{i}|\mathbf{z}_{\mathbf{c}}^{<i},\mathbf{x}_{\mathbf{c}})}[\mathbf{z}_{\mathbf{c}}^{i}]-\mathbb{E}_{q_{\phi}(\mathbf{z}^{i}|\mathbf{z}^{<i},\mathbf{x})}[\mathbf{z}^{i}]\right\|_{1}, $$ 

which optimizes the noise-robust encoder to align the multi-level encodings of the defective shapes to the intact ones. We compute the loss only on sparse voxels shared by both defective and intact shapes. We employ sparse convolution (Tang et al., 2023) in all three modules for efficient processing of sparse voxel grids. However, at the coarsest level, the voxel grid becomes a dense one, where we use dense convolution for further processing. More details are provided in the appendix.

## 5 EXPERIMENTS

To validate the effectiveness of our proposed method, we conduct evaluations on several different shape restoration subtasks. We use an unconditional model (i.e., with only shape condition) for shape restoration without additional conditions like text to demonstrate the model's understanding capability of the impaired shapes. We first conduct experiments on our proposed general shape restoration task in Sec. 5.1, and then verify our model's effectiveness on existing benchmarks of shape restoration subtasks in Secs. 5.2 and 5.3. We train the model on a dataset that combines Objaverse and ShapeNet, comprising approximately 120k object models and 800k (defective, intact) shape pairs. Intact shapes are represented with  $ 256^{3} $  TSDF grids, while impaired shapes, originally at various resolutions, are upsampled to  $ 256^{3} $  and used as conditional inputs for the model. More details on data preprocessing and results postprocessing are provided in the appendix.

### 5.1 GENERAL SHAPE RESTORATION

The proposed task of general shape restoration contains different types of shape impairments. We evaluate the model separately on known categories and in-the-wild instances to quantify its capability. The known categories consist of 13 classes from ShapeNet with a substantial number of training samples. The in-the-wild categories refer to other unseen categories, which may have very few or no samples included in the training set. For known categories, we use the ShapeNet-13 (Liu et al., 2020) test set. For in-the-wild categories, we constructed the test set with GSO and ABO categories which are not included in ShapeNet-13.

Evaluation metrics. Following MSC (Wu et al., 2020), we evaluate the quality, diversity, and fidelity of the restoration results separately with the Minimum Matching Distance (MMD), Total Mutual Difference (TMD), and Average Matching Distance (AMD) metrics. For each impaired shape, we sample k = 10 samples in unit cubes and calculate these metrics. The reported metrics are multiplied by  $ 10^{3} $ . More details are provided in the appendix.

Results. We present both quantitative and qualitative results across multiple dataset subsets and restoration subtasks. Quantitative results in Tab. 1 highlight difficulty variations among different data and subtasks. Benefiting from the advantage of more training samples, the model performs better on known categories compared to in-the-wild ones. Among subtasks, noisy completion and noise-free completion demand stronger semantic understanding and generation capabilities on broader semantics from the model, resulting in generally worse metrics across several dataset sub-

<div style="text-align: center;">Table 1: Quantitative results of general shape restoration.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">MMD  $ \downarrow $  / AMD  $ \downarrow $  / TMD  $ \uparrow $</td><td colspan="2">In-the-wild Categories</td><td colspan="2">Known Categories</td></tr><tr><td style='text-align: center;'>GSO</td><td style='text-align: center;'>ABO</td><td colspan="2">ShapeNet-13</td></tr><tr><td style='text-align: center;'>super-resolution</td><td style='text-align: center;'>0.174 / 0.213 / 0.192</td><td style='text-align: center;'>1.177 / 1.358 / 0.255</td><td colspan="2">0.174 / 0.315 / 0.251</td></tr><tr><td style='text-align: center;'>noise-free completion</td><td style='text-align: center;'>2.985 / 5.920 / 4.201</td><td style='text-align: center;'>1.267 / 2.701 / 1.916</td><td colspan="2">0.462 / 1.313 / 1.060</td></tr><tr><td style='text-align: center;'>noisy refinement</td><td style='text-align: center;'>0.344 / 0.466 / 0.395</td><td style='text-align: center;'>1.195 / 1.513 / 0.419</td><td colspan="2">0.204 / 0.304 / 0.240</td></tr><tr><td style='text-align: center;'>noisy completion</td><td style='text-align: center;'>0.922 / 1.467 / 1.046</td><td style='text-align: center;'>1.201 / 1.875 / 0.873</td><td colspan="2">0.319 / 0.648 / 0.506</td></tr></table>

<div style="text-align: center;">Table 2: Quantitative results of noise-free shape completion on the 3DQD benchmark.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td rowspan="2">Resolution</td><td colspan="3">Half</td><td colspan="3">Octant</td></tr><tr><td style='text-align: center;'>MMD  $ \downarrow $</td><td style='text-align: center;'>AMD  $ \downarrow $</td><td style='text-align: center;'>TMD  $ \uparrow $</td><td style='text-align: center;'>MMD  $ \downarrow $</td><td style='text-align: center;'>AMD  $ \downarrow $</td><td style='text-align: center;'>TMD  $ \uparrow $</td></tr><tr><td style='text-align: center;'>PoinTr</td><td style='text-align: center;'>point cloud</td><td style='text-align: center;'>5.316</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>21.57</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>N/A</td></tr><tr><td style='text-align: center;'>SeedFormer</td><td style='text-align: center;'>point cloud</td><td style='text-align: center;'>4.972</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>23.99</td><td style='text-align: center;'>N/A</td><td style='text-align: center;'>N/A</td></tr><tr><td style='text-align: center;'>AutoSDF</td><td style='text-align: center;'>64 $ ^{3} $</td><td style='text-align: center;'>3.510</td><td style='text-align: center;'>8.200</td><td style='text-align: center;'>4.660</td><td style='text-align: center;'>5.720</td><td style='text-align: center;'>12.79</td><td style='text-align: center;'>8.260</td></tr><tr><td style='text-align: center;'>3DQD</td><td style='text-align: center;'>64 $ ^{3} $</td><td style='text-align: center;'>2.933</td><td style='text-align: center;'>6.302</td><td style='text-align: center;'>4.780</td><td style='text-align: center;'>4.690</td><td style='text-align: center;'>10.93</td><td style='text-align: center;'>9.600</td></tr><tr><td style='text-align: center;'>NeuSDFusion</td><td style='text-align: center;'>sdf field</td><td style='text-align: center;'>2.290</td><td style='text-align: center;'>5.900</td><td style='text-align: center;'>4.760</td><td style='text-align: center;'>3.030</td><td style='text-align: center;'>9.590</td><td style='text-align: center;'>8.320</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>256 $ ^{3} $</td><td style='text-align: center;'>2.268</td><td style='text-align: center;'>6.143</td><td style='text-align: center;'>5.330</td><td style='text-align: center;'>3.840</td><td style='text-align: center;'>9.930</td><td style='text-align: center;'>9.080</td></tr><tr><td style='text-align: center;'>Ours (more samples)</td><td style='text-align: center;'>256 $ ^{3} $</td><td style='text-align: center;'>1.844</td><td style='text-align: center;'>4.752</td><td style='text-align: center;'>3.960</td><td style='text-align: center;'>3.560</td><td style='text-align: center;'>8.400</td><td style='text-align: center;'>7.020</td></tr></table>

sets. Fig. 5 illustrates the model’s performance on different subtasks, showing its ability to comprehend severely missing or noisy inputs and generate multi-modal outputs.

### 5.2 NOISE-FREE (MULTI-MODAL) SHAPE COMPLETION

We use the 3DQD (Li et al., 2023) benchmark to evaluate the multi-modal completion for noise-free partial shapes. 3DQD trains a category-conditional model on ShapeNet-13, using either half or octant samples as partial shapes, and evaluates MMD, TMD, and AMD metrics separately. To generate category-conditioned samples, we pre-train our model on Objaverse using text conditions based on captions provided by Cap3D (Luo et al., 2024) and fine-tune it on ShapeNet-13 using category names as the text conditions. Notably, we do not train our model solely on the noise-free completion subtask but jointly on all restoration subtasks, ensuring its generalizability, which cannot be effectively evaluated on the 3DQD benchmark. More details are provided in the appendix.

Results. Our method achieves SoTA completion quality (MMD) on the half subset, significantly outperforming previous baselines based on various 3D representations, as shown in Tab. 2. It also delivers better overall results (AMD) on the octant subset. In our vanilla training data, noise-free completion accounts for only a small fraction of the samples (approximately 1/8 of the entire dataset). As shown in Ours (more samples), by introducing more noise-free completion samples, we can further improve the model's performance. As illustrated in Fig. 6, our method provides restoration that better preserve the given partial input and generate outputs with finer details. Conditional generative models are known to trade diversity for quality (Sadat et al., 2024; Ho & Salimans, 2022); our model similarly exhibits slightly lower diversity than baselines. However, TMD measures diversity without considering plausibility. Our method achieves more plausible restorations than baselines.

### 5.3 Noisy Shape Completion

The PatchComplete (Rao et al., 2022) benchmark focuses on restoring noisy 3D scans of unknown categories. It includes synthetic and real-world subsets with severely incomplete, noisy partial shapes constructed by virtual rendering and depth fusion on ShapeNet, as well as cropping from ScanNet scans. Restoration quality is assessed by IoU and Chamfer Distance. Similar to 3DQD, we fine-tune our pre-trained model on PatchComplete's training set. The base model is trained on the Objaverse subset of our dataset, with all novel categories presented in the test set filtered.

Results. As shown in Tab. 3, our method achieves SoTA results on novel categories from both synthetic and real-world data. Notably, inputs of this benchmark are very low-resolution  $ (32^{3}) $  TSDF grids. As illustrated in Fig. 7, our model can handle such low-resolution noisy inputs, generating high-resolution geometry while maintaining consistency with the partial ones. Unlike baseline methods, we do not leverage the observability information contained in the original TSDF grid but rely solely on the geometric information near the surface. This leads to a more general and challenging problem, yet the proposed method still yields superior results compared to baselines.

<div style="text-align: center;">Table 3: Quantitative results of noisy shape completion on ShapeNet and ScanNet objects of novel categories. We leave the full results of each novel category to the appendix.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>CD  $ \downarrow $  / IoU  $ \uparrow $</td><td style='text-align: center;'>3D-EPN</td><td style='text-align: center;'>Auto-SDF</td><td style='text-align: center;'>PatchComplete</td><td style='text-align: center;'>DiffComplete</td><td style='text-align: center;'>SC-Diff</td><td style='text-align: center;'>Ours</td></tr><tr><td style='text-align: center;'>ShapeNet (avg. of 8 categories)</td><td style='text-align: center;'>5.58 / 59.4</td><td style='text-align: center;'>5.86 / 45.2</td><td style='text-align: center;'>4.27 / 65.4</td><td style='text-align: center;'>4.10 / 67.5</td><td style='text-align: center;'>4.08 / 68.3</td><td style='text-align: center;'>3.90 / 70.6</td></tr><tr><td style='text-align: center;'>ScanNet (avg. of 6 categories)</td><td style='text-align: center;'>9.09 / 44.0</td><td style='text-align: center;'>8.90 / 38.9</td><td style='text-align: center;'>7.52 / 49.5</td><td style='text-align: center;'>7.18 / 51.3</td><td style='text-align: center;'>7.04 / 51.9</td><td style='text-align: center;'>6.75 / 53.3</td></tr></table>

<div style="text-align: center;">Table 4: Ablation study on joint subtasks learning.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">MMD  $ \downarrow $  / AMD  $ \downarrow $  / TMD  $ \uparrow $</td><td colspan="2">ABO (in-the-wild categories)</td><td colspan="2">ShapeNet-13 (known categories)</td></tr><tr><td style='text-align: center;'>noisy refinement</td><td style='text-align: center;'>noisy completion</td><td style='text-align: center;'>noisy refinement</td><td style='text-align: center;'>noisy completion</td></tr><tr><td style='text-align: center;'>noisy refinement only</td><td style='text-align: center;'>1.53 / 1.75 / 0.258</td><td style='text-align: center;'>4.25 / 5.44 / 0.653</td><td style='text-align: center;'>0.278 / 0.360 / 0.195</td><td style='text-align: center;'>1.84 / 2.49 / 0.408</td></tr><tr><td style='text-align: center;'>noisy completion only</td><td style='text-align: center;'>1.49 / 1.82 / 0.372</td><td style='text-align: center;'>1.95 / 2.68 / 0.693</td><td style='text-align: center;'>0.265 / 0.385 / 0.219</td><td style='text-align: center;'>0.640 / 0.965 / 0.391</td></tr><tr><td style='text-align: center;'>Ours (joint training)</td><td style='text-align: center;'>1.19 / 1.51 / 0.419</td><td style='text-align: center;'>1.20 / 1.87 / 0.873</td><td style='text-align: center;'>0.204 / 0.304 / 0.240</td><td style='text-align: center;'>0.319 / 0.648 / 0.506</td></tr></table>

<div style="text-align: center;">Table 5: Ablation study on conditional encoder pre-alignment fine-tuning.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>MMD  $ \downarrow $  / AMD  $ \downarrow $  / TMD  $ \uparrow $</td><td style='text-align: center;'>noisy refinement</td><td style='text-align: center;'>noisy completion</td></tr><tr><td style='text-align: center;'>w/o pre-alignment</td><td style='text-align: center;'>0.308 / 0.400 / 0.299</td><td style='text-align: center;'>0.401 / 0.577 / 0.435</td></tr><tr><td style='text-align: center;'>w/ pre-alignment</td><td style='text-align: center;'>0.299 / 0.383 / 0.296</td><td style='text-align: center;'>0.376 / 0.534 / 0.407</td></tr></table>

### 5.4 ABLATION STUDIES

Effectiveness of joint training on subtasks. In this ablation study, we aim to validate the effectiveness of joint training on multiple shape restoration subtasks compared to training each subtask individually. We train two models separately on our dataset's noisy completion subset and the noisy refinement subset. Then, we test their performance on each task, comparing the results with those of the jointly trained model. As shown in Tab. 4, joint training on a dataset composed of various shape restoration scenarios can effectively improve performance across different cases. Joint training proves effective for both rare and common categories.

Effectiveness of conditional encoding pre-alignment. In this experiment, we aim to validate the effectiveness of our proposed conditional encoder training mechanism. We compare two setups: one using the base VAE encoder directly without fine-tuning as the conditional encoder and the other applying our proposed pre-alignment fine-tuning strategy to it. Training the conditional encoder from scratch is excluded, as loading the uncompressed high-resolution conditional shapes during training significantly slows down training, making it impractical. We compare these setups on a single ShapeNet category. As shown in Tab. 5 pre-alignment fine-tuning yields better results on noisy, defective shapes with varying levels of incompleteness, i.e., noisy refinement and completion.

## 6 CONCLUSION

In this work, we unify multiple shape restoration subtasks and simulate diverse scenarios — such as varying degrees of incompleteness and noise — to build a cohesive synthetic dataset for training a general-purpose shape restoration model. We employ a conditional hierarchical latent diffusion model for shape restoration, enabling multi-level understanding of the defective shapes and efficient generation of the intact ones. Additionally, we enhance the model's robustness to extreme OOD inputs through patch-wise and noise-robust encoding. Benefiting from the proposed scalable training strategy, we can pre-compress defective shapes, significantly improving the training efficiency of the generative model. We demonstrate our model's effectiveness across different shape restoration subtasks on multiple datasets.

Limitation and future works. Our conditional diffusion model achieves high-quality results but sacrifices diversity for quality, a common tradeoff in such models. Although effective in certain multi-object scenarios, our method is limited by the training data and cannot handle large-scale scenes. Exploring geometric restoration in large scenes through compositional or holistic approaches is a promising direction. Restoring shapes with significant incompleteness requires the understanding and generation capabilities of broad semantics. Enhancing our model by training on larger-scale datasets could improve its robustness. Additionally, representing objects with sparse TSDF grids remains insufficiently compact; hence, shape restoration based on primitive-based or CAD-related representations is a potential area for further improvement.

Restorations

GT

Input

Restorations

GT

### Figure 5: Qualitative results of shape restoration on our general shape restoration benchmark. UniRestore3D can handle various types of impairments with high quality

<div style="text-align: center;"><img src="imgs/img_in_image_box_253_158_970_1397.jpg" alt="Image" width="58%" /></div>


Ours

GT

### Figure 6: Qualitative results of noise-free shape completion on the 3DQD benchmark. UniRestore3D achieves better quality and fidelity compared to baselines

Input

Ours

GT

<div style="text-align: center;">Figure 7: Qualitative results of noisy shape completion on the PatchComplete benchmark. UniRestore3D can restore low-resolution and noisy inputs effectively.</div>


## ACKNOWLEDGEMENTS

This work was partially supported by the NSFC (No. U24B20154, No. 62322207, No. 62402427), Zhejiang Provincial Natural Science Foundation of China (No. LD25F030001), Ant Group Research Fund and Information Technology Center and State Key Lab of CAD&CG, Zhejiang University.