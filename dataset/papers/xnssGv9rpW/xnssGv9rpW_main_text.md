# SYMMCD: SYMMETRY-PRESERVING CRYSTAL GENERATION WITH DIFFUSION MODELS

Daniel Levy $ ^{*1,2} $ , Siba Smarak Panigrahi $ ^{*1,2,3} $ , Sékou-Oumar Kaba $ ^{*1,2} $ , Qiang Zhu $ ^{4} $ , Kin Long Kelvin Lee $ ^{5} $ , Mikhail Galkin $ ^{5} $ ,

Santiago Miret $ ^{5} $ , Siamak Ravanbakhsh $ ^{1,2} $ 

 $ ^{1} $ McGill University,  $ ^{2} $ Mila,  $ ^{3} $ École Polytechnique Fédérale de Lausanne (EPFL),  $ ^{4} $ University of North Carolina at Charlotte,  $ ^{5} $ Intel Labs



## ABSTRACT

Generating novel crystalline materials has the potential to lead to advancements in fields such as electronics, energy storage, and catalysis. The defining characteristic of crystals is their symmetry, which plays a central role in determining their physical properties. However, existing crystal generation methods either fail to generate materials that display the symmetries of real-world crystals, or simply replicate the symmetry information from examples in a database. To address this limitation, we propose SymmCD $ ^{1} $ , a novel diffusion-based generative model that explicitly incorporates crystallographic symmetry into the generative process. We decompose crystals into two components and learn their joint distribution through diffusion: 1) the asymmetric unit, the smallest subset of the crystal which can generate the whole crystal through symmetry transformations, and; 2) the symmetry transformations needed to be applied to each atom in the asymmetric unit. We also use a novel and interpretable representation for these transformations, enabling generalization across different crystallographic symmetry groups. We showcase the competitive performance of SymmCD on a subset of the Materials Project, obtaining diverse and valid crystals with realistic symmetries and predicted properties.

## 1 INTRODUCTION

Crystals serve as the fundamental building blocks of many materials, and the discovery of new crystalline materials is expected to lead to diverse technological breakthroughs in fields ranging from energy storage to computing hardware (Miret et al., 2024). Generative models have the potential to greatly accelerate this process by proposing new candidates materials, and possibly conditioning on desired properties or compositions.

The defining characteristic of crystals is their symmetry. These symmetries are Euclidean transformations that map the crystal structure back to itself. They can in general be some specific translations, rotations, reflections and combinations of these. The set of these operations is called the space group of the crystal. It is known that space groups in three dimensions fall into 230 distinct classes (Hahn et al., 1983). The symmetry of a crystal plays a crucial role in determining its stability along with its thermodynamic, electronic and mechanical properties (Nye, 1985). A classic example is given by piezoelectricity, the ability of a material to generate an electric dipole under mechanical stress, which can only be manifested in materials lacking inversion symmetry.

Importantly, many of the recently proposed generative models for crystals do not generate samples with non-trivial symmetry: for example, the most frequently generated crystals by DiffCSP (Jiao et al., 2023) and CDVAE (Xie et al., 2022) are in the low-symmetry P1 space group, which is very rare in nature. MatterGen (Zeni et al., 2023) can generate crystals conditioned on a desired space group for space groups that are highly represented in the dataset, but they only recover the target space group roughly 20% of the time, dropping to about 10% for more symmetric space groups.

<div style="text-align: center;"><img src="imgs/img_in_image_box_252_135_971_450.jpg" alt="Image" width="58%" /></div>


<div style="text-align: center;">Figure 1: Illustration of the SymmCD method. Left. Representation of the unit cell of a 2D crystal with  $ p4m $  symmetry where the asymmetric unit and the site symmetries of the atoms are highlighted. Leveraging symmetry results in a much more compact, yet complete representation. Right. Diffusion and denoising on the different components of the representation. For site symmetries and atom types, discrete diffusion is used. For the coordinate and asymmetric unit continuous diffusion is used. The diffusion and denoising processes preserve the space group symmetry.</div>


Cheetham & Seshadri (2024) analyse the space groups of the stable crystal structures proposed by the GNoME model of Merchant et al. (2023), finding that the top 4 most commonly generated space groups account for 34% of all generated crystals, even though each of those 4 space groups appears in less than 1% of crystals in the Inorganic Crystal Structure Database (Hellenbrandt, 2004).

In this work, we propose a novel approach for generative modeling of inorganic crystals that ensures any desired distribution of space groups. The idea is similar to that of creating a paper snowflake, where we fold the paper to create an unconstrained space, and after an unconstrained cutting of the paper in this space, its unfolding creates an object with desired symmetries. In the context of crystals, the unconstrained space is called the asymmetric unit, which is a maximal subset of the unit cell with no redundancy. To unfold the asymmetric unit, we need to generate the site symmetry of each atom inside the unit, i.e., the symmetry transformations that fix the atoms in place. In our generative process, the atomic positions are made consistent with generated site symmetries, enabling the unfolding of an asymmetric unit into a symmetric crystal; see Figure 1.

Crystals and their individual atoms have many different types of symmetries, so we need to address the issue of data-fragmentation. By representing symmetry information using standard crystallographic notations, such as Hermann–Mauguin notation or Wyckoff position labels (Hahn et al., 1983), we are faced with many crystals and site symmetries that have a low frequency in the training data. To address this problem, we introduce a novel representation of symmetries as binary matrices, which enables information-sharing and generalization across both crystal and site symmetries.

The main contributions of this work are as follows: I) We demonstrate a novel approach to generating crystals through the unconstrained generation of asymmetric units, along with their symmetry information. II) We introduce a physically-motivated representation for crystallographic site symmetries that generalizes across space groups. III) We experimentally evaluate our method, finding that it performs on par with previous methods at generating stable structures, while offering significantly improved computational efficiency due to our representation. IV) We analyse the symmetry and diversity of crystal structures generated by existing generative models.

## 2 RELATED WORK

There has been a growing body of work in developing machine-learning methods for crystal structure modeling, including the development of datasets and benchmarks (Jain et al., 2013; Saal et al., 2013; Chanussot et al., 2021; Miret et al., 2023; Lee et al., 2023; Choudhary et al., 2024). Recent work has also focused on developing architectures that are equivariant to various symmetries Duval et al. (2023) or are specifically designed to include inductive biases useful for crystal structures (Xie & Grossman, 2018; Kaba & Ravanbakhsh, 2022; Goodall et al., 2022; Yan et al., 2022; 2024).

In addition to structure-based modeling, prior work has also generated full-atom crystal structures, in which all atoms of the three-dimensional structure are generated. A range of generation methods including VAEs (Noh et al., 2019; Xie et al., 2022), GANs (Nouira et al., 2018; Kim et al., 2020), reinforcement learning (Govindarajan et al., 2023), diffusion models (Zeni et al., 2023; Yang et al., 2023; Jiao et al., 2023; Klipfel et al., 2024), flow-matching models (Miller et al., 2024), and active learning based discovery (Merchant et al., 2023) have been used. These follow similar works in 3D molecule generation (Hoogeboom et al., 2022; Garcia Satorras et al., 2021; Xu et al., 2022), but extend them by incorporating crystal periodicity. In addition to full-atom crystal generation, prior work has also applied text-based methods to understand and generate crystals using language models (Flam-Shepherd & Aspuru-Guzik, 2023; Gruver et al., 2024; Alampara et al., 2024).

Other works have pointed out the importance of symmetry of the generated structures. DiffCSP++ (Jiao et al., 2024), does so by using predefined structural templates from the training data and learning atomic types and coordinates compatible with the templates. While this is an interesting solution, we show that predefining the templates in this way severely limits the diversity and novelty of the generated samples. CrystalGFN (AI4Science et al., 2023) incorporates constraints on the lattice parameters and composition based on space groups but does not guarantee that the atomic positions respect the desired symmetry. Finally, the concurrent works CrystalFormer (Cao et al., 2024) and WyCryst (Zhu et al., 2024) generate symmetric crystals by predicting atom symmetries. However, they directly use the labels of Wyckoff positions to encode symmetries, which does not enable generalization across groups. The methods are therefore limited to generating from space groups that are common in the dataset. By contrast, our method generalizes across groups and can generate valid crystals even from groups that are rare in the dataset.

## 3 BACKGROUND

Lattices and unit cells Crystals are macroscopic atomic systems characterized by a periodic structure. A crystal can be described as an infinite 3-dimensional lattice of identical unit cells, each containing atoms in set positions. We can represent a crystal with the tuple  $ \mathcal{C} = (\mathbf{L}, \mathbf{X}, \mathbf{A}) $ , where  $ \mathbf{L} = (\mathbf{l}_1, \mathbf{l}_2, \mathbf{l}_3) \in \mathbb{R}^{3 \times 3} $  is a matrix of lattice vectors,  $ X \in [0, 1)^{3 \times N} $  represents the fractional coordinates of N atoms within a unit cell, and  $ A \in \{0, 1\}^{Z \times N} $  is a matrix of one-hot vectors of Z possible elements for each atom. The lattice describes the tiling of unit cells: the cartesian coordinates of atoms can be given by  $ X^c = LX $ , and if  $ x_i^c $  is the cartesian coordinate of an atom in a unit cell, then the crystal will also contain an identical atom at  $ x_i^c + Lj, \forall j \in Z^3 $ .

Crystal symmetries In addition to the translational symmetry of the lattice, crystals typically have many other symmetries. Understanding these symmetries is fundamental in characterizing crystals and directly relates to many of the properties of these materials. The space group G of a crystal is the group of all Euclidean transformations that leave the crystal invariant, i.e., that simply permutes atoms of the same type. As space groups are subgroups of the Euclidean group, their elements can be represented as  $ (\mathbf{O}, \mathbf{t}) $ , where  $ \mathbf{O} \in O(n) $  and  $ t \in R^{3} $ , with action on  $ x \in R^{3} $  defined as  $ (\mathbf{O}, \mathbf{t}) \mathbf{x} = \mathbf{O x} + \mathbf{t} $ . The operations that are part of a space group can be generally understood as belonging to different types: translations, rotations, inversions, reflections, screw axes (combinations of rotations and translations), and glide planes (combinations of mirroring and translation).

Two space groups belong to the same type if all their operations can be mapped to each other by an orientation-preserving Euclidean transformation (coordinate change). We denote the set of all space group types as G. In 3 dimensions, there are only 230 unique space group types. By choosing a canonical coordinate system, we can in general work only with space group types. The point group P of a space group G is the image of the homomorphism  $ (\mathbf{O}, \mathbf{t}) \mapsto \mathbf{O} $ , i.e., the group obtained by keeping only the orthogonal parts of G. By contrast with space groups, any point group must at least preserve a single point, that is, the origin. By a similar procedure to space groups, we can classify point groups and find that there are 32 crystallographic point group types, consisting of inversions, rotations, and reflections. We denote the set of all point group types as P.

Wyckoff positions Having classified symmetry groups, we can now also classify points of space using symmetry considerations. This will be important to our method, as we will seek to use these semantically meaningful classes to guide the generation process. Given a space group G, we say that two points  $ x, x^{\prime} \in R^{3} $  are part of the same crystallographic orbit if there is a  $ (\mathbf{O}, \mathbf{t}) \in G $  such that

 $ (\mathbf{O},\mathbf{t})\mathbf{x}=\mathbf{x}^{\prime} $ . The orbits form a partition of  $ R^{3} $ ; they can be understood as the finest level of classification under G. We define the site symmetry group of a point  $ \mathbf{x} $ ,  $ S_{\mathbf{x}}=\left\{(\mathbf{O},\mathbf{t})\in G\mid(\mathbf{O},\mathbf{t})\mathbf{x}=\mathbf{x}\right\} $  as the subgroup of G that leaves x invariant. It is clear that the site symmetry must be a point group (since translations do not preserve any point), and is a subgroup of P. From the orbit-stabilizer theorem (see e.g. Dummit & Foote (2004)), we can find that the number of points in the orbit x and in the unit cell is given by  $ |P|/|S_{x}| $ . Points in highly symmetric positions, therefore, result in smaller orbits. A point is in a general position if its site symmetry group is trivial. In this case, there is a one-to-one correspondence between points in the orbit and group members. If the site symmetry is non-trivial, a point is said to be in a special position.

Points in the same orbit have conjugate site symmetry groups. Therefore, site symmetry groups related by conjugation can be understood as equivalent. This motivates a coarser level of classification that will be very useful. Two points  $ x, x^{\prime} \in R^{3} $  are part of the same Wyckoff position if their site symmetry group is conjugate. Wyckoff positions have a clear meaning: they classify regions of space in terms of their type of symmetry. The multiplicity of a Wyckoff position is the number of equivalent atoms that must occupy that position and is equal to the  $ \left|P\right|/\left|S_{x}\right| $  ratio introduced earlier.

Asymmetric Units The unit cell of a crystal can further be reduced into an asymmetric unit, which is a small part of the unit cell that contains no symmetry but can be used to generate the whole unit cell by applying the symmetry transformations of the space group. An asymmetric unit will only contain a single atom from each orbit.

## 4 METHOD: SYMMETRIC CRYSTAL DIFFUSION (SYMMCD)

### 4.1 REPRESENTATION OF CRYSTALS WITH WYCKOFF POSITIONS

As explained in the previous section, a crystal structure can in general be represented by the tuple  $ \mathcal{C} = (\mathbf{L}, \mathbf{X}, \mathbf{A}) $ . This representation has been used in previous generative models for crystals (Xie et al., 2022; Jiao et al., 2023; Luo et al., 2023; Zeni et al., 2023). However, a fundamental limitation of a model based on this representation is that it does not leverage the inductive bias of crystal symmetry and offers no guarantees for the crystal to satisfy anything but a trivial space group.

We introduce an alternative representation that respects symmetry in addition to having many desirable properties. First, we explicitly specify the space group type of the crystal  $ G \in G $  in the representation. Given the space group, instead of representing each of the N atoms individually with  $ X \in R^{3 \times N} $  and  $ A \in R^{Z \times N} $ , we represent the M crystallographic orbits; replicating the atoms within the orbit then creates the crystal. As explained in Section 3, the Wyckoff position identifies a set of orbits by site symmetries. Therefore, specifying the site symmetry and an arbitrary orbit representative is sufficient to identify a crystallographic orbit. This corresponds to a representation of an asymmetric unit within the unit cell. We thus define the set of orbit representatives with their Wyckoff positions as the tuple  $ C' = (\mathbf{k}, \mathbf{X}', \mathbf{S}, \mathbf{A}') $ , where k is a parametrization of the lattice (to be explained later),  $ X' = [x_{1}', \ldots, x_{M}'] \in R^{3 \times M} $  are the representative's fractional coordinates in the asymmetric unit,  $ S = [S_{x_{1}', \ldots, x_{M}'}'] \in P^{M} $  are the site symmetry groups and  $ A' = [a_{1}', \ldots, a_{M'}'] \in R^{Z \times M} $  are the atomic types.

From the set of representatives, we can go back to the representation X and A in a unique way. This is done by generating the orbits using the replication operation that depends on the group G and the site symmetry S. The replication operation essentially consists of applying all of the symmetry operations of the space group except for the ones included in the site symmetry group. The details of this operation are included in Appendix A. Finally, the lattice L can be constrained to be compatible with the space group in a convenient way using the vector  $ k \in R^{6} $  (Jiao et al., 2024):  $ \log(\mathbf{L}) = \sum_{i=1}^{6} k_{i} \mathbf{b}_{i} $ , where the  $ B_{i} \in R^{3 \times 3} $  is a standard basis over symmetric matrices. This basis and the constraints on k for each space group are described in Appendix B.

Our representation of crystals that explicitly takes into account symmetry is therefore given by the tuple  $ \mathcal{C}^{\prime}=(G,\mathbf{k},\mathbf{X}^{\prime},\mathbf{S},\mathbf{A}^{\prime}) $ . We convert crystal structures to this representation using the SPGLIB symmetry finding algorithm (Togo & Tanaka, 2018b) provided by PYMATGEN (Ong et al., 2013).

In addition to accounting for the symmetry, this representation of a crystal provides two important advantages compared to existing methods. First, it provides the generative model with a powerful

<div style="text-align: center;"><img src="imgs/img_in_image_box_229_122_992_313.jpg" alt="Image" width="62%" /></div>


<div style="text-align: center;">Figure 2: Crystal symmetry axes. The different axes describe the directions along which symmetry operations can occur. For each of the 15 axes, there are 13 possible symmetry operations.</div>


physically-motivated inductive bias. It is known from crystallography that atoms are typically not located in arbitrary positions in the unit cell (Aroyo, 2013). Rather, it is energetically more favourable for atoms to occupy positions of high symmetry, e.g. special Wyckoff positions. The representation in terms of positions X does not make this explicit. The representation using Wyckoff positions  $ (\mathbf{X}^{\prime}, \mathbf{S}^{\prime}) $  provides explicit supervision to the model and guides the generation process: the model decides in which type of high-symmetry position an atom should be located and generates a position compatible with that type. Second, the representation in terms of Wyckoff positions is much more compact than the representation that operates on individual atoms. M is often significantly smaller than N. In the MP-20 dataset (a subset of the Materials Project dataset (Jain et al., 2013)) for example, the average number of orbits is M = 4.7 whereas the average number of atoms per unit cells is  $ \bar{N} = 18.9 $ , representing a fourfold difference $ ^{2} $ . We therefore eliminate the redundant information from the representation and increase the computational efficiency of our method.

### 4.2 SYMMETRY REPRESENTATION

A key component of our representation of crystals with Wyckoff positions is the encoding of the space group G and site symmetry groups  $ S^{\prime} $ . While there are many existing methods to encode these symmetries, they generally do not make explicit the commonalities between the site symmetries of Wyckoff positions in the same space group, and the commonalities between different space groups across crystal systems. This is an important limitation: because there are 230 space groups, not having a representation that is common across space groups results in dividing the effective amount of data the model is trained on by a large amount. For example, in the MP-20 dataset (Jain et al., 2013; Xie & Grossman, 2018) 113 space groups out of 169 in the training set have fewer than a hundred samples associated with them, and specific Wyckoff positions in each group have even fewer samples. We propose a method to represent the site symmetries of different Wyckoff positions and to encode the symmetries of space groups to address this shortcoming.

We represent atom site symmetries using a binary representation based on the oriented site symmetry symbol used by the International Tables for Crystallography to describe Wyckoff positions (Hahn et al., 1983; Donnay & Turrell, 1974). The oriented site symmetry symbols denote generators of the site symmetry group along different possible axes, illustrated in Figure 2. In total, there are 15 possible axes of symmetry in a crystal, corresponding to each of the Cartesian axes, along with body and face diagonals.

Examples of possible symmetry operations along each axis include rotations and roto-inversions, as well as mirror symmetry along a plane perpendicular to the axis. There are 13 possible symmetries along each axis. Listing out the site symmetry operation along each axis yields a  $ 15 \times 13 $  binary matrix, or equivalently 15 different one-hot vectors. There is an injective mapping between site symmetries and site symmetry matrix representations, so a representative atom can be replicated to produce a full orbit using this representation.

The space group G can also be encoded into a binary representation using a similar scheme, by listing out the 15 possible axes of symmetry and listing out the possible symmetry operations along each axis. Unlike the point group symmetries of atoms, these space group symmetry operations may involve translations and so include screw and glide transformations, leading to 26 possible symmetry operations. Further details are included in Appendix 4.1. As a part of this project, both

<div style="text-align: center;"><img src="imgs/img_in_image_box_292_122_929_413.jpg" alt="Image" width="52%" /></div>


<div style="text-align: center;">Figure 3: SymmCD training and sampling pipeline. For training (green), the crystal structures are pre-processed to find the space group G along with the site symmetries S and a set of orbit representatives inside the asymmetric unit. The denoising model is a GNN with fully connected graphs, followed by a decoder. For sampling (blue), the positions are projected to the closest one compatible with their site symmetry. Then, the asymmetric unit is replicated to obtain the unit cell.</div>


the site symmetry representations and the space group representations have been incorporated as functions in the PyXtal software library (Fredericks et al., 2021).

### 4.3 DIFFUSION MODEL

We can now describe the generative model and training process. In SymmCD, the space group and the number of orbit representatives are first sampled from separate distributions obtained from data, such that the distribution over crystal structures is  $  p(\mathcal{C}) = p(\mathbf{k}, \mathbf{X}^{\prime}, \mathbf{S}, \mathbf{A}^{\prime} \mid M, G) p(M \mid G) p(G)  $ . We will seek to model the conditional distribution  $  p(\mathbf{k}, \mathbf{X}^{\prime}, \mathbf{S}, \mathbf{A}^{\prime} \mid M, G)  $  with a denoising diffusion model (Sohl-Dickstein et al., 2015; Ho et al., 2020).

We leverage our binary representation for incorporating crystal symmetry information (described in Section 4.2) and perform joint diffusion over lattice representation (k), fractional coordinates of atoms  $ (\mathbf{X}^{\prime}) $ , their types  $ (\mathbf{A}^{\prime}) $ , and the associated binary representation of site symmetry (S).

Diffusion process We consider a separate diffusion process over the different components of the crystal representation. We apply discrete diffusion from Austin et al. (2021) for site symmetries and atom types. Rather than adding Gaussian noise as in conventional diffusion, we add noise to categorical features by multiplying probability vectors by a transition matrix and sampling from the new probabilities. Inspired by Vignac et al. (2023), the transition matrices are parameterized so that the process converges to the marginals from the data distribution for atom types and site symmetries. The loss function used for discrete diffusion on atomic types is

 $$ \mathcal{L}_{\mathbf{A}^{\prime}}=\mathbb{E}_{\mathbf{a}_{t}\sim\mathrm{C a t}(\mathbf{a}_{0}^{\top}\bar{\mathbf{Q}}_{t}),t\sim\mathrm{U}(1,T)}\sum_{i=1}^{M}\mathrm{C r o s s E n t r o p y}(\mathbf{a}_{i},\hat{\mathbf{a}}_{i}), $$ 

where  $ a_{0} $  is the initial one-hot encoding of the atom types for a single representative and  $ \bar{Q}_{t} = \prod_{i=1}^{t} Q_{i} \in R^{Z \times Z} $  is the cumulative product of transition matrices between timesteps, and  $ \hat{a}_{i} $  are the predicted denoised probabilities. The same loss function is used for site symmetries.

Continuous diffusion is used for fractional coordinates and lattice parameters, similar to Jiao et al. (2023). The loss function for the continuous diffusion on lattice parameters is

 $$ \mathcal{L}_{\mathbf{k}}=\mathbb{E}_{\mathbf{\epsilon  _{k}}\sim\mathcal{N}(0,\mathbf{I}),t\sim\mathrm{U}(1,T)}\left[||m\odot\mathbf{\epsilon  _{k}}-\hat{\mathbf{\epsilon  _{k}}}(\mathcal{C}_{t}^{\prime},t)||_{2}^{2}\right], $$ 

where m is a space group-dependent mask, and  $ \hat{\epsilon}_{k} $  is the predicted denoising vector. The same loss function is used for the fractional coordinates, except that to capture their periodic nature, we use a wrapped normal distribution  $ \mathcal{WN}(0,1)^{3\times M} $ . We provide more details in Appendix D.

Denoising network The architecture of the denoiser is a message-passing neural network that operates on a fully connected graph of representatives, based on Jiao et al. (2023). Features for each representative  $ h_{i} $  are initialized using an embedding of their atom types  $ a_{i} $  and their site symmetries.

 $ S_{i} $ , along with the graph-level features of the diffusion timestep t, the lattice features k, and an embedding of the space group G. At each layer, messages  $ m_{ij} $  are computed between representatives i and j by applying an MLP to  $ h_{i}, h_{j} $ , and a Fourier embedding of the vector  $ x_{i} - x_{j} $  to respect periodic invariance. These messages are then used to update  $ h_{i} $ . More details on the architecture are included in Appendix E.1. Note that the denoising network is not equivariant. It is not necessary since the unit cell axes provide a canonical reference system (Kaba et al., 2023). We also found that using an equivariant denoising network like EGNN Satorras et al. (2021) did not work well in part due to the fact that since we use periodic encodings, the crystal structure input has a translational symmetry. An equivariant model may not be able to break that symmetry (Kaba & Ravanbakhsh, 2023) resulting in an inability to output correct positions in the asymmetric unit (or unit cell).

Putting it all together The algorithm for training our diffusion model is outlined in Algorithm 1. We use different loss coefficients  $ \lambda_{k} $ ,  $ \lambda_{X'} $ ,  $ \lambda_{A'} $  and  $ \lambda_{S} $  to weigh the importance of the different components of the model. The algorithm for sampling from the diffusion model is shown in Algorithm 2. The full pipeline is summarized in Figure 3. Since both the diffusion and the denoising process both operate only on the asymmetric unit, they fully preserve the symmetry of the crystal.

Algorithm 1 Training SymmCD

1: Input: Dataset of crystals D

2: while not converged do

3: Sample a crystal $\mathcal{C} = (\mathbf{L}, \mathbf{X}, \mathbf{A})$ from dataset $\mathcal{D}$, and a timestep $t \sim \text{Uniform}(1, T)$

4: Derive the asymmetric representation $\mathcal{C}' = (G, \mathbf{k}, \mathbf{X}', \mathbf{A}', \mathbf{S})$ from $\mathcal{C}$

5: Add noise to $\mathbf{k}$, $\mathbf{X}', \mathbf{A}', \mathbf{S}'$:

6: $\mathbf{k}_t = \sqrt{\alpha_t} \mathbf{k}_0 + \sqrt{1 - \alpha_t} \mathbf{\epsilon}_{\mathbf{k}}$, $\epsilon_{\mathbf{k}} \sim \mathcal{N}(0, \mathbf{I})$

7: $\mathbf{X}'_t = \sqrt{\alpha_t} \mathbf{X}'_0 + \sqrt{1 - \alpha_t} \mathbf{\epsilon}_{\mathbf{X}'}$, $\epsilon_{\mathbf{X}'} \sim \mathcal{WN}(0, \mathbf{I})$

8: $\mathbf{A}'_t \sim \text{Cat}(\mathbf{A}\mathbf{Q}_{a,t})$

9: $\mathbf{S}_{u,t} \sim \text{Cat}(\mathbf{S}\mathbf{Q}_{u,G,t})$

10: Use denoising network $\phi$ to predict $\hat{\epsilon}_{\mathbf{k}}$, $\hat{\mathbf{X}'}, \hat{\mathbf{A}'}, \hat{\mathbf{S}}$ from noisy $\mathcal{C}_t = (G, \mathbf{k}_t, \mathbf{X}'_t, \mathbf{A}'_t, \mathbf{S}_t)$, $t$

11: Compute losses $\mathcal{L}_{\mathbf{k}}$, $\mathcal{L}_{\mathbf{X}'}$, $\mathcal{L}_{\mathbf{A}'}$, $\mathcal{L}_{\mathbf{S}'}$

12: Update the denoising network $\phi$ using total loss:

13: $\mathcal{L} = \lambda_{\mathbf{k}} \mathcal{L}_{\mathbf{k}} + \lambda_{\mathbf{X}'} \mathcal{L}_{\mathbf{X}'} + \lambda_{\mathbf{A}'} \mathcal{L}_{\mathbf{A}'} + \lambda_{\mathbf{S}} \mathcal{L}_{\mathbf{S}'}$

14: end while

Algorithm 2 Sampling from SymmCD

1: Input: Target space group G, Number of representatives M
2: Initialize:
3: Sample  $ \mathbf{k}_{T} \sim \mathcal{N}(0, \mathbf{I}) $ 
4: Sample  $ \mathbf{X}_{T}' \sim \mathcal{U}(0, 1)^{3 \times M} $ 
5: Sample  $ \mathbf{A}_{T}' \sim p_{\text{marginal}}(\mathbf{A}') $ 
6: Sample  $ \mathbf{S}_{T}' \sim p_{\text{marginal}}(\mathbf{S}'|G) $  (site symmetries)
7: for t = T to 1 do
8: Compute  $ \hat{\epsilon}_{k}, \hat{\epsilon}_{X'}, \hat{A}'', \hat{S} $  using denoising network  $ \phi(\cdot) $ 
9: Sample  $ k_{t-1}, X'_{t-1}, A'_{t-1}, S'_{t-1} $  using  $ \hat{\epsilon}_{k}, \hat{\epsilon}_{X'}, \hat{A}'', \hat{S} $ 
10: end for
11: Project  $ S'_{0} $  onto nearest valid point group
12: Project  $ X'_{0} $  onto nearest Wyckoff position with that site symmetry
13: Replicate representative atoms  $ X'_{0} $  using site symmetries  $ S'_{0} $  to generate full crystal  $ X_{0} $ 
14: Output: Crystal structure  $ X_{0} $ , Atom types  $ A_{0} $ , lattice  $ L_{0} $ 

## 5 EXPERIMENTS

We test our model on de novo crystal generation using the MP-20 dataset (Xie et al., 2022), a subset of the Materials Project (Jain et al., 2013) consisting of 40,476 crystals, each with up to 20 atoms per primitive unit cell. The data is preprocessed to use the conventional unit cell rather than the primitive unit cell, as the former has more conveniently expressed symmetries and constraints. A conventional unit cell may be larger than a primitive unit cell, which results in up to 80 atoms in the unit cell. We withhold 20% of the dataset as a validation set, and 20% as a test set.

<div style="text-align: center;"><img src="imgs/img_in_image_box_256_115_885_293.jpg" alt="Image" width="51%" /></div>


<div style="text-align: center;">Figure 4: Proportion of space group symmetries of the dataset, and each method. The width of each color segment represents the proportion of crystals with that symmetry. From left to right, the first few space groups are: P1, Fm $ \overline{3} $ m, Cm, P $ \overline{1} $ , C2/m, I4/mmm, Pm $ \overline{3} $ m, P6 $ _{3} $ /mmc, and Pm.</div>


We empirically demonstrate our contributions, particularly in ensuring we generate crystals with desired symmetries while being competitive with existing baselines. We compare our proposed method with four recent strong baselines: CDVAE (Xie et al., 2022), DiffCSP (Jiao et al., 2023), DiffCSP++ (Jiao et al., 2024) and FlowMM (Miller et al., 2024). We retrained each method according to their given hyperparameters, and generated 10,000 crystals each. We also consider a variant of our model (10 SGs) where we only sample from the 10 most common space groups in the MP-20 dataset, similar to (Cao et al., 2024) $ ^{3} $ . This is to provide a more nuanced comparison with other methods, which are not constrained in matching the space group distribution. This choice still captures a large portion of the data distribution, since these are the most prevalent space groups.

### 5.1 SYMMETRY AND STRUCTURAL DIVERSITY

First, we evaluate the different methods on their ability to generate crystals with diverse structures and space groups. This aspect has not been investigated yet for the considered baselines, yet it is significant in understanding if they generate realistic structures.

Space groups To detect the space group of the generated structures, we use spglib's symmetry finding method (Togo & Tanaka, 2018a; Ong et al., 2013) with a tolerance of 0.1Å. The distribution of space groups of the generated structures is shown in Figure 4. It can be observed that while SymmCD matches the highly diverse data distribution, CDVAE mostly generates crystals with trivial P1 symmetry, and DiffCSP and FlowMM generate many crystals with low symmetry and generally have lower diver-



<div style="text-align: center;">Table 1: Template statistics for each model.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'># Unique</td><td style='text-align: center;'>% in Train</td><td style='text-align: center;'># New</td></tr><tr><td style='text-align: center;'>Training Set</td><td style='text-align: center;'>3318</td><td style='text-align: center;'>100</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>CDVAE</td><td style='text-align: center;'>797</td><td style='text-align: center;'>28.7</td><td style='text-align: center;'>568</td></tr><tr><td style='text-align: center;'>DiffCSP</td><td style='text-align: center;'>1347</td><td style='text-align: center;'>43.2</td><td style='text-align: center;'>764</td></tr><tr><td style='text-align: center;'>DiffCSP++</td><td style='text-align: center;'>1905</td><td style='text-align: center;'>94.2</td><td style='text-align: center;'>110</td></tr><tr><td style='text-align: center;'>FlowMM</td><td style='text-align: center;'>1291</td><td style='text-align: center;'>41.7</td><td style='text-align: center;'>753</td></tr><tr><td style='text-align: center;'>SymmCD</td><td style='text-align: center;'>2794</td><td style='text-align: center;'>40.8</td><td style='text-align: center;'>1654</td></tr></table>

sity of space groups. We also consider a new quantitative metric to characterize the space group distribution,  $ d_{sg} $ , which is calculated as the Jensen-Shannon distance between the distribution of space groups of the generated structures and the test set. We report it for the different methods in the rightmost column of Table 3. The results confirm that SymmCD and DiffCSP++ are the only methods that accurately match the distribution of space groups in the dataset.

Unique Templates We also evaluate the ability of the different methods to generate diverse crystal structures. We define a structural template to be a combination of a space group and a multiset of occupied Wyckoff positions, regardless of the atomic types in the Wyckoff position. Templates, also known as Wyckoff sequences, are used in practice to classify crystals by their symmetry. They have the advantage of providing a notion of a structure that is highly flexible, while being robust to perturbations of coordinates that do not change the position of atoms with respect to symmetry elements. Most potential templates have not yet been experimentally observed, motivating the development of methods that can discover materials with new templates (Hornfeck, 2022).

The training dataset contains 3318 such unique templates. We examine the templates for the 10,000 crystals generated by each method, and report results in Table 1. We find that SymmCD performs best out of all models, proposing unique and novel templates. This highlights an important limitation of DiffCSP++. While it can produce diverse space groups and to a certain extent diverse templates, since it uses pre-defined templates it fails to generate structures with novel templates. Our method does not suffer from this problem since it learns to generate templates.

###### 5.2 STABLE, UNIQUE AND NOVEL (S.U.N.) STRUCTURES

Regardless of their target application, generative models for crystals should produce sets of crystals that are thermodynamically stable, unique (not duplicated within the predicted set), and novel (not already in the training data), or S.U.N. To this end, we adapt the evaluation procedure of Miller et al. (2024) to assess the capability of our model to generate S.U.N. materials. Thermodynamic stability is determined by estimating the energy of a material with respect to a convex hull. The convex hull gives linear combinations of known phases that represent the lowest-energy mixtures of materials;

<div style="text-align: center;">Table 2: Percent of stable and S.U.N. samples produced from an initial set of 10,000 generated crystals for each method.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Initial Stable</td><td style='text-align: center;'>Relaxed Stable</td><td style='text-align: center;'>Relaxed S.U.N.</td></tr><tr><td style='text-align: center;'>CDVAE</td><td style='text-align: center;'>0.24%</td><td style='text-align: center;'>4.42%</td><td style='text-align: center;'>4.26%</td></tr><tr><td style='text-align: center;'>DiffCSP</td><td style='text-align: center;'>7.82%</td><td style='text-align: center;'>11.32%</td><td style='text-align: center;'>8.92%</td></tr><tr><td style='text-align: center;'>DiffCSP++</td><td style='text-align: center;'>6.99%</td><td style='text-align: center;'>11.36%</td><td style='text-align: center;'>8.62%</td></tr><tr><td style='text-align: center;'>FlowMM</td><td style='text-align: center;'>4.26%</td><td style='text-align: center;'>9.05%</td><td style='text-align: center;'>6.49%</td></tr><tr><td style='text-align: center;'>SymmCD (All SGs)</td><td style='text-align: center;'>4.96%</td><td style='text-align: center;'>9.34%</td><td style='text-align: center;'>6.86%</td></tr><tr><td style='text-align: center;'>SymmCD (10 SGs)</td><td style='text-align: center;'>6.85%</td><td style='text-align: center;'>10.92%</td><td style='text-align: center;'>7.59%</td></tr></table>

if a material has an energy above the hull, it is energetically favorable for it to decompose into a combination of stable phases and is therefore thermodynamically unstable. We assess the stability of generated crystals by estimating their energies using a pretrained CHGNet model (Deng et al., 2023), and comparing that to a convex hull computed for Materials Project (Riebesell et al., 2024).

We predict the stability of the 10,000 samples generated by each method, and then use CHGNet to compute relaxed structures for each crystal, which results in higher stability. Finally, we check whether the stable relaxed crystals are also unique and novel. Details of this procedure are included in Miller et al. (2024). The results are shown in Table 2. We see that SymmCD performs on par with the baselines. Sampling over the most common space groups results in more stable structures.

### 5.3 VALIDATION WITH DENSITY FUNCTIONAL THEORY RELAXATIONS

We carried out electronic structure calculations as a more accurate way to evaluate the crystal structures generated by the different methods. Since these computations are significantly more expensive than the ones with CHGNet, we performed them on 100 structures sampled from each model. Concretely, these involved using the CP2K (Kühne et al., 2020) suite of programs to perform cell and geometry relaxations to assess the quality of generated structures based on their distances from "true" local minima. We expect a better method to produce structures with less distance traveled, smaller per-atom forces (i.e. energy gradient with respect to positions), and with fewer iterations.

In these calculations, we find that the convergence rates of methods are not significantly different (see Figure 10). Comparing expectation values of the maximum force after optimization, however, we see that SymmCD is significantly more successful in generating structures that are readily optimized. The full results of this analysis, as well as the parameters used for CP2K can be found in Appendix F.6.

### 5.4 PROXY METRICS

Although ultimately we care about the stability, novelty, and properties of crystals after structural relaxation, we also compare the different methods using the proxy metrics established by Xie et al. (2022), as they are cheap enough to compute over a large set of generated samples and demonstrate the ability of different methods to capture the properties of a target distribution. These metrics include heuristics of validity, coverage of the test set, and the Wasserstein distances between the distributions of three properties of the generated samples and the test set: atomic densities  $ d_{\rho} $ , number of unique elements  $ d_{elem} $ , and predicted formation energy  $ d_{E} $ . We also include  $ d_{sg} $ , the Jensen-Shannon distance between space group distributions. Further details are included in Appendix F.5.

For each method, we retrained a model with 5 different seeds, generating 10,000 samples for evaluation per seed. The results are shown in Table 3. Although SymmCD performs similarly to other methods for most metrics, it is more likely to generate structurally invalid crystals, possibly because it cannot easily see distances between atoms. We also note the large variance in the results across different seeds for each method.

<div style="text-align: center;">Table 3: Results for comparing the validity, coverage, and property distribution metrics.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">MP-20</td><td colspan="2">Validity (%) ( $ \uparrow $ )</td><td colspan="2">Coverage (%) ( $ \uparrow $ )</td><td colspan="4">Property Distribution ( $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>Struct.</td><td style='text-align: center;'>Comp.</td><td style='text-align: center;'>Recall</td><td style='text-align: center;'>Precision</td><td style='text-align: center;'>$ d_{\rho} $</td><td style='text-align: center;'>$ d_{E} $</td><td style='text-align: center;'>$ d_{\text{elem}} $</td><td style='text-align: center;'>$ d_{\text{sg}} $</td></tr><tr><td style='text-align: center;'>CDVAE</td><td style='text-align: center;'>99.97 $ \pm $ 0.03</td><td style='text-align: center;'>85.61 $ \pm $ 1.20</td><td style='text-align: center;'>99.31 $ \pm $ 0.06</td><td style='text-align: center;'>99.47 $ \pm $ 0.19</td><td style='text-align: center;'>0.70 $ \pm $ 0.10</td><td style='text-align: center;'>0.24 $ \pm $ 0.06</td><td style='text-align: center;'>1.28 $ \pm $ 0.06</td><td style='text-align: center;'>0.69 $ \pm $ 0.007</td></tr><tr><td style='text-align: center;'>DiffCSP</td><td style='text-align: center;'>97.43 $ \pm $ 3.10</td><td style='text-align: center;'>82.50 $ \pm $ 1.34</td><td style='text-align: center;'>99.55 $ \pm $ 0.08</td><td style='text-align: center;'>98.73 $ \pm $ 1.34</td><td style='text-align: center;'>0.18 $ \pm $ 0.08</td><td style='text-align: center;'>0.14 $ \pm $ 0.05</td><td style='text-align: center;'>0.56 $ \pm $ 0.07</td><td style='text-align: center;'>0.44 $ \pm $ 0.009</td></tr><tr><td style='text-align: center;'>DiffCSP++</td><td style='text-align: center;'>99.44 $ \pm $ 0.06</td><td style='text-align: center;'>86.50 $ \pm $ 0.85</td><td style='text-align: center;'>99.72 $ \pm $ 0.06</td><td style='text-align: center;'>99.61 $ \pm $ 0.08</td><td style='text-align: center;'>0.12 $ \pm $ 0.04</td><td style='text-align: center;'>0.05 $ \pm $ 0.01</td><td style='text-align: center;'>0.33 $ \pm $ 0.04</td><td style='text-align: center;'>0.16 $ \pm $ 0.009</td></tr><tr><td style='text-align: center;'>FlowMM</td><td style='text-align: center;'>96.67 $ \pm $ 0.57</td><td style='text-align: center;'>83.25 $ \pm $ 0.13</td><td style='text-align: center;'>99.49 $ \pm $ 0.05</td><td style='text-align: center;'>99.58 $ \pm $ 0.10</td><td style='text-align: center;'>0.23 $ \pm $ 0.10</td><td style='text-align: center;'>0.09 $ \pm $ 0.02</td><td style='text-align: center;'>0.08 $ \pm $ 0.02</td><td style='text-align: center;'>0.50 $ \pm $ 0.011</td></tr><tr><td style='text-align: center;'>SymmCD $ _{\text{All}} $  SGs</td><td style='text-align: center;'>90.34 $ \pm $ 3.39</td><td style='text-align: center;'>85.81 $ \pm $ 0.87</td><td style='text-align: center;'>99.58 $ \pm $ 0.08</td><td style='text-align: center;'>97.76 $ \pm $ 0.85</td><td style='text-align: center;'>0.23 $ \pm $ 0.06</td><td style='text-align: center;'>0.21 $ \pm $ 0.04</td><td style='text-align: center;'>0.40 $ \pm $ 0.06</td><td style='text-align: center;'>0.164 $ \pm $ 0.003</td></tr><tr><td style='text-align: center;'>SymmCD $ _{\text{10 SGs}} $</td><td style='text-align: center;'>92.30 $ \pm $ 6.10</td><td style='text-align: center;'>87.13 $ \pm $ 0.62</td><td style='text-align: center;'>97.33 $ \pm $ 0.59</td><td style='text-align: center;'>98.78 $ \pm $ 0.69</td><td style='text-align: center;'>0.53 $ \pm $ 0.23</td><td style='text-align: center;'>0.21 $ \pm $ 0.09</td><td style='text-align: center;'>0.16 $ \pm $ 0.02</td><td style='text-align: center;'>0.469 $ \pm $ 0.002</td></tr></table>

### 5.5 COMPUTATIONAL EFFICIENCY

We demonstrate significant computational efficiency gains and reduced memory footprint due to using a more compact representation based on crystallographic orbits. We compare our model to an equivalent model that looks at a full unit cell, rather than just the asymmetric unit. It also uses a fully connected graph to

<div style="text-align: center;">Table 4: Computational efficiency of our representation</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Asymmetric Unit (ours)</td><td style='text-align: center;'>Conventional Unit Cell</td></tr><tr><td style='text-align: center;'>Maximum batch size ( $ \uparrow $ )</td><td style='text-align: center;'>8192</td><td style='text-align: center;'>512</td></tr><tr><td style='text-align: center;'>Memory for 512 batch size ( $ \downarrow $ )</td><td style='text-align: center;'>3.6 GB</td><td style='text-align: center;'>31 GB</td></tr><tr><td style='text-align: center;'>Time for 1 training epoch ( $ \downarrow $ )</td><td style='text-align: center;'>27 sec.</td><td style='text-align: center;'>52 sec.</td></tr></table>

represent the atoms in the unit cell, but does not predict site symmetry representations. This makes the model essentially equivalent to DiffCSP (Jiao et al., 2023), but with the same architecture and hyperparameters as SymmCD for consistent comparison. We compare the two representations for one epoch of training using 40GB of RAM and a single NVIDIA MIG A100 and report the results in Table 4. These results highlight SymmCD's memory efficiency and faster training capabilities.

### 5.6 SCALING UP TO MPTS-52

To validate that SymmCD can scale up to datasets of larger crystals thanks to its compact representation, we also validate our model by training it on the more challenging MPTS-52 dataset (Baird et al., 2024), which contains crystals with up to 52 atoms per primitive unit cell, and includes more chemical elements. Further details of this dataset along with the results are included in Appendix F.1.

We note that none of the baselines we compare against (CDVAE, DiffCSP, FlowMM, and DiffCSP++) present results for de-novo generation on this dataset. We can see from these results that SymmCD is able to scale to much larger crystals than those in the MP-20 dataset and is still capable of producing valid, stable, and novel crystals that match the training distribution of the dataset. The metrics reported are worse than those for MP-20, as this is a more difficult dataset.

## Conclusion

In this paper, we introduced a novel approach for generating crystals with precise symmetry properties. We proposed to leverage asymmetric units and site symmetry representations within a diffusion model framework. This approach ensures that the generated crystals inherently preserve desired symmetries while allowing greater diversity, computational efficiency, and flexibility in the generation process. To encode crystal and site symmetries we introduced a new representation of crystal symmetries that enables information sharing across space groups, improving generalization when learning with a diverse set of crystal symmetries. Our results indicate that this method produces stable, novel, and structurally diverse crystals, with improved computational efficiency, showing promise for discovery in materials science. In this work, we focused on inorganic crystals, but SymmCD could potentially be promising for applications on molecular crystals and co-crystals, which also have non-trivial symmetries. Beyond materials, other data types such as molecules and graphs often exhibit complex symmetries, and future work could investigate if symmetry constraints could also be useful in generative models for these modalities.

A limitation of our framework is that it makes it more challenging to perform crystal structure prediction given a composition since it relies on sampling a space group first, and then a composition conditioned on the space group. Finally, an important area of future work in generative models for crystals is also to go beyond single crystals, and consider the generation of polycrystalline materials. These types of materials are common in applications, yet not suited to generation using single unit cells or asymmetric units.

## ACKNOWLEDGMENTS

This project is supported by Intel Labs, CIFAR and the NSERC Discovery grant. S.-O. K.'s research is also supported by IVADO and the DeepMind Scholarships, and D.L.'s research is partly supported by the FRQNT. Computational resources were provided by Mila and Intel Labs. We are thankful to Alexandra Volokhova, Victor Schmidt, Alex Hernandez-Garcia, Alexandre Duval, and Félix Therrien for helpful discussions.