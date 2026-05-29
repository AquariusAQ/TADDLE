## A REPLICATION

We define the replication operator as  $ R: G \times P \times R^{3} \to 2^{R^{3}} $ . This operation is defined by considering the group  $ S_{x} \ltimes T_{S} $ , with  $ T_{S} $  being the group of translations defined by the lattice.  $ S_{x} \ltimes T_{S} $  is the set of operations that preserve the position of x within the unit cell as opposed to within the crystal. We can then consider the coset decomposition of the space group with respect to that group  $ G/(S_{x} \ltimes T_{S}) $ . Then, we denote by  $ [G/(S_{x} \ltimes T_{S})]_{0} $  a system of coset representatives where the translation parts are chosen to move only within the unit cell. This defines the set of operations that move a position x within its orbit and the unit cell. The replication operation then simply consists of applying all these operations:

 $$ \mathsf{R}\left(G,S_{\mathbf{x}},\mathbf{x}\right)=\left(\left(\mathbf{O},\mathbf{t}\right)\mathbf{x}\mid\left(\mathbf{O},\mathbf{t}\right)\in\left[G/\left(S_{\mathbf{x}}\ltimes T_{\mathbf{S}}\right)\right]_{\mathbf{0}}\right) $$ 

The representation in terms of individual atoms is then:

 $$ \mathbf{X}=\bigoplus_{i}^{M}\mathsf{R}\left(G,S_{\mathbf{x}_{i}^{\prime}},\mathbf{x}_{i}^{\prime}\right) $$ 

 $$ \mathbf{A}=\bigoplus_{i}^{M}\operatorname{r e p e a t}\left(\mathbf{a}_{i},\left[G:\left(S_{\mathbf{x}}\ltimes T_{\mathbf{S}}\right)\right]\right) $$ 

where repeat  $ (\mathbf{a}, n) $  repeats the vector a n times and  $ [G : (S_{x} \ltimes T_{S})] $  is the multiplicity of the orbit.

In our diffusion model, our predicted site symmetries  $ \hat{S} $  do not always necessarily correspond to a valid crystallographic point group. To get around this, we project  $ \hat{S} $  to the nearest point group that is a subgroup of the given space group, as measured by the Frobenius Norm of their difference. Once a point group is chosen, the PyXtal search_closest_wp function is used to get the nearest coordinates to  $ X^{\prime} $  that correspond to a Wyckoff position with the given site symmetry, and  $ X^{\prime} $  is updated to be placed on those coordinates (Fredericks et al., 2021). Finally, the representative atoms at the Wyckoff position are replicated, using operations implemented in PyXtal.

## B LATTICE REPRESENTATION

We use the lattice representations derived by Jiao et al. (2024), as they are useful for constraining lattices to respect the symmetries of a given space group. The authors found that any lattice matrix L can be written as  $ \mathbf{L} = \mathbf{Q} \exp(\mathbf{S}) $  for some orthogonal Q (which we can ignore, as orthogonal transformations do not change the lattice), and symmetric S. The matrix S can then be decomposed into a sum of the following basis lattices:

 $$ \mathbf{B}_{1}=\begin{pmatrix}{{{0}}}&{{{1}}}&{{{0}}} \\{{{1}}}&{{{0}}}&{{{0}}} \\{{{0}}}&{{{0}}}&{{{0}}}\end{pmatrix},\quad\mathbf{B}_{2}=\begin{pmatrix}{{{0}}}&{{{0}}}&{{{1}}} \\{{{0}}}&{{{0}}}&{{{0}}} \\{{{1}}}&{{{0}}}&{{{0}}}\end{pmatrix},\quad\mathbf{B}_{3}=\begin{pmatrix}{{{0}}}&{{{0}}}&{{{0}}} \\{{{0}}}&{{{0}}}&{{{1}}} \\{{{0}}}&{{{1}}}&{{{0}}}\end{pmatrix}, $$ 

 $$ \mathbf{B}_{4}=\begin{pmatrix}{{{1}}}&{{{0}}}&{{{0}}} \\{{{0}}}&{{{-1}}}&{{{0}}} \\{{{0}}}&{{{0}}}&{{{0}}}\end{pmatrix},\quad\mathbf{B}_{5}=\begin{pmatrix}{{{1}}}&{{{0}}}&{{{0}}} \\{{{0}}}&{{{1}}}&{{{0}}} \\{{{0}}}&{{{0}}}&{{{-2}}}\end{pmatrix},\quad\mathbf{B}_{6}=\begin{pmatrix}{{{1}}}&{{{0}}}&{{{0}}} \\{{{0}}}&{{{1}}}&{{{0}}} \\{{{0}}}&{{{0}}}&{{{1}}}\end{pmatrix}. $$ 

with  $ S = \sum_{i=1}^{6} k_{i} B_{i} $ . They derive constraints on  $ k_{i} $  depending on the space groups that a crystal belongs to:

• Triclinic:  $ \mathbf{k}=(k_{1},k_{2},k_{3},k_{4},k_{5},k_{6}) $ 

 $$ \mathbf{M o n o c l i n i c:}\mathbf{k}=(0,k_{2},0,k_{4},k_{5},k_{6}) $$ 

• Orthorhombic: $\mathbf{k}=(0,0,0,k_{4},k_{5},k_{6})$

• Tetragonal:  $ \mathbf{k}=(0,0,0,0,k_{5},k_{6}) $ 

• Hexagonal:  $ \mathbf{k}=(-\log(3)/4,0,0,0,k_{5},k_{6}) $ 

• Cubic:  $ \mathbf{k}=(0,0,0,0,0,k_{6}) $ 

## C SITE SYMMETRY REPRESENTATION

The 15 possible symmetry axes of crystals are: [001], [010], [100], [111],  $ [1\bar{1}1] $ ,  $ [\bar{1}11] $ ,  $ [\bar{1}11] $ , [110],  $ [1\bar{1}0] $ , [101], [101], [011], [011], [210], [120], [110]. These axes are written in short form: for example, [110] denotes the direction of the vector  $ (1,-1,0) $ . They are shown in Figure 2. The axes depend on the symmetries of the crystal system: for example, in an orthorhombic crystal (a rectangular prism whose side lengths are not necessarily equal), a crystal may have different site symmetries oriented around the x, y, or z-axes. Conversely, in a tetragonal crystal (a rectangular prism with a square base), any site symmetry oriented along the x-axis must also be along the y-axis, there may be additional symmetries along the diagonal of the x-y plane.

The possible set of symmetry elements along each axis for a site symmetry group correspond to the identity 1; an inversion  $ \bar{1} $ ; rotations of different orders 2, 3, 4, and 6; rotoinversions  $ \bar{2} $  (equivalent to a mirror symmetry m across a plane perpendicular to the axis),  $ \bar{3} $ ,  $ \bar{4} $ , and  $ \bar{6} $ ; and combinations of rotations and mirror reflections 2/m, 4/m, and 6/m. This enumeration yields 13 possible symmetries along each axis.

The possible symmetry elements along each axis for a space group correspond to the identity 1; an inversion  $ \bar{1} $ ; rotations of different orders 2, 3, 4, and 6; rotoinversions  $ \bar{2} $  (equivalent to a mirror symmetry m across a plane perpendicular to the axis),  $ \bar{3} $ ,  $ \bar{4} $ , and  $ \bar{6} $ ; screws  $ 2_{1} $ ,  $ 3_{1} $ ,  $ 3_{2} $ ,  $ 4_{1} $ ,  $ 4_{2} $ ,  $ 4_{3} $ ,  $ 6_{1} $ ,  $ 6_{2} $ ,  $ 6_{3} $ ,  $ 6_{4} $ ,  $ 6_{5} $ , and glides a, b, c, n, d, e.

To encode a space group, an additional 7-dimensional one-hot encoding is used to denote the Bravais lattice to which the space group belongs. This yields a  $ (26 \times 15) + 7 = 397 $  dimensional binary representation of space group.

The representations can now be accessed using the symmetry module of PyXtal (Fredericks et al., 2021).

## D DIFFUSION AND DENOISING PROCESS DETAILS

Diffusion on lattice parameters k Inspired by Jiao et al. (2024), we perform diffusion over k, the  $ O(3) $ -invariant lattice representation. The forward noising process is given by  $ q(\mathbf{k}_{t}|\mathbf{k}_{0}) \sim \mathcal{N}(\mathbf{k}_{t}|\sqrt{\alpha_{t}}\mathbf{k}_{0}, (1 - \bar{\alpha}_{t})\mathbf{I}) $ , where  $ k_{t} $  is the noised version of  $ k_{0} $  at timestep t. Here, similar to Nichol & Dhariwal (2021),  $ \bar{\alpha}_{t} = \Pi_{j=1}^{t}(1 - \beta_{j}) $ , where  $ \beta_{j} \in (0, 1) $  determines variance in each step controlled by the cosine scheduler. During the generation process, we start with  $ \mathbf{k}_{T} \sim \mathcal{N}(0, \mathbf{I}) $  and use learned denoising network to generate  $ k_{t-1} $  from  $ k_{t} $ :

 $$ \begin{aligned}&p_{\theta}(\mathbf{k}_{t-1}|\mathcal{C}_{t}^{\prime})=\mathcal{N}\Big(\mathbf{k}_{t-1}|\boldsymbol{\mu}_{\mathbf{k}}(t),\boldsymbol{\sigma}(t)\mathbf{I}\Big),\\&\boldsymbol{\mu}_{\mathbf{k}}(t)=\frac{1}{\sqrt{\bar{\alpha}_{t}}}\Big(\mathbf{k}_{t}-\frac{\beta_{t}}{\sqrt{1-\bar{\alpha}_{t}}}\hat{\epsilon}_{\mathbf{k}}(\mathcal{C}_{t}^{\prime},t)\Big),\boldsymbol{\sigma}(t)=\beta_{t}\frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_{t}}.\\ \end{aligned} $$ 

Here,  $ C_{t}^{\prime} $  is the noised crystal and  $ \hat{\epsilon}_{k}(\mathcal{C}_{t}^{\prime},t) $  is the predicted denoising term predicted from a denoising network  $ \phi(\mathcal{C}_{t}^{\prime},t) $ . We also use a mask m to only implement diffusion over unconstrained dimensions of  $ k_{t} $ , since depending upon space groups, certain dimensions have fixed values (Appendix B). The mask can be represented as  $ m\in\{0,1\}^{6} $  and  $ m_{i}=1 $  indicates that  $ i^{th} $  index of k is unconstrained. The corresponding loss used to train the denoising network is:

 $$ \mathcal{L}_{\mathbf{k}}=\mathbb{E}_{\mathbf{\epsilon  _{k}}\sim\mathcal{N}(0,\mathbf{I}),t\sim\mathrm{U}(1,T)}[||m\odot\mathbf{\epsilon  _{k}}-\hat{\mathbf{\epsilon  _{k}}}(\mathcal{C}_{t}^{\prime},t)||_{2}^{2}] $$ 

where  $ \odot $  is the elementwise product and  $ U(1,T) $  is a uniform distribution over timesteps.

Diffusion over representative fractional coordinates  $ X^{\prime} $ . We perform diffusion over the fractional coordinates using the same method as Jiao et al. (2023). Due to the periodicity of fractional coordinates, the noising process  $ q(\mathbf{X}_{t}|\mathbf{X}_{0}) $  is determined by a Wrapped Normal distribution rather than a Gaussian distribution, and we initialize the fractional coordinates  $ X_{T} $  with the uniform distribution  $ U(0,1) $  when sampling.

Diffusion on atom types  $ A^{\prime} $  We use discrete diffusion from Austin et al. (2021) to sample the atom types of each representative. If  $ a_{0} \in \{0,1\}^{Z} $  is the one-hot encoding of atom types for a single representative, then we can noise it as:  $  q(\mathbf{a}_{t}|\mathbf{a}_{0}) = \operatorname{Cat}(\mathbf{a}_{t}; \mathbf{p} = \mathbf{a}_{0}^{\top} \bar{\mathbf{Q}}_{t})  $ , where  $ Q_{t} = \prod_{i=1}^{t} Q_{i} \in R^{Z \times Z} $  is the cumulative product of transition matrices between timesteps. Inspired by Vignac et al. (2023), the transition matrix can be parametrized as  $ Q_{t} = \alpha_{t} I + \beta_{t} m_{a} $ , where  $ m_{a} $  are the marginals over the atom types in the data, and  $ \alpha_{t} $  and  $ \beta_{t} $  are scheduling parameters. The effect of this noising scheme is that regardless of  $ a_{0} $ , the fully noised  $ a_{T} = a_{0}^{\top} Q_{T} = m_{a} $ , so we can sample from the prior distribution  $ m_{a} $ , which is close to the data distribution. The discrete diffusion model is trained using a cross-entropy loss:

 $$ \mathcal{L}_{\mathbf{A}^{\prime}}=\mathbb{E}_{\mathbf{a}_{t}\sim\mathrm{C a t}(\mathbf{a}_{0}^{\top}\bar{\mathbf{Q}}_{t}),t\sim\mathrm{U}(1,T)}\sum_{i=1}^{M}\mathrm{C r o s s E n t r o p y}(\mathbf{a}_{i}^{\prime},\hat{\mathbf{a}}_{i}^{\prime}), $$ 

where  $ \hat{a}_{i} $  are the probabilities predicted by the model  $ \phi(\mathcal{C}_{t}^{\prime}, t) $ . To sample from the discrete diffusion model, we sample from the marginal distribution over atom types  $ m_{a} $ , then progressively denoise using:

 $$ q(\mathbf{a}_{t-1}|\mathbf{a}_{t},\mathbf{a}_{0})=\mathrm{C a t}\left(\mathbf{a}_{t-1};\mathbf{p}=\frac{\mathbf{a}_{t}^{\top}\mathbf{Q}_{t}^{\top}\odot\mathbf{a}_{0}^{\top}\bar{\mathbf{Q}}_{t-1}}{\mathbf{a}_{0}^{\top}\bar{\mathbf{Q}}_{t}\mathbf{a}_{t}}\right) $$ 

More details of this implementation can be seen in Vignac et al. (2023).

Diffusion for site symmetries S The site symmetry representation matrices described in Section 4.2 can be thought of as 15 separate 13-dimensional categorical variables: one site symmetry operation per axis. Our diffusion model over site symmetries is almost identical to the method for atom types, applying discrete diffusion separately over each of the axes. Because the site symmetries depend strongly on the space group, we use transition matrices that are different for each space group:  $ Q_{t,i,G} = \alpha_{t}I + \beta_{t}m_{S_{\alpha},G} $ , where  $ m_{S_{\alpha},G} $  denotes the marginals over site symmetry operations for axis  $ S_{u} $  given space group G. For each representative node, we average the cross-entropy loss over each of the axes.

## E ARCHITECTURE DETAILS

### E.1 Denoising Model

We use a graph neural network based on the architecture of Jiao et al. (2023). We embed the timestep t using sinusoidal embeddings,  $ \psi_{t}(t) $ . We embed our space group representation from Section 4.2 using an MLP,  $ \phi_{G}(G) $ . We embed our site symmetries by separately embedding each axis using the same network, and feeding the resulting embeddings into a secondary MLP:  $ \phi_{S}(\bigoplus_{u=1}^{15}\phi_{U}(S_{u})) $ . These are all used to initialize the node embeddings  $ h_{i} $ .

 $$ \mathbf{h}_{i}\leftarrow\phi_{h}(\mathbf{a}_{i},\mathbf{x}_{i},\phi_{S}\left(\bigoplus_{u=1}^{15}\phi_{U}(S_{u})\right),\phi_{G}(G),\psi_{t}(t)). $$ 

As noted earlier, we directly use coordinates x, because we are working a conventional or canonical lattice, and so Euclidean symmetries are not necessarily useful here.

At each layer we compute messages and use them to update node embeddings:

 $$ \begin{aligned}\mathbf{m}_{ij}&\leftarrow\phi_{m}(\mathbf{h}_{i},\mathbf{h}_{j},\mathbf{k},\psi(\mathbf{x}_{i}-\mathbf{x}_{j}))\\\mathbf{h}_{i}&\leftarrow\mathbf{h}_{i}+\phi_{h}(\mathbf{h}_{i},\sum_{j}^{M}\mathbf{m}_{ij})\end{aligned} $$ 

Here,  $ \psi $  is a Fourier embedding,  $ \phi_{m} $  and  $ \phi_{h} $  are MLPs acting on edges and nodes respectively. We use a SiLU activation function for each MLP. Finally, we output predicted  $ \hat{\epsilon}_{X'} $ ,  $ \hat{A}' $  and  $ \hat{S} $  using the node embeddings  $ h_{i} $ , and  $ \hat{\epsilon}_{k} $  using  $ \sum_{i}^{M} h_{i} $ .

<div style="text-align: center;"><img src="imgs/img_in_chart_box_240_97_949_232.jpg" alt="Image" width="57%" /></div>


<div style="text-align: center;">Figure 5: Proportion of space group symmetries of the MPTS-52 dataset. From left to right, the first few space groups are: Pnma, P2_{1}/c, Fm $ \overline{3} $ m, I4/mmm, P6_{3}/mmc, Pm $ \overline{3} $ m, C2/m, C2/c,  $ \overline{1} $ , and Cmcm.</div>


<div style="text-align: center;">Table 5: Template statistics for MPTS-52</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'># Unique</td><td style='text-align: center;'>% in Train</td><td style='text-align: center;'># New</td></tr><tr><td style='text-align: center;'>Training Set</td><td style='text-align: center;'>4452</td><td style='text-align: center;'>100</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>SymmCD</td><td style='text-align: center;'>2772</td><td style='text-align: center;'>48.3</td><td style='text-align: center;'>1432</td></tr></table>

<div style="text-align: center;">Table 6: Percent of stable and S.U.N. samples produced from an initial set of 10,000 generated crystals for SymmCD trained on the MPTS-52 dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Initial Stable</td><td style='text-align: center;'>Relaxed Stable</td><td style='text-align: center;'>Relaxed S.U.N.</td></tr><tr><td style='text-align: center;'>1.72%</td><td style='text-align: center;'>5.97%</td><td style='text-align: center;'>4.62%</td></tr></table>

<div style="text-align: center;">Table 7: The validity, coverage, and property distribution metrics for SymmCD trained on the MPTS-52 dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Validity (%) ( $ \uparrow $ )</td><td style='text-align: center;'>Coverage (%) ( $ \uparrow $ )</td><td style='text-align: center;'>Property Distribution ( $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>Struct. Comp.</td><td style='text-align: center;'>Recall</td><td style='text-align: center;'>Precision</td></tr><tr><td style='text-align: center;'>90.1</td><td style='text-align: center;'>79.2</td><td style='text-align: center;'>99.6</td></tr></table>

### E.2 Model Hyperparameters

The graph neural network has 8 layers, and we use a representation dimension of 1024 for  $ h_{i} $ . We encode distances between nodes using a sinusoidal embedding, with 128 different frequencies. We encode the timestep t into a 10 dimensional vector. We apply layer normalization at each layer of the GNN. The loss coefficients selected were  $ \lambda_{k}=5 $ ,  $ \lambda_{X^{\prime}}=1 $ ,  $ \lambda_{A^{\prime}}=0.1 $  and  $ \lambda_{S}=10 $ .

We performed two hyperparameter sweeps: we first tested each combination of  $ \lambda_{k} $ ,  $ \lambda_{A'} $  and  $ \lambda_{S} $  set to values in  $ \{0.1, 0.5, 1, 5, 10\} $ , while keeping  $ \lambda_{X'} $  fixed at 1, and then selected the loss coefficients that lead to the highest structural validity. Next, we performed a random sweep of other architecture parameters, running 150 different hyperparameter combinations and choosing a model that had high performance on structural validity, compositional validity, and  $ d_{E} $ . We varied the number of GNN layers in  $ \{6, 8, 12, 16\} $ , representation dimension in  $ \{256, 512, 1024\} $ , time embedding dimension in  $ \{10, 64, 256\} $ , and varied whether layer normalization was used.

## F EXTENDED RESULTS

### F.1 MPTS 52 DATASET

In addition to MP-20, we also trained SymmCD on the MPTS-52 dataset (Baird et al., 2024), a more challenging subset of the Materials Project that contains materials with up to 52 atoms per primitive unit cell. Unlike MP-20, it does not filter out materials containing radioactive elements. The dataset contains 40,476 samples with a train/validation/test split of 27,380/5,000/8,096 crystals. The splits are in chronological order, with the materials in the test set having been discovered most recently, and the materials in the training set having been discovered earliest. The distribution of space groups found in the dataset are shown in Figure 5. None of the diffusion and flow-based methods we compared against (CDVAE, DiffCSP, DiffCSP++, or FlowMM) have reported de-novo generation results on this dataset.

We trained SymmCD on MPTS-52 using all of the same hyperparameters as were used for the MP-20 dataset, but trained for 1500 epochs. We sampled 10,000 crystals, and checked the same proxy metrics. We also relaxed the generated crystals using CHGNet and checked for whether the generated crystals were S.U.N. The unique templates (as described in Section 5.1) of the dataset and of our generated crystals are shown in Table 5. The results for the proxy metrics are shown in Table 7 and stability results are shown in Table 6.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_416_167_802_400.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">Figure 6: Histogram of the number of atoms in crystals from MP-20 and generated by SymmCD when trained on MP-20.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_222_485_1000_744.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure 7: Histogram of the number of atoms in crystals from MPTS-52 and generated by SymmCD when trained on MPTS-52.</div>


### F.2 NUMBER OF ATOMS

To demonstrate that SymmCD is able to correctly predict reasonable site symmetries, we show here that the distribution of number of atoms per crystal matches the dataset it is trained on. This is not a trivial task, as the model needs to learn the multiplicity of different possible site symmetries, which depends on both the different symmetry elements of the site symmetry and the space group that it belongs to. The comparison for MP-20 is shown in Figure 6, and the comparison for MPTS-52 is shown in Figure 7.

### F.3 PROPERTY PREDICTION TASK

We test the usefulness of our site symmetry representation using a regression experiment. We selected formation energy per atom as the target property to predict.

We use DimeNet++ (Gasteiger et al., 2020b;a) as a base model to perform ablation over the type of input graph and encoding site symmetry information per node.

One input format is a multi-graph (Xie et al., 2022), which describes the unit cell as a graph with nodes as atoms and edges between them according to a cutoff radius. These edges could potentially span to neighbouring unit cells. The other input format is the asymmetric unit that we use in SymmCD. Under these two inputs, we test the effects of including a site symmetry encoding for each node. We report the Mean Absolute Error (MAE) for the test set in Table 8. We see that the effect of including site symmetry

Table 8: Mean average error when predicting crystal formation energy. The input could be the asymmetric unit or a multi-graph, and the site symmetry information can be encoded or ignored. We observe that our encoding of site symmetry helps predict the target property.




<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Multigraph</td><td style='text-align: center;'>Asymm. Unit</td></tr><tr><td style='text-align: center;'>W/out S</td><td style='text-align: center;'>0.0214</td><td style='text-align: center;'>0.0711</td></tr><tr><td style='text-align: center;'>With S</td><td style='text-align: center;'>0.0212</td><td style='text-align: center;'>0.0490</td></tr></table>

information is minimal when we have access to the full

graph. However, we see that when we are restricted to only using the asymmetric unit, having access to the site symmetry info greatly helps, showing that we can recover some geometric information lost when using just an asymmetric unit by also including symmetry.

### F.4 EXAMPLES

In Figure 8, we include 6 randomly sampled crystals generated by SymmCD along with their respective space groups.

<div style="text-align: center;"><img src="imgs/img_in_image_box_217_433_424_630.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">(a) PaTi_{3} Pm $ \overline{3} $ m</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_488_444_697_631.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(b)  $ CeSiGe_{2}Os $  I4mm</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_773_452_983_626.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(c)  $ Cs_{2}CuO_{4} $  I4mm</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_217_734_426_929.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(d)  $ SrYO_{3} $  I4 $ _{m} $ cm</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_497_738_697_927.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">(e) Ta₂Nb₄V₄CoMo₂C Pmmm</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_767_725_973_930.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">(f) TbInAu_{2}  Fm $ \overline{3} $ </div>


<div style="text-align: center;">Figure 8: Example materials generated by SymmCD, along with their chemical formulae and space groups symmetries.</div>


### F.5 PROXY METRICS

We compare the different methods using the metrics established by Xie et al. (2022), measuring the validity, coverage, and property statistics of the generated crystals. We measure the validity by checking structural validity, defined as whether no two atoms are closer than 0.5 Å apart, and compositional validity, defined as whether the charges are balanced as determined by SMACT (Davies et al., 2019). It should be noted that the compositional validity of the MP-20 dataset is only 92%. To determine coverage, we examine the CrystalNN structural fingerprints (Zimmermann & Jain, 2020) and Magpie compositional fingerprints (Ward et al., 2016) of the valid generated crystals, and look at their distances to the fingerprints of the crystals in the test set. If the distance is under some cutoff, then the crystals are matched, giving both recall and precision metrics. We look at the distances between the properties of the valid generated crystals and the crystals from the test set to compare the ability of each model to match the data distribution. We specifically compare the Wasserstein distances between the atomic densities  $ d_{\rho} $ , number of unique elements  $ d_{elem} $ , and the formation energy  $ d_{E} $  predicted by a pretrained DimeNet++ model (Gasteiger et al., 2020a). We also look at the Jensen-Shannon distance between the space groups of the generated valid structures and the space groups of the crystals in the test set, denoted by  $ d_{sg} $ . The coverage and property statistics are computed only for a random subset of 1000 valid crystals per method. For each method, we

<div style="text-align: center;"><img src="imgs/img_in_chart_box_311_179_910_520.jpg" alt="Image" width="48%" /></div>


<div style="text-align: center;">Figure 9: Distribution of total atom displacements over the course of relaxation. Displacements are averaged over structures for a given method.</div>


train 5 different models with different random seeds so that we could see the variance for each of the metrics.

### F.6 DENSITY FUNCTIONAL THEORY

We performed cell and geometry relaxation calculations using the PBE functional, DZVP-MOLOPT-SR-GTH basis set, and GTH-PBE pseudopotential with the QUICKSTEP program from CP2K (Perdew et al., 1996; Kühne et al., 2020). Table 9 shows the settings—particularly convergence thresholds—used for performing the relaxations. Values were generally tuned to balance between a feasible computational budget and a fair benchmark between methods: as an example, we allow relaxations to attempt to continue even with poorly convergent SCF in the hopes that optimization trajectories will still have a chance of finding a local minimum. With that in mind, we note that the convergence thresholds are also generally set to relatively "lax" values, and for end property values stricter convergence thresholds may be necessary.

Figure 9 shows distributions of the cumulative atom displacement, averaged over structures for each method: in other words, the total distance the average atom travels over the full course of the relaxation. Naturally, the less atom displacement, the more likely a given generative method produces high fidelity crystal structure samples. We see that DiffCSP++ and both treatments of SymmCD produce samples with displacements that peak close to zero, while other methods peak closer to  $ \sim4.5\AA $ . The relatively long tails out to high cumulative displacements seen with CDVAE and FlowMM are attributed to trajectories that do not converge after the specified number of relaxation steps. From this, we can conclude that with the exception of SymmCD and DiffCSP++, the tested generative methods demonstrate a similar degree of fidelity requiring some degree of optimization.

Figure 10 shows empirical distribution functions of the maximum gradient value at the end of the relaxation trajectory, regardless of the state of convergence: the smaller the maximum gradient is, the closer the relaxation ended in a local minimum. The first observation is that across all methods, a significant portion ( $ \sim $ 60%) of trajectories fail to converge (the portion to the right of the dashed line)—the majority of sampled structures fail to converge, based on our naive attempt to relax them using our reasonable choice of method, basis set, and pseudopotential. Another observation is the long tail towards low values for the 10 space group treatment of SymmCD, which provides compelling evidence for extremely high fidelity samples being produced by SymmCD. The expected values of the maximum gradient for each method are shown in Table 10, where we can observe that SymmCD tends to have a lower maximum gradient after relaxation.

<div style="text-align: center;">Table 9: Configuration settings for CP2K. Settings that are omitted from this table assume their default values.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Parameter</td><td style='text-align: center;'>Value</td></tr><tr><td colspan="2">Base SCF settings</td></tr><tr><td style='text-align: center;'>EPS_SCF</td><td style='text-align: center;'>10^{-7}</td></tr><tr><td style='text-align: center;'>MAX_SCF</td><td style='text-align: center;'>300</td></tr><tr><td style='text-align: center;'>MAX_ITER_LUMO</td><td style='text-align: center;'>400</td></tr><tr><td style='text-align: center;'>IGNORE_CONVERGENCE_FAILURE</td><td style='text-align: center;'>T</td></tr><tr><td colspan="2">Orbital transformation</td></tr><tr><td style='text-align: center;'>Orbital transformation method</td><td style='text-align: center;'>IRAC</td></tr><tr><td style='text-align: center;'>ENERGY_GAP</td><td style='text-align: center;'>10^{-3}</td></tr><tr><td style='text-align: center;'>MINIMIZER</td><td style='text-align: center;'>DIIS</td></tr><tr><td style='text-align: center;'>LINESEARCH</td><td style='text-align: center;'>2PNT</td></tr><tr><td style='text-align: center;'>PRECONDITIONER</td><td style='text-align: center;'>FULL_ALL</td></tr><tr><td colspan="2">Outer SCF settings</td></tr><tr><td style='text-align: center;'>MAX_SCF</td><td style='text-align: center;'>20</td></tr><tr><td style='text-align: center;'>EPS_SCF</td><td style='text-align: center;'>10^{-6}</td></tr><tr><td colspan="2">Cell optimization</td></tr><tr><td style='text-align: center;'>TYPE</td><td style='text-align: center;'>DIRECT_CELL_OPT</td></tr><tr><td style='text-align: center;'>MAX_ITER</td><td style='text-align: center;'>100</td></tr><tr><td style='text-align: center;'>OPTIMIZER</td><td style='text-align: center;'>BFGS</td></tr><tr><td colspan="2">Geometry optimization</td></tr><tr><td style='text-align: center;'>MAX_DR</td><td style='text-align: center;'>3 \times 10^{-3}</td></tr><tr><td style='text-align: center;'>MAX_FORCE</td><td style='text-align: center;'>9 \times 10^{-4}</td></tr><tr><td style='text-align: center;'>RMS_DR</td><td style='text-align: center;'>1.5 \times 10^{-3}</td></tr><tr><td style='text-align: center;'>RMS_FORCE</td><td style='text-align: center;'>6 \times 10^{-4}</td></tr><tr><td style='text-align: center;'>MAX_ITER</td><td style='text-align: center;'>100</td></tr><tr><td style='text-align: center;'>OPTIMIZER</td><td style='text-align: center;'>BFGS</td></tr><tr><td style='text-align: center;'>BFGS TRUST_RADIUS</td><td style='text-align: center;'>0.25</td></tr></table>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_310_974_913_1320.jpg" alt="Image" width="49%" /></div>


<div style="text-align: center;">Figure 10: Empirical cumulative distribution plots for the maximum, absolute value of the gradient across atoms after relaxation, regardless of convergence. The dashed line indicates the convergence threshold stated in Table 9.</div>


<div style="text-align: center;">Table 10: Expected values of the maximum gradient for each method from integrating the curves in Figure 10.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'>$ \mathbb{E}[|\nabla|_{\max}] $  ( $ \downarrow $ )</td></tr><tr><td style='text-align: center;'>DiffCSP</td><td style='text-align: center;'>0.000494</td></tr><tr><td style='text-align: center;'>CDVAE</td><td style='text-align: center;'>0.000235</td></tr><tr><td style='text-align: center;'>FlowMM</td><td style='text-align: center;'>0.000083</td></tr><tr><td style='text-align: center;'>DiffCSP++</td><td style='text-align: center;'>0.000072</td></tr><tr><td style='text-align: center;'>SymmCD (All groups)</td><td style='text-align: center;'>0.000057</td></tr><tr><td style='text-align: center;'>SymmCD (10 SGs)</td><td style='text-align: center;'>0.000038</td></tr></table>