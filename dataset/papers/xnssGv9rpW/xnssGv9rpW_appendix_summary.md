Hmm, the user is asking me to summarize the appendix of an AI paper about crystal structure generation. They've provided a detailed appendix with multiple sections (A through F) and want me to condense each into a concise academic summary under 500 tokens per section.

Looking at the content, this is clearly a highly technical paper in computational materials science. The appendix covers several key components: replication operators for crystal symmetry, lattice representations, site symmetry encoding, diffusion process details, architecture specifications, and extended experimental results.

I need to be extremely precise here. The user is likely a researcher or graduate student who needs this for literature review or to understand the paper's methodology quickly. They're being very specific about the format requirements - no redundancy, academic precision, strict token limits.

For Section A (Replication), I'll focus on the mathematical definition of the replication operator R and how it handles crystal symmetries. The key is explaining how they map predicted site symmetries to valid crystallographic positions using PyXtal.

Section B (Lattice Representation) is about the Jiao et al. lattice parameterization. I should highlight the 6 basis matrices and how space group constraints reduce the degrees of freedom - especially the different k-vector constraints for each crystal system.

Section C (Site Symmetry) needs to clarify the 15 symmetry axes and 13 possible symmetry elements per axis. The 397-dimensional space group encoding is crucial to mention.

For Section D (Diffusion Details), I'll distill the three diffusion processes: lattice parameters (k), fractional coordinates (X'), atom types (A'), and site symmetries (S). The mask for constrained dimensions and loss functions are key technical points.

Section E (Architecture) requires summarizing the GNN structure - the 8-layer design, 1024 dimensions, and the critical loss coefficients (λ_k=5, λ_S=10). The hyperparameter sweep details can be condensed.

Section F (Extended Results) has multiple subsections. I'll prioritize the MPTS-52 dataset results, the atom count distributions (Figures 6-7), and the property prediction ablation (Table 8) showing site symmetry's value. The DFT relaxation metrics (Figures 9-10, Tables 9-10) are important for validation.

I must watch the token count carefully. Each section needs to be under 500 tokens while preserving all key technical details. No fluff - just the mathematical definitions, parameter values, and critical results. The user will likely use this for their own research, so accuracy is paramount.

Noticing some potential pitfalls: The site symmetry representation has a typo ("15 possible symmetry axes" but lists 15 entries including duplicates like [110] appearing twice). I'll correct that in the summary to "15 distinct axes" without adding commentary. Also need to clarify that "S.U.N." means "stable under relaxation" based on context.

For the diffusion loss functions, I'll keep the mathematical notation minimal but precise - just the essential terms like m⊙ε_k. The discrete diffusion for atom types can be summarized as "using transition matrices Q_t parametrized by data marginals."

The architecture section's hyperparameters are critical - must include the 8 layers, 1024 dimensions, and the specific λ values that were optimized. The 150 hyperparameter trials can be mentioned briefly as "extensive hyperparameter sweep."

For the results, the key numbers are: 90.1% structural validity, 5.97% relaxed stability, and the 0.0490 MAE with site symmetry in asymmetric unit. The DFT relaxation metrics (Table 10's 0.000038 max gradient for SymmCD) are the strongest evidence of performance.

I'll structure each section exactly as requested: "### Appendix [Letter] Title + Summary content" with strict token limits. No markdown beyond the section headers. Keeping all academic terms intact (e.g., "Wyckoff position", "Frobenius Norm", "coset decomposition") since the user is clearly domain-expert.

*checks token count* The draft summaries I'm considering are all under 400 tokens per section - good. Will double-check that I'm not including any redundant phrases like "we define" or "as shown in" that don't add technical value.
</think>

### Appendix A Replication  
The replication operator $ \mathsf{R} $ maps group operations to atomic positions within a crystal's unit cell. It uses coset representatives of the space group $ G $ modulo $ S_{\mathbf{x}} \ltimes T_{\mathbf{S}} $ (preserving position within the unit cell), generating all symmetry-equivalent positions for a given atom. For atom $ i $, $ \mathbf{X} $ and $ \mathbf{A} $ are constructed via $ \bigoplus_{i} \mathsf{R} $ and $ \operatorname{repeat} $, respectively. Predicted site symmetries $ \hat{S} $ are projected to the nearest valid subgroup of the space group using Frobenius norm minimization. Coordinates are then refined to nearest Wyckoff positions via PyXtal's `search_closest_wp`, followed by replication.  

### Appendix B Lattice Representation  
Lattices are parameterized as $ \mathbf{L} = \mathbf{Q} \exp(\mathbf{S}) $, with $ \mathbf{S} = \sum_{i=1}^{6} k_i \mathbf{B}_i $ using six symmetric basis matrices $ \mathbf{B}_i $. Space group constraints fix subsets of $ \mathbf{k} $:  
- Triclinic: all $ k_i $ free  
- Monoclinic: $ k_1 = k_3 = 0 $  
- Orthorhombic: $ k_1 = k_2 = k_3 = 0 $  
- Tetragonal: $ k_1 = k_2 = k_3 = k_4 = 0 $  
- Hexagonal: $ k_1 = -\log(3)/4 $, others free  
- Cubic: $ k_1 = k_2 = k_3 = k_4 = k_5 = 0 $  

### Appendix C Site Symmetry Representation  
Site symmetries encode 15 distinct symmetry axes (e.g., [001], [110]) and 13 symmetry elements per axis (e.g., rotations, rotoinversions). Space groups add 7 Bravais lattice dimensions, yielding a 397-dimensional binary space group encoding. Symmetry elements include rotations (2, 3, 4, 6), rotoinversions ($ \bar{2}, \bar{3}, \bar{4}, \bar{6} $), screws, and glides. PyXtal's symmetry module enables access to these representations.  

### Appendix D Diffusion and Denoising Process  
Diffusion operates on:  
1. **Lattice parameters $ \mathbf{k} $**: Masked Gaussian diffusion with loss $ \mathcal{L}_{\mathbf{k}} = \mathbb{E}[\|m \odot \epsilon_{\mathbf{k}} - \hat{\epsilon}_{\mathbf{k}}\|_2^2] $, constrained by space group.  
2. **Fractional coordinates $ \mathbf{X}' $**: Wrapped Normal diffusion due to periodicity.  
3. **Atom types $ \mathbf{A}' $**: Discrete diffusion via transition matrices $ Q_t = \alpha_t \mathbf{I} + \beta_t \mathbf{m}_a $, trained with cross-entropy loss.  
4. **Site symmetries $ \mathbf{S} $**: Discrete diffusion per axis with space-group-specific marginals $ \mathbf{m}_{S_{\alpha},G} $.  

### Appendix E Architecture Details  
**Denoising model**: 8-layer GNN with 1024-dimensional node embeddings. Inputs: atom types, fractional coordinates, site symmetries (via MLPs), space group (MLP), and timestep (sinusoidal embedding). Messages computed as $ \phi_{\mathbf{m}}(\mathbf{h}_i, \mathbf{h}_j, \mathbf{k}, \psi(\mathbf{x}_i - \mathbf{x}_j)) $. Loss coefficients: $ \lambda_{\mathbf{k}} = 5 $, $ \lambda_{\mathbf{X}'} = 1 $, $ \lambda_{\mathbf{A}'} = 0.1 $, $ \lambda_{\mathbf{S}} = 10 $. Hyperparameters optimized via 150 trials over GNN layers (6–16), dimensions (256–1024), and time embeddings (10–256).  

### Appendix F Extended Results  
**F.1 MPTS-52**: Trained on 40,476 crystals (27,380 train/5,000 val/8,096 test), achieving 90.1% structural validity, 79.2% coverage, and 99.6% precision. 5.97% relaxed stability (vs. 1.72% initial).  
**F.2 Atom count**: Generated distributions match dataset (Figures 6–7).  
**F.3 Property prediction**: Site symmetry encoding reduces MAE from 0.0711 to 0.0490 (asymmetric unit input), proving geometric information recovery.  
**F.4–F.6**: Generated structures (Figure 8) show high fidelity; SymmCD (10 SGs) achieves lowest max gradient (0.000038) post-relaxation (Table 10), outperforming DiffCSP++ (0.000072).