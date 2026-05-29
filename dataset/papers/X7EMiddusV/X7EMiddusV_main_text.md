# DIRECTED GRAPH TRANSFORMERS MEET METABOLIC NETWORKS

Anonymous authors

Paper under double-blind review

## ABSTRACT

Technical advances in sequencing have allowed the reconstruction of genome-scale metabolic models (GEMs) for a wide range of microorganisms. These models have been particularly useful for the prediction of essential genes and reactions, which are potential targets for antimicrobial therapies. However, current methods for essentiality prediction are computationally limited and are not able to accommodate the increasingly available data. Motivated by the success of data-driven approaches in other domains, this work introduces the metabolic transformer, a model designed for holistic identification of essential reactions in genome-scale models, entirely trained on synthetic knock-out data. It is demonstrated that the problem of essential reaction prediction can be theoretically formulated as the identification of redundant nodes in directed bipartite graphs. This reveals the limitations of message-passing schemes and motivates the development of a novel graph transformer architecture specifically tailored for metabolic networks. The proposed architecture is capable of addressing the essential reaction identification problem by capturing both the directionality and global structure of metabolic networks. To demonstrate the effectiveness of our approach, we composed a large-scale dataset of genome-scale models reconstructed from real microorganisms. $ ^{1} $ 

## 1 INTRODUCTION

In recent years, the emergence of high-throughput technologies allowed the integration of transcriptomic data of multiple pathogens into large biological datasets. This integration paved the way for the reconstruction of microorganism metabolism which directly led to the possibility of modelling these systems computationally Ric (2020).

Metabolism is the set of basic life processes that take place in the cell. All the metabolic chemical reactions that occur in a cell form a metabolic network. Genome-Scale models (GEM) are structured biochemical, genetic, and genomic databases for an organism, that aim to cover the whole metabolic network of a cell. As of 2019, more than 6.000 GEMs have been reconstructed for organisms including bacteria, archaea and eukaryotes and over 140.000 automated reconstructions are available from over 2.600 organisms Büchel et al. (2013); Gu et al. (2019); Monk et al. (2017).

GEMs have been particularly useful for phenotype simulation and prediction. These models have shown great accuracy in predicting the growth rate of microorganisms along with the flux carried out for each reaction of metabolism Orth et al. (2010). This has been particularly interesting for the identification of essential reactions and genes. Essential reactions are those that are required for the growth of the organism Henry et al. (2006). Consequently, when these reactions are knocked out the growth in the model is null, i.e., the organism dies. The genes that encode these reactions are appealing therapeutic targets.

Although the results on essentiality prediction are promising, current methods can be computationally demanding and their application to uncertain data can be challenging. Improving the accuracy of predictions is not trivial and requires an increasing complexity on the models that sometimes involves an exponential combinatorial cost or requires the estimation of missing data O'Brien et al. (2013); Salvy & Hatzimanikatis (2020). In addition, recent advances in metabolism profiling technology Judge et al. (2019) will produce large amounts of data that could be hard to accommodate into current

<div style="text-align: center;"><img src="imgs/img_in_image_box_211_158_1020_403.jpg" alt="Image" width="66%" /></div>


<div style="text-align: center;">Figure 1: The metabolic transformer. a) The model is trained entirely from synthetic knock-out data obtained from metabolic network reconstructions from different microorganisms. b) Once trained, the model enables the computation of essential reactions in models where synthetic knock-out computations were previously infeasible. c) The model trained on synthetic data can serve as a backbone for finetuning with real data.</div>


modelling algorithms. In this context, there is a growing belief that data-driven methods (i.e. deep learning) have the potential to improve metabolic modelling Zampieri et al. (2019).

In this work, we propose a supervised learning approach, fully trained on synthetic knock-out data, for the holistic prediction of essential reactions in genome-scale models using graph neural networks and transformers models. Graph Neural Networks (GNN), first proposed in Scarselli et al. (2008), have become the reference algorithm for learning on graph-structured data. It is well known, however, that GNNs suffer from over-squashing and fail to model long-range relationships in graphs, which is a fundamental requirement for metabolic modelling. Motivated by the success of transformers models in natural language processing and computer vision, many works have proposed the use of transformers for graph-structured data Min et al. (2022); Rampasek et al. (2022); Bo et al. (2023). One of the early applications of transformers to graphs was presented in Dwivedi & Bresson (2020), where the authors proposed a generalization of transformers for graphs by applying the attention mechanism to the neighbourhood of each node. Later examples such as Kreuzer et al. (2021) and Kim et al. (2022) propose approaches where attention is applied to the whole graph. Other works have also proposed the integration transformers with GNNs to capture both the local and global information of the graph Rong et al. (2020); Rampasek et al. (2022).

This work proposes the metabolic transformer, a graph transformer model that combines directed message passing with positional encodings for essential reaction prediction in genome-scale models. Previous models have been proposed before for essentiality computation on GEMs Hasibi et al. (2024). These methods however rely only on message-passing schemes or are narrowed down to a specific dataset, which makes it application to other tasks or microorganisms limited. In this work, a model is proposed that is able to generalize essentiality reaction prediction across different microorganisms by pre-training on synthetic knock-out data. It is shown that the model can serve for effective transfer learning to real metabolic data, in this case, gene essentiality obtained experimentally in the lab. This is in contrast with existing LP methods that cannot generalize outside the modelled problem. To the best of our knowledge, this is the first model to successfully demonstrate transfer learning on metabolic data.

In summary, our proposed approach has the following properties:

• For inference, the proposed method only takes as input the topology of the metabolic network. This is in contrast with previous methods that require precomputing synthetic knock-out data for inference Freischem et al. (2022); Hasibi et al. (2024).

- The method predicts the essentiality of all reactions considering the whole network at once. This property is a direct consequence of the graph transformer architecture, which can take the features of all nodes as input. This overcomes the need for GNN-based approaches that require pre-computing global features on the network Yang et al. (2022).

• Given the high throughput of deep learning models, we are able to compute predictions for all reactions from only one forward pass of the model. This is an improvement compared

with the traditional synthetic knock-out simulation, which requires the computation of the growth rate for each reaction individually Oyelade et al. (2018).

• Metabolic data available for many microorganisms is scarce or limited. Additionally, metabolic modelling is not limited to essentiality prediction, but also includes a wide range of other modelling tasks such as flux prediction, control, or drug target identification. By training a model on synthetic data from many diverse microorganisms, we developed a model that can serve as backbone for effective transfer learning on different organisms or tasks, where data can be limited. We thus believe that this is a significant step towards the creation of foundational models of metabolism.

Currently, the majority of GEMs publicly available are built through automated reconstruction pipelines Gu et al. (2019). As a consequence, most of these models lack manual curation and, hence, traditional methods for knock-out simulations based on linear programming cannot be applied. In this context, since the proposed method only depends on the topology of the network, and does not require any additional data to be curated, it has the potential to enable the prediction of essential reactions in a wide range of models for which current methods are unable to produce predictions.

In addition to the proposed methods, we are releasing a carefully curated dataset of 23k GEMs from more than 100 different microorganisms. The dataset consists of large-scale graphs with an average of 2k nodes. This is the first dataset of GEMs and, given its dimensions, we believe it is of interest for the graph-learning community, making it one of the largest datasets in terms of number of nodes Hu et al. (2021).

Although this work is particularly focused on genome-scale metabolic models, the architecture and methods proposed in this work are not limited and can be applied to other topics traditionally modelled with directed bipartite graphs, such as control and distributed systems Murata (1989); Heiner et al. (2008), chemical reaction networks Wen et al. (2023) or retrosynthesis Chen et al. (2020).

The rest of the paper is organised as follows. In Section 2 preliminary definitions related to constraint-based models are introduced. Section 3 introduces the background for graph transformers used in this work. Section 4 describes the proposed architecture, and Section 5 describes the results obtained in the created dataset. Finally, Section 6 concludes the paper.

## 2 PRELIMINARY DEFINITIONS

In this Section, we introduce the preliminary definitions that are used throughout the paper. In particular, we introduce the formal definition of constraint-based models, flux balance analysis, and essential reactions.

### 2.1 CONSTRAINT-BASED MODELS

A constraint-based model Varma & Palsson (1994); Orth et al. (2011) is a tuple  $ \{R, M, S, L, U\} $  where R is a set of reactions, M is a set of metabolites,  $ S \in R^{|M| \times |R|} $  is the stoichiometric matrix, and  $ L, U \in R^{|R|} $  are lower and upper flux bounds of the reactions.

 $$ 2H_{2}+O_{2}\rightarrow2H_{2}O $$ 

<div style="text-align: center;"><img src="imgs/img_in_image_box_408_1169_552_1268.jpg" alt="Image" width="11%" /></div>


(a) Chemical reaction.

<div style="text-align: center;">(b) Petri net.</div>


Figure 2: A constraint-based model consisting only of the reaction in (a), can be modelled by the Petri net of (b).

All the reactions of the model are associated with a set of reactant metabolites and a set of product metabolites. For example, the reaction  $ r{:}A \rightarrow 2B $  has a reactant metabolite A, and a product metabolite B, with stoichiometric weights 1 and 2 respectively, i.e. reaction r consumes one molecule of type A and produces two molecules of type B. The stoichiometric matrix S accounts for all the stoichiometric weights of the reactions, i.e.  $ S[m,r] $  is the stoichiometric weight of metabolite  $ m \in M $  for reaction  $ r \in R $ .

Constraint-based models are inherently bipartite directed graphs and thus they can be represented graphically as Petri nets Murata (1989); Heiner et al. (2008), where places, drawn as circles, model metabolites, and transitions, drawn as squares, model reactions. The presence of an arc from a





place(transition) to a transition(place) means that the place is a reactant(product) of the reaction modelled by the transition. The weights of the arcs of the Petri net account for the stoichiometry of the constraint-based model.

Example 2.1. The Petri net in Figure 2b represents a constraint-based model consisting only of one reaction, the well-known reaction of equation 2a.

### 2.2 FLUX BALANCE ANALYSIS

Flux Balance Analysis (FBA) Orth et al. (2010) is a mathematical procedure for the estimation of steady-state fluxes in constraint-based models. FBA is generally used to predict the maximum growth rate of an organism. Let  $ v \in R^{|R|} $  be the vector of fluxes of reactions and v[r] denote the flux of reaction r. At a steady state, it holds that  $ S \cdot v = 0 $ , where S is the stoichiometric matrix. Let  $ r_{g} $  be the reaction that models growth (or biomass production). Without loss of generality, it will be assumed that  $ L[r_{g}] \geq 0 $ . A theoretical optimum for the growth rate can be obtained by the linear programming problem (LPP) for FBA:

 $$ \begin{aligned}\max v[r_{g}]\\ st.\ \mathcal{S}\cdot v=0\\L\leq v\leq U\end{aligned} $$ 

where the maximum  $ v[r_{g}] $  obtained by the above LPP (1) is the maximum growth rate.

### 2.3 ESSENTIAL REACTIONS

A reaction is said to be essential if it is required by the organism to grow. In other words, the deletion of an essential reaction implies null growth. Consequently, these reactions have the potential to cause the death of the modelled organism Oyelade et al. (2018).

Definition 2.1. Oyelade et al. (2018) A reaction  $ r \in R $  is an essential reaction if the solution of the following LPP:

 $$ \begin{aligned}\max v[r_{g}]\\ st.\ \mathcal{S}\cdot v=0\\L\leq v\leq U\\v[r]=0\end{aligned} $$ 

is equal to 0 or the LPP is infeasible. In other words, a reaction  $ r \in R $  is essential if the maximum growth rate that the model can achieve is 0 (i.e.  $ \max v[r_{g}] = 0 $ ) when the reaction r is removed (i.e. when  $ v[r] = 0 $ ).

The set of essential reactions, can be computed straightforwardly by solving equation 2 for each  $ r \in R $ .

Essential reactions can also be interpreted as those reactions that are required for the production of biomass, and that lack any alternative pathway with similar functionality. To exemplify how the topology of the network conditions the essentiality of reactions, here we provide three samples of non-essential reactions:

Example 2.2. Let us consider the constraint-based model in Figure 3a, where reaction  $ r_{g} $  is the biomass reaction. It can be seen that metabolite  $ m_{2} $  is required for the production of biomass. This metabolite can be both produced through reactions  $ r_{1} $  and  $ r_{2} $ . If reaction  $ r_{1} $  is removed, then reaction  $ r_{2} $  can still produce metabolite  $ m_{2} $ . Therefore, reaction  $ r_{1} $  is not essential. Similarly, reaction  $ r_{2} $  is not essential either.

Example 2.3. Let us now consider the constraint-based model in Figure 3b. Metabolite  $ m_{4} $  is a product of the biomass reaction. Once this metabolite is produced it requires to be evacuated from the network. This can be done through reaction  $ r_{7} $  or reactions  $ r_{5} $  and  $ r_{6} $ . If reaction  $ r_{7} $  is removed, then reactions  $ r_{5} $  and  $ r_{6} $  can still evacuate metabolite  $ m_{4} $ . Therefore, reaction  $ r_{7} $  is not essential. Similarly, reactions  $ r_{5} $  and  $ r_{6} $  are not essential either.

Example 2.4. Let us now consider the constraint-based model in Figure 3c. Here, metabolite  $ m_{3} $  can be diverted through the biomass reaction or through an alternative path (reactions  $ r_{8} $ ,  $ r_{9} $ ) that does not contribute to biomass production. If one of these two reactions is removed, then the other reactions can still produce metabolite  $ m_{3} $ . Therefore, reactions  $ r_{8} $  and  $ r_{9} $  are not essential.

<div style="text-align: center;"><img src="imgs/img_in_image_box_224_175_443_388.jpg" alt="Image" width="17%" /></div>


<div style="text-align: center;">(a) Redundancy in the production of biomass.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_499_172_721_376.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(b) Redundancy in the evacuation of biomass products.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_766_176_996_373.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">(c) Biomass reachability.</div>


<div style="text-align: center;">Figure 3: Three sample conditions of reactions essentiality.</div>


## 3 DIRECTED GRAPH TRANSFORMERS

In this section, we introduce all the concepts that compose a directed graph transformer model. We start by introducing the concept of graph neural networks, and directed graph neural networks and we end by introducing positional encodings for graphs. A definition of transformers models is provided in Appendix A.

### 3.1 GRAPH NEURAL NETWORKS

Graph Neural Networks (GNN) Scarselli et al. (2008); Li et al. (2015); Kipf & Welling (2016) are machine learning models that learn on data that is accompanied by a graph structure. GNNs are composed of layers of message passing networks. In each layer, the embedding vector  $ h_{i} $  of node i is computed from the aggregation of the embeddings of their neighbour nodes  $ \mathcal{N}(i) $  of the previous layer. The initial embedding vector is usually the input feature vector that each node is given. As described in You et al. (2020), a general k-th GNN layer can be defined formally as:

 $$ h_{v}^{(k+1)}=\mathbf{A G G}\left(\left\{\mathbf{A C T}\left(\mathbf{W}^{(k)}h_{u}^{(k)}+b^{(k)}\right),u\in\mathcal{N}(v)\right\}\right) $$ 

where  $ h_{v}^{(k)} $  is the k-th layer embedding of node v,  $ \mathbf{W}^{(k)} $  and  $ b^{(k)} $  are the trainable weight matrix and bias respectively, ACT is an activation function and AGG is a commutative aggregation function such as maximisation, summation or mean. The different variations proposed to GNNs have shown to be very effective in learning on graphs data Hamilton et al. (2017); Veličković et al. (2017); Li et al. (2015). However, GNNs are known to suffer from limited expressivity Xu et al. (2018a), over-smoothing, this is, converging to a single solution after many layers Li et al. (2018), and oversquashing, this is, losing information from long-range nodes due to bottlenecks in the graph Alon & Yahav (2020). Their application to modelling long-range dependencies is thus considered limited.

### 3.2 DIRECTED GRAPH NEURAL NETWORKS

Most current GNN applications assume that graphs are undirected. However, the application of GNNs on directed graphs is not straightforward. This usually requires either transforming the directed graph into an undirected graph or propagating messages only over incoming (or outgoing) edges, which could lead to information loss. These approaches have had a great performance on benchmarks that have been historically homophilic, this is, where neighbours nodes tend to share the same label. However, the case of directed and heterophilic graphs has been less explored. To overcome this, in Rossi et al. (2023) the authors introduce the framework of Directed Graph Neural Networks (Dir-GNN). Dir-GNNs are generic GNNs that can be applied to directed graphs by aggregating messages from both the incoming and outgoing edges of each node. This extension of GNNs showed a great improvement in heterophilic datasets and seemed to outperform previous approaches for directed graphs. For this reason, we will use Dir-GNNs as the base model for our models. Formally,

<div style="text-align: center;"><img src="imgs/img_in_image_box_292_170_915_386.jpg" alt="Image" width="50%" /></div>


<div style="text-align: center;">(a) Second eigenvector  $ v_{1} $  of the graph from Figure 3.(b) First element of the eigenvectors of the magnetic Network direction has been omitted to represent that Laplacian of the graph from Figure 3. The size of the directionality is missed in the combinatorial Laplacian. nodes is proportional to the real part of the eigenvector and the color is proportional to the imaginary part.</div>


<div style="text-align: center;">Figure 4: Unipartite representation of the graph in Figure 3 showing the spectral information of the combinatorial and magnetic Laplacians. Here values have been normalized for visualization purposes.</div>


the k-th layer can be defined as:

 $$ \begin{aligned}&m_{u,\leftarrow}^{(k+1)}=\mathbf{A G G}\left(\left\{\mathbf{A C T}\left(\mathbf{W}_{\leftarrow}^{(k)}h_{v}^{(k)}+b_{\leftarrow}^{(k)}\right),(v,u)\in\mathcal{E}\right\}\right)\\&m_{u,\rightarrow}^{(k+1)}=\mathbf{A G G}\left(\left\{\mathbf{A C T}\left(\mathbf{W}_{\rightarrow}^{(k)}h_{v}^{(k)}+b_{\rightarrow}^{(k)}\right),(u,v)\in\mathcal{E}\right\}\right)\\&h_{u}^{(k+1)}=\mathbf{C O M}^{(k)}\left(\left\{m_{u,\leftarrow}^{(k+1)},m_{u,\rightarrow}^{(k+1)}\right\}\right)\\ \end{aligned} $$ 

where  $ \mathbf{W}_{\leftarrow}^{(k)} $  and  $ \mathbf{W}_{\rightarrow}^{(k)} $  are the learnable weight matrices of the k-th layer,  $ b_{\leftarrow}^{(k)} $  and  $ b_{\rightarrow}^{(k)} $  are the learnable bias vectors of the k-th layer, ACT is an activation function, AGG is a commutative aggregation function, and  $ \operatorname{COM}^{(k)} $  is a function that combines the incoming and outgoing messages. In this work  $ \operatorname{COM}^{(k)} $  is implemented as:  $ \alpha^{(k)}m_{u,\leftarrow}^{(k+1)} + (1 - \alpha^{(k)})m_{u,\rightarrow}^{(k+1)} $  with  $ \alpha^{(k)} \in [0,1] $  being a learnable parameter of the k-th layer.

### 3.3 POSITIONAL ENCODINGS

The attention mechanism of transformers is known to be permutation invariant. Hence, in order to capture positional information, Positional Encodings (PEs) are used. In Vaswani et al. (2017), the authors propose to add a sinusoidal PE to capture the position of each word in the sentence. However, the generalization of the sinusoidal signal to graph-structured data is not trivial. The most common approach for computing positional encodings in graph data is to use the Laplacian matrix of the graph. It is argued that the eigenvectors of the Laplacian matrix generalise the sinusoidal encodings to graphs Dwivedi & Bresson (2020).

#### 3.3.1 EIGENVECTORS OF LAPLACIAN

Let  $ G^{(U)} = (V, E) $ , be an undirected graph without self loops, where V are the nodes and E the edges. The adjacency matrix  $ \mathbf{A}^{(U)} \in \mathbb{R}^{N \times N} $  is defined as  $ \mathbf{A}_{ij}^{(U)} = 1 $  if  $ (i, j) \in E $  and  $ \mathbf{A}_{ij}^{(U)} = 0 $  otherwise, with  $ N = |V| $ . Given the degree of each node  $ d_i = \sum_{j=1}^{N} \mathbf{A}_{ij}^{(U)} $ , the degree matrix is defined as  $ \mathbf{D} = \text{diag}(d_1, \ldots, d_N) $ . The (combinatorial) Laplacian and symmetrized Laplacian are defined as:

 $$ \mathbf{L}_{u}=\mathbf{D}-\mathbf{A}^{(U)}\in\mathbb{R}^{N\times N}\quad\mathbf{L}_{s}=\mathbf{D}^{-1/2}\mathbf{L}\mathbf{D}^{-1/2}\in\mathbb{R}^{N\times N} $$ 

Since, the  $ L_{u} $  of an undirected graph is real symmetric,  $ L_{u} $  can be decomposed as  $ L_{u} = U\LambdaU^{T} $  where U is the orthonormal matrix of eigenvectors  $ U = v_{0}, \ldots, v_{N-1} $  and  $ \Lambda $  is the diagonal matrix of eigenvalues  $ \Lambda = \lambda_{0}, \ldots, \lambda_{N-1} $ . Given the ordered eigenvalues  $ \lambda_{0} \leq \cdots \leq \lambda_{N-1} $ , and the corresponding eigenvectors  $ v_{0}, \ldots, v_{N-1} $ , it always holds that  $ \lambda_{0} = 0 $  and  $ v_{0} = 1 $ .

In Beaini et al. (2021), it was shown that using only the eigenvector  $ v_{1} $  is enough to distinguish graphs not distinguishable by the 1-WL test. Additionally, in Kreuzer et al. (2021), the authors suggest that transformers with the full set of eigenvectors are universal approximators of the graph isomorphism problem.

Example 3.1. Let us consider the constraint-based model of Figure 3. The graph in Figure 4a shows the equivalent unipartite graph where the direction of the network has been omitted. In this graph, the intensity of the color of the nodes is proportional to the value of the second eigenvector of the combinatorial Laplacian  $ v_{1} $ . Generally, it is known that the second eigenvector is able to represent closeness in the network, this is, nodes that are close to each other in the network will have similar values and distant nodes will have a larger difference in their values. In this sense, the combinatorial Laplacian is able to provide information of the global positioning of a node in a network. Recall that directionality is missed in the combinatorial Laplacian. However, the second eigenvector is able to represent some redundancies of the network. For instance, in this graph, it happens that nodes  $ r_{6} $  and  $ r_{7} $ , which belong to a redundant path, have similar values in the second eigenvector (i.e.,  $ v_{1,r6} \approx v_{1,r7} $ ). The same case happens with nodes  $ r_{1} $  and  $ r_{2} $  (i.e.,  $ v_{1,r1} \approx v_{1,r2} $ ).

#### 3.3.2 MAGNETIC LAPLACIAN

As it was pointed out in Furutani et al. (2020) and Geisler et al. (2023), the combinatorial Laplacian fails to distinguish directionality on graphs. To address this issue, the authors propose the use of the eigenvectors of the magnetic Laplacian, where in addition to the connectivity of the network, the directionality of the network is encoded in the complex plane. Let  $ G^{(D)} = (V, E) $  be a directed graph with adjacency matrix  $ \mathbf{A}^{(D)} $ , where  $ \mathbf{A}_{ij}^{(D)} = 1 $  if  $ (i, j) \in E $  and  $ \mathbf{A}_{ij}^{(D)} = 0 $  otherwise. Additionally, let  $ \mathbf{A}^{(U)} $  be the adjacency matrix of the undirected version of  $ G^{(D)} $ . The magnetic Laplacian is defined as:

 $$ \mathbf{L}_{m}=\mathbf{D}-\mathbf{\Gamma  _{m}}\odot\mathbf{A}^{(U)}\in\mathbb{C}^{N\times N} $$ 

where  $ \odot $  is the Hadamard product and  $ \Gamma_{m}\inC^{N\times N} $  is a Hermitian matrix whose  $ (i,j) $  is equal to:  $ e^{i2\pi q(\mathbf{A}_{ij}^{(D)}-\mathbf{A}_{ji}^{(D)})} $ , where  $ q\in[0,1) $  is a rotation parameter or potential. Notice that when q=0, the magnetic Laplacian is equal to the combinatorial Laplacian. In addition, notice that  $ e^{i2\pi q(\mathbf{A}_{ij}^{(D)}-\mathbf{A}_{ji}^{(D)})} $  encodes the direction of the edge  $ (i,j) $ . For undirected edges, i.e.  $ \mathbf{A}_{ij}^{(D)}=\mathbf{A}_{ji}^{(D)}=1 $ , the value is equal to 1. For directed edges, i.e.  $ \mathbf{A}_{ij}^{(D)}=1 $ ,  $ \mathbf{A}_{ji}^{(D)}=0 $  and  $ \mathbf{A}_{ij}^{(D)}=0 $ ,  $ \mathbf{A}_{ji}^{(D)}=1 $ , the value is equal to  $ e^{i2\pi q} $  and  $ e^{-i2\pi q} $  respectively. From here onwards, we will assume the eigenvalues of the magnetic Laplacian to be ordered:  $ 0=\lambda_{0}\leq\lambda_{1}\leq\cdots\leq\lambda_{N-1} $ .

Example 3.2. Let us consider again the constraint-based model of Figure 3. The graph in Figure 4b shows the equivalent unipartite graph. In this graph, the size of the nodes is proportional to the real part of the first eigenvector of the magnetic Laplacian, denoted  $ \operatorname{Re}(\phi_{0}^{(q)}) $ , and the color of the nodes is proportional to the imaginary part  $ \operatorname{Im}(\phi_{0}^{(q)}) $ . As it is shown in Furutani et al. (2020); Geisler et al. (2023), and as it can be seen in the graph, the magnetic Laplacian provides a topological sorting of the network in the complex plane. This is particularly useful in the case of metabolic networks, as it is able to capture redundancies in the network while also considering the directionality of the network.

For instance, notice that, in this graph, nodes  $ r_{1} $  and  $ r_{2} $ , which are both part of a redundant path, have the same color and size (i.e.  $ \mathrm{Re}(\phi_{0,r_{1}}^{(q)}) \approx \mathrm{Re}(\phi_{0,r_{2}}^{(q)}) $  and  $ \mathrm{Im}(\phi_{0,r_{1}}^{(q)}) \approx \mathrm{Im}(\phi_{0,r_{2}}^{(q)}) $ ). The same case happens with nodes  $ r_{3} $ ,  $ r_{4} $  and  $ r_{5} $ . Notice also that, unlike the combinatorial Laplacian, the magnetic Laplacian is able to capture the fact that paths  $ m_{3} \rightarrow r_{g} $  and  $ m_{3} \rightarrow r_{8} $  are parallel.

#### 3.3.3 NOTATION

To denote the set of eigenvalues and eigenvectors of the magnetic Laplacian, we will use the following notation:  $ \mathbf{\Lambda}^{(q)} = \lambda_{0}^{(q)}, \ldots, \lambda_{N-1}^{(q)} $  and  $ \mathbf{\Phi}^{(q)} = \phi_{0}^{(q)}, \ldots, \phi_{N-1}^{(q)} $ , where  $ \phi_{i}^{(q)} $  is the i-th eigenvector of the magnetic Laplacian for rotation parameter q. We use  $ \phi_{i,j}^{(q)} $  to denote the entry of the i-th eigenvector corresponding with the j-th node.

We will use the fact that the magnetic Laplacian with q = 0 is equivalent to the combinatorial Laplacian to denote the eigenvalues and eigenvectors of the combinatorial Laplacian as  $ \mathbf{\Lambda}^{(0)} = \mathbf{\Lambda}^{(0)} = \mathbf{\Lambda}^{(0)} $ .

 $ \lambda_{0}^{(0)},\ldots,\lambda_{N-1}^{(0)} $  and  $ \Phi^{(0)}=\phi_{0}^{(0)},\ldots,\phi_{N-1}^{(0)} $  respectively. Similarly, we denote  $ \phi_{i,j}^{(0)} $  as the entry of the i-th eigenvector of the combinatorial Laplacian corresponding with the j-th node.

## 4 ARCHITECTURE

In this section, we present the organization of the proposed metabolic transformer. The architecture is a directed graph transformer, which is composed of a positional encoding layer (that may include directional information) and a graph transformer layer with directed message passing.

This architecture is motivated by the fact that message passing alone is not enough to identify the sufficient conditions that were introduced in Section 3. In Appendix B.1 an example is provided where the asymptotic complexity of identifying redundancies in a network is of  $ O(D) $  message passing steps, with D being the diameter of the network. Another significant property of this architecture is that it is able to capture directionality in the network. Again, it is easy to see that

<div style="text-align: center;"><img src="imgs/img_in_image_box_539_312_1000_610.jpg" alt="Image" width="37%" /></div>


<div style="text-align: center;">(a) Directed graph transformer (b) Directed graph transformer based on the SAT architecture based on the GPS architecture Chen et al. (2022). Rampášek et al. (2022).</div>


<div style="text-align: center;">Figure 5: Two proposed architectures for the directed graph transformer.</div>


identifying redundancies in a network requires directional information. An example of this is provided in Appendix B.2.

### 4.1 SPECTRAL ENCODING

As mentioned before, bipartite metabolic networks will be treated as non-bipartite networks. This will allow us to compute spectral positional encodings as described before. This procedure is detailed in Appendix G. In Section 3 we described how the eigenvalues and eigenvectors of the combinatorial Laplacian and the magnetic Laplacian capture information about the global position of nodes in a graph. To be able to learn useful representations from this information, we will use the Laplacian Positional Encoder (LapPE), introduced in Kreuzer et al. (2021) and the Magnetic Laplacian Positional Encoder (MagLapPE), introduced in Geisler et al. (2023). A detailed description of these positional encoders is provided in Appendix C.

### 4.2 METABOLIC TRANSFORMER

Now that the positional encoding layers are introduced, we are ready to introduce the metabolic transformer layer. The approach taken is a combination of a directed message passing layer to capture the local structure of the network, and a self-attention layer over the whole set of nodes to capture global relations in the network. This combination of message passing with global attention has been presented before Chen et al. (2022); Rampášek et al. (2022), with promising results in many graph-related tasks. These works however, were limited to undirected graphs. In this work, two different flavours of graph transformers have been extended to include directed message passing.

1. The first approach (SAT), follows the work of Chen et al. (2022) and can be seen in Figure 5a. This approach is intended to capture structural similarities through the attention mechanism. The layer achieves this by using the GNN output as the query and the key of the attention mechanism, and the graph embeddings as the value. The output of the attention mechanism is used then as the input of the next layer. Formally, the l-th layer of the graph transformer

is defined as follows:

 $$ \begin{aligned}\mathbf{X}_{G}^{(l+1)}&=\mathbf{Dir-GNN}(\mathbf{X}^{(l)},\mathbf{E}^{(l)},\mathbf{A}^{(D)})\\\mathbf{X}_{A}^{(l+1)}&=\mathbf{MHA}(\mathbf{X}_{G}^{(l+1)},\mathbf{X}_{G}^{(l+1)},X^{(l+1)})\\\mathbf{X}^{(l+1)}&=\mathbf{Norm}(\mathbf{FFN}(\mathbf{X}_{A}^{(l+1)}+\mathbf{X}^{(l)}))\end{aligned} $$ 

where  $ \mathbf{X}^{(l+1)}\in\mathbb{R}^{N\times d} $  is the output of the l-th layer,  $ \mathbf{A}^{(D)} $  is the adjacency matrix of the graph,  $ \mathbf{E}^{(0)} $  are the initial edge features, and  $ \mathbf{X}^{(0)} $  are the sum of the initial graph features and positional encoder features.

2. The second approach (GPS), presented in Rampášek et al. (2022), is depicted in Figure 5b. This approach applies simultaneously a GNN layer, that captures the local information of the network, and a self-attention layer over the whole set of nodes embeddings. The output of both layers is added and the output of the layer is passed as input to the next layer. Formally, the l-th layer of the graph transformer is defined as follows:

 $$ \begin{aligned}\mathbf{X}_{G}^{(l+1)}&=\mathbf{Dir}-\mathbf{GNN}(\mathbf{X}^{(l)},\mathbf{E}^{(l)},\mathbf{A}^{(D)})\\\mathbf{X}_{A}^{(l+1)}&=\mathbf{MHA}(\mathbf{X}^{(l)})\\\mathbf{X}^{(l+1)}&=\mathbf{Norm}(\mathbf{FFN}(\mathbf{X}_{G}^{(l+1)}+\mathbf{X}_{A}^{(l+1)}+\mathbf{X}^{(l)}))\end{aligned} $$ 

where  $ \mathbf{X}^{(l+1)}\in\mathbb{R}^{N\times d} $  is the output of the l-th layer,  $ \mathbf{A}^{(D)} $  is the adjacency matrix of the graph,  $ \mathbf{E}^{(0)} $  are the initial edge features, and  $ \mathbf{X}^{(0)} $  are the sum of the initial graph features and positional encoder features.

## 5 RESULTS

This section reports the results obtained in a dataset of 23k genome-scale models with an average of 2k node. Since no previous dataset exists for GEM, we are releasing the curated dataset along with this publication. The details of the dataset curation can be found in Section E. The results presented here include those obtained with the proposed graph transformer models with the different positional encoders. We are particularly interested in studying the effect of the different positional encoders and different GNN layers on the overall performance. To do so, we compare the results obtained with baseline GNN models, which include: Graph Convolutional Networks (GCN) Kipf &


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>Params</td><td style='text-align: center;'>Epoch time (s)</td><td style='text-align: center;'>F1  $ \uparrow $</td></tr><tr><td style='text-align: center;'>SAT+LapPE+Dir-GCN</td><td style='text-align: center;'>298393</td><td style='text-align: center;'>619.71</td><td style='text-align: center;'>0.6622</td></tr><tr><td style='text-align: center;'>SAT+LapPE+Dir-GAT</td><td style='text-align: center;'>349849</td><td style='text-align: center;'>809.96</td><td style='text-align: center;'>0.6328</td></tr><tr><td style='text-align: center;'>SAT+LapPE+Dir-GINE</td><td style='text-align: center;'>398209</td><td style='text-align: center;'>757.61</td><td style='text-align: center;'>0.5651</td></tr><tr><td style='text-align: center;'>SAT+LapPE+Dir-Gated</td><td style='text-align: center;'>498049</td><td style='text-align: center;'>824.91</td><td style='text-align: center;'>0.6628</td></tr><tr><td style='text-align: center;'>SAT+MagLapPE+Dir-GCN</td><td style='text-align: center;'>269145</td><td style='text-align: center;'>614.65</td><td style='text-align: center;'>0.6370</td></tr><tr><td style='text-align: center;'>SAT+MagLapPE+Dir-GAT</td><td style='text-align: center;'>320601</td><td style='text-align: center;'>813.43</td><td style='text-align: center;'>0.6786</td></tr><tr><td style='text-align: center;'>SAT+MagLapPE+Dir-GINE</td><td style='text-align: center;'>368961</td><td style='text-align: center;'>735.50</td><td style='text-align: center;'>0.5877</td></tr><tr><td style='text-align: center;'>SAT+MagLapPE+Dir-Gated</td><td style='text-align: center;'>468801</td><td style='text-align: center;'>639.33</td><td style='text-align: center;'>0.6974</td></tr><tr><td style='text-align: center;'>GPS+LapPE+Dir-GCN</td><td style='text-align: center;'>298393</td><td style='text-align: center;'>557.39</td><td style='text-align: center;'>0.6659</td></tr><tr><td style='text-align: center;'>GPS+LapPE+Dir-GAT</td><td style='text-align: center;'>349849</td><td style='text-align: center;'>646.62</td><td style='text-align: center;'>0.6718</td></tr><tr><td style='text-align: center;'>GPS+LapPE+Dir-GINE</td><td style='text-align: center;'>398209</td><td style='text-align: center;'>587.89</td><td style='text-align: center;'>0.6195</td></tr><tr><td style='text-align: center;'>GPS+LapPE+Dir-Gated</td><td style='text-align: center;'>498049</td><td style='text-align: center;'>940.53</td><td style='text-align: center;'>0.7145</td></tr><tr><td style='text-align: center;'>GPS+MagLapPE+Dir-GCN</td><td style='text-align: center;'>269145</td><td style='text-align: center;'>546.68</td><td style='text-align: center;'>0.6589</td></tr><tr><td style='text-align: center;'>GPS+MagLapPE+Dir-GAT</td><td style='text-align: center;'>320601</td><td style='text-align: center;'>712.57</td><td style='text-align: center;'>0.6708</td></tr><tr><td style='text-align: center;'>GPS+MagLapPE+Dir-GINE</td><td style='text-align: center;'>368961</td><td style='text-align: center;'>735.50</td><td style='text-align: center;'>0.5984</td></tr><tr><td style='text-align: center;'>GPS+MagLapPE+Dir-Gated</td><td style='text-align: center;'>468801</td><td style='text-align: center;'>657.18</td><td style='text-align: center;'>0.6841</td></tr></table>

<div style="text-align: center;">Table 1: Results of the different models on the test set.</div>


<div style="text-align: center;">Welling (2016), Graph Attention Networks (GAT) Veličković et al. (2017), Graph Isomorphism Networks (GINE) Xu et al. (2018a), and Gated Graph Neural Networks (Gated) Li et al. (2015). For each of the baseline models, we used the directed version of the model, as described in Section 4. Details on the hyperparameters and experimental setup can be found in Appendix F.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>F1 (+)  $ \uparrow $</td><td style='text-align: center;'>ROC AUC (+)  $ \uparrow $</td><td style='text-align: center;'>F1 (-)  $ \uparrow $</td><td style='text-align: center;'>ROC AUC (-)  $ \uparrow $</td></tr><tr><td style='text-align: center;'>FlowGAT Hasibi et al. (2024)</td><td style='text-align: center;'>0.85  $ \pm $  0.033</td><td style='text-align: center;'>0.495  $ \pm $  0.053</td><td style='text-align: center;'>0.02  $ \pm $  0.04</td><td style='text-align: center;'>0.525  $ \pm $  0.075</td></tr><tr><td style='text-align: center;'>Metabolic Transformer (ours)</td><td style='text-align: center;'>0.845  $ \pm $  0.027</td><td style='text-align: center;'>0.857  $ \pm $  0.022</td><td style='text-align: center;'>0.69  $ \pm $  0.026</td><td style='text-align: center;'>0.849  $ \pm $  0.017</td></tr></table>

<div style="text-align: center;">Table 2: Performance comparison in synthetic reaction essentiality prediction on  $ E.coli $  model iML1515 Monk et al. (2017). Here (+) indicates essentiality as positive class, while (-) indicates the opposite problem, this is, prediction of non-essentiality.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>F1 (+)  $ \uparrow $</td><td style='text-align: center;'>ROC AUC (+)  $ \uparrow $</td><td style='text-align: center;'>F1 (-)  $ \uparrow $</td><td style='text-align: center;'>ROC AUC (-)  $ \uparrow $</td></tr><tr><td style='text-align: center;'>FlowGAT Hasibi et al. (2024)</td><td style='text-align: center;'>0.86  $ \pm $  0.015</td><td style='text-align: center;'>0.56  $ \pm $  0.121</td><td style='text-align: center;'>0.161  $ \pm $  0.099</td><td style='text-align: center;'>0.587  $ \pm $  0.134</td></tr><tr><td style='text-align: center;'>Metabolic Transformer (ours)</td><td style='text-align: center;'>0.68  $ \pm $  0.031</td><td style='text-align: center;'>0.845  $ \pm $  0.03</td><td style='text-align: center;'>0.624  $ \pm $  0.093</td><td style='text-align: center;'>0.797  $ \pm $  0.083</td></tr></table>

<div style="text-align: center;">Table 3: Performance comparison in real gene essentiality prediction on  $ E.coli $  model iML1515 Monk et al. (2017). Here (+) indicates essentiality as positive class, while (-) indicates the opposite problem, this is, prediction of non-essentiality.</div>


As it is explained in Appendix E, the generated dataset is highly imbalanced, with an average of 8.8% of positive samples. To account for this, we report the F1 score of the models to compare their performance. From the results shown in Table 1, we can see that there is a clear dependence of the results on the message passing scheme used. Overall, the results suggest that message passing layers that aggregate the messages in a non-linear way (Dir-GAT, Dir-Gated) perform better than those that weight each message contribution the same (Dir-GCN, Dir-GINE). This suggests that the solution suffers from the well-known over-squashing problem Di Giovanni et al. (2023). Regarding the graph transformer architecture, generally, GPS-based solutions seem to perform slightly better than SAT-based solutions. If we compare now the results obtained with the different positional encoders, surprisingly, there is no clear advantage of using one over the other. It seems that the MagLapPE performs better with the SAT-based solutions, while the LapPE performs better with the GPS-based solutions. The best results are obtained with the GPS + LapPE + Dir-Gated model. Recall that LapPE does not capture the directional information of the graph. Despite this, LapPE is equally competitive than MagLapPE and outperforms MagLapPE in the case mentioned. This might suggest that (i) direction information is not as important for the task at hand or (ii) the information captured with the message passing is enough to capture the directional information of the graph. These results also suggest that, for this task, the use of the message passing scheme seems to be more important than the positional encoding scheme.

In Appendix D we provide an ablation study of the different components of the proposed architecture. A performance comparison between GPU and CPU inference for essential reactions computation is presented in Appendix H



<div style="text-align: center;"><img src="imgs/img_in_chart_box_612_770_812_976.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_818_770_1024_975.jpg" alt="Image" width="16%" /></div>


In order to evaluate the performance of the metabolic transformer against the existing state-of-the-art, we compared the results obtained with the FlowGAT model Hasibi et al. (2024). As metabolic transformer, here we used the best-performing model according to Table 1, this is, GPS+LapPE+Dir-Gated. Table 2 shows the results obtained in synthetic knock-out essential reaction prediction, using the iML1515 model of Escherichia coli Monk et al. (2017). As mentioned previously in this work, since FlowGAT relies in locality information, the capability to model essential reactions is limited, which explains the difference in performance. Table 3 shows the results obtained in real gene essentiality prediction using the same model. A finetuning was performed with the metabolic transformer to adapt the model to the task at hand. The results show that FlowGAT shows a higher F1 score, while performing poorly in the ROC AUC metric. In order to clarify this divergence in scores, confusion matrices from both problems are shown in Figures 7 and 6. In this Figure it can be seen that FlowGAT tends to classify all reactions as positive, while almost no discrimination is done with negative classes. Since the

<div style="text-align: center;">(a) FlowGAT Hasibi et al. (2024)</div>


<div style="text-align: center;">(b) Metabolic Transformer (ours)</div>


<div style="text-align: center;">Figure 6: Confusion matrix for synthetic reaction essentiality prediction on  $ E. $ coli model iML1515 Monk et al. (2017), obtained from best F1-score.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_612_1121_817_1327.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_819_1122_1024_1327.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">(a) FlowGAT Hasibi et al. (b) Metabolic Transformer (2024) (ours)</div>


<div style="text-align: center;">Figure 7: Confusion matrix for real gene essentiality prediction on  $ E.\ coli $  model iML1515 Monk et al. (2017), obtained from best F1-score.</div>


positive class here is the majority class, the F1 score obtained is large, while the ROC AUC shows that discriminative power of FlowGAT is considerably lower than the metabolic transformer introduced in this work. We note that the evaluation dataset used consists of a small number of samples with a high presence of positive samples, which explains the fluctuation in the ROC AUC metric. Note also that, while FlowGAT is a specialized model narrowed down for this particular tasks, our approach is a general model finetuned for gene essentiality prediction.

## 6 CONCLUSIONS

This work proposes the metabolic trasnformers, a novel approach for the prediction of essential reactions in genome-scale models by using directed graph transformers, entirely trained with synthetic knock-out data. The proposed approach is based on the use of directed GNNs as the base model to capture local information, and the use of positional encodings based on the combinatorial Laplacian and the magnetic Laplacian. To compare the performance of the proposed approach, we built a dataset of genome-scale models from the metabolism of different microorganisms from public databases. Additionally, we compared the performance of the proposed approach with existing methods for essential reactions and genes prediction, and we showed that the proposed approach achieves state-of-the-art performance in both tasks. The study on the architectural choices show that the performance is highly dependent on the message passing scheme used. In particular, the best results are obtained with the use of gated GNNs with a GPS architecture and Laplacian-based positional encodings. The results also show no clear advantage of using magnetic Laplacian-based positional encodings over Laplacian-based positional encodings, when combined with directed message passing. This work, however, is fundamentally limited by the scalability of the model, both from a data and memory perspective. Transformers are well known to have a  $ O(N^{2}) $  memory complexity. In our case, we used graphs from 113 microorganisms with an average of 2190 nodes, with the largest one having 5861 nodes. Currently, the largest available genome-scale model is the human model Recon3D Brunk et al. (2018) with 17683 nodes, which poses a challenge to the scalability of the model. Possible solutions to this issue include using more efficient attention mechanisms (Choromanski et al. (2020)).

## 7 ETHICS STATEMENT

The contribution of this work is to provide a novel approach to identify essential reactions in GEMs. This has the potential to facilitate the identification of essential reactions in a broad range of automatically reconstructed GEMs, a task previously deemed infeasible or requiring extensive manual curation of models. Current approaches that identify essential reactions in GEMs are aimed at contributing to the development of new antibiotics or cancer treatments, among others. Therefore we do not expect any ethical or societal issues to arise from this work.

## 8 REPRODUCIBILITY STATEMENT

Both the code and datasets required for the reproducibility of the results reported will be made available under GNU-GPLv3 license upon acceptance of the paper. Additionally, the steps taken to generate the dataset and hyperparameters used can be found in the Appendix F, E and G.