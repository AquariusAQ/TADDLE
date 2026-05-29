## A TRANSFORMERS

Transformers are neural network models that have been shown to be very effective in a wide range of tasks, ranging from language modelling Vaswani et al. (2017), to computer vision Dosovitskiy et al. (2020) or approximating combinatorial problems such as the Travelling Salesman Problem Kool et al. (2018). Transformers are composed of two main blocks: self-attention and feed-forward layers. The self-attention layer is a function that takes as input the N node features  $ X \in R^{N \times d} $ , it linearly projects X into the query Q, key K and value V this is,  $ Q = XW_{Q} $ ,  $ K = XW_{K} $  and  $ V = XW_{V} $ , and computes self-attention as:

 $$ \mathrm{SelfAttention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\mathrm{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d}}\right)\mathbf{V}\in\mathbb{R}^{N\times d} $$ 

where d is the dimension of Q and  $ W_{Q}, W_{K}, W_{V} \in R^{d \times d} $  are trainable weight matrices. Generally, multi-head attention is used, this is, the self-attention is computed h times with different weight matrices, and the resulting outputs are concatenated. The multi-head attention layer is then followed by a feed-forward layer. The overall transformer architecture can be defined as:

 $$ \begin{aligned}\mathbf{X}^{\prime}&=\mathbf{X}+\mathbf{MHA}(\mathbf{X})\\ Transformer(\mathbf{X})&=FFN(\mathbf{X}^{\prime})=ReLU(\mathbf{X}^{\prime}\mathbf{W}_{1})\mathbf{W}_{2}\end{aligned} $$ 

where MHA is the multi-head attention layer, ReLU is the activation function, and  $ W_{1}, W_{2} \in R^{d \times d} $  are trainable weight matrices. In the above equation,  $ \mathrm{MHA}(\mathbf{X}, \mathbf{X}, \mathbf{X}) $  has been written as  $ \mathrm{MHA}(\mathbf{X}) $  for simplicity.

## B CHALLENGES

This section provides an insight into the existing challenges to identify redundancies in metabolic networks, and how these challenges have driven our architectural decisions.

### B.1 LONG-RANGE DEPENDENT

It is straightforward to check that the message passing framework is constrained when it comes to identifying redundancies on networks. To see this, consider the example network of Figure 8. In this network, reaction  $ r_{s} $  is a source reaction in the network, and reaction  $ r_{g} $  is the objective reaction. It is clear that there are two different paths from  $ r_{s} $  to  $ r_{g} $ , one path that goes through reaction  $ r_{1} $  and another path that goes through reaction  $ r_{2} $ . This makes reactions  $ r_{1} $  and  $ r_{2} $  not essential. Notice that between reaction  $ r_{1} $  and  $ r_{2} $  there are a total of  $ 2N+2 $  nodes. This means that, if we wanted to transfer information from  $ r_{1} $  to  $ r_{2} $  using message passing, we would need at least  $ 2N+2 $  steps (assuming undirected message passing). Ultimately, this means that, for reaction  $ r_{1} $  to acknowledge the existence of reaction  $ r_{2} $ , it would need to pass information through at least  $ 2N+2 $  steps. Therefore, it is clear that the number of steps required to identify redundancies in the network scales linearly with the number of nodes in the network, this is, given a network N, the worst case number of steps required is  $ O(D) $ , where D is the diameter of N.

Given this limitation of message passing, we need to look for a different mechanism for capturing the global information of the network. This shortcoming in message passing may be addressed by using positional encodings Dwivedi & Bresson (2020). Since positional encodings try to capture the global position of a node in the network, they may reduce the number of steps required to identify redundancies in the network. However, as shown in Section D, GNNs with positional encodings are not powerful enough to identify redundancies in the network.

<div style="text-align: center;"><img src="imgs/img_in_image_box_248_163_968_267.jpg" alt="Image" width="58%" /></div>


<div style="text-align: center;">(a) Example Petri net where reac- (b) Example Petri net where reac- (c) Undirected graph whose topol- tions  $ r_{1} $  and  $ r_{2} $  are essential reac- tion  $ r_{1} $  is not essential and reaction -ogy matches the one of Figures (a) and (b).</div>


<div style="text-align: center;">Figure 9: The directionality of the network determines the essentiality of the reactions. Both networks in Figures (a) and (b) share the same undirected topology, but the directionality of the edges is different. This makes reaction  $ r_{1} $  essential in Figure (a) and not essential in Figure (b).</div>


### B.2 DIRECTIONALITY

Given the wide application of graph learning methods to undirected graphs, one might be tempted to use undirected graphs to represent metabolic networks. Again, it is straightforward to show that the directionality of the network is a key property in the identification of redundancies and that ignoring it leads to a loss of information in the network. To see this, consider the Petri nets of Figures 9a and 9b, where reaction  $ r_{g} $  is the objective reaction. Notice that, given the direction of the reactions, reaction  $ r_{1} $  is essential in Figure 9a but not essential in Figure 9b, since in the last, alternative reactions exist that are able to supply the objective reaction. Notice that, if we ignore the directionality of the reactions, both networks share the same topology, depicted in Figure 9c.

In this graph, reaction  $ r_{2} $  is always involved in a path that leads to the objective reaction  $ r_{g} $ . Therefore, it is easy to infer that this reaction will always be essential, even if we ignore the directionality of the reactions. However, without directionality, it is impossible to infer whether reaction  $ r_{1} $  is essential or not. It is clear then, that any approach that dismisses the direction of the network will suffer a loss of key information.

## C SPECTRAL ENCODING

This section provides a detailed description of the positional encoding proposed in Section 4. In particular, we will definitions are provided for the combinatorial Laplacian Positional Encoder (LapPE) Kreuzer et al. (2021) and the Magnetic Laplacian Positional Encoder (MagLapPE) Geisler et al. (2023).

### C.1 LAPLACIAN POSITIONAL ENCODER

To encode the spectral information of the combinatorial Laplacian we use an approach similar to the one in Kreuzer et al. (2021). In this work, the first k eigenvectors with the minimum eigenvalues are selected. These eigenvectors are concatenated with their corresponding eigenvalues, passed through a linear layer, a self-attention layer and finally each embedding is pooled node-wise through a sum pooling. More formally, given the k eigenvectors corresponding to the node i, which we denote as  $ \phi_{:k,i} $ , and the k eigenvalues, which we denote as  $ \lambda_{:k} $ , the Laplacian Positional encoding for node i is defined as follows:

 $$ \begin{aligned}X_{i}&=\mathrm{FFN}\left(\begin{bmatrix}\phi_{:k,i}\\ \lambda_{:k}^{T}\end{bmatrix}\right)\\X_{i}^{\prime}&=\mathrm{MHA}(X_{i})\\X_{i}^{\prime\prime}&=\mathrm{FFN}\left(\sum_{j}^{k}X_{i,j}^{\prime}\right)\end{aligned} $$ 

where  $ X_{i}^{\prime\prime}\inR^{d} $  is the final embedding of the node i, with d being the dimension of the embedding. Normalization has been omitted from the above definition. To handle the sign invariance of the eigenvectors, the sign of  $ \phi_{:k,i} $  is randomly flipped, as it was done in Dwivedi & Bresson (2020).

### C.2 MAGNETIC LAPLACIAN POSITIONAL ENCODER

To encode the spectral information of the magnetic Laplacian, we will use the positional encoder proposed in Geisler et al. (2023). Similarly as in the Laplacian Positional Encoder, the first k eigenvectors with the minimum eigenvalues are selected. Let us denote,  $ \phi_{:k,i} $  as the k eigenvectors corresponding to the node i, and  $ \lambda_{:k} $  as the k eigenvalues. The Magnetic Laplacian Positional Encoder is defined as follows:

 $$ \begin{aligned}\boldsymbol{X}_{i}&=\mathbf{FFN}\left(\mathbf{Re}(\phi_{:k,i})\parallel\mathbf{Im}(\phi_{:k,i})\right)\parallel\boldsymbol{\lambda}_{:k}^{T}\\\boldsymbol{X}_{i}^{\prime}&=\mathbf{MHA}(\boldsymbol{X}_{i})+\boldsymbol{X}_{i}\\\mathbf{X}^{\prime \prime}&=\mathbf{FFN}\left(\bigparallel_{i=1}^{N}\boldsymbol{X}_{i}^{\prime}\right)\\\mathbf{X}^{\prime \prime \prime}&=\mathbf{MHA}(\mathbf{X}^{\prime \prime})+\mathbf{X}^{\prime \prime}\end{aligned} $$ 

where  $ X^{\prime\prime\prime} \in R^{N \times d} $  is the final matrix of embeddings of the N nodes, with d being the dimension of the embedding. Normalization and dropout have been omitted from the definition. Notice that, unlike the Laplacian Positional Encoder, this encoder includes skip connections after the self-attention layer, and performs self-attention before and after concatenating the embeddings of the nodes. Empirically, we did not observe any significant advantage of using self-attention. To handle the sign invariance of the eigenvectors, we use the same approach proposed in Geisler et al. (2023), this is, the sign of each eigenvector is determined such that the maximum real magnitude is positive. The authors also propose the use of SignNet Lim et al. (2022) to handle the sign invariance, however, empirically we did not observe any significant advantage using SignNet. The eigenvector normalization is performed as described in the original paper Geisler et al. (2023).

## D ABLATION STUDY

In this section, we perform an ablation study to show the importance of the different components of the proposed model. In particular, we will show the performance obtained without the use of message passing, without the use of transformers, and the use of directed GNNs.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>Params</td><td style='text-align: center;'>Epoch time (s)</td><td style='text-align: center;'>F1  $ \uparrow $</td></tr><tr><td style='text-align: center;'>Transformer + MagLapPE</td><td style='text-align: center;'>482305</td><td style='text-align: center;'>548.12</td><td style='text-align: center;'>0.4166</td></tr></table>

<div style="text-align: center;">Table 4: Ablation without message passing.</div>


First, let us consider the model without the use of message passing, this is, using only a transformer layer and the positional encodings. Here we used directly magnetic Laplacian positional encodings, since combinatorial Laplacian positional encodings are not able to capture directionality. The results are shown in Table 4. Here we can see that the model performance is not as good, with an F1 score of 0.4166. Based on this, it seems that the model struggles to capture local information relying only on the positional encodings.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>Params</td><td style='text-align: center;'>Epoch time (s)</td><td style='text-align: center;'>F1  $ \uparrow $</td></tr><tr><td style='text-align: center;'>SAT + LapPE + Gated</td><td style='text-align: center;'>374785</td><td style='text-align: center;'>642.65</td><td style='text-align: center;'>0.6003</td></tr><tr><td style='text-align: center;'>GPS + LapPE + Gated</td><td style='text-align: center;'>374785</td><td style='text-align: center;'>581.06</td><td style='text-align: center;'>0.6006</td></tr><tr><td style='text-align: center;'>SAT + MagLapPE + Gated</td><td style='text-align: center;'>345537</td><td style='text-align: center;'>607.56</td><td style='text-align: center;'>0.6022</td></tr><tr><td style='text-align: center;'>GPS + MagLapPE + Gated</td><td style='text-align: center;'>345537</td><td style='text-align: center;'>553.92</td><td style='text-align: center;'>0.5524</td></tr></table>

<div style="text-align: center;">Table 5: Ablation using non-directed message passing.</div>


Let us now consider the model without directed GNNs. Here we used both the SAT and the GPS architecture with an undirected gated GNN layer. We resort to the gated GNN layer since it shows a better performance than other GNN layers in our given dataset. The results are shown in Table 5. Here we can see that the model shows a better performance, reaching up to 0.6022 F1 score. Notice

that this outperforms some of the models presented in Table 1. However, it is still far from the best performance obtained on the task. Compared to the results in Table 4, there is a clear improvement in the score. This suggests that a great performance can be obtained in the task even without the directionality information.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>Params</td><td style='text-align: center;'>Epoch time (s)</td><td style='text-align: center;'>F1  $ \uparrow $</td></tr><tr><td style='text-align: center;'>LapPE + Dir-Gated</td><td style='text-align: center;'>662785</td><td style='text-align: center;'>167.62</td><td style='text-align: center;'>0.6445</td></tr><tr><td style='text-align: center;'>MagLapPE + Dir-Gated</td><td style='text-align: center;'>591169</td><td style='text-align: center;'>166.88</td><td style='text-align: center;'>0.6512</td></tr></table>

<div style="text-align: center;">Table 6: Ablation without transformer.</div>


Finally, let us consider the model without the use of transformers. Here we used directly a sequence of GNN layers without any transformer layer. For this task, we used a directed gated GNN layer. The results are shown in Table 6. Here we can see that the model shows a surprisingly good performance, reaching up to 0.6512 F1 score. This performance again outperforms some of the models presented in Table 1, however, it is still far from the best performance obtained on the task. This highlights the importance of the message passing layer and, in particular, the importance of the directed GNN layer. Finally, it shows that, without the global information provided by the transformer, the model is still not able to fully capture the redundancies in the network.

## E DATASET

To generate the dataset used in this work, we gathered 197 genome-scale models from the BiGG King et al. (2016) and Biomodels Malik-Sheriff et al. (2020) databases. The models correspond with a total of 113 different organisms, having multiple strains for some of them. The amount of metabolites of the models ranges from 7 to 2038 metabolites with an average of 772. The amount of reactions ranges from 6 to 4047 reactions with an average of 1115. The largest model contains a total of 5861 nodes. The number of essential reactions is on average 26% of the total reactions with a standard deviation of 0.17. If we compare the number of essential reactions with the total number of nodes, on average, 14% of the nodes are essential reactions with a standard deviation of 0.14. It can be seen that the amount of essential reactions is highly imbalanced, with an average of 218 essential reactions. To generate our dataset the models were split into 153 models for training, 21 models for validation and 21 models for testing. Since models of the same organism tend to share more similar network topologies, models of the same organism were not split into different sets. The split between training, validation and test sets was done manually to ensure that the models in all sets have a balanced number of reactions, metabolites, and essential reactions, as well as the maximum degree of the nodes and the degree of the objective reaction. The detailed list of models and splits is available as supplementary material.

Essential reactions computation To compute the essential reactions of each model, we used the COBRApy Ebrahim et al. (2013) framework. For all the models, we remove restrictions that enforce a non-null flux in the reactions (i.e. we set L = 0), this is, we assume a rich medium and we impose no constraint on growth production or any other metabolite. Notice that, the proposed method in this work does not limit the possibility of adding such restrictions for reactions. Future approaches could include reaction fluxes as additional features.

Data augmentation Since we are able to compute essential reactions of any model, we performed random modifications in the models as a way to augment the dataset. To generate a modified model, we performed randomly two types of modifications: (i) we randomly chose any reaction of the model and used it as the objective reaction and (ii) we took the biomass reaction and randomly added or removed between 1 and 10 reactants and between 1 and 10 products of the model. In the case of producing an infeasible problem, the generated model was discarded. Notice that both presented modifications alter the objective function and therefore the computed essential reactions. The resulting dataset after the data augmentation process contained a total of 18703 samples with 14419 train samples, 2142 validation samples and 2142 test samples, with 2190 nodes on average and an 8.8% of essential reactions on average. This dataset is particularly interesting as it contains a large number of graphs, which is a feature more common in graph-prediction tasks, but on average, it

also contains a large number of nodes for node-prediction. In addition, the node label depends on the global structure of the graph.

## F EXPERIMENTAL SETUP

In this section, we describe the experimental setup used to train and evaluate the models. In addition to the architectural decision described in Section 4, we also used jumping knowledge Xu et al. (2018b) with max aggregation and skip connections with GNNs (ommitted in Figure 5). In the case of GAT, we used 4 attention heads. Edge features were included in all GNN models except for GCN. All the models were trained with the AdamW optimizer Loshchilov & Hutter (2017).

The training was performed using NVIDIA A10 and NVIDIA Geforce RTX 3090 GPUs, both with 24 GB RAM. Given the size of the data samples, the maximum batch size that could accommodate the GPU memory was 3 samples. The number of epochs was set to a limit of 100 epochs with 10 warmup epochs. The training of a single graph transformer model took around 24 hours.

Hyperparameters Given the computational cost of training the models, we performed a limited hyperparameter search. Generally, we optimized each of the hyperparameters in a greedy fashion, this is, we fixed the rest of the hyperparameters and optimized one at a time. Table 7 shows the main hyperparameters used for training the models.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>Value</td></tr><tr><td rowspan="7">Base</td><td style='text-align: center;'>Layers</td><td style='text-align: center;'>6</td></tr><tr><td style='text-align: center;'>FNN layers</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Hidden dim</td><td style='text-align: center;'>64</td></tr><tr><td style='text-align: center;'>Heads</td><td style='text-align: center;'>4</td></tr><tr><td style='text-align: center;'>Dropout</td><td style='text-align: center;'>0.0</td></tr><tr><td style='text-align: center;'>GNN Aggr</td><td style='text-align: center;'>Sum</td></tr><tr><td style='text-align: center;'>Norm</td><td style='text-align: center;'>Layer Norm</td></tr><tr><td rowspan="2">Optimizer</td><td style='text-align: center;'>Learning rate</td><td style='text-align: center;'>0.0008</td></tr><tr><td style='text-align: center;'>Weight decay</td><td style='text-align: center;'>$ 1 \times 10^{-5} $</td></tr><tr><td style='text-align: center;'>Laplacian</td><td style='text-align: center;'>Freq.</td><td style='text-align: center;'>10 minimum eigv.</td></tr><tr><td rowspan="3">Encoder</td><td rowspan="2">Use attention</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>(layers = 1, heads = 2)</td></tr><tr><td style='text-align: center;'>FFN layers</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Magnetic</td><td style='text-align: center;'>Freq.</td><td style='text-align: center;'>10 minimum eigv.</td></tr><tr><td style='text-align: center;'>Laplacian</td><td style='text-align: center;'>Use attention</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>Encoder</td><td style='text-align: center;'>FFN layers</td><td style='text-align: center;'>2</td></tr></table>

<div style="text-align: center;">Table 7: Hyperparameters used for graph transformer models.</div>


For the ablation studies presented in Section D, in the case of the model without message passing, we used the hyperparameters in Table 8. In the case of graph transformers with undirected GNNs, we used the same hyperparameters as in Table 7. Finally, in the case of the model without transformers, we used the hyperparameters in Table 9. For the latter, we used 350 epochs instead of 100.

The FlowGAT model was trained using the same code and hyperparameters provided in the original paper Hasibi et al. (2024). The metabolic trasnformer was finetuned for the  $ E.coli $  model using the above hyperparameters, learning rate of 0.0001 and freezing the transformer layers, except for the two prediction heads.

Loss function Since we are training a binary classification model, we used binary cross-entropy as loss function. Through this work, we have dealt with the prediction of reaction essentiality in a bipartite graph. Clearly, the loss  $ \mathcal{L}(y,\hat{y}) $  was computed only for reaction nodes and not for metabolites nodes. This is:

 $$ \mathcal{L}(y,\hat{y})=-\sum_{i\in\mathcal{R}}y_{i}\log\hat{y}_{i}+(1-y_{i})\log(1-\hat{y}_{i}) $$ 


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>Value</td></tr><tr><td rowspan="6">Base</td><td style='text-align: center;'>Layers</td><td style='text-align: center;'>6</td></tr><tr><td style='text-align: center;'>FNN layers</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Hidden dim</td><td style='text-align: center;'>96</td></tr><tr><td style='text-align: center;'>Heads</td><td style='text-align: center;'>4</td></tr><tr><td style='text-align: center;'>Dropout</td><td style='text-align: center;'>0.0</td></tr><tr><td style='text-align: center;'>Norm</td><td style='text-align: center;'>Layer Norm</td></tr><tr><td rowspan="2">Optimizer</td><td style='text-align: center;'>Learning rate</td><td style='text-align: center;'>0.001</td></tr><tr><td style='text-align: center;'>Weight decay</td><td style='text-align: center;'>$ 1 \times 10^{-5} $</td></tr><tr><td style='text-align: center;'>Magnetic</td><td style='text-align: center;'>Freq.</td><td style='text-align: center;'>10 minimum eigv.</td></tr><tr><td style='text-align: center;'>Laplacian</td><td style='text-align: center;'>Use attention</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>Encoder</td><td style='text-align: center;'>FFN layers</td><td style='text-align: center;'>2</td></tr></table>

<div style="text-align: center;">Table 8: Hyperparameters used for transformer models.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>Value</td></tr><tr><td rowspan="5">Base</td><td style='text-align: center;'>Layers</td><td style='text-align: center;'>6</td></tr><tr><td style='text-align: center;'>Hidden dim</td><td style='text-align: center;'>96</td></tr><tr><td style='text-align: center;'>Dropout</td><td style='text-align: center;'>0.0</td></tr><tr><td style='text-align: center;'>GNN Aggr</td><td style='text-align: center;'>Sum</td></tr><tr><td style='text-align: center;'>Norm</td><td style='text-align: center;'>Layer Norm</td></tr><tr><td rowspan="2">Optimizer</td><td style='text-align: center;'>Learning rate</td><td style='text-align: center;'>0.0008</td></tr><tr><td style='text-align: center;'>Weight decay</td><td style='text-align: center;'>0.0</td></tr><tr><td style='text-align: center;'>Laplacian</td><td style='text-align: center;'>Freq.</td><td style='text-align: center;'>10 minimum eigv.</td></tr><tr><td rowspan="3">Encoder</td><td rowspan="2">Use attention</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>(layers = 1, heads = 2)</td></tr><tr><td style='text-align: center;'>FFN layers</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Magnetic</td><td style='text-align: center;'>Freq.</td><td style='text-align: center;'>10 minimum eigv.</td></tr><tr><td style='text-align: center;'>Laplacian</td><td style='text-align: center;'>Use attention</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>Encoder</td><td style='text-align: center;'>FFN layers</td><td style='text-align: center;'>2</td></tr></table>

<div style="text-align: center;">Table 9: Hyperparameters used for Dir-GNN models.</div>


where  $ y_{i} $  is the ground truth label for reaction i and  $ \hat{y}_{i} $  is the predicted label for reaction i. Since the dataset is highly imbalanced, we used the weighted version of the loss function, where the weight of the positive class is the inverse of the proportion of positive samples in the dataset.

## G METABOLIC NETWORK GRAPH CONSTRUCTION

In this Section, we explain how the metabolic network is transformed into a graph and how node features are extracted. The general pipeline for the transformation is shown in Figure 10. We will use this figure to exemplify the process and the design decisions taken. We start from the metabolic network shown on the left of the figure. This network is composed of 3 reactions and 2 metabolites. The first challenge that arises is the fact that certain metabolic reactions are considered reversible. This means that these reactions can be performed in both directions. We therefore need to model this fact. One naive solution is to simply link all input and output metabolites of a reaction with undirected edges. However, this solution removes the 2 sets of consumed and produced metabolites. Another solution is to label the edges of each side of the reaction (e.g., use a one-hot encoding on the edges of one side of the reaction [0, 1] and the opposite encoding on the other side [1, 0]). However, this unintentionally creates two separate sets of metabolites which might not be desirable. To solve this problem, we resorted to the approach used with LPPs when dealing with metabolic networks Ebrahim et al. (2013), and created two separate reactions for each reversible reaction. For instance, if we assume that reaction  $ r_{2} $  is reversible, we create two reactions with opposite direction  $ r_{2}^{\leftarrow} $  and  $ r_{2}^{\rightarrow} $ , as shown in the first step of the figure.

Once the reversible reactions are resolved, to compute the positional encodings we need to transform the bipartite graph into a unipartite graph. To achieve this, we just use a one-hot encoding to

<div style="text-align: center;"><img src="imgs/img_in_image_box_204_160_1223_441.jpg" alt="Image" width="83%" /></div>


<div style="text-align: center;">Figure 10: Metabolic network transformation. The Figure shows the steps undertaken to transform a metabolic network into a graph and how the spectral features are computed.</div>


differentiate between reactions and metabolites and directly treat the graph as unipartite. This is shown in the third graph of the figure. On this unipartite graph, we are now able to compute the eigenvalues and eigenvectors of the magnetic Laplacian as explained in the paper.

In the cases when it is desired to also compute the combinatorial Laplacian of the graph, we simply ignore the direction of the edges and treat the graph as undirected. This results in the fourth graph of the figure.

In addition to the positional encodings and the node features aforementioned, to provide more information to the model, we also include a flag in each node features indicating whether the node is the objective reaction, and another flag indicating whether the node is a source or sink reaction in the network.

With the procedure exposed until now, empirically it usually happened that the two reactions created from a reversible reaction had different labels. This is, if  $ r_{2} $  is reversible, then  $ r_{2}^{\leftarrow} $  and  $ r_{2}^{\rightarrow} $  could be classified differently. This is an issue since reactions  $ r_{2}^{\rightarrow} $  and  $ r_{2}^{\rightarrow} $  are the same reaction, just with different directions. To solve this problem, we decided to add an extra edge between the two reactions created from a reversible reaction. This helps to propagate the information during message passing and seems to solve the problem. This is shown in the fifth graph of the figure. To differentiate between the previous edges of the network, and the new edges created, we used a one-hot encoding and included it as edge features.

## H PERFORMANCE COMPARISON

Our proposed approach enables the use of GPUs to identify essential reactions in GEMs, which was not possible before. To show this, we used the model MODEL1011090001 from the Biomodels database Malik-Sheriff et al. (2020) which contains 3393 reactions and 2572 metabolites. The wall clock time, using the GPS + LapPE + Dir-Gated model described in Table 1, was of 0.09248 seconds with a GPU NVIDIA 3090 RTX with 24GB of memory. The wall clock time solving the FBA LPPs with GLPK solver was 6.7725 seconds with CPU Intel Core i5-9300H CPU @ 2.40GHz x 8.