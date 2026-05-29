## A EXPERIMENT DETAILS

### A.1 Task Description

Here we describe each task and how the experiments are performed in more detail; we also specify the exact model components which comprise the circuits we analyzed in the main paper. In this section, we use the convention of specifying attention heads in a model by the tuple (layer index, head index). For the IOI task, we add a third entry for output of the head at a specific sequence position, and follow their nomenclature for semantically labeling the sequence positions.

##### A.1.1 INDIRECT OBJECT IDENTIFICATION (IOI) (WANG ET AL., 2023)

The indirect object identification (IOI) task is to predict the indirect object in a sentence with two entities, such as identifying "Mary" in the sequence "When Mary and John went for a walk, John gave an apple to ...". The objective originally defined for this task, which we also use, is the difference between the predicted logit of the indirect object (IO) token and the subject (S) token. Wang et al. (2023) carefully design their experiment and dataset code so that mean ablation occurs using the mean activations over the corrupted "ABC dataset", which replaces the three name tokens in the task with random names. Likewise, when setting the decomposition of a node, we set the "relevant" component to the deviation from the mean activations on the ABC dataset. This task was performed on GPT2-small. GPT-2 correctly performs this task about 99 percent of the time, so we did not take special measures to account for those samples where it doesn't in our circuit analysis. We identify circuits using 25 IOI samples drawn from mixed templates, and mean ablation is conducted using the corrupted ABC dataset. Another set of 100 IOI samples are used in evaluation. Manual circuit of IOI compared against in the ROC AUC experiment is: [(2, 2), (4, 11), (0, 1), (3, 0), (0, 10), (5, 5), (6, 9), (5, 8), (5, 9), (7, 3), (7, 9), (8, 6), (8, 10), (10, 7), (11, 0), (9, 9), (9, 6), (10, 0), (9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)], which is from Figure 2 in Wang et al. (2023).

Details of Circuit Analysis For this task, in order to provide some intuitions about what CD-T calculates, we provide a number of heatmaps of the relevance scores at specific positions during various iterations of the circuit analysis. In this section, we don’t follow our automated circuit discovery algorithm exactly, but instead partially follow Wang et al. (2023)'s analysis to decide what sequence positions to search over and visualize.

The first iteration of the algorithm finds the Name Mover Heads: (9, 9, end), (10, 0, end), and (9, 6, end); the Negative Name Mover Heads: (10, 7, end), (11, 10, end); and some Backup Name Mover Heads: (10, 2, end), (10, 6, end), (10, 10, end), described by Wang et al. (2023)

Following Wang et al. (2023)'s analysis further, and deviating from the normal course of our circuit-finding algorithm, we compute the relevance of nodes to the Name Mover Heads on just the end

<div style="text-align: center;">Relevance of nodes to IOI task metric at last sequence position</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_343_184_843_580.jpg" alt="Image" width="40%" /></div>


position, and find what they named the "S-Inhibition Heads", at (8, 10, end), (7, 9, end), and (7, 3, end), though there are some other relevant-looking contenders:

<div style="text-align: center;">Relevance of nodes to output residuals of Name Mover Heads at last sequence position, normalized by mean rel per layer</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_386_733_877_1123.jpg" alt="Image" width="40%" /></div>


Continuing to follow their analysis, Wang et al. (2023)'s analysis further, we compute the relevance of nodes to the S-Inhibition Heads at the S2 position, and find our first minor disagreement with their process: though we find two of what they named the Induction Heads at (5, 5, S2), (5, 8, S2), (5, 9, S2) we don't find the one at (6, 9, S2) but instead find one at (5, 10, S2), and find a head their later analysis called a Duplicate Token Head, at (3, 0, S2).

Next we compute the relevance of nodes to the Induction Heads at the S2 position, and find that  $ (3, 0, S2) $  mostly drowns out the signal of the other Duplicate Token Heads:

Finally, we compute the relevance of nodes to the Induction Heads at the S1+1 position, and find  $ (4, 11, S1+1) $ , with the other Previous Token Head they found at  $ (2, 2, S1+1) $  a top contender, though there are other heads not accounted for in their analysis:

Overall, there is significant but not perfect agreement with the results and analysis of the IOI paper. We also attempted analysis by computing relevance to intermediate matrices (i.e., the key, query,

<div style="text-align: center;">Relevance of nodes to output residuals of S-inhibition heads at S2 sequence position, normalized</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_376_196_871_602.jpg" alt="Image" width="40%" /></div>


<div style="text-align: center;">(Duplicate Token Heads) Relevance of nodes to output residuals of Induction heads at S2 position, normalized for mean relevance among heads in layer</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_389_674_886_1067.jpg" alt="Image" width="40%" /></div>


value vectors) in the attention calculation, but found the plots to be qualitatively similar, possibly due to the fact that CD-T by design propagates relevances from these vectors to the attention head outputs as well.

Another fact of note is that the scales on these plots vary substantially. It would be desirable to find an interpretable normalization method to put these on the same scale with some intrinsic meaning, or otherwise explain the causes of this phenomenon.

##### A.1.2 GREATER-THAN (HANNA ET AL., 2023)

The Greater than task is to predict the last two digits in an incomplete sentence following the template “The [noun] lasted from the year XYY to the year XX”. And we expect the model to assign higher probability to years greater than YY. The objective originally defined for this task, which we also use, is the sum of probabilities assigned to tokens corresponding to greater years, minus the sum of probabilities assigned to tokens corresponding to lesser years. (Some probability is assigned to tokens which don’t correspond to numbers at all.) For our “mean-ablation”, we simply take the

<div style="text-align: center;"><img src="imgs/img_in_chart_box_317_170_884_609.jpg" alt="Image" width="46%" /></div>


mean over the activations over 100 negative datapoints (impossible completions, with the ending year preceding the starting century), and as above, when setting the decomposition at a source node, define the relevant component to be the deviation from the mean activation over this distribution. This task was performed on GPT2-small. GPT-2 correctly performs this task about 99 percent of the time, so we did not take special measures to account for those samples where it doesn't in our circuit analysis. We identify circuits using a random sample of 100 datapoints provided by Hanna et al. (2023), and mean ablation is conducted using the negative impossible completion samples. Another set of 100 samples are used in evaluation.

It should be noted that our result and the results in Hanna et al. (2023) are not directly comparable, since they also attempt to investigate the influence of MLPs and their resulting circuit removes most of the MLPs in GPT-2. For the comparisons found in the main paper, we have compared our result (with all MLPs) to the circuit which is obtained by taking all the attention heads named in Hanna et al. (2023) (also with all MLPs), which is a circuit distinct from the one found in their work.

Manual circuit of Greater-than compared against in the ROC AUC experiment is:  $ [(5, 1), (5, 5), (6, 1), (6, 9), (7, 10), (8, 8), (8, 11), (9, 1)] $ .

Independently of the comparison, it remains true that keeping the relatively small proportion of attention heads in our circuit results in recovering almost all of GPT-2's capability on this task; see below.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Task-specific metric</td><td style='text-align: center;'>Correct guess rate</td></tr><tr><td style='text-align: center;'>Full model</td><td style='text-align: center;'>0.817</td><td style='text-align: center;'>0.992</td></tr><tr><td style='text-align: center;'>All attention heads ablated</td><td style='text-align: center;'>-2.095</td><td style='text-align: center;'>0</td></tr><tr><td style='text-align: center;'>Their circuit</td><td style='text-align: center;'>0.768</td><td style='text-align: center;'>0.989</td></tr><tr><td style='text-align: center;'>Their circuit (most MLPs ablated)</td><td style='text-align: center;'>0.727</td><td style='text-align: center;'>N/A</td></tr><tr><td style='text-align: center;'>Our circuit</td><td style='text-align: center;'>0.761</td><td style='text-align: center;'>0.981</td></tr></table>

#### A.1.3 DOSTRING (HEIMERSHEIM & JANIAK, 2023)

The goal of the Docstring task is to predict the next variable name in a Python docstring. For example, given a function with variable names LOAD, SIZE, FILES, and LAST, the task is to predict the word after one :PARAM. According to docstring conventions, this should be the variable name in the function definition which follows the most recent variable name to appear after :PARAM. The objective originally defined for this task is the logit assigned to the correct variable name, minus the max logit assigned to all other variable name tokens found in the function signature. For

our “mean-ablation”, we use their random_random dataset which randomize both the variable names in the function definition and in the docstring of prompts, and correspondingly, when setting the decomposition at a source node, define the relevant component to be the deviation from the mean activation over this distribution. We identify circuits using a 100 datapoints sampling for the dataset provided by Heimersheim & Janiak (2023), and mean ablation is conducted using the corrupted random_random dataset. Another set of 100 samples are used in evaluation. This task was performed on a 4-layer attention-only transformer trained on natural language and Python code (attn-only-41) released with the TransformerLens library for the express purpose of facilitating mechanistic interpretability research. Another complication is that the toy model only guesses the correct token between 60 and 65 percent of the time. To account for this, we perform our circuit analysis and evaluation on the subset of input examples for which the model performs the task correctly.

Our circuit consists of the nodes  $ (3, 0) $ ,  $ (3, 6) $ ,  $ (1, 4) $ ,  $ (0, 5) $ ,  $ (0, 0) $ ,  $ (1, 0) $ ,  $ (1, 4) $ ,  $ (0, 1) $ ,  $ (2, 3) $ ,  $ (1, 2) $ . Heimersheim & Janiak (2023) find the circuit  $ (0, 2) $ ,  $ (0, 4) $ ,  $ (0, 5) $ ,  $ (1, 2) $ ,  $ (1, 4) $ ,  $ (2, 0) $ ,  $ (3, 0) $ ,  $ (3, 6) $  with their initial analysis, and heuristically observe that three heads help to obtain the “augmented circuit”  $ (0, 2) $ ,  $ (0, 4) $ ,  $ (0, 5) $ ,  $ (1, 2) $ ,  $ (1, 4) $ ,  $ (2, 0) $ ,  $ (3, 0) $ ,  $ (3, 6) $ ,  $ (1, 0) $ ,  $ (0, 1) $ ,  $ (2, 3) $ .

Manual circuit of Docstring compared against in the ROC AUC experiment is:  $ [(0, 2), (0, 4), (0, 5), (1, 2), (1, 4), (2, 0), (3, 0), (3, 6)] $ .


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Task-specific metric</td><td style='text-align: center;'>Correct guess rate</td></tr><tr><td style='text-align: center;'>Full model</td><td style='text-align: center;'>3.661</td><td style='text-align: center;'>1.0</td></tr><tr><td style='text-align: center;'>All attention heads ablated</td><td style='text-align: center;'>-2.095</td><td style='text-align: center;'>0</td></tr><tr><td style='text-align: center;'>Their circuit</td><td style='text-align: center;'>2.884</td><td style='text-align: center;'>0.54</td></tr><tr><td style='text-align: center;'>Augmented circuit</td><td style='text-align: center;'>3.524</td><td style='text-align: center;'>0.66</td></tr><tr><td style='text-align: center;'>Our circuit</td><td style='text-align: center;'>3.235</td><td style='text-align: center;'>0.57</td></tr></table>

## B ALGORITHM DETAILS

### B.1 HEURISTICS

In addition to the greedy pruning presented in Algorithm 1 as a refinement step, here we describe other heuristic elements to the algorithm.

• The threshold for determining which set S of highest-contributing nodes may be varied: it is possible to pick the top N nodes or fraction of nodes, or to automatically detect outliers. Empirically, we find that the distribution of node contributions varies quite severely, so finding a heuristic which works in all cases is actually an object of future work. In this paper, we set the threshold by varying the percentile of top nodes to extract, in the range of  $ [90, 99] $  to obtain the ROC AUC in section 4.2.1.

• Empirically we find, for a fixed target node, the relevance scores of source nodes in different layers to this target node may have different expected magnitudes, due to the numerical effects of propagation through the network. To account for this, we normalize the relevance scores by dividing by the average magnitude across scores found in a given layer.

• To avoid the risk of propagation equations becoming numerically unstable after a large number of iterations, especially if the relevant and irrelevant constituents differ in sign at a specific index, we set the value of one of rel/irrel at this position to 0 and the other to the sum of the two terms.

### B.2 COMPLEXITY ANALYSIS

To provide more clarity, the computational complexity of the algorithm satisfy the following properties:

• Each decomposition (of a set of target nodes with respect to a set of source nodes) requires cost in FLOPs similar to one forward pass of the model (and often less, since values prior

to the source nodes can be cached and values after the target nodes do not need to be calculated).

• The core of the algorithm is a loop, where in each iteration, we search over all nodes which can potentially have high relevance to the target nodes. (This means that heuristically excluding some nodes from the search can potentially significantly decrease cost in FLOPs, and cost in FLOPs increases linearly with respect to the granularity with which we separate the nodes in the model.)

- The memory footprint of a single decomposition, including the forward pass, is a small (less than 3) constant multiple of the cost of a forward pass; the only added costs are to keep track of the relevant and irrelevant constituent tensors separately, as well as bookkeeping of components of the target decomposition metric.

- The cost (in FLOPs or memory) of performing the analysis on a set of input examples is linear in the number of input examples, since the same set of computations needs to be done with respect to each example.