# GENERALIZATION V.S. MEMORIZATION: TRACING LANGUAGE MODELS' CAPABILITIES BACK TO PRETRAINING DATA

Xinyi Wang $ ^{1*} $ , Antonis Antoniades $ ^{1*} $ , Yanai Elazar $ ^{2,3} $ , Alfonso Amayuelas $ ^{1} $ , Alon Albalak $ ^{4} $ , Kexun Zhang $ ^{5} $ , William Yang Wang $ ^{1} $ 

 $ ^{1} $ University of California, Santa Barbara,  $ ^{2} $ Allen Institute for AI,



 $ ^{3} $ University of Washington,  $ ^{4} $  SynthLabs,  $ ^{5} $ Carnegie Mellon University {xinyi_wang, antonis}@ucsb.edu, william@cs.ucsb.edu



## ABSTRACT

The impressive capabilities of large language models (LLMs) have sparked debate over whether these models genuinely generalize to unseen tasks or predominantly rely on memorizing vast amounts of pretraining data. To explore this issue, we introduce an extended concept of memorization, distributional memorization, which measures the correlation between the LLM output probabilities and the pretraining data frequency. To effectively capture task-specific pretraining data frequency, we propose a novel task-gram language model, which is built by counting the co-occurrence of semantically related n-gram pairs from task inputs and outputs in the pretraining corpus. Using the Pythia models trained on the Pile dataset, we evaluate four distinct tasks: machine translation, factual question answering, world knowledge understanding, and math reasoning. Our findings reveal varying levels of memorization, with the strongest effect observed in factual question answering. Furthermore, while model performance improves across all tasks as LLM size increases, only factual question answering shows an increase in memorization, whereas machine translation and reasoning tasks exhibit greater generalization, producing more novel outputs. This study demonstrates that memorization plays a larger role in simpler, knowledge-intensive tasks, while generalization is the key for harder, reasoning-based tasks, providing a scalable method for analyzing large pretraining corpora in greater depth. $ ^{1} $ 

## 1 INTRODUCTION

Large language models (LLMs), such as GPT-4, have achieved remarkable performance across a wide range of tasks, yet the debate persists regarding whether these models are truly generalizing to unseen test cases or merely memorizing their extensive training data (Magar and Schwartz, 2022; Srivastava et al., 2024; Bender et al., 2021; Merrill et al., 2024; Shaib et al., 2024). Previous research has primarily investigated memorization in LLMs through verbatim recall of long segments from the training corpus (Zhang et al., 2023; Jiang et al., 2024; Carlini et al., 2022). However, the exact reproduction of long text is relatively uncommon when examining high-level capabilities such as translation and reasoning, particularly when the task output is short, as with world knowledge questions. To advance this line of inquiry, a more flexible definition of memorization is necessary.

Several works have explored the relationship between memorization and generalization (Feldman, 2020; Feldman and Zhang, 2020; Zhang et al., 2023), often employing counterfactual memorization, which measures the difference in model performance when a specific training example is excluded. However, these studies typically use small models and datasets, or subsets of larger datasets, and focus primarily on quantifying how much model behavior depends on memorization. They do not fully examine how different model capabilities emerge from the interplay between memorization and

<div style="text-align: center;"><img src="imgs/img_in_image_box_214_163_1007_452.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 1: Overview of our proposed analysis pipeline. For the selected evaluation tasks, we first construct a task-gram table by matching semantically similar n-grams from task inputs  $ (x) $  and targets  $ (y) $ . These n-grams are then searched within the pretraining corpus, yielding their counts and source documents. We then build a task-gram language model from the obtained counts and then analyze their relationship with LLM predictions.</div>


generalization. Moreover, the limited scale of these analyses is partly constrained by the definition of counterfactual memorization, which requires expensive model retraining.

In this paper, we introduce a new framework for understanding memorization at scale, enabling us to analyze various LLM capabilities. We define distributional memorization as the correlation between the distribution of LLM outputs and the distribution of pretraining data. Similarly, we define distributional generalization as the divergence between the LLM's output distribution and the pretraining data distribution. For simplicity, we will refer to these concepts as memorization and generalization throughout the paper.

Our approach to defining memorization and generalization requires estimating the distribution of LLMs' pretraining corpora, which is a challenging task given the sheer size of these datasets, often containing trillions of tokens. To address this, we propose a novel method for modeling language distributions by counting semantically related n-gram pairs extracted from a task's input-output pairs. For example, in machine translation, these n-grams would correspond to phrase pairs from the source and target languages, as shown in Figure 1. This allows us to construct a set of n-gram pairs that characterize the task. Drawing inspiration from the phrase table used in machine translation (Passban et al., 2016), which consists of translation pairs, we refer to this set as the task's task-gram table. When n-grams from the input and output co-occur within a document, they are often separated by significant distances. By counting these co-occurrences, we are able to model long-range, task-relevant dependencies Elazar et al. (2019; 2022). The resulting n-gram language model, built from the task-gram table, is referred to as a task-gram language model. In contrast, classical n-gram LMs capture only local lexical dependencies within a single n-gram. Although Liu et al. (2024) introduced a  $ \infty $ -gram LM, which extends n to infinity through backoff from infinitely long n-grams, the effective length after backoff remains limited and cannot capture the long-range dependencies modeled by our task-gram LM. These n-gram LMs can be viewed as empirical approximations of the pretraining data distribution since they follow the observed frequency of n-grams (or pairs) in the data. Using these n-gram LMs, we measure the degree of memorization by correlating LLM-predicted probabilities with the probabilities generated by the n-gram LMs.

Our experiments focus on the Pythia model family (Biderman et al., 2023), pretrained on the Pile dataset (Gao et al., 2020), and evaluate performance across three tasks: translation (WMT (Callison-Burch et al., 2009)), factual question answering (TriviaQA (Joshi et al., 2017)), world knowledge questions (MMLU (Hendrycks et al., 2020)), and math reasoning (GSM8K (Cobbe et al., 2021)). The high-level overview of our analysis pipeline is depicted in Figure 1. We first construct a task-gram table using supervised task data, then search for the co-occurrence of n-gram pairs in the pretraining corpus using WIMBD (Elazar et al., 2024). Finally, we construct a task-gram LM from the co-occurrence counts and compare it with LLM predictions. Our results demonstrate that the task-gram LM effectively captures task-relevant data distributions, providing better explanations for LLM behaviors compared to the  $ \infty $ -gram LM (Liu et al., 2024).

Among the four tasks, our analysis reveals that TriviaQA exhibits the strongest memorization effect, with task performance highly correlated to the n-gram distributions in the pretraining data. In contrast, MMLU shows weaker memorization, and WMT exhibits only an insignificant memorization effect. Additionally, our results show that as the model size increases, the source of performance gains varies across tasks: for TriviaQA, improved memorization plays a key role, while for MMLU, WMT, and GSM8K, increased generalization is more crucial. These findings align with the nature of the tasks: TriviaQA relies on factual recall, MMLU and GSM8K require more complex reasoning, and WMT reflects transferable translation skills.

To complement our distributional memorization analysis, we also conduct a gradient-based estimation of the training influence of pretraining documents on test examples. We find that, consistent with our distributional memorization results, pretraining documents have the largest impact on TriviaQA, followed by MMLU, and the least impact on WMT. Documents containing n-gram pairs from the task-gram table exert a greater influence than those containing only individual n-grams.

To our knowledge, this work presents one of the first comprehensive analyses of LLM capabilities by tracing their origins to pretraining corpora at scale. Our task-gram language model offers a scalable, generalizable approach for understanding LLM behavior across a variety of tasks. $ ^{2} $ 

## 2 METHOD

Diverse abilities have been observed from LLMs trained on large pretraining corpora. Many of them are distinct in nature, like knowledge retrieval and math reasoning. It is reasonable to hypothesize that these capabilities come from different subsets of texts from pretraining. To better identify these task-relevant texts, we propose to construct a task-gram table from a set of supervised task data  $ D_{T}=\{(x_{i},y_{i})\}_{i} $  corresponding to task T, to characterize the task-relevant documents. More specifically, we mine semantically similar n-gram pairs  $ (s^{x},s^{y}) $  from corresponding task input x and output y respectively. i.e.,  $ s^{x}\subseteq x $ , and  $ s^{y}\subseteq y $ . We use the cosine similarity between the embeddings of the n-grams to measure the semantic closeness of the n-grams. The task-gram table is then constructed by all possible such n-gram pairs in  $ D_{T} $ . The mined n-gram pairs capture task-specific supervision signals as the output n-gram can be viewed as “answering” the input n-gram. We formally define the task-gram table as follows:

Definition 1. Denote all possible combinations of input-output n-gram pairs from task data  $ D_{T}=\{(x_{i},y_{i})\}_{i} $  by  $ A_{n}(T)=\cup_{i}[G_{n}(x_{i})\times G_{n}(y_{i})] $ , where  $ G_{n}(\cdot) $  denotes the set of all possible n-grams in a piece of text. Then the task-gram table  $ H_{n}(T) $  is defined as:

 $$ H_{n}(T)=\{(s_{j}^{x},s_{j}^{y})\mid cos(E(s_{j}^{x}),E(s_{j}^{y}))>\gamma_{T},s^{x}\neq s^{y},(s_{j}^{x},s_{j}^{y})\in A_{n}(T)\}, $$ 

where  $ \gamma_{T}\in(0,1) $  is a threshold chosen as a hyperparameter, E is a pretrained text embedding model, and  $ \cos(\cdot,\cdot) $  denotes the cosine similarity between two vectors.

Based on the task-gram table, we can then construct a task-gram language model, which can describe the distribution of the characterized task-related data in a large pretraining corpus. In the following paper, we use C as the counting function. We define the number of co-occurrence of a n-gram pair  $ (s^{x}, s^{y}) $  in the same document in the pretraining copus D by  $ C((s^{x}, s^{y}), \mathcal{D}) $ . We also define the number of occurrences of a n-gram  $ s^{y} $  in the pretraining corpus D by  $ C(s^{y}, \mathcal{D}) $ . Then a task-gram language model can be defined as follows:

Definition 2. A task-gram language model is defined over its task-gram table  $ H_{n}(T) $ . For  $ \forall(s^{x}, s^{y}) \in H_{n}(T) $  and a corpus of interest D, we define the following probability distribution

 $$ P_{n,\mathcal{D}}(s^{y}|s^{x})=C((s^{x},s^{y}),\mathcal{D})/C(s^{x},\mathcal{D}). $$ 

In practice, when we observe that an n-gram  $ s^{x} $  exists in an unseen text input, the chance of the corresponding n-gram  $ s^{y} $  appearing in the output can be estimated by  $ P_{n,\mathcal{D}}(s^{y}|s^{x}) $ .

Suppose we have an LLM pretrained on the corpus D. Considering the zero-shot setting where we prompt the LLM with an instruction text u and an input text  $ x^{3} $ . Suppose  $ s^{x} \subseteq x $  and  $ s^{y} \subseteq y $ , we

want to define an LLM version of the above n-gram conditional distribution using the LLM predicted probability of  $ s^{y} $  in the context of the concatenated testing example  $ u \oplus x \oplus y $ :

 $$ P_{\mathrm{L L M}}(s^{y}|s^{x})=\prod_{t\in s^{y}}P_{\mathrm{L L M}}(t|u\oplus x\oplus y_{[1:m-1]}). $$ 

Here m denotes the location index of the n-gram  $ s^{y} $  found in y, and t is each token in the tokenized n-gram  $ s^{y} $ . We can then formally define the distributional memorization by Spearman correlation  $ \rho $  between the task-gram language model probabilities and LLM predicted probabilities of testing data:

Definition 3. For a testing set  $ D_{T}^{\prime}=\{(x_{i},y_{i})\}_{i} $ , we denote all n-gram pairs found in it by  $ \Phi=\{(s^{x},s^{y})|\forall(s^{x},s^{y})\in[G_{n}(x)\times G_{n}(y)]\cap H_{n}(T),\forall(x,y)\in D_{T}^{\prime}\} $ . Then we define the extent of an LLM distributional memorize the pretraining corpus D when performing task T as follows:

 $$ M e m_{n}(L L M,\mathcal{D}|T)=\rho(\log P_{n,\mathcal{D}}(Y|X),\log P_{L L M}(Y|X)), $$ 

where  $ \rho $  denotes Spearman correlation,  $ \log P_{n,\mathcal{D}}(Y|X)=\{\log P_{n,\mathcal{D}}(s^{y}|s^{x})|\forall(s^{x},s^{y})\in\Phi\} $ , and  $ \log P_{LLM}(Y|X)=\{\log P_{LLM}(s^{y}|s^{x})|\forall(s^{x},s^{y})\in\Phi\} $ .

Similar to normal Spearman correlations, the significance of distributional generalization can be measured by p-value. The distributional generalization is then defined as the opposite of the distributional memorization: increased memorization implies decreased generalization, as the LLM predictions and the n-gram LM are more distributionally correlated, and vice versa.

Distributional memorization describes to what extent the LLM-predicted probability of the ground truth testing data can be viewed as a monotonic function of task-gram LM probability. This can be viewed as a measure of the predictability of task-gram LM to the LLM probability. In the following sections, we show comprehensive empirical evidence of LLMs performing knowledge-intensive tasks depending on distributional memorization while performing reasoning-intensive tasks depending on distributional generalization, as shown in Figure 2.



<div style="text-align: center;"><img src="imgs/img_in_chart_box_609_623_788_775.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_794_643_970_775.jpg" alt="Image" width="14%" /></div>


## 3 EXPERIMENTAL SETUP

<div style="text-align: center;"><img src="imgs/img_in_chart_box_608_779_786_907.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_793_779_969_907.jpg" alt="Image" width="14%" /></div>


<div style="text-align: center;">Figure 2: Expected distributional memorization and generalization trend for different types of tasks.</div>


In this section, we introduce the datasets, models, and tools we use for analyzing the memorization and generalization behaviors of LLMs.



Models and Pretraining Corpus We use a family of fully open-sourced, Transformer-decoder-based LMs: Pythia (Biderman et al., 2023), with a wide range of model sizes ranging from 13M to 12B parameters. All Pythia models are trained on Pile (Gao et al., 2020), a diverse pretraining corpus consisting of approximately 207B tokens. We also include some results with 1B and 7B OLMo models (Groeneveld et al., 2024), pretrained on Dolma (Soldaini et al., 2024b) with 3T tokens.

Downstream Tasks We use four types of tasks: machine translation, factual question answering, world knowledge questions, and reasoning.

For translation, we use the WMT-09 dataset (Callison-Burch et al., 2009) with a 2.5K testing set. WMT is a classic annual machine translation shared task with different languages. We chose WMT09 as our testing data instead of newer versions of WMT because WMT09 contains more European languages, which is more prominent in the Pile.

For factual question answering, we use the TriviaQA dataset (Joshi et al., 2017) with a 10K testing set, which is a knowledge-intensive question-answering dataset with the questions originating from trivia enthusiasts. Since the answers are usually single words or short phrases, we regard the whole answer text as the output n-gram  $ s^{y} $  no matter the value of n.

For world knowledge questions, we use the MMLU benchmark (Hendrycks et al., 2020), covering 57 tasks including elementary mathematics, US history, computer science, law, and more. The aim of

<div style="text-align: center;"><img src="imgs/img_in_chart_box_211_161_399_353.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_398_161_606_355.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_604_160_804_351.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_803_161_1009_352.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">Figure 3: Task performance v.s. n-gram pair count in the Pile with different Pythia model sizes, for four different tasks, from left to right: WMT, TriviaQA, MMLU, and GSM8K.</div>


For WMT, the x-axis shows the counts of n-gram pairs found in the testing set of six different languages: Hungarian, Czech, German, Italian, Spanish, and French, from left to right.

MMLU is to test models' world knowledge and problem-solving ability, so different from TriviaQA, there is a large portion of questions that require logical/math reasoning skills.

For complex reasoning, we use the GSM8K dataset (Cobbe et al., 2021), which contains linguistically diverse grade school math world problems with step-by-step solutions. To solve GSM8K, the LLM needs to first generate the chain-of-thoughts (CoT) reasoning step, and then draw the final conclusion, which requires advance math and logical reasoning capabilities.

Searching over Pretraining Data at Scale Given the scale of the LLM pretraining corpus, searching n-grams over the whole corpus D is non-trivial. We use What's In My Big Data? (WIMBD) (Elazar et al., 2024) and the  $ \infty $ -gram (Liu et al., 2024) platform, which are designed to search and retrieve documents in huge corpora. Both of them have indexed the Pile, allowing us to search it using API calls. We use WIMBD for accurately counting the co-occurrence of n-gram pairs as the co-occurrence frequency is usually low, and the approximate counting function in  $ \infty $ -gram sometimes fails to capture these low-frequency n-gram pairs. We use  $ \infty $ -gram for counting the occurrence of single n-grams as they appear more frequently in the corpus and the counting approximation has a relatively small effect on the search results. We also use the  $ \infty $ -gram API to produce the  $ \infty $ -gram probability of single n-grams, as detailed in Section 5.

#### 4 n-GRAM DATA FREQUENCY V.S. TASK PERFORMANCE

In this section, we investigate the overall relation between the n-gram pair counts and the LLM task performance, which can be viewed as a rough estimation of the importance of the task-related data. We confirm that the Pile is not contaminated by any of the datasets we used, by ensuring there are no large n-grams  $ (n = 8 $  and  $ n = 14) $  overlaps between the Pile and the testing data. This decontamination method is adopted from the GPT3 technical report (Brown, 2020).

We estimate the probability of a test example  $ (x,y) $  appearing in the pretraining corpus as the probability of any of the n-gram pairs from the task-gram table found in  $ (x,y) $  appearing:

 $$ P_{\mathcal{D},n}(x,y)\propto\sum_{(s^{x},s^{y})\in(x,y)}C((s^{x},s^{y}),\mathcal{D})\mathbb{1}_{(s^{x},s^{y})\in H_{n}(T)} $$ 

In Figure 3, we plot the task performances v.s. the count of n-gram pairs per example as defined in Equation (4), for WMT  $ (n=2) $ , TriviaQA  $ (n=5) $ , and MMLU  $ (n=3) $ . For WMT, since the test set size for each language is the same, thus we show the sum of count per language here. For TriviaQA, MMLU, and GSM8K, the x-axis represents bins that group test examples based on the number of n-gram pairs. For instance, in the case of TriviaQA, data points at x=0 correspond to test examples where the number of n-gram pairs falls between 0 and 2000. In general, TriviaQA has more n-gram pair counts than WMT, MMLU, and GSM8K.

The WMT performance is evaluated by the BLEU score between greedily generated translation and the reference translations. The TriviaQA performance is evaluated by the accuracy of the exact match of the generated answer. The MMLU performance is evaluated by the accuracy of the option

with the highest LM predicted probability. Since there are four choices for each question, the random performance of MMLU is 25%. For GSM8K, since the performance of Pythia models is low ( $ < 5\% $  accuracy), the accuracy plot is too sparse to show any trend. We compute the BERTScore (Zhang $ ^{*} $  et al., 2020) (precision) between the model-generated chain-of-thoughts (CoT) and the ground truth CoT instead.

When the model size is small (< 410m), WMT and TriviaQA have near-zero performance regardless of the n-gram pair count, while interestingly, MMLU reaches the lowest performance (<10%) significantly lower than random guessing when the n-gram pair count is around 150. A closer inspection of the test examples in this interval reveals that they contain more reasoning or math problems, which appear to be harder for Pythia models. The noisier performance curve of MMLU and GSM8K is likely due to the weaker capabilities of Pythia models on this benchmark.

In general, all task performance increases when the number of task-related n-gram pairs increases when the model size is large enough (> 410m). And the trend of performance improvement is more significant when the model size is larger. For GSM8K, the Pythia 2.8B model shows the most significant increasing trend, while larger models show a less significant trend. This seems to indicate that memorization plays an essential role in LM's capabilities for all four tasks, and larger models memorize more. However, these performance curves can also be explained by the improved generalization ability of LMs when there is more relevant pretraining data. In the next section, we use pre-defined distributional memorization and generalization to investigate the possible causes behind these performance trends.

#### 5 n-GRAM DISTRIBUTION V.S. LLM DISTRIBUTION

In this section, we compute the distributional memorization  $ Mem_{n}(\text{LLM}, \mathcal{D}|T) $  as defined in Definition 3, for T = WMT, TriviaQA, and MMLU respectively. In addition to computing the distributional memorization with the task-gram language model as defined in Definition 2, we consider computing another version of distributional memorization with an n-gram language model defined by single n-grams. Specifically, we consider the  $ \infty $ -gram language model (Liu et al., 2024) that uses an n as large as possible for predicting the probability of each token, which is shown to be better aligned with human written text compared with classical n-gram LMs.

An  $ \infty $ -gram LM can be viewed as an n-gram LM initialized with  $ n = \infty $ , and then backoff when the n-gram count equals zero. This way, the probability of each token is dependent on its longest prefix that exists in the pretraining corpus. Considering concatenating the input and output text  $ u \oplus x \oplus y $  as the context, such distribution can be written as

 $$ P_{\infty,\mathcal{D}}(s^{y}|u\oplus x\oplus y)=\prod_{t_{i}\in s^{y}}P_{\infty,\mathcal{D}}(t_{i}|t_{[1:i-1]})\\=\prod_{t_{i}\in s^{y}}C(t_{[i-(n_{i}-1):i]})\\/C(t_{[i-(n_{i}-1):i-1]}). $$ 

Here i is the location index of the token  $ t_{i} $  in the concatenated text  $ u \oplus x \oplus y $ , and  $ n_{i} $  is the size of the longest prefix of  $ t_{i} $  that can be found in the pretraining corpus, i.e.,  $ n_{i} = \max \left\{ n' \in [1, i] \mid C(t_{[i-(n'-1):i-1]}) > 0 \right\} $ . In practice, we ignore the tokens with zero probability and set the  $ \infty $ -gram probability of this token to one, i.e.,  $ P_{\infty, \mathcal{D}}(s^{y}) = 0 $  only when all its tokens have zero probability. Then the alternative version of distributional memorization using  $ \infty $ -gram LM is defined as:

 $$ \mathbf{M e m}_{\infty}(\mathbf{L L M},\mathcal{D}|T)=\rho(\log P_{\infty,\mathcal{D}}(Y|X),\log P_{\mathbf{L L M}}(Y|X)), $$ 

with all the notations similarly defined as in Definition 3. We then compute the two versions of distributional memorization for the different datasets, WMT, TriviaQA, and MMLU, and show the results in Figure 4.

Translation ability does not come from memorization. In Figure 4, we do not show any distributional memorization values of WMT because none of them are statistically significant. This indicates that while the translation performance is strongly positively correlated to the n-gram counts as shown in Figure 3, the performance gain does not come from initiating the pretraining data distribution.

To further investigate how different the LLM-generated translation text is from the pretraining data, we show the number of novel n-gram pairs that LLMs generated on WMT that have never been seen

<div style="text-align: center;"><img src="imgs/img_in_chart_box_251_164_969_546.jpg" alt="Image" width="58%" /></div>


<div style="text-align: center;">Figure 4: Visualization of distributional memorization with different-sized Pythia models on four tasks: WMT, TriviaQA, MMLU, and GSM8K. We also show results with OLMo models on GSM8K. For WMT, we show the number of new n-gram pairs generated by LLMs as the distributional memorization is not significant. For MMLU, we divide the tasks into two categories: knowledge-intensive and reasoning-intensive. For GSM8K, we show the Kendall tau ranking distance instead of Spearman correlation to quantify the distributional generalization effect as the distributional memorization is not significant. Solid lines show distributional memorization computed with our task-gram LM and dashed lines are computed with the  $ \infty $ -gram LM. Statistical significant  $ (p < 0.05) $  values are marked with solid round markers while statistically insignificant values  $ (p > 0.05) $  are marked with gray star markers.</div>


in any pretraining documents in the top left panel of Figure 4. It shows that larger LLMs generate more novel n-grams, which indicates a larger discrepancy in text distribution from the pretraining data and better distributional generalization. This implies the performance increase in Figure 3 comes from a better generalization ability learned from more relevant data instead of memorization. Our hypothesis is that this generalization is the transfer of translation skills between different languages.

Such a result contradicts the observations in Merrill et al. (2024), which show that larger LLMs are less novel in n-gram generation. This is likely because they evaluate LLM generation using the prompts from an in-distribution dataset (validation set of the Pile) to the pretraining corpus, which might encourage LLMs to memorize. On the other hand, the translation dataset we use is out-of-distribution, which encourages LLMs to use its generalization capabilities.

Knowledge-intensive question-answering ability relies more on memorization. In the top middle panel of Figure 4, we show that TriviaQA has a significant distributional memorization effect, in terms of both task-gram LM and  $ \infty $ -gram LM.  $ \mathsf{Mem}_{n=3}(\mathsf{LLM},\mathcal{D}|T) $  (>0.35) is significantly more profound than  $ \mathsf{Mem}_{n=5}(\mathsf{LLM},\mathcal{D}|T) $  (<0.25) and  $ \mathsf{Mem}_{\infty}(\mathsf{LLM},\mathcal{D}|T) $  (<0.25). This indicates that LLMs memorize small, long-range parallel data pieces more than large, local data pieces.

Also, when the LM size increases, our  $ \operatorname{Mem}_{n=5}(\operatorname{LLM},\mathcal{D}|T) $  (>0.35) also increases. Since the larger model also has better performance as shown in Figure 3, this memorization behavior is highly correlated with LMs' factual QA ability. This might be because factual QA requires retrieving knowledge from training data, thus memorization plays a critical role in this task.

For MMLU shown in the two rightmost panels of Figure 4, we divide the 57 MMLU tasks into two groups: knowledge-intensive and reasoning-intensive. We consider knowledge-intensive tasks as tasks that can be answered by retrieving static knowledge, while reasoning-intensive tasks as ones that need computation or logical reasoning over the knowledge. $ ^{4} $  In general, knowledge-intensive tasks show a more significant memorization effect than reasoning-intensive tasks. This discrepancy is the most profound when n = 3, while the memorization level of knowledge-intensive tasks and reasoning-intensive tasks is similar when n = 5. This echoes our previous hypothesis that LLMs demonstrate a stronger memorization behavior when performing knowledge-intensive tasks.

Recalling rare knowledge requires generalization. Similar to TriviaQA, the top right panel of Figure 4 shows that our  $ \mathsf{Mem}_{n=3}(\mathsf{LLM},\mathcal{D}|T) $  (>0.25) remains most pronounced for the knowledge-intensive MMLU tasks. The primary distinction between the MMLU tasks and TriviaQA is that  $ \mathsf{Mem}_{n}(\mathsf{LLM},\mathcal{D}|T) $  decreases as the LLM size increases. This may be attributed to the fact that MMLU involves more specialized and less common knowledge compared to TriviaQA, making its occurrence in the pretraining corpus relatively infrequent. Consequently, for larger models to perform better on MMLU tasks, they may need to adjust the probability of recalling this knowledge, resulting in a decrease in distributional memorization.

Reasoning-intensive abilities rely more on generalization. The reasoning-intensive MMLU tasks in the bottom right panel of Figure 4 show a very different picture compared to the knowledge-intensive MMLU tasks and TriviaQA. In this case, our  $ \text{Mem}_{n=5}(\text{LLM}, \mathcal{D}|T) $  is the most significant, which indicates that the memorization of large text segments is more significant. This might be because some concepts can be meaningfully expressed in large text segments while the small text segments are meaningless in a reasoning-intensive context. The decreasing trend of memorization when the model size increases also indicates that memorization is not the driving force of performance improvement.

For GSM8K shown in the two bottom panels of Figure 4, we did not observe a significant memorization effect with either Pythia models or OLMo models. To quantify the distributional generalization, we substitute the Spearman correlation with the normalized Kendall tau ranking distance, which represents the fraction of data pairs that disagree on their rankings. For both Pythia models and OLMo models, the distributional generalization increases when the model size increases, while the LLMs' probabilities agree more with task-gram probabilities than the inf-gram probabilities. The GSM8K results confirm that generalization is the driving force of performance improvement.

Task-gram LM can better explain LLM predicted probabilities than  $ \infty $ -gram LM. With both TriviaQA and MMLU, we observe that  $ \operatorname{Mem}_{\infty}(\operatorname{LLM},\mathcal{D}|T) $  is always less or equal to our  $ \operatorname{Mem}_{n}(\operatorname{LLM},\mathcal{D}|T) $ . And larger LLMs always show less  $ \infty $ -gram memorization effect. This shows that our task-gram LM is a better way to model the data distribution so that it is more correlated to what is memorized by LLMs.

In general, the LLM shows decreased memorization and increased generalization when the model size increases. This implies LLMs leverage better generalization to solve hard tasks, whether they are hard in terms of the rarity of the knowledge or in terms of the requirement of reasoning. Our task-gram LM also better models the memorized data distribution than the  $ \infty $ -gram LM.

## 6 INFLUENCE OF n-GRAM DATA THROUGH PRETRAINING

Since our results so far only show correlations between pretraining data frequency and the LLM predictions, we wish to explore the causal relationship of training data on LLM predictions. In this section, we estimate the influence of pretraining data on related testing predictions by accumulating the gradient dot products through model checkpoints, as introduced in Pruthi et al. (2020).

More specifically, we approximate the influence of a pretraining document at training time by the dot product between its pretraining loss gradient and the testing loss gradient. For an n-gram pair  $ (s^{x}, s^{y}) \in H_{n}(T) $  found in some example  $ (x, y) $ , the test loss is defined as:

 $$ \ell(\theta,(x,y),s^{y})=-\log P_{\mathrm{L L M}}(s^{y}|u,x,y_{[1:m-1]}). $$ 

Here m denotes the location index of the n-gram  $ s^{y} $  found in y, and  $ \theta $  denotes all trainable parameters of the LLM. The training loss of a pretraining document  $ d \in D $  containing the n-gram pair  $ (s^{x}, s^{y}) $  is defined as:

 $$ \ell(\theta,d,s^{y})=-\log P_{\mathrm{L L M}}(s^{y}|d_{[1:h-1]}). $$ 

Here h denotes the location index of the n-gram  $ s^{y} $  found in d. Suppose we have k evenly spaced pretraining checkpoints of the LLM  $ \theta_{1}, \theta_{2}, \ldots, \theta_{k} $ , then the influence of d on the testing example  $ (x, y) $  through the pretraining is defined as:

 $$ I n(d,(x,y))=\sum_{i=1}^{k}\sum_{(s^{x},s^{y})\in\Phi(x,y)}\nabla_{\theta_{i}}\ell(\theta_{i},d,s^{y})\cdot\nabla_{\theta_{i}}\ell(\theta_{i},(x,y),s^{y}). $$ 

<div style="text-align: center;"><img src="imgs/img_in_chart_box_288_161_929_360.jpg" alt="Image" width="52%" /></div>


<div style="text-align: center;">Figure 5: Training influence of pretraining documents v.s. Pythia model size with WMT, TriviaQA, and MMLU. Green lines correspond to documents containing n-gram pairs, while blue lines correspond to documents containing only the output n-gram in n-gram pairs.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td colspan="2">TriviaQA</td><td colspan="2">GSM8K</td></tr><tr><td style='text-align: center;'>Memorization</td><td style='text-align: center;'>Generalization</td><td style='text-align: center;'>Memorization</td><td style='text-align: center;'>Generalization</td></tr><tr><td style='text-align: center;'>Pythia (6.9B)</td><td style='text-align: center;'>17%</td><td style='text-align: center;'>9%</td><td style='text-align: center;'>2.6%</td><td style='text-align: center;'>2.8%</td></tr><tr><td style='text-align: center;'>Pythia-Instruct (6.9B)</td><td style='text-align: center;'>23.5%</td><td style='text-align: center;'>23.2%</td><td style='text-align: center;'>6.3%</td><td style='text-align: center;'>7.3%</td></tr><tr><td style='text-align: center;'>Pythia (12B)</td><td style='text-align: center;'>28.7%</td><td style='text-align: center;'>23.2%</td><td style='text-align: center;'>2.7%</td><td style='text-align: center;'>2.8%</td></tr><tr><td style='text-align: center;'>OLMo (7B)</td><td style='text-align: center;'>36.4%</td><td style='text-align: center;'>29.8%</td><td style='text-align: center;'>2.5%</td><td style='text-align: center;'>3.1%</td></tr><tr><td style='text-align: center;'>OLMo-instruct (7B)</td><td style='text-align: center;'>29%</td><td style='text-align: center;'>10%</td><td style='text-align: center;'>6.3%</td><td style='text-align: center;'>7.9%</td></tr></table>

<div style="text-align: center;">Table 1: Zero-shot accuracy on TriviaQA and GSM8K test set with memorization encouraged task prompt (maximize counts) and generalization encouraged task prompt (minimize counts).</div>


Here, we use  $ \Phi(x,y) $  to denote all n-gram pairs found in  $ (x,y) $  and exist in the task-gram table  $ H_{n}(T) $ . Such an influence function can be viewed as an estimation of the total reduction in loss on the test example  $ (x,y) $  that is induced by the training process whenever document d is utilized in the pretraining. To estimate how much influence the documents containing n-gram pairs have on each testing task, we compute the average influence over the testing set  $ D_{T}^{\prime} $  and R randomly retrieved documents from pretraining corpus D for each testing example. The average influence obtained can then be written as:

 $$ I n(\mathcal{D},T)=\frac{1}{|D_{T}^{\prime}|}\frac{1}{R}\sum_{(x,y)\in D_{T}^{\prime}}\sum_{i=1}^{R}I n(d_{i},(x,y)) $$ 

Note that the computation of the full gradient is relatively expensive, so we choose a relatively small R = 50. This analysis does not aim to cover the full pretraining corpus but to give complementary causal evidence to the previous findings. We consider two retrieval schemes: 1. retrieving documents where the test sample's n-gram pair appears together. 2. retrieving documents that contain the output n-gram from the test sample's n-gram pair. In Figure 5, we plot the average influence of these two schemes by green and blue curves, respectively.

Across all three datasets, we observe that pretraining documents containing n-gram pairs consistently contribute more to the testing examples than documents containing only the output n-gram in n-gram pairs, over different model sizes. This suggests that our task-gram table identifies important task-relevant data better than single n-grams. The difference between n-gram pair and output n-gram is the smallest on WMT, as well as the value of the influence function, which echoes the insignificant memorization effect of LLMs on this task. In general, when LLM size increases, WMT and MMLU decrease in data influence, while TriviaQA slightly increases, and has the highest influence value at the largest model size. This indicates that memorization is likely caused by more influence of the relevant data at training time.

## 7 PRACTICAL IMPLICATIONS: PROMPT OPTIMIZATION

An important observation of our study is that knowledge-intensive tasks benefit from LLMs' distributional memorization, while reasoning-intensive tasks benefit from LLMs' distributional generalization. Then it is possible to design or rewrite the prompt according to this principle to improve an LLM's task performance, based on the hypothesis that the LLM generation distribution is strongly affected by the prompt distribution. More specifically, to encourage memorization, we can rewrite the task instruction

to be more similar to pretraining data in terms of n-gram counts. To encourage generalization, we can rewrite the prompt to be less similar to training data.

We implement a simple prompt optimizer based on GPT4o and the WIMBD n-gram count feedback. More specifically, we instruct GPT4o (Achiam et al., 2023) to rewrite a given task prompt at each iteration, and give the average n-gram count in the pretraining corpus of the rewritten prompt to GPT4o in the next iteration as the reward. We instruct GPT4o to maximize this reward if we want to encourage memorization, and instruct it to minimize this reward if we want to encourage generalization. Here, we show a maximization and a minimization result for TriviaQA and GSM8K respectively. We report zero-shot testing accuracy with the Pythia models and OLMo models in Table 1. The meta prompt we used to perform such optimization and the optimized task prompts are included in Appendix E.

Note that the lengths of the optimized prompts are not significantly different, while TriviaQA significantly benefits from the prompts that are more similar to the pretraining data, and GSM8K benefits from the prompts that are less similar. More sophisticated prompt optimization algorithms with more detailed distributional memorization feedback can be designed based on a similar idea. We leave the investigation of other possibilities for future work.

## 8 RELATED WORK

Understanding LLMs' capabilities from training data. Most work on understanding LLMs analyzes their capabilities via synthetic experiments or small-scale studies (Arora and Goyal, 2023; Prystawski et al., 2023; Wang et al., 2024; Xie et al., 2022; Wang et al., 2023; Chan et al., 2022; Razeghi et al., 2023; Chen et al., 2024), despite the importance of scaling. Kirchenbauer et al. (2024) estimate dependencies between model capabilities and subsets of training data using kernel-based statistical evidence but rely on a small fraction of pretraining data due to computational constraints. To address this, Elazar et al. (2024) introduce WIMBD, a system for efficient n-gram retrieval over massive datasets, while Merrill et al. (2024) develop an unbounded n-gram search method and find larger LLMs generate fewer novel n-grams. Shaib et al. (2024) demonstrate how syntactic templates in training data influence LLM outputs, and Liu et al. (2024) propose  $ \infty $ -gram language models to estimate text corpora distributions, focusing on local dependencies rather than complex capabilities.

In this work, we analyze the origins of LLMs' zero-shot capabilities by leveraging WIMBD and  $ \infty $ -gram frameworks for full-scale pretraining data exploration.

Memorization vs. generalization. LLM memorization, defined as exact recall of training data, has been extensively studied, including memorization of rare or private data (Zhang et al., 2023), test set contamination (Jiang et al., 2024), and higher prevalence of verbatim recall in larger models (Carlini et al., 2022). Hartmann et al. (2023) provides a comprehensive summarization of different types of LLM memorization. The relationship between memorization and generalization is explored in Feldman (2020), showing that memorization can improve generalization. Extensions to this work quantify memorization via performance differences when specific training examples are included or excluded (Feldman and Zhang, 2020; Zhang et al., 2023). However, this approach is infeasible for large-scale LLMs due to retraining requirements. We propose a scalable definition of distributional memorization using n-gram counts to enable efficient large-scale analysis. $ ^{5} $ 

## 9 CONCLUSION

In this paper, we introduce the task-gram language model, a scalable approach for modeling the task-relevant distribution of pretraining data. We trace the capabilities of LLMs back to this data by defining distributional memorization, measured through the Spearman correlation between task-gram LM probabilities and LLM probabilities. Through extensive experiments using Pythia models across four distinct tasks, we find that LLMs tend to memorize more when engaged in simpler, knowledge-intensive tasks, while they generalize more in harder, reasoning-intensive tasks. Our analysis provides a comprehensive examination of the origins of LLM capabilities and offers a scalable framework for investigating the fine-grained task-relevant characteristics of pretraining corpora.