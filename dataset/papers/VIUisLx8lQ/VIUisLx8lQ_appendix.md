## A APPENDIX

### A.1 PROMPT

We introduce the simple and practical definition of four reasoning types in Table 8. Table 16 lists the few-shot examples of each reasoning type. The full few-shot examples can be found in the supplementary materials. We use the same few-shot examples for the logical problems and create another set of examples for the mathematics problems.

<div style="text-align: center;">Table 8: Description of different reasoning types. We give informal definitions that are easy to follow and illustrate simple examples for each reasoning type.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Type</td><td style='text-align: center;'>Definition</td><td style='text-align: center;'>Example</td></tr><tr><td style='text-align: center;'>Deduction</td><td style='text-align: center;'>Deduce conclusion based on the general rules and premise.</td><td style='text-align: center;'>From the premises ‘all frogs are amphibians’ and ‘no cats are amphibians’, we can infer the conclusion ‘no cats are frogs’.</td></tr><tr><td style='text-align: center;'>Induction</td><td style='text-align: center;'>Make broad generalizations from specific observations.</td><td style='text-align: center;'>Starting from the empirical observation that ‘all ravens I have seen so far are black’, inductive reasoning can be used to infer that ‘all ravens are black’.</td></tr><tr><td style='text-align: center;'>Abduction</td><td style='text-align: center;'>Assume one candidate is correct and check whether it meets the condition in the problem.</td><td style='text-align: center;'>Guess that it has rained to explain that the streets are wet. A tsunami could also explain why the streets are wet but this is usually not the best explanation.</td></tr><tr><td style='text-align: center;'>Analogy</td><td style='text-align: center;'>Retrieve several relevant information and draw the conclusion of this problem based on the similarity.</td><td style='text-align: center;'>Infer information about humans from medical experiments on animals: (1) rats are similar to humans; (2) birth control pills affect the brain development of rats; (3) therefore they may also affect the brain development of humans.</td></tr></table>

The prompt used by the meta-thinker is:

Given the question below, please identify the type of reasoning required to provide a solution. You may choose the following reasoning types: Deductive, Inductive, Analogical, Abductive Reasoning, or None. None indicates that no specific reasoning type is needed for this problem. Please assign an effectiveness score for each reasoning type from 0 to 1, where 0 represents no effective and 1 represents full effective. Please return the reasoning types and their corresponding effectiveness scores in the JSON format.

For instance, if you think the question can be solved using both deductive and inductive reasoning, with an effectiveness of 0.5 for deductive reasoning and 0.3 for inductive reasoning, you should return: [["ReasoningType": "Deductive", "Effectiveness": 0.5},["ReasoningType": "Inductive", "Effectiveness": 0.3},["ReasoningType": "Analogical", "Effectiveness": 0},["ReasoningType": "None", "Effectiveness": 0)].

The prompt used by the reasoner is listed below. The definition is based on Table 8.

Use  $ [f_{k}] $  reasoning to solve the given question.  $ [f_{k}] $  reasoning is [definition].

### A.2 DATASET

#### A.2.1 DATA PROCESSING

The dataset statistics of the four benchmarks are detailed in Table 9. For multiple-choice questions, we calculate accuracy using the exact match criterion. For mathematics problems, we compare the model’s response with the ground truth using mathematical equality.

LogiQA (Liu et al., 2021; 2023a) is a multi-choice understanding benchmark for logical reasoning. It follows the definition of DeLancey (2017) and categorizes the problems into categorical reasoning,

<div style="text-align: center;">Table 9: Logical benchmarks and mathematic benchmarks we used in this paper. We follow the standard train/test split on LogiQA and follow the split in Toshniwal et al. (2024) for GSM8k and MATH. For BBH, we randomly split the dataset. The synthesized data is described in Section 3.1. BBH includes 16 tasks while MATH includes math problems of 7 categories. Policy indicates the data used to train the meta-reasoner and SFT indicates the instruction-following in reasoner finetuning.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td colspan="5">Benchmark</td><td colspan="2">Empirical Dataset</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'># Task</td><td style='text-align: center;'># Train</td><td style='text-align: center;'># Val</td><td style='text-align: center;'># Test</td><td style='text-align: center;'># Total</td><td style='text-align: center;'># Meta-thinker</td><td style='text-align: center;'># Reasoner</td></tr><tr><td style='text-align: center;'>LogiQA</td><td style='text-align: center;'>1</td><td style='text-align: center;'>3757</td><td style='text-align: center;'>500</td><td style='text-align: center;'>511</td><td style='text-align: center;'>4768</td><td style='text-align: center;'>$ \sim $ 2k</td><td style='text-align: center;'>$ \sim $ 6k</td></tr><tr><td style='text-align: center;'>BBH</td><td style='text-align: center;'>16</td><td style='text-align: center;'>1904</td><td style='text-align: center;'>320</td><td style='text-align: center;'>1600</td><td style='text-align: center;'>3824</td><td style='text-align: center;'>$ \sim $ 1k</td><td style='text-align: center;'>$ \sim $ 3.5k</td></tr><tr><td style='text-align: center;'>GSM8k</td><td style='text-align: center;'>1</td><td style='text-align: center;'>6473</td><td style='text-align: center;'>1000</td><td style='text-align: center;'>1319</td><td style='text-align: center;'>8792</td><td style='text-align: center;'>$ \sim $ 4k</td><td style='text-align: center;'>$ \sim $ 4k</td></tr><tr><td style='text-align: center;'>Math</td><td style='text-align: center;'>7</td><td style='text-align: center;'>6500</td><td style='text-align: center;'>1000</td><td style='text-align: center;'>5000</td><td style='text-align: center;'>12500</td><td style='text-align: center;'>$ \sim $ 1k</td><td style='text-align: center;'>$ \sim $ 1k</td></tr></table>

sufficient conditional reasoning, necessary conditional reasoning, disjunctive reasoning, and conjunctive reasoning. These reasoning categories are not orthogonal and one problem can belong to multiple categories. We follow the standard training/validation split and only keep examples with more than 3 reasoning categories. This makes the problem more diverse and difficult to solve. We take the validation set as the test set and randomly select 500 examples from the training set for validation.

BBH (Suzgun et al., 2022) is a set of hard problems borrowed from Big Bench (Srivastava et al., 2022). They are also formatted as multi-choice problems. We pick the English tasks with more than 2 options, resulting in 16 tasks: date understanding disambiguation qa, geometric shapes, hyperbaton, logical deduction three, logical deduction five, logical deduction seven, movie recommendation, penguins in a table, reasoning color, ruin names, snarks, temporal sequences, tracking shuffled three, tracking shuffled five, and tracking shuffled seven. For each task, we randomly select 100 examples as the test set and 20 examples as the validation. The rest are used as training examples.

GSM8k (Cobbe et al., 2021) is a commonly used math benchmark to evaluate LLMs' capability in math reasoning. It contains 8.5K grade school math word problems, which are split into 7.5k training examples and 1k test problems. Each problem usually takes between 2 and 8 steps to solve. MATH (Hendrycks et al., 2021) is also a popular math benchmark for LLMs. It contains 12,500 challenging competition mathematics problems with 7 categories. There are 7.5k training examples and 5k test problems. We follow Toshniwal et al. (2024) to process the dataset.

Contexthub (Hua et al., 2024) is a new propositional logic benchmark. It contains abstract and contextualized logical problems from 12 categories with 4 levels of difficulty (Zhu et al., 2024). We follow the standard split of the original paper and use the subset of difficult level 4 to test the complex logic reasoning capabilities. The abstract logical problems only contain the symbolic variable without natural language description, which can be viewed as symbolic reasoning problems.

Livebench (White et al., 2024) is a recently proposed benchmark with 18 diverse tasks across 6 categories, specifically designed to minimize data contamination. All problems have verifiable, objective ground-truth answers, allowing hard questions to be scored accurately and automatically. We evaluate our models on three tasks (spatial, web of lies v2, zebra puzzle) from the reasoning category, splitting them 0.7/0.3 for training and testing.

#### A.2.2 DATASET EXAMPLES

We demonstrate one example for each dataset below.

## LogiQA

One seminar had 18 participants. It is known that :(1) At least 5 young teachers are female; (2) At least 6 female teachers are over middle age; (3) At least 7 young women are teachers; According to the above information, which can be concluded?

Options:

(A) Some young teachers are not women



(B) Some young women are not teachers

(C) There are at least 11 young teachers

(D) There are at least 13 female teachers

## BBH: logical deduction three objects

The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. In a golf tournament, there were three golfers: Ada, Mel, and Mya. Mya finished below Ada. Mel finished above Ada.

Options:

(A) Ada finished last





(B) Mel finished last

(C) Mya finished last

## GSM8k

A 40 meters rope was cut into 2 parts in the ratio of 2:3. How long is the shorter part?

## MATH: Algebra Level 1

 $ 361 + 2(19)(6) + 36 = x $ . Solve for x.

## ContextHub: Abstract - Level 2

(wqeq or mnze) → zkx.

(NOT ttjmx) → kottz.



(kottz or zkx) → pofk.

Given pofk is False, what is the value of ttjmx?



## LiveBench: reasoning - zebra puzzle

There are 3 people standing in a line numbered 1 through 3 in a left-to-right order.

Each person has a set of attributes: Nationality, Music-Genre, Transport.

The attributes have the following possible values:

- Nationality: spanish, argentine, canadian

- Music-Genre: punk, rock, reggae

- Transport: train, jet-ski, trike

and exactly one person in the line has a given value for an attribute.













Given the following premises about the line of people:

- the person who is argentine avoids getting on a train

- the person who is spanish is somewhere between the person who listens to punk and the person who listens to rock

- the person who listens to punk is not anywhere to the right of the person that travels by trike







- the person who listens to punk is on the immediate right of the person that travels by jet-ski

Answer the following question:

What is the nationality of the person who listens to rock? Return your answer as a single word, in the following format: ***X***, where X is the answer.

### A.3 TRAINING DETAILS

For self-training of TypedThinker, we use the splits in the original papers for LogiQA and follow the split of Toshniwal et al. (2024) for GSM8k and MATH. For BBH, we utilize 16 English multiple-choice tasks and randomly select 100 examples per task as the test set, with 20 examples as the hold-out validation set. The detailed statistics are listed in Table 9. Finally, the curated generation dataset covers 67.2% problems on the LogiQA benchmark, 69.7% on BBH, 74.88% on GSM8k, and 36.27% on MATH. We finetune a unified meta-thinker for both math and logical problems, and a unified reasoner for all reasoning types. We use Huggingface (Wolf et al., 2019) with deepspeed (Rasley et al., 2020). The finetuning is conducted on 2 A6000 GPUs. The batch size is 64 and the learning rate is  $ 1e^{-5} $ . The maximum epoch is 3 for the meta-thinker and 2 for the reasoner.

### A.4 ANALYSIS OF TYPED REASONING

Accuracy of Typed Reasoning We calculate the accuracy for each reasoning type on our empirical dataset  $ \overline{D} $ , shown in Figure 6. We can find that on LogiQA and MATH, the accuracy of different reasoning types is similar. However, deductive and analogical reasoning outperform the other two on BBH while inductive and abductive reasoning are more effective. The results illustrate that after our carefully designed demonstration for each reasoning type, LLM's capabilities in other reasoning types achieve comparable performance with deductive reasoning. This ensures the quality and the balance of our collected dataset on each reasoning type.

Comparing Figure 1 and Figure 6, we can see that if correctly selected, the specific reasoning type can enhance the model performance by handling problems that cannot be solved by other reasoning types, such as inductive on MATH. However, the unsuitable reasoning type can also mislead the model, leading to poor performance.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_337_757_878_1068.jpg" alt="Image" width="44%" /></div>


<div style="text-align: center;">Figure 6: Accuracy of the solutions on different reasoning types. It indicates that the effectiveness of reasoning types varies in different problems.</div>


Diversity of typed Reasoning To further verify whether the reasoning types can make the solutions more diverse, we compare the diversity between solutions under different sampling settings in Table 10. We use Levenshtein Distance (Levenshtein, 1966) and the n-gram overlaps between sentences to evaluate diversity. Specifically, for K generations  $ G = \{g_{1}, \cdots, g_{K}\} $  of the same problem, we calculate the distance between each pair and normalize them with the sentence length. Then the average distance over these paired results is used as the distance of these K generations. If we denote the normalized Levenshtein Distance function as  $ f_{ld} $ , this process can be represented as:

 $$ f_{\mathrm{l d}}(G)=\frac{2}{K(K-1)}\sum_{i=0}^{K}\sum_{j=i+1}^{K}f_{\mathrm{l d}}(g_{i},g_{j}). $$ 

The calculation of the n-gram overlap is defined in the same way. For each setting, we present the average score over the problems in the test set in Table 10. A larger Levenshtein distance and a smaller overlap indicate a more diverse solution set. The zero-shot setting does not include examples

in the prompt, and the zero-shot setting + types only include the definition of the reasoning type (as listed in Table 8). The few-shot setting has 5 examples, and the few-shot setting with types has different 6 examples for each type. For zero-shot / few-shot @5, we use repeated sampling with temperature = 1 for 5 times. For zero-shot / few-shot + 5 types, we sample one solution per reasoning type.

From Table 10, we can see that after adding the reasoning types, the diversity of both zero-shot and few-shot increases significantly. It indicates that the introduction of various reasoning types can make the LLM's reasoning more diverse. We can also find that in most cases, the few-shot with reasoning types has the highest diversity, while in BBH, the zero-shot setting can benefit more from the reasoning types.

<div style="text-align: center;">Table 10: Adding reasoning types can enhance diversity in both zero-shot and few-shot sampling settings. It can significantly increase the distance and reduce the n-gram overlaps between generations. For each setting, we use Mistral 7B to sample 5 solutions with temperature = 1. @5 indicates repeated sampling 5 times, +5 types indicates sampling one solution per reasoning type. The diversity is averaged over the whole test set.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Benchmark</td><td style='text-align: center;'>Sampling Setting</td><td style='text-align: center;'>Levenshtein Distance  $ \uparrow $</td><td style='text-align: center;'>Unigram overlap  $ \downarrow $</td><td style='text-align: center;'>4-gram overlap  $ \downarrow $</td></tr><tr><td rowspan="4">LogiQA</td><td style='text-align: center;'>Zero-shot @ 5</td><td style='text-align: center;'>0.304</td><td style='text-align: center;'>0.588</td><td style='text-align: center;'>0.537</td></tr><tr><td style='text-align: center;'>Zero-shot + 5 types</td><td style='text-align: center;'>0.600</td><td style='text-align: center;'>0.258</td><td style='text-align: center;'>0.186</td></tr><tr><td style='text-align: center;'>Few-shot @ 5</td><td style='text-align: center;'>0.573</td><td style='text-align: center;'>0.232</td><td style='text-align: center;'>0.123</td></tr><tr><td style='text-align: center;'>Few-shot + 5 types</td><td style='text-align: center;'>0.644</td><td style='text-align: center;'>0.175</td><td style='text-align: center;'>0.077</td></tr><tr><td rowspan="4">BBH</td><td style='text-align: center;'>Zero-shot @ 5</td><td style='text-align: center;'>0.517</td><td style='text-align: center;'>0.310</td><td style='text-align: center;'>0.216</td></tr><tr><td style='text-align: center;'>Zero-shot + 5 types</td><td style='text-align: center;'>0.712</td><td style='text-align: center;'>0.128</td><td style='text-align: center;'>0.050</td></tr><tr><td style='text-align: center;'>Few-shot @ 5</td><td style='text-align: center;'>0.599</td><td style='text-align: center;'>0.224</td><td style='text-align: center;'>0.119</td></tr><tr><td style='text-align: center;'>Few-shot + 5 types</td><td style='text-align: center;'>0.650</td><td style='text-align: center;'>0.175</td><td style='text-align: center;'>0.076</td></tr><tr><td rowspan="4">GSM8k</td><td style='text-align: center;'>Zero-shot @ 5</td><td style='text-align: center;'>0.624</td><td style='text-align: center;'>0.195</td><td style='text-align: center;'>0.091</td></tr><tr><td style='text-align: center;'>Zero-shot + 5 types</td><td style='text-align: center;'>0.683</td><td style='text-align: center;'>0.151</td><td style='text-align: center;'>0.054</td></tr><tr><td style='text-align: center;'>Few-shot @ 5</td><td style='text-align: center;'>0.498</td><td style='text-align: center;'>0.312</td><td style='text-align: center;'>0.176</td></tr><tr><td style='text-align: center;'>Few-shot + 5 types</td><td style='text-align: center;'>0.710</td><td style='text-align: center;'>0.137</td><td style='text-align: center;'>0.048</td></tr><tr><td rowspan="4">MATH</td><td style='text-align: center;'>Zero-shot</td><td style='text-align: center;'>0.673</td><td style='text-align: center;'>0.157</td><td style='text-align: center;'>0.070</td></tr><tr><td style='text-align: center;'>Zero-shot + 5 types</td><td style='text-align: center;'>0.729</td><td style='text-align: center;'>0.112</td><td style='text-align: center;'>0.035</td></tr><tr><td style='text-align: center;'>Few-shot</td><td style='text-align: center;'>0.659</td><td style='text-align: center;'>0.174</td><td style='text-align: center;'>0.080</td></tr><tr><td style='text-align: center;'>Few-shot + 5 types</td><td style='text-align: center;'>0.732</td><td style='text-align: center;'>0.115</td><td style='text-align: center;'>0.038</td></tr></table>

### A.5 MORE EXPERIMENTAL RESULTS

#### A.5.1 Results on More Backbone LLMs

We conducted further experiments using Qwen 2-7B-Instruct (Bai et al., 2023) as our backbone LLM. The Qwen series of open-source large language models have demonstrated comparable or even superior performance to the Mistral and LLaMA families across multiple tasks. The results are shown in Table 11. Our method achieves approximately 7% improvement over the few-shot baseline in both single-generation and majority-vote settings (+SC @5). These results demonstrate that TypedThinker is a general and effective method for enhancing the reasoning capabilities of various LLMs.

<div style="text-align: center;">Table 11: Qwen 2-7B-Instruct results. The annotations are the same with Table 1.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>LogiQA</td><td style='text-align: center;'>BBH</td><td style='text-align: center;'>GSM8K</td><td style='text-align: center;'>MATH</td><td style='text-align: center;'>Avg.</td></tr><tr><td style='text-align: center;'>Few-shot</td><td style='text-align: center;'>0.552</td><td style='text-align: center;'>0.471</td><td style='text-align: center;'>0.646</td><td style='text-align: center;'>0.417</td><td style='text-align: center;'>0.521</td></tr><tr><td style='text-align: center;'>+ SC @ 5</td><td style='text-align: center;'>0.579</td><td style='text-align: center;'>0.554</td><td style='text-align: center;'>0.763</td><td style='text-align: center;'>0.497</td><td style='text-align: center;'>0.598</td></tr><tr><td style='text-align: center;'>CoT Selection</td><td style='text-align: center;'>0.554</td><td style='text-align: center;'>0.516</td><td style='text-align: center;'>0.772</td><td style='text-align: center;'>0.451</td><td style='text-align: center;'>0.573</td></tr><tr><td style='text-align: center;'>+ SC @ 5</td><td style='text-align: center;'>0.560</td><td style='text-align: center;'>0.528</td><td style='text-align: center;'>0.780</td><td style='text-align: center;'>0.497</td><td style='text-align: center;'>0.591</td></tr><tr><td style='text-align: center;'>Zeroshot MoR</td><td style='text-align: center;'>0.573</td><td style='text-align: center;'>0.498</td><td style='text-align: center;'>0.492</td><td style='text-align: center;'>0.407</td><td style='text-align: center;'>0.493</td></tr><tr><td style='text-align: center;'>Fewshot MoR</td><td style='text-align: center;'>0.589</td><td style='text-align: center;'>0.581</td><td style='text-align: center;'>0.889</td><td style='text-align: center;'>0.551</td><td style='text-align: center;'>0.652</td></tr><tr><td style='text-align: center;'>TypedThinker</td><td style='text-align: center;'>0.595</td><td style='text-align: center;'>0.534</td><td style='text-align: center;'>0.779</td><td style='text-align: center;'>0.474</td><td style='text-align: center;'>0.596</td></tr><tr><td style='text-align: center;'>+ SC @ 5</td><td style='text-align: center;'>0.644</td><td style='text-align: center;'>0.584</td><td style='text-align: center;'>0.880</td><td style='text-align: center;'>0.565</td><td style='text-align: center;'>0.668</td></tr></table>

<div style="text-align: center;">Table 12: TypedThinker outperforms other baselines on LiveBench without extra finetuning. Here the results are based on the majority vote over 5 responses (+SC @5).</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Mistral 7B</td><td style='text-align: center;'>LLaMA3 8B</td></tr><tr><td style='text-align: center;'>Few-shot</td><td style='text-align: center;'>0.178</td><td style='text-align: center;'>0.200</td></tr><tr><td style='text-align: center;'>CoT Selection</td><td style='text-align: center;'>0.244</td><td style='text-align: center;'>0.200</td></tr><tr><td style='text-align: center;'>TypedThinker</td><td style='text-align: center;'>0.267</td><td style='text-align: center;'>0.267</td></tr></table>

#### A.5.2 Results on more benchmarks

We further evaluate our methods on LiveBench (White et al., 2024) to test the generalization capability of our method. The experimental settings are the same as described in Section 4.5. The results are shown in Table 12. It demonstrates that TypedThinker outperforms other baselines, further supporting its generalization capability across diverse tasks.

#### A.5.3 MORE ABLATION STUDIES

The primary reason for comparing our method with the few-shot baseline is that fine-tuning for specific reasoning types is an integral part of our approach. Therefore, we evaluate the impact of our fine-tuned reasoner through a separate ablation study. However, comparing TypedThinker to few-shot baselines without fine-tuning may not fully account for the benefits of fine-tuning. Therefore, we conduct two more experiments to verify the influence of the fine-tuned LLMs.

Comparison with base LLM + one module Ablation studies in Section 4.4 investigate the contribution of each component by removing one component each time. Here we provide additional ablation results by adding one component to the base LLM each time, resulting in two variants: Base LLM + Meta-thinker and Base LLM + Collection. For a more reliable conclusion, we ran experiments three times to calculate the average and std and present the result in Table 13 and 14.

<div style="text-align: center;">Table 13: Mistral 7B results are based on three repetitive experiments. Avg. indicates the average accuracy over four benchmarks.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>LogiQA</td><td style='text-align: center;'>BBH</td><td style='text-align: center;'>GSM8K</td><td style='text-align: center;'>MATH</td><td style='text-align: center;'>Avg.</td></tr><tr><td style='text-align: center;'>Few-shot</td><td style='text-align: center;'>0.493 ± 0.007</td><td style='text-align: center;'>0.347 ± 0.01</td><td style='text-align: center;'>0.372 ± 0.014</td><td style='text-align: center;'>0.071 ± 0.003</td><td style='text-align: center;'>0.321 ± 0.006</td></tr><tr><td style='text-align: center;'>CoT Selection</td><td style='text-align: center;'>0.475 ± 0.009</td><td style='text-align: center;'>0.361 ± 0.01</td><td style='text-align: center;'>0.377 ± 0.011</td><td style='text-align: center;'>0.104 ± 0.008</td><td style='text-align: center;'>0.329 ± 0.004</td></tr><tr><td style='text-align: center;'>LLM + Meta Thinker</td><td style='text-align: center;'>0.512 ± 0.003</td><td style='text-align: center;'>0.377 ± 0.006</td><td style='text-align: center;'>0.379 ± 0.004</td><td style='text-align: center;'>0.106 ± 0.009</td><td style='text-align: center;'>0.343 ± 0.004</td></tr><tr><td style='text-align: center;'>LLM + Collection</td><td style='text-align: center;'>0.519 ± 0.007</td><td style='text-align: center;'>0.398 ± 0.005</td><td style='text-align: center;'>0.363 ± 0.004</td><td style='text-align: center;'>0.086 ± 0.008</td><td style='text-align: center;'>0.342 ± 0.001</td></tr><tr><td style='text-align: center;'>TypedThinker</td><td style='text-align: center;'>0.553 ± 0.004</td><td style='text-align: center;'>0.430 ± 0.008</td><td style='text-align: center;'>0.390 ± 0.012</td><td style='text-align: center;'>0.103 ± 0.01</td><td style='text-align: center;'>0.369 ± 0.006</td></tr></table>

<div style="text-align: center;">Table 14: LLaMA 3 8B results are based on three repetitive experiments. Avg. indicates the average accuracy over four benchmarks.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>LogiQA</td><td style='text-align: center;'>BBH</td><td style='text-align: center;'>GSM8K</td><td style='text-align: center;'>MATH</td><td style='text-align: center;'>Avg.</td></tr><tr><td style='text-align: center;'>Few-shot</td><td style='text-align: center;'>0.569 ± 0.003</td><td style='text-align: center;'>0.319 ± 0.006</td><td style='text-align: center;'>0.476 ± 0.004</td><td style='text-align: center;'>0.102 ± 0.005</td><td style='text-align: center;'>0.366 ± 0.001</td></tr><tr><td style='text-align: center;'>CoT Selection</td><td style='text-align: center;'>0.558 ± 0.011</td><td style='text-align: center;'>0.376 ± 0.007</td><td style='text-align: center;'>0.360 ± 0.006</td><td style='text-align: center;'>0.104 ± 0.009</td><td style='text-align: center;'>0.349 ± 0.007</td></tr><tr><td style='text-align: center;'>LLM + Meta Thinker</td><td style='text-align: center;'>0.538 ± 0.005</td><td style='text-align: center;'>0.434 ± 0.005</td><td style='text-align: center;'>0.508 ± 0.005</td><td style='text-align: center;'>0.118 ± 0.005</td><td style='text-align: center;'>0.400 ± 0.001</td></tr><tr><td style='text-align: center;'>LLM + Collection</td><td style='text-align: center;'>0.574 ± 0.011</td><td style='text-align: center;'>0.497 ± 0.004</td><td style='text-align: center;'>0.438 ± 0.005</td><td style='text-align: center;'>0.109 ± 0.006</td><td style='text-align: center;'>0.404 ± 0.001</td></tr><tr><td style='text-align: center;'>TypedThinker</td><td style='text-align: center;'>0.546 ± 0.004</td><td style='text-align: center;'>0.534 ± 0.003</td><td style='text-align: center;'>0.535 ± 0.001</td><td style='text-align: center;'>0.203 ± 0.009</td><td style='text-align: center;'>0.455 ± 0.002</td></tr></table>

Results show that the retrieval component improves performance on logical tasks but may mislead models on mathematical datasets. This is consistent with our findings in the ablation study in Table 2: the retrieved solutions with digits may mislead the model. Meanwhile, compared with the ICL reasoner, our finetuned reasoner shows better capability in identifying the suitable reasoning type.

### A.6 DISCUSSION ON MORE REASONING PROBLEMS

In this paper, we mainly focus on logical and math reasoning problems. However, our TypedThinker can also be extended to symbolic or commonsense reasoning without extra ef-

<div style="text-align: center;">Table 15: TypedThinker performs best on the abstract category of Contexthub. Here the results are based on the majority vote over 5 responses (+SC @5).</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Mistral 7B</td><td style='text-align: center;'>LLaMA3 8B</td></tr><tr><td style='text-align: center;'>Few-shot</td><td style='text-align: center;'>0.25</td><td style='text-align: center;'>0.35</td></tr><tr><td style='text-align: center;'>CoT Selection</td><td style='text-align: center;'>0.55</td><td style='text-align: center;'>0.40</td></tr><tr><td style='text-align: center;'>Zero-shot MoR</td><td style='text-align: center;'>0.55</td><td style='text-align: center;'>0.35</td></tr><tr><td style='text-align: center;'>Few-shot MoR</td><td style='text-align: center;'>0.45</td><td style='text-align: center;'>0.45</td></tr><tr><td style='text-align: center;'>Self-Discover</td><td style='text-align: center;'>0.25</td><td style='text-align: center;'>0.55</td></tr><tr><td style='text-align: center;'>TypedThinker</td><td style='text-align: center;'>0.75</td><td style='text-align: center;'>0.55</td></tr></table>

forts. For example, the problems under abstract category in ContextHub can be viewed as symbolic reasoning, as shown in A.2.2. Therefore, in addition to the overall performance shown in Table 4, we present specific performance on the abstract category of symbolic reasoning in Table 15. As we can see, TypedThinker outperforms baseline methods, even without further fine-tuning the meta-thinker and reasoner for this symbolic reasoning task.

While the LogiQA dataset contains problems requiring commonsense reasoning, the dataset lacks explicit annotations (such as a specific category) for such tasks. For example, here is one case that requires commonsense knowledge about manufacturing costs, market dynamics, and consumer preferences.

## LogiQA: A commonsense reasoning example

Traditionally, the most highly sought cars have been the sports cars and similar two-door models. Nevertheless, Zincstone Motors has chosen to eliminate the last two-door models and produce only four-door models. Which of the following would, if true, most help to explain Zincstone Motors' strategy?

(A) In almost every instance, Zincstone Motors models lead all comparable models of competitors in fuel efficiency and have lower average maintenance costs as well.

(B) After a spate of recent additional safety requirements, the cost of frame and doors of Zincstone Motors' standard two-door models are now three times as expensive as standard four-door frame and doors.

(C) Many of Zincstone Motors models are exported and sold overseas, including in some countries like Japan, which import a significant number of cars into the United States.

(D) As American consumers lose access to car manufacturers who produce two-door cars, and as two-door cars occupy smaller and smaller shares of the United States car market, American consumers' tastes tend to shift from two-door cars.

For reasoning problems that significantly differ from the existing domains (logic and math), additional demonstrations tailored to the task are recommended to guide reasoning. For example, inductive reasoning might involve deriving a general program from exemplar input-output test cases in code generation (Shao et al., 2024; Yang et al., 2024). In such cases, providing a few task-specific examples or conducting light fine-tuning on the reasoner can further enhance performance.

### A.7 IMPACT STATEMENT

This work will enhance current LLMs with better reasoning capability, which can make them more useful in problem-solving. There might be some potential societal consequences of our work, none of which we feel must be specifically highlighted here. However, it might be misused as we release all our code and data for reproduction. We will try our best to avoid the potential misuse.

<div style="text-align: center;">Table 16: Examples of four reasoning types.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Type</td><td style='text-align: center;'>Query</td><td style='text-align: center;'>Example</td></tr><tr><td style='text-align: center;'>Deduction</td><td style='text-align: center;'>Alice, Bob, and Claire are dancers at a square dance. At the start of a song, they each have a partner: Alice is with Lola, Bob is with Rodrigo, and Claire is with Patrick. Throughout the song, Alice and Bob switch partners; Claire and Bob switch; Finally, Bob and Alice switch. At the end of the dance, Alice is dancing with Options: (A) Lola (B) Rodrigo (C) Patrick</td><td style='text-align: center;'>(0) At the start: Alice: Lola, Bob: Rodrigo, Claire: Patrick.(1) Alice and Bob switch: Alice: Rodrigo, Bob: Lola, Claire: Patrick. (2) Claire and Bob switch: Alice: Rodrigo, Bob: Patrick, Claire: Lola. (3) Bob and Alice switch: Alice: Patrick, Bob: Rodrigo, Claire: Lola. At the end of the dance, Alice is dancing with Patrick. So the answer is  $ \left[\frac{(C)}{(A)}\right] $ .</td></tr><tr><td style='text-align: center;'>Induction</td><td style='text-align: center;'>Students who told a lie overestimated how many people could detect it. Volleyball players performing poorly thought teammates noticed more than they actually did. A student wearing a funny T-shirt expected everyone to notice, but only a few did. Which option best illustrates the statements above? (A) People overestimate how often others notice their appearance and behavior. (B) People rarely notice the appearance or behavior of others. (C) We are less observant of others&#x27; appearance and behavior than we think. (D) People are less aware of their appearance and behavior than others are.</td><td style='text-align: center;'>By examining each of the scenarios mentioned, we notice all these examples indicate that individuals overestimate the level and extent of attention their actions, appearances, or behaviors receive from others. This consistency across different contexts illustrates a broader psychological phenomenon. So the answer is  $ \left[\frac{(A)}{(A)}\right] $ .</td></tr><tr><td style='text-align: center;'>Abduction</td><td style='text-align: center;'>The integer m is between 30 and 80 and is a multiple of 6. When m is divided by 8, the remainder is 2. Similarly, when m is divided by 5, the remainder is 2. What is the value of m?</td><td style='text-align: center;'>To solve this problem using abductive reasoning, we assume that one possible value of m exists that abides by the constraints and check if this assumption holds.1. Fi rst, filter values of m that are multiples of 6 between 30 and 80. 2. Next, apply the condition that when m is divided by 8, the remainder is 2. Only 42, 66 fit this condition.3. Apply the third condition, that when divided by 5, m should leave a remainder of 2. Testing the applicable values so far and find 42 meets the requirement. So the answer is  $ \left[\frac{42}{42}\right] $ .</td></tr><tr><td style='text-align: center;'>Analogy</td><td style='text-align: center;'>John is 24 years younger than his dad. The sum of their ages is 68 years. How many years old is John?</td><td style='text-align: center;'>Retrieval: Question: Lisa is 10 years younger than her mom. The sum of their ages is 70 years. How old is Lisa? Answer: Lisa is 30 years old and her mom is 40 years old. These are solved using the same approach as the problem about John and his dad&#x27;s ages, i.e., setting up two equations based on the information given and then solving for the two variables representing the ages. Therefore, for the given question, John is  $ \left[\frac{22}{22}\right] $  years old.</td></tr></table>