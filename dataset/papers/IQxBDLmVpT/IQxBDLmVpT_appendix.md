## APPENDIX

## A LIMITATIONS

While this paper provides valuable insights into how large-scale pretraining corpus contributes to the emergent abilities of LLMs through n-gram search, there are a few limitations that we want to list out. First, the model we use, Pythia (Biderman et al., 2023), and the pretraining corpus, Pile (Gao et al., 2020), are slightly outdated and have been outperformed by many new open-source LLMs. However, most open-source LLMs lack corresponding pretraining data and have limited model sizes and pre-training checkpoints, hindering scaling effect studies. The current WIMBD system also has limitations in searching larger corpora like Dolma (Soldaini et al., 2024a) (3T tokens), calling for improved searching and retrieval methods. The quality of task-relevant n-gram pairs is highly sensitive to the filtering method, and while the current embedding similarity-based approach is effective, better filtering methods could significantly enhance the analysis, which is left for future research.

## B ETHICS STATEMENT

The insights and methodologies developed in this paper have several significant implications for the broader field of artificial intelligence, particularly in the development and deployment of large language models (LLMs). Understanding the balance between memorization and generalization within LLMs is crucial for both advancing the theoretical foundation of machine learning and addressing practical concerns related to their use.

Enhanced Model Interpretability: By extending the definition of memorization and examining how LLMs utilize their pretraining data, our research contributes to a deeper understanding of the internal mechanics of these models. This improved interpretability can help researchers and practitioners diagnose and mitigate issues related to data bias, model robustness, and unexpected behaviors in AI systems.

Privacy and Security Considerations: Our findings have direct implications for privacy and security in AI. Demonstrating how LLMs memorize and potentially recall training data underscores the need for rigorous data handling and anonymization techniques. It raises awareness about the risks of inadvertent leakage of sensitive information, thereby informing policy and best practices for data usage in training large models.

Economic and Societal Impact: As LLMs become more integral to various industries, understanding their capabilities and limitations can have significant economic and societal implications. Our research can help businesses and policymakers make informed decisions about deploying these models, ensuring they are used ethically and effectively. This, in turn, can lead to more reliable and trustworthy AI systems, fostering greater public trust and acceptance.

## C RELATED WORK DETAILS

Understanding LLMs' capabilities from training data Prystawski et al. (2023) and Wang et al. (2024) discuss how the reasoning ability of language models is a consequence of their pretraining data. Prystawski et al. (2023) discuss how chain-of-thought reasoning is effective in autoregressive language models because of local structure within pretraining data, and Wang et al. (2024) derive novel conclusions from known facts by aggregating reasoning paths seen in pretraining data. On the other hand, Xie et al. (2022) and Wang et al. (2023) discuss how in-context learning is a by-product of learning the pretraining data distribution. They both suggest that language models learn to implicitly infer a latent variable from the given prompt, as the pretraining data is generated from some unknown latent variable. Additionally, Chan et al. (2022) propose that the distributional properties of training data drive emergent in-context learning behaviors in large language models, whereas Razeghi et al. (2023) show the influence the pretraining data on the mathematical abilities of LLMs. Chen et al. (2024) also highlight the significance of parallel structures in pretraining data for the emergence of in-context learning.

However, the small-scale nature of such analysis is antithetical to the commonly believed main driving factor behind the performance of LLMs: scaling. Recently, Kirchenbauer et al. (2024) proposes to provide statistical evidence of the dependence of a target model capabilities on subsets of its training data, by estimating the data distribution with an embedding-induced kernel. However, their estimation is based on a very small portion of the pretraining data (around 0.3%) as computing the embeddings of a huge dataset is very non-trivial. To get a better estimation of the whole distribution of the pretraining data, Elazar et al. (2024) construct a retrieval system, WIMBD, that can efficiently search n-gram phrases over hundreds and thousands of GBs of pretraining data. Merrill et al. (2024) propose an efficient data structure that enables unbounded-length n-gram searches in massive pretraining datasets, and find that larger LLMs generate less novel large n-grams compared to human written texts, and (Shaib et al., 2024) shows how syntactic templates from the training data cause models to re-use such templates after training. Liu et al. (2024) proposes to build a prefix-based efficient retrieval system for pretraining corpora, and then construct large-scale  $ \infty $ -gram language models to estimate the distribution of text corpora. However, such distribution only models local contextual dependency, which might not be useful for understanding complex LLM capabilities.

New methods and analysis to investigate these capabilities at scale and to understand the role of scaling are needed to obtain useful insights into real-world LLMs. In this work, we aim to provide an in-depth analysis of the origin of the general zero-shot capabilities of LLMs, by performing full searches across the whole pretraining corpus with the WIMBD and  $ \infty $ -gram framework.

Memorization v.s. generalization The phenomenon of machine learning models being able to perfectly memorize the training data has been studied in many previous works. Most of them define LLM memorization as exactly recalling the training examples by designed prompting, including the memorization of rare long-tail data, like private information (Zhang et al., 2023), and the contamination of testing sets (Jiang et al., 2024). Carlini et al. (2022) found that the exact copy and pasting behaviors are more prevalent in larger LMs. Hartmann et al. (2023) provides a comprehensive summarization of LLM memorization.

Several papers have studied the interplay between memorization and generalization of training data. Feldman (2020) prove that memorizing the training data is in fact required for optimal generalization on testing data. Works along this line (Feldman and Zhang, 2020; Zhang et al., 2023) extend the original definition of memorization by quantifying the extent of memorizing a training example with the performance difference when including and excluding this specific example in training data. However, this definition is impractical for large-scale analysis of LLMs as it requires retraining an LLM from scratch to analyze one data point. In this paper, we propose a new definition of distributional memorization by using n-gram counts, which is more suitable for large-scale analysis with LLMs.

## D EXPERIMENT DETAILS

Dataset choice The choice of dataset comes from the consideration of balancing different types of tasks, and we believe the combination of translation + factual QA + reasoning can well represent the spectrum of possible tasks. To enhance the selection of reasoning tasks, we also add GSM8K as a new task as described in point 1 of the general response. We found both the performance v.s. n-gram count results and the distributional generalization results on GSM8K align with our previous findings.

Embedding model choice The choice of embedding model comes from accommodating the need for different tasks. We use the LASER embedding model for the Translation task because it is specialized for translation. We use E5 for other tasks because it is a general-purpose sentence embedding model that performs cosine-similarity-based retrieval.

Task-gram table construction To build a task-gram table, we use a two-step process. First, we filter all possible input-output n-gram pairs in a supervised dataset by the cosine similarity between their embeddings. Second, we search through the pretraining corpus and keep n-gram pairs that have nonzero counts.

For WMT, we use the Europarl (Koehn, 2005) parallel corpus to construct a large task-gram table. For TriviaQA, we use its training set, while for MMLU, we use the testing set as there is no training set available. We employ two different models to filter the n-grams pairs. For translation tasks, we

use the LASER embeddings (Schwenk and Douze, 2017), which provide language-agnostic sentence representations, in order to assess cross-lingual similarity. We mine n-gram pairs from the Europarl corpus, which consists of around 20 million parallel sentences extracted from the proceedings of the European Parliament (Koehn, 2005). For all other tasks, we use E5 (Wang et al., 2022), which are multilingual sentence representations trained using contrastive learning on a diverse range of tasks. For TriviaQA, we mine the n-gram pairs from TriviaQA's training set. For MMLU, since the training set is very small (100 – 500 examples for each task), we mine the n-gram pairs from the test sets directly. For GSM8K, due to the short time limitation during rebuttal, we also mine the n-gram pairs from the 1K test sets directly.

In WMT, the cosine similarity thresholds we use are: 0.85, 0.8, 0.75, and 0.7 for 2 to 5-gram pairs, respectively; For TriviaQA and MMLU, the values are 0.75 and 0.65 for 3 and 5-gram pairs, respectively. We use lower thresholds for larger n-grams because larger n-grams inherently impose stricter alignment, and are therefore less likely. We show a cosine similarity sensitivity ablation in Figure 6.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_511_481_727.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_481_509_747_728.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_749_510_1009_728.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">Figure 6: Visualization of distributional memorization with different-sized Pythia models on TriviaQA with different cosine similarity threshold  $ \gamma\in\{0.7,0.75,0.8\} $ . The trend of distributional memorization does not change with different thresholds.</div>


We perform our experiments on 8 GPU 40G A100 working stations. Below is the license information for the datasets we used:

• Pile: MIT license. URL: https://github.com/EleutherAI/the-pile/tree/master

• Tulu: ODC-BY license. URL: https://huggingface.co/datasets/allenai/tulu-v2-sft-mixture

- WMT-09: published with the WMT workshop. URL: https://www.statmt.org/wmt09/translation-task.html

• TriviaQA: Apache License 2.0. URL: https://nlp.cs.washington.edu/triviaqa/

• MMLU: MIT license. URL: https://github.com/hendrycks/test

Knowledge-intensive MMLU tasks:

'prehistory', 'business_ethics', 'philosophy',
'moral_disputes', 'medical_genetics', 'high_school_government_and_politics',
'human_aging', 'us_foreign_policy', 'high_school_macroeconomics',
'logical_fallacies', 'international_law', 'computer_security',
'sociology', 'professional_psychology', 'marketing', 'human_sexuality',
'anatomy', 'high_school_us_history', 'public_relations',
'high_school_microeconomics', 'clinical_knowledge', 'security_studies',
'nutrition', 'world_religions', 'high_school_psychology',
'high_school_geography', 'management', 'global_facts',
'high_school_world_history', 'high_school_european_history',
'jurisprudence', 'virology', 'astronomy', 'miscellaneous'

Reasoning-intensive MMLU tasks:

'econometrics', 'professional_law', 'abstract_algebra', 'college_medicine', 'college_chemistry', 'moral_scenarios', 'college_mathematics', 'high_school_chemistry', 'professional_accounting', 'college_computer_science', 'college_biology', 'high_school_computer_science', 'high_school_mathematics', 'college_physics', 'professional_medicine', 'elementary_mathematics', 'machine_learning', 'electrical_engineering', 'high_school_physics', 'conceptual_physics', 'high_school_statistics', 'high_school_biology'

<div style="text-align: center;"><img src="imgs/img_in_chart_box_392_405_806_763.jpg" alt="Image" width="33%" /></div>


<div style="text-align: center;">Figure 7: GSM8K accuracy v.s. n-gram pair count in Dolma with different OLMo model sizes.</div>


## E PROMPT OPTIMIZATION DETAILS


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Corpus</td><td style='text-align: center;'>Dataset</td><td style='text-align: center;'>Memorization</td><td style='text-align: center;'>Generalization</td></tr><tr><td rowspan="2">Pile</td><td style='text-align: center;'>TriviaQA</td><td style='text-align: center;'>Deliver an exact single word or concise phrase in response to the factual question. (avg. 3-gram count: 557948.5)</td><td style='text-align: center;'>Formulate a distinctive and concise term or phrase to clearly answer the factual question. (avg. 3-gram count: 1714.9)</td></tr><tr><td style='text-align: center;'>GSM8K</td><td style='text-align: center;'>Carefully analyze each math word problem presented, break it down step-by-step, and clearly state the final answer. (avg. 3-gram count: 45028.9)</td><td style='text-align: center;'>Dissect each math word problem into straightforward, logical steps; solve each part systematically for precise solutions. (avg. 3-gram count: 43.1)</td></tr><tr><td rowspan="2">Dolma</td><td style='text-align: center;'>TriviaQA</td><td style='text-align: center;'>Provide a single word or concise phrase in response to the following factual question. (Avg. 3-gram count: 5566475.5)</td><td style='text-align: center;'>Provide a clear and specific word or brief phrase in response to the factual question below. (Avg. 3-gram count: 43436.2)</td></tr><tr><td style='text-align: center;'>GSM8K</td><td style='text-align: center;'>Solve the following math word problem by methodically breaking it down into simple, clear steps to find the correct solution efficiently. (Avg. 3-gram count: 3736735.0)</td><td style='text-align: center;'>Solve the upcoming math word problem by sequentially explaining each calculation and logical step, ensuring clarity and coherence in your solution. (Avg. 3-gram count: 4563.6)</td></tr></table>

<div style="text-align: center;">Table 2: Prompts optimized for memorization and generalization for TriviaQA and GSM8K.</div>


Text in the square brackets are comments that are not involved in the actual prompt.

## Meta prompt for prompt optimization

**Task Description**:
You are tasked with optimizing a given prompt to guide an open-source language model (LM) in completing a specific task effectively. You will receive:
- The current prompt for the task.
- Its corresponding memorization score (Average frequency of task-related n-grams found in the LM's pretraining corpus).
- A few example input-output pairs illustrating the intended task.
- A history of previous prompt optimization iterations.
**Optimization Goals**:
Clearly describe the intended task with a general instruction that effectively guides the LM to perform the task.

[If trying to encourage reasoning ...]
Minimizing the memorization score of the updated prompt. The memorization score reflects the distributional correlation between the prompt and the LM's pretraining corpus. A lower score encourages the LM to generate more novel outputs.

[If trying to encourage memorization ...]
Maximize the memorization score of the updated prompt. The memorization score reflects the distributional correlation between the prompt and the LM's pretraining corpus. A higher score encourages better alignment with the LM's learned knowledge.
**Example task input-output pairs**:
[Example input-output pairs for current task]
**Output**:
Produce an updated prompt that balances clarity of task instruction with an lower memorization score.

## F n-GRAM PAIR EXAMPLES

In this section, we present some representative examples collected from the analysis for the different tasks evaluating, including Translation and Question-Answering (MMLU, TriviaQA). In order to show examples from the different experiments, we show examples with different model sizes and number of n-grams.

### G MODEL GENERATIONS VS. n-GRAM FREQUENCY IN PRETRAINING

To further analyze the behavior of a model based on the pairs found in pretraining, we compared the alignment of the generations of the model when that n-gram was generated, vs. n-gram frequency in the pretraining corpus. We found that as the n-gram frequency increased, the model n-gram pairs the model generated were more aligned.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Source</td><td style='text-align: center;'>(English , Spanish)</td><td style='text-align: center;'>Result</td></tr><tr><td colspan="3">Pythia 12b - 4gram</td></tr><tr><td style='text-align: center;'>The reaction of the market to the results of the vote in the American House of Representatives, which refused to support the plan for the stabilization of the financial sector there, has manifested itself here as well.</td><td style='text-align: center;'>(of the vote in, de la votación en) (the vote in the, la votación en el) (results of the vote, resultados de la votación) (to the results of, ante los resultados de) (has manifested itself here, se ha manifestado aquí) (market to the results, mercado ante los resultados) (plan for the stabilization, plan para la estabilización) (stabilization of the financial, estabilización del sector financiero) (reaction of the market, La reacción del mercado)</td><td style='text-align: center;'>La reacción del mercado ante los resultados de la votación en el Congreso de los Estados Unidos, que rechazó el plan para la estabilización del sector financiero allí, se ha manifestado aquí también.</td></tr><tr><td colspan="3">Pythia 410m - 4gram</td></tr><tr><td style='text-align: center;'>The reaction of the market to the results of the vote in the American House of Representatives, which refused to support the plan for the stabilization of the financial sector there, has manifested itself here as well.</td><td style='text-align: center;'>(of the vote in, de la votación en) (the vote in the, la votación en el) (results of the vote, resultados de la votación) (has manifested itself here, se ha manifestado aquí)</td><td style='text-align: center;'>&#x27;El resultado de la votación en la Cámara de Diputados, que no aceptó la propuesta de la Comisión de Desarrollo Regional para la Seguridad Social, ha manifestado aquí también.&#x27;</td></tr></table>

<div style="text-align: center;">Figure 8: Examples of mined pairs for the translation Task (English to Spanish) using Pythia Models with 4-Gram analysis. Models evaluated include those with 12 billion and 410 million parameters.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Source</td><td style='text-align: center;'>(English, Spanish)</td><td style='text-align: center;'>Result</td></tr><tr><td colspan="3">Pythia 12b - 2gram</td></tr><tr><td style='text-align: center;'>The new law is also meant to constrain the &quot;golden parachutes&quot; for the heads of the companies participating in the program and establishes a committee which will oversee the fund.</td><td style='text-align: center;'>(new law, nueva ley)(companies participating, empresas participan)(program establishes, programa establece)(committee oversee, comité supervisará)(establishes committee, establece comité)(oversee fund, supervisará fondo)(heads companies, jefes empresas)(participating program, participan programa)</td><td style='text-align: center;'>La nueva ley también pretende limitar los &quot;bonos de la parrilla&quot; para los jefes de las empresas que participan en el programa y establece un comité que supervisará el fondo.</td></tr><tr><td colspan="3">Pythia 410m - 2gram</td></tr><tr><td style='text-align: center;'>The new law is also meant to constrain the &quot;golden parachutes&quot; for the heads of the companies participating in the program and establishes a committee which will oversee the fund.</td><td style='text-align: center;'>(new law, nueva ley)(companies participating, empresas participan)(program establishes, programa establece)(establishes committee, establece comité)(participating program, participan programa)</td><td style='text-align: center;'>El programa de la nueva ley es también para limitar los &quot;champios de la empresa&quot; para los que participan en el programa y establece un comité que se encargará de gestionar el fondo.</td></tr></table>

<div style="text-align: center;">Figure 9: Examples of mined pairs for the translation Task (English to Spanish) using Pythia Models with 2-Gram analysis. Models evaluated include those with 12 billion and 410 million parameters.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Question</td><td style='text-align: center;'>Choices</td><td style='text-align: center;'>(Key, Value)</td><td style='text-align: center;'>Answer</td><td style='text-align: center;'>Category</td></tr><tr><td colspan="5">Pythia 12b - 3gram</td></tr><tr><td style='text-align: center;'>These are chains of organizations that are concerned with the management of the processes and activities involved in creating and moving products from producers and manufacturers to end-user customers. The organizations involved with any one journey, are collectively termed as a:</td><td style='text-align: center;'>(a) Distribution(b) Vendor(c) Communication(d) Selling</td><td style='text-align: center;'>(management processes activities, Distribution)(organizations involved one, Communication)(management processes activities, Communication)(activities involved creating, Communication)(processes activities involved, Communication)</td><td style='text-align: center;'>(a) Distribution</td><td style='text-align: center;'>marketing</td></tr><tr><td style='text-align: center;'>A 2-month-old female is brought to the office for her first routine health maintenance examination and for her immunization update. In order to determine whether or not any contraindications exist for diphtheria, tetanus, pertussis (DtaP) immunization, the parents should be questioned regarding</td><td style='text-align: center;'>(a) allergy to eggs(b) Apgar scores at birth(c) gestational age at birth(d) previous seizures</td><td style='text-align: center;'>(diphtheria tetanus pertussis, previous seizures)(tetanus pertussis dtap, allergy to eggs)(routine health maintenance, gestational age at birth)(pertussis dtap immunization, allergy to eggs)</td><td style='text-align: center;'>(d) previous seizures</td><td style='text-align: center;'>professional medicine</td></tr><tr><td colspan="5">Pythia 6.9 - 5gram</td></tr><tr><td style='text-align: center;'>&#x27;In which of the following Asian countries would one find special economic zones (SEZs)?&#x27;</td><td style='text-align: center;'>(a) Japan(b) South Korea(c) China(d) Vietnam</td><td style='text-align: center;'>(find special economic zones, seas, China)(following asian countries would one, China)(asian countries would one find, China)(would one find special economic, China)</td><td style='text-align: center;'>(c) China</td><td style='text-align: center;'>High School Geography</td></tr></table>

<div style="text-align: center;">Figure 10: Examples of mined pairs for the MMLU Task using Pythia Models with bigram and 5-gram analysis. Models include 12 billion and 6.9 billion parameters.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Question</td><td style='text-align: center;'>(Key, Value)</td><td style='text-align: center;'>Answer</td></tr><tr><td colspan="3">Pythia 12b - 3gram</td></tr><tr><td style='text-align: center;'>A tropical plant that grows several feet high, with a large fleshy root that contains an acrid milk juice, poisonous in its natural form, produces from the juice, extracted under pressure and purified, a staple foodstuff called what?</td><td style='text-align: center;'>(fleshy root contains, tapioca)(tropical plant grows&#x27;, &#x27;tapioca)(large fleshy root, &#x27;tapioca)(grows several feet, &#x27;tapioca)(natural form produces, &#x27;tapioca)</td><td style='text-align: center;'>Tapioca</td></tr><tr><td style='text-align: center;'>Feel Like Making Love and The First Time Ever I Saw Your Face were hit singles for which female artist?</td><td style='text-align: center;'>(love first time, roberta flack)(time ever saw, roberta flack)(ever saw face, roberta flack)(first time ever, roberta flack)(feel like making, roberta flack)</td><td style='text-align: center;'>Roberta Flack</td></tr></table>

<div style="text-align: center;">Figure 11: Examples of mined pairs for the TriviaQA Task using Pythia Models (12b) with trigram analysis.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_336_672_604_883.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_673_880_882.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">Figure 12: Model translation generations alignment for pair instances identifies in pretraining. As n-gram rises, the larger models are able to reproduce more aligned pairs from pretraining.</div>