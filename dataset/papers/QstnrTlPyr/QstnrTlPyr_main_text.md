# BSM: SMALL BUT POWERFUL BIOLOGICAL SEQUENCE MODEL FOR GENES AND PROTEINS

Anonymous authors

Paper under double-blind review

## ABSTRACT

Modeling biological sequences such as DNA, RNA, and proteins is crucial for understanding complex processes like gene regulation and protein synthesis. However, most current models either focus on a single type or treat multiple types of data separately, limiting their ability to capture cross-modal relationships. We propose that by learning the relationships between these modalities, the model can enhance its understanding of each type. To address this, we introduce BSM, a small but powerful mixed-modal biological sequence foundation model, trained on three types of data: RefSeq, Gene Related Sequences, and interleaved biological sequences from the web. These datasets capture the genetic flow, gene-protein relationships, and the natural co-occurrence of diverse biological data, respectively. By training on mixed-modal data, BSM significantly enhances learning efficiency and cross-modal representation, outperforming models trained solely on unimodal data. With only 110M parameters, BSM achieves performance comparable to much larger models across both single-modal and mixed-modal tasks, and uniquely demonstrates in-context learning capability for mixed-modal tasks, which is absent in existing models. Further scaling to 270M parameters demonstrates even greater performance gains, highlighting the potential of BSM as a significant advancement in multimodal biological sequence modeling.

## 1 INTRODUCTION

Biological sequences—such as DNA, RNA, and proteins—are fundamental to an organism’s functions, as they encode genetic information that determines structure, function, and regulatory mechanisms (Watson & Crick, 1953; Nirenberg & Matthaei, 1961). Understanding these sequences is vital for unraveling the mysteries of biological evolution, deciphering disease mechanisms, and elucidating molecular interactions.

By applying machine learning algorithms to large-scale biological sequence data, one can capture evolutionary effects and extract complex patterns in gene transcription and protein translation. This not only enhances our understanding of gene regulation and protein function but also enables the prediction and generation of complex biological functions, significantly advancing our comprehension of biology and life processes (Nguyen et al., 2024a).

Despite the rapid advancements in modeling biological sequences with machine learning, current efforts have primarily focused on creating unimodal models specialized for DNA, such as DNABert2 (Zhou et al., 2023), HyenaDNA (Nguyen et al., 2024b), Caduceus (Schiff et al., 2024), NT (Dalla-Torre et al., 2023); RNA, including RNA-FM (Chen et al., 2022); or proteins, like ESM2 (Lin et al., 2023), ProTrans (Ahmed et al., 2020), ProGen2 (Nijkamp et al., 2023). However, complex biological processes such as gene regulation, CRISPR immunity, and genetic transposition involve interactions across multiple modalities.

Recently, several methods have focused on developing models capable of handling both gene and protein data. For example, Evo (Nguyen et al., 2024a) is a 7B genomic foundation model pretrained on DNA sequences, which inherently contain the potential to express other modalities. It learns from large genomic regions to capture systems-wide interactions and enables the design of more sophisticated biological functions. LucaOne (He et al., 2024) is a 1.8B model that is trained on both gene and protein data separately, allowing the model to process and analyze both types of data concurrently. Although these models demonstrate excellent performance across various tasks,

<div style="text-align: center;"><img src="imgs/img_in_chart_box_223_168_608_350.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_169_998_350.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;">Figure 1: Impact of mixed-modal data on model learning efficiency. After the first round of training, the model shows slow learning efficiency when using only Round 1 data (single-modal data: DNA, RNA, and protein). However, when trained on the newly introduced mixed-modal data, it achieves significantly lower validation loss, indicating greatly improved learning efficiency. Similarly, the introduction of new web data after the second round further reduces validation loss. All validation losses across rounds are computed on the same Round 1 validation data for consistency.</div>


they achieve their capabilities primarily by increasing computational resources—LucaOne, for instance, utilizes 800 billion tokens for pretraining—and by scaling up model size to the billions to accommodate multiple modalities. This raises the question of whether smaller models can achieve similar capabilities, making advanced biological sequence modeling more accessible and practical.

Recent advancements in small language models (SLMs) have shown significant progress (Minaee et al., 2024). These models have demonstrated emerging capabilities and achieved performance levels comparable to much larger models. The secret behind this achievement lies in strategic training choices, such as using "textbook-quality" data. Capable SLMs are faster to run and easier to serve; meanwhile, ongoing improvements, as seen with the Phi series (Abdin et al., 2024), indicate that SLMs are still under-trained and have great potential for further improvement.

In this paper, we explore the construction of high-quality biological sequence data. According to the central dogma of molecular biology (Crick, 1970), which highlights the sequential flow of genetic information, DNA, RNA, and proteins are intrinsically interconnected. We propose that by learning the relationships among these three types of sequences, the model can enhance its understanding of each modality. In light of this, we introduce three types of mixed-modal data for pretraining: RefSeq, Gene Related Sequences, and interleaved biological sequences from the web. These datasets capture genetic flow, gene and protein relationships, and the natural co-occurrence of diverse biological data types, respectively. Unlike previous work, which trains models using unimodal data or combines different types in a simplistic manner (where each training sample consists of only one type), we explicitly train the model on this mixed-modal data to learn the relationships among them.

We emphasize that incorporating mixed-modal data facilitates a more comprehensive understanding of biological sequences and enables more effective acquisition of cross-modal representations by better learning the relationships between these modalities. As illustrated in Figure 1, our experiments reveal that relying on unimodal data to learn these capabilities results in slow learning efficiency and requires substantial data and model sizes. In contrast, training on mixed-modal data significantly lowers validation loss, computed on the same Round 1 data across rounds, indicating greatly improved learning efficiency and better representation across all single modalities.

Based on these newly introduced types of high-quality mixed-modal data, we develop a small but powerful biological sequence foundation model, BSM, through a structured multi-round training approach. In this process, we conduct three rounds of training, progressively incorporating different types of mixed-modal data in the latter two rounds. By employing an annealing strategy, we optimize the data mix to ensure the best possible integration of these datasets. Our experiments demonstrate that BSM achieves performance comparable to that of billion-scale models on both single-modal and complex mixed-modal tasks. This proves the effectiveness of our method and its significant potential.

To conclude, our work makes the following contributions: 1) We propose that explicitly learning the relationships between genes and proteins can enhance the model's understanding of each modality. We introduce three types of mixed-modal data and strategically integrate them with unimodal data, ultimately resulting in our small but powerful BSM model. 2) We conduct extensive experiments

<div style="text-align: center;"><img src="imgs/img_in_image_box_219_162_999_581.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure 2: Overview of the pretraining data and training process of the BSM model. BSM utilizes three types of mixed-modal data: RefSeq, Gene Related Sequences, and interleaved biological sequences from the web for pretraining. It undergoes three rounds of training to enhance its ability to learn complex relationships among different types of biological data.</div>


demonstrating that BSM achieves performance comparable to billion-parameter models across various tasks, particularly excelling in mixed-modal tasks, and exhibits unique and strong few-shot learning capabilities with different modality combinations. 3) We conduct scaling experiments to show that BSM scales effectively; when increased to 270M parameters, it achieves better results, highlighting the potential of our approach.

## 2 METHODS

### 2.1 ARCHITECTURE

BSM employs a single-nucleotide tokenizer with a vocabulary that includes nucleotides, amino acids, and special tokens. It uses an autoregressive architecture to model biological sequences such as genes and proteins. By learning next-token prediction, the model reasons over sequences causally and captures statistical patterns and dependencies in the training data, enabling effective representation and generation of biological sequences. Furthermore, the autoregressive architecture's sequential nature effectively handles long-range dependencies, which is crucial in biological sequences like DNA, RNA, and proteins, where long-context information can reveal critical functional relationships or structural interactions.

The BSM family includes models of two sizes, specifically BSM-110M and BSM-270M. BSM-110M is a decoder-only Transformer with 12 layers, each having 12 attention heads and a hidden dimension of 768, and BSM-270M features 20 layers, 16 attention heads, and a hidden dimension of 896. Both models utilize rotary position embedding (RoPE) (Su et al., 2024) with a base frequency hyperparameter of 100,000 and support a context length of 1024 tokens. To accelerate training, we employ flash-attention mechanisms (Dao, 2023).

### 2.2 PRETRAINING DATA

High-quality biological data play a key role in developing effective models for biological sequences. In addition to using unimodal protein and gene data, we incorporate three types of mixed-modal data for continued pretraining, each containing valuable information that helps the model learn diverse dependencies and interactions in biological sequences, as well as critical functional relationships.

These multi-modal datasets enhance the model’s learning efficiency and its ability to understand cross-modal relationships, enabling it to handle various biological data. In the following section, we introduce the data used in pretraining. Details of the data construction are listed in the Appendix. B. Additionally, we perform data cleansing by removing instances of the downstream task test data from the pretraining data.

Single-modal Data In the initial pretraining phase, we utilize single-modal datasets that exclusively contain nucleic acid (DNA or RNA) sequences or protein sequences, enabling the model to concentrate on understanding the essential structures, patterns, and unique characteristics of each modality. These datasets are sampled from the pretraining dataset of LucaOne (He et al., 2024), and we only use the sequence information from RefSeq (O'Leary et al., 2016), UniProt (Consortium, 2015) and ColabFold (Mirdita et al., 2022) without incorporating additional biological annotations, resulting in a data volume of 220 billion tokens. Single-modal data not only enhances the model's ability to understand individual modality sequences but also establishes a solid foundation for continued pretraining on more complex multimodal data.

Mixed-modal Pair Data - NCBI RefSeq To better learn the relationships among DNA, RNA, and proteins, we incorporate mixed-modal data from the NCBI RefSeq database (O'Leary et al., 2016). RefSeq provides a comprehensive and curated collection of annotated reference sequences that illustrate the flow of genetic information in the central dogma of molecular biology—DNA to RNA to protein. This resource is crucial for facilitating the understanding of gene-protein interactions and regulatory mechanisms, capturing essential details about transcription and translation processes.

We use data from 15 different species, including Bubalus Bubalis, Camelus Dromedarius, Human, and several others. Each species has its own unique DNA, RNA, and protein sequences, which are used to construct our gene-protein pairing dataset, resulting in a dataset of 9.2 billion tokens. This extensive data ensures a rich representation of genetic information, allowing the model to leverage diverse biological contexts for effective learning while recognizing the inherent links between DNA sequences and their corresponding proteins.

Mixed-modal Pair Data - Gene Related Sequences To better understand the complex relationships between gene-gene and gene-protein, we incorporate data from the NCBI Gene database (Brown et al., 2015), which offers a detailed collection of gene-related sequences and annotations. The Gene Related Sequences data within this collection contains related sequences to the gene and provides links to the corresponding records in Entrez Nucleotide (Maglott et al., 2010), Entrez Protein (Ostell, 2012) or UniProtKB (Boutet et al., 2007).

We have constructed a dataset from the Gene Related Sequences data by sampling from multiple species, resulting in a diverse collection of 8.3 billion tokens. This dataset enables the model to capture complex dependencies, recognize patterns of gene regulation and protein expression, and enhances its understanding of gene and protein functions.

Mixed-modal Interleaved Data - Filtered Web Data To simulate the natural co-occurrence of diverse biological data types and provide the model with a more realistic learning context, we integrate filtered web data from FineWeb-Edu (Penedo et al., 2024), which consists of high-quality web-crawled documents. We specifically filter this data to extract biological sequences within documents, using a special token, <sep>, to separate interleaved biological sequences, resulting in a final dataset of 33 million tokens.

By incorporating this curated dataset into our pretraining process, BSM can leverage a rich and diverse collection of arbitrarily interleaved biological sequences. This dataset features a greater number of interleaved sequences, enhancing the model's ability to capture complex biological relationships and enabling it to understand and generate gene and protein sequences in any arbitrary context.

### 2.3 PRETRAINING PROCEDURE

Three-round Training We pretrain BSM models from scratch in an end-to-end manner, which includes a three-round training process, as shown in Figure 2. It begins by establishing a foundational understanding of individual types of biological sequences (DNA, RNA, or proteins) using 100B single-modal tokens. The model then advances to incorporate multi-modal data, enhancing its ability to understand relationships and transitions between different biological data types, which is crucial for tasks involving mixed modalities. Specifically, in the second round, it utilizes an additional 120B

single-modal data along with a certain amount of multi-modal pair data from RefSeq (24B) and Gene Related Sequences (12B). In the third round, it trains on a small amount of high-quality mixed-modal data, including pair data from RefSeq (1.3B) and Gene Related Sequences (1.3B) as well as web interleaved data (13.3B). By upsampling these high-quality but relatively small multi-modal datasets during continued pretraining, we significantly enhance the BSM model's performance across a range of biological tasks, making it a powerful tool for decoding the complexities of molecular biology.

We set the learning rate to decay from 2e-5 to 1e-5 in the first round. In the second round, it decays from 1e-5 to 1e-7, and in the third round, it decays from 1e-6 to 0. The tokenizer is trained with a peak learning rate of 2e-5.

Simulated Annealing & Data Mixing To obtain a high-quality biological model, it is essential to carefully determine the proportion of different data sources in pretraining. Similar to Blakeney et al. (2024) and Llama 3.1 (Dubey et al., 2024), we find that upsampling and annealing help efficiently select the optimal ratio for mixing new mixed-modal datasets. We evaluate these datasets by training the BSM-110M over 1000/500 steps with linearly annealed learning rates. Through these annealing experiments, we identify the best data mixing ratio based on the lowest validation loss in the second and third rounds. Ultimately, we employ a ratio of 10:2:1 for single-modal data, RefSeq, and Gene Related Sequences in the second round, and a ratio of 1:1:10 for RefSeq, Gene Related Sequences, and web interleaved data in the third round. After determining the best mix, we train a larger model (BSM-270M) on this selected data mix.

## 3 EXPERIMENTS

We evaluate BSM’s capabilities in understanding and generating biological sequences across a variety of tasks, including both mixed-modal and single-modal tasks. BSM demonstrates outstanding performance on multi-modal tasks, even surpassing many billion-scale models. We also assess BSM’s in-context learning (ICL) ability in a few-shot setting for mixed-modal tasks, demonstrating that BSM possesses this capability, which has not yet been observed in other biological sequence models. Additionally, we investigate the model’s supervised fine-tuning (SFT) and zero-shot performance on protein and gene-related tasks. Scaling experiments confirm that further increasing the model size continues to enhance its performance. We conduct an ablation study on the performance of models from different rounds to verify the value of multi-modal data. Finally, we also evaluate the model’s generative abilities using the perplexity metric. Details of evaluation datasets are listed in the Appendix C.

Implementation Details For tasks requiring SFT, we fine-tune the model using a learning rate of  $ 1e^{-6} $  and a batch size of 16. For tasks that require two sequences as input, other baseline models lack the ability to simultaneously process both sequences, especially when it comes to handling gene and protein pairs. Instead, they use a dual-tower structure, employing two independent encoders to encode each sequence separately. In contrast, BSM directly connects the two sequences as input using a  $ \langle sep\rangle $  token, allowing the model to evaluate their relationship directly in a unified context. Implementation details are listed in the Appendix A.

### 3.1 MIXED-MODAL MODELING & FEW-SHOT EVALUATION

As shown in Figure 3, in mixed-modal tasks, such as RNA-protein interactions, BSM outperforms larger models like LucaOne. In the Central Dogma task, which focuses on DNA-protein associations, BSM achieves performance comparable to LucaOne. Additionally, in the few-shot learning setting without fine-tuning, BSM achieves performance close to SFT. Notably, BSM is the only existing biological sequence model capable of few-shot learning on mixed-modal data. These results highlight BSM's ability to efficiently process and analyze mixed-modal biological sequence data, positioning it as a leading model in the field despite its smaller size.

ncRPI The ncRNA-Protein Interactions (ncRPI) (Han & Zhang, 2023) task is a binary classification task aimed at predicting interactions between various non-coding RNAs (ncRNAs) and proteins, which is crucial for understanding cellular functions. Both BSM-110M and BSM-270M surpass the performance of billion-scale biological models, such as DNABert2 + ESM2-3B and LucaOne 1.8B. Notably, the results for BSM-270M are comparable to those of ncRPI-LGAT, a model specifically tailored for this task.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_219_179_573_388.jpg" alt="Image" width="28%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_577_180_817_389.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_820_182_1012_391.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">Figure 3: Results on mixed-modal tasks and few-shot evaluation. In the RNA-protein mixed-modal task (ncRPI), BSM outperforms larger models like LucaOne. In the DNA-protein mixed-modal task (Central Dogma), BSM achieves performance comparable to LucaOne. In few-shot learning settings without fine-tuning, BSM performs similarly to SFT, making it the only biological sequence model capable of few-shot learning on mixed-modal data.</div>


Central Dogma The Central Dogma task is a binary classification task curated by LucaOne, aimed at recognizing the intrinsic association between DNA sequences and their corresponding proteins based on the central dogma. Specifically, the DNA-protein pairs are constructed from the RefSeq database (O'Leary et al., 2016). To ensure data integrity, we removed 57 instances of test data that were present in our pretraining dataset. We conducted two experimental settings for this task: one involved SFT with 3,200 training samples, while the other utilized few-shot learning without fine-tuning the model.

In the SFT experiment, BSM-270M performs comparably to LucaOne despite using a much smaller model size. Additionally, BSM outperforms DNABERT2 + ESM2-3B in performance.

Few-shot Learning In the few-shot learning setting, we concatenate few-shot demonstrations with the test sample, each containing a DNA and protein sequence, as input for BSM. We then calculate the log probability for each token in the tested protein sequence. This allows us to obtain the overall generation probability for the tested protein sequence. We then set a threshold for this generation probability to classify whether the protein is linked to the tested DNA, and subsequently compute the prediction accuracy.

Despite not being fine-tuned, the BSM model demonstrates strong performance in identifying correct DNA-protein associations. The experiments show that increasing the number of demonstrations further enhances performance, with the few-shot learning results approaching those of SFT. This experiment highlights BSM's in-context learning capability, particularly in mixed-modal tasks, a capability that other existing models do not possess.

### 3.2 PROTEIN MODELING EVALUATION

We evaluate BSM’s capabilities on four protein tasks, with results shown in Figure 4. Notably, we surpass all baseline models in both the PPI and ProtLoc tasks, achieving the best results. In the ProtStab task, we obtain results comparable to LucaOne. Additionally, in the zero-shot protein fitness prediction task, we achieve performance similar to Evo-7B and Progen2-large. These results highlight BSM’s capability in modeling protein sequences through a deep understanding of protein functions and activities, despite its smaller size.

PPI The Protein-Protein Interaction (PPI) task is pivotal for mapping out how proteins interact within biological systems. We use the DeepPPI (Sun et al., 2017) database that contains human protein interactions for binary classification. The models are fine-tuned on this dataset, and their performances are assessed based on prediction accuracy. Both BSM-110M and BSM-270M surpass protein-specific models like DeepPPI and ESM2-3B, as well as multimodal biological sequence models like LucaOne.

Prokaryotic Protein Subcellular Location (ProtLoc) ProtLoc (Xu et al., 2009) predicts the subcellular localization of prokaryotic proteins, classifying them into six compartments like the cell membrane and cytoplasm. It uses a strategy similar to DeepLocPro (Moreno et al., 2024), helping to

<div style="text-align: center;"><img src="imgs/img_in_chart_box_216_167_580_351.jpg" alt="Image" width="29%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_588_167_984_348.jpg" alt="Image" width="32%" /></div>


<div style="text-align: center;">(d) Zero-shot Protein Fitness Prediction</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_380_580_536.jpg" alt="Image" width="29%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_587_369_1002_540.jpg" alt="Image" width="33%" /></div>


<div style="text-align: center;">Figure 4: Results on four protein tasks. BSM outperforms all baseline models in PPI and ProtLoc, achieving the best results. In ProtStab, its performance matches LucaOne. Additionally, in the zero-shot protein fitness prediction task, BSM shows comparable results to Evo-7B and Progen2-large.</div>


understand protein functions based on their locations. In this task, both BSM-110M and BSM-270M achieve the best results.

Protein Stability (ProtStab) evaluates protein stability by correlating features with stability measurements from the TAPE dataset (Rao et al., 2019). This task is important for understanding protein activity and can assist in drug design and biotechnology applications. In this task, BSM outperforms ESM-2-3B and TAPE, attaining results comparable to LucaOne.

Zero-shot Protein Fitness Prediction This task evaluates models' ability to predict the impact of mutations on protein function without task-specific fine-tuning. It uses Deep Mutational Scanning (DMS) datasets (Jacquier et al., 2013; Firnberg et al., 2014; Adkar et al., 2012; Tsuboyama et al., 2023; Kelsic et al., 2016), where a comprehensive set of mutations is introduced into protein-coding sequences to measure their effects on fitness. Fitness serves as a metric for how effectively a protein performs a specific function.

Following the implementation of Evo, the model predicts fitness scores based solely on its understanding of the protein sequence in a zero-shot setting. In the experiments, Evo-7B and NT-500M use gene sequences as input, while other protein models like ESM-2 650M and Progen2-large rely on protein sequences. In contrast, BSM utilizes both gene and protein sequences due to its mixed-modal modeling capability. Our experiments demonstrate that incorporating gene data enhances BSM-110M performance on this task, increasing the SRCC from 40.7% to 42.3%. This advancement not only establishes BSM's broad applicability in computational biology but also showcases its forward-looking nature in improving protein modeling capabilities through the integration of genetic information. Ultimately, BSM-270M achieves performance comparable to Evo-7B and Progen2-large, although it does not surpass ESM-2 650M.

### 3.3 GENE MODELING EVALUATION

We evaluate BSM’s capabilities on several critical genomic challenges, with results shown in Figure 5. BSM outperformed Evo 7B in the zero-shot ncRNA fitness prediction task, leveraging its understanding of genomic sequences to predict the effects of mutations on ncRNA functionality without task-specific fine-tuning. It also performed well in the ncRNA Fam multi-class classification task. These collective achievements underscore the model’s comprehensive strength in genomic analysis and its potential to contribute significantly to molecular biology research.

Zero-shot ncRNA Fitness Prediction This task investigates the model’s ability to predict the functional implications of mutations in non-coding RNAs (ncRNAs), including tRNAs, rRNAs,

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_166_595_362.jpg" alt="Image" width="31%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_602_167_999_361.jpg" alt="Image" width="32%" /></div>


<div style="text-align: center;">Figure 5: Results on two gene-related tasks. BSM outperformed Evo 7B in the zero-shot ncRNA fitness prediction task, accurately predicting the effects of mutations on ncRNA functionality without task-specific fine-tuning. It also performed well in the ncRNA Fam multi-class classification task.</div>


and ribozymes (Kobori et al., 2015; Andreasson et al., 2020; Domingo et al., 2018; Guy et al., 2014). Understanding these roles is crucial for cellular processes like protein synthesis and gene regulation. We use ncRNA Deep Mutational Scanning (DMS) data for evaluation, which includes various mutations and their effects on ncRNA functionality. Following Evo, we adopt a zero-shot approach to assess whether the pretrained BSM can generalize its understanding of genomic sequences to accurately predict the impact of mutations on ncRNA fitness without task-specific fine-tuning. Experimental results show that both BSM-110M and BSM-270M achieve the best performance, surpassing other methods, including Evo-7B.

Non-coding RNA Family (ncRNAFam) The ncRNAFam task is a sophisticated multi-class classification challenge, requiring models to accurately categorize non-coding RNA (ncRNA) sequences into 88 distinct families (Noviello et al., 2020; Rossi et al., 2019). These ncRNAs, while not coding for proteins, play indispensable roles in gene expression regulation and other cellular processes. Our fine-tuned BSM model achieves a remarkable accuracy of 97.5%, slightly below LucaOne but surpassing DNABert2 110M. This achievement underscores BSM's proficiency in discerning the subtleties of ncRNA sequences, showcasing its advanced capability to classify these crucial non-coding elements with high precision.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_223_914_1000_1330.jpg" alt="Image" width="63%" /></div>


<div style="text-align: center;">Figure 6: Effectiveness of scaling BSM and incorporating cross-modal data. Validation loss curves for the BSM-110M and BSM-270M show that scaling enhances model capabilities. Adding cross-modal data in Round 2 and Round 3 continuously reduces validation loss compared to the BSM-110M-single.</div>


### 3.4 Scaling Up BSM

To unlock the full potential of BSM, we investigate its scaling properties by increasing its parameters from 110M to 270M. As shown in Figure 6, BSM-270M exhibits lower validation loss across all three rounds of training, demonstrating a significant improvement compared to BSM-110M. Experiments across diverse biological tasks also indicate that a larger model enhances performance. This scaling shows that increasing model size can further enhance BSM's capabilities, underscoring the effectiveness of our approach and the considerable value of incorporating mixed-modal data in advancing biological sequence modeling.

### 3.5 ABLATION STUDY ON MIXED-MODAL DATA

We compared models trained on single-modal versus with mixed-modal data under the same token budget on BSM-110M. The results in Table 1 show that BSM-110M (R2) outperforms BSM-110M-single (R2), and BSM-110M (R3) outperforms BSM-110M-single (R3). This demonstrates that incorporating mixed-modal data in both Round 2 and Round 3 leads to a significant improvement in model performance, both in single-modal and mixed-modal tasks. Additionally, without mixed-modal data, BSM-110M-single performs significantly worse than billion-scale models. However, when mixed-modal data is included, its performance matches or even exceeds that of these larger models. Figure 6 shows the validation loss of BSM-110M-single consistently higher than BSM-110M. This aligns with Figure 1, confirming that introducing mixed-modal data significantly reduces validation loss and improves single-modal representations.

<div style="text-align: center;">Table 1: Ablation study on mixed-modal data.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>ncRPI</td><td style='text-align: center;'>PPI</td><td style='text-align: center;'>ProtLoc</td><td style='text-align: center;'>Protein Fitness</td><td style='text-align: center;'>ncRNA fitness</td></tr><tr><td style='text-align: center;'>ESM2-3B</td><td style='text-align: center;'>0.9332</td><td style='text-align: center;'>0.9745</td><td style='text-align: center;'>0.9496</td><td style='text-align: center;'>/</td><td style='text-align: center;'>/</td></tr><tr><td style='text-align: center;'>LucaOne</td><td style='text-align: center;'>0.938</td><td style='text-align: center;'>0.9748</td><td style='text-align: center;'>0.9452</td><td style='text-align: center;'>/</td><td style='text-align: center;'>/</td></tr><tr><td style='text-align: center;'>Evo</td><td style='text-align: center;'>/</td><td style='text-align: center;'>/</td><td style='text-align: center;'>/</td><td style='text-align: center;'>0.452</td><td style='text-align: center;'>0.243</td></tr><tr><td style='text-align: center;'>BSM-110M-single (R2)</td><td style='text-align: center;'>0.9216</td><td style='text-align: center;'>0.9648</td><td style='text-align: center;'>0.9019</td><td style='text-align: center;'>0.373</td><td style='text-align: center;'>0.21</td></tr><tr><td style='text-align: center;'>BSM-110M (R2)</td><td style='text-align: center;'>0.9422</td><td style='text-align: center;'>0.9722</td><td style='text-align: center;'>0.9401</td><td style='text-align: center;'>0.403</td><td style='text-align: center;'>0.239</td></tr><tr><td style='text-align: center;'>BSM-110M-single (R3)</td><td style='text-align: center;'>0.922</td><td style='text-align: center;'>0.9651</td><td style='text-align: center;'>0.9038</td><td style='text-align: center;'>0.379</td><td style='text-align: center;'>0.211</td></tr><tr><td style='text-align: center;'>BSM-110M (R3)</td><td style='text-align: center;'>0.9494</td><td style='text-align: center;'>0.975</td><td style='text-align: center;'>0.9685</td><td style='text-align: center;'>0.423</td><td style='text-align: center;'>0.256</td></tr></table>

### 3.6 COMPARISON OF BSM WITH MODELS OF SIMILAR SIZE

We compared our models with ESM-150M, which share similar size with ours. Results in Table 2 show that the performance of ESM-150M is far lower than its larger models, and both BSM-110M and BSM-270M significantly outperform ESM-150M, highlighting the advantages of our approach and the importance of mixed-modal data. We clarify that we report ESM-650M for Zero-shot Protein Fitness Prediction because it is the best size for this task (Nguyen et al., 2024a).

<div style="text-align: center;">Table 2: Comparison of BSM with ESM-150M of Similar Size</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Tasks</td><td style='text-align: center;'>ESM-150M</td><td style='text-align: center;'>ESM-650M</td><td style='text-align: center;'>ESM-3B</td><td style='text-align: center;'>BSM-110M</td><td style='text-align: center;'>BSM-270M</td></tr><tr><td style='text-align: center;'>PPI</td><td style='text-align: center;'>0.8139</td><td style='text-align: center;'>/</td><td style='text-align: center;'>0.9745</td><td style='text-align: center;'>0.975</td><td style='text-align: center;'>0.9788</td></tr><tr><td style='text-align: center;'>ProtLoc</td><td style='text-align: center;'>0.8644</td><td style='text-align: center;'>/</td><td style='text-align: center;'>0.9496</td><td style='text-align: center;'>0.9685</td><td style='text-align: center;'>0.9713</td></tr><tr><td style='text-align: center;'>Protein Stability</td><td style='text-align: center;'>0.7129</td><td style='text-align: center;'>/</td><td style='text-align: center;'>0.7556</td><td style='text-align: center;'>0.7653</td><td style='text-align: center;'>0.7681</td></tr><tr><td style='text-align: center;'>Protein Fitness</td><td style='text-align: center;'>0.408</td><td style='text-align: center;'>0.512</td><td style='text-align: center;'>/</td><td style='text-align: center;'>0.423</td><td style='text-align: center;'>0.441</td></tr></table>

### 3.7 PERPLEXITY EVALUATION

Perplexity (PPL) is one of the most common metrics for evaluating the generation capabilities of language models. It is defined as the exponentiated average negative log-likelihood of a sequence, reflecting how well a model can predict the next word based on the preceding context. A lower perplexity score indicates a better ability of the model to accurately predict the next word. We evaluate BSM's generation capability using perplexity on our validation protein data. As illustrated in Table 3, BSM outperforms the larger

<div style="text-align: center;">Table 3: Comparison of BSM with various protein sequence models based on perplexity.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>PPL.</td></tr><tr><td style='text-align: center;'>Progen2 2.7B</td><td style='text-align: center;'>8.92</td></tr><tr><td style='text-align: center;'>Progpt2 700M</td><td style='text-align: center;'>9.75</td></tr><tr><td style='text-align: center;'>BSM-270M</td><td style='text-align: center;'>9.47</td></tr></table>

model ProGPT2 700M, although it doesn’t surpass Progen2 2.7B. This demonstrates that using mixed-modal data for pretraining allows smaller models to effectively model and generate protein sequences.

## 4 RELATED WORK

### 4.1 BIOLOGICAL SEQUENCE MODEL

Modeling biological sequences has traditionally involved unimodal approaches tailored to specific data types, such as DNA, RNA, or proteins. While significant progress has been made with models like DNABert2 (Zhou et al., 2023), RNA-FM (Chen et al., 2022), and ESM2 (Lin et al., 2023), these models often struggle with capturing complex mixed-modal interactions inherent in biological processes. Recent advancements, such as LucaOne (He et al., 2024) and Evo (Nguyen et al., 2024a), have begun to handle both gene and protein data, demonstrating the potential of large-scale models in modeling multi-modal biological data. However, insufficient attention has been given to exploring diverse data, especially high-quality mixed-modal data, which is crucial for models to acquire comprehensive capabilities. Our work proves that learning from mixed-modal data significantly enhances learning efficiency and improves both single and mixed-modal representations.

### 4.2 SMALL LANGUAGE MODEL

Small Language Models (SLMs) like Phi (Gunasekar et al., 2023) and Gemma 2 (Team et al., 2024) illustrate that with strategic training approaches, such as high-quality data utilization and knowledge distillation, SLMs can achieve impressive performance. Unlike current trends in biological sequence modeling that focus on scaling model size to the billion-parameter level, our research explores the potential of leveraging rich and high-quality mixed-modal bio-sequence data, which has rarely been studied or utilized in this field. We demonstrate that introducing mixed-modal data can enable smaller models to achieve performance close to or even surpass that of billion-scale models, highlighting the critical importance of data quality and diversity. Our work strongly demonstrates the tremendous potential of expanding both mixed-modal data and model size, paving the way for more powerful models in bioinformatics.

## 5 CONCLUSION

In this study, we have demonstrated that high-quality mixed-modal biological data is essential for enhancing both cross-modal and single-modal learning capabilities in our BSM models. The results indicate that protein-gene interleaving data has considerable potential to improve model performance, highlighting the importance of data quality in training effective biological models.

However, our research has certain limitations. We utilized only a partial dataset from RefSeq and Gene Related Sequence data, which lacks exploration of other valuable types of cross-modal data, such as gene-protein interactions data. Additionally, we mined only a relatively small dataset of interleaved biological sequences from the web, it still yielded continuous improvements in model performance. This suggests that there is substantial room for further investigation, and we believe that leveraging larger and more diverse datasets could enhance our model's capabilities even further. Our future work will focus on exploring additional types of cross-modal data to fully realize the potential of mixed-modal approaches in biological sequence modeling and contribute to advancements in the field.