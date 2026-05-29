## A APPENDIX

### A.1 HYPERPARAMETERS

Table 7 contains the hyperparameters used for NPSSM (700k params) and Table 8 contains the hyperparameters used for MSA Transformer (700k) params.

<div style="text-align: center;">Table 7: Hyperparameters for 700k parameter NPSSM</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Embedding dim  $ d $</td><td style='text-align: center;'>156</td></tr><tr><td style='text-align: center;'>Number of NPSSM layers</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Initial learning rate</td><td style='text-align: center;'>$ 1e^{-3} $</td></tr><tr><td style='text-align: center;'>$ p_{\text{mask}} $</td><td style='text-align: center;'>0.10</td></tr><tr><td style='text-align: center;'>optimizer</td><td style='text-align: center;'>Adam</td></tr><tr><td style='text-align: center;'>effective batch size</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>Starting  $ \lambda $</td><td style='text-align: center;'>1.0</td></tr><tr><td style='text-align: center;'>$ \lambda $  annealing schedule</td><td style='text-align: center;'>linear</td></tr><tr><td style='text-align: center;'>Ending  $ \lambda $  value</td><td style='text-align: center;'>0.5</td></tr></table>

<div style="text-align: center;">Table 8: Hyperparameters for 700k parameter MSA Transformer</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Embedding dim  $ d $</td><td style='text-align: center;'>100</td></tr><tr><td style='text-align: center;'>Num attention heads</td><td style='text-align: center;'>10</td></tr><tr><td style='text-align: center;'>Number of MSA-Transformer layers</td><td style='text-align: center;'>4</td></tr><tr><td style='text-align: center;'>Activation Dropout</td><td style='text-align: center;'>0.1</td></tr><tr><td style='text-align: center;'>Attention Dropout</td><td style='text-align: center;'>0.1</td></tr><tr><td style='text-align: center;'>Initial learning rate</td><td style='text-align: center;'>$ 6e^{-4} $</td></tr><tr><td style='text-align: center;'>$ p_{\text{mask}} $</td><td style='text-align: center;'>0.10</td></tr><tr><td style='text-align: center;'>optimizer</td><td style='text-align: center;'>Adam</td></tr><tr><td style='text-align: center;'>effective batch size</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>Starting  $ \lambda $</td><td style='text-align: center;'>1.0</td></tr><tr><td style='text-align: center;'>$ \lambda $  annealing schedule</td><td style='text-align: center;'>linear</td></tr><tr><td style='text-align: center;'>Ending  $ \lambda $  value</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Datapoint Positional Embedding</td><td style='text-align: center;'>None</td></tr><tr><td style='text-align: center;'>Tied Row Attention</td><td style='text-align: center;'>True</td></tr></table>

### A.2 LAYER ABLATIONS

To investigate the effects of each individual layer, we construct five identical models with the same hyperparameters as in 7 with a context size of k = 150. We then train these models for 100,000 steps and evaluate the results on the same set of data as in Table 2. We include these results in Table 9. We observe that permuting the order of the layers results in a noticeable decrease in performance, indicating that the ordering of the layer application is an important consideration for these class of models. We also observe that flattening the input, that is concatenating all 150 back to back to form a single sequence as an input, performance significantly worse than either of the models that operate on the MSA, achieving only 88% of the performance that the base model achieves. The final two ablations correspond to applying the layers only over the attributes/rows (i.e. a standard SSM) and only over the data points/columns. The  $ SSM_{attr} $  model performance similarly to the flattened version even though the  $ SSM_{attr} $  model does not have access to any external context set. This might be indicative that the performance of the Flattened version of the model was impacted due to the increased difficulty of using the context set without leveraging the MSA structure. The  $ SSM_{data} $ 

model observes only, which for this task without any relevant context likely limits the model to performing some sort of weighted average. This model converged fairly quickly (under 10,000 steps) and performs similarly to the KNN baseline from Table 2 (0.758).

### A.3 ADDITIONAL DATASETS

We additionally report performance of Beagle and the two best Non-Parametric Models on a different chromosome of 1k genomes, chr14 (Table 10) and on a different dataset HapMap (The International HapMap 3 Consortium (2010)) (Table 11).

<div style="text-align: center;">Table 9: Imputation Performance  $ (r^{2}) $  evaluated on 9159 variants from 516 haplotypes on chromosome 20. Models were trained for 100,000 steps, each with the same model configuration. Flattened is taking the input  $ H_{ref} $  and flattening it down into a single sequence (i.e. laying all the sequences in  $ H_{ref} $  back to back.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model</td><td style='text-align: center;'>$ r^{2} \pm \sigma $</td></tr><tr><td style='text-align: center;'>(Base) SSM $ _{\text{attr}} $   $ \mapsto $  SSM $ _{\text{data}} $</td><td style='text-align: center;'>0.9406  $ \pm $  0.0026</td></tr><tr><td style='text-align: center;'>SSM $ _{\text{data}} $   $ \mapsto $  SSM $ _{\text{attr}} $</td><td style='text-align: center;'>0.9308  $ \pm $  0.0029</td></tr><tr><td style='text-align: center;'>Flattened</td><td style='text-align: center;'>0.8241  $ \pm $  0.0067</td></tr><tr><td style='text-align: center;'>SSM $ _{\text{attr}} $  Only</td><td style='text-align: center;'>0.8206  $ \pm $  0.0074</td></tr><tr><td style='text-align: center;'>SSM $ _{\text{data}} $  Only</td><td style='text-align: center;'>0.7783  $ \pm $  0.0049</td></tr></table>

<div style="text-align: center;">Table 10: Imputation performance  $ (r^{2}) $  evaluated on  $ \sim19,369 $  untyped variants (dev set) from 516 haplotypes on chromosome 14. The Non-Parametric Models were trained on chromosome 20.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Class</td><td style='text-align: center;'>Method</td><td style='text-align: center;'>$ k $</td><td style='text-align: center;'>$ r^{2} $</td></tr><tr><td style='text-align: center;'>HMM</td><td style='text-align: center;'>Beagle (Browning et al. (2018))</td><td style='text-align: center;'>4388</td><td style='text-align: center;'>0.964 \pm 0.002</td></tr><tr><td rowspan="2">Non-Parametric Models</td><td style='text-align: center;'>MSA Transformer</td><td style='text-align: center;'>650</td><td style='text-align: center;'>0.963 \pm 0.002</td></tr><tr><td style='text-align: center;'>NPSSM</td><td style='text-align: center;'>2000</td><td style='text-align: center;'>0.967 \pm 0.002</td></tr></table>

<div style="text-align: center;">Table 11: Imputation performance  $ (r^{2}) $  evaluated on 962 untyped variants from 400 haplotypes from Hapmap on chromosome 14. The Non-Parametric Models were trained on chromosome 20 on 1000 Genomes.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Class</td><td style='text-align: center;'>Method</td><td style='text-align: center;'>$ k $</td><td style='text-align: center;'>$ r^{2} $</td></tr><tr><td style='text-align: center;'>HMM</td><td style='text-align: center;'>Beagle (Browning et al. (2018))</td><td style='text-align: center;'>1828</td><td style='text-align: center;'>0.891  $ \pm $  0.008</td></tr><tr><td rowspan="2">Non-Parametric Models</td><td style='text-align: center;'>MSA Transformer</td><td style='text-align: center;'>650</td><td style='text-align: center;'>0.921  $ \pm $  0.007</td></tr><tr><td style='text-align: center;'>NPSSM</td><td style='text-align: center;'>1828</td><td style='text-align: center;'>0.919  $ \pm $  0.007</td></tr></table>