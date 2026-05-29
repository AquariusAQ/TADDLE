## A RELATED WORK OF EVALUATION METHOD

Before the advent of LLMs, traditional evaluation methods for assessing machine-generated text involved both model-free and model-based metrics. The former refers to metrics that compare the output to a reference text, with BLEU (Papineni et al., 2002) and ROUGE (Lin, 2004) being the most commonly used. However, Krishna et al. (2021) highlighted the shortcomings of reference-based metrics like ROUGE, noting their unreliability for effective evaluation. Recently, there has been a shift towards model-based evaluation methods, including BERTScore (Zhang et al., 2019), BLEURT (Sellam et al., 2020), and BARTScore (Yuan et al., 2021), which focus on capturing semantic meaning rather than solely assessing lexical similarities. These are traditional evaluation methods, yet they are not optimally equipped to evaluate the complexity of large language models.

## B EVALUATION METRICS

• Pearson is a measure of the linear correlation between two variables, which measures the strength and direction of the linear relationship between the two variables.

• Spearman is a nonparametric statistical measure designed to assess the strength and direction of the monotonic relationship between two variables.

• Kendall-Tau is a nonparametric statistical method used to assess the correlation between two variables, especially when the variables are categorical.

• micro-F1 calculates the harmonic mean of precision and recall by considering the contributions of each prediction equally, regardless of the class.

<div style="text-align: center;">Table 6: Question categories of the AttriData dataset.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>First Level of Question</td><td style='text-align: center;'>Total Number</td></tr><tr><td style='text-align: center;'>NLP Basic</td><td style='text-align: center;'>2658</td></tr><tr><td style='text-align: center;'>Text Generation</td><td style='text-align: center;'>2726</td></tr><tr><td style='text-align: center;'>Question and Answer</td><td style='text-align: center;'>2383</td></tr><tr><td style='text-align: center;'>Reasoning</td><td style='text-align: center;'>6335</td></tr><tr><td style='text-align: center;'>Math</td><td style='text-align: center;'>4965</td></tr><tr><td style='text-align: center;'>Professional Field</td><td style='text-align: center;'>2647</td></tr><tr><td style='text-align: center;'>Multi-Turn Dialogue</td><td style='text-align: center;'>1118</td></tr></table>

<div style="text-align: center;">Table 7: The information about the amount of misattribution data.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>First Level of Misattribution</td><td style='text-align: center;'>Total Number</td></tr><tr><td style='text-align: center;'>Response Quality</td><td style='text-align: center;'>2726</td></tr><tr><td style='text-align: center;'>Instruction Following</td><td style='text-align: center;'>477</td></tr><tr><td style='text-align: center;'>Knowledge Ability</td><td style='text-align: center;'>2120</td></tr><tr><td style='text-align: center;'>Reasoning Capability</td><td style='text-align: center;'>4471</td></tr><tr><td style='text-align: center;'>Multi-Turn Dialogue</td><td style='text-align: center;'>124</td></tr><tr><td style='text-align: center;'>Creative Ability</td><td style='text-align: center;'>103</td></tr><tr><td style='text-align: center;'>Comprehension Ability</td><td style='text-align: center;'>340</td></tr><tr><td style='text-align: center;'>Other Errors</td><td style='text-align: center;'>143</td></tr><tr><td style='text-align: center;'>Safety</td><td style='text-align: center;'>8</td></tr></table>

<div style="text-align: center;"><img src="imgs/img_in_image_box_138_152_1009_948.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Figure 6: The prompt template of feedback generation by GPT-4.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_134_158_1008_1449.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Figure 7: The prompt utilized in the experiments in English.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_136_284_1009_1278.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Figure 8: The prompt utilized in the experiments in Chinese.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_135_181_1008_1390.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Figure 9: The case of Response Quality and Instruction Following.</div>


 $$ P(A) $$ 

<div style="text-align: center;"><img src="imgs/img_in_image_box_135_184_1010_1393.jpg" alt="Image" width="71%" /></div>


 $$ \dot{ 字母 }n $$ 

 $$ P(A)=\frac{m}{n} $$ 

 $$  字母 n $$ 

 $$ P(n)=\frac{Number of occurrences of letter“n”}{Total number of all letters in the word}=\frac{2}{10}=\frac{1}{5} $$ 

 $$ \frac{ 字母 \text{“}n”出现的次数 }{ 单词中所有字母的总数 }=\frac{2}{10}=\frac{1}{5} $$ 

 $$ P(n)= $$ 

 $$ \text{“}n\text{”} $$ 

 $$ P(A)=\frac{m}{n} $$ 

 $$ \mathrm{n} $$ 

 $$ \frac{1}{5} $$ 

 $$ P(A) $$ 

 $$ \dot{\textbf{n}} $$ 

 $$ \frac{1}{5}. $$ 

 $$  ``n^{\prime \prime} $$ 

 $$ \bar{s}\quad\text{“n”}\quad\text{is?} $$ 

 $$ \text{“n”} $$ 

 $$ \text{“n”} $$ 

 $$ \text{“n”} $$ 

 $$ P(A) $$ 

 $$ P(n)=\frac{Number of occurrences of letter“n”}{Total number of all letters in the word}=\frac{1}{10}. $$ 

 $$ P(A)=\frac{m}{n} $$ 

 $$ \frac{1}{5}. $$ 

 $$ \dot{ 中 } 字母 n $$ 

 $$ \frac{ 字母 n 出现的次数 }{ 单词中所有字母的总数 }=\frac{1}{10} $$ 

 $$ P(A)=\frac{m}{n} $$ 

 $$ P(A) $$ 

 $$ \frac{1}{5} $$ 

 $$ P(n)= $$ 

<div style="text-align: center;">Figure 10: The case of Knowledge Ability, Reasoning Capability and Creative Ability.</div>


<div style="text-align: center;"><img src="imgs/img_in_image_box_135_157_1008_1457.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Figure 11: The case of Multi-Turn Dialogue, Comprehension Ability and Safety. 21</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_234_157_981_901.jpg" alt="Image" width="61%" /></div>


<div style="text-align: center;">Figure 12: The overview of Misattribution Framework.</div>