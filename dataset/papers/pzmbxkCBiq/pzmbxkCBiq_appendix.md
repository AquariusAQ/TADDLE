## APPENDIX OVERVIEW

The appendix is structured as follows:

Appendix §A provides a detailed description of evaluation datasets, including examples and statistical summaries.

Appendix §B presents supplementary experimental results, including analyses of win probability, likelihood scaling, and the effects of different regularization techniques.

Appendix §C further investigates model performance on NATURALQUESTIONSOPEN and TRIVIAQA.

## A DATASETS

This section provides an in-depth look at the datasets used in our experiments, focusing on the BINARIZEDPREF, LLM-as-a-Judge framework, NATURALQUESTIONSOPEN, and TRIVIAQA datasets.

BINARIZEDPREF Dataset. The BINARIZEDPREF collection process used a robust multi-source approach combining professional annotators, multiple independent annotation pipelines, and various validation methods. The foundation comes from professional annotation services (70% of data), with rigorous quality control through multi-annotator consensus, adversarial validation sets, and specialized verification datasets for issues like hallucination and repetition. We've ensured broad domain coverage, incorporating specialised modules for code generation, RAG interactions, STEM, and medical domains while maintaining strong multilingual capabilities across French, Spanish, Korean, Japanese, German, and Italian - including dedicated datasets for handling code-mixing and language transition cases. Quality control is implemented through multiple layers: consensus-based annotation (1-3 annotators depending on complexity), dedicated adversarial validation sets, and specific datasets targeting quality aspects like anti-repetition, length control, and format adherence. The data is predominantly recent (2024), with carefully weighted components and explicit test sets for key capabilities. We use strategic copy multipliers (up to 5x) for crucial capabilities, and the entire dataset is organised into functional groups (multilingual, code, RAG) to ensure balanced training across all target capabilities.

LLM-as-a-Judge Framework Dataset. We utilize a diverse set of prompts for the LLM-as-a-Judge framework. Figure 1 illustrates a representative example from this dataset, showcasing different generations from various competitor models. To provide insight into the composition of our LLM-as-a-Judge dataset, Figure 6 presents the distribution of prompt examples. This visualisation helps to understand the variety and balance of the prompts used in our evaluation framework.

NATURALQUESTIONSOPEN Dataset. Table 2 presents examples from the NATURALQUESTIONSOPEN dataset $ ^{2} $ , showcasing the types of questions and answers used in our evaluation. The NATURALQUESTIONSOPEN dataset, introduced by Kwiatkowski et al. (2019), is an open-domain question-answering benchmark. It consists of English questions paired with possible answer strings, all answerable using English Wikipedia content. Each data instance contains a question field and an answer field with potential correct responses. We use the validation set for our evaluation. Table 2 presents representative examples from this dataset, illustrating the types of questions and answers used in our evaluation.

TRIVIAQA Dataset. The TRIVIAQA dataset $ ^{3} $  is a comprehensive reading comprehension benchmark containing over 650,000 question-answer-evidence triples Joshi et al. (2017). It includes 95,000 question-answer pairs, each accompanied by an average of six independently gathered evidence documents. This structure provides high-quality distant supervision for question-answering.

<div style="text-align: center;">Table 1: Examples for datasets.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Role</td><td style='text-align: center;'>Example</td></tr><tr><td style='text-align: center;'>User</td><td style='text-align: center;'>I am looking to put a house on the market. 4 bedrooms, 3 bathrooms, 3000 sq ft. Nice neighbourhood and local school, strong community feel. Safe for kids, lots of young families. Can you write me a general description for the house in around 100-150 words? Make it professional sounding, emphasizing key aspects that you think would be attractive to young families.</td></tr><tr><td style='text-align: center;'>GPT-3.5-turbo</td><td style='text-align: center;'>Welcome to this spacious and inviting 4 bedroom, 3 bathroom home located in a desirable neighbourhood with a strong sense of community. Boasting over 3000 sq ft of living space, this property is perfect for growing families looking for a safe and welcoming environment to call home. The open-concept layout offers ample room for entertaining, while the well-appointed bedrooms provide a peaceful retreat for all. The nearby local school and abundance of young families in the area make this the ideal setting for children to thrive and make lasting friendships. Don’t miss out on the opportunity to make this house your family’s forever home.</td></tr><tr><td style='text-align: center;'>GPT-4o</td><td style='text-align: center;'>Welcome to your dream family home! Nestled in a vibrant, family-friendly neighborhood, this spacious 4-bedroom, 3-bathroom residence offers 3,000 sq ft of comfortable living space. The open floor plan seamlessly connects the living, dining, and kitchen areas, making it perfect for both everyday living and entertaining. Each bedroom is generously sized, providing ample space for relaxation and privacy. \n\nLocated in a top-rated school district, this home ensures your children will receive a quality education. The community boasts a strong, supportive atmosphere with plenty of young families, making it easy to form lasting friendships. Safety is a priority here, with well-maintained streets and parks where kids can play freely. Don’t miss the opportunity to create cherished memories in this exceptional home. Schedule a viewing today!</td></tr><tr><td style='text-align: center;'>Claude-3-Sonnet</td><td style='text-align: center;'>Nestled in the heart of a vibrant community, this spacious 3,000 square foot home offers the perfect blend of comfort and convenience for growing families. With four generously sized bedrooms and three beautifully appointed bathrooms, this residence provides ample space for daily living and entertaining. The neighborhood boasts a strong sense of community, with a local school nearby and a safe, family-friendly environment where children can thrive. The area is home to numerous parks, playgrounds, and recreational facilities, ensuring endless opportunities for outdoor activities and quality family time. Designed with modern amenities and a welcoming ambiance, this property is an exceptional choice for families seeking a comfortable and secure haven to call home.</td></tr><tr><td style='text-align: center;'>Llama-3-70b-chat-hf</td><td style='text-align: center;'>Here is a general description for your house: \n\nWelcome to this stunning 4 bedroom, 3 bathroom residence, perfectly situated in a highly sought-after neighborhood that embodies the essence of community living. With its spacious 3000 square feet of living space, this beautiful home offers the perfect blend of comfort, functionality, and charm. Ideal for young families, this property boasts a prime location within walking distance to a top-rated local school, ensuring a safe and convenient commute for kids. The neighborhood itself is a hub of activity, with a strong sense of community and a plethora of young families, making it the perfect environment for children to grow and thrive. With its generous layout, abundance of natural light, and exceptional curb appeal, this property is sure to impress even the most discerning buyers. Don’t miss out on this incredible opportunity to make this house your dream home! \n\nLet me know if you’d like me to make any changes!</td></tr><tr><td style='text-align: center;'>Llama-3-8b-chat-hf</td><td style='text-align: center;'>Here is a general description for your house: \n\nWelcome to this stunning 4-bedroom, 3-bathroom residence, perfectly situated in a highly sought-after neighborhood. This beautiful home offers a spacious 3000 square feet of living space, ideal for growing families. With its prime location, you’ll enjoy the convenience of being close to top-rated local schools, making it an ideal choice for young families. The neighborhood exudes a strong sense of community, with plenty of opportunities for kids to make friends and for parents to connect with neighbors. The home itself is a haven of comfort and style, with ample natural light, modern fixtures, and plenty of storage. With its safe and family-friendly atmosphere, this property is the perfect place to call home. Don’t miss this incredible opportunity to make memories with your loved ones in this wonderful community! \n\nLet me know if you’d like me to make any changes!</td></tr></table>

<div style="text-align: center;">Table 2: Examples for NATURALQUESTIONSOPEN.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Question</td><td style='text-align: center;'>Answer</td></tr><tr><td style='text-align: center;'>who does the voice of mickey mouse on mickey mouse clubhouse?</td><td style='text-align: center;'>[&#x27;Bret Iwan&#x27;, &#x27;Wayne Allwine&#x27;]</td></tr><tr><td style='text-align: center;'>who wrote knock knock knocking on heavens door?</td><td style='text-align: center;'>[&#x27;Bob Dylan&#x27;]</td></tr></table>

<div style="text-align: center;">Table 3: Examples for TRIVIAQA.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Question</td><td style='text-align: center;'>Answer</td></tr><tr><td style='text-align: center;'>Who was the next British Prime Minister after Arthur Balfour?</td><td style='text-align: center;'>[&#x27;Sir Henry Campbell-Bannerman&#x27;, &#x27;Campbell-Bannerman&#x27;, &#x27;Campbell Bannerman&#x27;, &#x27;Sir Henry Campbell Bannerman&#x27;, &#x27;Henry Campbell Bannerman&#x27;, &#x27;Henry Campbell-Bannerman&#x27;]</td></tr><tr><td style='text-align: center;'>Which Lloyd Webber musical premiered in the US on 10th December 1993?</td><td style='text-align: center;'>[&#x27;Sunset Blvd&#x27;, &#x27;West Sunset Boulevard&#x27;, &#x27;Sunset Boulevard&#x27;, &#x27;Sunset Bulevard&#x27;, &#x27;Sunset Blvd.&#x27;]</td></tr></table>

tasks. However, we do not use any evidence in our experiments. We use the validation set for our evaluation. Table 3 presents representative examples from the TRIVIAQA dataset.

## B ADDITIONAL EXPERIMENTAL RESULTS

As supplementary of the main experiment, we provide the following experiments.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_161_1003_550.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 6: Distribution of LLM-as-the-judge prompt dataset.</div>


Win Probability vs. Better Completion Likelihood. Figure 7 illustrates the relationship between win probability and better mean likelihood across different competitor models, including GPT-4, Claude-3-Sonnet, Llama-3-8B, and Llama-3-70B-Chat. We record points every 500 steps across varying hyperparameters for each method. Our results are consistent with our findings in the main text ( $ §4.2 $ ), suggesting that simply increasing the likelihood of better completions does not consistently result in performance improvements.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_214_819_611_1082.jpg" alt="Image" width="32%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_612_821_1009_1082.jpg" alt="Image" width="32%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_212_1085_612_1347.jpg" alt="Image" width="32%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_612_1087_1010_1347.jpg" alt="Image" width="32%" /></div>


<div style="text-align: center;">Figure 7: Win Probability vs Better Mean Likelihood Scaling Law. with different competitor models. including GPT-4o, Claude-3-Sonnet, Llama-3-8B, and Llama-3-70B-Chat</div>


IPO Learning curves with 7B model on the ULTRAFEEDBACK dataset. To demonstrate the generalisability of our findings, we experiment with the IPO using three different values of  $ \tau $ , adding NLL loss as an auxiliary loss with four  $ \lambda $  coefficients on the ULTRAFEEDBACK dataset using the 7B model. Figure 8 illustrates several key findings:

1. Likelihood and Performance Correlation: As shown in the first and second rows of the figure, a Higher likelihood for better completions and larger gaps between better and worse completions do not necessarily translate to improved model performance.

2. Likelihood and Cross-Input Diversity: Lower completion likelihood tends to enhance the models’ Cross-Input Diversity, as shown in the second and fourth rows, where lower better completion likelihood generally corresponds to improved Cross-Input Diversity.

3. Entropy and Over-optimisation: Decreasing entropy over top-k tokens (Per-Input Diversity) appears to be an indicator of over-optimisation for diversity. The fifth row demonstrates that curves with lower entropy typically do not perform as well, as reflected in their win probability. Particularly, this result shows that the turning points of the entropy, which transits from the increasing diversity to the decreasing entropy, is a strong indicator of the over-optimisation for diversity.

4. Probability Mass Distribution: We do not observe a decrease in probability mass in top k tokens in this case, as shown in the last row of the figure. This observation aligns with our findings: in runs without decreasing entropy, we do not observe a significant decline in win probability.

Learning curves with 7B model on the BINARIZEDPREF dataset. To demonstrate the generalisability of our findings, we perform additional experiments using the 7B model on the BINARIZEDPREF dataset. The results, consistent with our previous observations, underscore the broad applicability of our insights across various datasets. Figure 9 illustrates several key findings:

1. Likelihood and Performance Correlation: Higher likelihood for better completions and larger gaps between better and worse completions do not necessarily translate to improved model performance. This is evident in the first and second rows of the figure, where models with the highest better completion likelihood do not achieve the best performance.

2. Likelihood and Cross-Input Diversity: Lower completion likelihood tends to enhance the models’ Cross-Input Diversity. This trend is observable when comparing the second and fourth rows, where lower better completion likelihood generally corresponds to improved Cross-Input Diversity.

3. Entropy and Over-optimisation: Decreasing entropy over top-k tokens (Per-Input Diversity) appears to be a good indicator of over-optimisation for diversity. The fifth row demonstrates that curves with overly low entropy do not perform as well (i.e., pink curves), as reflected in their win probabilities. Additionally, as the entropy begins to rise again, an improvement in win probability is also observed.

4. Probability Mass Distribution: We do not observe a decrease in probability mass in top k tokens in this case, as shown in the last row of the figure. This observation aligns with our findings: in runs without decreasing entropy, we do not observe a significant decline in win probability.

Learning curves with 35B model on the BINARIZEDPREF dataset. To demonstrate the generalisability of our findings, we perform additional experiments using the 35B model on the BINARIZEDPREF dataset. The results align well with our previous observations. Figure 10 illustrates several key findings:

1. Likelihood and Performance Correlation: Similarly, results from larger model sizes suggest that higher likelihoods for better completions and larger gaps between better and worse completions do not necessarily lead to improved model performance, as shown in the first and second rows of the figure.

2. Likelihood and Cross-Input Diversity: Lower completion likelihood tends to enhance the models’ Cross-Input Diversity. Specifically, the curve with a lower better completion likelihood generally tends to have a higher Cross-Input Diversity.

3. Entropy and Over-Optimisation: A decrease in entropy over the top-k tokens (Per-Input Diversity) appears to indicate over-optimisation for diversity. For instance, the pink lines for DPO and IPO show a clear drop in entropy after 500 steps, accompanied by a decline in win probability.

4. Probability Mass Distribution: Similarly, we do not observe a decrease in probability mass in top k tokens in this case, as shown in the last row of the figure.

Training Negative Log-Likelihood Loss on better completions has limited influence on the model when it cannot affect completion likelihood. To demonstrate the generalisability of our findings, we perform further experiments with 35B models on the BINARIZEDPREF dataset. As shown in Figure 11, we experiment with DPO using three different values of  $ \beta $ , adding NLL loss as an auxiliary loss with four distinct coefficients for each  $ \beta $ . Similarly to our findings in the main text, results indicate that when there is limited impact on the likelihood, the NLL loss has minimal impact on model performance. Training Negative Log-Likelihood Loss on better completions remains susceptible to over-optimisation.

## Table 4: Examples for TRIVIAQA

Question: {question}
Reference Answer: {reference_answer}
Model Output: {model_output}

Evaluate the correctness of the model output compared to the reference answer.
Respond with EXACTLY ONE of the following options:
- Yes
- No
- Unsure

Guidelines:
- Yes: If the model output is correct or equivalent to the reference answer.
- No: If the model output is incorrect or contradicts the reference answer.
- Unsure: If you can't determine the correctness or if there's insufficient information.

Do not provide any explanation or additional text. Your entire response must be a single word.

Your response:

Discussion about Relationship Between KL and Completion likelihood. We report the  $ L_{2} $  loss between the policy model and the reference model with respect to the likelihood. This serves as a proxy for KL divergence, as both measure the divergence between the policy and reference models. While we could not generate a direct KL vs. Likelihood plot due to access restrictions, this proxy analysis allows us to provide relevant insights without requiring additional model retraining.

As shown in Figure 12, our experiments reveal that likelihood does not strictly correlate with the  $ L_{2} $  loss: lower likelihood (higher cross-entropy loss) does not necessarily correspond to a higher  $ L_{2} $  loss. This result suggests that the relationship between the likelihood of preferred completions and the divergence between the models is more nuanced than a simple monotonic association. In particular, the observed patterns reinforce the idea that likelihood and KL divergence, while connected under specific assumptions, are not directly interchangeable.

## C FURTHER INVESTIGATIONS FOR QUESTION ANSWERING TASKS

Case studies for NATURALQUESTIONSOPEN and TRIVIAQA tasks. Table 5 provides two examples for NATURALQUESTIONSOPEN and TRIVIAQA tasks, respectively.

LLM-as-a-Judge for the NATURAL QUESTIONS OPEN task. We implement a more flexible evaluation method to understand the potential issue of stylistic variations in answers. Instead of relying on exact string matching, which can be overly rigid, we employ an LLM-as-a-Judge using the

<div style="text-align: center;">Table 5: Model output examples for NATURALQUESTIONSOPEN and TRIVIAQA.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td colspan="3">Examples for NATURALQUESTIONSOPEN</td></tr><tr><td style='text-align: center;'>Field</td><td style='text-align: center;'>Content</td><td style='text-align: center;'>F1 Word</td></tr><tr><td style='text-align: center;'>Question</td><td style='text-align: center;'>Where is dakar located on the world map?</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>High Likelihood Answer</td><td style='text-align: center;'>Senegal</td><td style='text-align: center;'>100.0%</td></tr><tr><td style='text-align: center;'>Mid Likelihood Answer</td><td style='text-align: center;'>Dakar is the capital of Senegal and is located in West Africa. It is situated on the western coast of the country, on the Atlantic Ocean.</td><td style='text-align: center;'>8.7%</td></tr><tr><td colspan="3">Examples for TRIVIAQA</td></tr><tr><td style='text-align: center;'>Field</td><td style='text-align: center;'>Content</td><td style='text-align: center;'>F1 Word</td></tr><tr><td style='text-align: center;'>Question</td><td style='text-align: center;'>How many Rings of Power were there, in total?</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>High Likelihood Answer</td><td style='text-align: center;'>20</td><td style='text-align: center;'>100.0%</td></tr><tr><td style='text-align: center;'>Mid Likelihood Answer</td><td style='text-align: center;'>There were 20 Rings of Power in total, 3 of which were given to the Elves, 7 to the Dwarves, and 9 to the Men.</td><td style='text-align: center;'>8.7%</td></tr></table>

GPT4o model. As shown in Table 4, this LLM-based evaluation system is presented with the original question, the reference answer, and the model's output. It then assesses whether the model's output is correct, incorrect, or if there's not enough information to make a determination, responding with "Yes", "No", or "Unsure" respectively. We compute the model performance based on the percentage of "Yes". Figure 13 shows the model performance on the ULTRAFEEDBACK dataset using the 7B model. Our analysis reveals that while the LLM-as-a-Judge evaluation method demonstrates a trend similar to the  $ F_{1} $  score, it consistently yields higher performance metrics.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_139_161_1018_1362.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Figure 8: Learning curves across training steps for various metrics. Results are reported for the 7B models using IPO on the ULTRAFEEDBACK dataset with varying values of  $ \tau $  and  $ \lambda $ .</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_138_160_1018_1360.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Figure 9: Learning curves across training steps for various metrics. Results are reported for the 7B models using the Hinge, DPO, and IPO on the BINARIZEDPREF dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_137_159_1018_1360.jpg" alt="Image" width="71%" /></div>


<div style="text-align: center;">Figure 10: Learning curves across training steps for various metrics. Results are reported for the 35B models using the Hinge, DPO, and IPO on the BINARIZEDPREF dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_212_158_1013_760.jpg" alt="Image" width="65%" /></div>


<div style="text-align: center;">Figure 11: Control Likelihood via training on better completion on the BINARIZEDPREF dataset, using the 35B model. When different runs have similar likelihoods, the win probability and diversity of their model outputs tend to follow the same trend throughout training.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_893_466_1120.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_495_898_732_1120.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_773_901_1009_1121.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_217_1125_456_1348.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_500_1124_731_1350.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_768_1126_991_1351.jpg" alt="Image" width="18%" /></div>


<div style="text-align: center;">Figure 12: Our results indicate that completion likelihood does not strictly correlate with the  $ L_{2} $  loss: lower likelihood (higher cross-entropy loss) does not necessarily correspond to a higher  $ L_{2} $  loss.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_213_665_479_863.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_479_665_745_863.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_744_665_1008_863.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">Figure 13: NATURALQUESTIONSOPEN vs Better Mean LLH on the ULTRAFEEDBACK dataset using the 7B model. The  $ F_{1} $  score and LLM-as-a-Judge results are reported.</div>