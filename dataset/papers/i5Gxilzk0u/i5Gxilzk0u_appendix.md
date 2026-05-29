## A THEORETICAL ANALYSIS FOR DECOUPLE-THEN-COMBINE

Our proposed MOCO relies on the assumption that the joint conditional probability  $ p(x_{t-1} \mid c_{\mathrm{text}}, c_{\mathrm{audio}}, x_{t}) $  can be approximated by  $ p(x_{t-1,\mathrm{lower}} \mid c_{\mathrm{text}}, x_{t}) \cdot p(x_{t-1,\mathrm{upper}} \mid c_{\mathrm{audio}}, x_{t}) $ , expressed as:

 $$ p(x_{t-1}\mid c_{text},c_{audio},x_{t})\approx p(x_{t-1,lower}\mid c_{text},x_{t})\cdot p(x_{t-1,upper}\mid c_{audio},x_{t}), $$ 

where  $ x_{t} $  denotes the motion at denoising step t, composed of upper-body motion  $ x_{t,upper} $  and lower-body motion  $ x_{t,lower} $ .

We provide a detailed derivation of Equation 15, outlining the two approximations involved in the decomposition process. The derivation follows these steps:

 $$ p(x_{t-1}\mid c_{text},c_{audio},x_{t})=p(x_{t-1,lower},x_{t-1,upper}\mid c_{all}),\ where c_{all}=\{c_{text},c_{audio},x_{t}\} $$ 

 $$ =p(x_{t-1,lower}\mid c_{all})\cdot p(x_{t-1,upper}\mid c_{all},x_{t-1,lower}) $$ 

 $$ \approx p(x_{t-1,lower}\mid c_{all})\cdot p(x_{t-1,upper}\mid c_{all}) $$ 

 $$ \approx p(x_{t-1,lower}\mid c_{all}\setminus\{c_{audio}\})\cdot p(x_{t-1,upper}\mid c_{all}\setminus\{c_{text}\}) $$ 

 $$ =p(x_{t-1,lower}\mid c_{text},x_{t})\cdot p(x_{t-1,upper}\mid c_{audio},x_{t}). $$ 

The first approximation occurs in the transition from Equation 16 to Equation 17. Here, we approximate:

 $$ \begin{align*}p(x_{t-1,upper}\mid c_{all},x_{t-1,lower})&=p(x_{t-1,upper}\mid c_{text},c_{audio},x_{t,upper},x_{t,lower},x_{t-1,lower})\\&\approx p(x_{t-1,upper}\mid c_{text},c_{audio},x_{t,upper},x_{t,lower})\\&=p(x_{t-1,upper}\mid c_{all}).\end{align*} $$ 

This approximation assumes that  $ x_{t} $  already encapsulates sufficient information about  $ x_{t-1} $ , allowing us to neglect the influence of  $ x_{t-1,lower} $  when estimating  $ x_{t-1,upper} $ . This simplification is justified by the proximity of the diffusion steps and the strong correlation between the states at steps t and t-1.

The second approximation occurs in the transition from Equation 17 to Equation 18, where we decouple modality-specific influences:

 $$ \begin{aligned}&p(x_{t-1,lower}\mid c_{all})\approx p(x_{t-1,lower}\mid c_{all}\setminus\{c_{audio}\}),\\&p(x_{t-1,upper}\mid c_{all})\approx p(x_{t-1,upper}\mid c_{all}\setminus\{c_{text}\}).\\ \end{aligned} $$ 

This approximation leverages the observation that text input  $ (c_{\mathrm{text}}) $  primarily influences lower-body movements (e.g., walking or shifting stance), while audio input  $ (c_{\mathrm{audio}}) $  predominantly affects upper-body movements (e.g., gestures or facial expressions). By excluding  $ c_{audio} $  from the conditioning set for  $ x_{t-1,lower} $  and  $ c_{text} $  for  $ x_{t-1,upper} $ , we ensure the conditioning focuses on the most relevant modality for each body part.

## B RULES FOR MANAGING MULTI-MODAL ASYNCHRONOUS CONDITIONS

In this section, we outline the rules of the MOCO framework for managing multi-modal asynchronous conditions. Our rules build upon the excellent work of STMC (Petrovich et al., 2024) and extend them to accommodate multi-modal scenarios.

## Default:

1. Single Active Condition: When only one condition is active, it governs the movement of the entire body.

2. Two Active Conditions of Different Modalities: When two conditions from different modalities (e.g., speech and text) are active simultaneously, speech by default controls upper body movements (i.e., head and arms), while text by default controls lower body movements (i.e., legs and spine).

## Flexible:

To achieve more nuanced control, we leverage STMC's rules. When two conditions are active simultaneously:

1. Different Body Parts: If the conditions control different body parts, each condition governs its respective parts without conflict.

2. Overlapping Body Parts: If both conditions attempt to control the same body parts, the condition controlling fewer body parts takes precedence for those specific parts.

3. Equal Control Scope: If both conditions control an equal number of body parts, the condition with the earlier start time takes precedence. The later-starting condition will only control movement after the earlier condition has concluded.

## C LIMITATIONS OF WEIGHTED AVERAGING IN MULTI-MODAL MOTION GENERATION

<div style="text-align: center;"><img src="imgs/img_in_chart_box_212_473_401_623.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(a) T, text = "standing"</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_411_473_602_623.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(b) A, text="standing"</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_473_802_622.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(c) T, text="sitting"</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_813_473_1002_621.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(d) A, text="sitting"</div>


Figure 5: Comparison of differences calculated by the speech-to-gesture model and the text-to-motion model during motion updates. “T” denotes using text-to-motion model to update motion, while “A” denotes using speech-to-gesture to update motion. The results show that the speech-to-gesture model computes larger differences than the text-to-motion model, indicating it adjusts the motion more aggressively based on the conditions. This explains why the weighted averaging method’s generated results closely resemble those produced entirely by the speech-to-gesture model. Additionally, when the text condition is “sitting,” the speech-to-gesture model calculates larger differences in the legs than in the arms, which is counterintuitive and may be attributed to data bias in the speech-to-motion dataset.

To understand why the Weighted Average method performs similarly to the Audio-Only—achieving good results in speech-to-gesture metrics but poor performance in text-to-motion metrics—we conducted the following experiments.

Given both speech and text inputs, we updated the motion using only the text-to-motion model. At each denoising step t, we computed the difference  $ diff_{t} $  between the speech-to-gesture model's prediction—based on the speech input and the current motion from the text-to-motion model—and the current motion from the text-to-motion model. This difference quantifies how much the speech-to-gesture model perceives a mismatch between the speech condition and the current motion. Conversely, when we used only the speech-to-gesture model to update the motion, the calculated difference indicated how much the text-to-motion model perceived a mismatch between the text condition and the current motion. A larger difference suggests a greater mismatch and that the model will update the motion more aggressively.

We recorded these differences in both scenarios and divided them into whole body, arms, and legs for clearer illustration. Comparing Figures 5 (a) and (b), as well as Figures 5 (c) and (d), we found that the differences calculated by the speech-to-gesture model are larger than those by the text-to-motion model. This indicates that the speech-to-gesture model adjusts the motion more aggressively based on its conditions than the text-to-motion model does. This explains why, when using the weighted averaging method, the generated result closely resembles that produced entirely by the speech-to-gesture model.

Furthermore, by comparing Figures 5 (a) and (c), which have different text conditions, we observe that when the text condition is "sitting," the differences calculated by the speech-to-gesture model in the legs are larger than in the arms. This is counterintuitive since speech is typically associated with upper-body gestures rather than lower-body movements. Conversely, when the text condition is "standing," the differences in the legs are smaller than in the arms, aligning with expectations. This

phenomenon may be attributed to data bias in the speech-to-motion dataset, where most motions are performed in standing positions.

These observations reveal the limitations of weighted averaging in multi-modal motion generation and suggest the validity of our proposed decoupled denoising process.

## D COMPUTATIONAL COMPLEXITY


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Parameters (M)</td><td style='text-align: center;'>Model Size (MB)</td><td style='text-align: center;'>FLOPs (G)</td><td style='text-align: center;'>Inference Time (ms/frame)</td></tr><tr><td style='text-align: center;'>$ G_{T2M} $</td><td style='text-align: center;'>27.01</td><td style='text-align: center;'>103.02</td><td style='text-align: center;'>5.19</td><td style='text-align: center;'>2.26</td></tr><tr><td style='text-align: center;'>$ G_{S2G} $</td><td style='text-align: center;'>36.86</td><td style='text-align: center;'>140.62</td><td style='text-align: center;'>6.72</td><td style='text-align: center;'>4.30</td></tr><tr><td style='text-align: center;'>$ G_{T2V} $</td><td style='text-align: center;'>0.34</td><td style='text-align: center;'>1.31</td><td style='text-align: center;'>0.06</td><td style='text-align: center;'>6.20</td></tr><tr><td style='text-align: center;'>$ G_{S2D} $</td><td style='text-align: center;'>36.94</td><td style='text-align: center;'>140.94</td><td style='text-align: center;'>6.74</td><td style='text-align: center;'>4.37</td></tr></table>

<div style="text-align: center;">Table 3: Complexity of each denoiser of MOCO.</div>


Our framework, MOCO, comprises four transformer-based denoisers:  $ G_{T2M} $  for text-to-motion (T2M),  $ G_{S2G} $  for speech-to-gesture (S2G),  $ G_{T2V} $  for trajectory-to-velocity (T2V), and  $ G_{S2D} $  for speech-to-details (S2D), which manages facial expressions and hand poses. To clearly illustrate the computational complexity of MOCO, we present various metrics, including the number of parameters, model size, FLOPs, and inference time on a single NVIDIA 4090 GPU, as shown in Table 3.

As indicated in the table, our framework is overall lightweight and sufficiently fast. Specifically, the speech-to-gesture denoiser  $ G_{S2G} $  and the speech-to-details denoiser  $ G_{S2D} $  are relatively larger than the other denoisers due to additional cross-attention parameters. In contrast, the trajectory-to-velocity denoiser  $ G_{T2V} $  is the most lightweight module, featuring fewer hidden state dimensions and transformer layers because the task it handles involves low-dimensional data. However, the introduction of a guidance mechanism for more accurate predictions results in  $ G_{T2V} $  having the longest inference time.

Finally, to generate the motion sequences for a 35-second demo video consisting of nine clips under different conditions and with a total duration of 54 seconds, our method completed the body motion generation task in only 3.72 seconds. This fast generation time highlights the potential of our approach for real-time applications.

## E ADDITIONAL EXPERIMENTS

### E.1 EVALUATION OF TRAJECTORY CONTROL


<table border=1 style='margin: auto; width: max-content;'><tr><td colspan="2">Method</td><td colspan="2">Location</td><td colspan="2">Orientation</td></tr><tr><td style='text-align: center;'>CFG</td><td style='text-align: center;'>L-BFGS</td><td style='text-align: center;'>Average Difference</td><td style='text-align: center;'>Goal Difference</td><td style='text-align: center;'>Average Difference</td><td style='text-align: center;'>Goal Difference</td></tr><tr><td style='text-align: center;'>✗</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>0.5641</td><td style='text-align: center;'>1.2068</td><td style='text-align: center;'>0.7059</td><td style='text-align: center;'>1.2583</td></tr><tr><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>0.5676</td><td style='text-align: center;'>1.3177</td><td style='text-align: center;'>0.8276</td><td style='text-align: center;'>1.5115</td></tr><tr><td style='text-align: center;'>✗</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>0.0747</td><td style='text-align: center;'>0.1235</td><td style='text-align: center;'>0.6009</td><td style='text-align: center;'>1.1031</td></tr><tr><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>0.1121</td><td style='text-align: center;'>0.1950</td><td style='text-align: center;'>0.7111</td><td style='text-align: center;'>1.2845</td></tr></table>

<div style="text-align: center;">Table 4: Evaluation of Trajectory Control.</div>


Table 4 evaluates trajectory control methodologies by assessing the effects of classifier-free guidance (CFG) and L-BFGS optimization on both location (meters) and orientation (radians). For each category, two primary metrics are reported: Average Difference, quantifying the mean deviation between the generated trajectory and the ground truth (GT), and Goal Difference, measuring the discrepancy at the final point relative to the GT. The results show that L-BFGS optimization significantly reduces location differences and modestly improves orientation accuracy. Notably, for the same trajectory, orientation can be diverse, so the generated orientation does not need to closely match the GT. In contrast, incorporating CFG does not enhance trajectory accuracy. These findings indicate that while L-BFGS is a robust optimization strategy for trajectory control, integrating CFG may not provide complementary advantages and could interfere with the optimization process.


<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2"></td><td colspan="5">Text2Motion</td><td colspan="3">Speech2Gesture</td><td style='text-align: center;'>Transition</td></tr><tr><td style='text-align: center;'>FID+↓</td><td style='text-align: center;'>R1↑</td><td style='text-align: center;'>R3↑</td><td style='text-align: center;'>M2T↑</td><td style='text-align: center;'>M2M↑</td><td style='text-align: center;'>FID-A↓</td><td style='text-align: center;'>BC↑</td><td style='text-align: center;'>L1div↑</td><td style='text-align: center;'>MTD↓</td></tr><tr><td style='text-align: center;'>Ground Truth</td><td style='text-align: center;'>0.000</td><td style='text-align: center;'>40.0</td><td style='text-align: center;'>72.5</td><td style='text-align: center;'>0.781</td><td style='text-align: center;'>1.000</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>2.9</td></tr><tr><td style='text-align: center;'>Synchronous</td><td style='text-align: center;'>0.896</td><td style='text-align: center;'>23.8</td><td style='text-align: center;'>45.9</td><td style='text-align: center;'>0.638</td><td style='text-align: center;'>0.629</td><td style='text-align: center;'>4.41</td><td style='text-align: center;'>2.62</td><td style='text-align: center;'>9.47</td><td style='text-align: center;'>5.5</td></tr><tr><td style='text-align: center;'>Asynchronous</td><td style='text-align: center;'>0.862</td><td style='text-align: center;'>24.6</td><td style='text-align: center;'>46.9</td><td style='text-align: center;'>0.649</td><td style='text-align: center;'>0.639</td><td style='text-align: center;'>3.83</td><td style='text-align: center;'>2.72</td><td style='text-align: center;'>8.62</td><td style='text-align: center;'>5.3</td></tr></table>

<div style="text-align: center;">Table 5: Comparison of MOCO in synchronous and asynchronous conditions.</div>


### E.2 PERFORMANCE UNDER SYNCHRONOUS AND ASYNCHRONOUS CONDITIONS

In Table 5, we compare the performance of MOCO under synchronous and asynchronous conditions. As illustrated in the table, MOCO generates slightly better motions under asynchronous conditions compared to synchronous ones. This improvement may be attributed to asynchronous conditions allowing a single modality to control the entire body, rather than using multiple modalities to control different parts simultaneously. Such an approach is likely simpler for the model, as it was trained on data where single modalities govern the whole body. Additionally, motions generated under single-modality conditions more closely align with the distribution of the GT in the test set, which also consists of motions under single-modality conditions. Consequently, this alignment results in better performance metrics.

### E.3 SINGLE MODALITY PERFORMANCE


<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Methods</td><td colspan="3">R-Precision</td><td rowspan="2">FID $ \downarrow $</td><td rowspan="2">MM Dist $ \downarrow $</td><td rowspan="2">Diversity $ \uparrow $</td><td rowspan="2">MM $ \uparrow $</td></tr><tr><td style='text-align: center;'>Top 1</td><td style='text-align: center;'>Top 2</td><td style='text-align: center;'>Top 3</td></tr><tr><td style='text-align: center;'>Ground Truth</td><td style='text-align: center;'>0.511±0.003</td><td style='text-align: center;'>0.703±0.003</td><td style='text-align: center;'>0.797±0.002</td><td style='text-align: center;'>0.002±0.000</td><td style='text-align: center;'>2.974±0.008</td><td style='text-align: center;'>9.503±0.065</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>T2M-GPT (Zhang et al., 2023a)</td><td style='text-align: center;'>0.491±0.003</td><td style='text-align: center;'>0.680±0.003</td><td style='text-align: center;'>0.775±0.002</td><td style='text-align: center;'>0.116±0.004</td><td style='text-align: center;'>3.118±0.011</td><td style='text-align: center;'>9.761±0.081</td><td style='text-align: center;'>1.856±0.011</td></tr><tr><td style='text-align: center;'>MDM (Tevet et al., 2022)</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>0.611±0.007</td><td style='text-align: center;'>0.544±0.044</td><td style='text-align: center;'>5.566±0.027</td><td style='text-align: center;'>9.559±0.086</td><td style='text-align: center;'>2.799±0.072</td></tr><tr><td style='text-align: center;'>MOCO (Ours)</td><td style='text-align: center;'>0.434±0.010</td><td style='text-align: center;'>0.618±0.008</td><td style='text-align: center;'>0.720±0.008</td><td style='text-align: center;'>0.530±0.044</td><td style='text-align: center;'>3.563±0.049</td><td style='text-align: center;'>9.856±1.066</td><td style='text-align: center;'>2.663±0.068</td></tr><tr><td style='text-align: center;'>FineMoGen (Zhang et al., 2023b)</td><td style='text-align: center;'>0.504±0.002</td><td style='text-align: center;'>0.690±0.002</td><td style='text-align: center;'>0.784±0.002</td><td style='text-align: center;'>0.151±0.008</td><td style='text-align: center;'>2.998±0.008</td><td style='text-align: center;'>9.263±0.094</td><td style='text-align: center;'>2.696±0.079</td></tr><tr><td style='text-align: center;'>MoMask (Guo et al., 2024)</td><td style='text-align: center;'>0.521±0.002</td><td style='text-align: center;'>0.713±0.002</td><td style='text-align: center;'>0.807±0.002</td><td style='text-align: center;'>0.045±0.002</td><td style='text-align: center;'>2.958±0.008</td><td style='text-align: center;'>-</td><td style='text-align: center;'>1.241±0.040</td></tr><tr><td style='text-align: center;'>LMM-Tiny (Zhang et al., 2025)</td><td style='text-align: center;'>0.496±0.002</td><td style='text-align: center;'>0.685±0.002</td><td style='text-align: center;'>0.785±0.002</td><td style='text-align: center;'>0.415±0.002</td><td style='text-align: center;'>3.087±0.012</td><td style='text-align: center;'>9.176±0.074</td><td style='text-align: center;'>1.465±0.048</td></tr><tr><td style='text-align: center;'>LMM-Large (Zhang et al., 2025)</td><td style='text-align: center;'>0.525±0.002</td><td style='text-align: center;'>0.719±0.002</td><td style='text-align: center;'>0.811±0.002</td><td style='text-align: center;'>0.040±0.002</td><td style='text-align: center;'>2.943±0.012</td><td style='text-align: center;'>9.814±0.076</td><td style='text-align: center;'>2.683±0.054</td></tr></table>

<div style="text-align: center;">Table 6: Quantitative results of text-to-motion generation on the HumanML3D test set.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Methods</td><td style='text-align: center;'>FGD $ \downarrow $</td><td style='text-align: center;'>BC</td><td style='text-align: center;'>Diversity $ \uparrow $</td><td style='text-align: center;'>MSE $ \downarrow $</td><td style='text-align: center;'>LVD $ \downarrow $</td></tr><tr><td style='text-align: center;'>FaceFormer (Fan et al., 2022)</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>7.787</td><td style='text-align: center;'>7.593</td></tr><tr><td style='text-align: center;'>CodeTalker (Xing et al., 2023)</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>8.026</td><td style='text-align: center;'>7.766</td></tr><tr><td style='text-align: center;'>S2G (Ginosar et al., 2019a)</td><td style='text-align: center;'>28.15</td><td style='text-align: center;'>4.683</td><td style='text-align: center;'>5.971</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>Trimodal (Yoon et al., 2020)</td><td style='text-align: center;'>12.41</td><td style='text-align: center;'>5.933</td><td style='text-align: center;'>7.724</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>HA2G (Liu et al., 2022c)</td><td style='text-align: center;'>12.32</td><td style='text-align: center;'>6.779</td><td style='text-align: center;'>8.626</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>DisCo (Liu et al., 2022a)</td><td style='text-align: center;'>9.417</td><td style='text-align: center;'>6.439</td><td style='text-align: center;'>9.912</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>CaMN (Liu et al., 2022b)</td><td style='text-align: center;'>6.644</td><td style='text-align: center;'>6.769</td><td style='text-align: center;'>10.86</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>DiffStyleGesture (Yang et al., 2023)</td><td style='text-align: center;'>8.811</td><td style='text-align: center;'>7.241</td><td style='text-align: center;'>11.49</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>TalkShow (Yi et al., 2023)</td><td style='text-align: center;'>6.209</td><td style='text-align: center;'>6.947</td><td style='text-align: center;'>13.47</td><td style='text-align: center;'>7.791</td><td style='text-align: center;'>7.771</td></tr><tr><td style='text-align: center;'>EMAGE (Liu et al., 2023a)</td><td style='text-align: center;'>5.512</td><td style='text-align: center;'>7.724</td><td style='text-align: center;'>13.06</td><td style='text-align: center;'>7.680</td><td style='text-align: center;'>7.556</td></tr><tr><td style='text-align: center;'>ProbTalk (Liu et al., 2024)</td><td style='text-align: center;'>6.170</td><td style='text-align: center;'>8.099</td><td style='text-align: center;'>10.43</td><td style='text-align: center;'>8.990</td><td style='text-align: center;'>8.385</td></tr><tr><td style='text-align: center;'>MOCO (Ours)</td><td style='text-align: center;'>5.543</td><td style='text-align: center;'>7.089</td><td style='text-align: center;'>14.05</td><td style='text-align: center;'>7.285</td><td style='text-align: center;'>7.573</td></tr></table>

<div style="text-align: center;">Table 7: Quantitative results of speech-to-gesture generation on the BEATX test set.</div>


To demonstrate MOCO's performance in single-modality scenarios, we trained it from scratch on HumanML3D for text-to-motion and on BEATX for speech-to-gesture, respectively, ensuring a fair comparison. The results, presented in Tables 6 and 7, show that in the HumanML3D text-to-motion benchmark (Table 6), our model achieves performance comparable to the widely-used MDM. This outcome is expected since our text-to-motion denoiser,  $ G_{T2M} $ , is based on MDM. In the BEATX speech-to-gesture benchmark (Table 7), MOCO attains competitive performance compared to state-of-the-art methods.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Method</td><td style='text-align: center;'>Better Text Following (%)</td><td style='text-align: center;'>Better Beat Synchronization (%)</td></tr><tr><td style='text-align: center;'>Neither</td><td style='text-align: center;'>1.0</td><td style='text-align: center;'>13.0</td></tr><tr><td style='text-align: center;'>Pseudo-Text</td><td style='text-align: center;'>0.0</td><td style='text-align: center;'>12.5</td></tr><tr><td style='text-align: center;'>MOCO (Ours)</td><td style='text-align: center;'>99.0</td><td style='text-align: center;'>74.5</td></tr><tr><td style='text-align: center;'>Neither</td><td style='text-align: center;'>0.0</td><td style='text-align: center;'>13.5</td></tr><tr><td style='text-align: center;'>Weighted Average</td><td style='text-align: center;'>0.0</td><td style='text-align: center;'>3.5</td></tr><tr><td style='text-align: center;'>MOCO (Ours)</td><td style='text-align: center;'>100.0</td><td style='text-align: center;'>83.0</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'>Better Body Coherence (%)</td><td style='text-align: center;'>Better Temporal Fluidity (%)</td></tr><tr><td style='text-align: center;'>Neither</td><td style='text-align: center;'>12.0</td><td style='text-align: center;'>12.6</td></tr><tr><td style='text-align: center;'>Combine Only Last Time</td><td style='text-align: center;'>17.0</td><td style='text-align: center;'>42.6</td></tr><tr><td style='text-align: center;'>MOCO (Ours)</td><td style='text-align: center;'>71.0</td><td style='text-align: center;'>44.8</td></tr></table>

<div style="text-align: center;">Table 8: User study.</div>


### E.4 User Study

Table 8 presents the results of a user study comparing MOCO with three baseline methods. Specifically, we evaluate MOCO against Pseudo-Text and Weighted Average, introduced in Section 4.3, to assess overall performance in text following and audio beat synchronization. Additionally, we compare MOCO with Combine Only Last Time, which also employs the decoupling strategy but applies the combining strategy only at the final diffusion step. This comparison aims to evaluate whole body coherence and temporal fluidity.

As shown in the table, MOCO achieves significant advantages over both Pseudo-Text and Weighted Average, demonstrating the effectiveness of our decouple-then-combine strategy in generating motion aligned with multi-modal conditions. Furthermore, when compared to Combine Only Last Time, our method was rated significantly higher in both body coherence and temporal fluidity. This indicates that MOCO does more than merely combine different body parts controlled by separate conditions; it ensures that each body part aligns with its corresponding condition while enhancing coordination among all body parts.

## F DETAILS OF MULTI-MODAL BENCHMARK

To effectively evaluate our proposed task, we developed a multi-modal benchmark comprising 1,000 test clips, following the methodology outlined in Petrovich et al. (2024). Each test clip is automatically generated and includes two text descriptions and two audio clips.

For the text descriptions, we manually curated a set of 40 texts focusing on lower-body movements commonly associated with speech delivery or conversation. These descriptions provide the necessary context for evaluating the corresponding movements within the benchmark. Regarding the audio clips, we selected recordings from the BEATX dataset, specifically choosing eight speakers with speaker IDs below 10. These audio files were segmented into clips using a Voice Activity Detector (VAD), resulting in 694 audio clips with an average duration of 9.14 seconds.

The 1,000 test clips were generated through an automated process that utilizes the curated text descriptions and audio clips. For each test clip, two text descriptions are randomly selected and assigned random durations. Subsequently, two neighboring audio clips are randomly chosen. The start times for both the text and audio intervals are determined randomly, allowing the sequence to commence with either text or audio. This process results in the creation of four intervals that correspond to the selected text descriptions and audio clips.

Optional text descriptions are listed in the next page.