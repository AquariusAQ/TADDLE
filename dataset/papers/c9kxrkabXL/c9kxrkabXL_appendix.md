## A ADDITIONAL BACKGROUND, EXPERIMENTAL SETUP AND TRAINING DETAILS

### A.1 S4 LAYER

S4Gu et al. (2021a) layer is a variant of linear and time-invariant (LTI) state-space model (SSM)Gu et al. (2021b) which adopts the HIPPO Gu et al. (2020)-based initializations in order to better capture longer contexts, and proposes efficient ways for kernel computations and parallel training.

#### A.1.1 RECURRENT VIEW

Given an input scalar function  $  u(t) : \mathbb{R} \to \mathbb{R}  $ , the continuous LTI SSM is defined by the following first-order differential equation:

 $$ \dot{x}(t)=A x(t)+B u(t),\quad y(t)=C x(t)+D u(t) $$ 

The model maps the input stream  $  u(t)  $  to  $  y(t)  $ . It was shown that initializing A by the HIPPO matrix Gu et al. (2020) grants the state-space model (SSM) the ability to capture long-range dependencies. Similar to previous works Gu et al. (2021a); Gupta et al. (2022), D is replaced by parameter-based skip-connection and is omitted from the SSM by assuming D = 0.

This SSM operates on continuous sequences, and it is discretized by a step size  $ \Delta $  to operate on discrete sequences. Let the discretization matrices be  $ \bar{A}, \bar{B}, \bar{C} $ :

 $$ \bar{A}=(I-\Delta A/2)^{-1}(I+\Delta A/2),\quad\bar{B}=(I-\Delta A/2)^{-1}\Delta B,\quad\bar{C}=C $$ 

These matrices allow us to rewrite Eq. 10:

 $$ x_{k}=\bar{A}x_{k-1}+\bar{B}u_{k},\quad y_{k}=\bar{C}x_{k} $$ 

Using the recurrent Eq.12, SSM asymptotically allows for constant  $ O(1) $  time and memory inference for each token/timestep, as compared to  $ O(L^{2}) $  inference for transformers. SSM can be interpreted as a linear RNN in which  $ \bar{A} $  is the state-transition matrix, and  $ \bar{B}, \bar{C} $  are the input and output matrices. Thus, it essentially requires  $ O(L) $  training, L being the sequence length, as compared to  $ O(L^{2}) $  (parallelizable) training complexity for transformers.

#### A.1.2 CONVOLUTIONAL VIEW

The recurrent SSM view is not practical for training over long sequences, as the training cannot be parallelized across the sequence dimension and results in instabilities from vanishing gradient issues. However, the LTI SSM can be rewritten as a convolution, which allows for efficient parallelizable training. The S4 convolutional view is obtained as follows:

Given a sequence of scalars  $  u = (u_{0}, u_{1}, \ldots, u_{L-1})  $  of length L, the S4 recurrent view can be unrolled to the following closed form:

 $$ \forall i\in[L-1]:x_{i}\in\mathbb{R}^{N},\quad x_{0}=\bar{B}u_{0},\quad x_{1}=\bar{A}\bar{B}u_{0}+\bar{B}u_{1},\quad...,\quad x_{L-1}=\sum_{i=0}^{L-1}\bar{A}^{L-1-i}\bar{B}u_{i} $$ 

 $$ y_{i}\in\mathbb{R},\quad y_{0}=\bar{C}\bar{B}u_{0},\quad y_{1}=\bar{C}\bar{A}\bar{B}u_{0}+\bar{C}\bar{B}u_{1},\quad...,\quad y_{L-1}=\sum_{i=0}^{L-1}\bar{C}\bar{A}^{L-1-i}\bar{B}u_{i} $$ 

Where N is the state size. Inputs and outputs are scalars.

Since the recurrent rule is linear, it can be computed in closed form with matrix multiplication or non-circular convolution:

<div style="text-align: center;"><img src="imgs/img_in_chart_box_409_167_803_451.jpg" alt="Image" width="32%" /></div>


<div style="text-align: center;">Figure 6: Average Returns obtained in SMAC tasks by passing S4 output versus S4 latent states.</div>


 $$ \begin{aligned}\begin{bmatrix}{{{y_{0}}}} \\{{{y_{1}}}} \\{{{\vdots}}} \\{{{y_{L-1}}}}\end{bmatrix}=\begin{bmatrix}{{{\bar{C}\bar{B}}}}&{{{0}}}&{{{0}}}&{{{0}}}&{{{0}}} \\{{{\bar{C}\bar{A}\bar{B}}}}&{{{\bar{C}\bar{B}}}}&{{{0}}}&{{{0}}}&{{{0}}} \\{{{\bar{C}\bar{A}^{2}\bar{B}}}}&{{{\bar{C}\bar{A}\bar{B}}}}&{{{\bar{C}\bar{B}}}}&{{{0}}}&{{{0}}} \\{{{\vdots}}}&{{{\vdots}}}&{{{\vdots}}}&{{{\vdots}}}&{{{\vdots}}} \\{{{\bar{C}\bar{A}^{L-2}\bar{B}}}}&{{{\bar{C}\bar{A}^{L-3}\bar{B}}}}&{{{\bar{C}\bar{A}^{L-4}\bar{B}}}}&{{{\ldots}}}&{{{\bar{C}\bar{B}}}}\end{bmatrix}\begin{bmatrix}{{{u_{0}}}} \\{{{u_{1}}}} \\{{{\vdots}}} \\{{{u_{L-1}}}}\end{bmatrix}\end{aligned} $$ 

i.e.,  $ y = \bar{k} * u $  for some kernel  $ \bar{k} $ , which can be calculated by fixing the sequence length L before training. This kernel can be efficiently computed using FFT operations; for example, Gu et al. (2021a) computes the kernel via inverse FFT on the spectrum of  $ \bar{k} $ , which is calculated via Cauchy kernel and the Woodbury Identity. This benefits from the "Normal Plus Low Rank" parameterization of the HIPPO-initialized state transition matrix A, and other more efficient parameterizations are proposed in Gupta et al. (2022).

The SSM, as represented above, operates on scalars or one channel of inputs. To handle vector inputs  $ \inR^{H} $ , H copies of the 1-D SSM layer are stacked, one for each input channel, and a linear mixing layer in the after block of the S4 layer mixes the information from different channels to produce outputs  $ \inR^{H} $ .

### A.2 SHARING HIDDEN STATE REPRESENTATIONS

The raw outputs from the S4 layer consist of  $ y_{k} = \bar{C} x_{k} $ , where  $ y_{k} \in R^{H} $  and the latent states  $ x_{k} \in R^{N \times H} $  for H input channels. Since the outputs are linear projections and offer a compact representation of the latent states (or, memory of the agent), this has been used as the message that is transmitted from one agent to the next in the SE-MDP. This offers several advantages: i) results in better team performance; ii) offers scalable cooperation between agents, which eliminates the need for a centralized transformer or a critic, which requires access to information from all agents; one agent needs access to only its immediate neighbor in the sequence; (iii) allows parallel training via convolution.

We also experimented with passing the raw hidden states  $ x_{k} \in R^{N \times H} $  from one agent to another. The hidden states can be complex, depending on the parameterization of the S4 kernel. Therefore, before passing the latent states directly, we first linearly mix the hidden states across the H channels to obtain  $ x_{k} \in C^{N} $ . Then, we linearly project the real and imaginary parts of  $ x_{k} $  after concatenation. This mode of information transfer, however, has notable drawbacks: i) it requires computing the S4 hidden state at every timestep, which requires recurrent rollouts of the S4 kernel, and ii) it fails to outperform the method of passing the S4 outputs; possibly due to errors accumulated during recurrent training. A comparison of performance using S4 output representation versus S4 latent state representation is shown in Figure6, where passing S4 outputs resulted in better performance across all tasks.

It is, however, noted that hidden states at each timestep may be efficiently obtained utilizing the parallel (associate) scan operation as done in Smith et al. (2022); Lu et al. (2024), but this requires JAX implementation and is currently not supported by PyTorch.

### A.3 PRELIMINARY STUDY USING MAMBA

We also explored Mamba as an alternative to LTI S4-based models. Mamba allows time-variant parameters to be considered in the SSM equations. Though convolution cannot be applied here since the kernel cannot be computed apriori since the parameters B, C are input-dependent, efficient parallel scan operation allows for parallelizable  $ O(\log L) $  complexity. However, preliminary analysis utilizing Mamba resulted in suboptimal performance, and it requires more extensive analysis.

### A.4 EXPERIMENTAL SETUP AND TRAINING

In all experiments, we set the input channel size to H = 96 and the S4 state size to N = 96. Offline training is conducted on batches of 64 trajectories, with the maximum trajectory length in the offline dataset used as the length for each batch. The shorter trajectories are zero-padded to a constant length. The training was performed using Adam optimizer with a learning rate of  $ 10^{-4} $ .

The offline trained model is fine-tuned online using on-policy MAPPO. During the initial stage of fine-tuning, the actor network is kept frozen, and the critic is first trained for the first 50,000 iterations. After this, both the actor and critic are trained simultaneously, with a slower learning rate for the actor network  $ (10^{-5}) $  compared to the critic  $ (10^{-4}) $ . During on-policy fine-tuning, the returns-to-go is set at 10% higher than the highest returns encountered during training. On-policy training is conducted in batches of 64. To mitigate the issue of deteriorating performance with prolonged on-policy training, the S4 kernel A can be kept frozen. All experiments were run on a single NVIDIA RTX 2080Ti GPU. Experiments on the RWARE domain take less than 2 hrs to reach optimal performance, and experiments on the SMAC domain take less than 6 hrs, 12 hrs, 12 hrs, and 30 hrs for maps 2c vs. 64zg, 5m vs. 6m, 6h vs. 8z and Corridor, respectively.

## B DATASETS AND BASELINES

### B.1 MULTI-ROBOTWAREHOUSE (RWARE)

The offline dataset on RWARE (Papoudakis et al., 2020) is obtained from (Matsunaga et al., 2023), which contains an expert dataset with diverse behaviors obtained by training MAT on small and tiny maps. The dataset consists of 1000 trajectories, each trajectory consisting of 500 timesteps. The dataset statistics are in Table 3. The longest trajectories consist of timesteps in the range of 500 in all the datasets.

The baseline results are obtained from (Matsunaga et al., 2023), which currently holds the state-of-the-art results of the baselines listed on this dataset.


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Map Name</td><td style='text-align: center;'>Maximum</td><td style='text-align: center;'>Minimum</td><td style='text-align: center;'>Average</td></tr><tr><td style='text-align: center;'>small 2 agents</td><td style='text-align: center;'>12.37</td><td style='text-align: center;'>1.13</td><td style='text-align: center;'>7.12</td></tr><tr><td style='text-align: center;'>small 4 agents</td><td style='text-align: center;'>12.08</td><td style='text-align: center;'>3.93</td><td style='text-align: center;'>9.49</td></tr><tr><td style='text-align: center;'>small 6 agents</td><td style='text-align: center;'>12.69</td><td style='text-align: center;'>7.59</td><td style='text-align: center;'>10.76</td></tr><tr><td style='text-align: center;'>tiny 2 agents</td><td style='text-align: center;'>16.81</td><td style='text-align: center;'>1.97</td><td style='text-align: center;'>12.77</td></tr><tr><td style='text-align: center;'>tiny 4 agents</td><td style='text-align: center;'>18.63</td><td style='text-align: center;'>10.40</td><td style='text-align: center;'>15.67</td></tr><tr><td style='text-align: center;'>tiny 6 agents</td><td style='text-align: center;'>19.97</td><td style='text-align: center;'>11.88</td><td style='text-align: center;'>17.45</td></tr></table>

<div style="text-align: center;">Table 3: RWARE datasets</div>


### B.2 SMAC

The offline SMAC (Samvelyan et al., 2019) dataset is obtained from (Wang et al., 2024). This dataset is obtained by randomly sampling 1000 trajectories from the original dataset provided by (Meng et al., 2021). We consider 4 representative battle maps, including 2 hard maps (5m vs 6m,

<div style="text-align: center;">2c vs 64zg) and 2 super hard maps (6h vs 8z, corridor), which are detailed in Table 4. The average returns for the dataset are listed in Table 5. The longest trajectories are encountered in the Corridor map, which typically comprises about 100 timesteps.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Map Name</td><td style='text-align: center;'>Ally Units</td><td style='text-align: center;'>Enemy Units</td><td style='text-align: center;'>Type</td></tr><tr><td style='text-align: center;'>5m_vs_6m</td><td style='text-align: center;'>5 Marines</td><td style='text-align: center;'>6 Marines</td><td style='text-align: center;'>homogeneous &amp; asymmetric</td></tr><tr><td style='text-align: center;'>2c_vs_64zg</td><td style='text-align: center;'>2 Colossi</td><td style='text-align: center;'>64 Zerglings</td><td style='text-align: center;'>micro-trick: positioning</td></tr><tr><td style='text-align: center;'>6h_vs_8z</td><td style='text-align: center;'>6 Hydralisks</td><td style='text-align: center;'>8 Zealots</td><td style='text-align: center;'>micro-trick: focus fire</td></tr><tr><td style='text-align: center;'>corridor</td><td style='text-align: center;'>6 Zealots</td><td style='text-align: center;'>24 Zerglings</td><td style='text-align: center;'>micro-trick: wall off</td></tr></table>

<div style="text-align: center;">Table 4: SMAC maps for experiments.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Map Name</td><td style='text-align: center;'>Quality</td><td style='text-align: center;'>Average Return</td></tr><tr><td rowspan="3">5m_vs_6m</td><td style='text-align: center;'>good</td><td style='text-align: center;'>20.00</td></tr><tr><td style='text-align: center;'>medium</td><td style='text-align: center;'>11.03</td></tr><tr><td style='text-align: center;'>poor</td><td style='text-align: center;'>8.50</td></tr><tr><td rowspan="3">2c_vs_64zg</td><td style='text-align: center;'>good</td><td style='text-align: center;'>19.94</td></tr><tr><td style='text-align: center;'>medium</td><td style='text-align: center;'>13.00</td></tr><tr><td style='text-align: center;'>poor</td><td style='text-align: center;'>8.89</td></tr><tr><td rowspan="3">6h_vs_8z</td><td style='text-align: center;'>good</td><td style='text-align: center;'>17.84</td></tr><tr><td style='text-align: center;'>medium</td><td style='text-align: center;'>11.96</td></tr><tr><td style='text-align: center;'>poor</td><td style='text-align: center;'>9.12</td></tr><tr><td rowspan="3">corridor</td><td style='text-align: center;'>good</td><td style='text-align: center;'>19.88</td></tr><tr><td style='text-align: center;'>medium</td><td style='text-align: center;'>13.07</td></tr><tr><td style='text-align: center;'>poor</td><td style='text-align: center;'>4.93</td></tr></table>

<div style="text-align: center;">Table 5: SMAC datasets.</div>


The offline RL-based baseline results are obtained from (Wang et al., 2024), and MADT results are obtained by running the code available with (Meng et al., 2021).

## C HYPERPARAMETERS AND ADDITIONAL ANALYSIS

### C.1 S4 MODEL SIZE PARAMETERS

We analyze the impact of the S4 model size parameters, specifically the number of input channels  $ (H) $  and the latent state size  $ (N) $ , on the model performance, as shown in Table 6. We compare the total number of parameters against the 1.8 million parameters reported for MADTKD in Tseng et al. (2022). Our biggest model with N=96 and H=96 was used in all our experiments, which consists of about 200k parameters.

### C.2 THE EFFECT OF CONTEXT LENGTH AND S4 PARAMETERS

The context length used for pretraining significantly impacts performance, which is also evident for transformer-based models. In our experiments, we used the maximum trajectory lengths encountered in the offline datasets for pretraining. Representative results are shown in Figure 7, which illustrates the effects of truncating the trajectory lengths to various percentages of the maximum length in the offline dataset for the SMAC map 2c vs 64zg.

<div style="text-align: center;">Table 6: Results of smaller models on the RWARE small map. Each of the smaller models is denoted by (i) N, the S4 state size, and (ii) H, the number of input/output channels.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Environments</td><td style='text-align: center;'>(N=96,H=96)</td><td style='text-align: center;'>(N=64,H=64)</td><td style='text-align: center;'>(N=32,H=32)</td><td style='text-align: center;'>(N=64,H=96)</td><td style='text-align: center;'>(N=96,H=64)</td><td style='text-align: center;'>(N=32,H=64)</td><td style='text-align: center;'>MADTKD</td></tr><tr><td style='text-align: center;'>2 agents</td><td style='text-align: center;'>6.58</td><td style='text-align: center;'>6.21</td><td style='text-align: center;'>5.53</td><td style='text-align: center;'>6.53</td><td style='text-align: center;'>6.25</td><td style='text-align: center;'>5.87</td><td style='text-align: center;'>3.65</td></tr><tr><td style='text-align: center;'>4 agents</td><td style='text-align: center;'>9.47</td><td style='text-align: center;'>8.86</td><td style='text-align: center;'>8.57</td><td style='text-align: center;'>9.15</td><td style='text-align: center;'>8.88</td><td style='text-align: center;'>8.64</td><td style='text-align: center;'>6.85</td></tr><tr><td style='text-align: center;'>6 agents</td><td style='text-align: center;'>10.87</td><td style='text-align: center;'>10.31</td><td style='text-align: center;'>9.55</td><td style='text-align: center;'>10.76</td><td style='text-align: center;'>9.97</td><td style='text-align: center;'>9.85</td><td style='text-align: center;'>7.85</td></tr><tr><td style='text-align: center;'>% Parameters (Ours)</td><td style='text-align: center;'>100</td><td style='text-align: center;'>60</td><td style='text-align: center;'>40</td><td style='text-align: center;'>81</td><td style='text-align: center;'>82</td><td style='text-align: center;'>55</td><td style='text-align: center;'>100</td></tr><tr><td style='text-align: center;'>% Parameters (MADTKD)</td><td style='text-align: center;'>12</td><td style='text-align: center;'>7</td><td style='text-align: center;'>5</td><td style='text-align: center;'>8</td><td style='text-align: center;'>8</td><td style='text-align: center;'>6</td><td style='text-align: center;'>100</td></tr></table>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_458_164_766_366.jpg" alt="Image" width="25%" /></div>


<div style="text-align: center;">Figure 7: The effect of truncating the trajectory length during training. The average returns are normalized with the maximum returns encountered in the offline dataset.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_335_464_611_629.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_611_463_888_629.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;">Figure 8: The effect of freezing S4 kernel parameter A in the SMAC 6h vs 8z map. Freezing A in the right Figure results in more stable performance during the on-policy recurrent finetuning.</div>


### C.3 EFFECT OF FREEZING A DURING ON-POLICY FINETUNING

The degrading effect on MADS4 performance during recurrent on-policy finetuning can be mitigated by freezing the S4 kernel parameter A while updating only parameters B and C, as illustrated in Figure 8. A similar observation has also been reported in Bar-David et al. (2023).

### C.4 EFFECT OF ORDER OF AGENTS

To assess the impact of agent ordering on MADS4's performance, we compared two training settings: (1) Random Order, where the agent order is randomly shuffled during training, and (2) Fixed Order, where agents are trained in the same sorted order as in the offline dataset. The results in Figure 9 indicate minimal to no performance difference between the two settings, demonstrating that MADS4 is robust to agent ordering within the SE-MDP framework. Nonetheless, we recommend using a random order during training to avoid introducing potential biases into the learning process.

### C.5 EFFECT OF GLOBAL STATES AS INPUTS

Building on prior work such as MADT, the proposed S4-based MADS4 agents utilize global states as inputs. However, in certain environments, access to the global state may be restricted or unavailable.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_335_1206_610_1371.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_612_1205_888_1371.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;">Figure 9: The effect of having a shuffled random order vs. a fixed sorted order of the agents in the SE-MDP framework on the SMAC domain in the 2c vs. 64zg map (left) and RWARE domain in the small 6 agents scenario (right).</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_334_162_609_327.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_161_888_327.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;">Figure 10: Performance comparison on RWARE small map with 6 agents (left) and SMAC map 2c vs 64 zg (right). The results demonstrate that excluding global states as inputs in MADS4 agents has minimal impact on performance.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_334_438_608_602.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_614_439_888_602.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;">Figure 11: The comparison of performance of MADS4 vs. MADS4-dec (decentralized MADS4) on RWARE small map with 6 agents (left) and SMAC map 2c vs 64 zg (right).</div>


To address this, we present an ablation study (Figure 10) evaluating the impact of using global state variables as inputs. The results indicate that omitting the global state does not lead to a significant drop in performance.

#### C.6 MADS4 vs. DECENTRALIZED MADS4

When decisions are made at the current timestep, all decisions from the previous timestep will already be finalized. As a result, the memory information of all agents is readily available for use. This eliminates the need for any agent to wait for its peer to decide the current timestep. By utilizing the memory information from the previous timestep, agents can make decisions without relying on sequential dependencies during the current timestep. Since memory accumulates over multiple timesteps, relying on the previous timestep's information does not compromise performance, as demonstrated in Figure 11. This modification enables our algorithm to function effectively in decentralized policy settings without performance degradation.