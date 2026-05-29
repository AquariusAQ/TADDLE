## A APPENDIX

### A.1 FURTHER BACKGROUND

## I NDEPENDENT PPO

Proximal Policy Optimisation (PPO) Schulman et al. (2017) is a method initially developed for single-agent RL. PPO aims to address performance collapse in policy gradient methods. It does this by bounding the ratio of action probabilities between the old and new policies. PPO optimises the objective function,

 $$ \mathbb{E}_{s\sim d^{\pi},a\sim\pi}\left[\min\left(\frac{\tilde{\pi}(a\mid s)}{\pi(a\mid s)}A^{\pi}(s,a),clip\left(\frac{\tilde{\pi}(a\mid s)}{\pi(a\mid s)},1-\epsilon,1+\epsilon\right)A^{\pi}(s,a)\right)\right], $$ 

where  $ \operatorname{clip}(t,a,b) $  is a function that outputs a if t < a, b if t > b and t otherwise. We consider the extension to multi-agent setting by using independent learning (de Witt et al., 2020). Here, each agent treats the others as part of the environment and learns a critic using its local AOH.

## OBL: ZERO-SHOT COORDINATION

Most SP methods exhibit a tendency towards brittleness and over-coordination, leading to suboptimal performance when paired with independently trained policies. To address this, OBL (Hu et al., 2021) was proposed as an algorithm that learns optimal grounded policies without relying on arbitrary conventions or assumptions about other agents. OBL has demonstrated remarkable success in ZSC scenarios, making it a compelling baseline for ad-hoc teamwork, even though it does not utilise any of the human data provided for the challenge.

### A.2 ADDITIONAL RESULTS

We provide additional results for performance on validation and test sets in Table 5 and 6.

<div style="text-align: center;">Table 5: Cross-entropy loss on the validation and test sets for BC, BR-BC and HDR-IPPO agents in the two-player setting. For human proxies we report results over two available agents. For the rest, we report results over three different seeds. Even though agents are trained with different seeds, they achieve almost identical results. We report loss  $ \pm SE $ .</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Data Used</td><td style='text-align: center;'>Method</td><td colspan="2">Validation Set</td><td colspan="2">Test Set</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'>Loss</td><td style='text-align: center;'>Accuracy</td><td style='text-align: center;'>Loss</td><td style='text-align: center;'>Accuracy</td></tr><tr><td rowspan="3">1,000 Games</td><td style='text-align: center;'>BC</td><td style='text-align: center;'>0.87  $ \pm $  0.0</td><td style='text-align: center;'>0.46  $ \pm $  0.0</td><td style='text-align: center;'>0.87  $ \pm $  0.0</td><td style='text-align: center;'>0.46  $ \pm $  0.0</td></tr><tr><td style='text-align: center;'>BR-BC</td><td style='text-align: center;'>10.93  $ \pm $  0.1</td><td style='text-align: center;'>0.25  $ \pm $  0.0</td><td style='text-align: center;'>10.96  $ \pm $  0.1</td><td style='text-align: center;'>0.25  $ \pm $  0.0</td></tr><tr><td style='text-align: center;'>HDR-IPPO</td><td style='text-align: center;'>0.96  $ \pm $  0.0</td><td style='text-align: center;'>0.41  $ \pm $  0.0</td><td style='text-align: center;'>0.97  $ \pm $  0.0</td><td style='text-align: center;'>0.40  $ \pm $  0.0</td></tr><tr><td rowspan="2">Entire Dataset</td><td style='text-align: center;'>BC</td><td style='text-align: center;'>0.47  $ \pm $  0.0</td><td style='text-align: center;'>0.67  $ \pm $  0.0</td><td style='text-align: center;'>0.48  $ \pm $  0.0</td><td style='text-align: center;'>0.67  $ \pm $  0.0</td></tr><tr><td style='text-align: center;'>Human Proxy</td><td style='text-align: center;'>0.53  $ \pm $  0.0</td><td style='text-align: center;'>0.63  $ \pm $  0.0</td><td style='text-align: center;'>0.54  $ \pm $  0.0</td><td style='text-align: center;'>0.63  $ \pm $  0.0</td></tr></table>

<div style="text-align: center;">Table 6: Cross-entropy loss on the validation and test sets for BC, BR-BC and HDR-IPPO agents in the three-player setting. For human proxies we report results over two available agents. For the rest, we report results over three different seeds. Even though agents are trained with different seeds, they achieve almost identical results. We report loss  $ \pm $  SE.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Data Used</td><td style='text-align: center;'>Method</td><td colspan="2">Validation Set</td><td colspan="2">Test Set</td></tr><tr><td style='text-align: center;'></td><td style='text-align: center;'></td><td style='text-align: center;'>Loss</td><td style='text-align: center;'>Accuracy</td><td style='text-align: center;'>Loss</td><td style='text-align: center;'>Accuracy</td></tr><tr><td rowspan="3">1,000 Games</td><td style='text-align: center;'>BC</td><td style='text-align: center;'>0.71  $ \pm $  0.0</td><td style='text-align: center;'>0.39  $ \pm $  0.0</td><td style='text-align: center;'>0.70  $ \pm $  0.0</td><td style='text-align: center;'>0.40  $ \pm $  0.0</td></tr><tr><td style='text-align: center;'>BR-BC</td><td style='text-align: center;'>32.30  $ \pm $  1.4</td><td style='text-align: center;'>0.07  $ \pm $  0.0</td><td style='text-align: center;'>32.48  $ \pm $  1.6</td><td style='text-align: center;'>0.07  $ \pm $  0.0</td></tr><tr><td style='text-align: center;'>HDR-IPPO</td><td style='text-align: center;'>0.81  $ \pm $  0.0</td><td style='text-align: center;'>0.31  $ \pm $  0.0</td><td style='text-align: center;'>0.81  $ \pm $  0.0</td><td style='text-align: center;'>0.31  $ \pm $  0.0</td></tr><tr><td rowspan="2">Entire Dataset</td><td style='text-align: center;'>BC</td><td style='text-align: center;'>0.55  $ \pm $  0.0</td><td style='text-align: center;'>0.51  $ \pm $  0.0</td><td style='text-align: center;'>0.54  $ \pm $  0.0</td><td style='text-align: center;'>0.51  $ \pm $  0.0</td></tr><tr><td style='text-align: center;'>Human Proxy</td><td style='text-align: center;'>0.62  $ \pm $  0.0</td><td style='text-align: center;'>0.43  $ \pm $  0.0</td><td style='text-align: center;'>0.62  $ \pm $  0.0</td><td style='text-align: center;'>0.44  $ \pm $  0.0</td></tr></table>

### A.3 TRAINING DETAILS

## BEHAVIOURAL CLONING (BC)

For each sampled mini-batch of games, we extract individual player trajectories, effectively decomposing each game into n trajectories, where n represents the number of players. This yields sets of trajectories, denoted by  $ U_{d}=\{u_{0},\ldots,u_{n}\} $ , with one trajectory per player for each game. Each trajectory,  $ u_{i}=\{(o_{t}^{i},a_{t}^{i})\}_{t=1}^{T} $ , consists of local player observations,  $ o_{t}^{i}\inO^{i} $ , and corresponding actions,  $ a_{t}^{i}\inA^{i} $ , taken at timestep t. Our BC model takes a single player's trajectory as input and predicts an action for each timestep, conditioned on the AOH. A mini-batch of size b is constructed by concatenating multiple sets of trajectories:  $ D_{batch}=U_{0}\parallel\ldots\parallel U_{b} $ , where  $ \parallel $  denotes concatenation. To augment the training data and improve generalisation, we randomly shuffle the colour space for both observations and actions within each mini-batch before feeding it to the network, as done in the previous works to enhance performance of our model (Hu et al., 2021).

The loss for each batch is calculated as

 $$ \mathcal{L}^{B C}(\theta)=-\frac{1}{\sum_{\tau_{i}\in\mathcal{D}_{b a t c h}}\left|\tau_{i}\right|}\sum_{\tau_{i}\in\mathcal{D}_{b a t c h}}\sum_{\left(o_{t},a_{t}\right)\in\tau_{i}}\ell(\pi_{\theta}^{B C}(a|o_{t},\phi_{t}),a_{t}), $$ 

where  $ D_{batch} $  is the batch of trajectories  $ \tau_{i} $  and  $ \ell $  is the standard cross-entropy loss. The policy,  $ \pi_{\theta}^{BC} $ , is conditioned on both the hidden state  $ \phi_{t}^{i} $ , encoding the observation history, and the current local observation  $ o_{t}^{i} $ .

## BEST RESPONSE TO BEHAVIOURAL CLONING (BR-BC)

When training with an embedded human model, we initially train in SP and anneal the amount of SP linearly to zero. Then, we continue training with the BC policy, as Carroll et al. (2019) find that this improves agents' performance. Additionally, we consider the three agent settings that was not considered by Carroll et al. (2019). When annealing, based on the annealing factor, we sample agents to decide whether we train in SP, with a single BC policy or with two BC policies.

### A.4 HUMAN PROXIES: ADDITIONAL DETAILS

## Dataset

Dataset used for training human proxies contains 147,621 Hanabi games, comprising 101,096 two-player and 46,525 three-player games. Table 7 summarises key statistics for scores and game lengths across both player configurations.

<div style="text-align: center;">Table 7: The dataset used for training human proxies comprises a total of 147,621 games, including 101,096 two-player and 46,525 three-player games. The table presents descriptive statistics for game scores and lengths across both player configurations.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Setting</td><td style='text-align: center;'>Metric</td><td style='text-align: center;'>Min</td><td style='text-align: center;'>Max</td><td style='text-align: center;'>Avg</td><td style='text-align: center;'>Median</td><td style='text-align: center;'>Std</td></tr><tr><td rowspan="2">Two-Player</td><td style='text-align: center;'>Scores</td><td style='text-align: center;'>1</td><td style='text-align: center;'>25</td><td style='text-align: center;'>23.09</td><td style='text-align: center;'>24.00</td><td style='text-align: center;'>2.16</td></tr><tr><td style='text-align: center;'>Game Lengths</td><td style='text-align: center;'>2</td><td style='text-align: center;'>88</td><td style='text-align: center;'>65.70</td><td style='text-align: center;'>66.00</td><td style='text-align: center;'>3.63</td></tr><tr><td rowspan="2">Three-Player</td><td style='text-align: center;'>Scores</td><td style='text-align: center;'>2</td><td style='text-align: center;'>25</td><td style='text-align: center;'>22.94</td><td style='text-align: center;'>23.00</td><td style='text-align: center;'>2.10</td></tr><tr><td style='text-align: center;'>Game Lengths</td><td style='text-align: center;'>34</td><td style='text-align: center;'>78</td><td style='text-align: center;'>58.38</td><td style='text-align: center;'>59.00</td><td style='text-align: center;'>3.36</td></tr></table>

## ARCHITECTURES

The architectures and hyperparameters used to train human proxy agents are shown in Table 8. Each model includes a fully connected layer, a multi-layer LSTM block, and a decoder fully connected network that maps the LSTM encodings to a probability distribution over actions. In our implementation, we employ loss masking to ensure that the probability of selecting an illegal action is effectively set to zero during sampling from the policy. Optimisation is performed using the Adam optimiser (Kingma & Ba, 2014) across all configurations. When applying a linear learning rate schedule, the learning rate is reduced to its minimum allowable value during the final 10% of the training process.

## HYPERPARAMETER SEARCH

We run a hyperparameter search for the BC policies used for training human proxies. We performed a full grid search across a range of hyperparameters, considering both two-player and three-player game settings. The hyperparameter search space is shown in Table 9.

From these configurations, we selected the two best-performing settings for both two-player and three-player scenarios. These configurations were subsequently employed in the second step of the HDR-IPPO procedure. Respective agents are what we refer to as human proxies.

## ROLE OF REGULARISATION

Our human proxy policies have demonstrated successful coordination in cross-play with both other human proxies and the original BC policies. Here, we show that arbitrary policies fail to coordinate well with them. We introduce a new set of agents trained using the same HDR-IPPO procedure as our human proxies, but with a crucial difference, the human data regularisation weight is set to zero,  $ \lambda = 0 $ .

These new policies start from the same BC policy weights as our human proxies and utilise identical hyperparameters, except for the KL regularisation term, which is eliminated. We focus on two specific agents, one for the two-player setting (based on  $ \pi_{\theta^{2}}^{BC} $ ) and one for the three-player setting (based on  $ \pi_{\theta^{4}}^{BC} $ ). We use the weights obtained at the final training timestep as checkpoints for these model weights.

Firstly, we observed a rapid deterioration in SP performance for the non-regularised policies. The agents quickly lost the ability to perform well, even in SP. This degradation is evident in Figure 4 and Figure 5. This sharp decline suggests a divergence from the strategies initially learned during the BC phase. Without the regularisation term guiding the policy towards human-like behaviour, the agents appear to explore alternative strategies that may lead to suboptimal performance in the context of the game.

<div style="text-align: center;">Table 8: Human proxy agent training configurations and architectures. We showcase both BC and IPPO hyperparameters in a single table.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>$ \pi_{\theta^{1}}^{HP} $</td><td style='text-align: center;'>$ \pi_{\theta^{2}}^{HP} $</td><td style='text-align: center;'>$ \pi_{\theta^{3}}^{HP} $</td><td style='text-align: center;'>$ \pi_{\theta^{4}}^{HP} $</td></tr><tr><td colspan="5">Network Architecture</td></tr><tr><td style='text-align: center;'>Num Players</td><td style='text-align: center;'>2</td><td style='text-align: center;'>2</td><td style='text-align: center;'>3</td><td style='text-align: center;'>3</td></tr><tr><td style='text-align: center;'>Activation</td><td style='text-align: center;'>GELU</td><td style='text-align: center;'>GELU</td><td style='text-align: center;'>GELU</td><td style='text-align: center;'>GELU</td></tr><tr><td style='text-align: center;'>LSTM Layers</td><td style='text-align: center;'>512, 512, 512</td><td style='text-align: center;'>512, 512</td><td style='text-align: center;'>512, 512, 512</td><td style='text-align: center;'>512, 512</td></tr><tr><td style='text-align: center;'>Input Embedding</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td></tr><tr><td style='text-align: center;'>Decoder MLP</td><td style='text-align: center;'>512</td><td style='text-align: center;'>256</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td></tr><tr><td colspan="5">BC</td></tr><tr><td colspan="5">Optimization</td></tr><tr><td style='text-align: center;'>Batch Size</td><td style='text-align: center;'>256</td><td style='text-align: center;'>256</td><td style='text-align: center;'>128</td><td style='text-align: center;'>256</td></tr><tr><td style='text-align: center;'>Dropout</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.3</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>LR Schedule</td><td style='text-align: center;'>Linear</td><td style='text-align: center;'>Linear</td><td style='text-align: center;'>Linear</td><td style='text-align: center;'>Linear</td></tr><tr><td style='text-align: center;'>Initial LR</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>0.005</td></tr><tr><td style='text-align: center;'>Final LR</td><td style='text-align: center;'>5.0e-05</td><td style='text-align: center;'>5.0e-05</td><td style='text-align: center;'>1.0e-05</td><td style='text-align: center;'>1.0e-05</td></tr><tr><td style='text-align: center;'>Epochs</td><td style='text-align: center;'>50</td><td style='text-align: center;'>50</td><td style='text-align: center;'>70</td><td style='text-align: center;'>70</td></tr><tr><td colspan="5">Training</td></tr><tr><td style='text-align: center;'>Permute Colours</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>Self-Play Eval Games</td><td style='text-align: center;'>5000</td><td style='text-align: center;'>5000</td><td style='text-align: center;'>5000</td><td style='text-align: center;'>5000</td></tr><tr><td colspan="5">IPPO</td></tr><tr><td colspan="5">Optimization</td></tr><tr><td style='text-align: center;'>Learning Rate</td><td style='text-align: center;'>0.0003</td><td style='text-align: center;'>0.0003</td><td style='text-align: center;'>0.0005</td><td style='text-align: center;'>0.0003</td></tr><tr><td style='text-align: center;'>Gamma Discount</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.99</td></tr><tr><td style='text-align: center;'>GAE Lambda</td><td style='text-align: center;'>0.95</td><td style='text-align: center;'>0.95</td><td style='text-align: center;'>0.95</td><td style='text-align: center;'>0.95</td></tr><tr><td style='text-align: center;'>Clip Epsilon</td><td style='text-align: center;'>0.2</td><td style='text-align: center;'>0.2</td><td style='text-align: center;'>0.2</td><td style='text-align: center;'>0.2</td></tr><tr><td style='text-align: center;'>Entropy Coefficient</td><td style='text-align: center;'>1.0e-05</td><td style='text-align: center;'>0.0001</td><td style='text-align: center;'>0.0001</td><td style='text-align: center;'>0.0001</td></tr><tr><td style='text-align: center;'>Value Function Coeff</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Max Gradient Norm</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Update Epochs</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td></tr><tr><td style='text-align: center;'>Num Minibatches</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td></tr><tr><td colspan="5">Critic Network</td></tr><tr><td style='text-align: center;'>Critic MLP</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr><tr><td colspan="5">Training</td></tr><tr><td style='text-align: center;'>BC Policy KL Weight</td><td style='text-align: center;'>0.3</td><td style='text-align: center;'>0.2</td><td style='text-align: center;'>0.1</td><td style='text-align: center;'>0.15</td></tr><tr><td style='text-align: center;'>Total Timesteps</td><td style='text-align: center;'>5e9</td><td style='text-align: center;'>5e9</td><td style='text-align: center;'>5e9</td><td style='text-align: center;'>5e9</td></tr><tr><td colspan="5">Environments</td></tr><tr><td style='text-align: center;'>Num Env Steps</td><td style='text-align: center;'>90</td><td style='text-align: center;'>90</td><td style='text-align: center;'>90</td><td style='text-align: center;'>90</td></tr><tr><td style='text-align: center;'>Num Train Envs</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td></tr><tr><td style='text-align: center;'>Num Eval Envs</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr></table>

<div style="text-align: center;">Table 9: Hyperparameter search space for BC when developing human proxies.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>Values</td></tr><tr><td style='text-align: center;'>Dropout</td><td style='text-align: center;'>0.3, 0.5</td></tr><tr><td style='text-align: center;'>Batch Size</td><td style='text-align: center;'>64, 128, 256</td></tr><tr><td style='text-align: center;'>LSTM Layers</td><td style='text-align: center;'>(512, 512), (512, 512, 512), (1024, 1024), (1024, 1024, 1024)</td></tr><tr><td style='text-align: center;'>Input Embedding</td><td style='text-align: center;'>512, 1024, (512, 512)</td></tr><tr><td style='text-align: center;'>Decoder MLP</td><td style='text-align: center;'>512, 1024</td></tr></table>

After the initial performance drop, the scores of the non-regularised policies gradually improve during training. However, with the current set of hyperparameters, this progress is slow and we do not observe convergence. The agents seem unable to efficiently re-learn a different, yet still effective, policy. It is important to note that this behaviour is not observed with different hyperparameter settings, but for the purpose of direct comparison in this analysis, we maintained the same configuration as the human proxy agents.

When paired with human proxies and BC policies, the non-regularised agents exhibit significantly lower coordination and overall performance compared to the pairings between human proxies and

BC policies exclusively. This discrepancy highlights the crucial role of the human data regularisation term in ensuring that the learned strategies remain aligned with human-like play, facilitating successful coordination.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_276_266_580_561.jpg" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_643_266_948_562.jpg" alt="Image" width="24%" /></div>


<div style="text-align: center;">Figure 4: Cross-play performance matrix for two-player agents, comparing a human proxy agent  $ (\pi_{\theta^{2}}^{HP}) $ , its corresponding BC policy  $ (\pi_{\theta^{2}}^{BC}) $ , and a non-regularised HDR-IPPO agent initialised from the same BC policy,  $ \pi_{\theta^{2}}^{BC} $ , but where  $ \lambda = 0 $ . Each matrix element represents the average score achieved by a team, averaged over both player orderings. Averages are calculated based on 15,000 games per team permutation.</div>


Consider the three-player cross-play matrix depicted in Figure 5. The BC policy, while achieving a low SP score of 7.19, significantly improves to 18.92 when paired with two human proxies. This substantial boost underscores the BC policy's inherent understanding of human-like strategies, even if it struggles to execute them independently. In contrast, the non-regularised policy, with a SP score of 3.99, only sees a marginal improvement to 6.93 in cross-play with human proxies. Furthermore, pairing the BC policy (the stronger player in SP) with two non-regularised policies actually decreases the performance compared to the SP performance of non-regularised policies. This degradation suggests a fundamental incompatibility between the strategies learned by the non-regularised policy and the human-like conventions given by the BC policy.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_276_980_580_1273.jpg" alt="Image" width="24%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_644_980_948_1273.jpg" alt="Image" width="24%" /></div>


<div style="text-align: center;">Figure 5: Cross-play performance matrix for three-player setting, with scores for a human proxy agent  $ (\pi_{\theta^{4}}^{HP}) $ , its corresponding BC policy  $ (\pi_{\theta^{4}}^{BC}) $ , and a non-regularised HDR-IPPO agent initialised from the same BC policy,  $ \pi_{\theta^{4}}^{BC} $ , but with  $ \lambda=0 $ . Each matrix element  $ (i,j) $  represents the average score achieved by a team composed of two instances of one agent on the x-axis and one instance of agent on the y-axis, averaged over all possible permutations of player positions. Averages are calculated based on 15,000 games per team permutation.</div>


### A.5 BEHAVIOUR ANALYSIS

The dataset we use in this work comes from the human policies that adhere to H-Group Conventions. This distinguishes our work from previous human-AI coordination studies, as these strategies can be explicitly described in natural language and are documented.

We also include behavioural metrics proposed by (Canaan et al., 2020), for both the trajectories found in the dataset and the ones created by the human proxies. Results are shown in Table 10.

<div style="text-align: center;">Table 10: We compare behaviour features available in the dataset and once present in trajectories generated by our human proxies. Precisely, we compute Information per Play (IPP) and Communicativeness. We show metrics are almost identical for both cases.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Trajectories</td><td style='text-align: center;'>Proxy</td><td style='text-align: center;'>IPP</td><td style='text-align: center;'>Communicativeness</td></tr><tr><td rowspan="3">Two-Player</td><td style='text-align: center;'>2P Dataset</td><td style='text-align: center;'>0.44</td><td style='text-align: center;'>0.47</td></tr><tr><td style='text-align: center;'>$ \pi_{\theta^{1}}^{HP} $</td><td style='text-align: center;'>0.43</td><td style='text-align: center;'>0.45</td></tr><tr><td style='text-align: center;'>$ \pi_{\theta^{2}}^{HP} $</td><td style='text-align: center;'>0.44</td><td style='text-align: center;'>0.48</td></tr><tr><td rowspan="3">Three-Player</td><td style='text-align: center;'>3P Dataset</td><td style='text-align: center;'>0.42</td><td style='text-align: center;'>0.49</td></tr><tr><td style='text-align: center;'>$ \pi_{\theta^{3}}^{HP} $</td><td style='text-align: center;'>0.44</td><td style='text-align: center;'>0.47</td></tr><tr><td style='text-align: center;'>$ \pi_{\theta^{4}}^{HP} $</td><td style='text-align: center;'>0.44</td><td style='text-align: center;'>0.46</td></tr></table>

Furthermore, we provide a qualitative assessment by analysing games played by our human proxies. Our findings show that our agents exhibit highly human-like behaviour and in most instances adhere to H-convention strategies. Sample renders can be found in the anonymous repository we provide and here we analyse the first game provided. We analyse each of the moves below, marking each move as a success or failure, depending on whether the convention is followed correctly. We also find that the mistakes made when not following conventions are very human-like mistakes. For example, on Turn 2, the agent violates the good touch principle, but this move also looks a lot like a 2 save. Hence, it violates a convention because it tried to follow a different one, which was not applicable in this case. Most of the failure cases we found come from confusing conventions that should be played in that particular instance, which is a human-like mistake.

1. Turn 0: Success (Hint blue, play clue on the blue 1)

2. Turn 1: Success (Play clue on both the 1's)

3. Turn 2: Failure (Looks like a 2 save, but doubles the yellow 2's, violating good touch principle)

4. Turn 3: Success (Actor 1 plays card slot 2, knowing it's a B1, which follows the "play clue" convention)

5. Turn 4: Success (Actor 0 gives a 5 save when Red 5 is on the chop)

6. Turn 5: Success (Actor 1 plays a known playable card)

7. Turn 6: Failure (Actor 0 should have 2-saved the Red 2 on chop)

8. Turn 7: Success (Actor 1 discards Red 2, the chop card)

9. Turn 8: Success (Actor 0 plays slot 0, a known playable 1)

10. Turn 9: Success (Actor 1 discards chop on slot 3)

11. Turn 10: Success (Fix clued the duplicate cards)

12. Turn 11: Success (Plays known playable card)

13. Turn 12: Success (Gives play clue to the Red 1)

14. Turn 13: Success (Plays the Red 1)

15. Turn 14: Success (Discards chop)

16. Turn 15: Success (Gives play clue to the Yellow 3)

17. Turn 16: Success (Plays Yellow 3)

18. Turn 17: Success (Discards known trash)

19. Turn 18: Success (Discards chop)

20. Turn 19: Failure (Doesn't understand chop-focus, giving play clue to wrong card, and violates good touch principle by clueing Red 1 and Red 3 twice)

21. Turn 20: Failure (The focus of the last clue was the chop, so it should have played the Red 3, but played Red 2 instead)

22. Turn 21: Success (Fix clue on the duplicated Red 3's)

23. Turn 22: Success (Plays known playable Red 3)

24. Turn 23: Success (Discards chop, position 1)

25. Turn 24: Success (Gives play clue to White 2)

26. Turn 25: Success (Plays White 2)

27. Turn 26: Success (Play clue on the Red 4)

28. Turn 27: Success (Plays the Red 4)

29. Turn 28: Success (Play clue on Blue 4, and filling in White 3)

30. Turn 29: Success (Plays White 3)

31. Turn 30: Success (Discards known trash)

32. Turn 31: Success (Play clue on White 4)

33. Turn 32: Success (Plays White 4)

34. Turn 33: Success (Plays known playable Red 5)

35. Turn 34: Success (Discards known trash Red 1)

36. Turn 35: Failure (Should have played its Blue 3 because of the play clue, instead gave a 5 hint off chop, which is illegal after the late game)

37. Turn 36: Success (Discards chop)

38. Turn 37: Failure (Should have played its Blue 3, instead gave a 2 hint which is illegal)

39. Turn 38: Success (Discards chop)

40. Turn 39: Failure (Should have played its Blue 3, instead discarded chop)

41. Turn 40: Success (Play clue on Blue 4, filling in Blue 3)

42. Turn 41: Success (Plays Blue 3)

43. Turn 42: Success (Discards chop)

44. Turn 43: Success (Plays Blue 4)

45. Turn 44: Success (Discards chop)

46. Turn 45: Success (Hint Blue, filling in Blue 5)

47. Turn 46: Success (Plays Blue 5)

48. Turn 47: Success (Play clue on Green 1)

49. Turn 48: Success (Plays Green 1)

50. Turn 49: Success (Discards chop)

51. Turn 50: Success (Play clue on Yellow 4)

52. Turn 51: Success (Plays Yellow 4)

53. Turn 52: Success (Plays Green 2)

54. Turn 53: Success (Discards chop)

55. Turn 54: Success (Reveals Green 4 identity)

56. Turn 55: Success (Discards chop)

1026 57. Turn 56: Success (Plays Yellow 5)

1027 58. Turn 57: Success (Discards chop)

1028 59. Turn 58: Success (5 save on Green 5)

1030 60. Turn 59: Success (Stalling, hinting 1s)

1031 61. Turn 60: Failure (Hinting Green is seen as a play clue on Green 1, which is illegal)

1033 62. Turn 61: Success (Plays Green 1, which it thought was Green 3 because of convention)

1034 63. Turn 62: Success (Play clue on Green 3)

1036 64. Turn 63: Success (Plays Green 3)

1037 65. Turn 64: Success (Hints White 5)

1038 66. Turn 65: Success (Plays Green 4)

In summary, in this game, the human proxy followed H-group conventions for 88% of the moves and used various strategies while playing the game.

## • Successful H-Group conventions played:

- Giving play clue (14x)

- Responding to play clue (23x)

- 2 Save (1x)

- 5 Save (2x)

- Discard chop (13x)

- Fix clue (3x)

- Discard known trash (3x)

- Stall clue (1x)

• H-Group Conventions violated:

– Violating Good touch Principle (1x)

– Failure to 2 save (1x)

– Incorrectly gives chop focus clue (1x)

– Incorrectly responding to chop focus clue (1x)

– Failure to play play clue (3x)

– Incorrectly give play clue (1x)

### A.6 AH2AC2: CHALLENGE IMPLEMENTATION DETAILS

We have established a dedicated website for the AH2AC2 challenge at https://ah2ac2.com/. Participants can request to join the challenge through this website, where they will also find a leaderboard displaying existing results.

Upon submitting a participation request, candidates will be notified when the challenge becomes public and will receive an API key for testing their implementations and for official evaluation. The results of candidate agents will automatically appear on the leaderboard once all games are completed. Participation in the action prediction challenge is currently optional but encouraged.

For the evaluation, we assess the performance of each candidate agent over a total of 2,000 games played with our human proxy agents: 1,000 games in the two-player setting and 1,000 games in the three-player setting. We consider all seating configurations and all combinations of agents and seating positions, including scenarios where candidates control multiple agents (except for self-play situations). The evaluation is conducted uniformly across different seating configurations to ensure a fair and comprehensive assessment. This setup follows the procedure introduced in the seminal work on ad hoc teamwork evaluation by (Stone et al., 2010). Participants can obtain evaluation results and information for their agents at any time through our dedicated API.

### A.7 HDR-IPPO: ABLATION STUDY

We conduct an ablation study to investigate the impact of the human data regularisation term on the performance and behaviour of agents trained with the HDR-IPPO. By systematically varying the strength of the regularisation, we aim to gain insights into its role in guiding policy learning towards strategies learned during BC (in this case, human-like strategies).

## METHODOLOGY

We focus on the two-player setting, where BC has demonstrated strong performance as a baseline. Due to the computational and time constraints associated with training HDR-IPPO agents, it was not feasible to extend this study to the three-player setting. To systematically analyse the effect of the KL regularisation term introduced in the HDR-IPPO algorithm, we adopt the following methodology:

1. We train a new BC policy that serves as both a starting point for subsequent HDR-IPPO training and a baseline for comparison in this ablation study.

2. From the baseline BC policy, we train multiple HDR-IPPO agents. These agents share identical architectures, hyperparameters, and training procedures, with the sole exception of the human data regularisation weight,  $ \lambda $ . We vary the weight of regularisation term across a range of values, from no regularisation ( $ \lambda = 0.00 $ ) to very high regularisation ( $ \lambda = 0.70 $ ) to comprehensively investigate the effects of this term during both training and evaluation. The final policy weights for each agent are stored for subsequent analysis.

3. We evaluate, analyse and compare the trained HDR-IPPO agents and the baseline BC policy. Precisely, we:

(a) Assess the SP performance of each agent to understand how the strength of KL regularisation influences its ability to play Hanabi effectively on its own.

(b) Evaluate the cross-play performance of each HDR-IPPO agent when paired with the baseline BC policy. Higher scores in this setting indicate that the HDR-IPPO agent has learned strategies compatible with the human-like conventions exhibited by the BC agent.

(c) Conduct cross-play evaluations among different HDR-IPPO agents to identify potential coordination patterns.

(d) Evaluate each agent on the held-out validation and test sets of human data. Good performance on this data indicates that the agent has effectively retained the conventions observed in the human demonstrations.

(e) Plot the KL divergence between the action distributions of each HDR-IPPO agent and the baseline BC policy throughout the training process. This visualisation allows us to identify potential convergence or divergence patterns for the KL divergence term.

## EXPERIMENTAL SETUP

The hyperparameters used for training the agents in this ablation study are listed in Table 11.

<div style="text-align: center;">Table 11: Hyperparameters used for training agents in the ablation study. We showcase both BC and IPPO hyperparameters in a single table.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>Value</td></tr><tr><td colspan="2">Network</td></tr><tr><td style='text-align: center;'>Num Players</td><td style='text-align: center;'>2</td></tr><tr><td style='text-align: center;'>Activation</td><td style='text-align: center;'>GELU</td></tr><tr><td style='text-align: center;'>LSTM Layers</td><td style='text-align: center;'>512</td></tr><tr><td style='text-align: center;'>Input Embedding</td><td style='text-align: center;'>1024</td></tr><tr><td style='text-align: center;'>Decoder MLP</td><td style='text-align: center;'>512</td></tr><tr><td style='text-align: center;'>Dropout</td><td style='text-align: center;'>0.5</td></tr><tr><td colspan="2">BC</td></tr><tr><td colspan="2">Optimisation</td></tr><tr><td style='text-align: center;'>Optimiser</td><td style='text-align: center;'>Adam (Kingma &amp; Ba, 2014)</td></tr><tr><td style='text-align: center;'>Batch Size</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>LR Schedule</td><td style='text-align: center;'>Linear</td></tr><tr><td style='text-align: center;'>Initial LR</td><td style='text-align: center;'>0.005</td></tr><tr><td style='text-align: center;'>Final LR</td><td style='text-align: center;'>1.0e-05</td></tr><tr><td style='text-align: center;'>Epochs</td><td style='text-align: center;'>70</td></tr><tr><td colspan="2">Training</td></tr><tr><td style='text-align: center;'>Permute Colours</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>Self-Play Eval Games</td><td style='text-align: center;'>5000</td></tr><tr><td colspan="2">IPPO</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td style='text-align: center;'>Learning Rate</td><td style='text-align: center;'>0.0005</td></tr><tr><td style='text-align: center;'>Gamma Discount</td><td style='text-align: center;'>0.99</td></tr><tr><td style='text-align: center;'>GAE Lambda</td><td style='text-align: center;'>0.95</td></tr><tr><td style='text-align: center;'>Clip Epsilon</td><td style='text-align: center;'>0.2</td></tr><tr><td style='text-align: center;'>Entropy Coefficient</td><td style='text-align: center;'>0.01</td></tr><tr><td style='text-align: center;'>Value Function Coeff</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Max Gradient Norm</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Update Epochs</td><td style='text-align: center;'>4</td></tr><tr><td style='text-align: center;'>Num Minibatches</td><td style='text-align: center;'>4</td></tr><tr><td colspan="2">Critic Network</td></tr><tr><td style='text-align: center;'>Critic MLP</td><td style='text-align: center;'>512</td></tr><tr><td colspan="2">Training</td></tr><tr><td style='text-align: center;'>Total Timesteps</td><td style='text-align: center;'>2e10</td></tr><tr><td colspan="2">Environments</td></tr><tr><td style='text-align: center;'>Num Env Steps</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>Num Train Envs</td><td style='text-align: center;'>1024</td></tr><tr><td style='text-align: center;'>Num Eval Envs</td><td style='text-align: center;'>512</td></tr><tr><td style='text-align: center;'>Num Test Actors</td><td style='text-align: center;'>1024</td></tr><tr><td style='text-align: center;'>Num Test Envs</td><td style='text-align: center;'>512</td></tr></table>

In addition to the listed parameters, we consider setups with the following human data KL regularisation weights: 0.00, 0.01, 0.08, 0.13, 0.20, 0.30, 0.50, and 0.70. This range allows us to explore the effects of regularisation from a complete absence to a very strong influence on the learning process.

## Results and Discussion

We start by presenting the cross-play matrix in Figure 6.

From the cross-play matrix, we observe consistently high SP scores across all agents, indicating that all variations of HDR-IPPO lead to significant improvements over the baseline BC policy's mean SP score of 19.53. The HDR-IPPO agents with the lowest SP scores are those at the extremes of the regularisation spectrum: the non-regularised policy ( $ \lambda = 0.00 $ ) and the one with the highest regularisation ( $ \lambda = 0.70 $ ). The agent showing the best SP performance, with a mean score of 23.42, is the one trained with  $ \lambda = 0.13 $ .

This result provides initial empirical evidence supporting the effectiveness of the regularisation employed in HDR-IPPO. We hypothesise that excessively high regularisation weights might prevent

<div style="text-align: center;"><img src="imgs/img_in_chart_box_216_160_587_509.jpg" alt="Image" width="30%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_643_161_1006_508.jpg" alt="Image" width="29%" /></div>


<div style="text-align: center;">Figure 6: Cross-play performance matrix. Each element  $ (i,j) $  represents the average score achieved by a team comprising of Agent j and Agent i, averaged over possible permutations of player positions. The average score for each team is calculated based on 25,000 games per permutation.</div>


the policy to generalise, while the complete absence of regularisation could lead to the adoption of strategies that deviate significantly from human-like conventions.

Next, we examine the distribution of perfect and zero-score games. As shown in the Table 12 below, with an appropriate level of regularisation, we acquire a policy that achieves a high number of perfect games while completely eliminating games with a score of zero. In contrast, the non-regularised policy  $ (\lambda = 0.00) $  yields only 143 perfect games, and its minimum score is comparable to that of the policy with  $ \lambda = 0.13 $ , which boasts 1599 perfect games. This discrepancy further supports our hypothesis that these agents employ fundamentally different strategies.

<div style="text-align: center;">Table 12: Perfect and zero-score games count out of 5,000 SP games. We also show the minimum score in the set of 5,000 games.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Perfect</td><td style='text-align: center;'>Zero</td><td style='text-align: center;'>Min</td></tr><tr><td style='text-align: center;'>$ \lambda=0.00 $</td><td style='text-align: center;'>143</td><td style='text-align: center;'>0</td><td style='text-align: center;'>13</td></tr><tr><td style='text-align: center;'>$ \lambda=0.01 $</td><td style='text-align: center;'>247</td><td style='text-align: center;'>0</td><td style='text-align: center;'>14</td></tr><tr><td style='text-align: center;'>$ \lambda=0.08 $</td><td style='text-align: center;'>1351</td><td style='text-align: center;'>0</td><td style='text-align: center;'>14</td></tr><tr><td style='text-align: center;'>$ \lambda=0.13 $</td><td style='text-align: center;'>1599</td><td style='text-align: center;'>0</td><td style='text-align: center;'>13</td></tr><tr><td style='text-align: center;'>$ \lambda=0.20 $</td><td style='text-align: center;'>1821</td><td style='text-align: center;'>0</td><td style='text-align: center;'>15</td></tr><tr><td style='text-align: center;'>$ \lambda=0.30 $</td><td style='text-align: center;'>1564</td><td style='text-align: center;'>3</td><td style='text-align: center;'>0</td></tr><tr><td style='text-align: center;'>$ \lambda=0.50 $</td><td style='text-align: center;'>1280</td><td style='text-align: center;'>21</td><td style='text-align: center;'>0</td></tr><tr><td style='text-align: center;'>$ \lambda=0.70 $</td><td style='text-align: center;'>1166</td><td style='text-align: center;'>113</td><td style='text-align: center;'>0</td></tr></table>

Let us now analyse the cross-play results from the perspective of the BC policy. Examining the BC policy column in Figure 6, we observe that cross-play scores generally increase compared to the initial BC SP score when paired with most HDR-IPPO agents. However, a notable exception occurs for pairings with agents trained using  $ \lambda = 0.00 $  and  $ \lambda = 0.01 $ . Despite these two policies having significantly higher SP scores than the BC policy, their coordination with the BC agent is poor, resulting in a substantial drop in cross-play performance. This widening gap between SP and cross-play scores is a recognised indicator of poor coordination. Additionally, the policy with the highest regularisation weight ( $ \lambda = 0.70 $ ) does not exhibit this coordination breakdown, even though it performs worse in SP compared to the non-regularised agent ( $ \lambda = 0.00 $ ). This observation provides empirical evidence that training with insufficient regularisation can lead to a divergence from the strategies encountered in the dataset used for BC, even when the final policy develops strong strategies and performs well in isolation.

Furthermore, let's examine the cross-play results from the perspective of the non-regularised policy ( $ \lambda = 0.00 $ ). A clear trend emerges where cross-play performance deteriorates as the regularisation weight of the partner policy increases. This suggests that the non-regularised policy, having diverged from human-like conventions, struggles to coordinate effectively with agents that adhere more closely to those conventions. In contrast, for policies trained with higher regularisation weights, we observe the opposite trend. The gap between SP and cross-play scores diminishes, and in some cases, cross-play even yields higher scores than the weaker policy's SP performance. This indicates that these policies, guided by the KL regularisation term, converge towards similar strategies, enabling them to coordinate exceptionally well, even surpassing individual SP scores in certain instances. The same holds true when pairing these agents with the baseline BC agent, further underscoring their compatibility with human-like conventions. This analysis provides strong empirical evidence that policies trained with higher regularisation weights tend to converge to a shared set of strategies, and that those strategies align closely with ones learned during the BC procedure.

We now shift our focus to evaluating the agents on the held-out validation and test sets, as presented in Table 13. Increasing the regularisation weight generally leads to improved performance on the held-out data, suggesting better adherence to the conventions present in the training set. Notably, the increase from  $ \lambda = 0.00 $  to  $ \lambda = 0.08 $  results in a substantial improvement of over 20% accuracy on both the validation and test sets. Importantly, subsequent increases yield only marginal gains.

Importantly, agents that perform well in cross-play tend to have a lot stronger performance on the held-out datasets. Hence, we again show that they converge to similar conventions, but we also show that these conventions are very close to ones learned during training.

<div style="text-align: center;">Table 13: Performance on hold out sets for all agents. We show the best result in bold.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'></td><td style='text-align: center;'>Val Loss</td><td style='text-align: center;'>Val Acc</td><td style='text-align: center;'>Test Loss</td><td style='text-align: center;'>Test Acc</td></tr><tr><td style='text-align: center;'>BC</td><td style='text-align: center;'>0.466</td><td style='text-align: center;'>0.676</td><td style='text-align: center;'>0.468</td><td style='text-align: center;'>0.674</td></tr><tr><td style='text-align: center;'>$ \lambda=0.00 $</td><td style='text-align: center;'>7.368</td><td style='text-align: center;'>0.330</td><td style='text-align: center;'>7.385</td><td style='text-align: center;'>0.327</td></tr><tr><td style='text-align: center;'>$ \lambda=0.01 $</td><td style='text-align: center;'>1.354</td><td style='text-align: center;'>0.435</td><td style='text-align: center;'>1.369</td><td style='text-align: center;'>0.433</td></tr><tr><td style='text-align: center;'>$ \lambda=0.08 $</td><td style='text-align: center;'>0.716</td><td style='text-align: center;'>0.574</td><td style='text-align: center;'>0.725</td><td style='text-align: center;'>0.571</td></tr><tr><td style='text-align: center;'>$ \lambda=0.13 $</td><td style='text-align: center;'>0.618</td><td style='text-align: center;'>0.607</td><td style='text-align: center;'>0.628</td><td style='text-align: center;'>0.605</td></tr><tr><td style='text-align: center;'>$ \lambda=0.20 $</td><td style='text-align: center;'>0.540</td><td style='text-align: center;'>0.636</td><td style='text-align: center;'>0.548</td><td style='text-align: center;'>0.634</td></tr><tr><td style='text-align: center;'>$ \lambda=0.30 $</td><td style='text-align: center;'>0.488</td><td style='text-align: center;'>0.659</td><td style='text-align: center;'>0.495</td><td style='text-align: center;'>0.656</td></tr><tr><td style='text-align: center;'>$ \lambda=0.50 $</td><td style='text-align: center;'>0.475</td><td style='text-align: center;'>0.665</td><td style='text-align: center;'>0.481</td><td style='text-align: center;'>0.663</td></tr><tr><td style='text-align: center;'>$ \lambda=0.70 $</td><td style='text-align: center;'>0.464</td><td style='text-align: center;'>0.671</td><td style='text-align: center;'>0.469</td><td style='text-align: center;'>0.668</td></tr></table>

Finally, we turn our attention to the evolution of the KL divergence term throughout the training process. Figure 7 illustrates the KL divergence for all trained HDR-IPPO policies. It is immediately apparent that the KL terms for  $ \lambda = 0.00 $  and  $ \lambda = 0.01 $  are significantly higher than the others, rendering the remaining curves barely visible in the plot. This observation aligns with our previous findings, further reinforcing the notion that insufficient regularisation can lead to substantial divergence from the human-like strategies learned through BC.

To gain a clearer understanding of the KL divergence dynamics for policies with higher regularisation weights, we present Figure 8, which excludes the policies with  $ \lambda = 0.00 $  and  $ \lambda = 0.01 $ . While policies with lower regularisation weights show increasing KL divergence during training, those with higher weights demonstrate a decreasing trend. This suggests that stronger regularisation effectively prevents the policy from deviating too far from the human-like strategies captured by the BC policy.

We do observe an initial increase in KL divergence for all policies during the first few update steps. This is likely attributable to the dominance of the IPPO loss over the KL divergence term in the early stages of training, particularly as the value function is being learned from scratch.

In conclusion, our ablation study provides compelling evidence that the KL regularisation term in HDR-IPPO effectively prevents divergence from the human-like strategies learned through BC. However, it is crucial to set the regularisation weight,  $ \lambda $ , to a sufficiently large value to ensure sustained adherence to these conventions throughout the training process, particularly in the context of extended training duration.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_330_184_875_600.jpg" alt="Image" width="44%" /></div>


<div style="text-align: center;">Figure 7: KL divergence throughout training for all trained HDR-IPPO policies.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_331_714_874_1125.jpg" alt="Image" width="44%" /></div>


<div style="text-align: center;">Figure 8: KL divergence throughout training where  $ \lambda \geq 0.08 $ </div>


### A.8 BASELINES: HYPERPARAMETERS

For OBL, we use open-sourced weights and hyperparameters (Hu et al., 2021).

<div style="text-align: center;">Table 14: Hyperparameters used for training BC, HDR-IPPO baselines on a 1,000-game data limit challenge. BC policies trained as baselines are used for starting points in HDR-IPPO and later for BR-BC.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>Two-Player Setting</td><td style='text-align: center;'>Three-Player Setting</td></tr><tr><td colspan="3">Network Architecture</td></tr><tr><td style='text-align: center;'>Num Players</td><td style='text-align: center;'>2</td><td style='text-align: center;'>3</td></tr><tr><td style='text-align: center;'>Activation</td><td style='text-align: center;'>GELU</td><td style='text-align: center;'>GELU</td></tr><tr><td style='text-align: center;'>LSTM Layers</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr><tr><td style='text-align: center;'>Input Embedding</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>512</td></tr><tr><td style='text-align: center;'>Decoder MLP</td><td style='text-align: center;'>256</td><td style='text-align: center;'>256</td></tr><tr><td colspan="3">BC</td></tr><tr><td colspan="3">Optimisation</td></tr><tr><td style='text-align: center;'>Batch Size</td><td style='text-align: center;'>32</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>Dropout</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>LR Schedule</td><td style='text-align: center;'>Linear</td><td style='text-align: center;'>Linear</td></tr><tr><td style='text-align: center;'>Initial LR</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>0.005</td></tr><tr><td style='text-align: center;'>Final LR</td><td style='text-align: center;'>0.0001</td><td style='text-align: center;'>0.0001</td></tr><tr><td style='text-align: center;'>Epochs</td><td style='text-align: center;'>70</td><td style='text-align: center;'>50</td></tr><tr><td colspan="3">Training</td></tr><tr><td style='text-align: center;'>Permute Colours</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>Self-Play Eval Games</td><td style='text-align: center;'>5000</td><td style='text-align: center;'>5000</td></tr><tr><td colspan="3">IPPO during HDR-IPPO</td></tr><tr><td colspan="3">Optimisation</td></tr><tr><td style='text-align: center;'>Learning Rate</td><td style='text-align: center;'>0.0005</td><td style='text-align: center;'>0.0005</td></tr><tr><td style='text-align: center;'>Linear Schedule</td><td style='text-align: center;'>True</td><td style='text-align: center;'>True</td></tr><tr><td style='text-align: center;'>Gamma Discount</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.99</td></tr><tr><td style='text-align: center;'>GAE Lambda</td><td style='text-align: center;'>0.95</td><td style='text-align: center;'>0.95</td></tr><tr><td style='text-align: center;'>Clip Epsilon</td><td style='text-align: center;'>0.2</td><td style='text-align: center;'>0.2</td></tr><tr><td style='text-align: center;'>Entropy Coefficient</td><td style='text-align: center;'>0.001</td><td style='text-align: center;'>0.001</td></tr><tr><td style='text-align: center;'>Value Function Coeff</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Max Gradient Norm</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Update Epochs</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td></tr><tr><td style='text-align: center;'>Num Minibatches</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td></tr><tr><td colspan="3">Critic Network</td></tr><tr><td style='text-align: center;'>Critic MLP</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr><tr><td colspan="3">Training</td></tr><tr><td style='text-align: center;'>BC Policy KL Weight</td><td style='text-align: center;'>0.25</td><td style='text-align: center;'>0.25</td></tr><tr><td style='text-align: center;'>Total Timesteps</td><td style='text-align: center;'>1e10</td><td style='text-align: center;'>1e10</td></tr><tr><td colspan="3">Environments</td></tr><tr><td style='text-align: center;'>Num Env Steps</td><td style='text-align: center;'>128</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>Num Train Envs</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td></tr><tr><td style='text-align: center;'>Num Eval Envs</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr></table>

<div style="text-align: center;">Table 15: Hyperparameters used for training BC, HDR-IPPO baselines on a 5,000-game data limit challenge. BC policies trained as baselines are used for starting points in HDR-IPPO and later for BR-BC.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>Two-Player Setting</td><td style='text-align: center;'>Three-Player Setting</td></tr><tr><td colspan="3">Network Architecture</td></tr><tr><td style='text-align: center;'>Num Players</td><td style='text-align: center;'>2</td><td style='text-align: center;'>3</td></tr><tr><td style='text-align: center;'>Activation</td><td style='text-align: center;'>GELU</td><td style='text-align: center;'>GELU</td></tr><tr><td style='text-align: center;'>LSTM Layers</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr><tr><td style='text-align: center;'>Input Embedding</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>512</td></tr><tr><td style='text-align: center;'>Decoder MLP</td><td style='text-align: center;'>256</td><td style='text-align: center;'>256</td></tr><tr><td colspan="3">BC</td></tr><tr><td colspan="3">Optimisation</td></tr><tr><td style='text-align: center;'>Batch Size</td><td style='text-align: center;'>128</td><td style='text-align: center;'>256</td></tr><tr><td style='text-align: center;'>Dropout</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>LR Schedule</td><td style='text-align: center;'>Linear</td><td style='text-align: center;'>Linear</td></tr><tr><td style='text-align: center;'>Initial LR</td><td style='text-align: center;'>0.005</td><td style='text-align: center;'>0.005</td></tr><tr><td style='text-align: center;'>Final LR</td><td style='text-align: center;'>0.0001</td><td style='text-align: center;'>0.0001</td></tr><tr><td style='text-align: center;'>Epochs</td><td style='text-align: center;'>50</td><td style='text-align: center;'>50</td></tr><tr><td colspan="3">Training</td></tr><tr><td style='text-align: center;'>Permute Colours</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>Self-Play Eval Games</td><td style='text-align: center;'>5000</td><td style='text-align: center;'>5000</td></tr><tr><td colspan="3">IPPO during HDR-IPPO</td></tr><tr><td colspan="3">Optimisation</td></tr><tr><td style='text-align: center;'>Learning Rate</td><td style='text-align: center;'>0.0005</td><td style='text-align: center;'>0.0005</td></tr><tr><td style='text-align: center;'>Linear Schedule</td><td style='text-align: center;'>True</td><td style='text-align: center;'>True</td></tr><tr><td style='text-align: center;'>Gamma Discount</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.99</td></tr><tr><td style='text-align: center;'>GAE Lambda</td><td style='text-align: center;'>0.95</td><td style='text-align: center;'>0.95</td></tr><tr><td style='text-align: center;'>Clip Epsilon</td><td style='text-align: center;'>0.2</td><td style='text-align: center;'>0.2</td></tr><tr><td style='text-align: center;'>Entropy Coefficient</td><td style='text-align: center;'>0.001</td><td style='text-align: center;'>0.001</td></tr><tr><td style='text-align: center;'>Value Function Coeff</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Max Gradient Norm</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Update Epochs</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td></tr><tr><td style='text-align: center;'>Num Minibatches</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td></tr><tr><td colspan="3">Critic Network</td></tr><tr><td style='text-align: center;'>Critic MLP</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr><tr><td colspan="3">Training</td></tr><tr><td style='text-align: center;'>BC Policy KL Weight</td><td style='text-align: center;'>0.25</td><td style='text-align: center;'>0.25</td></tr><tr><td style='text-align: center;'>Total Timesteps</td><td style='text-align: center;'>1e10</td><td style='text-align: center;'>1e10</td></tr><tr><td colspan="3">Environments</td></tr><tr><td style='text-align: center;'>Num Env Steps</td><td style='text-align: center;'>128</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>Num Train Envs</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td></tr><tr><td style='text-align: center;'>Num Eval Envs</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr></table>

<div style="text-align: center;">Table 16: Hyperparameters used for training all IPPO and BR-BC baseline agents. Here, we use feed-forward architecture. BC agents used for BR-BC are the same as shown in Table 14 and 15, depending on the challenge variety.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Hyperparameter</td><td style='text-align: center;'>Two-Player Setting</td><td style='text-align: center;'>Three-Player Setting</td></tr><tr><td colspan="3">Network Architecture</td></tr><tr><td style='text-align: center;'>Num Players</td><td style='text-align: center;'>2</td><td style='text-align: center;'>3</td></tr><tr><td style='text-align: center;'>Activation</td><td style='text-align: center;'>RELU</td><td style='text-align: center;'>RELU</td></tr><tr><td style='text-align: center;'>MLP</td><td style='text-align: center;'>(512, 512)</td><td style='text-align: center;'>(512, 512)</td></tr><tr><td colspan="3">IPPO</td></tr><tr><td colspan="3">Optimisation</td></tr><tr><td style='text-align: center;'>Learning Rate</td><td style='text-align: center;'>0.0005</td><td style='text-align: center;'>0.0005</td></tr><tr><td style='text-align: center;'>Gamma Discount</td><td style='text-align: center;'>0.99</td><td style='text-align: center;'>0.99</td></tr><tr><td style='text-align: center;'>GAE Lambda</td><td style='text-align: center;'>0.95</td><td style='text-align: center;'>0.95</td></tr><tr><td style='text-align: center;'>Clip Epsilon</td><td style='text-align: center;'>0.2</td><td style='text-align: center;'>0.2</td></tr><tr><td style='text-align: center;'>Entropy Coefficient</td><td style='text-align: center;'>0.01</td><td style='text-align: center;'>0.01</td></tr><tr><td style='text-align: center;'>Value Function Coeff</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Max Gradient Norm</td><td style='text-align: center;'>0.5</td><td style='text-align: center;'>0.5</td></tr><tr><td style='text-align: center;'>Update Epochs</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td></tr><tr><td style='text-align: center;'>Num Minibatches</td><td style='text-align: center;'>4</td><td style='text-align: center;'>4</td></tr><tr><td colspan="3">Critic Network</td></tr><tr><td style='text-align: center;'>Critic MLP</td><td style='text-align: center;'>512</td><td style='text-align: center;'>512</td></tr><tr><td colspan="3">Training</td></tr><tr><td style='text-align: center;'>Total Timesteps</td><td style='text-align: center;'>1e10</td><td style='text-align: center;'>1e10</td></tr><tr><td colspan="3">Environments</td></tr><tr><td style='text-align: center;'>Num Env Steps</td><td style='text-align: center;'>128</td><td style='text-align: center;'>128</td></tr><tr><td style='text-align: center;'>Num Train Envs</td><td style='text-align: center;'>1024</td><td style='text-align: center;'>1024</td></tr><tr><td colspan="3">BR-BC</td></tr><tr><td style='text-align: center;'>BC Anneal Start</td><td style='text-align: center;'>1e9</td><td style='text-align: center;'>1e9</td></tr><tr><td style='text-align: center;'>BC Anneal End</td><td style='text-align: center;'>6e9</td><td style='text-align: center;'>6e9</td></tr></table>