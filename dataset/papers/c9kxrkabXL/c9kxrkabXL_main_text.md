# MULTI-AGENT DECISION S4: LEVERAGING STATE SPACE MODELS FOR OFFLINE MULTI-AGENT REINFORCEMENT LEARNING

Anonymous authors

Paper under double-blind review

## ABSTRACT

Goal-conditioned sequence-based supervised learning with transformers has shown promise in offline reinforcement learning (RL) for single-agent settings. However, extending these methods to offline multi-agent RL (MARL) remains challenging. Existing transformer-based MARL approaches either train agents independently, neglecting multi-agent system dynamics, or rely on centralized transformer models, which face scalability issues. Moreover, transformers inherently struggle with long-term dependencies and computational efficiency. Building on the recent success of Structured State Space Sequence (S4) models, known for their parameter efficiency, faster inference, and superior handling of long context lengths, we propose a novel application of S4-based models to offline MARL tasks. Our method utilizes S4's efficient convolutional view for offline training and its recurrent dynamics for fast on-policy fine-tuning. To foster scalable cooperation between agents, we sequentially expand the decision-making process, allowing agents to act one after another at each time step. This design promotes bi-directional cooperation, enabling agents to share information via their S4 latent states or memory with minimal communication. Gradients also flow backward through this shared information, linking the current agent's learning to its predecessor. Experiments on challenging MARL benchmarks, including Multi-Robot Warehouse (RWARE) and StarCraft Multi-Agent Challenge (SMAC), demonstrate that our approach significantly outperforms state-of-the-art offline RL and transformer-based MARL baselines across most tasks.

## 1 INTRODUCTION

Multi-agent reinforcement learning (MARL) has demonstrated significant success in learning complex policies that require coordination among multiple agents to maximize a shared objective (Cao et al., 2012; Berner et al., 2019; Ye et al., 2015). However, this success often relies on a substantial number of interactions with the environment, which can be computationally expensive in high-fidelity simulations or prohibitively risky in real-world applications. To enhance sample efficiency, offline reinforcement learning (RL) algorithms (Lee et al., 2021; Fujimoto et al., 2019; Kumar et al., 2019; 2020; Kostrikov et al., 2021; Xu et al., 2022a; Li et al., 2022; Xu et al., 2023) have been developed, enabling learning from pre-collected offline datasets, thus reducing the need for extensive online interactions.

Offline RL is plagued by the well-known issue of distribution shift, which leads to extrapolation errors when encountering out-of-distribution (OOD) samples during policy training. This occurs when the learned policy deviates from the unknown behavior policy used to collect the training data. To mitigate this, various forms of regularization (Kumar et al., 2019) are introduced to ensure that the learned policy remains close to the behavior policy (Kumar et al., 2020; Xu et al., 2023; 2022a). In multi-agent settings, the joint state-action space expands exponentially as the number of agents increases. This makes it challenging to apply these regularization techniques globally on the joint state-action space, leading to sparse and less effective regularization constraints, especially when working with a limited and less diverse offline dataset.

Sequence-based supervised learning has been concurrently applied to address offline MARL, leveraging the significant success of supervised learning in capturing complex patterns from large offline datasets. This approach, first introduced by the Decision Transformer (DT) (Chen et al., 2021), has demonstrated its ability to learn policies in an autoregressive fashion by predicting the next action based on the current state, previous action and the desired return-to-go. While efforts have been made to extend DT-based architectures to offline MARL (Meng et al., 2021; Tseng et al., 2022), certain limitations persist. MADT (Meng et al., 2021) adapts DT independently for each agent in the multi-agent settings, failing to explicitly model cooperation between agents. (Tseng et al., 2022) adopts a similar approach and trains a centralized teacher policy to capture agent interactions, with individual agents learning through policy distillation. However, centralized transformers face scalability issues, requiring training on information from all agents. Additionally, these approaches inherit the inherent limitations of transformers, such as large model sizes, inefficient runtime inference, and restricted ability to capture long-range dependencies due to fixed window size constraints.

Structured State Space Sequence (S4) models (Gu et al., 2021a) have recently been shown to outperform transformer-based models in single-agent offline RL tasks (Bar-David et al., 2023). Building on this success, we propose a sequence-learning-based offline MARL algorithm leveraging S4 variants. These models provide superior parameter efficiency compared to transformers, effectively capture longer temporal contexts, and enable constant-time inference over the quadratic time complexity of transformers with respect to the sequence length. Unlike previous works, such as MADT, which trains agents independently, our method explicitly models cooperation through a Sequentially Expanded MDP (SE-MDP) paradigm. In this framework, recently used in online MARL settings (Li et al., 2023), each decision step is divided into mini-steps, with agents acting sequentially based on their predecessors' actions. Unlike (Li et al., 2023), we enable limited communication, requiring each agent to access only its immediate predecessor's information, shared through the latent state representation of the S4 model. Utilizing this hidden state of the S4 module of the current agent, information on all its prior agents is efficiently passed down to the next agent, and gradients flow backward from the current agent through this shared memory to the previous agents during training. This design enables scalable training with constant memory communication overhead, unlike traditional communication-based MARL algorithms, where memory overhead increases quadratically with the number of agents. Additionally, this streamlined information-sharing mechanism helps mitigate non-stationarity issues during online fine-tuning. This form of training also shares similarities with the way information is passed between segments of long sequences in Recurrent Memory Transformer (RMT) (Bulatov et al., 2022).

The S4-based agents are trained directly on sequences or trajectories from the offline dataset in an efficient convolutional manner. The offline pre-trained models can be further used for sample-efficient online fine-tuning based on individual tuples instead of sequences leveraging the recurrent view of S4. We evaluate the performance of our developed algorithm, called Multi-Agent Decision S4 (MADS4), on the challenging offline MARL benchmarks of Multi-Robot Warehouse (RWARE) (Papoudakis et al., 2020) and StarCraft2 Multi-Agent Challenge (SMAC) (Samvelyan et al., 2019), where MADS4 achieves superior performance across many tasks over state-of-the-art offline RL-based and transformer-based baselines.

## 2 RELATED WORK

Offline Reinforcement Learning Offline RL allows for policy learning based on pre-collected datasets without having access to active interactions with the environment (Levine et al., 2020), which is then directly used as the final policy or is used as a starting point for further improvement (Uchendu et al., 2023). This learning paradigm however results in severe distribution shift and extrapolation errors during policy evaluation on OOD samples not present in the offline dataset (Kumar et al., 2019; Fujimoto et al., 2019). Several approaches have been developed to mitigate this issue which typically involves various types of regularizations to be near the offline data distribution. Policy-based regularizations implicitly or explicitly constrain the policy to be close to the behavior policy of the dataset (Wu et al., 2019; Xu et al., 2021; Cheng et al., 2024; Li et al., 2022). Value-based regularizations aim to learn conservative value functions on OOD samples (Kumar et al., 2020; Kostrikov et al., 2021; Xu et al., 2022b). Other approaches involve including uncertainty (Wu et al., 2021; Bai et al., 2022) or penalizing OOD rewards (Yu et al., 2020).

<div style="text-align: center;"><img src="imgs/img_in_image_box_352_178_874_511.jpg" alt="Image" width="42%" /></div>


<div style="text-align: center;">Figure 1: Multi-Agent MDP (MMDP) is restructured into a Sequentially-Expanded MDP (SE-MDP) where the multi-agent state transition at each timestep is decomposed into n intermediate states. In this framework, each agent processes its input alongside information received from its preceding agent, then takes an action and passes updated information to the next agent in the sequence. During training, gradient flows backward, enabling earlier agents to receive updates based on the information passed by later agents.</div>


On the other hand, Decision Transformer (DT) (Chen et al., 2021) takes a goal-conditioned supervised learning (GCSL) approach to formulate offline RL as a sequence modeling task and outperforms many state-of-the-art offline RL algorithms. Following the success of this training regime, Decision S4 (Bar-David et al., 2023) proposes using S4 model variants for higher parameter efficiency, capturing longer sequences and faster inference.

Offline MARL Extending single-agent RL methods to multi-agent settings presents significant challenges due to the exponential growth of the joint state-action space. Most MARL algorithms adopt the Centralized Training with Decentralized Execution (CTDE) paradigm. In CTDE, global information is shared during training, and local, decoupled policies are used for execution (Oliehoek et al., 2008; Sunehag et al., 2017; Rashid et al., 2020; Son et al., 2019; Wang et al., 2020; Foerster et al., 2017; Lowe et al., 2017; Yu et al., 2021). Recently, offline RL-based MARL algorithms have emerged, typically applying regularizations on local policies or value functions (Yang et al., 2021; Jiang & Lu, 2023; Pan et al., 2022). On the other hand, (Meng et al., 2021) extends the Decision Transformer (DT) (Chen et al., 2021) to a multi-agent setting, where agents are trained independently by sharing weights within a goal-conditioned supervised learning framework.

However, these algorithms do not provide guarantees of global-level regularizations and fail to explicitly or implicitly learn cooperative behavior. Only a few recent works have tried to tackle these limitations. For example, (Wang et al., 2024) uses an implicit global to-local regularization, and (Tseng et al., 2022) uses knowledge distillation to distill cooperation in the local policies.

S4 S4(Gu et al., 2021a;b) and their variants (Gupta et al., 2022; Smith et al., 2022), which are developed on time-invariant linear state space layers, have outperformed transformers in capturing long-range contexts. These models require far fewer parameters and have constant time inference; hence, they have been suitably utilized in reinforcement learning domains in single-agent learning (Bar-David et al., 2023) and in-context learning (Lu et al., 2024). Commonly, the model uses the convolutional mode for efficient parallelizable training (where the whole input sequence is seen ahead of time) and switched into a recurrent mode for efficient autoregressive inference (where the inputs are seen one timestep at a time).

## 3 METHODOLOGY

Sequentially Expanded MDP In this section, we present our approach based on sequence learning with state-space layers. In this work, the Multi-agent Markov Decision Process (MMDP) is transformed into a Sequentially Expanded Markov Decision Process (SE-MDP), where each timestep is divided into n mini timesteps and a multi-agent decision by n agents is expanded into a sequence of n individual decisions, with only one agent acting during each mini-timestep. Thus, a single-step transition in the original MMDP  $ (s^{t}, \boldsymbol{a}^{t}, s^{t+1}) $  resulting in a shared reward  $ r(s^{t}, \boldsymbol{a}^{t}) $  is composed of a sequence of n intermediate transitions, which in turn result in the same shared reward  $ r(s^{t}, \boldsymbol{a}^{t}) $ , as shown in Figure 1.

 $$ (s^{t},\boldsymbol{a}^{t},s^{t+1})=\{(s^{t},a_{1}^{t},s_{a_{1}}^{t}),(s_{a_{1}}^{t},a_{2}^{t},s_{a_{1:2}}^{t}),...,(s_{a_{1:n-1}}^{t},a_{n}^{t},s_{a_{1:n}}^{t}=s^{t+1})\} $$ 

Within this framework, at each timestep, an agent's action is based on information passed by the immediately preceding agent in the sequence. This creates a bidirectional dependency between the agents as shown in Figure 1: in the forward direction, an agent's action is influenced by the actions of its predecessors, while in the backward direction, gradients can propagate from the current agent back to the previous agents.

Sequence-based reinforcement learning Offline RL is formulated as a supervised learning problem by predicting actions in an autoregressive manner typically conditioned on current states, previously executed actions, and desired returns to go. This paradigm was introduced in (Chen et al., 2021), where the DT is trained to predict current actions based on returns to go instead of current rewards in order to have better actions that are correlated with better future rewards. This work utilizes this supervised learning setting, where the state, action, and reward of  $ i^{th} $  agent at each timestep are denoted as  $ s_{i}, a_{i}, r_{i} $ , and its trajectories  $ \tau : (s_{0}, a_{0}, r_{0}, s_{1}, a_{1}, r_{1}, \ldots, s_{L}, a_{L}, r_{L}) $  consist of sequences of state, action, and reward tuples. Since the models are trained on returns-to-go, the trajectories are restructured as  $ \tau : (R_{0}, s_{0}, a_{0}, R_{1}, s_{1}, a_{1}, \ldots, R_{L}, s_{L}, a_{L}) $  where  $ R_{i} = \sum_{t=i}^{N} r_{i} $  is the returns to go from  $ i^{th} $  time step.

S4-based Agent At each time step, the model takes  $ u(t) $  as input and updates the latent state/memory  $ x(t) $  and, in turn, returns an output  $ y(t) $  by following the first-order differential equation parameterized by linear and time-invariant dynamics:



 $$ \begin{aligned}\dot{\boldsymbol{x}}(t)&=\boldsymbol{A}\boldsymbol{x}(t)+\boldsymbol{B}\boldsymbol{u}(t)\\\boldsymbol{y}(t)&=\boldsymbol{C}\boldsymbol{x}(t)+\boldsymbol{D}\boldsymbol{u}(t)\end{aligned} $$ 

The SSM operates on continuous time sequences where A, B, C, D are matrices of appropriate sizes. To model discrete sequences, the continuous SSM can be discretized with a fixed step size  $ \Delta $  following any discretization scheme, such as the bilinear method (Tustin, 1947), to obtain the following discretized linear recurrence relations:

 $$ \begin{aligned}x_{k}=\bar{A}x_{k-1}+\bar{B}u_{k}(t)\\y_{k}=\bar{C}x_{k}+\bar{D}u_{k}\end{aligned} $$ 

<div style="text-align: center;"><img src="imgs/img_in_image_box_580_894_1000_1164.jpg" alt="Image" width="34%" /></div>


<div style="text-align: center;">Figure 2: MADS4 Agent Actor Network. In addition to the state, action, and reward encoder, the memory/S4 state representation from the previous agent  $ (H_{n-1}) $  is encoded through a memory encoder. The updated S4 states and outputs from the S4 blocks pass through separate projection layers.</div>


where  $ \bar{A} $ ,  $ \bar{B} $ ,  $ \bar{C} $ ,  $ \bar{D} $  are calculated based on A, B, C, D and  $ \Delta $ .

(Gu et al., 2020) showed that the initialization of A by the HIPPO matrix enables the SSM model to capture better long-range contexts. Various techniques have since been developed to improve the model's performance, stability, and training efficiency (Gu et al., 2021a; Gupta et al., 2022). Similar to these works, D is represented here by a skip connection.

Since Eq. 3 is linear and time-invariant, the output sequence  $ y(t) $  can be computed directly in parallel based on input sequence  $ u(t) $  by convolution as follows:

 $$ \begin{aligned}y_{k}&=\bar{C}\bar{A}^{k}\bar{B}u_{0}+\bar{C}\bar{A}^{k-1}\bar{B}u_{1}+\ldots+\bar{C}\bar{A}\bar{B}u_{k-1}+\bar{C}\bar{B}u_{k}\\y&=\bar{K}*u\end{aligned} $$ 

where  $ \bar{K} $  is the SSM convolution kernel or filter which is a function of  $ \bar{A}, \bar{B}, \bar{C}, \bar{D} $  and context length L which is pre-fixed during training. This non-circular convolution can be computed efficiently across all time steps, allowing for parallelizable training. The recurrent view of the SSM also allows for faster inference with low memory. This is a key advantage over transformers, which makes the use of SSMs very effective in reinforcement learning settings, which require faster inference for the collection of online interactions with the environment. Additional details can be found in Appendix A.

Information sharing with limited communication To enable scalable cooperation among S4-based agents, we design a communication mechanism limited to consecutive agents in the SE-MDP sequence. Each agent's memory, represented by the hidden state of its S4 module, encodes information about all prior agents in the sequence. A projection of this latent state,  $ h_{i-1}^{t} $ , is passed as input to the next agent along with other inputs  $ \hat{u}_{i}^{t} $ , influencing its action  $ a_{i}^{t} $  and its memory  $ h_{i}^{t} $ :

 $$ a_{i}^{t},h_{i}^{t}=\pi_{i}(\hat{u}_{i}^{t},h_{i-1}^{t};\theta_{i}) $$ 

During training, gradients flow backward through the shared latent states, enabling the entire system to learn cooperative strategies:

 $$ \frac{\partial J}{\partial\theta_{i}}=\frac{\partial J}{\partial a_{i}^{t}}\cdot\frac{\partial a_{i}^{t}}{\partial\theta_{i}}+\frac{\partial J}{\partial a_{i+1}^{t}}\cdot\frac{\partial a_{i+1}^{t}}{\partial h_{i}^{t}}\cdot\frac{\partial h_{i}^{t}}{\partial\theta_{i}}. $$ 

where J represents the supervised loss function computed across all agents in the system. This sequential flow of information eliminates the need for an agent to communicate with more than one peer or identify useful collaborators, a challenge that grows with the number of agents. In contrast to typical communication-based MARL algorithms, which scale poorly due to the quadratic growth in memory requirements during training and execution, our mechanism is highly efficient, requiring only constant memory per agent.

We first adapt Decision S4 (DS4) for each agent with parameter sharing, similar to the Multi-Agent Decision Transformer (MADT) (Meng et al., 2021). Unlike MADT, however, the Multi-Agent Decision S4 (MADS4) is trained in a sequentially dependent manner, where agents can share accumulated memory information with the next agent in the sequence. The offline version of MADS4, trained on pre-collected trajectories, is detailed in the next section. These pre-trained models can be further fine-tuned in an on-policy setting using MAPPO (Yu et al., 2021).

### 3.1 MADS4: Offline Training

Input Formulation In the offline training setup of MADS4, each agent is trained over the offline trajectories consisting of sequences of previously seen observations, its own previously executed actions, the latent state representation of its preceding agent, and returns to go from the current time step. Similar to MADT, the state of each agent at each time step  $ s_{i}^{t} $  is composed of global environment state  $ s_{gi}^{t} $  and its local observation  $ o_{i}^{t} $ . However, our model performs very similarly without using the global state information in the input, as shown in Appendix C.5. Thus, a trajectory for the  $ i^{th} $  agent, which is taken as input to the S4-based model, consists of the following:

 $$ \tau_{i}=(u^{1},u^{2},...,u^{T})\quad where\quad u^{t}=\{R^{t},s_{gi}^{t},o_{i}^{t},a_{i}^{t-1},h_{i-1}^{t}\} $$ 

where  $ R^{t} $  is the returns-to-go from current time step t,  $ s_{gi}^{t} $  is the current shared global state,  $ o_{i}^{t} $  is the current local observation,  $ a_{i}^{t-1} $  is the previously executed action of the  $ i^{th} $  agent,  $ h_{i-1}^{t} $  is the current hidden state representation of the preceding agent. The model is trained to predict actions (action logits) at time step t in an autoregressive manner based on the data seen so far. The output of the S4-based model is used as the action probabilities which are sampled after applying the action availability masks.

 $$ \hat{a}_{i}^{t}=\arg\max_{a}P(a_{i}^{t}|\tau_{i}^{<=t};\theta) $$ 

where  $ \theta $  are the parameters of the MADS4. In this work, parameter sharing is allowed across the agents for training stability, and thus, essentially, a single model is trained, which takes into account different inputs for different agents along with their specific one-hot agent IDs.

Network Architecture and Training The MADS4 architecture consists of three key components, as illustrated in Figure 2: (i) Input, state, and output projection layers: each of these consists of a fully connected layer followed by ReLU activations; (ii) Input encoder layers: These layers handle states, actions, and rewards/returns, each implemented as fully connected linear layers; (iii) Sequence modeling component: This component consists of stacked S4 blocks, where each block consists of Batch Normalization layer followed by S4 layer, linear mixing layer with GELU activation, and a dropout layer. We employed various kernels for S4, with the "Normal Plus Low Rank" kernel initialized using HIPPO, achieving the best performance. In all experiments, we set the input channel size to H = 96 and the S4 state size to N = 96. An ablation study detailing the effects of varying state and input sizes is provided in Appendix C.1.

In the offline setting, the S4 model is trained efficiently using the convolutional view on entire trajectories sampled randomly from the offline dataset. The trajectories are zero-padded to a constant context length. Unlike transformers, which face limitations on context length due to the expensive quadratic time and space complexity of self-attention, S4-based models can be trained on complete trajectories that are often much longer than those typically used for transformers in most environments. The impact of truncating trajectory lengths has significant implications for model performance, as detailed in Appendix C.2. Actions are predicted based on the action logit outputs of the model, and the model is trained based on loss computed using cross-entropy between the true action labels and the predicted actions.

### 3.2 MADS4: ONLINE FINETUNING

For online fine-tuning, the offline pre-trained agent is used to interact with the online environment and is further updated based on an on-policy training scheme. The agent interacts with the environment while creating the buffer, which stores the local observations and actions of the individual agents, shared global states of the environment, rewards, returns-to-go, and also the latent states of the S4 modules of the agents. Within the well-known MAPPO-based (Yu et al., 2021) actor-critic framework, the offline-pretrained S4-based model is used as actor networks of the agents which predict actions via the recurrent view based on latent states and other inputs as used in the pretraining stage. The critic network is conditioned on both the global states of the environment as well as the encoded latent S4 states to evaluate the state value function.

Network Architecture and Training The pre-trained model is loaded as the actor network, which predicts action probabilities and next states as:

 $$ p_{i}^{t},h_{i}^{t}=\pi(u_{i}^{t},h_{i}^{t-1};\theta)\quad where\quad u_{i}^{t}=\{R_{i}^{t},s_{gi}^{t},o_{i}^{t},a_{i}^{t-1},h_{i-1}^{t}\} $$ 

The critic network is parameterized by fully connected layers with ReLU activations, which take the shared global state of the environment and encoded latent states and evaluate the value function, which is used to update the S4-based actor parameters ( $ \theta $ ) using the policy gradient theorem. For

Algorithm 1 MADS4-Offline Training

Input: Offline dataset $\mathcal{D}: \{\tau_i : \langle s_{gi}^t, o_i^t, a_i^t, v_i^t, d_i^t, R_i^t \rangle_{t=1}^T\}_{i=1}^n$, where $n$ is the number of agents, and $v_i^t$ denotes the available actions for the $i^{th}$ agent at time $t$; $d_i^t$ denotes the done signal for an episode

Initialize $\alpha$ as the learning rate, $K$ as the context length

Initialize $\theta$ for the S4 models based on HIPPO initialization

1: for $i = 1:n$ do ▷ Iterate over each agent
2: From $\tau_i$ and $h_{i-1}^t$, create $X_i: \{R_i^t, s_{gi}^t, o_i^t, a_i^{t-1}, h_{i-1}^t\}_{t=1}^T$, where $h_{i-1}^t$ is the latent state representation of the previous agent, assuming $h_{-1}^t = 0$
3: Zero-pad the trajectory to a constant length $K$ when $d_i^t$ is true ▷ Pad when the agent is done
4: Compute action output sequence $\hat{a}_i = \{\hat{a}_i^1, \ldots, \hat{a}_i^T\}$ and latent state projections $h_i = \{h_1^1, \ldots, h_i^T\}$
5: for $t = 1:T$ do ▷ Loss calculation over time steps
6: Mask illegal actions via $P(\hat{a}_{ij}^t \mid \tau_i^{<t}; \theta) = 0$ if $v_{ij}^t$ is False, where $j$ is the unavailable action index
7: Predict the action $\hat{a}_i^t = \arg \max_j P(\hat{a}_{ij} \mid \tau_i^{<t}; \theta)$
8: Update $\theta$:
    $\theta \leftarrow \arg \max_\theta \frac{1}{K} \sum_{t=1}^{K} P(a_i^t) \log P(\hat{a}_i^t \mid \tau_i^{<t}; \theta)$
9: end for
10: end for
Return: $\theta$

more stable training. the actor network is kept frozen initially, and the critic is solely trained on the recorded data collected using the pre-trained actor. After sufficient training of the critic, the actor and critics are simultaneously trained. During exploration, the desired returns-to-go is set at 10% higher than the current model's highest return. Additional details on the experimental setup and training are provided in Appendix A.4.

## 4 EXPERIMENTS

Datasets We evaluate the performance of MADS4 on challenging cooperative MARL benchmarks of Multi-Robot Warehouse (RWARE) (Papoudakis et al., 2020) and StarCraft2 Multi-Agent Challenge (SMAC) (Samvelyan et al., 2019). The offline datasets in the RWARE domain are obtained from (Matsunaga et al., 2023), which consists of diverse trajectories collected by training Multi-Agent Transformer (MAT) (Wen et al., 2022). The RWARE datasets consist of expert policies trained on 2 maps (tiny and small) with different numbers of agents. For the SMAC domain, the datasets provided by (Meng et al., 2021) have been used, which consists of trajectories collected with online trained MAPPO agents. The datasets consist of three trained quality levels of the agents, good, medium, and poor, tested on the different SMAC maps. For this work, we chose four representative maps consisting of two hard (5m vs. 6m, 2c vs. 64zg) and two super hard (6h vs. 8z, corridor) maps for evaluating MADS4. Additional statistics on the offline datasets can be found in Appendix B.

Baselines For comparisons on both domains, we compare with several recent offline MARL algorithms from the paradigms of both offline reinforcement learning and sequence-based supervised learning. The offline RL baselines considered for comparison are Behaviour Cloning (BC) (Fujimoto et al., 2019), OptiDICE (Lee et al., 2021), AlberDICE (Matsunaga et al., 2023), ICQ (Yang et al., 2021), OMAR(Pan et al., 2022) and OMIGA(Wang et al., 2024). The sequence-based learning algorithms considered in this work include MADT (Meng et al., 2021) and MADTKD (Tseng et al., 2022), which are based on transformers. MADT policies do not involve any cooperation during learning, whereas MADTKD incorporates a degree of cooperation distilled into the agents from the centralized teacher model.

Algorithm 2 MADS4: On-policy finetuning

1: Copy the model weights  $ \theta $  to the actor or policy network  $ \pi : \pi(u_i) $ , where  $ u_i = \{P_i^t, s_{gi}^t, o_i^t, a_i^{t-1}, h_{i-1}^t\} $ ; Initialize  $ \phi $ , the parameters for critic V

2: Set learning rates  $ \alpha_\pi $ ,  $ \alpha_V $  for actor and critic

3: for iterations = 1, M do

4: Set data buffer  $ D = \{\} $ 

5: for i = 1 to batch_size do

6:  $ \tau = [] $  ▷ Empty list

7: Initialize  $ h_0^{(1)}, \ldots, h_0^{(n)} $  actor S4 states

8: for each timestep t in the environment do

9: for agent = 1 : n do

10:  $ p_i^t, h_i^t = \pi(u_i^t, h_{i-1}^t; \theta) $ 

11: Sample  $ a_i^t \sim p_i^t $ 

12: end for

13: Execute actions  $ a^t $ , observe  $ r^t $ ,  $ s_g^{t+1} $ ,  $ o^{t+1} $ 

14:  $ \tau^+ = [s_t, o_t, h_t, a_t, r_t, R_t, s_{t+1}, o_{t+1}] $ 

15: end for

16: Calculate advantage  $ A^t $  via GAE on  $ \tau $  and store  $ \tau $  with  $ A^t $  in the buffer D

17: end for

18: for  $ k = 1, \ldots, K $  training steps do

19: Sample batch from replay-buffer  $ B = \{(s_g^t, o_t^t, a^{t-1}, R^{t-1}, h^{t-1}, s_g^{t+1}, o^{t+1}, a^t, h^t, r^t, R^t, A^t)\} \subset D $  for each agent

20: Calculate Bellman target estimate:  $ y = r^t + \gamma V(s^{t+1}, h^t) $ 

21: Update critic:  $ \phi_V = \phi_V - \alpha_V \nabla_{\phi_V} (V(s^t, h^{t-1}) - y)^2 $ 

22: If actor freezing is over, update actor:

 $ \theta_\pi \leftarrow \arg \max_{\theta_\pi} E_s \sim \rho_{\theta_\pi}, a \sim \pi_{\theta_\pi} $  [clip(w, 1 -  $ \epsilon $ , 1 +  $ \epsilon $ )  $ A^t] $ 

where the importance weight  $ w = \frac{\pi_\theta(a_i|o_i)}{\pi_{\theta_\pi}(a_i|o_i)} $ 

23: end for

24: end for

Offline training Here, we compare the performance of offline trained MADS4, where the agents share information in the form of their latent state projections. The trained agents are deployed on the online RWARE and StarCraft2 environments for evaluation. Tables 1 and 2 show the mean and standard deviation of average returns in RWARE and SMAC domains, respectively, evaluated over 30 episodes and 5 different training seeds. During evaluation, the desired returns-to-go is set at 10% higher than the highest returns encountered in the offline datasets.

RWARE environment is a warehouse simulation consisting of agents moving and delivering goods to workstations in partially observable settings while avoiding collisions. This domain poses challenges due to high-dimensional observations and the need for strong cooperation, especially in high-density settings where agents must navigate narrow passages. In this domain, MADS4 outperforms all baselines across the maps, with a larger performance gap on the small and tiny maps involving 6 agents, where tight coordination is crucial to avoid collisions in confined spaces. MADS4 also outperforms transformer-based baselines like MADTKD, likely due to the long trajectories in the RWARE datasets (up to 500 timesteps), which are often truncated to reduce transformer training costs. In contrast, MADS4 processes full trajectories, capturing longer contexts with fewer parameters.

In the SMAC domain, MADS4 demonstrates consistent performance that is similar to or better than the considered baselines across all studied maps. Notably, the model outperforms all baselines in the hard and superhard maps, specifically in the 2c vs. 64zg and 6h vs. 8z scenarios.

On-policy fine-tuning We evaluate whether the performance of offline pre-trained models can be enhanced through on-policy fine-tuning. During this phase, MADS4 interacts with the environment, collecting trajectories that are stored in a buffer and used to update the S4-based models via recur-

<div style="text-align: center;">Table 1: Average returns and standard deviations over 5 random seeds on the Warehouse domain.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">Method</td><td colspan="3">Tiny (11x11)</td><td colspan="3">Small (11x20)</td></tr><tr><td style='text-align: center;'>(N=2)</td><td style='text-align: center;'>(N=4)</td><td style='text-align: center;'>(N=6)</td><td style='text-align: center;'>(N=2)</td><td style='text-align: center;'>(N=4)</td><td style='text-align: center;'>(N=6)</td></tr><tr><td style='text-align: center;'>BC</td><td style='text-align: center;'>8.80  $ \pm $  0.25</td><td style='text-align: center;'>11.12  $ \pm $  0.19</td><td style='text-align: center;'>14.06  $ \pm $  0.32</td><td style='text-align: center;'>5.54  $ \pm $  0.06</td><td style='text-align: center;'>7.88  $ \pm $  0.14</td><td style='text-align: center;'>8.90  $ \pm $  0.13</td></tr><tr><td style='text-align: center;'>ICQ</td><td style='text-align: center;'>9.38  $ \pm $  0.75</td><td style='text-align: center;'>12.13  $ \pm $  0.44</td><td style='text-align: center;'>14.59  $ \pm $  0.16</td><td style='text-align: center;'>5.43  $ \pm $  0.19</td><td style='text-align: center;'>7.93  $ \pm $  0.19</td><td style='text-align: center;'>8.87  $ \pm $  0.22</td></tr><tr><td style='text-align: center;'>OMAR</td><td style='text-align: center;'>6.77  $ \pm $  0.64</td><td style='text-align: center;'>14.39  $ \pm $  0.91</td><td style='text-align: center;'>16.13  $ \pm $  1.21</td><td style='text-align: center;'>4.40  $ \pm $  0.34</td><td style='text-align: center;'>7.12  $ \pm $  0.38</td><td style='text-align: center;'>8.41  $ \pm $  0.49</td></tr><tr><td style='text-align: center;'>MADTKD</td><td style='text-align: center;'>6.24  $ \pm $  0.60</td><td style='text-align: center;'>9.90  $ \pm $  0.21</td><td style='text-align: center;'>13.06  $ \pm $  0.19</td><td style='text-align: center;'>3.65  $ \pm $  0.34</td><td style='text-align: center;'>6.85  $ \pm $  0.36</td><td style='text-align: center;'>7.85  $ \pm $  0.52</td></tr><tr><td style='text-align: center;'>OptiDICE</td><td style='text-align: center;'>8.70  $ \pm $  0.06</td><td style='text-align: center;'>11.13  $ \pm $  0.44</td><td style='text-align: center;'>14.02  $ \pm $  0.36</td><td style='text-align: center;'>4.84  $ \pm $  0.32</td><td style='text-align: center;'>7.68  $ \pm $  0.09</td><td style='text-align: center;'>8.47  $ \pm $  0.26</td></tr><tr><td style='text-align: center;'>AlberDICE</td><td style='text-align: center;'>11.15  $ \pm $  0.35</td><td style='text-align: center;'>13.11  $ \pm $  0.32</td><td style='text-align: center;'>15.72  $ \pm $  0.36</td><td style='text-align: center;'>5.97  $ \pm $  0.11</td><td style='text-align: center;'>8.18  $ \pm $  0.19</td><td style='text-align: center;'>9.65  $ \pm $  0.13</td></tr><tr><td style='text-align: center;'>MADS4 (ours)</td><td style='text-align: center;'>11.79  $ \pm $  0.61</td><td style='text-align: center;'>15.52  $ \pm $  0.20</td><td style='text-align: center;'>17.29  $ \pm $  0.76</td><td style='text-align: center;'>6.58  $ \pm $  0.28</td><td style='text-align: center;'>9.47  $ \pm $  0.15</td><td style='text-align: center;'>10.87  $ \pm $  0.55</td></tr></table>

<div style="text-align: center;">Table 2: Average returns and standard deviations over 5 random seeds on the SMAC domain.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td rowspan="2">SMAC Map</td><td rowspan="2">Data</td><td colspan="3">RL-based</td><td colspan="2">Sequence-based</td></tr><tr><td style='text-align: center;'>ICQ</td><td style='text-align: center;'>OMAR</td><td style='text-align: center;'>OMIGA</td><td style='text-align: center;'>MADT</td><td style='text-align: center;'>MADS4 (ours)</td></tr><tr><td rowspan="3">5m vs 6m (H)</td><td style='text-align: center;'>G</td><td style='text-align: center;'>7.87 \pm 0.30</td><td style='text-align: center;'>7.40 \pm 0.63</td><td style='text-align: center;'>8.25 \pm 0.37</td><td style='text-align: center;'>8.15 \pm 0.63</td><td style='text-align: center;'>8.00 \pm 0.45</td></tr><tr><td style='text-align: center;'>M</td><td style='text-align: center;'>7.77 \pm 0.30</td><td style='text-align: center;'>7.08 \pm 0.51</td><td style='text-align: center;'>7.92 \pm 0.57</td><td style='text-align: center;'>7.80 \pm 0.56</td><td style='text-align: center;'>7.85 \pm 0.57</td></tr><tr><td style='text-align: center;'>P</td><td style='text-align: center;'>7.26 \pm 0.19</td><td style='text-align: center;'>7.27 \pm 0.42</td><td style='text-align: center;'>7.52 \pm 0.21</td><td style='text-align: center;'>7.23 \pm 0.48</td><td style='text-align: center;'>7.67 \pm 0.15</td></tr><tr><td rowspan="3">2c vs 64zg (H)</td><td style='text-align: center;'>G</td><td style='text-align: center;'>18.82 \pm 0.17</td><td style='text-align: center;'>17.27 \pm 0.78</td><td style='text-align: center;'>19.15 \pm 0.32</td><td style='text-align: center;'>18.90 \pm 0.78</td><td style='text-align: center;'>19.40 \pm 0.55</td></tr><tr><td style='text-align: center;'>M</td><td style='text-align: center;'>15.57 \pm 0.61</td><td style='text-align: center;'>10.20 \pm 0.20</td><td style='text-align: center;'>16.03 \pm 0.19</td><td style='text-align: center;'>16.92 \pm 0.20</td><td style='text-align: center;'>17.27 \pm 0.15</td></tr><tr><td style='text-align: center;'>P</td><td style='text-align: center;'>12.56 \pm 0.18</td><td style='text-align: center;'>11.33 \pm 0.50</td><td style='text-align: center;'>13.02 \pm 0.66</td><td style='text-align: center;'>13.33 \pm 0.50</td><td style='text-align: center;'>14.67 \pm 0.32</td></tr><tr><td rowspan="3">6h vs 8z (SH)</td><td style='text-align: center;'>G</td><td style='text-align: center;'>11.81 \pm 0.12</td><td style='text-align: center;'>9.85 \pm 0.28</td><td style='text-align: center;'>12.54 \pm 0.21</td><td style='text-align: center;'>12.55 \pm 0.67</td><td style='text-align: center;'>12.75 \pm 0.15</td></tr><tr><td style='text-align: center;'>M</td><td style='text-align: center;'>11.13 \pm 0.33</td><td style='text-align: center;'>10.36 \pm 0.16</td><td style='text-align: center;'>12.31 \pm 0.22</td><td style='text-align: center;'>12.36 \pm 0.16</td><td style='text-align: center;'>12.57 \pm 0.25</td></tr><tr><td style='text-align: center;'>P</td><td style='text-align: center;'>10.55 \pm 0.10</td><td style='text-align: center;'>10.63 \pm 0.25</td><td style='text-align: center;'>11.67 \pm 0.19</td><td style='text-align: center;'>11.63 \pm 0.25</td><td style='text-align: center;'>11.89 \pm 0.43</td></tr><tr><td rowspan="3">corridor (SH)</td><td style='text-align: center;'>G</td><td style='text-align: center;'>15.54 \pm 1.12</td><td style='text-align: center;'>6.74 \pm 0.69</td><td style='text-align: center;'>15.88 \pm 0.89</td><td style='text-align: center;'>17.81 \pm 1.14</td><td style='text-align: center;'>16.02 \pm 0.97</td></tr><tr><td style='text-align: center;'>M</td><td style='text-align: center;'>11.30 \pm 1.57</td><td style='text-align: center;'>7.26 \pm 0.71</td><td style='text-align: center;'>11.66 \pm 1.30</td><td style='text-align: center;'>12.75 \pm 1.18</td><td style='text-align: center;'>12.80 \pm 1.12</td></tr><tr><td style='text-align: center;'>P</td><td style='text-align: center;'>4.47 \pm 0.33</td><td style='text-align: center;'>4.28 \pm 0.49</td><td style='text-align: center;'>5.61 \pm 0.35</td><td style='text-align: center;'>8.76 \pm 0.49</td><td style='text-align: center;'>8.57 \pm 0.54</td></tr></table>

In the presence of a non-policy training build upon and improves the offline pretraining results. Furthermore, on-policy training without pretraining consistently results in sub-optimal performance across all tasks, underscoring the importance of pretraining for achieving superior results.

However, prolonged on-policy training can sometimes degrade the performance of pre-trained models, as shown in Appendix C.2. This degradation likely arises from the inherent instability of training S4 modules in a recurrent setup, compared to the more stable convolution-based operations employed during offline pretraining, leading to error accumulation. To address this, we mitigate the issue by freezing the S4 kernel parameter A, which governs state-to-state transitions independent of inputs, and fine-tuning only the input-dependent parameters B and C.

The effect of sharing information The sharing of information between agents leads to significantly improved cooperative behavior, as reflected in the higher average rewards shown in Figure 4. This performance boost is particularly pronounced in more complex tasks that involve a greater number of agents and demand precise coordination. Importantly, this method of sharing information is scalable, where an agent only needs to communicate with the next agent, minimizing overhead while ensuring efficient coordination. Information can be shared in multiple forms, namely by pass-

<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_1255_415_1394.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_413_1255_612_1394.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_610_1255_809_1394.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_808_1255_1006_1394.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">Figure 3: Training curves of on-policy training of MADS4 with and without offline pretraining. Mean and standard deviations of average returns are plotted over 5 independent runs.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_165_415_305.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_412_165_611_306.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_609_166_810_306.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_807_166_1006_305.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">Figure 4: Training curves of MADS4 with information sharing between consecutive agents and IDS4 where agents are trained independently. Mean and standard deviations of average returns are plotted over 5 independent runs.</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_215_426_414_562.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">(a)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_411_425_612_562.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_607_424_807_562.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">(b)</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_806_424_1007_562.jpg" alt="Image" width="16%" /></div>


<div style="text-align: center;">Figure 5: (a) The effect of having a shuffled random order vs. a fixed sorted order of the agents in the SE-MDP framework on the RWARE domain in the small 6 agents scenario and SMAC domain in the 2c vs. 64zg map. (b) The comparison of the performance of MADS4 vs. MADS4-dec (decentralized MADS4) on RWARE small map with 6 agents and SMAC map 2c vs 64zg.</div>


ing the action logit outputs from the S4 model, the latent state representations, or a combination of both from the preceding agent. A more detailed analysis of how different types of shared information impact performance is provided in Appendix A.2.

Effect of order of agents We evaluated the impact of agent ordering on MADS4's performance by comparing two settings: (1) Random Order, where agents are shuffled during training, and (2) Sorted Order, where the dataset order is preserved. Figure 5(a) shows similar performance, demonstrating MADS4's robustness to agent ordering in the SE-MDP framework. Notably, our approach only requires each agent to communicate with one unique peer which can be selected randomly to ensure that every agent's information is passed across the network without the need for any centralized optimization or sophisticated coordination.

MADS4 in decentralized setting To adapt MADS4 for a decentralized setting, where agents act in parallel, we leverage the hidden state information of each agent from the previous timestep as a proxy for the current timestep. This approach removes the sequential dependency in updating agent memory, as all agents' memory information from the previous timestep is available when making decisions at the current timestep. Since memory accumulates over multiple timesteps, relying on the previous timestep's information does not compromise performance, as demonstrated in Figure 5(b). This modification enables our algorithm to function effectively in decentralized policy settings without performance degradation.

## 5 CONCLUSIONS

In this work, we showcase the effectiveness of S4-based models in surpassing transformer-based architectures for sequence-to-sequence offline multi-agent reinforcement learning (MARL) tasks. By restricting communication to the exchange of information between unique, arbitrarily selected pairs of agents, MADS4 fosters superior cooperation compared to state-of-the-art offline RL and centralized transformer-based baselines, which require complete access to all agents' information during training. MADS4 offers a low-latency and lightweight model that can be trained more efficiently than transformers and fine-tuned online using recurrent computations.