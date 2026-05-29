# PRF: PARALLEL RESONATE AND FIRE NEURON FOR LONG SEQUENCE LEARNING IN SPIKING NEURAL NETWORKS

Anonymous authors

Paper under double-blind review

## ABSTRACT

Recently, there is growing demand for effective and efficient long sequence modeling, with State Space Models (SSMs) proving to be effective for long sequence tasks. To further reduce energy consumption, SSMs can be adapted to Spiking Neural Networks (SNNs) using spiking functions. However, current spiking-formalized SSMs approaches still rely on float-point matrix-vector multiplication during inference, undermining SNNs' energy advantage. In this work, we address the efficiency and performance challenges of long sequence learning in SNNs simultaneously. First, we propose a decoupled reset method for parallel spiking neuron training, reducing the typical Leaky Integrate-and-Fire (LIF) model's training time from  $ O(L^{2}) $  to  $ O(L \log L) $ , effectively speeding up the training by  $ 6.57 \times $  to  $ 16.50 \times $  on sequence lengths 1, 024 to 32, 768. To our best knowledge, this is the first time that parallel computation with a reset mechanism is implemented achieving equivalence to its sequential counterpart. Secondly, to capture long-range dependencies, we propose a Parallel Resonate and Fire (PRF) neuron, which leverages an oscillating membrane potential driven by a resonate mechanism from a differentiable reset function in the complex domain. The PRF enables efficient long sequence learning while maintaining parallel training. Finally, we demonstrate that the proposed spike-driven architecture using PRF achieves performance comparable to Structured SSMs (S4), with two orders of magnitude reduction in energy consumption, outperforming Transformer on Long Range Arena tasks. $ ^{1} $ 

## 1 INTRODUCTION

Long Sequence Modeling. Long sequence modeling is a fundamental problem in machine learning. This significant advancement has yielded wide-ranging impact across various fields (such as Transformer (Vaswani et al., 2017) and Mamba (Gu & Dao, 2023)), including reinforcement learning (e.g., robotics and autonomous driving) (Chen et al., 2021), autoregressive task (e.g., large language models) (Zhao et al., 2023), generative tasks (e.g., diffusion model) (Peebles & Xie, 2023) and etc. The Transformer architecture (Vaswani et al., 2017), which combines token mixing with self-attention and channel mixing with dense matrices, has been successfully applied in these fields, since sequence data (with length L) is well modeled by the self-attention mechanism.

State space models (SSMs) (Gu et al., 2022a) effectively address the limitations of self-attention mechanism in processing long sequences by reducing computational complexity from  $ O(L^{2}) $  to  $ O(L) $  during inference (Feng et al., 2024). The recurrent form of SSMs allows themselves to scale to longer sequence lengths more efficiently. Furthermore, the (Structured) SSMs utilize orthogonal polynomial bases to initialize recurrent weights (Gu et al., 2020) for token mixing, and to compress input history (Gu et al., 2021) into hidden state that enables long sequence learning yielding superior performance than Transformer models on long sequence tasks (Gu et al., 2022a). As a result, variants of SSMs (e.g., Mamba (Gu & Dao, 2023)) are successfully applied across various fields. However, SSMs still requires large number of float-point matrix-vector multiplications for both token mixing and channel mixing to effectively extract information. These dense matrix multiplication computations scale quadratically with model size (D), consuming significant amount of energy during inference.

Spiking Neural Networks. SNNs benefit from the ability to convert the Float-Point (FP) Multiply Accumulate (MAC) computations into sparse Accumulate (AC) computations through their spike-driven mechanism (Hu et al., 2021; Yao et al., 2024). Prior arts incorporated the spiking function at the input of token and channel mixing in SSMs to reduce energy consumption in channel mixing (Stan & Rhodes, 2023; Bal & Sengupta, 2024; Shen et al., 2024). However, the token mixing operation within the SSMs blocks has yet been optimized, resulting in extensive FP matrix-vector multiplications and thus compromising the overall energy efficiency (Stan & Rhodes, 2023). To avoid FP matrix-vector multiplication computations in both channel mixing and token mixing, it is also crucial to reduce the complexity of token mixing. Therefore, we must revisit and design novel the spiking neurons in SNNs to enabl effective long sequence learning capabilities, while maintaining computation efficiency.

The Challenges for SNNs. There are two challenges in implementing long sequence learning in SNNs. (i) First, the Backpropagation Through Time (BPTT) is the commonly used training method for SNNs (Wu et al., 2018). However, BPTT leads to a quadratic growth in training time with respect to sequence length, scaling as  $ O(L^{2}) $  (Kag & Saligrama, 2021). While there are some existing SNN efficient training methods that can improve training efficiency (Bellec et al., 2020; Xiao et al., 2022; Yin et al., 2023), they are still outperformed by BPTT for longer sequences (Meng et al., 2023). (ii) Secondly, commonly used spiking neuron models struggle to capture long-range dependencies. Specifically, the widely used LIF neuron has difficulty in capturing and distinguishing dependencies in membrane potential over long intervals, which limits its performance on long-range tasks. Previous work attempted to address the training efficiency and performance improvement separately by improving neuron models (Fang et al., 2024; Spieler et al., 2024). (Fang et al., 2024; Spieler et al., 2024). However, efforts made to tackle these challenges on long sequence tasks at the same time remains unveiled. In this work, we aim for addressing these two challenges simultaneously. Our contributions are summarized as follows:

• To accelerate the training process, we propose a novel decoupled reset method to implement parallel training that is equivalent to sequential training. This approach accelerates the back propagation by three orders of magnitude and can be applied to any types of spiking neurons.

• To effectively extract long range dependencies, we propose the Parallel Resonate and Fire (PRF) neuron with oscillating membrane potential in complex domain leveraging an adaptive and differentiable reset mechanism.

• To minimize inference energy, we further incorporate PRF into the design of Spike-Driven Temporal and Channel Mixer (SD-TCM) module. This module achieves performance comparable to S4 in long range arena tasks while reducing the inference energy by two orders of magnitude.

## 2 RELATED WORK

State Space Models A general discrete form of SSMs is given by the equation:  $ u_{t} = Au_{t-1} + Bx_{t} $ ,  $ y_{t} = Cu_{t} $ , where  $ A \in R^{H \times H} $ ,  $ B \in R^{D \times H} $ ,  $ C \in R^{H \times D} $  matrices is shape with the model size D and hidden size H. The success of the Structured SSMs (S4) arises from the fact that the coefficients of the orthogonal bases are solved to fit an arbitrary sequence curve (Gu et al., 2020). By leveraging orthogonal polynomial bases for initializing structured matrices A and B, S4 effectively compresses input history and outperforms transformers in long-range sequence tasks (Gu et al., 2021). Subsequent variants of SSMs have also achieved great success on long sequence tasks (Gu et al., 2022a; Goel et al., 2022b; Gu et al., 2023; 2022b; Orvieto et al., 2023). However, these SSMs-based models still exist power consumption issues that scale quadratically with model size D. This is largely due to the use of the dense matrices for channel mixing after  $ y_{t} $ , such as in the GLU block (Dauphin et al., 2017), which requires  $ 2D^{2} $  computation in matrices, leading to a significant number of FP-MAC operations.

Spikinglized SSMs The spike mechanism can alleviate the energy problem in dense matrices by converting the FP-MAC as sparse FP-AC computation. Some spiking-formalized (spikinglized) approaches integrate the spike function into SSMs after token mixing to reduce the FP-MAC computation of channel mixing. For instance, Oliver et al.(Stan & Rhodes, 2023) make intersection of SNNs with S4D model (Gu et al., 2022b) for long-range sequence modelling, by adding Heaviside function to token mixing output,  $ y_{t} $ , at each SSMs layer. Similarly, Abhronil et al.(Bal & Sengupta,

(2024) combine stochastic spiking function at the output of  $ y_{t} $ . These spikingized methods can harvest the long sequence learning capability of SSMs models, but also retain nonlinear activation computation and FP matrix-vector multiplications, as they retain the A, B and C matrices during recurrent inference. However, this retention limits the energy efficiency advantages of SNNs and poses challenges for deploying the model on neuromorphic chips. Therefore, we aim to further optimize the inference process by reducing matrices A and B to vectors and eliminating C. This requires rethinking the role of spiking neurons in long sequence learning.

## 3 PROBLEM FORMULATION

In this section, we first introduce the commonly used Leaky Integrate-and-Fire (LIF) neuron, then we describe the two main challenges for long range learning ability with spiking neuron.

### 3.1 THE LEAKY INTEGRATE-AND-FIRE (LIF) NEURON

The LIF model is a widely used spiking neuron model. It simulates neurons by integrating input signals and firing spikes when the membrane potential exceeds a threshold. The dynamic of membrane potential  $ u(t) $  is followed by:

 $$ \frac{d u(t)}{d t}=-\frac{1}{\tau}\left(u(t)-u_{\mathrm{r e s e t}}\right)+\frac{R}{\tau}c(t), $$ 

where the  $ c(t) $  is the input current. The constants  $ \tau $ ,  $ u_{reset} $ , and R denote the membrane time constant, reset potential, and resistance. We use  $ R = \tau $  as in previous work. When  $ u(t) $  reaches the threshold  $ V_{th} $ , the neuron fires a spike, and then  $ u(t) $  resets to  $ u_{reset} $ . The discrete LIF model is expressed as:

 $$ u_{t}=\beta\cdot\left(u_{t-1}-V_{\mathrm{th}}s_{t-1}\right)+c_{t}, $$ 

 $$ s_{t}=\mathcal{H}(u_{t}-V_{\mathrm{th}})=\begin{cases}1,&\mathrm{if}u_{t}\geq V_{\mathrm{th}}\\ 0,&\mathrm{otherwise}\end{cases}, $$ 

where  $ \beta\triangleq1-\frac{1}{\tau}\in(0,1) $ , and the discrete timesteps  $ t=1,2,\ldots,T $ , the initial situation  $ s_{0}=u_{0}=0 $ . After firing, the membrane potential resets according to previous spike  $ s_{t-1} $ . This spiking neuron face two primary challenges for long sequence tasks: (i) The coupled reset prevents parallel training along timesteps, causing training time significantly for long sequences. (ii) The commonly used LIF neuron model struggles to capture long-range dependencies, hindering the performance on long sequence tasks. The overview as shown in Figure 1.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_232_1011_731_1155.jpg" alt="Image" width="40%" /></div>


<div style="text-align: center;">(b) Decoupled Reset to Parallel</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_751_1027_987_1155.jpg" alt="Image" width="19%" /></div>


<div style="text-align: center;">(c) Problem2: Fast Decay</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_234_1173_492_1315.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">(d) Problem2: Slow Decay</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_496_1172_746_1314.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">(e) Resonate to Distinguish</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_751_1179_1006_1310.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">Figure 1: (a) The reset mechanism prevents parallel computation of timesteps as the state update relies on previous spike output, causing  $ O(L^{2}) $  timing cost. (b) The proposed decoupled reset mechanism enables parallel computation. (c) Fast decay causes long-range dependencies to vanish in the membrane potential. (d) Slow decay could generate dependency over long sequence, but causes dependency ambiguity. (e) The resonate mechanism with membrane potential helps distinguish relevant inputs.</div>


### 3.2 PROBLEM 1: COUPLED RESET PREVENTS PARALLELISM ALONG TIMESTEPS

The first problem is that the training cost of LIF-based sequential computation increases exponentially as the number of timesteps extends, with a complexity of  $ O(L^{2}) $ . This occurs because BPTT method requires the computational graph to expand along the time dimension (Kag & Saligrama, 2021). While the sequential computation of linear combination can be parallelized to reduce the time cost complexity from  $ O(L^{2}) $  to  $ O(L \log L) $  during training, as done in SSMs. However, the  $ u_{t} $  dependency on previous spikes output  $ s_{t-1} $  with nonlinear Heaviside function. By recursively expanding  $ u_{t} $  and simplifying in Equation 4:

 $$ u_{t}=c_{t}+\beta c_{t-1}+\ldots+\beta^{t-1}c_{1}-\beta V_{\mathrm{t h}}(s_{t-1}+s_{t-2}+\ldots+s_{1}). $$ 

The reset mechanism causes  $ u_{t} $  to depend on all previous spike outputs. This coupling hinders the neuron's ability to perform parallel computations, further limiting effective parallel computation for both  $ u_{t} $  and  $ s_{t} $ , as illustrated in Figure 1 (a). This forces the spiking neuron to compute sequentially. Consequently, as the number of timesteps increases during training, the time cost grows quadratically. To address this issue, we propose the decoupled reset method, which facilitates parallel computation of spiking neurons with the reset mechanism (Illustrated in Figure 1 (b) and discussed in detail in Method 4.1).

### 3.3 PROBLEM 2: COMMONLY USED LIF STRUGGLE WITH LONG-RANGE DEPENDENCIES

The second problem is that the commonly used LIF model struggles to capture and distinguish the dependencies in the membrane potential over long interval. This limitation arises due to the dilemma of decay factor  $ \beta $  in the membrane potential dynamics. When  $ \beta $  is small, the membrane potential decreases rapidly. This Fast Decay cause the neuron to quickly forget past inputs (Figure 1(c)). For example, in Case 1, if there are two inputs,  $ c_{t_{0}} $  and  $ c_{T} $ , separated by T time steps, result in  $ u_{T} $  becoming independent of  $ c_{t_{0}} $ . Similarly, in Case 2, with zero input  $ c_{t_{0}} $ , the result is identical with Case 1. This failure to retain long-term information leads to the vanishing of dependencies. Conversely, when  $ \beta $  is large, the membrane potential retains its value over an extended period T (Figure 1(d)). This Slow Decay can solve the dependency vanishing problem, but causing ambiguity between closely spaced inputs. For instance, input  $ c_{t_{0}} = c_{t_{0}'} $  with  $ t_{0}' = t_{0} + \delta t $  result in similar membrane potentials, there is small difference between the membrane potential  $ u_{T} = \beta^{(T)} \cdot c_{t_{0}} + c_{T} $  in Case 1 and the  $ u_{T} = \beta^{(T-\delta t)} \cdot c_{t_{0}'} + c_{T} $  in Case 2, as shown in Figure 1(d). To address this challenge, we propose enhancing the neuron's dynamics by incorporating resonate mechanism. This allows neurons to remain sensitive to input, even after a long interval of T. (Illustrated in Figure 1(e) and discussed in detail in Method 4.2).

## 4 METHOD

In this section, we present our approach to address two primary challenges in SNNs: parallel training and long-range dependency learning. For parallel training, we decouple the reset mechanism from the integrate computation to implement parallel training while maintaining equivalence with sequential computation. For long-range dependency learning, we propose the Parallel Resonate and Fire (PRF) Neuron by incorporating the reset process as an imaginary part into the time constant, enabling the neuron to achieve long-range learning ability.

### 4.1 DECOUPLED RESET FOR PARALLEL COMPUTATION

To enable parallel computation, we need to decouple the causal relationship with previous spikes, by separating the linear combination part from the nonlinear causal dependency part. To achieve this, we first substitute the Equation 4 into the Equation 3 by expanding the  $ u_{t} $  in the Heaviside function. Then we merge the reset part into the threshold  $ V_{th} $ . As such, Equation 3 is rewritten as Equation 5:

 $$ s_{t}=\mathcal{H}(\underbrace{c_{t}+\beta c_{t-1}+\ldots+\beta^{t-1}c_{1}}_{Leaky and Integrate\quad\triangleq\quad u_{t}^{\prime}}-\underbrace{V_{\mathrm{th}}\cdot(\beta\cdot(s_{t-1}+s_{t-2}+\ldots+s_{1})+1)}_{decoupled reset\quad\triangleq\quad d_{t}}), $$ 

where the linear combination of leaky and integrate part is defined as  $ u_{t}^{\prime} $ , the second part is defined as the decoupled reset  $ d_{t} $ . Now the spike output is:

 $$ s_{t}=\mathcal{H}\left(u_{t}^{\prime}-d_{t}\right)=\left\{\begin{aligned}&1,&&if u_{t}^{\prime}\geq d_{t}\\ &0,&&otherwise\end{aligned}\right., $$ 

where this spike output  $ s_{t} $  depends on  $ u_{t}^{\prime} $  and  $ d_{t} $ . Note that the first part is linear combination of  $ v_{t}^{\prime} = \beta u_{t-1}^{\prime} + c_{t} $ , we define the sequence of  $ U_{T}^{\prime} $  as the ordered set  $ \{u_{1}^{\prime}, u_{2}^{\prime}, \ldots, u_{T}^{\prime}\} $  over all timesteps (detailed notation is described in Appendix A). This linear combination of  $ u_{t}^{\prime} $  can be computed using convolution and further accelerated by converting the convolution into multiplication after applying the Fast Fourier Transform:  $ \mathbf{U}_{T}^{\prime} = \mathbf{C}_{T} * \mathbf{K}_{T} = \mathcal{F}^{-1}(\mathcal{F}(\mathbf{C}_{T}) \odot \mathcal{F}(\mathbf{K}_{T})) $ , with the  $ O(L \log L) $  computation complexity, avoiding the  $ O(L^{2}) $  training time in BPTT.

After efficient parallel computation of  $ u_{t}^{\prime} $ , the  $ d_{t} $  still remains the dependency relationship with previous spikes  $ s_{t} $ . To further decouple this dependency from the spike output  $ s_{t} $ , the key idea is converting the recursive form to the iterative form. We first define the dependency part as  $ A_{t} $ , as shown in Equation 7. Now, we only need to convert the  $ A_{t} $  from its recursive form into an iterative form. By separating the last spike from all previous spikes, we obtain Equation 8. The left part of this equation corresponds to Equation 6, while the right part refers to the recursive dependency itself, leading to Equation 9 and Equation 10:

 $$ A_{t}\triangleq_{t-1}s_{t-1}+s_{t-2}+\ldots+s_{1} $$ 

 $$ =s_{t-1}+\left(s_{t-2}+\ldots+s_{1}\right) $$ 

 $$ \begin{aligned}=&\mathcal{H}\left(u_{t-1}^{\prime}-d_{t-1}\right)+A_{t-1}\end{aligned} $$ 

 $$ \begin{aligned}=\begin{cases}1+A_{t-1},&if u_{t-1}^{\prime}\geq d_{t-1}\\A_{t-1},&if u_{t-1}^{\prime}<d_{t-1}\end{cases},\end{aligned} $$ 

where the sequence of  $ d_{t} $  is calculated according to the sequence of  $ u_{t}^{\prime} $ , by dynamically updating  $ A_{t} $ :

 $$ d_{t}=V_{\mathrm{th}}\cdot(\beta A_{t}+1),\quad A_{0}=u_{0}^{\prime}=0. $$ 

At this point, all recursive forms have been converted into dynamic equations. The formation of  $ d_{t} $  is completely independent of  $ s_{t} $ . The decoupled reset function  $ \mathbf{D}_{T}=f_{D}(\mathbf{U}_{T}^{\prime}) $  is deduced as shown in Equation 10 and 11, where  $ d_{t} $  can be dynamically scanned from all  $ u_{t}^{\prime} $  with  $ O(L) $  complexity.

In summary, we convert the all calculation of  $ s_{t} $  with  $ O(L^{2}) $  complexity into a combination of  $ u_{t}^{\prime} $  with  $ O(L \log L) $  complexity and subsequently  $ d_{t} $  with  $ O(L) $  complexity, achieving a training speed-up of approximately  $ L/(\log L + 1) $ . To summarize, this approach facilitates parallel computation:

 $$ \mathbf{U}_{T}^{\prime}=\mathbf{C}_{T}*\mathbf{K}_{T} $$ 

(12a)

 $$ \mathbf{U}_{T}^{\prime}=(u_{1}^{\prime},u_{2}^{\prime},\cdots,u_{T}^{\prime}) $$ 

 $$ =\mathcal{F}^{-1}\left(\mathcal{F}(\mathbf{C}_{T})\odot\mathcal{F}(\mathbf{K}_{T})\right) $$ 

(12b)

 $$ \mathbf{C}_{T}=(c_{1},c_{2},\cdots,c_{T}) $$ 

 $$ \mathbf{D}_{T}=f_{D}(\mathbf{U}_{T}^{\prime}) $$ 

(12c)

 $$ \mathbf{K}_{T}=(\boldsymbol{\beta}^{0},\boldsymbol{\beta}^{1},\ldots,\boldsymbol{\beta}^{T-1}) $$ 

 $$ \mathbf{S}_{T}=\mathcal{H}(\mathbf{U}_{T}^{\prime}-\mathbf{D}_{T}) $$ 

(12d)

 $$ \mathbf{D}_{T}=(d_{1},d_{2},\cdots,d_{T}) $$ 

Where the outputs  $ \mathbf{S}_{T} = (s_{1}, s_{2}, \ldots, s_{T}) $  from Equation 12d is equivalent with the sequential generated from Equation 3. The kernel vector  $ K_{T} $  with  $ \beta = 1 - \frac{1}{\tau} $ . By combining the above equations, we obtain the parallel computation process, as shown in Algorithm.1 and 2 in Appendix.B. Although parallelized LIF solves the training problem for long sequences (Experiment 5.1), it still does not perform well on long sequences (Experiment 5.2), due to the common issue of long-range dependencies in LIF models (Problem 3.3).

### 4.2 RESONATE FOR LONG-RANGE LEARNING ABILITY

To address the challenge of capturing long-range dependencies in spiking neural networks, we introduce the Parallel Resonate-and-Fire (PRF) neuron. This neuron model extends the standard LIF neuron by incorporating a resonance mechanism into its dynamics, enabling it to retain information over longer periods while maintaining computational efficiency.

Firstly, recall the commonly used LIF model from Equation 1, the dynamics are given by:  $ \frac{du(t)}{dt} = \gamma u(t) - \gamma u_{\text{reset}} + c(t) $ , where  $ \gamma \triangleq -1/\tau $ . To introduce more dynamic behavior, we use  $ r(t) $  and  $ \theta $ .

to replace  $ u_{reset} $  and  $ \gamma $  in the reset part, in respectively. Then the dynamic are rewritten as:

 $$ \frac{d u(t)}{d t}=\gamma u(t)-\theta r(t)+c(t). $$ 

To introduce membrane potential oscillations, we define a complex decay constant $\tilde{\gamma} = \gamma + i\theta$, where $i = \sqrt{-1}$. In the vanilla LIF model, the reset $r(t)$ is a non-continuous conditional function with $\theta = \gamma$ in Equation 14, and the reset result is instantaneous process:

 $$ \frac{d r(t)}{d t}=\begin{cases}V_{\mathrm{t h}}\delta(t),&\mathrm{i f}u(t)\geq V_{\mathrm{t h}}\\ 0,&\mathrm{i f}u(t)<V_{\mathrm{t h}}\end{cases}. $$ 

While this reset has proven effective, it poses challenges for efficient computation. We introduce a reset function that is both effective and computationally simple for both forward and backward propagation. We define the reset function to satisfy the condition:  $ \tilde{u}(t) \triangleq u(t) + ir(t) $ . Substituting this definition into Equation 14, we obtain the PRF model:

 $$ \frac{d\tilde{u}(t)}{d t}=\tilde{\gamma}\tilde{u}(t)+c(t), $$ 

where the real part  $ \Re\{\tilde{u}(t)\} $  corresponds to the membrane potential dynamics. This formulation incorporates the reset mechanism into the time constant via the imaginary component  $ \theta $ , introducing resonance into the neuron's behavior. When  $ \theta = 0 $ , the model reduces to the standard LIF neuron without reset. If we extend the Equation 16, we can be expressed in matrix form:

 $$ \begin{pmatrix}\dot{u}(t)\\ \dot{r}(t)\end{pmatrix}=\begin{pmatrix}\gamma&-\theta\\ \theta&\gamma\end{pmatrix}\begin{pmatrix}u(t)\\ r(t)\end{pmatrix}+\begin{pmatrix}c(t)\\ 0\end{pmatrix}, $$ 

which give us the insight that this reset process is continuous function:

 $$ \frac{d\boldsymbol{r}(t)}{d t}=\gamma\boldsymbol{u}(t)+\theta\boldsymbol{r}(t), $$ 

where this formulation treats the reset as a continuous function controlled by  $ \theta $ , allowing for a more gradual and reset process.

We then discretize the model (Equation 16) (details are provided in Appendix C), yielding the sequential and parallel formulations of the PRF neuron, as shown in Equation 19 and 20 in respectively:

 $$ \tilde{u}_{t}=\exp(\Delta\tilde{\gamma})\tilde{u}_{t-1}+\Delta c_{t}\qquad(19a) $$ 

 $$ s_{t}=\mathcal{H}\left(\Re\{\tilde{u}_{t}\}-V_{\mathrm{t h}}\right) $$ 

(19b)

 $$ \mathbf{\tilde{U}}_{T}=\mathcal{F}^{-1}\left(\mathcal{F}\left(\mathbf{C}_{T}\right)\odot\mathcal{F}(\mathbf{\tilde{K}}_{T})\right) $$ 

 $$ \tilde{\gamma}=\gamma+i\theta $$ 

(19c)

 $$ \mathbf{S}_{T}=\mathcal{H}\left(\Re\{\tilde{\mathbf{U}}_{T}\}-V_{\mathrm{t h}}\right) $$ 

Now the membrane potential  $ \widetilde{u}_{t}\inC $ , where  $ \Delta $  is the time step size. The details of parallel computation is described in Algorithm.3. Where  $ \mathbf{C}_{T}=(c_{1},c_{2},\ldots,c_{T}) $  is the input current sequence, and the kernel  $ \tilde{K}_{T} $  is calculated by:

 $$ \mathbf{\tilde{K}}_{T}=\left(\Delta A^{(0)},\Delta A^{(1)},\ldots,\Delta A^{(T-1)}\right), $$ 

with  $ A = \exp(\Delta\tilde{\gamma}) $ . By incorporating the imaginary component  $ \theta $ , the neuron exhibits oscillatory behavior, allowing it to resonate at specific input frequencies (see Appendix D for details). This oscillation phenomenon can be found in biological neurons (Izhikevich, 2001). Our model differs from variants of resonant models, like BHRF (Higuchi et al., 2024), which combine adaptive thresholds and refractory mechanisms. Instead, we incorporate the reset directly into the time constant through the imaginary component while enabling efficient parallel computation for training. Furthermore, the PRF neuron can also be deployed on neuromorphic chips for inference after parallel training, requiring only two additional multiplications and one more addition than LIF (see details in Appendix E).

### 4.3 THEORY ANALYSIS

(1) Sequential Perspective on Dynamic. Decoupling the reset can be viewed as transforming the reduction in membrane potential caused by increased threshold value. This means that the soft reset mechanism could be equivalent to the adaptive threshold mechanism (Bellec et al., 2020) as deduced in the Theorem 1. Furthermore, the LIF model without a reset can be considered a specific instance of a PRF, as demonstrated in Theorem 2. Moreover, the  $ u_{t} $  in PRF will converge as shown in Theorem 3. This indicates that the membrane potential of PRF is stable and bounded.

Theorem 1. Let  $ V_{th} = 1 $  and  $ \rho = 1 $  in Adaptive-LIF model, then the LIF neuron with soft reset model is equivalent to the Adaptive-LIF without reset mechanism. (The proof See Appendix.F.1)

Theorem 2. Let  $ \Delta = 1 $  and  $ \theta = 0 $ , then the PRF model could degenerate as Valina LIF model without reset mechanism. (The proof see Appendix.F.2)

Theorem 3. If the inputs  $ c_{t} \sim \mathcal{N}(0, \sigma^{2}) $  follow a normal distribution, then the membrane potential of PRF will converge as a distribution  $ u_{t} \sim \mathcal{N}(0, \frac{\tau \Delta}{2} \sigma^{2}) $ , as  $ t \to +\infty $ . (The proof see Appendix.F.3)

(2) Parallel Perspective on Gradients. The issue of membrane potential dependence described in Problem 3.3 is fundamentally a problem of gradients, which are correlated with the kernel and previous spikes. Assuming the input current is  $ C_{T}^{l} = W^{l}S_{T}^{l-1} $  across all T, the gradients are proportional to:

 $$ \nabla_{\boldsymbol{W}^{l}}\mathcal{L}\propto\underbrace{\frac{\partial\mathcal{L}}{\partial u_{T}^{l}}\sum_{t=1}^{T}\frac{\partial u_{T}^{l}}{\partial u_{t}^{l}}\frac{\partial u_{t}^{l}}{\partial\boldsymbol{W}^{l}}}_{Sequential Perspective}\propto\underbrace{\sum_{t=1}^{T}\beta^{(t)}s_{t}^{l-1}=\left\langle\mathbf{K}_{T},\mathbf{S}_{T:1}^{l-1}\right\rangle}_{Parallel Perspective}, $$ 

here  $ \langle\cdot,\cdot\rangle $  denotes the inner product,  $ \mathbf{S}_{T}^{l-1}=(s_{1}^{l-1},s_{2}^{l-1},\ldots,s_{T}^{l-1}) $  is the sequence of spike outputs from the previous layer l-1, and  $ K_{T} $  represents the parallelism kernel. The subscript  $ (T:1) $  indicates the sequence reversal. A small  $ \beta $  causes gradient vanishing due to a narrow receptive field, making it likely for sparse spikes to fall outside this field and yield an inner product close to zero. Conversely, a large  $ \beta $  results in a long-range, slow-decaying kernel, leading to overly consistent gradient values across spike positions. However, an oscillating kernel with a large  $ \beta $  can adjust the gradient at different spike positions, smoothing the gradients and improving the neuron's representational capability (see Appendix G for details). Experiment 5.2 verifies this insight.

### 4.4 ARCHITECTURE

The Spike-Driven Token and Channel Mixer (SD-TCM) Module (Figure 2, with further details in Appendix H) is inspired by token and channel mixing in Transformer and S4 architectures. For token mixing, we use a PRN Neuron followed by a Linear layer, while for channel mixing, the Neuron is replaced by a Spatial Neuron. The Spatial Neuron, a variant of the LIF Neuron, focuses on instantaneous information at each timestep by setting the time constant  $ \tau $  close to 1:  $ \lim_{\tau \to 1^{+}} u_{t} = \left(1 - \frac{1}{\tau}\right) u_{t-1} + c_{t} \approx c_{t} $ . This simplifies the output to  $ s_{t} = \mathcal{H}(c_{t} - V_{\mathrm{th}}) $ , resembling motor neurons in biology with rapid decay.

During training, a trainable amplitude  $ \alpha $  acts as a gate, and during inference,  $ \alpha $  can be merged into the Linear layer:  $ (\alpha s_{t}) \times \mathbf{W} \equiv s_{t} \times (\alpha \mathbf{W}) $ . Membrane shortcut residual connections are used to maintain event-driven, spike-based communication. As shown in Table 1, the PRF requires only  $ O(5D) $  computational complexity, while SSMs and Spikinglized SSMs require  $ O(H^{2} + 2DH) $ . This makes the architecture rely only on FP-AC and element-wise multiplication, reducing energy consumption and simplifying deployment on neuromorphic chips. As a result, it achieves lower computational complexity and energy consumption (see analysis in Appendix I).

<div style="text-align: center;"><img src="imgs/img_in_image_box_211_1186_486_1394.jpg" alt="Image" width="22%" /></div>


<div style="text-align: center;">Figure 2: Diagram of the SD-TCM.</div>


<div style="text-align: center;">Table 1: The comparison for inference complexity. Full table in Appendix.I (H: hidden dimension, D: model dimension, H denotes Heaviside function, fr: firing rate  $ \in(0,1) $ ).</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Token Mixing</td><td style='text-align: center;'>Dynamic Equation</td><td style='text-align: center;'>Infer. Complexity</td></tr><tr><td style='text-align: center;'>SSMs</td><td style='text-align: center;'>u_{t}=A^{H\times H}u_{t-1}+B^{D\times H}x_{t}y_{t}=C^{H\times D}u_{t}</td><td style='text-align: center;'>O(H^{2}+2DH)</td></tr><tr><td style='text-align: center;'>Spikingized SSMs</td><td style='text-align: center;'>u_{t}=A^{H\times H}u_{t-1}+B^{D\times H}x_{t}y_{t}=\mathcal{H}(C^{H\times D}u_{t}-V_{\mathrm{th}})</td><td style='text-align: center;'>O(H^{2}+2DH)</td></tr><tr><td style='text-align: center;'>PRF + Linear</td><td style='text-align: center;'>u_{t}=a^{D}\odot u_{t-1}+b^{D}\odot x_{t}s_{t}=\mathcal{H}(\Re\{u_{t}\}-V_{\mathrm{th}})y_{t}=\text{Linear}(s_{t})</td><td style='text-align: center;'>O(5D+fr\cdot D^{2})</td></tr></table>

## 5 EXPERIMENTS

To evaluate the efficiency and effectiveness of the parallel method, as well as the performance improvement on long sequence tasks, we first demonstrate that parallel significantly accelerates training while maintaining equivalence with sequential computation. Next, we explore the PRF neuron's ability to handle long-range dependencies, showing that kernel oscillations improve both performance and gradient stability. Finally, the SD-TCM module achieves performance comparable to S4 while reducing energy consumption by over 98.57% on Long Range Arena tasks. Detailed experimental setups and training hyperparameters are provided in Appendix J.

### 5.1 THE PARALLEL PROCESS

First, we compare the training runtime across different timesteps in Figure 3 (left) using three repeated experiments. Beyond 4 timesteps, parallel training consistently outperforms sequential training. For example, at 1,024 timesteps, sequential training takes 4.6 seconds per iteration, while parallel training takes only 0.7 seconds, achieving a speedup of  $ 6.57\times $ .

<div style="text-align: center;"><img src="imgs/img_in_chart_box_475_435_731_626.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_742_435_1000_625.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;">Figure 3: (left) Comparison of training runtime for LIF and Parallelized LIF models. (right) Comparison of sequential and parallel training runtime across different categories per batch.</div>


This acceleration becomes more significant as the number of timesteps increases, with a speedup of  $ 9.35 \times $  at 16,384 timesteps and  $ 16.50 \times $  at 32,768 timesteps. The equivalence between sequential and parallel computation is maintained during inference and training, with only a minor accuracy difference during training (details in Appendix K), which we tentatively attribute to numerical error.

The speedup is primarily due to parallel training avoiding the recursive unfolding of the computational graph. The forward and backward passes are accelerated by  $ 1.77\times $  and  $ 132.55\times $ , respectively, as shown in Figure 3 (right). The significant speedup in the backward pass occurs because sequential training requires unfolding the graph at each timestep, which is time-consuming, whereas parallel training computes the graph only once. Further details of the timing for both training and inference can be found in Appendix L.

### 5.2 THE LONG RANGE LEARNING ABILITY

Although parallelized LIF can speed up training, it still struggles with performance on simple long-sequence tasks, such as sequential MNIST, as shown in Figure 4 (left). However, after introducing oscillations in the kernel, the PRF neuron successfully solves the sequential-MNIST problem, achieving both training efficiency and effectiveness.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_475_1032_733_1224.jpg" alt="Image" width="21%" /></div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_744_1033_1000_1224.jpg" alt="Image" width="20%" /></div>


<div style="text-align: center;">Figure 4: (left) Ablation Study on sMNIST datasets (Par. means parallel training). (right) Accuracy across different decay on psMNIST.</div>


To better understand how oscillating membrane potential improves performance, we compare the performance of various  $ \beta $  values, with and without oscillations, on the more challenging permuted-sMNIST dataset. The results after training for 50 epochs are summarized in Figure 4 (right). (We fit  $ \Delta=1 $  and set  $ \theta=\frac{\pi}{2} $  while varying the  $ \beta $  hyperparameter without training.) Without the oscillating term, as  $ \beta $  increases, accuracy initially improves as expected but then decreases beyond a certain point. In contrast, the introduction of oscillations helps counteract this decline and further enhances performance, indicating that oscillations can improve performance for long sequences.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_221_171_408_308.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(a) Fast Decay</div>


<div style="text-align: center;">Gradients Vanishing</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_418_172_605_309.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(b) Moderate Decay</div>


<div style="text-align: center;">Gradients Appearing</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_613_172_801_309.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(c) Slow Decay</div>


<div style="text-align: center;">Gradients Ambiguity</div>


<div style="text-align: center;"><img src="imgs/img_in_chart_box_810_172_997_309.jpg" alt="Image" width="15%" /></div>


<div style="text-align: center;">(d) Resonate</div>


<div style="text-align: center;">Smooth Gradients</div>


<div style="text-align: center;">Figure 5: Comparison of loss landscapes: (a) Fast decay causes gradient vanishing. (b) Moderate decay improves but keeps optimal local. (c) Further slow decay makes gradient ambiguity, hindering optimization. (d) The resonance creates a smooth gradient field, aiding efficient convergence.</div>


We further examine the loss landscape contours (Li et al., 2018) for each case in Figure 5, corresponding to Figure 4 (right). Gradients are perpendicular to the contour lines, with sparse or dense contours indicating smaller or larger gradients, respectively. Extremely sparse or dense contours suggest vanishing or exploding gradients. In Figure 5 (a), fast decay with a small  $ \beta $  shows gradient vanishing, with extremely sparse contours. Increasing  $ \beta $  with moderate decay can alleviate gradient vanishing, but the model still encounters a local optima region (Figure 5 (b)). Further increasing  $ \beta $  with slow decay is expected to fully address the vanishing problem but introduces gradient ambiguity with overlapping contours (Figure 5 (c)). In contrast, the introduced oscillation from resonate results in smoother gradients (Figure 5 (d)), enabling more effective feature extraction.

As shown in Table 2, the PRF neuron uses only 68.9k parameters (768 for neurons and 67.2k for synapses) to achieve state-of-the-art (SOTA) results while maintaining training efficiency. Previous models required recurrent connections in the linear layers to solve the  $ (p) $ s-MNIST task. To align the training parameters, we modified the architecture to a feedforward linear layer with a  $ 1-(128)^{3}-10 $  structure. Using parallel computation, we completed the entire training process in only 1.60 hours with a batch size of 256 for 200 epochs.

<div style="text-align: center;">Table 2: The neuron for (permute)-sequential MNIST and sequential CIFAR task. The pixel-level image classification captures the hierarchical structure to verify the long-range capture ability. The symbol  $ * $  denotes including the feedback linear connection, while the symbol of  $ \uparrow $  and  $ \downarrow $  indicate that larger and smaller values are better, respectively. The BHRF result is referenced from (Higuchi et al., 2024), results of PSN family and PMSN are from (Chen et al., 2024), while other results are from (Zhang et al., 2024).</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Task (Length)</td><td style='text-align: center;'>Spiking Neuron</td><td style='text-align: center;'>Seq. Infer.</td><td style='text-align: center;'>Par. Train.</td><td style='text-align: center;'>No. Params.  $ \downarrow $</td><td style='text-align: center;'>Top-1 Test Acc. (%)  $ \uparrow $</td></tr><tr><td rowspan="12">sMNIST / psMNIST (784)</td><td style='text-align: center;'>LIF</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>155.1*</td><td style='text-align: center;'>89.28 / 80.26</td></tr><tr><td style='text-align: center;'>PLIF (Fang et al., 2021) $ ^{2021} $  ICCV</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>155.1 k*</td><td style='text-align: center;'>91.79 / -</td></tr><tr><td style='text-align: center;'>GLIF (Yao et al., 2022) $ ^{2022} $  NeurIPS</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>157.5 k*</td><td style='text-align: center;'>96.64 / 90.47</td></tr><tr><td style='text-align: center;'>TC-LIF (Zhang et al., 2024) $ ^{2024} $  AAAI</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>155.1 k*</td><td style='text-align: center;'>99.20 / 95.36</td></tr><tr><td style='text-align: center;'>ALIF (Yin et al., 2021) $ ^{2021} $  Nat. MI</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>156.3 k*</td><td style='text-align: center;'>98.70 / 94.30</td></tr><tr><td style='text-align: center;'>BHRF (Higuchi et al., 2024) $ ^{2024} $  ICML</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>68.9 k</td><td style='text-align: center;'>99.10 / 95.2</td></tr><tr><td style='text-align: center;'>PRF (Ours)</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>68.9 k</td><td style='text-align: center;'>99.18 / 96.87</td></tr><tr><td style='text-align: center;'>PSN (Fang et al., 2024) $ ^{2024} $  NeurIPS</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>2.5 M</td><td style='text-align: center;'>97.90 / 97.76</td></tr><tr><td style='text-align: center;'>Masked PSN (Fang et al., 2024) $ ^{2024} $  NeurIPS</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>153.7 k</td><td style='text-align: center;'>97.76 / 97.53</td></tr><tr><td style='text-align: center;'>Sliding PSN (Fang et al., 2024) $ ^{2024} $  NeurIPS</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>52.4 k</td><td style='text-align: center;'>97.20 / 82.84</td></tr><tr><td style='text-align: center;'>PMSN (Chen et al., 2024) $ ^{2024} $  ArXiv</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>156.4 k</td><td style='text-align: center;'>99.53 / 97.78</td></tr><tr><td style='text-align: center;'>PRF (Ours)</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>167.0 k</td><td style='text-align: center;'>99.39 / 97.90</td></tr><tr><td rowspan="7">seqCIFAR (1024)</td><td style='text-align: center;'>LIF</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>0.18 M</td><td style='text-align: center;'>45.07</td></tr><tr><td style='text-align: center;'>PSN (Fang et al., 2024) $ ^{2024} $  NeurIPS</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>6.47 M</td><td style='text-align: center;'>55.24</td></tr><tr><td style='text-align: center;'>Masked PSN (Fang et al., 2024) $ ^{2024} $  NeurIPS</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>0.38 M</td><td style='text-align: center;'>57.83</td></tr><tr><td style='text-align: center;'>Sliding PSN (Fang et al., 2024) $ ^{2024} $  NeurIPS</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>0.18 M</td><td style='text-align: center;'>70.23</td></tr><tr><td style='text-align: center;'>PMSN (Chen et al., 2024) $ ^{2024} $  ArXiv</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>0.21 M</td><td style='text-align: center;'>82.14</td></tr><tr><td style='text-align: center;'>PRF (Ours)</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>0.29 M</td><td style='text-align: center;'>82.37</td></tr><tr><td style='text-align: center;'>PRF (Ours)</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>1.10 M</td><td style='text-align: center;'>85.33</td></tr></table>

### 5.3 LONG RANGE ARENA TASKS

To demonstrate the long-range dependency analysis capability of our SD-TCM module, we evaluate it using the Long Range Arena (LRA) benchmark (Tay et al., 2020). This benchmark covers a wide range of classification tasks, including both textual and image domains. For the ListOps, Text, and Retrieval tasks, we use the causal architecture, while S4 employs a bidirectional architecture for all tasks. For the Image and Pathfinder tasks, we use a bidirectional architecture.

<div style="text-align: center;">Table 3: Comparison of Accuracy, Parameters and Energy. Table 4: Text (4096) ablation on  $ \alpha $  effective with casual architecture.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Metric</td><td style='text-align: center;'>Model</td><td style='text-align: center;'>ListOps</td><td style='text-align: center;'>Text</td><td style='text-align: center;'>Retrieval</td><td style='text-align: center;'>Image</td><td style='text-align: center;'>Pathfinder</td><td style='text-align: center;'>Avg.</td></tr><tr><td rowspan="2">En.(mJ)</td><td style='text-align: center;'>S4</td><td style='text-align: center;'>5.104</td><td style='text-align: center;'>3.718</td><td style='text-align: center;'>24.439</td><td style='text-align: center;'>19.222</td><td style='text-align: center;'>6.256</td><td style='text-align: center;'>11.748</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>0.075</td><td style='text-align: center;'>0.298</td><td style='text-align: center;'>0.211</td><td style='text-align: center;'>0.187</td><td style='text-align: center;'>0.067</td><td style='text-align: center;'>0.168</td></tr><tr><td rowspan="2">Acc.(%)</td><td style='text-align: center;'>S4</td><td style='text-align: center;'>59.60</td><td style='text-align: center;'>86.82</td><td style='text-align: center;'>90.90</td><td style='text-align: center;'>88.65</td><td style='text-align: center;'>94.20</td><td style='text-align: center;'>84.03</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>59.20</td><td style='text-align: center;'>86.33</td><td style='text-align: center;'>89.88</td><td style='text-align: center;'>84.77</td><td style='text-align: center;'>91.76</td><td style='text-align: center;'>82.39</td></tr><tr><td rowspan="2">Par.</td><td style='text-align: center;'>S4</td><td style='text-align: center;'>815 k</td><td style='text-align: center;'>843 k</td><td style='text-align: center;'>3.6 M</td><td style='text-align: center;'>3.6 M</td><td style='text-align: center;'>1.3 M</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>Ours</td><td style='text-align: center;'>272 k</td><td style='text-align: center;'>830 k</td><td style='text-align: center;'>1.1 M</td><td style='text-align: center;'>4.1 M</td><td style='text-align: center;'>1.3 M</td><td style='text-align: center;'>-</td></tr></table>


<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Text (4096)</td><td style='text-align: center;'>Acc. (%)</td><td style='text-align: center;'>Diff. (%)</td></tr><tr><td style='text-align: center;'>without  $ \alpha $</td><td style='text-align: center;'>85.75</td><td style='text-align: center;'>-</td></tr><tr><td style='text-align: center;'>+  $ \alpha $  on  $ SN $  &amp;  $ TN $</td><td style='text-align: center;'>85.69</td><td style='text-align: center;'>- 0.06</td></tr><tr><td style='text-align: center;'>+  $ \alpha $  on  $ TN $</td><td style='text-align: center;'>86.11</td><td style='text-align: center;'>+ 0.36</td></tr><tr><td style='text-align: center;'>+  $ \alpha $  on  $ SN $</td><td style='text-align: center;'>86.33</td><td style='text-align: center;'>+ 0.58</td></tr></table>

The SD-TCM module achieves performance comparable to S4 while reducing energy consumption by over 98.57% on average, as shown in Table 3. Specifically, the accuracy on ListOps is 59.60%, compared to S4's 59.20%. The key advantage is the significant reduction in energy consumption, as shown in Table 3, with ListOps dropping from 5.104 mJ to 0.075 mJ. This energy reduction mainly benefits from the extreme sparsity of spikes, with detailed firing rate statistics provided in Appendix M. Additionally, the model is sensitive to the imaginary part of  $ \theta $ , causing fluctuations with different initialization (see Appendix N for details). Table 4 shows an ablation study on the  $ \alpha $  parameter in the Text (4096) task. Applying  $ \alpha $  to both the spatial (S/N) and PRF (T/N) components slightly lowers accuracy from 85.75% to 85.69%, but applying it only to S/N improves accuracy to 86.33%, demonstrating its benefit for spatial dependencies. Finally, as shown in Table 5, our module achieves performance comparable to the S4 baseline across tasks while preserving spike-driven feature, avoiding nonlinear activation functions and FP MAC operations.

<div style="text-align: center;">Table 5: Test Accuracy Comparison on LRA Tasks (\%) ( $ \uparrow $ ). 'NL Act. Free' and 'FP MAC Free' denote models that do not use nonlinear activation functions or floating-point multiply-accumulate operations in the block, respectively. The underline and bold formatting indicate the SoTA result for Spikinglized SSMs and Improving Neuron methods, respectively.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Model (Input length)</td><td style='text-align: center;'>NL Act. -Free</td><td style='text-align: center;'>FP MAC -Free</td><td style='text-align: center;'>ListOps (2,048)</td><td style='text-align: center;'>Text (4,096)</td><td style='text-align: center;'>Retrieval (4,000)</td><td style='text-align: center;'>Image (1,024)</td><td style='text-align: center;'>Pathfinder (1,024)</td><td style='text-align: center;'>Avg.</td></tr><tr><td style='text-align: center;'>Random (Lower Bound)</td><td style='text-align: center;'>-</td><td style='text-align: center;'>-</td><td style='text-align: center;'>10.00</td><td style='text-align: center;'>50.00</td><td style='text-align: center;'>50.00</td><td style='text-align: center;'>10.00</td><td style='text-align: center;'>50.00</td><td style='text-align: center;'>34.00</td></tr><tr><td style='text-align: center;'>Transformer (Vaswani et al., 2017)</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>36.37</td><td style='text-align: center;'>64.27</td><td style='text-align: center;'>57.46</td><td style='text-align: center;'>42.44</td><td style='text-align: center;'>71.40</td><td style='text-align: center;'>54.39</td></tr><tr><td style='text-align: center;'>S4 (Bidirectional) (Gu et al., 2022a)</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>59.60</td><td style='text-align: center;'>86.82</td><td style='text-align: center;'>90.90</td><td style='text-align: center;'>88.65</td><td style='text-align: center;'>94.20</td><td style='text-align: center;'>84.03</td></tr><tr><td rowspan="2">Binary 54D (Stan &amp; Rhodes, 2023) 2024 Sci. Rep. ↔ + GSU &amp; GeLU</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>54.80</td><td style='text-align: center;'>82.50</td><td style='text-align: center;'>85.03</td><td style='text-align: center;'>82.00</td><td style='text-align: center;'>82.60</td><td style='text-align: center;'>77.39</td></tr><tr><td style='text-align: center;'>✗</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>59.60</td><td style='text-align: center;'>86.50</td><td style='text-align: center;'>90.22</td><td style='text-align: center;'>85.00</td><td style='text-align: center;'>91.30</td><td style='text-align: center;'>82.52</td></tr><tr><td style='text-align: center;'>Stoch. SpikingS4 (Bal &amp; Sengupta, 2024) 2024 arXiv</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>55.70</td><td style='text-align: center;'>77.62</td><td style='text-align: center;'>88.48</td><td style='text-align: center;'>80.10</td><td style='text-align: center;'>83.41</td><td style='text-align: center;'>77.06</td></tr><tr><td style='text-align: center;'>SpikingSSMs (Shen et al., 2024) 2024 arXiv</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>60.23</td><td style='text-align: center;'>80.41</td><td style='text-align: center;'>88.77</td><td style='text-align: center;'>88.21</td><td style='text-align: center;'>93.51</td><td style='text-align: center;'>82.23</td></tr><tr><td style='text-align: center;'>Spiking LMU (Liu et al., 2024) 2024 ICLR</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>37.30</td><td style='text-align: center;'>65.80</td><td style='text-align: center;'>79.76</td><td style='text-align: center;'>55.65</td><td style='text-align: center;'>72.68</td><td style='text-align: center;'>62.23</td></tr><tr><td style='text-align: center;'>ELM Neuron (Spicler et al., 2024) 2024 ICLR</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>✗</td><td style='text-align: center;'>44.55</td><td style='text-align: center;'>75.40</td><td style='text-align: center;'>84.93</td><td style='text-align: center;'>49.62</td><td style='text-align: center;'>71.15</td><td style='text-align: center;'>69.25</td></tr><tr><td style='text-align: center;'>Spike-Driven TCM</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>✓</td><td style='text-align: center;'>59.20</td><td style='text-align: center;'>86.33</td><td style='text-align: center;'>89.88</td><td style='text-align: center;'>84.77</td><td style='text-align: center;'>91.76</td><td style='text-align: center;'>82.39</td></tr></table>

## 6 CONCLUSION

This study aims to solve the SNNs problem of parallelization and performance on long sequences. We propose the decoupled reset method, enabling spiking neurons could parallel training. This method can be applied to any type of neuron to speed up. Additionally, we introduce the PRF neuron, incorporating the reset as an imaginary part to formulate oscillations in the membrane potential, which solves the long-range dependency problem. The SD-TCM model, combined with PRF neurons, achieves performance comparable to S4 on the LRA task while reducing energy consumption by two orders of magnitude. However, due to the sensitivity of training to neuron initialization, the PathX problem remains unsolved. This issue could be solved in the future by using better hyperparameters and initialization strategies.