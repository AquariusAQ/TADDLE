# TORCH TITAN: ONE-STOP PYTORCH NATIVE SOLUTION FOR PRODUCTION READY LLM PRETRAINING

Wanchao Liang $ ^{1} $ , Tianyu Liu $ ^{1*} $ , Less Wright $ ^{1} $ , Will Constable $ ^{1} $ , Andrew Gu $ ^{1} $ 

Chien-Chin Huang $ ^{1} $ , Iris Zhang $ ^{1} $ , Wei Feng $ ^{1} $ , Howard Huang $ ^{1} $ , Junjie Wang $ ^{1} $ 

Sanket Purandare $ ^{2†} $ , Gokul Nadathur $ ^{1} $ , Stratos Idreos $ ^{2} $ 

 $ ^{1} $ Meta,  $ ^{2} $ Harvard University

## ABSTRACT

The development of large language models (LLMs) has been instrumental in advancing state-of-the-art natural language processing applications. Training LLMs with billions of parameters and trillions of tokens require sophisticated distributed systems that enable composing and comparing several state-of-the-art techniques in order to efficiently scale across thousands of accelerators. However, existing solutions are complex, scattered across multiple libraries/repositories, lack interoperability, and are cumbersome to maintain. Thus, curating and empirically comparing training recipes require non-trivial engineering effort.

This paper introduces TORCHTITAN, an open-source $ ^{1} $ , PyTorch-native distributed training system that unifies and advances state-of-the-art techniques, streamlining integration and reducing engineering overhead. TORCHTITAN enables seamless application of 4D parallelism in a modular and composable manner, while featuring elastic scaling to adapt to changing computational requirements. The system provides comprehensive logging, efficient checkpointing, and debugging tools, ensuring production-ready training. Moreover, TORCHTITAN incorporates innovative hardware-software co-designed solutions, leveraging cutting-edge features like Float8 training and SymmetricMemory to maximize hardware utilization. As a flexible experimental test bed, TORCHTITAN facilitates the curation and comparison of custom recipes for diverse training contexts. By leveraging TORCHTITAN, we developed optimized training recipes for the Llama 3.1 family and provided actionable guidance on selecting and combining distributed training techniques to maximize training efficiency, based on our hands-on experiences.

We thoroughly assess TORCHTITAN on the Llama 3.1 family of LLMs, spanning 8 billion to 405 billion parameters, and showcase its exceptional performance, modular composability, and elastic scalability. By stacking training optimizations, we demonstrate accelerations ranging from 65.08% on Llama 3.1 8B at 128 GPU scale (1D), 12.59% on Llama 3.1 70B at 256 GPU scale (2D), to 30% on Llama 3.1 405B at 512 GPU scale (3D) on NVIDIA H100 GPUs over optimized baselines. We also demonstrate the effectiveness of 4D parallelism in enabling long context training.

## 1 INTRODUCTION

Large Language Models (LLMs) (Devlin, 2018; Liu et al., 2019; Radford et al., 2019; Chowdhery et al., 2023; Anil et al., 2023; Achiam et al., 2023; Dubey et al., 2024; Jiang et al., 2024; Abdin et al., 2024) have been the driving force behind the advancement of natural language processing (NLP) applications spanning language translation, content/code generation, conversational AI, text data analysis, creative writing and art, education, and research, etc.

Achieving state-of-the-art LLM performance requires massive scale, exemplified by top-performing models like Llama 3.1 (405B parameters, 15T tokens, 30.84M GPU hours, 16K H100 GPUs) (Dubey

et al., 2024) and Google's PaLM (540B parameters, 0.8T tokens, 9.4M TPU hours, 6144 TPUv4 chips) (Chowdhery et al., 2023). These models demonstrate exceptional natural language understanding and generation capabilities, but at the same time necessitate substantial computational resources, memory, and time to train, highlighting the significant investment required to advance natural language processing.

Training large language models (LLMs) at scale is a daunting task that requires a delicate balance of parallelism, computation, and communication, all while navigating intricate memory and computation trade-offs. The massive resources required for training make it prone to GPU failures, underscoring the need for efficient recovery mechanisms and checkpointing strategies to minimize downtime (Eisenman et al., 2022; Wang et al., 2023; Gupta et al., 2024; Maurya et al., 2024; Wan et al., 2024). To optimize resource utilization and achieve elastic scalability, it is crucial to combine multiple parallelism techniques, including Data Parallel (Li et al., 2020; Rajbhandari et al., 2020; Zhang et al., 2022; Zhao et al., 2023), Tensor Parallel (Narayanan et al., 2021; Wang et al., 2022; Korthikanti et al., 2023), Context Parallel (Liu et al., 2023; Liu & Abbeel, 2024; NVIDIA, 2023; Fang & Zhao, 2024), and Pipeline Parallel (Huang et al., 2019; Narayanan et al., 2019; 2021; Qi et al., 2023). By stacking these parallelisms with memory and computation optimization techniques, such as activation recomputation (Chen et al., 2016; Korthikanti et al., 2023; He & Yu, 2023; Purandare et al., 2023), mixed precision training (Micikevicius et al., 2018; 2022), and deep learning compilers (Bradbury et al., 2018; Yu et al., 2023; Li et al., 2024; Ansel et al., 2024), it is possible to maximize hardware utilization.

While state-of-the-art distributed training techniques have significantly advanced the field, existing systems that incorporate them still fall short in addressing critical challenges that hinder their usability, adoption and effectiveness for researchers and industry practitioners.

1. Non-composable: Existing systems struggle to integrate and stack parallelism techniques, limiting multi-dimensional exploration and integration with memory and computation optimizations, thereby reducing training efficiency.

2. Inflexible Architecture: Lack of modularity and extensibility hampers the integration of new techniques, optimizations, and hardware, limiting adaptability to evolving ML landscapes.

3. Inefficient Hardware Utilization: Poor leverage of advanced hardware features results in sub-optimal GPU efficiency and lack of customizable checkpointing strategies for memory-computation trade-offs.

4. Insufficient Support for Production Training: Limited distributed checkpointing scalability, cumbersome failure recovery, and inadequate debugging tools hinder production-grade workflows.

5. Framework Limitations: Dependence on external, poorly maintained dependencies and failure to harness PyTorch's optimized kernels, new features, and compiler support lead to inefficiencies and compatibility issues.

The non-composability and inflexibility of distributed systems stem from the absence of unified tensor and device abstractions applied consistently across the stack. Without these foundational components, parallelism strategies, checkpointing, and efficiency optimizations remain fragmented, limiting modularity, scalability, and extensibility.

TORCHTITAN's primary research contribution lies in identifying and unifying the core principles of parallelism and optimization techniques into a cohesive framework. By leveraging and extending PyTorch's Distributed Tensor (DTensor) and DeviceMesh (PyTorch Community, 2023a), TORCHTITAN provides a unified abstraction that simplifies the composition of parallelism strategies, and ensures correct single device semantics with its sharding primitives. Unlike existing systems that often rely on rigid or ad-hoc designs, TORCHTITAN introduces a unified template for distributed training, enabling researchers to systematically explore configurations, rigorously evaluate existing methods, and uncover novel techniques within the design space.

TORCHTITAN represents a complete distributed training system for large language models (LLMs), rather than merely a collection of individual techniques. Its modular, extensible architecture supports seamless composition of 4D parallelism, advanced training optimizations, and scalable distributed checkpoint save/load, all while harnessing PyTorch's native capabilities. The system not only en-

able production-grade training with thousands of GPUs, but also reduces complexity and fosters innovation, setting a new standard for scalable and flexible distributed training systems.

To develop and evaluate the capabilities of TORCHTITAN, we undertook several key steps, which represent the core contributions of this work, and are summarized as follows:

1. We advance DTensor by extending its sharding to support n-D parallelism, adding compatibility with torch.compile for compiler optimizations, and enabling efficient checkpointing of n-D models via state dict support. We also resolve critical bugs to bolster DTensor's production readiness.

2. We demonstrate how to compose various parallelism techniques, facilitating the exploration of multi-dimensional parallelism in large language model training ( $ §2.1 $ ).

3. We enable novel hardware-software co-designed solutions exploiting advanced hardware features to increase GPU efficiency, offer customizable activation checkpointing strategies for navigating memory-computation trade-offs, and utilize torch.compile to further optimize memory, computation, and communication ( $ §2.2 $ ).

4. We offer production grade training by incorporating scalable and efficient distributed checkpoint to facilitate fast failure recovery, integrating debugging tools like Flight Recorder to debug crashed/stuck jobs, and provide extensive logging metrics ( $ §2.3 $ ).

5. We extensively evaluate TORCHTITAN on Llama 3.1 family of models, stacking 1D to 4D parallelisms (respectively), at the scale from 8 to 512 GPUs to demonstrate elastic scalability while ensuring efficiency, convergence, and accuracy. In summary, we demonstrate training accelerations ranging from 65.08% on Llama 3.1 8B at 128 GPU scale (1D), 12.59% on Llama3.1 70B at 256 GPU scale (2D), to 30% on Llama3.1 405B at 512 GPU scale (3D), and the effectiveness of 4D parallelism in enabling long context training, on latest NVIDIA H100 GPUs over optimized baselines ( $ \S3.2 $ ).

6. We provide systematic training recipes and guidelines that empower users to navigate the complexities of distributed training, helping them optimize training efficiency for a range of model sizes and cluster configurations ( $ §3.3 $ ).

By providing an accessible and extensible platform, TORCHTITAN democratizes large language model (LLM) pretraining, empowering a wider range of researchers and developers to tap into the potential of LLMs and accelerate innovation in the field.

## 2 ELASTICITY THROUGH COMPOSABILITY

<div style="text-align: center;"><img src="imgs/img_in_image_box_218_996_1004_1178.jpg" alt="Image" width="64%" /></div>


<div style="text-align: center;">Figure 1: Composable and Modular TORCH TITAN initialization workflow.</div>


TORCHTITAN incorporates various parallelisms in a modular manner to enable easy, user-selectable combinations of multi-dimensional shardings. This composability enables the tackling of difficult scaling challenges by enhancing the ease of exploration for optimizing training efficiencies at scale.

The codebase of TORCHTITAN is organized purposefully to enable composability and extensibility. We intentionally keep three main components separate and as orthogonal as possible: (1) the model definition, which is parallelism-agnostic and designed for readability, (2) parallelism helpers, which apply parallelisms and training optimizations to a particular model, and (3) a generalized training loop. All these components are configurable via TOML files with command-line overrides, and it is easy to add new models and parallelism techniques on top of the existing codebase.

### 2.1 COMPOSABLE N-D PARALLELISM TRAINING

In this section, we will walk through the entire regime of scaling model training on large clusters, including meta device initialization and the core composable multi-dimensional parallelisms, to showcase how these techniques can be composed to train LLMs efficiently at increasing scale in TORCHTITAN. The corresponding code snippets in TORCHTITAN can be found in Appendix A.

#### 2.1.1 LARGE-SCALE MODEL INITIALIZATION USING META DEVICE

As LLMs grow exponentially, scaling challenges arise even before training begins, particularly in instantiating large models for sharding without exceeding CPU or GPU memory limits.

To address this, TORCHTITAN enables meta device initialization, where the model is first created on a meta device that stores only metadata, making initialization ultra-fast. The model is then sharded into Distributed Tensors (DTensors), with the local shard of each parameter residing on the meta device. Finally, parameter initialization is performed using user-defined functions, ensuring correct DTensor sharding layouts and proper RNG seed usage.

#### 2.1.2 FULLY SHARDED DATA PARALLEL

The original Fully Sharded Data Parallel (FSDP) (Zhao et al., 2023) is an effective implementation of ZeRO that offers large model training capability in PyTorch. However, the original implementation (FSDP1) in PyTorch suffers from various limitations due to its FlatParameter implementation.

Given these limitations, TORCHTITAN integrates a new version of Fully Sharded Data Parallel (FSDP2), which uses the per-parameter Distributed Tensor sharding representation and thus provides better composability with model parallelism techniques and other features that require the manipulation of individual parameters.

TORCHTITAN integrates and leverages FSDP2 as it's default 1D parallelism, benefiting from the improved memory management (often 7 percent lower per GPU memory requirement vs FSDP1) and the slight performance gains (average of 1.5 percent gain vs FSDP1). More details on FSDP2 and usage example are shown in Appendix B.1. TORCHTITAN makes it simple to run with FSDP2 by embedding appropriate defaults, including auto-sharding with your world size automatically.

For scaling to even larger world sizes, TORCHTITAN also integrates Hybrid Sharded Data Parallel (HSDP) which extends FSDP2 by creating 2D DeviceMesh with replica groups. Details are shown in Appendix B.2

#### 2.1.3 TENSOR PARALLEL

Tensor Parallel (TP) (Narayanan et al., 2021), together with Sequence Parallel (SP) (Korthikanti et al., 2023), is a key model parallelism technique to enable large model training at scale.

TP is implemented in TORCHTITAN using the PyTorch's RowwiseParallel and ColwiseParallel APIs, where the model parameters are partitioned to DTensors and perform sharded computation with it (Figure 3). By leveraging DTensor, the TP implementation does not need to touch the model code, which allows faster enablement on different models and provides better composability with other features mentioned in this paper.

Tensor and Sequence Parallel (TP/SP) While TP partitions the most computationally demanding aspects, Sequence Parallel (SP) performs a sharded computation for the normalization or dropout layers on the sequence dimension, which otherwise generates large replicated activation tensors, and thus can be challenging to memory constraints per GPU. See Appendix B.3 for more details, illustrations, and usage for both TP and FSDP + TP.

Due to the synergistic relationship between TP and SP, TORCHTITAN natively bundles these two together, and they are jointly controlled by the TP degree setting.

Loss Parallel When computing the loss function, model outputs are typically large, especially with TP/SP, where they are sharded across the vocabulary dimension. Naively computing cross-entropy loss requires gathering all shards, leading to high memory usage.

Loss Parallel enables efficient loss computation without fully gathering model outputs, significantly reducing memory consumption and improving training speed by minimizing communication overhead and enabling parallel sharded computation. Due to these advantages, TORCHTITAN implements Loss Parallel by default.

#### 2.1.4 PIPELINE PARALLEL

For large-scale pretraining, TORCHTITAN employs Pipeline Parallelism (PP), which minimizes communication overhead by leveraging P2P communications. PP divides the model into S stages, each running on a separate group of devices. Typically, each stage represents a model layer or a group of adjacent layers, but can include partial layers. During the forward pass, each stage receives input activations (except stage 0), computes locally, and sends output activations (except stage S - 1). The last stage computes the loss and initiates the backward pass, sending gradients in reverse order. To improve efficiency, the input batch is split into microbatches, and the pipeline schedule overlaps computation and communication across microbatches. TORCHTITAN supports various pipeline schedules (Narayanan et al., 2019; Huang et al., 2019; Narayanan et al., 2021; Qi et al., 2023). Recently, TORCHTITAN added support for new schedules including ZeroBubble and 'Flexible-Interleaved-1F1B', making use of pipeline IR to quickly express new schedules as a list of compute actions and rely on compiler passes to insert and optimize communication actions PyTorch Team 2024d.

The PP training loop differs from standard training by creating pipeline stages and executing schedules instead of directly invoking model.forward(). Since loss is computed per microbatch, TORCHTITAN introduces a shared loss_fn to unify pipeline and non-pipeline workflows, reducing code divergence.

torch.distributed.pipelining also simplifies interactions with data parallelism, ensuring that reductions occur only after the final microbatch and handling shard/unshard operations (e.g., with ZeRO-3), as well as applying gradient scaling transparently within the pipeline schedule executor. For more details on TORCHTITAN's implementation of PP, see Appendix B.4.

#### 2.1.5 CONTEXT PARALLELISM

TORCHTITAN has been extended to incorporate Context Parallelism (CP) (Liu et al., 2023; Liu & Abbeel, 2024; NVIDIA, 2023), enabling 4D parallelism by adding CP as an additional dimension to existing DP, TP, and PP. CP scales model training by splitting the context dimension across GPUs, significantly increasing the maximum trainable context length without causing out-of-memory (OOM) errors. For example, on Llama 3.1 8B with 8 H100 GPUs, using CP enabled training at context lengths up to 262,144 tokens, achieving minor MFU degradation as CP degree increases (PyTorch Team, 2025). For more details on CP integration please refer to Appendix B.5.

### 2.2 OPTIMIZING TRAINING EFFICIENCIES

#### 2.2.1 NAVIGATING COMPUTE-MEMORY TRADE-OFFS USING ACTIVATION CHECKPOINTING

Activation checkpointing (AC) (Chen et al., 2016; He & Yu, 2023; Purandare et al., 2023) and selective activation checkpointing (SAC) (Korthikanti et al., 2023) are standard training techniques to reduce peak GPU memory usage, by trading activation recomputation during the backward pass for memory savings. It is often needed even after applying multi-dimensional parallelisms.

TORCHTITAN offers flexible AC and SAC options utilizing torch.utils.checkpoint, applied at the TransformerBlock level. The AC strategies include "full" AC, op-level SAC, and layer-level SAC.

Within a TransformerBlock, full AC works by recomputing all activation tensors needed during the backward pass, whereas op-level SAC saves the results from computation-intensive PyTorch operations and only recomputes others. Layer-level SAC works in similar fashion as full AC, but the wrapping is applied to every x TransformerBlock (where x is specified by the user) to implement configurable trade-offs between memory and recompute. (Details are in Appendix B.6.)

##### 2.2.2 REGIONAL COMPILATION TO EXPLOIT TORCH. COMPILE OPTIMIZATIONS

torch.compile was released in PyTorch 2 (Ansel et al., 2024) with TorchDynamo as the frontend to extract PyTorch operations into an FX graph, and TorchInductor as the backend to compile the FX graph into fused Triton code to improve the performance.

In TORCHTITAN, we use regional compilation, which applies torch.compile to each individual TransformerBlock in the Transformer model. This has two main benefits: (1) we get a full graph (without graph breaks) for each region, compatible with FSDP2 and TP (and more generally torch.Tensor subclasses such as DTensor) and other PyTorch distributed training techniques; (2) since the Llama model stacks identical TransformerBlock layers one after another, torch.compile can identify the same structure is being repeatedly compiled and only compile once, thus greatly reducing compilation time.

torch.compile brings efficiency in both throughput and memory (see Section 3.2) via computation fusions and computation-communication reordering, in a model-agnostic way with a simple user interface. Below we further elaborate how torch.compile composability helps TORCHTITAN unlock hardware-optimized performance gain with simple user interface, with the integration of advanced features such as Asynchronous TP and Float8.

#### 2.2.3 ASYNCHRONOUS TENSOR PARALLEL TO MAXIMALLY OVERLAP COMMUNICATION

By default, TP incurs blocking communications before/after the sharded computations, causing computation resources to not be effectively utilized. Asynchronous TP (AsyncTP) (Wang et al., 2022) achieves computation-communication overlap by fractionalizing the TP matrix multiplications within attention and feed-forward modules into smaller chunks, and overlapping communication collectives in between each section. The overlap is achieved by a micro-pipelining optimization, where results are being communicated at the same time that the other chunks of the matmul are being computed.

PyTorchAsyncTP is based on a SymmetricMemory abstraction, which creates intra-node buffers to write faster communication collectives. This is done by allocating a shared memory buffer on each GPU in order to provide direct P2P access (PyTorch Team, 2024a).

With TORCHTITAN's integration of torch.compile,AsyncTP can be easily configured in TORCHTITAN to achieve meaningful end-to-end speedups (see Section 3.2 for details) on newer hardware (H100 or newer GPUs with NVSwitch within a node). Usage details are in Appendix B.7

#### 2.2.4 BOOSTING THROUGHPUT WITH MIXED PRECISION TRAINING AND FLOAT8 SUPPORT

Mixed precision training (Micikevicius et al., 2018) provides both memory and computational savings while ensuring training stability. FSDP2 has built-in support for mixed precision training with basic torch.dtype. This covers the popular usage of performing FSDP all-gather and computation in a low precision (e.g. torch.bfloat16), and perform lossless FSDP reduce-scatter (gradient) in high precision (e.g. torch.float32) for better numerical results. See Appendix B.8 for usage details.

TORCHTITAN also supports more advanced mixed precision training with Float8, a derived data type, applied selectively to linear layers (available on newer hardware like NVIDIA H100), achieving substantial performance gains while ensuring training stability (reported in Section 3.2). The Float8 feature from torchao.float8 supports multiple per-tensor scaling strategies, including dynamic, delayed, and static (see Micikevicius et al. (2022); PyTorch Community (2023b), Section 4.3 for details), while being composable with other key PyTorch-native systems such as autograd, torch.compile, FSDP2 and TP (with Float8 all-gather capability) (PyTorch Team, 2024c).

### 2.3 PRODUCTION READY TRAINING

To enable production-grade training, TORCHTITAN offers seamless integration with key features out of the box. These include (1) efficient checkpointing using PyTorch Distributed Checkpointing (DCP), and (2) debugging stuck or crashed jobs through integration with Flight Recorder.

#### 2.3.1 Scalable and efficient Distributed Checkpointing

Checkpoints are crucial in training large language models for two reasons: they facilitate model reuse in applications like inference and evaluation, and they provide a recovery mechanism in case of failures. An optimal checkpointing workflow should ensure ease of reuse across different parallelisms and maintain high performance without slowing down training. There are two typical checkpointing methods. The first aggregates the state (model parameters and optimizer states) into an unsharded version that is parallelism-agnostic, facilitating easy reuse but requiring expensive communication. The second method has each trainer save its local sharded state, which speeds up the process but complicates reuse due to embedded parallelism information.

DCP addresses these challenges using DTensor, which encapsulates both global and local tensor information independently of parallelism. DCP converts this information into an internal format for storage. During loading, DCP matches the stored shards with the current DTensor-based model parameters and optimizer states, fetching the necessary shard from storage. TORCHTITAN effectively uses DCP to balance efficiency and usability. Furthermore, DCP enhances efficiency through asynchronous checkpointing by processing storage persistence in a separate thread, allowing this operation to overlap with subsequent training iterations. TORCHTITAN utilizes DCP's asynchronous checkpointing to reduce the checkpointing overhead by 5-15x compared to synchronous distributed checkpointing for the Llama 3.1 8B model (PyTorch Team, 2024b).

#### 2.3.2 FLIGHT RECORDER TO DEBUG JOB CRASHES

Debugging NCCL collective timeouts at large scales is challenging due to the asynchronous nature of communication kernels. PyTorch's Flight Recorder addresses this by logging the start, end, and enqueue times for all collective and p2p operations, along with metadata like process groups, source/destination ranks, tensor sizes, and stack traces.

This data is invaluable for diagnosing hangs in parallelism code. For PP, it can pinpoint the latest send or recv completed on the GPU, helping debug schedule bugs. For FSDP and TP, it identifies ranks that failed to call collectives, aiding in uncovering issues with PP scheduling or TP logic.

## 3 EXPERIMENTATION

In this section, we demonstrate the effectiveness of elastic distributed training using TORCHTITAN, via experiments on Llama 3.1 8B, 70B, and 405B, from 1D parallelism to 4D parallelism, at the scale from 8 GPUs to 512 GPUs. We also share the knowledge and experience gained through TORCHTITAN experimentation. A walkthrough of the codebase on how we apply (up to) 4D parallelism can be found in Appendix A.

### 3.1 EXPERIMENTAL SETUP

The experiments are conducted on NVIDIA H100 GPUs $ ^{2} $  with 95 GiB memory, where each host is equipped with 8 GPUs and NVSwitch. Two hosts form a rack connected to a TOR switch. A backend RDMA network connects the TOR switches. In TORCHTITAN we integrate a checkpointable data loader and provide built-in support for the C4 dataset (en variant), a colossal, cleaned version of Common Crawl's web crawl corpus (Raffel et al., 2020). We use the same dataset for all experiments in this section. For the tokenizer, we use the official one (tiktoken) released together with Llama 3.1.

### 3.2 PERFORMANCE

To showcase the elasticity and scalability of TORCHTITAN, we experiment on a wide range of GPU scales (from 8 to 512), as the underlying model size increases (8B, 70B, and 405B) with a varying number of parallelism dimensions (up to 4D). To demonstrate the effectiveness of the optimization techniques introduced in Section 2.2, we show how training throughput improves when adding each

individual technique on appropriate baselines. In particular, when training on a higher dimensional parallelism with new features, the baseline is always updated to include all previous techniques.

We note that, throughout our experimentation, memory readings are stable across the whole training process $ ^{3} $ , whereas throughput numbers (token per second, per GPU) are calculated and logged every 10 iterations, and always read at the (arbitrarily determined) 90th iteration. We do not report Model FLOPS Utilization (MFU) (Chowdhery et al., 2023) because when Float8 is enabled in TORCHTiTAN, both BFLOAT16 Tensor Core and FP8 Tensor Core are involved in model training, but they have different peak FLOPS and the definition of MFU under such scenario is not well-defined. We note that the 1D Llama 3.1 8B model training on 8 or 128 H100 GPUs without Float8 achieves 33% to 42% MFU.

<div style="text-align: center;">Table 1: 1D parallelism (FSDP) on Llama 3.1 8B model, 8 GPUs. Mixed precision training. Selective activation checkpointing. Local batch size 2, global batch size 16. (Stats per GPU)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Techniques</td><td style='text-align: center;'>Throughput (Tok/Sec)</td><td style='text-align: center;'>Comparison</td><td style='text-align: center;'>Memory (GiB)</td></tr><tr><td style='text-align: center;'>FSDP</td><td style='text-align: center;'>6,258</td><td style='text-align: center;'>100%</td><td style='text-align: center;'>81.9</td></tr><tr><td style='text-align: center;'>+ torch.compile</td><td style='text-align: center;'>6,674</td><td style='text-align: center;'>+ 6.64%</td><td style='text-align: center;'>77.0</td></tr><tr><td style='text-align: center;'>+ torch.compile + Float8</td><td style='text-align: center;'>9,409</td><td style='text-align: center;'>+ 50.35%</td><td style='text-align: center;'>76.8</td></tr></table>

<div style="text-align: center;">Table 2: 1D parallelism (FSDP) on Llama 3.1 8B model, 128 GPUs. Mixed precision training. Selective activation checkpointing. Local batch size 2, global batch size 256. (Stats per GPU)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Techniques</td><td style='text-align: center;'>Throughput (Tok/Sec)</td><td style='text-align: center;'>Comparison</td><td style='text-align: center;'>Memory (GiB)</td></tr><tr><td style='text-align: center;'>FSDP</td><td style='text-align: center;'>5,645</td><td style='text-align: center;'>100%</td><td style='text-align: center;'>67.0</td></tr><tr><td style='text-align: center;'>+ torch.compile</td><td style='text-align: center;'>6,482</td><td style='text-align: center;'>+ 14.82%</td><td style='text-align: center;'>62.1</td></tr><tr><td style='text-align: center;'>+ torch.compile + Float8</td><td style='text-align: center;'>9,319</td><td style='text-align: center;'>+ 65.08%</td><td style='text-align: center;'>61.8</td></tr></table>

<div style="text-align: center;">Table 3: 2D parallelism (FSDP + TP) + torch.compile + Float8 on Llama 3.1 70B model, 256 GPUs. Mixed precision training. Full activation checkpointing. FSDP degree 32, TP degree 8. Local batch size 16, global batch size 512. (Stats per GPU)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Techniques</td><td style='text-align: center;'>Throughput (Tok/Sec)</td><td style='text-align: center;'>Comparison</td><td style='text-align: center;'>Memory (GiB)</td></tr><tr><td style='text-align: center;'>2D</td><td style='text-align: center;'>897</td><td style='text-align: center;'>100%</td><td style='text-align: center;'>70.3</td></tr><tr><td style='text-align: center;'>+AsyncTP</td><td style='text-align: center;'>1,010</td><td style='text-align: center;'>+12.59%</td><td style='text-align: center;'>67.7</td></tr></table>

<div style="text-align: center;">Table 4: 3D parallelism (FSDP + TP + PP) + torch.compile + Float8 +AsyncTP on Llama 3.1 405B model, 512 GPUs. Mixed precision training. Full activation checkpointing. FSDP degree 4, TP degree 8, PP degree 16. Local batch size 32, global batch size 128. (Stats per GPU)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Schedule</td><td style='text-align: center;'>Throughput (Tok/Sec)</td><td style='text-align: center;'>Comparison</td><td style='text-align: center;'>Memory (GiB)</td></tr><tr><td style='text-align: center;'>1F1B</td><td style='text-align: center;'>100</td><td style='text-align: center;'>100%</td><td style='text-align: center;'>78.0</td></tr><tr><td style='text-align: center;'>Interleaved 1F1B</td><td style='text-align: center;'>130</td><td style='text-align: center;'>+ 30.00%</td><td style='text-align: center;'>80.3</td></tr></table>

Additional experimental details and loss-convergence tests for correctness can be found in Appendix B.10.

### 3.3 Scaling with TorchTitan 4D Parallelism

Scaling large language models (LLMs) requires parallelism strategies to handle increasing model sizes and data on thousands of GPUs. TORCHTITAN enables efficient scaling through composable

<div style="text-align: center;">Table 5: FSDP + CP + torch.compile + Float8 on Llama 3.1 8B model, 8 GPUs. Mixed precision training. Full activation checkpointing. Local batch size 1. (Stats per GPU)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Schedule</td><td style='text-align: center;'>Sequence Length</td><td style='text-align: center;'>Throughput (Tok/Sec)</td><td style='text-align: center;'>Memory (GiB)</td></tr><tr><td style='text-align: center;'>FSDP 8, CP 1</td><td style='text-align: center;'>32,768</td><td style='text-align: center;'>3,890</td><td style='text-align: center;'>83.9</td></tr><tr><td style='text-align: center;'>FSDP 4, CP 2</td><td style='text-align: center;'>65,536</td><td style='text-align: center;'>2,540</td><td style='text-align: center;'>84.2</td></tr><tr><td style='text-align: center;'>FSDP 2, CP 4</td><td style='text-align: center;'>131,072</td><td style='text-align: center;'>1,071</td><td style='text-align: center;'>84.0</td></tr><tr><td style='text-align: center;'>FSDP 1, CP 8</td><td style='text-align: center;'>262,144</td><td style='text-align: center;'>548</td><td style='text-align: center;'>84.5</td></tr></table>

<div style="text-align: center;">Table 6: 4D parallelism (FSDP + TP + PP + CP) + torch.compile + Float8 +AsyncTP + 1F1B on Llama 3.1 405B model, 512 GPUs. Mixed precision training. Full activation checkpointing. TP degree 8, PP degree 8. Local batch size 8. (Stats per GPU)</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Schedule</td><td style='text-align: center;'>Sequence Length</td><td style='text-align: center;'>Throughput (Tok/Sec)</td><td style='text-align: center;'>Memory (GiB)</td></tr><tr><td style='text-align: center;'>FSDP 8, CP 1</td><td style='text-align: center;'>32,768</td><td style='text-align: center;'>76</td><td style='text-align: center;'>75.3</td></tr><tr><td style='text-align: center;'>FSDP 4, CP 2</td><td style='text-align: center;'>65,536</td><td style='text-align: center;'>47</td><td style='text-align: center;'>75.9</td></tr><tr><td style='text-align: center;'>FSDP 2, CP 4</td><td style='text-align: center;'>131,072</td><td style='text-align: center;'>31</td><td style='text-align: center;'>77.1</td></tr><tr><td style='text-align: center;'>FSDP 1, CP 8</td><td style='text-align: center;'>262,144</td><td style='text-align: center;'>16</td><td style='text-align: center;'>84.9</td></tr></table>

4D parallelism. This section highlights key observations and motivations for using TORCHTITAN 4D parallelism, focusing on a specific combination shown in Figure 2.

<div style="text-align: center;"><img src="imgs/img_in_image_box_236_733_988_1043.jpg" alt="Image" width="61%" /></div>


<div style="text-align: center;">Figure 2: Scaling with 4D Parallelism</div>


#### 3.3.1 Scaling with FSDP

FSDP (ZeRO) is a general technique applicable to any model architecture and is often sufficient as the first degree of parallelism when communication is faster than computation (e.g., up to 512 GPUs). However, with larger scales, collective latency increases linearly with the world size, limiting efficiency. To overcome this, model parallelism like TP and PP can be combined with FSDP.

#### 3.3.2 2D PARALLELISM: TP WITH FSDP

Tensor Parallelism (TP) reduces collective latency by distributing work across GPUs, enabling smaller effective batch sizes and reducing peak memory usage for large models or sequence lengths. Combining FSDP and TP allows strong scaling with a fixed problem/batch size (Details shown in Figure 4). TP also improves FLOP utilization by optimizing matrix multiplication shapes. However, TP introduces blocking collectives and is typically limited to intra-node scaling (e.g., NVLink), with degrees usually capped at 8. Scaling beyond 4192 GPUs requires combining TP with PP.

#### 3.3.3 3D PARALLELISM: PP WITH 2D PARALLELISM

Pipeline Parallelism (PP) reduces communication bandwidth requirements by transmitting only activations and gradients between stages in a peer-to-peer manner. PP is particularly effective for mitigating FSDP communication latency at larger scales or in bandwidth-limited clusters. The efficiency of PP depends on pipeline schedules and microbatch sizes, which influence the size of pipeline “bubbles.”

#### 3.3.4 LONG CONTEXT TRAINING AND 4D PARALLELISM

Context Parallelism (CP) allows ultra long context training by splitting the context (sequence) dimension across GPUs to avoid OOM errors. CP is mainly used for long context training, to give the model capability to capture more correlations for tokens, thus enhancing the overall model quality. For scaling sequence length, CP can be used alone or together with DP. When training large models or on a large number of GPUs, we can combine CP with 3D parallelism, where TP usually keeps the inner-most DeviceMesh dimension, and CP applies in the next outer DeviceMesh dimension.

## 4 RELATED WORK

Libraries such as Megatron-LM (Narayanan et al., 2021), DeepSpeed (Rasley et al., 2020), veScale (Inc., 2024) and PyTorch Distributed (Paszke et al., 2019; Meta Platforms, Inc., 2024) provide APIs for distributed workflows. However, these frameworks present challenges in flexibility, integration, and scalability. TORCHTITAN addresses these limitations with native support for key features absent in existing systems:

• Megatron-LM: Requires model modifications for TransformerEngine, lacks seamless FSDP integration with TP and PP, and does not support advanced pipeline schedules to minimize computation overhead.

• DeepSpeed: Depends on Megatron-LM for TP and CP, with limited support for FSDP and advanced pipeline schedules.

• veScale: Does not support FSDP, CP, SAC, Float8 training, or torch.compile, and offers only three pipeline schedules, compared to TORCHTITAN's six.

We note that each of these libraries has its own strengths, and TORCHTITAN is designed to provide foundational components that can be leveraged by all of them. A detailed comparison, including feature breakdowns and code complexity analysis, is available in Appendix B.9. Slapo (Chen et al., 2023) introduces a schedule language to convert a PyTorch model for common model training optimizations such as 3D parallelism, and supports progressive optimization through high-level primitives. In contrast, TORCHTITAN provides modular and composable APIs built on DTensor and DeviceMesh.

## 5 CONCLUSION

TORCHTITAN is a powerful and flexible framework for LLM training, enabling seamless composability of parallelism techniques (FSDP, TP, PP, CP), memory optimizations (Float8, activation checkpointing), and PyTorch compiler integration for enhanced efficiency. Its modular design supports evolving architectures and hardware, fostering innovation with multi-axis metrics.

Designed for interpretability and production-grade training, TORCHTITAN offers elastic scalability, comprehensive training recipes, and expert guidance on distributed training strategies. As demonstrated in experiments, it accelerates training by 65.08% on Llama 3.1 8B (128 GPUs, 1D), 12.59% on Llama 3.1 70B (256 GPUs, 2D), and 30% on Llama 3.1 405B (512 GPUs, 3D) over optimized baselines, while enabling long-context training with 4D composability. With its robust features and high efficiency, TORCHTITAN is an ideal one-stop solution for challenging LLM training tasks.