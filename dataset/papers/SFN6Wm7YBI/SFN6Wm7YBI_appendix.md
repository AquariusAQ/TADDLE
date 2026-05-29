## A COMPOSABLE 4D PARALLELISM WALKTHROUGH

We have discussed the scaling with TORCHTITAN 4D parallelism and the motivations to apply different parallelisms to scale training to thousands of GPUs. In this section we will walk through the 4D parallelism code in TORCHTITAN.

The first step is to create an instance of the model (e.g. the Transformer for Llama models) on the meta device. We then apply PP by splitting the model into multiple PP stages according to the pipeline_parallel_split_points config. Note that for PP with looped schedules, we may obtain multiple model_parts from PP splitting, where each item in model_parts is

one stage-model-chunk. Next we apply SPMD-style distributed training techniques including TP, activation checkpointing, torch.compile, FSDP, and mixed precision training for each model part, before actually initializing the sharded model on GPU.

# meta init
with torch.device("meta"):
    model = model_cls.from_model_args(model_config)

# apply PP
pp_schedule, model_parts = models_pipelining_fns[model_name](
    model, pp_mesh, parallel_dims, job_config, device, model_config, loss_fn
)

for m in model_parts:
    # apply SPMD-style distributed training techniques
    models_parallelize_fns[model_name](m, world_mesh, parallel_dims, job_config)
    # move sharded model to GPU and initialize weights via DTensor
    m.to_empty(device="cuda")
    m.init_weights()

To apply PP to the model, we run the following code at the high level. pipeline_llama_manual_split splits the model into multiple stages according to the manually given pipeline_parallel_split_points config, by removing the unused model components from a complete model (on the meta device). Then build_pipeline_schedule make the pipeline schedule with various options from torch.distributed.pipelining, including 1F1B (Narayanan et al., 2019), GPipe (Huang et al., 2019), interleaved 1F1B (Narayanan et al., 2021), etc. instructed by the pipeline_parallel_schedule config.

stages, models = pipeline_llama_manual_split(
    model, pp_mesh, parallel_dims, job_config, device, model_config
)

pp_schedule = build_pipeline_schedule(job_config, stages, loss_fn)
return pp_schedule, models

TP and FSDP are applied in the SPMD-style models_parallelize_fns function. To apply TP, we utilize the DTensor parallelize_module API, by providing a TP "plan" as the instruction of how model parameters should be sharded. In the example below, we showcase the (incomplete) code for sharding the repeated TransformerBlock.

for layer_id, transformer_block in model.layers.items():
    layer_tp_plan = {
        "attention_norm": SequenceParallel(),
        "attention": PrepareModuleInput(
            input_layout=(Shard(1), None),
            desired_input_layout=(Replicate(), None),
        ),
        "attention.wq": ColwiseParallel(),
    ...
}

parallelize_module(
    module=transformer_block,
    device_mesh=tp_mesh,
    parallelize_plan=layer_tp_plan,
)

Then, we apply the FSDP by wrapping each individual TransformerBlock and then the whole model. Note that the FSDP2 implementation in PyTorch comes with mixed precision training support. By default, we use torch.bfloat16 on parameters all-gather and activation computations, and use torch.float32 on gradient reduce-scatter communication and optimizer updates.

mp_policy = MixedPrecisionPolicy(param_dtype, reduce_dtype)
fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}

for layer_id, transformer_block in model.layers.items():
    # As an optimization, do not reshard_after_forward for the last
    # TransformerBlock since FSDP would prefetch it immediately
    reshard_after_forward = int(layer_id) < len(model.layers) - 1
    fully_shard(
        transformer_block,
        **fsdp_config,
        reshard_after_forward=reshard_after_forward,
    )
    fully_shard(model, **fsdp_config)

Independently, we can apply CP by running each training iteration under a Python context manager.

optional_context_parallel_ctx = (
    utils.create_context_parallel_ctx(
        cp_mesh=world_mesh["cp"],
        cp_buffers=[input_ids, labels] + [m.freqs_cis for m in model_parts],
        cp_seq_dims=[1, 1] + [0 for _ in model_parts],
        cp_no_restore_buffers={input_ids, labels},
        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
    )
    if parallel_dims.cp_enabled
    else None
)
...
with train_context(optional_context_parallel_ctx):
    pred = model(input_ids)
    loss = loss_fn(pred, labels)

## B SUPPLEMENTARY MATERIALS

### B.1 FULLY SHARDED DATA PARALLEL

FSDP2 advances the tensor sharding approach by replacing the original FSDP1 FlatParameter sharding. Specifically, parameters are now represented as DTensors sharded on the tensor dimension 0. This provides better composability with model parallelism techniques and other features that require the manipulation of individual parameters, allowing sharded state dict to be represented by DTensor without any communication, and provides for a simpler meta-device initialization flow via DTensor. For example, FSDP2 unlocks finer grained tensor level quantization, especially Float8 tensor quantization, which we will showcase in the results section.

As part of the rewrite from FSDP1 to FSDP2, FSDP2 implements an improved memory management system by avoiding using record stream. This enables deterministic memory release, and as a result provides lower memory requirements per GPU relative to FSDP1. For example on Llama 2 7B, FSDP2 records an average of 7% lower GPU memory versus FSDP1.

In addition, by writing efficient kernels to perform multi-tensor allgather and reduce scatter, FSDP2 shows on-par performance compared to FSDP1, with even slight performance gains - using the Llama 2 7B, FSDP2 shows an average gain of 1.5% faster throughput.

The performance gains are the result of employing two small performance improvements. First, only a single division kernel is run for the FP32 reduce scatter (pre-dividing the local FP32 reduce-scatter gradient by world size, instead of a two step pre and post divide by square root of world size). Secondly, in TORCHTITAN, FSDP2 is integrated with a default of not re-sharding the final block in a transformer layer during the forward pass, since it will be immediately re-gathered at the start of the backward pass.

Usage: TORCHTITAN has fully integrated FSDP2 as the default parallelism when training, and the data_parallel_shard_degree is the controlling dimension in the command line or TOML file. Note that for ease of use, the default data_parallel_shard_degree is -1, means to simply use all GPUs available, so user do not need to specify the actual world size.

### B.2 HYBRID SHARDED DATA PARALLEL

Hybrid Sharded Data Parallel (HSDP) is an extension of FSDP (Zhang et al., 2022). In FSDP, communication occurs between all devices within the FSDP group. However, at some point, the FSDP communication overhead exceeds its corresponding computation because the latency of allgather/reduce-scatter communications increases linearly with the number of devices. This results in low MFU and becomes worthless to add more GPUs for scaling.

HSDP obviates this to some degree by creating a 2-D DeviceMesh that contains replica groups on one dimension and shard groups on the other dimension, where each shard group runs FSDP and the replica group runs normal data parallel. This ensures the FSDP communications happen in a fraction of the original world size, with the addition of backward gradient allreduce across replica groups. HSDP reduces FSDP communication overhead and allows further scaling with data parallel.

Usage: TORCHTITAN makes it easy to experiment with HSDP by using the two configurable settings: data_parallel_shard_degree and data_parallel_replicate_degree, which controls the degree of the shard and replica groups we are creating. The product of both replicate and shard degree is the actual data parallel world size.

### B.3 TENSOR PARALLEL

TP partitions the attention and feed forward network (MLP) modules of a transformer layer across multiple devices, where the number of devices used is the TP degree. This allows for multiple GPUs to cooperatively process the same batch by using the local sharded model parameters, at the cost of adding all-reduce/all-gather/reduce-scatter operations to synchronize intermediate activations.

<div style="text-align: center;"><img src="imgs/img_in_image_box_300_864_925_1239.jpg" alt="Image" width="51%" /></div>


<div style="text-align: center;">Figure 3: Tensor Parallel in detail (2 GPUs, data moves from left to right).</div>


Due to the additional collectives introduced by TP, it needs to happen within a fast network (i.e. NVLink). When training LLMs, TP is usually combined with FSDP, where TP shards within nodes and FSDP shards across nodes to create the 2D hierarchical sharding on different DeviceMesh dimensions.

Usage: Because of the synergistic relationship between TP and SP, TORCHTITAN natively bundles these two together and they are jointly controlled by the TP degree setting in the command line.

<div style="text-align: center;"><img src="imgs/img_in_chart_box_305_168_914_609.jpg" alt="Image" width="49%" /></div>


<div style="text-align: center;">Figure 4: FSDP2 + Tensor Parallel (TP degree 4) sharding layout, with 2 nodes of 4 GPUs.</div>


or the TOML entry of tensor_parallel_degree. Setting this to 2 for example would mean that 2 GPUs within the node will share the computational load for each transformer layer's attention and MLP modules via TP, and normalization/dropout layers via Sequence Parallel. Loss Parallel is implemented via a context manager as it needs to control the loss computation outside of the model's forward computation. It can be enabled via enable_loss_parallel.

### B.4 PIPELINE PARALLEL

We expose several parameters to configure PP. pipeline_parallel_degree controls the number of ranks participating in PP. pipeline_parallel_split_points accepts a list of strings, representing layer fully-qualified-names before which a split will be performed. Thus, the total number of pipeline stages V will be determined by the length of this list. pipeline_parallel_schedule accepts the name of the schedule to be used. If the schedule is multi-stage, there should be V > 1 stages assigned to each pipeline rank, otherwise V == 1. pipeline_parallel_microbatches controls the number of microbatches to split a data batch into.

### B.5 Enabling 4D PARALLEL TRAINING: CONTEXT-PARALLEL (CP)

To address context scaling, we have incorporated Context Parallelism (CP) into TORCHTITAN. Following the principles of modular design of TORCHTITAN, CP was integrated via a context manager that dynamically replaces calls to attention operators (namely, scaled_dot_product_attention) with CP operations, ensuring no changes to the model code are required.

Under the hood, CP shards the DTensor along the sequence dimension across the CP device mesh. It extends the DTensor dispatcher to handle CP-specific operations, such as Ring Attention and causal attention load balancing, ensuring efficient operation. By extending DTensor's capabilities to support CP, TORCHTITAN ensures that CP is fully compatible with all other parallelisms (FSDP, TP, PP), optimizations (e.g., activation checkpointing, torch.compile), and DCP. This demonstrates the extensibility of TORCHTITAN's modular design, which accommodates future optimizations seamlessly while maintaining performance and compatibility.

### B.6 ACTIVATION CHECKPOINTING

TORCHTITAN offers two types of Selective Activation Checkpointing which allow for a more nuanced tradeoff between memory and recomputation. Specifically, we offer the option to selectively checkpoint "per layer" or "per operation". The goal for per operation is to free memory used by operations that are faster to recompute and save intermediates (memory) for operations that are slower to recompute and thus deliver a more effective throughput/memory trade-off.

Usage: AC is enabled via a two-line setting in the command line or TOML file. Specifically, mode can be either none, selective, or full. When selective is set, then the next config of selective_ac_type is used which can be either a positive integer to enable selective layer checkpointing, or op to enable selective operation checkpointing. Per layer takes an integer input to guide the checkpointing policy, where 1 = checkpoint every layer (same as full), 2 = checkpoint every other layer, 3 = checkpoint every third layer, etc. Per op(eration) is driven by the _save_list policy in parallelize_llama.py which flags high arithmetic intensity operations such as matmul (matrix multiplication) and SPDA (Scaled Dot Product Attention) for saving the intermediate results, while allowing other lower intensity operations to be recomputed. Note that for balancing total throughput, only every other matmul is flagged for saving.

### B.7 ASYNCTP

The SymmetricMemory collectives used in AsyncTP are faster than standard NCCL collectives and operate by having each GPU allocate an identical memory buffer in order to provide direct P2P access. SymmetricMemory relies on having NVSwitch within the node, and is thus generally only available for H100 or newer GPUs.

Usage:AsyncTP is enabled within the experimental section of the TORCHTITAN TOML config file and turned on or off via the enable_async_tensor_parallel boolean setting.

### B.8 CUSTOMIZING FSDP2 MIXED PRECISION IN TORCHTITAN

Mixed Precision is controlled by the MixedPrecisionPolicy class in the apply_fsdp function, which is then customized with param_dtype as BF16, and reduce_dtype defaulting to FP32 by default in TORCHTITAN. The reduce_dtype in FP32 means that the reduce-scatter in the backwards pass for gradient computation will take place in FP32 to help maximize both stability and precision of the gradient updates.

### B.9 TORCHTITAN: COMPREHENSIVE FEATURE SET AND REDUCED COMPLEXITY

#### B.9.1 TORCHTITAN ENABLES NEW DESIGNS

TORCHTITAN's extensive feature set and broad design space coverage are driven by its unified design principles i.e. modularity, composability, and extensibility. Leveraging these principles, TORCHTITAN seamlessly integrates diverse parallelism strategies (FSDP, TP, PP, and CP) and optimizations (e.g., SAC, Float8 training). This unified framework not only supports advanced pipeline schedules and multi-dimensional parallelism but also simplifies the integration of new techniques, making it highly adaptable for cutting-edge research and production-grade deployments.

The following table highlights TORCHTITAN’s capabilities in context of parallelism, checkpointing and compiler support offerings compared to Megatron-LM, DeepSpeed, and veScale:

#### B.9.2 Code Complexity and Maintainability

TORCHTITAN’s design principles also contribute to its significantly reduced code complexity. Despite offering a rich feature set, TORCHTITAN maintains a compact and modular codebase, making it easier to extend, maintain, and evolve while ensuring high performance. The following table compares the lines of code (LOC) for TORCHTITAN with Megatron-LM and DeepSpeed:

<div style="text-align: center;">Table 7: Comparison of TORCHTITAN with Megatron-LM, DeepSpeed, and veScale with respect to parallelism, compiler support, activation checkpointing, and model checkpointing.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Features</td><td style='text-align: center;'>TORCHTITAN</td><td style='text-align: center;'>Megatron-LM</td><td style='text-align: center;'>DeepSpeed</td><td style='text-align: center;'>veScale</td></tr><tr><td style='text-align: center;'>FSDP-Zero2</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>FSDP-Zero3</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>HSDP</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>TP</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>Async TP (Micro-pipelining)</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>CP</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>PP-Gpipe</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>PP-Interleaved (1F1B)</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>PP-Looped-BFS</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>PP-1F1B</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>PP-Flexible-Interleaved-1F1B</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>PP-ZeroBubble</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>(TP+SP)+PP</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>DDP+(TP+SP)+PP</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>FSDP+(TP+SP)</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>FSDP+(TP+SP)+PP</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>FSDP+(TP+SP)+PP+CP</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>MoE</td><td style='text-align: center;'>Ongoing</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>Full AC</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>Flexible SAC</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>DCP</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td></tr><tr><td style='text-align: center;'>Float8 Training</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No</td><td style='text-align: center;'>No</td></tr><tr><td style='text-align: center;'>torch.compile</td><td style='text-align: center;'>Yes</td><td style='text-align: center;'>No $ ^{4} $</td><td style='text-align: center;'>Partial</td><td style='text-align: center;'>No</td></tr></table>

<div style="text-align: center;">Table 8: Lines of Code (LOC) comparison across systems.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Lines of Code (LOC)</td><td style='text-align: center;'>TORCH TITAN</td><td style='text-align: center;'>Megatron-LM</td><td style='text-align: center;'>DeepSpeed</td></tr><tr><td style='text-align: center;'>Core Codebase</td><td style='text-align: center;'>7K</td><td style='text-align: center;'>93K</td><td style='text-align: center;'>94K</td></tr><tr><td style='text-align: center;'>Total Codebase (Including Utilis)</td><td style='text-align: center;'>9K</td><td style='text-align: center;'>269K</td><td style='text-align: center;'>194K</td></tr></table>

### B.10 Extended Experiments Analysis: Performance and Loss Converging

#### B.10.1 PERFORMANCE

Our experiments in Section 3.2 serve multiple objectives:

• Establish composability and modularity: TORCHTITAN demonstrates seamless integration of various parallelisms and optimization techniques.

• Showcase performance improvements: Significant speed-ups are observed across parallels and optimizations.

• Validate elastic scalability: TORCHTITAN scales effectively with both the model size and the number of GPUs.

• Ablation studies: Detailed performance gains for individual techniques are presented.

## I n particular

• Table 1: Highlights improvements from compiler support over eager execution, followed by further gains with Float8 training.

• Table 2: Demonstrates how earlier gains scale as the number of GPUs increases.

• Table 3: Shows speed-up achieved by AsyncTP (a HW/SW co-designed technique) over 2D training combined with torch.compile and Float8 training.

• Table 4: Quantifies the benefits of Interleaved 1F1B scheduling over 1F1B on top of AsyncTP, torch.compile, and Float8 training.

• Table 5: Demonstrates the effectiveness of CP on enabling long context training, even at small scale.

• Table 6: Demonstrate the composability of 4D parallelism, and the effectiveness of CP on enabling long context training at large scale.

For FSDP, the ZeRO-3 variant is used for all experiments except for those involving PP where the ZeRO-2 variant is used. This distinction is due to the inefficiency of ZeRO-3 in PP, where it incurs additional all-gather calls for each microbatch. In contrast, ZeRO-2 gathers parameters only once for the first microbatch and reshards after the last microbatch's backward pass.

#### B.10.2 Loss Converging

TORCHTITAN’s design principles have influenced the development of advanced distributed training features such as FSDP2, AsyncTP, PP, and CP in PyTorch's distributed library. Throughout these contributions, we have ensured the loss converging of individual techniques as well as their various combinations of parallelisms and optimizations.

For example, below is a series of loss-converging tests covering both parallelisms and training optimizations. We use notations of "FSDP 8" for an experiment in which the degree of FSDP is 8, "FSDP 8, CP 8" for an experiment on 64 GPUs where FSDP degree is 8 and CP degree is 8, etc. We assume the correctness of FSDP, which can be further verified by comparing it with DDP or even single-device jobs.

<div style="text-align: center;">Table 9: Loss-converging tests setup.</div>



<table border=1 style='margin: auto; width: max-content;'><tr><td style='text-align: center;'>Parallelism</td><td style='text-align: center;'>Techniques</td></tr><tr><td style='text-align: center;'>FSDP 8 (ground truth)</td><td style='text-align: center;'>default</td></tr><tr><td style='text-align: center;'>FSDP 8, TP 2, PP 2</td><td style='text-align: center;'>torch.compile, Float8, async TP, Interleaved 1F1B</td></tr><tr><td style='text-align: center;'>FSDP 8, TP 2, CP 2, PP 2</td><td style='text-align: center;'>torch.compile, Float8, async TP, Interleaved 1F1B</td></tr><tr><td style='text-align: center;'>FSDP 8, CP 8</td><td style='text-align: center;'>default</td></tr></table>

<div style="text-align: center;"><img src="imgs/img_in_chart_box_293_931_929_1155.jpg" alt="Image" width="51%" /></div>


<div style="text-align: center;">Figure 5: Loss converging tests on Llama 3.1 8B. C4 dataset. Local batch size 4, global batch size 32. 3000 steps, 600 warmup steps.</div>