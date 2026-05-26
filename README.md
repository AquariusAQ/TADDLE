# GRADER
GRADER: A Granular Agent for Defect Evaluation in LLM-Generated Peer Reviews

Here's a method section you can drop into your README:

## Method

GRADER detects defects in LLM-generated peer reviews by decomposing the task into four specialized analysis stages, an integration stage, and a final composition stage. The key idea is that different defect types leave different kinds of evidence — factual errors live in the review-vs-paper relationship, while bias and tone signals live in the review text itself — so we route each review through dedicated tools rather than relying on a single end-to-end classifier.


### Components

**Orchestrator.** Built on `Qwen3-30B-A3B-Thinking-2507` in thinking mode. It decomposes the review into evidence segments (factual claims, critical opinions, attitude expressions) and adaptively schedules tool calls under a budget of 8 calls per review. It sees only the review and the paper's abstract — full paper content is available downstream at the tools, where verification actually happens.

**Four analysis tools.** All four share the orchestrator's backbone but differ in prompts, paper access, and retrieval permissions:

- **VERIFY** — Fact-checks the review's claims against the paper. Emits typed evidence (factual error / no-evidence / careless omission) with confidence scores and supporting paper quotes. *Full paper access + external retrieval (Semantic Scholar / arXiv).*
- **CORRECT** — Classifies errors surfaced by VERIFY into `comprehension_error`, `factual_data_error`, `omission_error`, or `logic_error`, and flags unprofessional cases. *Operates on evidence segments; no retrieval.*
- **COMPLETE** — Checks whether each criticism is paired with an actionable revision direction. *Full paper access + external retrieval.*
- **TRANSFORM** — Detects subjective bias, hostile phrasing, and double standards. *Operates on review + relevant paper segments; no retrieval.*

Tool outputs are structured JSON.

**INTEGRATE.** A `Qwen3.5-9B` model fine-tuned with LoRA. It is the **only module with classification authority** and the only module whose parameters are updated — the orchestrator and the four tools are off-the-shelf. INTEGRATE consumes the review, paper representation, and the four tools' JSON traces, and emits a single JSON object containing the binary label `ẑ`, the multi-label defect vector `ŷ`, and a 5-point quality score. A runtime check enforces that INTEGRATE runs exactly once per review.

### Inference pipeline

1. **Paper parsing** — PDF → Markdown + figures via PaddleOCR-VL; figures captioned by Qwen3-VL-4B (≤200 tokens), appendix summarized by Qwen3-30B (≤500 tokens per subsection).
2. **Orchestration** — The orchestrator reads the review and abstract, plans a tool sequence, and dispatches calls. Recommended order is `VERIFY → CORRECT → COMPLETE → TRANSFORM`, but the orchestrator may short-circuit, revisit, or skip tools based on intermediate evidence.
3. **Analysis** — Each invoked tool returns structured JSON traces. Tools may call each other's outputs as context.
4. **Integration** — INTEGRATE synthesizes all traces into `(ẑ, ŷ, q)`. Thinking mode is disabled at this stage for latency.
5. **Composition** — The orchestrator emits the final user-facing JSON with per-defect explanations, evidence traces, and revision suggestions.

### Training

INTEGRATE is the only fine-tuned component. We use two-stage semi-supervised training:

- **Stage 1 (supervised).** LoRA fine-tuning on 1,080 expert-annotated reviews. Input: paper representation + review + four tools' JSON outputs. Target: standardized JSON with `ẑ`, `ŷ`, and quality score.
- **Stage 2 (semi-supervised).** The Stage-1 checkpoint pseudo-labels weakly-supervised synthetic reviews; we keep only samples whose predicted defects are consistent with the generation persona's weak label (~2,705 high-confidence samples), then continue training on the mix.

This split — frozen analysis tools, fine-tuned integrator — makes the system modular: tools can be swapped for stronger frontier models without retraining, and the integrator can be retrained as the defect taxonomy evolves.

## Fine-tuning Integrate

The training pipeline for Integrate is built on [LlamaFactory](https://github.com/hiyouga/LlamaFactory/). Please follow its official documentation to install the framework first.

```bash
git clone --depth 1 https://github.com/hiyouga/LlamaFactory.git
cd LlamaFactory
pip install -e .
pip install -r requirements/metrics.txt
```

Copy the dataset files into `LlamaFactory/data`:

- `integrate_trainer/data/dataset_stage_1.json`
- `integrate_trainer/data/dataset_stage_2.json`

Then add the following entries to `LlamaFactory/data/dataset_info.json`:

```json
"dataset_stage_1": {
  "file_name": "dataset_stage_1.json",
  "columns": {
    "prompt": "instruction",
    "response": "output"
  }
},
"dataset_stage_2": {
  "file_name": "dataset_stage_2.json",
  "columns": {
    "prompt": "instruction",
    "response": "output"
  }
}
```

Run **Stage 1** training:

```bash
bash Stage1_train.sh
```

Run **Stage 2** training:

```bash
bash Stage2_train.sh
```

### Example Launch Script

Below is an example command to serve the fine-tuned Integrate (Grader) model with vLLM, using the LoRA adapter obtained from the previous fine-tuning stage:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
    Qwen/Qwen3.5-9B \
    --max-parallel-loading-workers 32 \
    --port 8000 \
    --max-model-len 73728 \
    --enable-lora \
    --served-model-name grader \
    --enable-prefix-caching \
    --language-model-only \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --lora-modules masai=./saves/qwen3.5-9b/lora/sft/dataset_stage_2
```

This starts the server on port `8000`, with the LoRA module `masai` pointing to the Stage 2 training checkpoint. The `--served-model-name grader` flag ensures the model is accessible as `grader` (the expected name for the Integrate module elsewhere in the pipeline).


## Quick Start

`src/main.py` is the script for generating **persona-based LLM reviews** and performing **Deficiency Detection**.

This repository includes a sample paper. Follow the steps below to generate reviews and detect deficiencies:

---

### 1. Model Configuration

Configure model access in `src/configs/models.yaml`:

```yaml
# src/configs/models.yaml

qwen3_30b:
  model: "Qwen3-30B-A3B-Thinking-2507"   # Model deployed locally with vLLM
  temperature: 0.6
  max_completion_tokens: 16384
  base_url: "LOCAL_URL:default"          # vLLM service address
  api_key: "EMPTY"                       # Usually not required for local deployment
  extra_body:
    top_k: 20
    min_p: 0.0

grader:
  model: "grader"
  temperature: 0.1
  top_p: 0.95
  max_completion_tokens: 1024
  base_url: "LOCAL_URL:grader"           # Grader model service address
  api_key: "EMPTY"
  extra_body:
    lora_name: grader
```

**Deployment Notes:**

- Use **vLLM ≥ 0.19.1** to deploy `Qwen3-30B-A3B-Thinking-2507` and the **Grader model** (the Integrate module).  
- When deploying the Grader, add the following launch argument:
  ```bash
  --default-chat-template-kwargs '{"enable_thinking": false}'
  ```
- Fill in the `base_url` with the API endpoint of each model, and `api_key` with the API key (if required).

---

### 2. Review Generation

```bash
cd src/
python main.py \
    hydra/job_logging=none \
    hydra/hydra_logging=none \
    conferences=iclr \
    dataset.default_name="iclr" \
    dataset.default_paper_id="0a7TRHhhcS" \
    experiment.name="run_review" \
    experiment.workflow.phase_i=true \
    experiment.workflow.defense_initial=false \
    llm_reviewer='${qwen3_30b}' \
    agents.reviewer.max_workers_reviewer=5
```

- `conferences` : Controls the review scoring template.  
- `dataset.default_name` and `dataset.default_paper_id` : Dataset name and paper ID.  
- `experiment.workflow.phase_i` / `experiment.workflow.defense_initial` : Toggle review generation and deficiency detection (currently only generating reviews).  
- `llm_reviewer` : The model used for generating reviews.  
- `agents.reviewer.max_workers_reviewer` : Number of parallel review workers.

Generated reviews will be saved under `src/outputs/iclr_run_review/`. The file `results.json` contains all generated reviews.

---

### 3. Deficiency Detection

#### 3.1 Start the External Literature Search Backend

1. Create a file `key.json` inside `src/server/paper_searcher/` with your [Semantic Scholar](https://www.semanticscholar.org/) API key:
   ```json
   {
       "semanticscholar": "xxxxxxxxxx"
   }
   ```
2. Start the service:
   ```bash
   cd src/server/paper_searcher/
   pip install -r requirements.txt
   python paper_search_api.py
   ```
   The service will be running at **http://localhost:3101** by default.

#### 3.2 Run Deficiency Detection

```bash
cd src/
python main.py \
    hydra/job_logging=none \
    hydra/hydra_logging=none \
    conferences=iclr \
    dataset.default_name="iclr" \
    dataset.default_paper_id="0a7TRHhhcS" \
    experiment.name="run_detect" \
    experiment.input_path="outputs/iclr_run_review" \
    experiment.workflow.phase_i=false \
    experiment.workflow.defense_initial=true \
    llm_reviewer='${qwen3_30b}' \
    agents.defense_agent.input_paper_content="abstract" \
    llm_defense='${qwen3_30b}' \
    llm_tools='${qwen3_30b}' \
    llm_integrate_tool='${grader}' \
    agents.defense_agent.max_workers_analyze_initial=3
```

- `llm_defense` : Backbone model for the Orchestrator.  
- `llm_tools` : Backbone model for the content analysis tools.  
- `llm_integrate_tool` : Backbone model for the Integrate module (must use Grader).  
- `agents.defense_agent.max_workers_analyze_initial` : Number of parallel detection workers.

The output is saved to `src/outputs/iclr_run_detect/`.  
- `results.json` contains all detection results.  
- The `tool_history/` folder contains all tool-call histories.

---

## Dataset Crawling

Paper data is sourced from **OpenReview**. To obtain data beyond the provided sample:

1. In the `paper_crawler/` directory, create two files:
   - `username.key` : your OpenReview username
   - `password.key` : your OpenReview password
2. Run the crawling and processing scripts:
   ```bash
   cd MASA/GAVEL/paper_crawler
   python download_all.py
   python paper_processor.py
   ```
3. The processed data will be stored in `paper_crawler/dataset/processed/`.  
   For example, for NeurIPS 2025, all eligible paper PDFs are saved in  
   `paper_crawler/dataset/processed/NeurIPS2025/NeurIPS2025/`.

---

## Paper Parsing

Copy the crawled data to the parsing directory. For NeurIPS 2025:  
Copy the folder `paper_crawler/dataset/processed/NeurIPS2025/NeurIPS2025` into `paper_parser/datasets`.

### 1. PDF Parsing

Deploy the PaddleOCR backend (see [official documentation](https://www.paddleocr.ai/latest/version3.x/pipeline_usage/PaddleOCR-VL.html)), then start the service:

```bash
bash start_server.sh
```

By default, it runs on port **8118**. Then parse the PDFs:

```bash
python 1_parse_pdf.py --dataset NeurIPS2025
```

### 2. Image Caption Generation

Uses the `Qwen3-VL-4B-Instruct` model by default (recommended to deploy with vLLM). Run:

```bash
python 2_analyze_all_images.py --datasets NeurIPS2025 --url 127.0.0.1:8001
```

- `--datasets` : Specify the dataset name.  
- `--url` : Model service endpoint.

### 3. Appendix Summarization

Uses the `Qwen3-30B-A3B-Thinking-2507` model by default (recommended to deploy with vLLM). Run:

```bash
python 3_summarize_appendix.py --datasets NeurIPS2025 --url 127.0.0.1:8002
```

- `--datasets` : Specify the dataset name.  
- `--url` : Model service endpoint.