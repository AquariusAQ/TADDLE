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