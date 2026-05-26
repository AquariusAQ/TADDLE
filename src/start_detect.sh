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
    llm_integrate_tool='${taddle}' \
    agents.defense_agent.max_workers_analyze_initial=3