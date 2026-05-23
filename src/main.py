import os
import json
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import argparse  # New: Command line argument parsing
from typing import List, Dict, Optional
from agents import Author, Reviewer, MetaReviewer, DefenseAgent
from utils.logger import logger
from utils.parallel import parallel_run  # Import parallel tool
from utils.config import config  # Import global configuration variable
from utils.shared_data import DeepThreadSafeDict, result, paper  # Import thread-safe global variables
from utils.message import global_message_queue 

def validate_and_load_workflow() -> None:
    # 1. Define phase order mapping (phase_i -> Phase 1, and so on)
    review_phases = [
        "phase_i",   # Phase 1: Initial review
        "phase_ii",  # Phase 2: Author rebuttal
        "phase_iii", # Phase 3: Updated review
        "phase_iv",  # Phase 4: Meta-review
        "phase_v"    # Phase 5: Final decision
    ]
    
    # 2. Extract indices of all enabled review phases
    workflow = config["experiment"]["workflow"]
    enabled_phase_indices: List[int] = []
    for idx, phase in enumerate(review_phases):
        if workflow[phase]:
            enabled_phase_indices.append(idx)
    
    # 3. Validate that enabled review phases are consecutive
    if enabled_phase_indices:
        # Consecutive condition: max index - min index + 1 == number of enabled phases
        min_idx = min(enabled_phase_indices)
        max_idx = max(enabled_phase_indices)
        expected_count = max_idx - min_idx + 1
        actual_count = len(enabled_phase_indices)
        
        if expected_count != actual_count:
            # Extract non-consecutive phase names for debugging
            enabled_phases = [review_phases[idx] for idx in enabled_phase_indices]
            raise ValueError(
                f"Review process phases must be consecutive! Currently enabled non-consecutive phases: {enabled_phases}"
            )
    
    # 4. Check if starting from phase_1 (whether phase_i is enabled)
    start_from_phase1 = workflow["phase_i"]
    if not start_from_phase1:
        # 5. Validate input_path is provided
        input_path = config["experiment"]["input_path"].strip()
        if not input_path:
            raise ValueError("When not starting from phase_1, input_path must provide the output path of completed phases")
        input_path = os.path.join(input_path, f"{config['dataset']['default_paper_id']}")

        # 6. Define required file paths to read
        message_file = os.path.join(input_path, "message.jsonl")
        result_file = os.path.join(input_path, "results.json")
        config_file = os.path.join(input_path, "config.json")
        
        # 7. Read message.jsonl (parse JSON line by line)
        if not os.path.exists(message_file):
            raise FileNotFoundError(f"message.jsonl not found in {input_path}")
        
        message_input: List[Dict] = []
        with open(message_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                try:
                    msg = json.loads(line)
                    message_input.append(msg)
                except json.JSONDecodeError as e:
                    raise json.JSONDecodeError(
                        f"Invalid format in message.jsonl line {line_num}: {str(e)}",
                        doc=line,
                        pos=e.pos
                    ) from e
        logger.info(f"Loaded {len(message_input)} messages from previous phases")
        
        # 8. Read result.json
        if not os.path.exists(result_file):
            raise FileNotFoundError(f"result.json not found in {input_path}")
        
        with open(result_file, "r", encoding="utf-8") as f:
            try:
                result_input = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid format in result.json: {str(e)}",
                    doc=f.read(),
                    pos=e.pos
                ) from e
        logger.info(f"Loaded previous results from result.json")
        logger.info(result_input)
            
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"config.json not found in {input_path}")
        
        with open(config_file, "r", encoding="utf-8") as f:
            try:
                config_input = json.load(f)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid format in config.json: {str(e)}",
                    doc=f.read(),
                    pos=e.pos
                ) from e
        logger.info(f"Loaded previous configuration from config.json")
            
        # 9. Assign as required
        global_message_queue.load_messages(message_input)  # Assign to global message queue
        result.update(result_input)
        config["input_config"] = config_input  # Update current config with loaded config


# Use @hydra.main decorator to define main function
# config_path: Path to directory containing configuration files
# config_name: Name of configuration file (without .yaml extension)
@hydra.main(config_path="configs", config_name="review_config", version_base="1.3.2")
def main(cfg: DictConfig):
    """
    Main function.
    Hydra automatically passes the loaded configuration as a DictConfig object to the cfg parameter.
    """
    # 1. Load user-specified configuration file
    try:
        # with open(config_path, "r") as f:
        #     global_config = yaml.safe_load(f)
        # # Assign loaded configuration to global variable in utils.config
        # if not isinstance(global_config, dict):
        #     raise ValueError(f"Invalid configuration file format, parsed result type: {type(global_config)}")
        # config.update(global_config)  # Key: Allow other modules to access the configuration

        # Use configuration object provided by Hydra
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        config.update(config_dict)
        # if conference_config := config.get("conferences"):
        #     config.update(conference_config)
        config['dataset']['default_paper_id'] = str(config['dataset']['default_paper_id'])

        # Call logger instance's init_log_dir method to initialize logs (replace old independent function)
        if config.get('experiment', {}).get('name', None):
            log_name = f"{config['dataset']['default_name']}_{config['experiment']['name']}"
        else:
            log_name = None
        logger.init_log_dir(
            log_name=log_name,
            paper_id=config['dataset']['default_paper_id']
        )
        global_message_queue.set_message_log_path(logger.message_log_path)
        logger.save_json(config, "config.json")  # Save the actual configuration used

        logger.info(f"=== Starting multi-agent review process ===")
        logger.info(f"Using Hydra configuration file")
    except FileNotFoundError:
        # Note: If an error occurs before log initialization, temporarily initialize to avoid RuntimeError
        if not hasattr(logger, '_logger') or logger._logger is None:
            logger.init_log_dir()  # Temporarily initialize default log directory
        logger.error(f"Hydra configuration file not found")
        return
    except Exception as e:
        # Similarly, ensure logs are initialized when an error occurs
        if not hasattr(logger, '_logger') or logger._logger is None:
            logger.init_log_dir()
        logger.error(f"Failed to load configuration file: {str(e)}")
        return

    # 2. Parse dataset parameters
    dataset_name = config["dataset"]["default_name"]
    paper_id = config["dataset"]["default_paper_id"]
    paper["id"] = paper_id  # Store current paper ID in shared variable
    logger.info(f"Target paper: {dataset_name}/{paper_id}")
    ## Validate if phases are valid
    validate_and_load_workflow()

    # 3. Initialize Agents
    author = Author(config)
    reviewers = []
    skip_reviewers_list = [r.strip() for r in config["agents"]["reviewer"]["skip_reviewers"].strip().split(",")]
    for i in range(len(config["agents"]["reviewers"])):
        if config["agents"]["reviewers"][i]["name"] not in skip_reviewers_list:
            reviewers.append(Reviewer(config, i))
    meta_reviewer = MetaReviewer(config)
    defense_agent = DefenseAgent(config)

    # 4. Load paper content
    author._set_allowed_tools(["paper_processor"])

    paper_content = author.call_tool( "paper_processor", action="load", dataset_name=dataset_name, paper_id=paper_id)
    paper_title = author.call_tool( "paper_processor", action="extract_title", paper_content=paper_content)
    if not paper_content:
        logger.error("Failed to load paper, terminating process")
        return
    paper["title"] = paper_title
    paper["content"] = paper_content  # Store paper content in shared variable

    # Load paper abstract
    paper_abstract = author.call_tool("paper_processor", action="extract_abstract", paper_content=paper_content)
    if not paper_abstract:
        logger.error("Failed to load paper abstract, terminating process")
        return
    paper["abstract"] = paper_abstract  # Store paper abstract in shared variable

    author._clear_allowed_tools()  # Manually clear permissions
    # logger.info(paper["title"])
    # exit(0)

    # 5. Execute five-phase review process (same as before)
    # Phase I: Review evaluation
    if config["experiment"]["workflow"]["phase_i"]:
        logger.info(f"\n=== Phase I: Review Evaluation (Parallel Mode) ===")
        # Construct parallel tasks: generate_initial_review method for each reviewer
        phase_i_tasks = [
            (
                reviewer.run,  # Function
                (),  # Positional arguments
                {
                    "phase": "phase_i", 
                    "paper_content": paper_content,
                    "allowed_tools": []
                }
            ) 
            for reviewer in reviewers
        ]
        # Execute in parallel (3 reviewers call LLM simultaneously)
        parallel_run(phase_i_tasks, max_workers=config["agents"]["reviewer"]["max_workers_reviewer"], enable_prefix_cache=False)
        initial_reviews = meta_reviewer.collect_initial_reviews()  # Collect after all reviews are completed
    else:
        initial_reviews = meta_reviewer.collect_initial_reviews()  # Collect after all reviews are completed
        logger.info(f"\n=== Phase I: Review Evaluation skipped ===")

    # Defense analysis of initial reviews
    if config["experiment"]["workflow"]["defense_initial"]:
        logger.info(f"\n=== Analyzing Initial Reviews ===")
        if config["agents"]["defense_agent"]["input_paper_content"] == "full":
            defense_paper_content = "Full content: \n" + paper_content
        elif config["agents"]["defense_agent"]["input_paper_content"] == "abstract":
            defense_paper_content = "Abstract: \n" + paper_abstract
        else:
            raise(f"Unknown paper content type {config["agents"]["defense_agent"]["input_paper_content"]}")
        if config["tools"]["malice_defense_tool"]["enabled"]:
            defense_agent.run(phase="analyze_initial_reviews", 
                              initial_reviews=initial_reviews, 
                              paper_content=defense_paper_content, 
                              allowed_tools=["malice_defense_tool"])
        else:
            defense_agent.run(phase="analyze_initial_reviews", 
                              initial_reviews=initial_reviews, 
                              paper_content=defense_paper_content, 
                              allowed_tools=[])
        author.receive_malice_analysis()  # Receive malice analysis results from Defense Agent

    logger.info(f"\n=== Multi-agent review process completed ===")
    logger.info(f"Final decision: {meta_reviewer.final_decision}")
    logger.save_json(result, "results.json")  # Save final review results

import atexit
import sys
from utils.parallel import shutdown_background_executor

# Register exit hook: Force shutdown thread pool when program exits
atexit.register(shutdown_background_executor)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:  # Catch Ctrl+C
        shutdown_background_executor()
        print("\n✅ Received Ctrl+C, all threads/processes terminated")
        sys.exit(1)