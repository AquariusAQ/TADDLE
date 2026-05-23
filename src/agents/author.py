from .base_agent import BaseAgent
from utils.llm_client import call_llm, call_llm_with_tool
from utils.logger import logger
from utils.shared_data import result  # Import global result variable
from utils.parallel import parallel_run, run_in_background  # Import parallel processing utilities
import threading  # Import threading lock to protect shared variables
import json
import re

class Author(BaseAgent):
    def __init__(self, config):
        super().__init__(
            agent_type="author",
            name=config["agents"]["author"]["name"],
            config=config
        )
        self.reviews = []  # All collected reviews (Format: [{"sender": reviewer name, "content": review content, "msg_type":"initial_review"}])
        self.responses = []  # Store exclusive rebuttals for each review (Format: [{"reviewer_name": "", "review_content": "", "rebuttal": ""}])
        self.response_lock = threading.Lock()  # New: Thread lock to avoid conflicts in parallel writing to self.responses
        self.initial_malice_analysis = None  # Initial malice detection result
        self.updated_malice_analysis = None  # New: Updated malice detection result

    # Receive initial review malice analysis from Defense Agent
    def receive_malice_analysis(self):
        self.initial_malice_analysis = self.receive_messages(
            msg_type="malice_analysis_initial"
        )
        if self.initial_malice_analysis:
            logger.info(f"[{self.name}] Received {len(self.initial_malice_analysis)} initial review malice analysis records")
            # self._extract_and_save_initial_revision_suggestions()  # New: Extract and save
        else:
            logger.warning(f"[{self.name}] No initial review malice analysis results received")
        return self.initial_malice_analysis
    
    def receive_updated_malice_analysis(self):
        self.updated_malice_analysis = self.receive_messages(
            msg_type="malice_analysis_updated"
        )
        if self.updated_malice_analysis:
            logger.info(f"[{self.name}] Received {len(self.updated_malice_analysis)} updated review malice analysis records")
            # self._extract_and_save_updated_revision_suggestions()  # New: Extract and save
        else:
            logger.warning(f"[{self.name}] No updated review malice analysis results received")
        return self.updated_malice_analysis

    def collect_reviews(self):
        """Collect initial reviews from all reviewers"""
        self.reviews = self.receive_messages(msg_type="initial_review")  # Filter by message type
        logger.info(f"[{self.name}] Collected {len(self.reviews)} initial reviews")
        return self.reviews
    
    # New: Subfunction for processing single review (for parallel calls)
    def _process_single_review(self, review, paper_content, malice_analysis):
        """Process single review: Extract objections → Generate rebuttal → Safely write results (independent subtask)"""
        reviewer_name = review["sender"]
        review_content = review["content"]

        # New: Filter analysis for current review from malice analysis results (match reviewer name)
        current_malice = None
        if malice_analysis:
            for analysis in malice_analysis:
                for single_analysis in analysis["content"]:
                    if single_analysis["reviewer_name"] == reviewer_name:
                        current_malice = single_analysis["malice_analysis"]
                        break

        malice_prompt_suffix = ""
        if current_malice:
            malice_prompt_suffix = (f"\n[Defense Analysis Prompt]: "
                                    f"The following is a reference analysis of whether this review contains malice: {current_malice}\n"
                                    "If \\boxed{} is empty, the review is considered non-malicious\n"
                                    "If \\boxed{} is not empty, it indicates the type of malice in the review."
                                    "In this case, please specifically address the malicious points in the rebuttal (e.g., correct factual errors, supplement evidence to respond to unfounded criticism).")
        
        logger.info(f"[{self.name}] Processing review from reviewer {reviewer_name} in parallel")

        # 1. Prepare LLM request: Pass available LLM-driven tools (comment_analyzer)
        llm_tools = self.get_llm_tool_config()  # Get tool configurations
        llm_tools_name = self.get_llm_tool_names()  # Get list of tool names
        system_prompt = f"""You are the author of the paper, and you need to generate an exclusive rebuttal for the review from [Reviewer {reviewer_name}].
        {self.config["prompt"]["image_analyze_tool_usage_prompt"] if "image_analyzer" in llm_tools_name else ""}
        {self.config["prompt"]["comment_analyze_tool_usage_prompt"]  if "comment_analyzer" in llm_tools_name else ""}
        Requirements: Only respond to the objections in this review, do not involve other reviews, and provide well-founded arguments.
        If external literature citations are needed, ensure proper citation formatting.{malice_prompt_suffix}
        """
        messages = [
            {"role": "system", "content": system_prompt + f"\nPaper Content:\n{paper_content}"},
            # {"role": "user", "content": },
            {"role": "user", "content": f"[Review from Reviewer {reviewer_name}]:\n{review_content}"},
        ]

        final_rebuttal = call_llm_with_tool(
            messages=messages, 
            call_tool_agent=self, 
            llm_tools=llm_tools, 
            llm_tools_name=llm_tools_name, 
            model="llm_author"
        )

        result["reviews"][reviewer_name]["rebuttal"] = {
            "content": final_rebuttal
        }

        # 4. Original logic for storing rebuttals and sending messages remains unchanged...
        with self.response_lock:
            self.responses.append({
                "reviewer_name": reviewer_name,
                "review_content": review_content,
                "rebuttal": final_rebuttal
            })
        self.send_message(recipient=reviewer_name, content=final_rebuttal, msg_type="author_rebuttal")
        logger.info(f"[{self.name}] Completed rebuttal generation for reviewer {reviewer_name}")
        return final_rebuttal

    def generate_response(self, paper_content):
        if not self.reviews:
            logger.error(f"[{self.name}] No reviews received, cannot generate rebuttals")
            return

        logger.info(f"[{self.name}] Starting parallel processing (total {len(self.reviews)} reviews)")

        # Build parallel task list: each task corresponds to processing one review
        parallel_tasks = [
            (
                self._process_single_review, # Subtask function
                (review, paper_content, self.initial_malice_analysis), # Positional arguments (review object, paper content)
                {} # Keyword arguments (none)
            )
            for review in self.reviews  # Create a task for each review
        ]

        # Execute tasks in parallel (reuse existing parallel_run, control max concurrency)
        # max_workers is recommended to match the number of reviewers, or refer to LLM service concurrency capacity (e.g., 3-5)
        max_workers = min(len(self.reviews), 5)  # Avoid overwhelming LLM service with excessive concurrency
        parallel_run(parallel_tasks, max_workers=max_workers)

        logger.info(f"[{self.name}] Parallel processing of rebuttals for all reviews completed (total {len(self.responses)} rebuttals)")
        return self.responses

    def _extract_and_save_revision_suggestions(self, malice_analysis_list: list, title: str, filename: str):
        """
        General logic: Extract revision suggestions and generate Markdown
        :param malice_analysis_list: Flattened list of malice analysis results
        :param title: Markdown document title
        :param filename: Filename for saving
        """
        # Build Markdown content
        markdown_content = f"# {title}\n\n"
        # Regex match content inside \result{...} (supports multi-line, non-greedy matching)
        # result_pattern = re.compile(r"\\result\{(.*)\}", re.DOTALL)

        for analysis in malice_analysis_list:
            # Extract reviewer name (default: Unknown Reviewer + ID)
            reviewer_name = analysis.get("reviewer_name", f"Unknown_Reviewer_{analysis.get('review_id', '0')}")
            suggestions = analysis.get("malice_result", "").get("suggestions", "")

            # Extract revision suggestions inside \result{}
            # match = result_pattern.search(malice_text)
            # if match and match.group(1).strip():
            #     suggestions = match.group(1).strip()
            #     suggestions = suggestions.replace("\\n", "\n")  # Restore line breaks
            # else:
            #     suggestions = "No constructive revision suggestions"

            # Append reviewer section (level 2 heading + suggestion content)
            markdown_content += f"## {reviewer_name}\n\n"
            markdown_content += f"{suggestions}\n\n"
            markdown_content += "---\n\n"  # Separator line to distinguish different reviewers

        # Remove redundant separator line at the end
        markdown_content = markdown_content.rstrip("---\n\n")

        if self.config["experiment"].get("enable_llm_suggestion_md_tidy", False):
            run_in_background(
                lambda: (
                    # Call LLM
                    markdown_content_refine := call_llm(
                        prompt=f"Markdown Text: {markdown_content}",
                        system="You are proficient in organizing Markdown files. Without modifying the content, adjust this Markdown text to a unified structured format and logical ordered list.\n",
                        model="llm_author"
                    )["answer"],
                    # Save file
                    logger.save_markdown(content=markdown_content_refine, filename=filename)
                ),
                # Note: Any exceptions in the lambda will be caught by run_in_background's try-except
            )
        else:
            try:
                logger.save_markdown(content=markdown_content, filename=filename)
            except Exception as e:
                logger.error(f"[{self.name}] Failed to save revision suggestions: {str(e)}", exc_info=True)

    def _extract_and_save_initial_revision_suggestions(self):
        """Extract and save revision suggestions from initial reviews"""
        # Flatten data: message list → reviewer analysis list
        all_analysis = []
        for msg in self.initial_malice_analysis:
            all_analysis.extend(msg.get("content", []))
        # Generate and save Markdown
        self._extract_and_save_revision_suggestions(
            malice_analysis_list=all_analysis,
            title="Collection of Revision Suggestions from Initial Reviews",
            filename="initial_revision_suggestions.md"
        )

    def _extract_and_save_updated_revision_suggestions(self):
        """Extract and save revision suggestions from updated reviews"""
        # Flatten data (same logic as initial)
        all_analysis = []
        for msg in self.updated_malice_analysis:
            all_analysis.extend(msg.get("content", []))
        # Generate and save Markdown
        self._extract_and_save_revision_suggestions(
            malice_analysis_list=all_analysis,
            title="Collection of Revision Suggestions from Updated Reviews",
            filename="updated_revision_suggestions.md"
        )


    def run_child(self, phase, **kwargs):
        """Only supports rebuttal generation phase"""
        if phase == "phase_ii":
            return self.generate_response(kwargs["paper_content"])
        else:
            logger.warning(f"[{self.name}] Unsupported phase: {phase}")
            return None