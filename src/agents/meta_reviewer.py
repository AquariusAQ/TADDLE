import os
import json
from typing import List, Optional, Dict
from .base_agent import BaseAgent
from utils.llm_client import call_llm
from utils.logger import logger
from utils.shared_data import result  # Import global result variable
from utils.message import global_message_queue  # Key: Import global message queue
import re

SELF_DESCRIPTION = {
    "inclusive": {
        "type": "Inclusive",
        "description": "As an inclusive Area Chair (AC), you tend to listen to all reviewers' opinions and make final decisions based on your own judgment combined with their feedback."
    },
    "conformist": {
        "type": "Conformist",
        "description": "As a conformist Area Chair (AC), you perfunctorily perform your AC duties. You mostly follow reviewers' suggestions to write meta-reviews, score papers, and decide whether to accept papers."
    },
    "authoritarian": {
        "type": "Authoritarian",
        "description": "As an authoritarian Area Chair (AC), you tend to read papers independently, make judgments based on personal evaluation, and mostly ignore reviewers' opinions."
    }
}


class MetaReviewer(BaseAgent):
    def __init__(self, config):
        super().__init__(
            agent_type="meta_reviewer",
            name=config["agents"]["meta_reviewer"]["name"],
            config=config
        )
        self.type = config["agents"]["meta_reviewer"]["type"]  # Inclusive/Authoritarian/Conformist
        self.ignore_analysis_without_malice = config["agents"]["meta_reviewer"].get("ignore_analysis_without_malice", False)
        self.self_description = (f"{self.config['prompt']['meta_reviewer_prompt']['system_prefix']}"
                            f"{SELF_DESCRIPTION[self.type]['description']}")
        self.initial_reviews = []  # Initial reviews
        self.author_response = None  # Author rebuttals
        self.updated_reviews = []  # Updated reviews
        self.meta_review = None  # Meta review
        self.final_decision = None  # Final decision
        self.meta_rating = None  # Meta review score
        self.meta_decision = None  # Meta review decision
        # Store defense agent analysis results
        self.initial_malice_analysis = None
        self.updated_malice_analysis = None
        self.external_meta_reviews: List[Dict] = []  # Store external meta reviews + abstracts
        self.visible_reviewers = config["agents"]["meta_reviewer"].get("visible_reviewers", [])
        if self.visible_reviewers:
            logger.info(f"[{self.name}] Configured visible reviewers: {self.visible_reviewers}")
        else:
            logger.info(f"[{self.name}] All reviewers are visible (no filter configured)")


    # Receive malice analysis results of initial reviews
    def receive_initial_malice_analysis(self):
        raw_analysis = self.receive_messages(msg_type="malice_analysis_initial")
        filtered_analysis = []
        for msg in raw_analysis:
            msg_copy = msg.copy()
            msg_copy["content"] = self._filter_visible_reviews(msg.get("content", []), name_key="reviewer_name")
            filtered_analysis.append(msg_copy)
        self.initial_malice_analysis = filtered_analysis
        logger.info(f"[{self.name}] Received {len(self.initial_malice_analysis)} visible initial review malice analysis records")
        return self.initial_malice_analysis

    # Receive malice analysis results of updated reviews
    def receive_updated_malice_analysis(self):
        raw_analysis = self.receive_messages(msg_type="malice_analysis_updated")
        # 新增：过滤仅可见评审员的更新后恶意分析结果
        filtered_analysis = []
        for msg in raw_analysis:
            msg_copy = msg.copy()
            msg_copy["content"] = self._filter_visible_reviews(msg.get("content", []), name_key="reviewer_name")
            filtered_analysis.append(msg_copy)
        self.updated_malice_analysis = filtered_analysis
        logger.info(f"[{self.name}] Received {len(self.updated_malice_analysis)} visible updated review malice analysis records")
        return self.updated_malice_analysis
    
    def _filter_visible_reviews(self, input_data: list, name_key: str = "sender") -> list:
        """Filter reviews based on visible reviewers list (if configured)"""
        if not self.visible_reviewers:
            return input_data
        
        filtered_data = []
        for item in input_data:
            if not isinstance(item, dict):
                continue
            reviewer_name = item.get(name_key)
            if reviewer_name in self.visible_reviewers:
                filtered_data.append(item)
        
        logger.info(f"[{self.name}] Filtered data: retained {len(filtered_data)}/{len(input_data)} entries (visible reviewers only)")
        return filtered_data

    def collect_initial_reviews(self):
        """Collect initial reviews from Phase I (filtered by visible reviewers)"""
        raw_reviews = self.receive_messages(msg_type="initial_review")
        self.initial_reviews = self._filter_visible_reviews(raw_reviews, name_key="sender")
        logger.info(f"[{self.name}] Collected {len(self.initial_reviews)} visible initial reviews")
        return self.initial_reviews
    
    # New: Read all author rebuttals (no recipient restriction, filter from global queue)
    def collect_all_rebuttals(self):
        """Read all author_rebuttal type messages (filtered by visible reviewers)"""
        all_rebuttals = global_message_queue.receive(
            recipient=None,
            msg_type="author_rebuttal"
        )
        # 新增：过滤仅可见评审员的作者回复
        filtered_rebuttals = self._filter_visible_reviews(all_rebuttals, name_key="recipient")
        if not filtered_rebuttals:
            logger.warning(f"[{self.name}] No visible author rebuttals found")
            self.author_response = []
            return []
        
        formatted_rebuttals = [
            f"[Rebuttal for Reviewer {rebuttal['recipient']}]\n{rebuttal['content']}" 
            for rebuttal in filtered_rebuttals
        ]
        self.author_response = formatted_rebuttals
        logger.info(f"[{self.name}] Retrieved {len(formatted_rebuttals)} visible exclusive rebuttals")
        return formatted_rebuttals

    def guide_discussion(self):
        """Phase III: Guide reviewer-AC discussion (based on own type)"""
        logger.info(f"[{self.name}] Starting discussion guidance (Type: {self.type})")
        
        # New: First collect all rebuttals (ensure discussion based on complete information)
        self.collect_all_rebuttals()
        if not self.author_response:
            logger.warning(f"[{self.name}] No author rebuttals found, guiding discussion based on existing reviews")
        
        if self.type == "inclusive":
            guidance = f"All reviewers are requested to respond to previous objections one by one based on the author's personal rebuttal to you, explaining whether to adjust scores and the reasons"
        elif self.type == "authoritarian":
            # Authoritarian AC generates initial judgment based on full reviews + full rebuttals (ensure complete information)
            initial_judgment = call_llm(
                prompt=f"Initial Reviews: {self.initial_reviews}\nFull Author Rebuttals: {self.author_response}\nPlease provide your initial judgment",
                system="You are an authoritarian AC, quickly form initial judgments based on full interaction information",
                model="llm_meta_reviewer"
            )["answer"]
            guidance = f"I have made an initial assessment based on full information: {initial_judgment}\nAll reviewers are requested to supplement review comments based on the author's exclusive rebuttal to you"
        else:  # Conformist
            guidance = f"All reviewers are requested to fully discuss based on the author's exclusive rebuttal to you, and update scores after forming a majority opinion"

        # Send guidance to all reviewers (logic unchanged)
        for reviewer in self.config["agents"]["reviewers"]:
            reviewer_name = reviewer["name"]
            if self.visible_reviewers and reviewer_name not in self.visible_reviewers:
                continue
            self.send_message(
                recipient=reviewer_name,
                content=guidance,
                msg_type="discussion_guidance"
            )
        logger.info(f"[{self.name}] Discussion guidance sent successfully: {guidance}")
        return guidance

    def check_divergence(self):
        """Check if reviewer score divergence requires further mediation"""
        ratings = [int(r["content"].split("Updated Score:")[1].split("\n")[0]) for r in self.updated_reviews]
        max_r = max(ratings)
        min_r = min(ratings)
        logger.info(f"[{self.name}] Score divergence: {max_r - min_r} (Threshold: {self.config['process']['phase_iii']['divergence_threshold']})")
        return (max_r - min_r) > self.config["process"]["phase_iii"]["divergence_threshold"]

    def generate_meta_review(self, paper_content):
        """Phase IV: Generate meta review (integrate all information)"""
        logger.info(f"[{self.name}] Starting meta review generation (based on full interaction)")
        
        # Use defense agent analysis results (reduce weight of malicious reviews)
        malice_suffix = "Defense analysis results for updated reviews are as follows:\n"
        high_malice_reviewers = []
        if self.updated_malice_analysis:
            for analysis in self.updated_malice_analysis:
                for single_analysis in analysis["content"]:
                    if (not self.ignore_analysis_without_malice) or len(single_analysis["malice_result"]) > 0:
                        high_malice_reviewers.append({
                            "reviewer_name": single_analysis["reviewer_name"],
                            "malice_result": single_analysis["malice_result"],
                            "malice_analysis": single_analysis["malice_analysis"]
                        })
        for high_malice_reviewer in high_malice_reviewers:
            logger.info(f"[{self.name}] Identified reviewer: {high_malice_reviewer['reviewer_name']} with potential malice including {high_malice_reviewer['malice_result']}")
            malice_suffix += f"\n[Defense Analysis Prompt]: Reviewer {high_malice_reviewer['reviewer_name']}'s review contains potential malice of {high_malice_reviewer['malice_result']} (empty means no malice), analysis as follows: {high_malice_reviewer['malice_analysis']}"
        # for high_malice_reviewer in high_malice_reviewers:
        #     logger.info(f"[{self.name}] Identified high-impact malicious reviewer: {high_malice_reviewer['reviewer_name']}")
        #     malice_suffix += f"\n[Defense Analysis Prompt]: Reviewer {high_malice_reviewer['reviewer_name']}'s review contains high-impact malice ({single_analysis['malice_result']}), analysis as follows: {high_malice_reviewer['malice_analysis']}"
        if len(high_malice_reviewers) > 0:
            malice_suffix += "Please carefully reference these review comments in the meta review and appropriately reduce their weight.\n"

        # if high_malice_reviewers:
        #     malice_suffix = f"Note: Reviews from reviewers {high_malice_reviewers} contain high-impact malice and require careful reference\n"
        #     logger.info(f"[{self.name}] Identified high-impact malicious reviewers: {high_malice_reviewers}")
        
        # System Prompt explicitly states "based on full information"
        # Under normal circumstances, the meta review score follows the comprehensive suggestions of all reviewers, and generally will not be higher than the highest reviewer score nor lower than the lowest reviewer score.
        system_prompt = f"""{self.self_description}
        Now you need to evaluate the review comments provided by reviewers and write a meta review, including:
        {self.config['process']['phase_iv']['meta_review_sections']}
        Your meta review will be used to decide which papers are accepted or rejected.
        You need to give a final score ranging from {self.config['process']['phase_i']['rating_range']},
        Score Explanation:
        Rating:
        {self.config['prompt']['meta_reviewer_prompt']['score_explanation']}
        Additional scores:
        1. Soundness:
            - 1 point: Seriously unsound (the core method of the paper is wrong, the argumentation process is logically invalid, and cannot support the research conclusions);
            - 2 points: Basically sound but with obvious flaws (the core method has no principled errors, but the argumentation process has omissions, and some conclusions lack effective support);
            - 3 points: Sound (the core method is feasible, the argumentation logic is clear, there are no core flaws, and the conclusions are persuasive);
            - 4 points: Highly rigorous (the methodology is scientifically designed and impeccable, the argumentation process is progressive, logically rigorous, and the conclusions are reliable and verifiable).
        2. Presentation:
            - 1 point: Confusing expression (the paper structure is disorganized, the chapter logic is disconnected, there are a lot of terminology errors and grammatical errors, and readability is extremely poor);
            - 2 points: Basically clear but ambiguous (the paper structure is basically complete, the core content is identifiable, but some expressions are ambiguous, the use of terminology is not standardized enough, and the charts (if any) are not clear enough);
            - 3 points: Clear (the paper structure is complete and logically coherent, the use of terminology is standardized, the charts (if any) are standardized and can assist in explaining the content, and the expression is concise and easy to understand);
            - 4 points: Excellent (the expression is professional and standardized, the structure is rigorous and orderly, the logical context is clear, the charts (if any) are reasonably and beautifully designed, the language is fluent and easy to understand, and it meets the presentation requirements of academic papers).
        3. Contribution:
            - 1 point: No substantial contribution (the content of the paper completely repeats existing research, does not put forward any new viewpoints, new methods or new discoveries, and has no value to the research field);
            - 2 points: Minor incremental contribution (the paper has minor improvements or supplements on the basis of existing research, does not break through the existing research framework, and the contribution is limited);
            - 3 points: Obvious contribution (the paper puts forward new viewpoints, methods or research perspectives, fills small research gaps in the field, and has certain reference value for related research);
            - 4 points: Breakthrough contribution (the paper puts forward innovative methods, core viewpoints or research paradigms, significantly promotes the research progress in the field, has important guiding significance for subsequent research, and drives the development of the field).
        
        {self.config['prompt']['meta_reviewer_prompt']['meta_review_notes']}
        The score, corresponding level, and core explanation you give must strictly comply with the explanations in the above table.

        Please follow this JSON format output:
        {{
            "content": "Your meta-review content here, with clear sections and points, include summary, strengths, weaknesses, and suggestions for improvement.",
            "scores": {{
                "Rating": X,
                "Soundness": Y,
                "Presentation": Z,
                "Contribution": W
            }}
        }}
        IMPORTANT FORMAT RULES:
            1. The output must be ONLY the above JSON (no "Final Decision: \\boxed{{}}" wrapper, no extra text before/after JSON)
            2. The JSON must be STRICTLY valid: no trailing commas, no missing commas, correct double quotation marks, no extra newlines in string values, no extra spaces/symbols outside the JSON structure
            3. No other content (e.g., overall decision explanation, notes, reminders) is allowed in the output—only the JSON object that can be directly parsed

        """
        self.meta_review_raw = call_llm(
            prompt=(f"Paper Content:\n{paper_content}\n"
                    f"Full Initial Reviews: {self.initial_reviews}\n"
                    f"Full Author Rebuttals: {self.author_response}\n"
                    f"Full Updated Reviews: {self.updated_reviews}\n"
                    f"{malice_suffix}"
                    "Please generate a meta review based on full interaction."),
            system=system_prompt,
            model="llm_meta_reviewer"
        )["answer"]
        
        self.meta_rating_json = json.loads(self.meta_review_raw)
        self.meta_rating = self.meta_rating_json["scores"]
        self.meta_review = self.meta_rating_json["content"]
        logger.info(f"[{self.name}] Meta review generated successfully (Score: {self.meta_rating})")
        result["meta_review"]["content"] = self.meta_review
        result["meta_review"]["score"] = self.meta_rating

        self.send_message(
                recipient=self.name,
                content=self.meta_review,
                msg_type="meta_review"
            )
        return self.meta_review

    def make_final_decision(self, paper_abstract: Optional[str] = None, use_external_data: bool = False):
        """Phase V: Make final decision (supports single paper/multiple external papers modes)
        Args:
            paper_abstract: Abstract of single paper (original parameter)
            use_external_data: Whether to use externally loaded multi-paper data (new parameter)
        """
        if not use_external_data:
            logger.info(f"[{self.name}] Starting final decision-making")
    
            system_prompt = f"""{self.self_description}
            Now you need to make the final acceptance decision based on the meta review.
            Select the most appropriate decision from possible options and explain the reasons in detail.
            Present the final decision in the format "Final Decision: \\boxed{{}}" (only final decision keyword inside braces).
            This is an extremely rigorous top-tier conference. Strictly note that the paper acceptance rate should be around 32% (including all types of acceptance). This means that if this paper is accepted, it needs to be better than 68% of the papers you have reviewed.
            """
    
            self.final_decision = call_llm(
                prompt=f"Paper Abstract:\n{paper_abstract}\nMeta Review: {self.meta_review}\nPossible Decisions: {self.config['process']['phase_v']['possible_decisions']}\nPlease select the most appropriate decision and explain the reasons",
                system=system_prompt,
                model="llm_meta_reviewer"
            )["answer"]
    
            self.meta_decision = self._extract_result(self.final_decision)
            logger.info(f"[{self.name}] Final decision generated successfully: {self.meta_decision}")
            result["meta_review"]["final_decision"] = self.meta_decision
    
            return self.final_decision
        else:
            max_retries = self.config['process']['phase_v'].get('decision_retry_times', 3) 
            retry_count = 0
            decision_result = None
            prompt = "" 
            
            logger.info(f"Starting multi-paper final decision-making: Total {len(self.external_meta_reviews)} papers, acceptance rate {self.config['process']['phase_v']['acceptance_rate']*100}%")
            if not self.external_meta_reviews:
                logger.error("No valid external paper data loaded, cannot execute decision-making")
                return
    
            sorted_papers = sorted(
                self.external_meta_reviews,
                key=lambda x: x["meta_review_score"],
                reverse=True
            )
            accept_count = max(1, round(len(sorted_papers) * self.config['process']['phase_v']['acceptance_rate']))
    
            papers_summary = []
            for idx, paper in enumerate(sorted_papers, 1):
                papers_summary.append(
                    f"[Paper {idx}]"
                    f"ID: {paper['paper_id']} | Meta Review Score: {paper['meta_review_score']}\n"
                    f"Abstract: {paper['paper_abstract']}\n"
                    f"Meta Review: {paper['meta_review_content']}"
                )
    
            system_prompt = f"""{self.self_description}
            You need to make final acceptance decisions for multiple papers based on their meta reviews and abstracts, following an acceptance rate of {self.config['process']['phase_v']['acceptance_rate']*100}%.
            Core Rules:
            1. Acceptance Count = Total Papers × {self.config['process']['phase_v']['acceptance_rate']*100}% (round up/down based on paper quality for non-integers, at least 1 paper)
            2. Prioritize accepting papers with high meta review scores and outstanding core contributions
            3. Comprehensive judgment of academic value and innovation based on abstracts and meta review content
            4. Accepted papers must be labeled with acceptance type (Accept-oral/Accept-spotlight/Accept-poster)
            5. Rejected papers must include core reasons (e.g., low score, insufficient contribution, etc.)
            Output Format Requirements:
            - Provide overall decision explanation (acceptance count, screening criteria)
            - Present in two sections: "Accepted Papers" and "Rejected Papers"
            - Each paper must include ID, score, acceptance type, and reasons, following this JSON format:
            {{
                "accepted_papers": [
                    {{
                        "paper_id": "XXXX",
                        "meta_review_score": X.X,
                        "decision_type": "Accept-oral",
                        "acceptance_reason": "High score, outstanding contribution......"
                    }},
                    ...
                ],
                "rejected_papers": [
                    {{
                        "paper_id": "YYYY",
                        "meta_review_score": Y.Y,
                        "decision_type": "Reject",
                        "rejection_reason": "Low score, limited contribution......"
                    }},
                    ...
                ]
            }}
            IMPORTANT FORMAT RULES:
            1. The output must be ONLY the above JSON (no "Final Decision: \\boxed{{}}" wrapper, no extra text before/after JSON)
            2. The JSON must be STRICTLY valid: no trailing commas, no missing commas, correct double quotation marks, no extra newlines in string values, no extra spaces/symbols outside the JSON structure
            3. No other content (e.g., overall decision explanation, notes, reminders) is allowed in the output—only the JSON object that can be directly parsed
            Possible Acceptance Types: {self.config['process']['phase_v']['possible_decisions']}
            """
    
            while retry_count < max_retries and decision_result is None:

                if retry_count == 0:
                    prompt = f"Summary of Papers for Decision:\n{'\n'.join(papers_summary)}\n" \
                             f"Total Papers: {len(sorted_papers)} | Planned Acceptance Count: {accept_count}\n" \
                             f"Please make final decisions strictly following the rules and output VALID JSON."
                else:
                    prompt = f"{prompt}\n\nERROR: Your previous JSON output had syntax errors (e.g., missing commas, trailing commas, invalid formatting). Please CORRECT the format and re-generate the JSON. Ensure it is 100% valid JSON that can be parsed by Python's json library."
    
                self.final_decision = call_llm(
                    prompt=prompt,
                    system=system_prompt,
                    model="llm_meta_reviewer",
                    temperature=0.4
                )["answer"]
    
                logger.info(f"Multi-paper final decision generated (retry {retry_count}/{max_retries})")
                logger.info(f"Final Decision Content: {self.final_decision}")

                # 核心修改：直接处理LLM输出的JSON字符串（移除boxed解析逻辑）
                json_str = self.final_decision.strip()  # 去除首尾空白符（换行、空格、制表符等）
                try:
                    # 尝试直接解析纯JSON
                    decision_result = json.loads(json_str)
                except json.JSONDecodeError:
                    # 解析失败时的降级处理
                    logger.warning("⚠️  Standard JSON parsing failed, trying fallback solutions")
                    try:
                        # 尝试用json5解析（兼容松散格式）
                        import json5
                        decision_result = json5.loads(json_str)
                        logger.warning("⚠️  Used json5 for parsing (minor format issues fixed)")
                    except (ImportError, json5.JSON5DecodeError):
                        # 手动修复常见问题：尾随逗号
                        json_str_fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
                        try:
                            decision_result = json.loads(json_str_fixed)
                            logger.warning("⚠️  Fixed trailing commas and parsed successfully")
                        except json.JSONDecodeError as e:
                            # 所有修复方式都失败，触发重试
                            retry_count += 1
                            logger.error(f"❌ JSON parse failed (retry {retry_count}/{max_retries}): {e}")
                            logger.error(f"❌ Problematic JSON snippet: {json_str[max(0, e.pos-50):e.pos+50]}")
                            decision_result = None  # 标记为失败，触发重试

            if decision_result is None:
                logger.error(f"❌ All {max_retries} retries failed! Using empty decision result.")
                decision_result = {}
            else:
                logger.info("✅ Multi-paper final decision parsed successfully!")
    
            # 6. 保存结果（原有）
            result["multi_paper_final_decision"] = {
                "total_papers": len(sorted_papers),
                "accept_count": accept_count,
                "decision_content": self.final_decision,
                "decision_result": decision_result,
                "sorted_papers": [p["paper_id"] for p in sorted_papers],
                "retry_count": retry_count
            }
            return self.final_decision

    ####### Load external meta reviews and paper abstracts for final decision #######

    def load_external_meta_reviews(self, output_dirs: List[str], paper_ids: List[str]):
        """
        Load external meta reviews and paper abstracts from specified output folders (supports single folder with multiple papers)
        Scenario 1: Length of output_dirs = 1 → All papers are in "paper ID subfolders" under this folder
        Scenario 2: Length of output_dirs > 1 → output_dirs correspond to paper_ids one-to-one
        Args:
            output_dirs: List of output folder paths (supports 1 or multiple)
            paper_ids: List of paper IDs
        """
        if not output_dirs or not paper_ids:
            logger.error("output_dirs and paper_ids cannot be empty!")
            raise ValueError("Valid output folder list and paper ID list must be provided")
        if len(output_dirs) == 1:
            # Single folder scenario: All papers use the same parent folder, data in "parent folder/paper_id" subdirectories
            parent_dir = output_dirs[0]
            out_dir_list = [parent_dir for _ in paper_ids]  # Generate path list with same length as paper_ids
            logger.info(f"Single output folder mode detected: All paper data will be read from paper ID subfolders under parent folder [{parent_dir}]")
        else:
            # Multiple folders scenario: Strictly verify one-to-one correspondence between paths and paper IDs
            if len(output_dirs) != len(paper_ids):
                logger.error("In multi-output folder mode, output_dirs and paper_ids must have the same length!")
                raise ValueError("output_dirs and paper_ids must correspond one-to-one (single folder mode can pass 1 path)")
            out_dir_list = output_dirs
            logger.info(f"Multi-output folder mode detected: {len(out_dir_list)} paths correspond to {len(paper_ids)} papers respectively")
    

        # if len(output_dirs) != len(paper_ids):
        #     logger.error("output_dirs and paper_ids have different lengths, cannot associate paper data")
        #     raise ValueError("output_dirs and paper_ids must correspond one-to-one with the same length")

        # Verify if paper_processor tool is enabled
        if "paper_processor" not in self.direct_tools:
            logger.error("paper_processor tool is not enabled, please enable it in config")
            raise RuntimeError("paper_processor tool is required to extract abstracts, please enable it first")
        paper_processor = self.direct_tools["paper_processor"]
        dataset_name = self.config["dataset"]["default_name"]  # Get dataset name from config (unified source)
    

        self.external_meta_reviews = []
        for idx, (out_dir, paper_id) in enumerate(zip(out_dir_list, paper_ids)):
            logger.info(f"Loading {idx+1}th external paper: output_dir={out_dir}, paper_id={paper_id}")

            # 1. Read results.json (meta review data) from output folder
            results_path = os.path.join(out_dir, paper_id, "results.json")
            if not os.path.exists(results_path):
                logger.warning(f"Skipped: {results_path} file does not exist")
                continue
            
            try:
                with open(results_path, "r", encoding="utf-8") as f:
                    results_data = json.load(f)
                meta_review = results_data.get("meta_review", {})
                meta_review_content = meta_review.get("content", "").strip()
                meta_review_score = meta_review.get("score", 0)

                if not meta_review_content or not meta_review_score:
                    logger.warning(f"Skipped: {results_path} lacks core meta review data")
                    continue
            except Exception as e:
                logger.error(f"Skipped: Failed to read {results_path} - {str(e)}", exc_info=True)
                continue
            
            # 2. Load paper content and extract abstract (reuse paper_processor tool)
            dataset_name = self.config["dataset"]["default_name"]
            paper_abstract = paper_processor.call(action="extract_abstract", dataset_name=dataset_name, paper_id=paper_id)
            # paper_content = paper_processor._load_paper(dataset_name, paper_id)
            # if isinstance(paper_content, str) and "Paper file does not exist" in paper_content:
            #     logger.warning(f"Skipped: Failed to load paper {dataset_name}/{paper_id}")
            #     continue
            
            # paper_abstract = paper_processor._extract_abstract(paper_content)
            # if not paper_abstract or "Failed to extract abstract" in paper_abstract:
            #     logger.warning(f"No valid abstract extracted for paper {dataset_name}/{paper_id}")
            #     paper_abstract = "No valid abstract available"

            # 3. Store external data (associate path, ID, meta review, abstract)
            self.external_meta_reviews.append({
                "parent_output_dir": out_dir,
                "paper_id": paper_id,
                "meta_review_content": meta_review_content,
                "meta_review_score": float(meta_review_score),
                "paper_abstract": paper_abstract,
                "results_file_path": results_path  # Record file path for traceability
            })

        logger.info(f"External data loading completed: Successfully loaded {len(self.external_meta_reviews)}/{len(output_dirs)} paper records")
        return self.external_meta_reviews


    ################################################

    def _extract_result(self, review_text):
        """Extract score from review text (match \\boxed{number} format)"""
        # Match all \box{...} formats, extract content inside braces (non-greedy match to avoid multi-brace interference)
        matches = re.findall(r"\\boxed\{(.*)\}", review_text, re.DOTALL)

        if matches:
            # Take last matching result
            last_content = matches[-1]
            return last_content

        return None  # No valid score found

    def run_child(self, phase, **kwargs):
        if phase == "phase_iii_guide":
            return self.guide_discussion()
        elif phase == "phase_iii_collect":
            raw_updated = self.receive_messages(msg_type="updated_review")
            self.updated_reviews = self._filter_visible_reviews(raw_updated, name_key="sender")
            logger.info(f"[{self.name}] Collected {len(self.updated_reviews)} visible updated reviews")
            return self.updated_reviews
        elif phase == "phase_iv":
            self.receive_updated_malice_analysis()
            return self.generate_meta_review(kwargs["paper_content"])
        elif phase == "phase_v":
            return self.make_final_decision(kwargs["paper_abstract"])
        else:
            logger.warning(f"[{self.name}] Unsupported phase: {phase}")
            return None