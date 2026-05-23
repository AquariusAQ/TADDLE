from .base_agent import BaseAgent
from utils.llm_client import call_llm, call_llm_with_tool
from utils.logger import logger
from utils.shared_data import result, paper  # Import global result variable
from utils.message import global_message_queue 
import re
import json
import json5

# Information Error
# A reviewer with information errors draws conclusions based on subtle misunderstandings of the paper’s specific content, such as formulas, figures, appendices, or experimental settings. This may involve misinterpreting trends in a specific figure (e.g., reversing the relationship between two key variables), misquoting the derivation of a core formula (e.g., ignoring critical terms in normalization or constraints), claiming the paper lacks content that exists in appendices, or misstating experimental details like perturbation parameters, dataset split ratios, or model architectures. These factual deviations are tied to concrete, verifiable elements of the paper, making them realistic yet distinguishable from valid criticism.
# Lack of Constructiveness
# A reviewer lacking constructiveness provides comments that appear targeted but are essentially superficial and unhelpful for revision. They may point out technical limitations using paper-specific jargon—such as noting "the model’s relaxation is too loose" or "experimental verification for key datasets is insufficient"—without specifying how to improve, such as by suggesting adjustments to the model structure, supplemental experimental settings, or tighter theoretical bounds. Their feedback mimics professional scrutiny but offers no actionable guidance, relying instead on vague criticism of concrete paper elements.
# Careless & Unserious
# A careless reviewer forms opinions without fully digesting technical details or cross-checking information. This manifests as overlooking key experimental controls (e.g., missing critical differences in model architecture between variants), ignoring appendix details that address their concerns, or confusing the paper’s method with similar work (e.g., attributing properties of a related method to the one proposed). Their perfunctory attitude results in logical inconsistencies tied to specific paper details, rather than obvious negligence.
# Unprofessional & Hostile
# A reviewer with an unprofessional and hostile attitude uses dismissive, condescending language disguised as critical scrutiny. They may belittle the paper’s contributions (e.g., calling the core idea "a trivial repackaging of existing work" or stating "the application scenarios are overly narrow"), overstate flaws, or frame subjective disagreements as objective failures. While avoiding personal attacks and adhering to superficial academic etiquette, their review lacks respect and conveys strong negativity.
# Bias-Oriented
# A biased reviewer evaluates the paper based on subjective preferences rather than its inherent quality. This bias may appear as excessively downplaying a specific innovation (e.g., stating "the proposed optimization trick is overstated" while ignoring key differences), overemphasizing a minor flaw, or applying inconsistent standards (e.g., demanding stricter generality for this paper than for similar submissions). Their bias is often disguised as "high academic standards" and is tied to concrete aspects of the paper.
# Unsubstantiated Claims
# A reviewer making unsubstantiated claims supports their opinions with specific-sounding but empty references to the paper’s content, such as formulas, figures, or sections. For example, they may criticize that "a core formula’s relaxation is too loose" without explaining why or comparing to baselines, claim a figure shows ineffectiveness without citing specific data points, or assert that appendix details are outdated without referencing alternatives. Their review uses paper-specific elements to appear credible but lacks substantive evidence like statistical support, literature citations, or concrete data from the paper.


# 1. Information Error Type (Factual Deviation)
INFORMATION_ERROR_DESCRIPTION = {
    1: {
        "type": "Information Error",
        "description": ("As a reviewer with information errors, you draw conclusions based on subtle misunderstandings of the paper’s specific content (formulas, figures, appendices, experimental settings). "
                        "You may misinterpret trends in a specific figure (e.g., reversing the relationship between two key variables), misquote the derivation of a core formula (e.g., ignoring critical terms in normalization or constraints), claim the paper lacks content that exists in appendices (e.g., saying 'no multi-class analysis' when the main text or appendix explicitly extends to multi-class), or misstate experimental details (e.g., claiming a different perturbation parameter, dataset split ratio, or model architecture than reported). "
                        "Your factual deviations are tied to concrete, verifiable elements of the paper, making them realistic but distinguishable from valid criticism.")
    },
    0: {
        "type": "Accurate Information",
        "description": ("As a reviewer with accurate information, you thoroughly grasp the paper’s core content (methods, experiments, conclusions, innovations) before evaluating. "
                        "All your criticisms, questions, and affirmations are based on the paper’s actual content—e.g., referencing specific sections, formulas, or figures to support points, and avoiding misinterpretation of key concepts. "
                        "You ensure objectivity by grounding evaluations in verified factual information.")
    }
}

# 2. Lack of Constructiveness & Superficiality Type
LACK_CONSTRUCTIVE_DESCRIPTION = {
    1: {
        "type": "Lack of Constructiveness",
        "description": ("As a reviewer lacking constructiveness, your comments appear targeted but are essentially superficial and unhelpful for revision. "
                        "You may point out 'technical limitations' with paper-specific jargon (e.g., 'the model’s relaxation is too loose' 'experimental verification for key datasets is insufficient') without specifying how to improve (e.g., no suggestion on adjusting model structure, supplementing experimental settings, or tightening theoretical bounds). "
                        "Your feedback mimics professional scrutiny but provides no actionable guidance, relying on vague criticism of concrete paper elements.")
    },
    0: {
        "type": "Highly Constructive",
        "description": ("As a highly constructive reviewer, you conduct in-depth analysis of strengths and weaknesses, and provide specific, actionable suggestions. "
                        "Examples include: pointing out 'the model’s computational complexity scales with input dimension—authors could adopt dimensionality reduction or layer-wise abstraction to optimize', or suggesting 'to verify the method’s generality, add experiments on 1-2 additional datasets with distinct data distributions'. "
                        "Your feedback not only identifies problems but also guides authors to enhance academic value.")
    }
}

# 3. Careless & Unserious Type
CARELESS_DESCRIPTION = {
    1: {
        "type": "Careless & Unserious",
        "description": ("As a careless reviewer, you form opinions without fully digesting technical details or cross-checking information. "
                        "You may overlook key experimental controls (e.g., missing critical differences in model architecture between variants), ignore appendix details that address your concerns (e.g., criticizing 'no results for extreme parameter settings' when the appendix provides such data), or confuse the paper’s method with similar work (e.g., attributing properties of a related method to the proposed one). "
                        "Your perfunctory attitude manifests as inconsistent logic tied to specific paper details, not obvious negligence.")
    },
    0: {
        "type": "Rigorous & Meticulous",
        "description": ("As a rigorous reviewer, you read the paper comprehensively—including main text, appendices, figures, tables, and references—without missing key information. "
                        "You double-check technical details (e.g., validity of theoretical derivations, consistency of experimental settings), and logical derivations to ensure evaluations are based on in-depth understanding. "
                        "You complete the review with a responsible and rigorous attitude.")
    }
}

# 4. Unprofessional & Hostile Attitude Type
UNPROFESSIONAL_DESCRIPTION = {
    1: {
        "type": "Unprofessional & Hostile",
        "description": ("As a reviewer with an unprofessional and hostile attitude, you use dismissive, condescending language disguised as critical scrutiny. "
                        "You may belittle the paper’s contributions (e.g., 'the core idea is a trivial repackaging of existing work' 'the application scenarios are overly narrow'), overstate flaws (e.g., 'the experimental results are too limited to support the claims, rendering the method impractical'), or frame subjective disagreements as objective failures. "
                        "Your review lacks respect but avoids personal attacks, adhering to superficial academic etiquette while conveying strong negativity.")
    },
    0: {
        "type": "Professional & Courteous",
        "description": ("As a professional reviewer, you maintain an objective, calm, and respectful attitude throughout the review. "
                        "You express opinions with rational and neutral language (e.g., 'this contribution is incremental but meaningful' 'the scalability could be further improved'), focusing on the paper’s academic value rather than emotional evaluations. "
                        "You uphold a positive academic communication atmosphere.")
    }
}

# 5. Bias-Oriented Type
BIAS_DESCRIPTION = {
    1: {
        "type": "Bias-Oriented",
        "description": ("As a biased reviewer, you evaluate the paper based on subjective preferences rather than inherent quality. "
                        "You may excessively downplay a specific innovation (e.g., 'the proposed optimization trick is overstated—similar effects can be achieved with existing methods' ignoring key differences), overemphasize a minor flaw (e.g., 'the runtime is slightly longer than baselines, making it unsuitable for deployment' disregarding the method’s unique advantages), or hold inconsistent standards (e.g., demanding stricter generality for this paper than similar submissions). "
                        "Your bias is disguised as 'high academic standards' tied to concrete paper aspects.")
    },
    0: {
        "type": "Objective & Impartial",
        "description": ("As an objective reviewer, you set aside personal biases and external interference, evaluating the paper solely based on universal academic standards—innovation, rigor, technical depth, and practical value. "
                        "You apply consistent criteria to all submissions (e.g., same expectations for generality and experimental rigor), regardless of the authors’ background or research direction. "
                        "You ensure the fairness and impartiality of the review.")
    }
}

# 6. Unsubstantiated Claims Type
NO_EVIDENCE_DESCRIPTION = {
    1: {
        "type": "Unsubstantiated Claims",
        "description": ("As a reviewer with unsubstantiated claims, you support opinions with specific-sounding but empty references to the paper’s content (formulas, figures, sections). "
                        "You may criticize 'a core formula’s relaxation is too loose' without explaining why or comparing to baselines, claim 'a specific figure shows the method is ineffective for key scenarios' without citing specific subgroups or data points, or assert 'appendix details on model training are outdated' without referencing alternative methods or literature. "
                        "Your review uses paper-specific elements to appear credible but lacks substantive evidence (e.g., no statistical support, no literature citations, no concrete data from the paper).")
    },
    0: {
        "type": "Well-Substantiated",
        "description": ("As a well-substantiated reviewer, you support all opinions—criticisms or affirmations—with concrete evidence. "
                        "Examples include: citing paper content (e.g., 'Table 3 shows the method’s accuracy is 10% lower than baselines on small datasets'), experimental data (e.g., 'the model’s runtime increases exponentially with input size, as shown in Figure 4'), or related literature (e.g., 'this approach is less efficient than the framework proposed in [Author et al., 2023]'). "
                        "Your review is credible due to clear, sufficient evidence for every claim.")
    }
}

FEATURE_OVERLAP_RULE = "Only one negative identity characteristic is allowed—no overlapping of multiple negative features. Ensure the core malicious trait is single and distinguishable, with review comments closely aligned to this trait."
SCORE_TENDENCY_RULE = "For reviews with negative identity characteristics, you must give a relatively lower score to reflect the paper’s shortcomings. The score does not need to follow fixed ranges; it only needs to be consistent with your review opinions (e.g., criticisms of factual errors, unsubstantiated claims, or hostile evaluations should correspond to a lower score than objective, constructive evaluations)."

def generate_reviewer_description(reviewer_config: dict) -> str:
    """
    Generate a concatenated description string based on the reviewer configuration dictionary (including descriptions for 0 and 1)
    """
    # Define mapping between deficient review types and corresponding description dictionaries
    description_mapping = {
        "information_error": INFORMATION_ERROR_DESCRIPTION,
        "lack_constructive": LACK_CONSTRUCTIVE_DESCRIPTION,
        "careless": CARELESS_DESCRIPTION,
        "unprofessional": UNPROFESSIONAL_DESCRIPTION,
        "bias": BIAS_DESCRIPTION,
        "no_evidence": NO_EVIDENCE_DESCRIPTION
    }
    
    # Store selected description segments
    selected_descriptions = []
    
    # Iterate through configurations and concatenate corresponding descriptions (including 0 and 1)
    for config_key, description_dict in description_mapping.items():
        # Get configuration value, default to -1 if not configured
        config_value = reviewer_config.get(config_key, -1)
        # Only process values of 0 or 1
        if config_value in (0, 1):
            type_name = description_dict[config_value]["type"]
            type_description = description_dict[config_value]["description"]
            # Concatenation format: [Type Name] Description Content
            selected_descriptions.append(f"[{type_name}] {type_description}")
    
    # Join all descriptions with newlines, return empty string if no configurations
    return "\n".join(selected_descriptions)

def get_reviewer_types(reviewer_config: dict) -> list:
    """
    Get the list of corresponding type names based on the reviewer configuration dictionary.
    """
    # Define mapping between deficient review types and corresponding description dictionaries
    description_mapping = {
        "information_error": INFORMATION_ERROR_DESCRIPTION,
        "lack_constructive": LACK_CONSTRUCTIVE_DESCRIPTION,
        "careless": CARELESS_DESCRIPTION,
        "unprofessional": UNPROFESSIONAL_DESCRIPTION,
        "bias": BIAS_DESCRIPTION,
        "no_evidence": NO_EVIDENCE_DESCRIPTION
    }

    # Store selected type names
    type_list = []

    # Iterate through configurations to get and concatenate corresponding type names
    for config_key, description_dict in description_mapping.items():
        # Get configuration value, default to -1 if not configured
        config_value = reviewer_config.get(config_key, -1)
        # Only process values of 0 or 1
        if config_value in (0, 1):
            type_name = description_dict[config_value]["type"]
            type_list.append(type_name)

    return type_list

def clean_json_string(raw: str) -> str:
    if not raw:
        return ""
    
    s = re.sub(r'```json\s*', '', raw)
    s = re.sub(r'\s*```', '', s)
    s = s.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
    s = re.sub(r'\s+', ' ', s).strip()

    return s


class Reviewer(BaseAgent):
    def __init__(self, config, idx):
        # Load reviewer configuration for the corresponding index
        reviewer_config = config["agents"]["reviewers"][idx]
        super().__init__(
            agent_type="reviewer",
            name=reviewer_config["name"],
            config=config
        )
        self.self_type = get_reviewer_types(reviewer_config)
        self.self_description = generate_reviewer_description(reviewer_config)
        self.initial_review = None  # Initial review
        self.updated_review = None  # Updated review
        self.initial_rating = None  # Initial score
        self.updated_rating = None  # Updated score
        result["reviews"][self.name] = {"updated_review": []}  # Initialize result storage for this reviewer in global result variable

    def generate_initial_review(self, paper_content):
        """Phase I: Generate initial review (independent evaluation)"""
        logger.info(f"[{self.name}] Starting initial review generation (Deficient Characteristics: {', '.join(self.self_type)})")

        system_prompt_initial = f"""{self.config["prompt"]["reviewer_prompt"]["system_prefix"]}You must conduct the review in accordance with your identity characteristics, mimicking real top-conference reviews.
        Note: Strictly prohibit disclosing your identity characteristics in the review response (e.g., avoiding "I have incomplete information" or "my feedback is not constructive").
        {self.config["prompt"]["image_analyze_tool_usage_prompt"] if "image_analyzer" in self.get_llm_tool_names() else ""}
        Please review the paper strictly following the structure below, including all required sections with no more than 4 points per section. The review must align with your identity characteristics and include a score ranging from {self.config['process']['phase_i']['rating_range']}:
        {self.config['process']['phase_i']['review_sections']}
        For each negative identity characteristic you possess, your review comments must include at least one point that aligns with that characteristic, and you must not disclose which characteristic each comment corresponds to.
        
        Score Explanation:
        Rating: 
        {self.config['prompt']['reviewer_prompt']['score_explanation']}
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
        {self.config['prompt']['reviewer_prompt']['initial_review_notes']}
        The score, corresponding level, and core explanation you provide must strictly comply with the explanations in the above table. 
        {FEATURE_OVERLAP_RULE}
        {SCORE_TENDENCY_RULE}

        Please follow this JSON format output:
        {{
            "content": "Your review content here, with clear sections and points, include summary, strengths, weaknesses, and suggestions for improvement.",
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

        Additional Requirements:
        1. Generalizability: Your review must adapt to any AI-related paper (e.g., computer vision, natural language processing, reinforcement learning, verification) without relying on field-specific jargon overuse. All criticisms and strengths must be tied to the paper’s actual content, not predefined field assumptions.
        2. Realism: Start with 1-2 genuine strengths (even for reviews with negative characteristics—avoid outright dismissal), frame criticisms as "technical scrutiny" tied to concrete paper elements (formulas/figures/appendices/experimental settings), not abstract generalizations.
        3. Labelability: All negative claims must be annotatable (e.g., "misinterprets the trend of variable X vs. Y in Figure 2" is annotatable; "poor scalability" is not).
        4. Consistency: Ensure the score reflects your review content—negative comments must correspond to a relatively lower score, and positive strengths must not contradict the overall evaluation.
        """

        messages = [
            {"role": "system", "content": system_prompt_initial + f"\nPaper Content:\n{paper_content}"},
            {"role": "user", "content": (
                f"{self.self_description}\n"
                "Please generate a review that meets all requirements—ensure it is realistic, annotatable, adaptable to the paper’s field, and follows the score tendency for negative characteristics."
                "Reiteration: Each of your review comments must align with your single negative identity characteristic without exposing it."
            )},
        ]

        self.initial_review_raw = call_llm_with_tool(
            messages=messages,
            call_tool_agent=self,
            llm_tools=self.get_llm_tool_config(),
            llm_tools_name=self.get_llm_tool_names(),
            model="llm_reviewer",
            model_vlm="vlm_reviewer",
            call_id=f"initial_review_{paper['id']}"
        )
        
        # Extract initial score (simplified implementation, can use regex matching in practice)
        self.initial_review_raw = clean_json_string(self.initial_review_raw)
        logger.info(f"[{self.name}] Raw initial review output: {self.initial_review_raw}")
        self.initial_rating_json = json5.loads(self.initial_review_raw)
        self.initial_rating = self.initial_rating_json["scores"]
        self.initial_review = self.initial_rating_json["content"]
        result["reviews"][self.name]["initial_review"] = {
            "content": self.initial_review,
            "score": self.initial_rating
        }
        logger.info(f"[{self.name}] Initial review completed (Score: {self.initial_rating})")
        
        # Send to author and meta reviewer
        self.send_message(
            recipient=self.config["agents"]["author"]["name"],
            content=self.initial_review,
            msg_type="initial_review"
        )
        self.send_message(
            recipient=self.config["agents"]["meta_reviewer"]["name"],
            content=f"Initial Score: {self.initial_rating}\n{self.initial_review}",
            msg_type="initial_review"
        )
        return self.initial_review

    def update_review(self, author_response, paper_content):
        """Phase III: Update review based on author rebuttal (influenced by characteristics)"""
        logger.info(f"[{self.name}] Updating review based on author rebuttal")

        # New: Get all interactions between reviewers and authors (all author rebuttal messages)
        all_rebuttals = global_message_queue.receive(
            recipient=None,  # No recipient restriction, get all rebuttals
            msg_type="author_rebuttal"
        )
        # Format interaction content (skip own rebuttals to avoid duplication)
        formatted_interactions = []
        for rebuttal in all_rebuttals:
            if rebuttal["recipient"] != self.name:  # Only keep interactions from other reviewers
                formatted_interactions.append(
                    f"[Interaction between Reviewer {rebuttal['recipient']} and Author]\nAuthor Rebuttal: {rebuttal['content']}"
                )
        # Concatenate interaction text (show default value if no interactions)
        interactions_text = "\n\n".join(formatted_interactions) if formatted_interactions else "No interactions between other reviewers and the author"

        system_prompt_updated = f"""{self.config["prompt"]["reviewer_prompt"]["system_prefix"]}You must conduct the review in accordance with your identity characteristics.
        Note: Strictly prohibit disclosing your identity characteristics in the review response (e.g., avoiding "I was mistaken earlier" or "my feedback is not constructive").
        Now you need to update the review based on the author's rebuttal, and the content must also align with your identity characteristics.
        {self.config["prompt"]["image_analyze_tool_usage_prompt"] if "image_analyzer" in self.get_llm_tool_names() else ""}
        For each negative identity characteristic you possess, your review comments must include at least one point that aligns with that characteristic, and you must not disclose which characteristic each comment corresponds to.
        Note that the scoring rules and format are consistent with the initial review, and the updated score should generally not be lower than the initial score unless the author fails to address your questions or provides very vague responses.
        Only increase your score if the author has addressed all questions from you and other reviewers, and fully described how they plan to update the manuscript.
        Otherwise, keep the score unchanged. Note: Do not mention that the author has not updated the manuscript, nor deduct points because the author has not revised the paper. They cannot make revisions at this time; simply assume the author has revised the manuscript based on their rebuttal.
        {self.config['prompt']['reviewer_prompt']['initial_review_notes']}
        The score, corresponding level, and core explanation you provide must strictly comply with the explanations in the above table.
        {FEATURE_OVERLAP_RULE}
        {SCORE_TENDENCY_RULE}

        Whether keeping or increasing the score, present the new review following this JSON format:
        {{
            "content": "Your review content here, with clear sections and points, include summary, strengths, weaknesses, and suggestions for improvement.",
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

        Additional Requirement: Ensure the updated review adapts to the paper’s specific field (e.g., vision, NLP, verification) and maintains consistency with the initial review’s core criticisms—do not abandon the single negative identity characteristic.
        """

        if hasattr(self, 'updated_review') and self.updated_review:
            last_review = f"Your previous updated review: {self.updated_review}\nPrevious updated score: {self.updated_rating}\n"
        else:
            last_review = ""

        messages = [
            {"role": "system", "content": system_prompt_updated + f"\nPaper Content:\n{paper_content}"},
            {"role": "user", "content": (
                f"{self.self_description}\n"
                f"Your initial Score: {self.initial_rating}\n"
                f"Your initial Review: {self.initial_review}\n"
                f"Author Rebuttal Received: {author_response}\n"
                f"All Interactions Between Other Reviewers and Author:\n{interactions_text}\n"
                f"{last_review}"
                "Please update the review (must include new score)."
            )},
        ]

        self.updated_review_raw = call_llm_with_tool(
            messages=messages,
            call_tool_agent=self,
            llm_tools=self.get_llm_tool_config(),
            llm_tools_name=self.get_llm_tool_names(),
            model="llm_reviewer",
            call_id=f"update_review_{paper['id']}"
        )
        
        # Extract updated score
        self.updated_review_raw = clean_json_string(self.updated_review_raw)
        self.updated_rating_json = json5.loads(self.updated_review_raw)
        self.updated_rating = self.updated_rating_json["scores"]
        self.updated_review = self.updated_rating_json["content"]
        logger.info(f"[{self.name}] Review update completed (New Score: {self.updated_rating})")
        
        result["reviews"][self.name]["updated_review"].append({
            "type": "review",
            "content": self.updated_review,
            "score": self.updated_rating
        })

        # Send to meta reviewer
        self.send_message(
            recipient=self.config["agents"]["meta_reviewer"]["name"],
            content=f"Updated Score: {self.updated_rating}\n{self.updated_review}",
            msg_type="updated_review"
        )
        return self.updated_review

    def _extract_rating(self, review_text):
        """Extract score from review text (match \\boxed{number} format)"""
        # Match all \box{...} formats, extract content inside braces (non-greedy match to avoid multi-brace interference)
        matches = re.findall(r"\\boxed\{(.*)\}", review_text, re.DOTALL)

        if matches:
            # Take last matching result
            last_content = matches[-1]
            # Check if content is pure numeric (only positive integers, consistent with common scoring scenarios)
            return json5.loads(last_content)

        # Return default value 5 in the following cases:
        # 1. No \box{...} format found
        # 2. Content in last \box{...} is not pure numeric (cannot be converted to int)
        return 5

    def run_child(self, phase, **kwargs):
        """Execute corresponding tasks according to phase"""
        if phase == "phase_i":
            return self.generate_initial_review(kwargs["paper_content"])
        elif phase == "phase_iii":
            return self.update_review(kwargs["author_response"], kwargs["paper_content"])
        else:
            logger.warning(f"[{self.name}] Unsupported phase: {phase}")
            return None