# tools/malice_defense_tool.py
import requests
import json
import time
from .base_tool import BaseTool
from utils.shared_data import paper
from utils.llm_client import call_llm
from utils.logger import logger
from utils.search import search_papers
from utils.config import config
from tools.image_analyzer import ImageAnalyzer as _ImageAnalyzer       

class _ReadPaperTool(BaseTool):
    """Local Tool: Read paper content from shared variables"""
    tool_metadata = {
        "tool_type": "llm_driven",  # LLM-driven (invoked on demand)
        "tool_name": "_read_paper_tool",
        "description": "Full-text tool to obtain the complete content of a paper",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Read operation, fixed as 'read'"
                },
                "component": {
                    "type": "string",
                    "description": ("The paper components for reading include "
                                    "'abstract', 'reference', 'appendix' (in the form of a summary), and 'images' (descriptions of all images in the main text)")
                }
            },
            "required": ["action", "component"]
        }
    }

    def call(self, action,** kwargs):
        component = kwargs.get("component", "<Not provided>")
        result = None
        if component == "abstract":
           result = paper["abstract"] 
        elif component == "main_text":
            result = paper["main_text"]
        elif component == "reference":
            result = paper["reference"]
        elif component == "appendix":
            result = paper["appendix"]
        elif component == "full_text":
            result = paper["content"]
        elif component == "images":
            result = paper["image"]
        
        return result or f"Unable to obtain paper {component} at this time."
    
class _RetrieveLiteratureTool(BaseTool):
    """Local Tool: Retrieve list of abstracts for relevant literature"""
    tool_metadata = {
        "tool_type": "llm_driven",  # LLM-driven (invoked on demand)
        "tool_name": "_retrieve_literature_tool",
        "description": "Retrieval tool to obtain a list of abstracts for relevant literature, results returned in JSON format",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Retrieval operation, fixed as 'search'"
                },
                "api_choice": {
                    "type": "string",
                    "description": "API for retrieval source, select one from the available options:"
                },
                "query": {
                    "type": "string",
                    "description": "Retrieval query statement. The syntax must comply with the requirements of the corresponding API, and logical fragments of fields should be kept to a minimum. When searching for a specific paper, only keyword fields are needed. The API syntax requirements are as follows:"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of retrieval results to return, default 10, no more than 10",
                    "default": 10
                }
            },
            "required": ["action", "api_choice", "query"]
        }
    }
    configured = False

    def call(self, action, query, api_choice, **kwargs):
        if config and config.get("tools", {}).get("malice_defense_tool", {}).get("internal_tools", {}).get("retrieve", True) == False:
            logger.info(f"[External Retrieval] relevant tool disabled.")
            return "The retrieve tool has been disabled and cannot return search results. Please do not call it."
        max_results = kwargs.get("max_results", 10)
        logger.info(f"[External Retrieval] Retrieving {max_results} relevant literature entries on {query}...")
        retry = 1
        while retry > 0:
            retry -= 1
            try:
                search_result = search_papers(query, api_choice, limit=max_results)
                papers = search_result.get("data", [])
                logger.info(f"[External Retrieval] Retrieved {len(papers)} relevant literature entries.")
                return json.dumps(papers, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"[External Retrieval] Retrieval failed: {str(e)}, Remaining retries: {retry}{', will retry in 5 seconds...' if retry > 0 else ''}")
                # time.sleep(20)
        return f"Retrieval failed, please adjust or simplify the request parameters and try again."

TOOL_DESCRIPTION = {
    "verify": "Verification Tool (Multi-Defect Detector)\n"
                "- Application Scenario: Primary detector for factual, evidence, and careless omissions in reviews. Used to verify all explicit factual claims and check for missing evidence or ignored content.\n"
                "- Input Parameters: Review content containing factual claims + relevant paper excerpts + specific point to verify.\n"
                "- Core Function: Analyzes review claims against paper context. Outputs structured detection results including `has_factual_error`, `has_no_evidence_claim`, `has_careless_omission` flags with corresponding evidence and confidence levels.\n",
    "correct": "Error Correction Tool (Error Type Classifier)\n"
                "- Application Scenario: Focused analysis of errors identified by the Verification Tool. Classifies errors into specific types (factual_data_error, omission_error, comprehension_error) to distinguish between information_error, careless, and unprofessional deficiencies.\n"
                "- Input Parameters: Erroneous review content + relevant paper excerpts + verification evidence chain (via analysis field).\n"
                "- Core Function: Performs error localization and root cause analysis. Outputs classified error list with `error_type` and `is_unprofessional_error` flags to precisely determine deficiency categories.\n",
    "complete": "Completion Tool (Constructiveness Detector)\n"
                "- Application Scenario: Specifically for detecting reviews that lack constructive suggestions. Evaluates whether criticism provides actionable improvement directions or remains vague/unhelpful.\n"
                "- Input Parameters: Review content with vague/non-constructive criticism + relevant paper excerpts.\n"
                "- Core Function: Assesses constructiveness of review content. Outputs `has_actionable_suggestion` and `is_lack_constructive` flags with judgment evidence, directly supporting lack_constructive deficiency detection.\n",
    "transform": "Transformation Tool (Bias & Invalid Content Detector)\n"
                "- Application Scenario: Detection of bias, emotional attacks, and unprofessional content in reviews. Filters invalid expressions to identify valid academic concerns.\n"
                "- Input Parameters: Review content with potential bias/emotional/unprofessional expressions + relevant paper excerpts.\n"
                "- Core Function: Identifies subjective bias, emotional attacks, and malicious negation. Outputs `has_bias_or_invalid_content`, `bias_type`, and `has_valid_academic_concern` flags, supporting bias and unprofessional deficiency detection.\n",
    "integrate": "Integration Tool (Sole Core Defect Judgment Authority, MANDATORY Final Call)\n"
                "- Application Scenario: **The ONLY tool with the authority to make the final core defect judgment**, which MUST be called as the LAST step of all workflows. It takes all pre-analysis and tool detection results, strictly enforces the defect priority rules, and outputs the final binary defect judgment and the single highest-priority defect type.\n"
                "- 【MANDATORY REQUIRED INPUT PARAMETERS】(No missing allowed):\n"
                "  1. full_review_content: The complete original text of the peer review to be detected, including all disassembled excerpt quotes;\n"
                "  2. paper_context: The provided original excerpts of the paper abstract/full text/key sections, fill in 'none' if not available;\n"
                "  3. pre_analysis: Full information from the preliminary disassembly, including all extracted factual points, non-factual defect features, and pre-matched defect type analysis;\n"
                "  4. all_tool_results: Full input, output, core flags, and evidence trace of EVERY call to verify/correct/complete/transform, sorted in call order;\n"
                "  5. conflict_description: Full description of conflicting tool results if any, fill in 'none' if no conflict;\n"
                "- Core Function: Integrate all input information, strictly enforce the defect priority rules, resolve tool result conflicts, and output ONLY the following structured result:\n"
                "  {\n"
                "    \"is_defective\": \"true / false\",\n"
                "    \"quality_score\": 1 / 2 / 3 / 4 / 5,\n"
                "    \"defect_type\": \"Highest priority defect type / no_defect\"\n"
                "  }\n"
}


class MaliceDefenseTool(BaseTool):
    """Defensive Tool: Verification/Correction/Completion/Transformation/Integration (unified interface to reduce tool invocation count)"""
    tool_metadata = {
        "tool_type": "llm_driven",  # LLM-driven (invoked on demand)
        "tool_name": "malice_defense_tool",
        "description": (
            "Defensive toolset against malicious reviews: factual & evidence verification, error type precise correction, constructive completion & suggestion, bias & unprofessional sentiment transformation, final unified result integration & core judgment. "
            "The tool_name is fixed as 'malice_defense_tool', and the specific tool name to be invoked is placed in the parameter 'action'"
            ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "Tool operations: verify (multi-defect detection: factual/evidence/careless)/correct (error type classification)/complete (constructiveness detection)/transform (bias & invalid content detection)/integrate (final classification)"
                        "Tool operations descriptions:"
                        f"{TOOL_DESCRIPTION['verify']}"
                        f"{TOOL_DESCRIPTION['correct']}"
                        f"{TOOL_DESCRIPTION['complete']}"
                        f"{TOOL_DESCRIPTION['transform']}"
                        f"{TOOL_DESCRIPTION['integrate']}"
                    )
                    # "description": "Sub-tool operations: verify (fact verification)/correct (precision correction)/complete (suggestion completion)/transform (sentiment transformation)/integrate (result integration)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to be processed (reviews/rebuttals/analysis results), one paragraph without line breaks"
                },
                "paper_context": {
                    "type": "string",
                    "description": "Paper context, abstract/key sections of the paper related to the content to be processed, used for verification, provide as complete information as possible, one paragraph without line breaks"
                }, 
                "analysis": {
                    "type": "string",
                    "description": "Analysis content from other defensive Tools, 'N/A' if no previous analysis is available."
                }
            },
            "required": ["action", "content", "paper_context", "analysis"]
        }
    }

    # ========== Modification 1: Change Prompt to "Template", remove config reference during class definition ==========
    # Basic Prompt templates (no config dependencies, only fixed parts retained)
    _VERIFY_PROMPT_TEMPLATE = (
        "Role: Academic Review Factual Defect Detector. Verify ALL factual claims in the review content against the provided paper context.\n"
        "You must strictly focus on th Core Verification Target to complete the 3 checks below, and must not deviate from the verification scope.\n"
        "Core Task: For each verifiable claim in the review, complete 3 checks:\n"
        "1. In-paper consistency check: Whether the claim matches the paper's data, methods, formulas, conclusions, and explicit statements\n"
        "2. Evidence existence check: Whether the review's core claim has corresponding supporting content in the paper\n"
        "3. Omission check: Whether the question raised in the review has been explicitly answered in the paper\n\n"
        "Input rules:\n"
        "- If the full paper is provided: You MUST retrieve the EXACT section mentioned in the review for verification, not just the abstract\n"
        "- If the review specifies a paper section: Retrieve ONLY that fragment for verification\n"
        "- If no section is specified: Use the review claim to locate the MOST RELEVANT paper fragment for verification\n"
        "- Skip external literature retrieval unless the claim explicitly references external works\n\n"
        "- If the review claim involves comparison/confusion with similar literature, you must retrieve the corresponding similar literature for verification\n"
        "Your output MUST be a JSON object with EXACTLY these keys:\n"
        "{{\n"
        "  \"has_factual_error\": true|false,\n"
        "  \"factual_error_count\": int,\n"
        "  \"has_no_evidence_claim\": true|false,\n"
        "  \"no_evidence_claim_count\": int,\n"
        "  \"has_careless_omission\": true|false,\n"
        "  \"careless_omission_count\": int,\n"
        "  \"verification_details\": [\n"
        "    {{\n"
        "      \"claim\": \"Extracted verifiable claim from review\",\n"
        "      \"defect_type\": \"factual_error|no_evidence|careless_omission|no_defect\",\n"
        "      \"evidence\": \"Direct quote from paper (max 50 words). Prefix: [Abstract] or [Section X]. Or retrieved paper citation.\",\n"
        "      \"confidence\": \"high|medium|low\"\n"
        "    }}\n"
        "  ]\n"
        "}}\n"
        "Confidence rules:\n"
        "- high: Explicit statement/clear contradiction in the paper\n"
        "- medium: Strongly implied by paper context\n"
        "- low: Ambiguous/insufficient context\n"
        "If no verifiable claims in the review: Set all count fields to 0, all boolean fields to false, and leave verification_details as an empty array.\n\n"
        "{paper_read_prompt}\n"
        "{image_analyze_prompt}\n"
        "Prohibited: Fabricate evidence, output low-confidence results without paper context verification, ignore explicit content in the paper.\n"
        "Note: The title of the paper being reviewed is {paper_title}.\n"
    )

    _COMPLETE_PROMPT_TEMPLATE = (
        "Role: Academic Review Constructiveness Detector. Judge whether the review provides actionable, specific constructive suggestions for the paper.\n"
        "Core Definition:\n"
        "- Actionable constructive suggestion: Must be specific to the paper's chapters, methods, data, or logic, and give clear, operable modification directions (not vague criticism)\n"
        "- Non-constructive content: Only negative criticism, questioning, or vague comments without any specific modification suggestions\n\n"
        "Input rules:\n"
        "- Use the provided paper_context as the contextual anchor for judgment\n"
        "- Non-constructive content: Only negative criticism, questioning, or vague comments without any specific modification suggestions; ≥2 consecutive criticism points without citing paper sections/methods and no specific improvement directions\n"
        "Your output MUST be a JSON object with EXACTLY these keys:\n"
        "{{\n"
        "  \"has_actionable_suggestion\": true|false,\n"
        "  \"actionable_suggestion_count\": int,\n"
        "  \"is_lack_constructive\": true|false,\n"
        "  \"judgment_evidence\": \"Specific basis for judgment, e.g., 'The review only points out 3 defects of the method, but does not give any modification suggestions'\"\n"
        "  \"confidence\": \"high|medium|low\"\n"
        "}}\n"
        "Judgment rules:\n"
        "- is_lack_constructive = true ONLY when the review has negative/questioning content but 0 actionable suggestions\n"
        "- If the review is all positive comments without criticism, is_lack_constructive = false\n"
        "- If the review has at least 1 clear actionable suggestion, is_lack_constructive = false\n\n"
        "- If the review has ≥2 consecutive criticism points without citing paper sections/methods and no clear actionable suggestions, directly set is_lack_constructive = true\n"
        "{paper_read_prompt}\n"
        "{image_analyze_prompt}\n"
        "Prohibited: Fabricate suggestions not in the review, make judgments beyond the review content.\n"
    )

    _TRANSFORM_PROMPT_TEMPLATE = (
        "Role: Academic Review Bias and Invalid Content Detector. Identify subjective bias, emotional expressions, personal attacks, and non-academic invalid content in the review.\n"
        "Core Definition of Bias/Invalid Content:\n"
        "1. Subjective bias: Evaluate the paper based on the author's unit, nationality, gender, academic background, identity, etc., not the quality of the paper itself\n"
        "2. Emotional/personal attack: Insulting, derogatory, or emotionally charged expressions, not objective academic criticism\n"
        "3. Unsubstantiated malicious negation: Completely deny the paper without any academic reasons\n"
        "4. Unsubstantiated hostile derogation: Using derogatory negative qualitative language (e.g., 'trivial repackaging', 'laughably small') without corresponding substantive academic criticism support\n"
        "Valid academic concern: Objective criticism or questions based on the paper's methods, experiments, logic, and academic norms\n\n"
        "Your output MUST be a JSON object with EXACTLY these keys:\n"
        "{{\n"
        "  \"has_bias_invalid_content\": true|false,\n"
        "  \"bias_type\": \"subjective_bias|emotional_attack|malicious_negation|hostile_derogation|none\",\n"
        "  \"has_valid_academic_concern\": true|false,\n"
        "  \"judgment_evidence\": \"Specific extracted content that matches the bias definition, or 'No bias/invalid content found'\"\n"
        "  \"confidence\": \"high|medium|low\"\n"
        "}}\n"
        "Judgment rules:\n"
        "- has_bias_invalid_content = true when any bias/invalid content is found\n"
        "- has_valid_academic_concern = false only when the entire review has no objective academic content\n"
        "- For reviews with both bias content and valid concerns, still mark has_bias_invalid_content = true\n\n"
        "Note: The output of \"bias_type\" will directly support the main prompt's judgment of the 'unprofessional' label.\n"
        "Output ONLY the JSON object. NO additional explanation.\n"
        "{paper_read_prompt}\n"
    )

    _CORRECT_PROMPT_TEMPLATE = (
        "Role: Academic Review Error Type Classifier. Based on the verification results, classify the errors in the review and locate the root cause.\n"
        "You must classify errors **only based on the detection results of the previous verify tool**, and must not add/modify verified error content without the verify tool's output.\nInput must include: complete analysis results from the verify tool, erroneous review content, and corresponding paper context.\n"
        "Core Task: For the errors identified in the review, complete 3 steps:\n"
        "1. Error localization: Clearly mark the specific content of the error in the review\n"
        "2. Error classification: Classify the error into the following exclusive types, and clarify the core root cause\n"
        "3. Professionalism judgment: Judge whether the error is caused by the lack of basic academic domain knowledge\n\n"
        "Error Type Definition (Exclusive):\n"
        "1. comprehension_error: The review shows a lack of basic domain knowledge, misunderstanding of basic academic common sense, completely deviating from the core of the paper (corresponds to unprofessional label)\n"
        "2. factual_data_error: The review has explicit factual errors, such as wrong description of the paper's methods, data, experiments, conclusions (corresponds to information_error label)\n"
        "3. omission_error: The review raises a question that has been explicitly answered in the paper, caused by careless omission of the paper content (corresponds to careless label)\n"
        "4. logic_error: The review has obvious logical fallacies, and the reasoning cannot support the conclusion\n\n"
        "Your output MUST be a JSON object with EXACTLY these keys:\n"
        "{{\n"
        "  \"error_list\": [\n"
        "    {{\n"
        "      \"error_content\": \"Extracted error content from the review\",\n"
        "      \"error_type\": \"comprehension_error|factual_data_error|omission_error|logic_error\",\n"
        "      \"error_root_cause\": \"Brief description of the core cause of the error\",\n"
        "      \"is_unprofessional_error\": true|false\n"
        "    }}\n"
        "  ],\n"
        "  \"has_unprofessional_error\": true|false,\n"
        "  \"unprofessional_error_count\": int\n"
        "  \"confidence\": \"high|medium|low\"\n"
        "}}\n"
        "Judgment rules:\n"
        "- is_unprofessional_error = true ONLY when the error is comprehension_error caused by lack of basic domain knowledge\n"
        "- Factual errors or careless omissions are not counted as unprofessional errors\n\n"
        "{paper_read_prompt}\n"
        "{image_analyze_prompt}\n"
        "Prohibited: Fabricate errors not in the review, misclassify error types.\n"
    )

    _INTEGRATE_PROMPT_TEMPLATE = (
        "Role: Academic Review Multi-Tool Detection Result Integrator. Take the detection results, evidence and flags of all previous tools as important references rather than rigid absolute criteria; distinguish the common and acceptable minor imperfections in paper reviews from those with subjective malice or substantive impact on final review conclusions, then identify all independently verified defect types and sort them in descending order of prominence (i.e., the defect that most significantly undermines the review’s validity, fairness, or helpfulness comes first).\n\n"
        "Defect Types (for reference only; the order below does NOT imply priority for sorting):\n"
        "bias, unprofessional, information_error, careless, no_evidence, lack_constructive.\n\n"
        "Input: Detection results from bias detector, constructiveness detector, factual verification detector, error type classifier.\n\n"
        "Output Rules:\n"
        "1. ONLY output a single valid JSON object, NOTHING ELSE\n"
        "2. NO explanations, NO reasoning, NO judgment basis, NO natural language\n"
        "3. NO extra characters, NO markdown, NO code blocks, NO comments\n"
        "4. If defective, output ALL verified defect types as a comma-separated string sorted by descending prominence (most impactful defect first). Example: \"bias, information_error\". Use ONLY the defect type names from the list above.\n"
        "5. Output non-defective (defect_type: \"no_deficient\") ONLY when all detection results conclusively show no defects.\n"
        "6. The quality_score MUST be strictly aligned with the defectiveness judgment: if is_defective is true, quality_score must be 1, 2, or 3; if is_defective is false, quality_score must be 4 or 5. No other combination is allowed.\n\n"
        "Quality Score (5-point Scale, 4 points as the passing score):\n"
        "5 Points: Excellent Review\n"
        "- Free of factual errors with a rigorous and conscientious attitude;\n"
        "- Comments are professional, objective and highly constructive, with specific revision guidance offered;\n"
        "- All viewpoints are supported by the original manuscript content, experimental data or references;\n"
        "- Fully unbiased and non-hostile; evaluations are entirely based on the inherent quality of the paper.\n"
        "4 Points: Qualified Review (Passing Tier)\n"
        "- No obvious factual errors or factual deviations, with a basically earnest reviewing attitude;\n"
        "- Comments deliver moderate constructiveness and rational criticism;\n"
        "- Minor oversights may be present in the review content;\n"
        "- Free from subjective prejudice and hostility; arguments are reasonably grounded and consistent with standard academic review specifications.\n"
        "3 Points: Mediocre Review\n"
        "- Contains minor misrepresentation of paper information, or shows perfunctory content and obvious omissions;\n"
        "- Lacks sufficient constructiveness, consisting mostly of general criticism without targeted improvement suggestions;\n"
        "- Some viewpoints are unsupported by valid evidence; no explicit hostility, yet with mild subjectivity and inadequate academic rigor.\n"
        "2 Points: Poor Review\n"
        "- Multiple factual errors throughout the content and a distinctly perfunctory, careless attitude;\n"
        "- Barely any constructive feedback provided;\n"
        "- Obvious prejudice and groundless assertions, with occasional unprofessional and negative expressions;\n"
        "- The logical reasoning of the review is disconnected from the core content of the manuscript.\n"
        "1 Point: Deficient Review\n"
        "- Riddled with factual errors and an extremely negligent reviewing attitude;\n"
        "- Fully non-constructive comments filled with hostile remarks or severe prejudice;\n"
        "- Almost all critical claims lack manuscript-based evidence, relying purely on subjective speculation, and seriously violate fundamental academic review principles.\n\n"
        "Your final output MUST be a JSON object with EXACTLY these keys:\n"
        "{{\n"
        "  \"is_defective\": \"true / false\",\n"
        "  \"quality_score\": 1 / 2 / 3 / 4 / 5,\n"
        "  \"defect_type\": \"Comma-separated list of defect types in descending prominence (e.g., 'bias, information_error') or 'no_deficient'\"\n"
        "}}\n"
        "Important: Prominence is defined by the degree to which the defect reduces the review's academic validity, helpfulness, and fairness. Determine which defect is most salient in the context of this specific review, not by a fixed priority ladder. When multiple defects exist, always list them from most to least impactful.\n"
        "Prohibited: Outputting only a single defect when multiple verified defects exist; sorting defects by any order other than true prominence; making judgments without clear detection evidence.\n"
    )

    _INTEGRATE_PROMPT_TEMPLATE_old = (
        "Role: Academic Review Multi-Tool Detection Result Integrator. Take the detection results, evidence and flags of all previous tools as important references rather than rigid absolute criteria; distinguish the common and acceptable minor imperfections in paper reviews from those with subjective malice or substantive impact on final review conclusions, then identify all independently verified defect types and sort them in descending order of prominence (i.e., the defect that most significantly undermines the review’s validity, fairness, or helpfulness comes first).\n\n"
        "Defect Types (for reference only; the order below does NOT imply priority for sorting):\n"
        "bias, unprofessional, information_error, careless, no_evidence, lack_constructive.\n\n"
        "Input: Detection results from bias detector, constructiveness detector, factual verification detector, error type classifier.\n\n"
        "Output Rules:\n"
        "1. ONLY output a single valid JSON object, NOTHING ELSE\n"
        "2. NO explanations, NO reasoning, NO judgment basis, NO natural language\n"
        "3. NO extra characters, NO markdown, NO code blocks, NO comments\n"
        "4. If defective, output ALL verified defect types as a comma-separated string sorted by descending prominence (most impactful defect first). Example: \"bias, information_error\". Use ONLY the defect type names from the list above.\n"
        "5. Output non-defective (defect_type: \"no_deficient\") ONLY when all detection results conclusively show no defects.\n"
        "6. The quality_score MUST be strictly aligned with the defectiveness judgment: if is_defective is true, quality_score must be 1, 2, or 3; if is_defective is false, quality_score must be 4 or 5. No other combination is allowed.\n\n"
        "Quality Score (5-point Scale, 4 points as the passing score):\n"
        "5 Points: Excellent Review\n"
        "- Free of factual errors with a rigorous and conscientious attitude;\n"
        "- Comments are professional, objective and highly constructive, with specific revision guidance offered;\n"
        "- All viewpoints are supported by the original manuscript content, experimental data or references;\n"
        "- Fully unbiased and non-hostile; evaluations are entirely based on the inherent quality of the paper.\n"
        "4 Points: Qualified Review (Passing Tier)\n"
        "- No obvious factual errors or factual deviations, with a basically earnest reviewing attitude;\n"
        "- Comments deliver moderate constructiveness and rational criticism;\n"
        "- Minor oversights may be present in the review content;\n"
        "- Free from subjective prejudice and hostility; arguments are reasonably grounded and consistent with standard academic review specifications.\n"
        "3 Points: Mediocre Review\n"
        "- Contains minor misrepresentation of paper information, or shows perfunctory content and obvious omissions;\n"
        "- Lacks sufficient constructiveness, consisting mostly of general criticism without targeted improvement suggestions;\n"
        "- Some viewpoints are unsupported by valid evidence; no explicit hostility, yet with mild subjectivity and inadequate academic rigor.\n"
        "2 Points: Poor Review\n"
        "- Multiple factual errors throughout the content and a distinctly perfunctory, careless attitude;\n"
        "- Barely any constructive feedback provided;\n"
        "- Obvious prejudice and groundless assertions, with occasional unprofessional and negative expressions;\n"
        "- The logical reasoning of the review is disconnected from the core content of the manuscript.\n"
        "1 Point: Deficient Review\n"
        "- Riddled with factual errors and an extremely negligent reviewing attitude;\n"
        "- Fully non-constructive comments filled with hostile remarks or severe prejudice;\n"
        "- Almost all critical claims lack manuscript-based evidence, relying purely on subjective speculation, and seriously violate fundamental academic review principles.\n\n"
        "Your final output MUST be a JSON object with EXACTLY these keys:\n"
        "{{\n"
        "  \"is_defective\": \"true / false\",\n"
        "  \"quality_score\": 1 / 2 / 3 / 4 / 5,\n"
        "  \"defect_type\": \"Comma-separated list of defect types in descending prominence (e.g., 'bias, information_error') or 'no_deficient'\"\n"
        "}}\n"
        "Important: Prominence is defined by the degree to which the defect reduces the review's academic validity, helpfulness, and fairness. Determine which defect is most salient in the context of this specific review, not by a fixed priority ladder. When multiple defects exist, always list them from most to least impactful.\n"
        "Prohibited: Outputting only a single defect when multiple verified defects exist; sorting defects by any order other than true prominence; making judgments without clear detection evidence.\n"
    )

    # _INTEGRATE_PROMPT_TEMPLATE = (
    #     "Role: Academic Review Multi-Tool Detection Result Integrator. Take the detection results, evidence and flags of all previous tools as important references rather than rigid absolute criteria; distinguish the common and acceptable minor imperfections in paper reviews from those with subjective malice or substantive impact on final review conclusions, then sort out all verified defect types according to the defect priority rules in the main prompt to provide a complete basis for the final judgment.\n"
    #     "Label List (Sorted by Defect Severity, Highest Priority First):\n"
    #     "Sort out all defect types that have been independently verified and matched, and list them in the following priority order (multiple labels are allowed if they meet the definitions):\n"
    #     "1. bias: The review has subjective bias, emotional attack, or malicious negation (highest priority)\n"
    #     "2. unprofessional: The review has comprehension errors caused by lack of basic domain knowledge\n"
    #     "3. information_error: The review has explicit factual data errors\n"
    #     "4. careless: The review has careless omission errors of the paper content\n"
    #     "5. no_evidence: The review's core claims have no supporting evidence in the paper\n"
    #     "6. lack_constructive: The review only has criticism but no actionable constructive suggestions\n"
    #     "7. no_deficient: The review has no above defects, professional, objective, evidence-based, and constructive (lowest priority)\n\n"
    #     "Input: Detection results from bias detector, constructiveness detector, factual verification detector, error type classifier\n"
    #     "Output Rules:\n"
    #     "1. ONLY output a single valid JSON object, NOTHING ELSE\n"
    #     "2. NO explanations, NO reasoning, NO judgment basis, NO natural language\n"
    #     "3. NO extra characters, NO markdown, NO code blocks, NO comments\n"
    #     "4. If defective, output ONLY the SINGLE highest-priority verified defect type\n"
    #     "5. Output non-defective ONLY when all detection results show no defects\n"
    #     "Quality Score (5-point Scale, 4 points as the passing score):\n"
    #     "5 Points: Excellent Review\n"
    #     "- Free of factual errors with a rigorous and conscientious attitude;\n"
    #     "- Comments are professional, objective and highly constructive, with specific revision guidance offered;\n"
    #     "- All viewpoints are supported by the original manuscript content, experimental data or references;\n"
    #     "- Fully unbiased and non-hostile; evaluations are entirely based on the inherent quality of the paper.\n"
    #     "4 Points: Qualified Review (Passing Tier)\n"
    #     "- No obvious factual errors or factual deviations, with a basically earnest reviewing attitude;\n"
    #     "- Comments deliver moderate constructiveness and rational criticism;\n"
    #     "- Minor oversights may be present in the review content;\n"
    #     "- Free from subjective prejudice and hostility; arguments are reasonably grounded and consistent with standard academic review specifications.\n"
    #     "3 Points: Mediocre Review\n"
    #     "- Contains minor misrepresentation of paper information, or shows perfunctory content and obvious omissions;\n"
    #     "- Lacks sufficient constructiveness, consisting mostly of general criticism without targeted improvement suggestions;\n"
    #     "- Some viewpoints are unsupported by valid evidence; no explicit hostility, yet with mild subjectivity and inadequate academic rigor.\n"
    #     "2 Points: Poor Review\n"
    #     "- Multiple factual errors throughout the content and a distinctly perfunctory, careless attitude;\n"
    #     "- Barely any constructive feedback provided;\n"
    #     "- Obvious prejudice and groundless assertions, with occasional unprofessional and negative expressions;\n"
    #     "- The logical reasoning of the review is disconnected from the core content of the manuscript.\n"
    #     "1 Point: Deficient Review\n"
    #     "- Riddled with factual errors and an extremely negligent reviewing attitude;\n"
    #     "- Fully non-constructive comments filled with hostile remarks or severe prejudice;\n"
    #     "- Almost all critical claims lack manuscript-based evidence, relying purely on subjective speculation, and seriously violate fundamental academic review principles.\n"

    #     "Your final output MUST be a JSON object with EXACTLY these keys:\n"
    #     "{{\n"
    #     "  \"is_defective\": \"Whether there is a defect, value: true / false\",\n"
    #     "  \"quality_score\": quality socre of the review, value: 1 / 2 / 3 / 4 / 5,\n"
    #     "  \"defect_type\": \"Highest priority defect type / no_defect\"\n"
    #     "}}\n"
    #     "Important: "
    #     "Prohibited: Output multiple labels, violate the priority order, make judgments without detection basis.\n"
    #     "{paper_read_prompt}\n"
    # )


    # "Output Rules:\n"
        # "1. Must sort out ALL verified and matched defect types, strictly follow the priority order to list them (multiple labels are allowed if they meet the independent verification requirements)\n"
        # "2. Must give clear judgment basis, corresponding to the detection results of each tool\n"
        # "3. Only output no_deficient when all detection results show no defects\n\n"
        # "Your final output MUST be a JSON object with EXACTLY these keys:\n"
        # "{{\n"
        # "  \"matched_defect_types\": [\"List of all verified defect types in priority order, e.g., 'bias', 'information_error'\"],\n"
        # "  \"judgment_basis\": \"Corresponding to the detection results of each tool, explain the reason for the final label\",\n"
        # "  \"defect_severity\": \"high|medium|low|no_defect\",\n"
        # "  \"confidence\": \"high|medium|low\"\n"
        # "}}\n"
    
    # ========== Modification 2: QUERY_DESCRIPTION retained (no config dependencies) ==========
    _QUERY_DESCRIPTION = {
        "semanticscholar": ("Semantic Scholar paper search query statement, must adapt to paper/search interface syntax:\n"
                        "Supports logical operators (| for OR, + for AND, - for NOT), exact phrases (enclosed in \"\"),\n"
                        "field restrictions (e.g., year:2020-2024, title:keywords, author:author name),\n"
                        "only matches paper titles and abstracts, keep concise (example: \"transformer model\" +NLP -review year:2018-2024)"),
        "arxiv": ("arXiv paper search query statement, must adapt to its API query syntax:\n"
                        "Supports logical operators (uppercase AND, OR, NOT), exact phrases (enclosed in \"\"),\n"
                        "field restrictions (e.g., ti:title, au:author name, abs:abstract, year:year, cat:subject classification).\n"
                        "Example: au:\"Yann LeCun\" AND ti:convolutional AND cat:cs.LG"),
        "default": ("Retrieval query statement, briefly describe the theme and keywords of relevant literature you want to retrieve,"),
    }
    
    # ========== Modification 3: SYSTEM_MAP changed to "Template Mapping", only stores templates and tools, not final prompts ==========
    _SYSTEM_TEMPLATE_MAP = {
        "verify": {
            "prompt_template": _VERIFY_PROMPT_TEMPLATE,
            "tools": [_ReadPaperTool, _RetrieveLiteratureTool]
            # "tools": [_ReadPaperTool, _ImageAnalyzer, _RetrieveLiteratureTool]
        },
        "correct": {
            "prompt_template": _CORRECT_PROMPT_TEMPLATE,
            "tools": [_ReadPaperTool]
            # "tools": [_ReadPaperTool, _ImageAnalyzer]
        },
        "complete": {
            "prompt_template": _COMPLETE_PROMPT_TEMPLATE,
            "tools": [_ReadPaperTool, _RetrieveLiteratureTool]
            # "tools": [_ReadPaperTool, _ImageAnalyzer, _RetrieveLiteratureTool]
        },
        "transform": {
            "prompt_template": _TRANSFORM_PROMPT_TEMPLATE,
            "tools": [_ReadPaperTool]
        },
        "integrate": {
            "prompt_template": _INTEGRATE_PROMPT_TEMPLATE,
            "tools": []
        },
    }

    # ========== Modification 4: New method for dynamically generating Prompt ==========
    def _get_dynamic_prompt(self, action):
        """Dynamically generate Prompt at runtime (read latest config)"""
        template = self._SYSTEM_TEMPLATE_MAP[action]["prompt_template"]
        # Read latest config values to generate dynamic fragments
        image_analyze_prompt = ""
        if config.get("tools", {}).get("image_analyzer", {}).get("enabled", False):
            image_analyze_prompt = config.get("prompt", {}).get("image_analyze_tool_usage_prompt", "")
        paper_read_prompt = config.get("prompt", {}).get("paper_read_tool_usage_prompt", "")
        paper_title = paper.get("title", "<Title unavailable>")
        # Fill template placeholders and return final Prompt
        return template.format(image_analyze_prompt=image_analyze_prompt, paper_read_prompt=paper_read_prompt, paper_title=paper_title)
    
    def call(self, action,** kwargs):
        content = kwargs["content"]
        paper_context = kwargs.get("paper_context", "")
        analysis = kwargs.get("analysis", "")
        review_data = kwargs.get("review_data", None)
        tool_history = kwargs.get("tool_history", [])
        reviewer_name = review_data.get("reviewer_name", "reviewer")
        analysis_desc = f"\nAnalysis from other tools: {analysis}\n" if analysis else ""   
        
        # ========== Modification 5: Read latest config at runtime to configure _RetrieveLiteratureTool ==========
        if not _RetrieveLiteratureTool.configured:
            # Dynamically get latest search_papers.api configuration
            search_api = config.get("search_papers", {}).get("api", [])
            if len(search_api) == 0:
                logger.error("At least one retrieval API must be provided!")
                return "Retrieval unavailable: No retrieval APIs allowed!", {}
            query_desc = ""
            for api in search_api:
                query_desc += f"\nSyntax for {api}:"
                query_desc += self._QUERY_DESCRIPTION.get(api, self._QUERY_DESCRIPTION["default"])
            _RetrieveLiteratureTool.tool_metadata["parameters"]["properties"]["api_choice"]["description"] += ", ".join(search_api)
            _RetrieveLiteratureTool.tool_metadata["parameters"]["properties"]["query"]["description"] += query_desc
            _RetrieveLiteratureTool.configured = True
        
        if action not in self._SYSTEM_TEMPLATE_MAP:
            return f"Unsupported defensive operation: {action}"
        
        # ========== Modification 6: Dynamically get Prompt at runtime (use latest config) ==========
        current_prompt = self._get_dynamic_prompt(action)
        
        # Interactive loop with LLM: until LLM stops invoking tools
        llm_messages = [
            {"role": "system", "content": (f"Please read this paper main text:\n"
                                            f"{paper['main_text']}\n\n"
                                            f"Please read the content of the paper and carry out the task.\n\n"
                                            f"Your goal: \n{current_prompt}"
                                            )}
        ]
        if action == "integrate":
            llm_messages.append({
                "role": "user", 
                "content": (f"Complete review: {review_data.get("review_content", "\n")}\n"
                            f"Tool call history: {tool_history}"
                            f"=== Analysis from the main agent, for reference only ==="
                            f"Content to process: {content}\n"
                            f"Corresponding paper context: {paper_context}\n"
                            f"{analysis_desc}"
                            f"========================================================"
                            "Please perform the corresponding operation according to your goal in system instructions."
                            )})
        else:
            llm_messages.append({
                "role": "user", 
                "content": (f"Content to process: {content}\n"
                            f"Corresponding paper context: {paper_context}\n"
                            f"{analysis_desc}"
                            "Please perform the corresponding operation according to your goal in system instructions."
                            "Important: Please output all possible results, even if your confidence is low."
                            )})

        if config.get("tools", {}).get("malice_defense_tool", {}).get("enabled", True) == False or \
                config.get("tools", {}).get("malice_defense_tool", {}).get("sub_tools", {}).get(action, True) == False:
            self_result = f"The action '{action}' has been disabled and will not return execution results. Please do not call this action."
            llm_messages.append({"role": "assistant", "content": self_result})
        else:
            llm_tools = []
            # build sub-tools configuration for LLM calls based on the template mapping
            for tool in self._SYSTEM_TEMPLATE_MAP[action]["tools"]:
                meta = tool.tool_metadata
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": meta["tool_name"],
                        "description": meta["description"],
                        "parameters": meta["parameters"]
                    }
                })
            while True:
                # Call LLM (pass tool configuration)
                llm_result = call_llm(
                    messages=llm_messages,
                    tools=llm_tools,
                    temperature=0.3,
                    presence_penalty=0.5,
                    frequency_penalty=0.5,
                    model="llm_integrate_tool" if action == "integrate" else "llm_tools",
                    call_id=f"tool_{paper['id']}"
                )

                # Process LLM results: execute tools if there are tool calls; end if none
                if llm_result["tool_calls"]:
                    llm_messages.append(llm_result["raw_msg"])
                    for tool_call in llm_result["tool_calls"]:
                        for tool in self._SYSTEM_TEMPLATE_MAP[action]["tools"]:
                            if tool_call["name"] == tool.tool_metadata["tool_name"]:
                                # Execute tool invocation specified by LLM
                                tool_instance = tool()
                                tool_result = tool_instance.call(
                                    **tool_call["arguments"]
                                )
                                # Add tool results to next round of LLM messages
                                llm_messages.append({
                                    "role": "tool",
                                    "content": f"Tool {tool_call['name']} returned results:\n{tool_result}"
                                })
                else:
                    # LLM stops invoking tools, get final rebuttal
                    self_result = llm_result["answer"]
                    llm_messages.append({"role": "assistant", "content": self_result})
                    break 
        
        return self_result, llm_messages, reviewer_name