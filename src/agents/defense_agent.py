# agents/defense_agent.py
from .base_agent import BaseAgent
from utils.llm_client import call_llm
from utils.logger import logger
from utils.shared_data import result, paper  # Import global result variable
from utils.parallel import parallel_run
import re
import time
import json
import json5
import copy

class DefenseAgent(BaseAgent):
    def __init__(self, config):
        super().__init__(
            agent_type="defense_agent",
            name=config["agents"]["defense_agent"]["name"],
            config=config
        )
        
        self.max_retry = config["agents"]["defense_agent"].get("max_retry", 2)  # 最大重跑次数
        self.retry_delay = config["agents"]["defense_agent"].get("retry_delay", 1)  # 重跑延迟（秒）


    def _call_single_tool(self, task_idx, tool_name, tool_args):
        """
        Subfunction for single tool call (for parallel execution).
        Receives task_idx and includes it in return results for subsequent sorting.
        """
        try:
            # Execute tool call
            tool_result = self.call_tool(tool_name=tool_name, **tool_args)
            # Return dictionary containing task index and execution result
            return {
                "task_idx": task_idx,
                "result": tool_result
            }
        except Exception as e:
            logger.error(f"Tool call failed: {tool_name}, Parameters: {tool_args}, Error: {str(e)}", exc_info=True)
            # Return task index and error message even if failed
            return {
                "task_idx": task_idx,
                "result": f"Tool call failed: {str(e)}"
            }
    
    def _process_tool_calls(self, llm_result, llm_messages, llm_tools_name, review_data, tool_history):
        """Process tool calls in parallel (replace original serial for loop)"""
        tool_calls = llm_result.get("tool_calls", [])
        if not tool_calls:
            logger.warning("Find empty tool_calls")
            logger.warning(llm_result)
            return llm_messages, tool_history
        
        # Build parallel task list: each task corresponds to one tool call
        parallel_tasks = []
        for idx, tool_call in enumerate(tool_calls):
            if tool_call["name"] in llm_tools_name:
                # Store task information (including index for subsequent order restoration)
                # tool_call_arguments = tool_call["arguments"]
                tool_call_arguments = copy.deepcopy(tool_call["arguments"])
                tool_call_arguments["review_data"] = review_data
                tool_call_arguments["tool_history"] = tool_history
                parallel_tasks.append((
                    self._call_single_tool,
                    (idx, tool_call["name"], tool_call_arguments),
                    {}  # Pass original task index
                ))

        if not parallel_tasks:
            logger.warning("Find empty tool_calls")
            logger.warning(llm_result)
            return llm_messages, tool_history
        
        # Execute tool calls in parallel (control max concurrency to avoid tool service overload)
        max_workers = self.config["agents"]["defense_agent"].get("max_workers_tool_call") or min(len(parallel_tasks), 3)  # Adjust based on tool service capacity
        raw_results = parallel_run(parallel_tasks, max_workers=max_workers)
        
        # Sort tool results in original order (key: ensure consistency with tool call sequence)
        ordered_results = sorted(raw_results, key=lambda x: x["task_idx"])
        
        # Add tool results to next round of LLM messages
        for idx, tool_result in enumerate(ordered_results):
            tool_call = tool_calls[idx]  # Get tool call info in original order
            llm_messages.append({
                "role": "tool",
                "content": f"Tool {tool_call['name']}: {tool_call['arguments'].get('action', '')} returned results:\n{tool_result['result']}"
            })
            tool_history.append({
                "tool_call": tool_call,
                "tool_return": tool_result['result']
            })
        
        return llm_messages, tool_history

    def _analyze_single_review(self, review_data, review_type="initial"):
        """Analyze single review (initial/updated), judge multiple malice types in one LLM call"""
        if self.config["tools"]["malice_defense_tool"]["enabled"] == False:
            return self._analyze_single_review_without_tool(review_data, review_type=review_type)
        try:
            if review_type not in ["initial", "updated"]:
                logger.error(f"[{self.name}] Unsupported review type: {review_type}")
                return None

            system_prompt = f"""
# TOP 0 MANDATORY OUTPUT RULE (NO EXCEPTIONS, HIGHEST PRIORITY)
YOU MUST CHOOSE ONLY ONE OF THE FOLLOWING TWO OUTPUT MODES FOR EACH TURN, NEVER COMBINE THEM, NEVER FABRICATE ANY TOOL CALL RESULTS:
1. MODE A: TOOL CALL ONLY. If you need to perform verification/analysis via the tool, output ONLY the standard tool call instruction that matches the provided tools definition, NO final judgment, NO JSON content, NO simulated tool response.
2. MODE B: FINAL JSON OUTPUT ONLY. You may ONLY use this mode when ALL required tool calls have been completed and you have received the REAL official tool return results. Output ONLY the standard compliant JSON, NO additional tool call planning, NO made-up tool records. Enter this mode only when you receive the result of the integrate action call from the malice_defense_tool in the historical dialogue.
FORBIDDEN: Fabricating any tool call record, simulated tool response, or tool execution result in thinking content or final output.

# STANDARD TOOL CALL MANDATORY PARAMETER SPECIFICATION (100% COMPLIANCE REQUIRED)
All tool calls MUST strictly follow this format, include ALL 4 REQUIRED PARAMETERS, no omission allowed:
- Fixed tool_name: malice_defense_tool
- Required parameter 1: action
  * Type: string
  * Valid values & description: verify (multi-defect detection: factual/evidence/careless)/correct (error type classification)/complete (constructiveness detection)/transform (bias & invalid content detection)/integrate (final classification)
- Required parameter 2: content
  * Type: string
  * Description: Content to be processed (reviews/rebuttals/analysis results), one paragraph without line breaks
- Required parameter 3: paper_context
  * Type: string
  * Description: Paper context, abstract/key sections of the paper related to the content to be processed, used for verification, provide as complete information as possible, one paragraph without line breaks
- Required parameter 4: analysis
  * Type: string
  * Description: Analysis content from other defensive Tools, 'N/A' if no previous analysis is available.

# TOP 1-3 CORE BUSINESS RULES (MANDATORY, NO EXCEPTIONS)
1. YOU HAVE NO AUTHORITY TO ISSUE A FINAL DEFECT JUDGMENT. The `integrate` action of the official tool holds the SOLE AND FINAL decision-making power for all core judgments (whether the review is defective, defect type priority, classification value). YOU MUST CALL THE TOOL WITH `action: integrate` AS THE FINAL TOOL CALL BEFORE ANY FINAL JSON OUTPUT.
2. ALL FACTUAL JUDGMENTS MUST BE VERIFIED VIA THE OFFICIAL TOOL. The `verify` action has full access to the paper's complete full text, no unverifiable content. Factual judgments without real tool verification are strictly prohibited.
3. ALL ANALYTICAL FUNCTIONS MUST BE EXECUTED EXCLUSIVELY VIA THE ONLY OFFICIAL TOOL: `malice_defense_tool`. All operations are triggered by the `action` parameter, with ONLY 5 VALID VALUES: verify/correct/complete/transform/integrate. Skipping tool calls or performing analysis directly is strictly prohibited.

## CLEAR STAGE EXECUTION GUIDANCE (WHICH MODE TO CHOOSE)
### YOU MUST USE MODE A (TOOL CALL ONLY) WHEN:
- You have completed pre-analysis and identified any point that requires tool verification/operation
- You have not yet completed all mandatory tool calls for the current review
- You have not yet called the tool with `action: integrate` as the final step
### YOU MAY USE MODE B (FINAL JSON ONLY) WHEN:
- All mandatory tool calls have been completed, and you have received the REAL official return results for every call
- The final `action: integrate` call has been completed, and you have received its official final judgment
- All content in the JSON can be 100% supported by the real tool return results

## MANDATORY PRE-ANALYSIS WORKFLOW (NO SKIPPING, MUST COMPLETE BEFORE MODE SELECTION)
Before choosing output mode, you MUST complete these 3 sub-steps for the full review, sentence by sentence:
1. Factual Point Disassembly: Extract ALL verifiable factual points from the review, each including: (a) exact original quote from the review, (b) corresponding paper location, (c) core content to be verified via tool.
2. Non-Factual Defect Feature Disassembly: Extract ALL non-factual features matching the defect definitions, each including: (a) exact original quote from the review, (b) suspected defect type (sorted by priority), (c) core feature to be verified via tool.
3. Pre-Matching & Tool Call Planning: (a) Pre-match all extracted points/features to defect types per the fixed priority order; (b) Plan tool calls with clear action types and purposes; (c) Allocate tool quota in full compliance with rules.
After completing this workflow, you MUST choose MODE A (TOOL CALL ONLY) if any tool call is required. You may only choose MODE B if all required tool calls have been completed with real returns.
All content from this pre-analysis workflow must be fully retained for final input into the tool with `action: integrate`.

## ROLE & CORE TASK
You are a Deficient Peer Review Audit & Optimization Analyst. Your core task is to identify deficient peer reviews, conduct standardized multi-dimensional quality audits, and deliver dual constructive outputs: (1) actionable revision suggestions for the paper, (2) targeted optimization guidance for the review itself. All work must be based on the review content, provided paper context, and REAL official return results from the `malice_defense_tool`.

## INPUT BOUNDARIES (STRICTLY FOLLOW)
You will receive one of two input types, and must adhere exclusively to the corresponding rules:
- ONLY ABSTRACT PROVIDED: The `verify` action of the official tool has full access to the paper's complete full text. All factual defect judgments MUST be verified via the tool with `action: verify`. Factual judgments without real tool verification are strictly prohibited.
- FULL TEXT/KEY SECTIONS PROVIDED: You may use the provided content for cross-check, but all factual judgments still require mandatory verification via the tool with `action: verify`.
FORBIDDEN: Making factual judgments without real tool verification, or fabricating non-existent paper content.

## DEFECT CLASSIFICATION FRAMEWORK
### LABEL RULES (MANDATORY)
1. PRIORITY ORDER (HIGHEST to LOWEST, must match labels in this fixed sequence to avoid misjudgment. Higher-priority labels fully override lower-priority labels for the same content):
   Step 1: Attitude & Bias Defects (HIGHEST PRIORITY: unprofessional > bias)
   Step 2: Factual & Evidence Defects (MEDIUM PRIORITY: information_error > careless > no_evidence)
   Step 3: Constructiveness Defect (LOWEST PRIORITY: lack_constructive)
2. MUTUAL EXCLUSION: A single sentence/feature cannot trigger multiple labels. You MUST assign only the highest-priority applicable label to the same content, and must not assign lower-priority labels to content that already meets a higher-priority defect definition.
3. MULTI-LABEL RULE: Multi-label assignment is only allowed if distinct, independently verified defects meet multiple separate defect definitions. The exact defect types and their order returned by the `action: integrate` operation must be used directly in all outputs, without reordering or applying the predefined priority sequence.

### PREDEFINED DEFECT TYPES (Sorted by Priority | Definition + Mandatory Action + Valid Evidence Standard)
1. unprofessional (Unprofessional & Hostile): Explicit negative qualitative language (e.g., 'trivial repackaging', 'laughably small') with no substantive academic criticism, or excessive belittling of the paper's contributions that contradicts its actual documented innovations. → MANDATORY ACTION: Call tool with `action: transform`. → Valid Evidence: Real official return from tool (action: transform) with `has_valid_academic_concern: false` + original hostile quote, OR real return from tool (action: correct) with `has_unprofessional_error: true` and error_type: comprehension_error + full tool call trace.
2. bias (Bias-Oriented): Excessively downplaying the paper's documented innovations or over-amplifying minor flaws, or applying stricter evaluation standards than the verified norm in the target field. → MANDATORY ACTION: Call tool with `action: transform` + `action: verify` (for field standard validation). → Valid Evidence: Real official return from tool (action: transform) with `has_bias_or_invalid_content: true` + bias_type classification + original review quote + full tool call trace.
3. information_error (Factual Information Error): Explicit misinterpretation of core formulas/figure trends, false claims that the paper lacks content which clearly exists in the full text, or incorrect statement of key experimental parameters (e.g., dataset split ratio, learning rate). → MANDATORY ACTION: Call tool with `action: verify` (for core content/parameter validation); if factual error is confirmed, follow up with tool with `action: correct` for error classification. → Valid Evidence: Real official return from tool (action: verify) with `has_factual_error: true` + full tool call trace.
4. careless (Careless & Unserious Omission): Ignoring content explicitly stated in the paper's full text, or incorrectly confusing the target paper with other relevant works. → MANDATORY ACTION: Call tool with `action: verify` (for full text cross-check). → Valid Evidence: Real official return from tool (action: verify) with `has_careless_omission: true` + full tool call trace.
5. no_evidence (Unsubstantiated Claims): Core criticism lacks any support from the paper's data or domain literature, or cites formulas/figures but provides no specific supporting details (e.g., 'Fig.3 shows ineffectiveness' without specifying data points). → MANDATORY ACTION: Call tool with `action: verify` (for evidence check). → Valid Evidence: Real official return from tool (action: verify) with `has_no_evidence_claim: true` + full tool call trace.
6. lack_constructive (Lack of Constructiveness): Criticism uses vague statements without specific improvement directions, or ≥2 consecutive criticism points without citing the paper's specific sections/methods. → MANDATORY ACTION: Call tool with `action: complete` (to supplement actionable improvement directions). → Valid Evidence: Real official return from tool (action: complete) with `is_lack_constructive: true` + original vague criticism quote + full tool call trace.

## NON-NEGOTIABLE TOOL CALL RULES
1. FINAL STEP MANDATE: Regardless of previous tool results, YOU MUST CALL THE TOOL WITH `action: integrate` AS THE ABSOLUTE FINAL TOOL CALL. All pre-analysis content, review excerpts, paper context, and full real input/output of all prior tool calls must be included in the input of this final call. No final JSON output may be issued before this call is completed and real return is received.
2. QUOTA RULES (Total Quota: 8 points; 1 point per tool call regardless of action type; final `action: integrate` call uses the reserved 1 point exclusively):
   - YOU MUST RESERVE AT LEAST 1 POINT EXCLUSIVELY FOR THE FINAL `action: integrate` CALL. Early exhaustion is strictly prohibited.
   - Maximum 4 points may be allocated to factual verification operations (action: verify / action: correct).
   - Minimum 3 points must be allocated to attitude/bias & constructiveness verification operations (action: complete / action: transform), with at least 1 call for each action type.
   - Remaining quota should be used for secondary verification of disputed content.
3. CONFLICT RESOLUTION: If real tool returns are contradictory, YOU MUST NOT MAKE INDEPENDENT JUDGMENTS. Fully retain all conflicting real tool input/output, and submit all content to the tool with `action: integrate` for final resolution.
4. CALL PRIORITY ORDER (Aligned with Defect Priority, HIGHEST to LOWEST):
   1. Verify highest-priority attitude/bias defects by calling tool with `action: transform`
   2. Verify factual/evidence defects by calling tool with `action: verify`
   3. Classify confirmed factual errors by calling tool with `action: correct`
   4. Verify lowest-priority constructiveness defects by calling tool with `action: complete`
   5. Secondary verification of disputed content
   6. FINAL MANDATORY CALL: tool with `action: integrate`

## DUAL SCORING SYSTEM
You must calculate and output these two scores ONLY in MODE B, with clear explanations fully supported by REAL official tool return results and 100% aligned with the final judgment from the `action: integrate` operation.
1. Defect Classification Certainty
   - Definition: Confidence level of the final core defect judgment, independent of defect severity. Value range: 0.0-1.0 (1 decimal place).
   - Mandatory Rules: 0.5-1.0 if `action: integrate` operation returns `is_defective: true`; 0.0-0.5 if `action: integrate` operation returns `is_defective: false`; 0.5 only for completely contradictory real tool results.
   - Judgment Basis: Confidence levels from all real tool returns, consistency of tool results with `action: integrate` final judgment, sufficiency of verifiable evidence.
2. Quality Score (5-point Scale, 4 points as the passing score)
### 5 Points: Excellent Review
- Free of factual errors with a rigorous and conscientious attitude;
- Comments are professional, objective and highly constructive, with specific revision guidance offered;
- All viewpoints are supported by the original manuscript content, experimental data or references;
- Fully unbiased and non-hostile; evaluations are entirely based on the inherent quality of the paper.
### 4 Points: Qualified Review (Passing Tier)
- No obvious factual errors or factual deviations, with a basically earnest reviewing attitude;
- Comments deliver moderate constructiveness and rational criticism;
- Minor oversights may be present in the review content;
- Free from subjective prejudice and hostility; arguments are reasonably grounded and consistent with standard academic review specifications.
### 3 Points: Mediocre Review
- Contains minor misrepresentation of paper information, or shows perfunctory content and obvious omissions;
- Lacks sufficient constructiveness, consisting mostly of general criticism without targeted improvement suggestions;
- Some viewpoints are unsupported by valid evidence; no explicit hostility, yet with mild subjectivity and inadequate academic rigor.
### 2 Points: Poor Review
- Multiple factual errors throughout the content and a distinctly perfunctory, careless attitude;
- Barely any constructive feedback provided;
- Obvious prejudice and groundless assertions, with occasional unprofessional and negative expressions;
- The logical reasoning of the review is disconnected from the core content of the manuscript.
### 1 Point: Deficient Review
- Riddled with factual errors and an extremely negligent reviewing attitude;
- Fully non-constructive comments filled with hostile remarks or severe prejudice;
- Almost all critical claims lack manuscript-based evidence, relying purely on subjective speculation, and seriously violate fundamental academic review principles.

## KEY GUARDRAILS & REFERENCE EXAMPLES
### FALSE POSITIVE GUARDRAIL: A review cannot be classified as defective if it contains valid, substantive academic criticism/actionable suggestions, even with minor non-core defects. Only defects that impact the core validity of the review can trigger a deficient classification.
### FALSE NEGATIVE GUARDRAIL: A review can only be classified as non-deficient if it meets BOTH: (1) No confirmed defects after full real tool verification; (2) Contains at least one valid, substantive academic concern or actionable suggestion for the paper.

### REFERENCE EXAMPLES (Aligned with Priority Rules)
1. Highest-Priority Defect Example: Review claims 'this work is a trivial repackaging with no innovation' using dismissive language with no substantive criticism. `malice_defense_tool (action: transform)` official return confirms `has_valid_academic_concern: false` and `has_bias_or_invalid_content: true`. `malice_defense_tool (action: integrate)` final return outputs `is_defective: true`, defect_type: unprofessional. → classification=1, certainty=1.0, severity=4, final_conclusion=unprofessional.
2. Non-Deficient Example: Review identifies an accuracy gap in Table 3 and provides a specific dimensionality reduction optimization suggestion. All official tool returns confirm no defects and valid actionable feedback. `malice_defense_tool (action: integrate)` final return outputs `is_defective: false`. → classification=0, certainty=1.0, severity=0, final_conclusion=no_deficient.
3. Priority Override Example: Review uses hostile, unprofessional language AND makes a factual error. Per priority rules, ONLY the highest-priority 'unprofessional' label is applied to the hostile content, with the factual error as a secondary defect only if it is distinct and independently verified via official tool calls. The final output order of the defect labels follows exactly the list returned by the integrate action.

## STRICT JSON OUTPUT REQUIREMENTS (ONLY FOR MODE B)
1. Output ONLY a single line of json.loads()-parsable JSON. No line breaks, Markdown, or extra text outside the JSON.
2. String Escaping: Replace " with \\", replace \\ with \\\\.
3. MANDATORY CLASSIFICATION RULE: The `classification` field MUST be strictly and fully consistent with the real final return of tool (action: integrate): set classification=1 IF AND ONLY IF the tool returns `is_defective: true`; set classification=0 IF AND ONLY IF the tool returns `is_defective: false`. Any independent modification of this field is strictly prohibited.
4. MANDATORY QUALITY SCORE RULE: The `quality_score.score` field MUST be strictly and fully consistent with the real final return of tool (action: integrate). Any independent modification of this field is strictly prohibited.
5. Fixed JSON Structure (NO DELETION OR ADDITION OF TOP-LEVEL FIELDS; ALL FIELDS MUST BE FILLED):
{{
  "classification": 0/1,
  "defect_classification_certainty": {{"score": float, "explain": "Basis for the certainty score, with real official tool result references"}},
  "quality_score": {{"score": integer, "explain": "Basis for the quality score, explain with confirmed defect details from real tool returns"}},
  "result": {{
    "unprofessional": {{"label":true/false,"explain":"Exact review quote + verification reason + real tool result","evidence_trace":"malice_defense_tool + action type + core real output flag for this judgment"}},
    "bias": {{"label":true/false,"explain":"Exact review quote + verification reason + real tool result","evidence_trace":"malice_defense_tool + action type + core real output flag for this judgment"}},
    "information_error": {{"label":true/false,"explain":"Exact review quote + verification reason + real tool result","evidence_trace":"malice_defense_tool + action type + core real output flag for this judgment"}},
    "careless": {{"label":true/false,"explain":"Exact review quote + verification reason + real tool result","evidence_trace":"malice_defense_tool + action type + core real output flag for this judgment"}},
    "no_evidence": {{"label":true/false,"explain":"Exact review quote + verification reason + real tool result","evidence_trace":"malice_defense_tool + action type + core real output flag for this judgment"}},
    "lack_constructive": {{"label":true/false,"explain":"Exact review quote + verification reason + real tool result","evidence_trace":"malice_defense_tool + action type + core real output flag for this judgment"}}
  }},
  "tool_call_summary": [{{"action":"string (must be selected from: verify/correct/complete/transform/integrate)","call_purpose":"string (specific purpose of this tool call)","core_output":"string (REAL official return result from the tool, NO FABRICATION)","quota_consumed":integer}}],
  "final_conclusion": "no_deficient/defect types in the predefined priority order, separated by ", "",
  "valid_academic_suggestions": "Actionable, specific paper revision suggestions extracted only from verified valid criticism of the review via real tool returns",
  "review_optimization_suggestions": "Targeted suggestions for optimizing the peer review itself, to fix confirmed defects and improve review quality; for non-deficient reviews, fill with 'No optimization needed, the review is compliant and high-quality'"
}}
5. final_conclusion Rule: 0 → "no_deficient"; 1 → all defect types exactly as returned by the `action: integrate` operation, preserving the original order, separated by ", ".
6. MANDATORY ANTI-FAKE RULE: All content in `tool_call_summary`, `evidence_trace`, and `explain` fields must be 100% consistent with the real input/output records of `malice_defense_tool`. Any fabricated tool call records or results will be deemed invalid.

## FINAL CRITICAL REMINDER
If no callable `malice_defense_tool` is provided, or no real tool call is executed, you MUST explicitly state 'No available tools, unable to complete final judgment'. Direct output of any judgment result without the real official return of tool (action: integrate) is strictly prohibited.
"""

            prompt = (
                    #   f"The tools corresponding to the action parameter of malice_defense_tool are as follows:\n{str(TOOL_DESCRIPTION)}\n"
                      f"{review_type} Review Content: {review_data['review_content']}\n"
                      f""
                    )

            llm_tools = self.get_llm_tool_config()  # Get tool configurations
            llm_tools_name = self.get_llm_tool_names()  # Get list of tool names

            def is_valid_json5(json_str: str) -> bool:
                if not isinstance(json_str, str):
                    return False
                try:
                    json5.loads(json_str)
                    return True
                except Exception as e:
                    logger.error(f"Unexpected error during JSON5 validation: {str(e)}", exc_info=True)
                    return False

            # llm_messages = [
            #     {"role": "system", "content": system_prompt + f"\nPaper Content: {review_data.get('paper_content', 'Not provided')}\n"},
            #     # {"role": "user", "content": },
            #     {"role": "user", "content": prompt}
            # ]
            # malice_result_str = ""
            # tool_call_history = []
            max_retry_invalid_output = -1
            while True: # enforce call integrate tool
                if max_retry_invalid_output < 0:
                    # logger.warning(f"Having Retried generation for {max_retry_invalid_output} times but failed. Starting a new dialog.")
                    llm_messages = [
                        {"role": "system", "content": system_prompt + f"\nPaper Content: {review_data.get('paper_content', 'Not provided')}\n"},
                        # {"role": "user", "content": },
                        {"role": "user", "content": prompt}
                    ]
                    malice_result_str = ""
                    tool_call_history = []
                    max_retry_invalid_output = 3
                while True:
                    # Call LLM (pass tool configurations)
                    llm_result = call_llm(
                        messages=llm_messages,
                        tools=llm_tools,
                        model="llm_defense",
                        call_id=f"analyze_{paper['id']}"
                    )

                    # 3. Process LLM results: execute tools if called; end if none
                    if llm_result["tool_calls"] and llm_result["tool_calls"] != []:
                        llm_messages.append(llm_result["raw_msg"])
                        llm_messages, tool_call_history = self._process_tool_calls(llm_result, llm_messages, llm_tools_name, review_data, tool_call_history)
                    else:
                        # LLM stops calling tools, get final results
                        malice_result_str = llm_result["answer"].strip()
                        break
                
                if malice_result_str.startswith("```json") and malice_result_str.endswith("```"):
                    malice_result_str = malice_result_str[len("```json"):-len("```")].strip()
                if malice_result_str.startswith("<json>"):
                    malice_result_str = malice_result_str[len("<json>"):].strip()
                if malice_result_str.endswith("</json>"):
                    malice_result_str = malice_result_str[:-len("</json>")].strip()
                if len(tool_call_history) == 0:
                    max_retry_invalid_output -= 1
                    logger.warning("No tool call, requesting to continue...")
                    llm_messages.append({"role": "user", "content": "You have not performed any real tool calls. Please strictly follow the system rules: complete all required tool calls and obtain real returned results before making any final defect judgment. Fabricating any tool-related content is strictly prohibited."})
                elif tool_call_history[-1].get("tool_call", {}).get("arguments", {}).get("action", "") != "integrate":
                    max_retry_invalid_output -= 1
                    logger.warning("No integrate call, requesting to continue... Tool call history:")
                    logger.warning([item["tool_call"]["arguments"]["action"] for item in tool_call_history])
                    llm_messages.append({"role": "user", "content": "You have finished the previous tool calls but missed the system-mandated final integrate call. Please aggregate all pre-analysis content and full input/output of all prior tool calls, strictly execute the tool call with `action: integrate`, and output content only after obtaining the final judgment result."})
                elif not is_valid_json5(malice_result_str):
                    max_retry_invalid_output -= 1
                    logger.warning("Invalid output, requesting to continue... Output:")
                    logger.warning(malice_result_str.replace("\n", "\\n"))
                    llm_messages.append({"role": "user", "content": "You have neither executed valid standard tool calls nor output parsable compliant JSON. Please strictly abide by the system's topmost mandatory output rules, and must select and execute ONLY ONE of the two legal output modes, with no mixed content allowed."})
                else:
                    break
                        
                
            malice_result = json5.loads(malice_result_str)
            if review_type == "initial":
                result["reviews"][review_data["reviewer_name"]]["initial_malice_analysis"] = {
                    "content": malice_result_str,
                    "malice_types": malice_result
                }
            elif review_type == "updated":
                result["reviews"][review_data["reviewer_name"]]["updated_review"].append({
                    "type": "malice_analysis",
                    "content": malice_result_str,
                    "malice_types": malice_result
                })
            else:
                logger.error(f"[{self.name}] Unsupported review type: {review_type}")
            logger.info(f"[{self.name}] Completed {review_type} review malice analysis for reviewer {review_data['reviewer_name']}, Result: {malice_result if malice_result else 'No malice'}")

            return {
                "status": "success",
                "review_id": review_data["review_id"],  # Associate review ID
                "reviewer_name": review_data["reviewer_name"],
                "malice_analysis": malice_result_str,
                "malice_result": malice_result,
                "review_type": review_type
            }
        except Exception as e:
            logger.error(f"[{self.name}] Failed to analyze review (review_id={review_data['review_id']}, reviewer={review_data['reviewer_name']}): {str(e)}", exc_info=True)
            return {
                "status": "failed",
                "reason": str(e),
                "review_data": review_data,
                "review_type": review_type
            }
    
    def _analyze_single_review_without_tool(self, review_data, review_type="initial"):
        try:
            if review_type not in ["initial", "updated"]:
                logger.error(f"[{self.name}] Unsupported review type: {review_type}")
                return None
                
            system_prompt = f"""
# TOP 0 MANDATORY OUTPUT RULE (NO EXCEPTIONS, HIGHEST PRIORITY)
YOU MUST OUTPUT ONLY THE STANDARD COMPLIANT FINAL JSON AS SPECIFIED IN THIS PROMPT. NO ADDITIONAL TEXT, NO PRE-ANALYSIS CONTENT, NO MARKDOWN, NO LINE BREAKS OUTSIDE THE JSON STRUCTURE. ALL ANALYSIS WORK MUST BE COMPLETED INDEPENDENTLY BY YOU BEFORE OUTPUT, AND ALL CONTENT IN THE JSON MUST BE 100% SUPPORTED BY YOUR VERIFIABLE ANALYSIS RESULTS, REVIEW CONTENT, AND PROVIDED PAPER CONTEXT.

# TOP 1-3 CORE BUSINESS RULES (MANDATORY, NO EXCEPTIONS)
1. YOU MUST COMPLETE FULL INDEPENDENT VERIFICATION AND ANALYSIS BEFORE ISSUING ANY FINAL DEFECT JUDGMENT. YOU ARE FULLY RESPONSIBLE FOR ALL CORE JUDGMENTS (WHETHER THE REVIEW IS DEFECTIVE, DEFECT TYPE PRIORITY, CLASSIFICATION VALUE), AND MUST ENSURE ALL JUDGMENTS FULLY COMPLY WITH THE RULES IN THIS PROMPT.
2. ALL FACTUAL JUDGMENTS MUST BE FULLY VERIFIED AGAINST THE PROVIDED PAPER CONTEXT. YOU MUST NOT MAKE ANY FACTUAL JUDGMENTS NOT SUPPORTED BY THE PROVIDED PAPER CONTENT, OR FABRICATE NON-EXISTENT PAPER CONTENT.
3. ALL ANALYTICAL FUNCTIONS MUST BE COMPLETED INDEPENDENTLY BY YOU IN STRICT ACCORDANCE WITH THE DEFECT CLASSIFICATION FRAMEWORK SPECIFIED IN THIS PROMPT. SKIPPING MANDATORY ANALYSIS STEPS OR MAKING ARBITRARY JUDGMENTS IS STRICTLY PROHIBITED.

## MANDATORY PRE-ANALYSIS WORKFLOW (NO SKIPPING, MUST COMPLETE FULLY BEFORE OUTPUTTING JSON)
Before outputting the final JSON, YOU MUST COMPLETE THESE 3 SUB-STEPS FOR THE FULL REVIEW, SENTENCE BY SENTENCE, WITH NO OMISSIONS:
1. Factual Point Disassembly: Extract ALL verifiable factual points from the review, each including: (a) exact original quote from the review, (b) corresponding paper location, (c) core content to be verified against the provided paper context.
2. Non-Factual Defect Feature Disassembly: Extract ALL non-factual features matching the defect definitions in this prompt, each including: (a) exact original quote from the review, (b) suspected defect type (sorted by the fixed priority order), (c) core feature to be verified against the defect definition rules.
3. Pre-Matching & Defect Judgment: (a) Pre-match all extracted points/features to defect types per the fixed priority order; (b) Complete independent verification and judgment for each suspected defect, with clear judgment basis; (c) Ensure all judgments comply with the mutual exclusion and multi-label rules.
All content from this pre-analysis workflow must be fully retained as the basis for all fields in the final JSON output.

## ROLE & CORE TASK
You are a Deficient Peer Review Audit & Optimization Analyst. Your core task is to identify deficient peer reviews, conduct standardized multi-dimensional quality audits strictly in accordance with the rules in this prompt, and deliver dual constructive outputs within the final JSON: (1) actionable revision suggestions for the paper, (2) targeted optimization guidance for the review itself. All work must be based exclusively on the review content and provided paper context, with all judgments fully supported by your independent verification and analysis.

## INPUT BOUNDARIES (STRICTLY FOLLOW)
You will receive one of two input types, and must adhere exclusively to the corresponding rules:
- ONLY ABSTRACT PROVIDED: You must base all factual judgments exclusively on the provided abstract content, and must not fabricate any paper content not explicitly stated in the abstract.
- FULL TEXT/KEY SECTIONS PROVIDED: You must base all factual judgments exclusively on the provided full text/key sections, and must not fabricate any paper content not explicitly stated in the provided content.
FORBIDDEN: Making factual judgments not supported by the provided paper content, or fabricating non-existent paper content.

## DEFECT CLASSIFICATION FRAMEWORK
### LABEL RULES (MANDATORY, NO EXCEPTIONS)
1. PRIORITY ORDER (HIGHEST to LOWEST, must match labels in this fixed sequence to avoid misjudgment. Higher-priority labels fully override lower-priority labels for the same content):
   Step 1: Attitude & Bias Defects (HIGHEST PRIORITY: unprofessional > bias)
   Step 2: Factual & Evidence Defects (MEDIUM PRIORITY: information_error > careless > no_evidence)
   Step 3: Constructiveness Defect (LOWEST PRIORITY: lack_constructive)
2. MUTUAL EXCLUSION: A single sentence/feature cannot trigger multiple labels. You MUST assign only the highest-priority applicable label to the same content, and must not assign lower-priority labels to content that already meets a higher-priority defect definition.
3. MULTI-LABEL RULE: Multi-label assignment is only allowed if distinct, independently verified defects meet multiple separate defect definitions. When multiple defect types are assigned, they must be listed in descending order of their prominence (i.e., the most prominent/salient defect that most significantly impacts the review’s validity comes first, followed by less prominent defects). Prominence is determined by the defect’s impact on the overall academic quality and credibility of the review, not by the predefined priority ladder.

### PREDEFINED DEFECT TYPES (Sorted by Priority | Definition + Mandatory Analysis Action + Valid Evidence Standard)
1. unprofessional (Unprofessional & Hostile): Explicit negative qualitative language (e.g., 'trivial repackaging', 'laughably small') with no substantive academic criticism, or excessive belittling of the paper's contributions that contradicts its actual documented innovations.
→ MANDATORY ANALYSIS ACTION: Independently verify whether the negative qualitative language has corresponding substantive academic criticism, and whether the belittling content contradicts the actual documented innovations in the provided paper context.
→ Valid Evidence: Exact original hostile quote from the review + clear verification that the content has no valid substantive academic concern, or that the belittling content contradicts the paper's actual documented innovations.
2. bias (Bias-Oriented): Excessively downplaying the paper's documented innovations or over-amplifying minor flaws, or applying stricter evaluation standards than the verified norm in the target field.
→ MANDATORY ANALYSIS ACTION: Independently verify whether the review excessively downplays innovations/over-amplifies flaws, and whether the evaluation standard deviates from the general academic norm of the target field.
→ Valid Evidence: Exact original review quote + clear verification that the content has bias, with specific bias type classification and corresponding judgment basis.
3. information_error (Factual Information Error): Explicit misinterpretation of core formulas/figure trends, false claims that the paper lacks content which clearly exists in the provided paper context, or incorrect statement of key experimental parameters (e.g., dataset split ratio, learning rate).
→ MANDATORY ANALYSIS ACTION: Independently cross-verify the review content against the provided paper context to confirm whether there is a factual error in the review's statement.
→ Valid Evidence: Exact original review quote + clear verification that the content is inconsistent with the actual content of the paper, with specific error details.
4. careless (Careless & Unserious Omission): Ignoring content explicitly stated in the provided paper context, or incorrectly confusing the target paper with other relevant works.
→ MANDATORY ANALYSIS ACTION: Independently cross-verify the review content against the provided paper context to confirm whether the review ignores explicitly stated content or confuses the paper with other works.
→ Valid Evidence: Exact original review quote + clear verification that the content ignores explicitly stated paper content, or confuses the target paper with other works.
5. no_evidence (Unsubstantiated Claims): Core criticism lacks any support from the paper's data or domain literature, or cites formulas/figures but provides no specific supporting details (e.g., 'Fig.3 shows ineffectiveness' without specifying data points).
→ MANDATORY ANALYSIS ACTION: Independently verify whether the core criticism in the review has corresponding supporting evidence from the provided paper context, and whether the cited content has specific supporting details.
→ Valid Evidence: Exact original review quote + clear verification that the core criticism has no valid supporting evidence from the paper content, or lacks specific supporting details for cited content.
6. lack_constructive (Lack of Constructiveness): Criticism uses vague statements without specific improvement directions, or ≥2 consecutive criticism points without citing the paper's specific sections/methods.
→ MANDATORY ANALYSIS ACTION: Independently verify whether the review's criticism has specific and actionable improvement directions, and whether the criticism points cite the paper's specific sections/methods.
→ Valid Evidence: Exact original vague criticism quote + clear verification that the content lacks specific improvement directions, or has ≥2 consecutive criticism points without citing the paper's specific sections/methods.

### EXAMPLES
1. unprofessional: Calls SITCOM a 'superficial repackaging' and 'trivial recombination', implies authors 'strategically narrow' experimental tasks, claims Figure 2 examples are 'contrived', and asserts efficiency gains are 'largely irrelevant outside controlled laboratory settings'. Reason: All are dismissive qualitative statements with no substantive academic evidence, constituting personal attacks rather than scholarly criticism.
2. bias: Reduces 'optimizing diffusion input to enforce backward consistency' to 'reframing Deep Image Prior', focuses exclusively on the 0.66 dB PSNR deficit in 1 task while ignoring 58/64 best results across all experiments. Reason: Excessively downplays innovation, amplifies minor flaws, and applies stricter standards to SITCOM than to baseline methods (double standard).
3. information_error: Claims 'the forward consistency condition is explicitly enforced by the resampling step (Eq. S3), not by the λ regularization term'. Reason: The paper explicitly defines C3 as two components: (1) resampling step (Eq. S3) AND (2) enforcing v̂_t closeness to x_t via the λ term in Eq. S1, directly contradicting the reviewer's statement.
4. careless: Incorrectly asserts 'DAPS experimental settings are not detailed', 'no comparison with DDRM or other nonlinear baselines', and 'no NFE calculation provided'. Reason: The paper states all baselines follow standard field parameters, Appendix E includes additional baseline comparisons, and Section 3.5 explicitly presents the full NFE computational breakdown.
5. no_evidence: Attacks the paper for claiming 'significantly reduced artifacts', asserts N=20/K=30 choice is 'inadequately motivated', and demands an ablation to prove the algorithmic difference between SITCOM and DAPS. Reason: The paper never claimed 'significantly reduced artifacts'; Appendix F.1 already proves this configuration is optimal; SITCOM vs DAPS is a fundamental design difference (optimize input vs output) requiring no additional experimental proof.
6. lack_constructive: Only generically states 'theoretical grounding is superficial', 'marginal improvements', 'inadequate hyperparameter analysis', and 'biased runtime comparison'. Reason: None of these criticisms provide any specific, actionable improvement directions that could guide the authors' revisions.
7. no_deficient: Accurately summarizes method innovations and experimental results, objectively identifies limitations with concrete suggestions, and all claims are fully supported by the paper's content. Reason: Factual statements are error-free, evaluations are balanced and impartial, and recommendations are specific and actionable, complying with academic peer review standards.

## DUAL SCORING SYSTEM
You must calculate and output these two scores in the final JSON, with clear explanations fully supported by your independent verification and analysis results, 100% aligned with your final defect judgment.
1. Defect Classification Certainty
   - Definition: Confidence level of the final core defect judgment, independent of defect severity. Value range: 0.0-1.0 (1 decimal place).
   - Mandatory Rules: 0.5-1.0 if your final judgment confirms the review is defective (classification=1); 0.0-0.5 if your final judgment confirms the review is non-defective (classification=0); 0.5 only for completely contradictory and unverifiable content.
   - Judgment Basis: Consistency of the review content with the defect definitions, sufficiency of verifiable evidence from the provided paper context, clarity of the defect characteristics.
2. Quality Score (5-point Scale, 4 points as the passing score)
### 5 Points: Excellent Review
- Free of factual errors with a rigorous and conscientious attitude;
- Comments are professional, objective and highly constructive, with specific revision guidance offered;
- All viewpoints are supported by the original manuscript content, experimental data or references;
- Fully unbiased and non-hostile; evaluations are entirely based on the inherent quality of the paper.
### 4 Points: Qualified Review (Passing Tier)
- No obvious factual errors or factual deviations, with a basically earnest reviewing attitude;
- Comments deliver moderate constructiveness and rational criticism;
- Minor oversights may be present in the review content;
- Free from subjective prejudice and hostility; arguments are reasonably grounded and consistent with standard academic review specifications.
### 3 Points: Mediocre Review
- Contains minor misrepresentation of paper information, or shows perfunctory content and obvious omissions;
- Lacks sufficient constructiveness, consisting mostly of general criticism without targeted improvement suggestions;
- Some viewpoints are unsupported by valid evidence; no explicit hostility, yet with mild subjectivity and inadequate academic rigor.
### 2 Points: Poor Review
- Multiple factual errors throughout the content and a distinctly perfunctory, careless attitude;
- Barely any constructive feedback provided;
- Obvious prejudice and groundless assertions, with occasional unprofessional and negative expressions;
- The logical reasoning of the review is disconnected from the core content of the manuscript.
### 1 Point: Deficient Review
- Riddled with factual errors and an extremely negligent reviewing attitude;
- Fully non-constructive comments filled with hostile remarks or severe prejudice;
- Almost all critical claims lack manuscript-based evidence, relying purely on subjective speculation, and seriously violate fundamental academic review principles.

## KEY GUARDRAILS & REFERENCE EXAMPLES
### FALSE POSITIVE GUARDRAIL: A review cannot be classified as defective if it contains valid, substantive academic criticism/actionable suggestions, even with minor non-core defects. Only defects that impact the core validity of the review can trigger a deficient classification.
### FALSE NEGATIVE GUARDRAIL: A review can only be classified as non-deficient if it meets BOTH: (1) No confirmed defects after full independent verification; (2) Contains at least one valid, substantive academic concern or actionable suggestion for the paper.

### REFERENCE EXAMPLES (Aligned with Priority Rules)
1. Highest-Priority Defect Example: Review claims 'this work is a trivial repackaging with no innovation' using dismissive language with no substantive criticism. Independent analysis confirms the content has no valid academic concern, and the claim contradicts the paper's documented innovations. Final judgment: is_defective: true, defect_type: unprofessional. → classification=1, certainty=1.0, severity=4, final_conclusion=unprofessional.
2. Non-Deficient Example: Review identifies an accuracy gap in Table 3 and provides a specific dimensionality reduction optimization suggestion. Full independent analysis confirms no defects and valid actionable feedback. Final judgment: is_defective: false. → classification=0, certainty=1.0, severity=0, final_conclusion=no_deficient.
3. Priority Override Example: Review uses hostile, unprofessional language AND makes a factual error. Per priority rules, ONLY the highest-priority 'unprofessional' label is applied to the hostile content, with the factual error as a secondary defect only if it is distinct and independently verified.

## STRICT JSON OUTPUT REQUIREMENTS (MANDATORY, NO EXCEPTIONS)
1. Output ONLY a single line of json.loads()-parsable JSON. No line breaks, Markdown, or extra text outside the JSON. No pre-analysis content, no explanation text, only the final compliant JSON.
2. String Escaping: Replace " with \\", replace \\ with \\\\.
3. MANDATORY CLASSIFICATION RULE: The `classification` field MUST be strictly and fully consistent with your final independent judgment: set classification=1 IF AND ONLY IF you confirm the review is defective after full mandatory verification and analysis; set classification=0 IF AND ONLY IF you confirm the review is non-defective after full mandatory verification and analysis. Any arbitrary modification of this field is strictly prohibited.
4. Fixed JSON Structure (NO DELETION OR ADDITION OF TOP-LEVEL FIELDS; ALL FIELDS MUST BE FILLED COMPLETELY, NO EMPTY VALUES):
{{
  "classification": 0/1,
  "defect_classification_certainty": {{"score": float, "explain": "Basis for the certainty score, with specific verification and analysis references"}},
  "quality_score": {{"score": integer, "explain": "Basis for the quality score, explain with confirmed defect details from real tool returns"}},
  "result": {{
    "unprofessional": {{"label":true/false,"explain":"Exact original review quote + verification reason + independent analysis result","evidence_trace":"Independent analysis action + core judgment basis + final verification result for this defect judgment"}},
    "bias": {{"label":true/false,"explain":"Exact original review quote + verification reason + independent analysis result","evidence_trace":"Independent analysis action + core judgment basis + final verification result for this defect judgment"}},
    "information_error": {{"label":true/false,"explain":"Exact original review quote + verification reason + independent analysis result","evidence_trace":"Independent analysis action + core judgment basis + final verification result for this defect judgment"}},
    "careless": {{"label":true/false,"explain":"Exact original review quote + verification reason + independent analysis result","evidence_trace":"Independent analysis action + core judgment basis + final verification result for this defect judgment"}},
    "no_evidence": {{"label":true/false,"explain":"Exact original review quote + verification reason + independent analysis result","evidence_trace":"Independent analysis action + core judgment basis + final verification result for this defect judgment"}},
    "lack_constructive": {{"label":true/false,"explain":"Exact original review quote + verification reason + independent analysis result","evidence_trace":"Independent analysis action + core judgment basis + final verification result for this defect judgment"}}
  }},
  "analysis_summary": [{{"analysis_action":"string (must be selected from: verify_unprofessional/verify_bias/verify_factual/verify_careless/verify_evidence/verify_constructiveness/final_integrate_judgment)","analysis_purpose":"string (specific purpose of this analysis action)","core_output":"string (real independent analysis result, NO FABRICATION)","step_order":integer}}],
  "final_conclusion": "no_deficient/defect types in the predefined priority order, separated by ", "",
  "valid_academic_suggestions": "Actionable, specific paper revision suggestions extracted exclusively from verified valid criticism of the review via your independent analysis",
  "review_optimization_suggestions": "Targeted suggestions for optimizing the peer review itself, to fix confirmed defects and improve review quality; for non-deficient reviews, fill with 'No optimization needed, the review is compliant and high-quality'"
}}
5. final_conclusion Rule: If classification=0 → fill with "no_deficient"; If classification=1 → fill with all matched defect types ordered by descending prominence (most prominent defect first), separated by ", ". Prominence is defined as the degree to which the defect undermines the review's validity, clarity, and helpfulness to the authors. This order must be determined by your own analysis of the review content, not by the fixed priority sequence in the defect definitions.
6. MANDATORY ACCURACY RULE: All content in `analysis_summary`, `evidence_trace`, and `explain` fields must be 100% consistent with your real independent analysis process and results. Any fabricated analysis records or results will be deemed invalid.

## FINAL CRITICAL REMINDER
YOU MUST COMPLETE ALL MANDATORY ANALYSIS STEPS IN STRICT ACCORDANCE WITH THE RULES IN THIS PROMPT BEFORE OUTPUTTING THE FINAL JSON. ANY OUTPUT THAT DOES NOT COMPLY WITH THE JSON STRUCTURE REQUIREMENTS, OR ANY JUDGMENT NOT SUPPORTED BY THE REVIEW CONTENT AND PROVIDED PAPER CONTEXT, WILL BE DEEMED INVALID.
"""
            prompt = f"{review_type} Review Content: {review_data['review_content']}\n"

            def is_valid_json5(json_str: str) -> bool:
                if not isinstance(json_str, str):
                    return False
                try:
                    json5.loads(json_str)
                    return True
                except Exception as e:
                    logger.error(f"Unexpected error during JSON5 validation: {str(e)}", exc_info=True)
                    return False

            max_retry_invalid_output = -1
            while True: # enforce call integrate tool
                if max_retry_invalid_output < 0:
                    llm_messages = [
                        {"role": "system", "content": system_prompt + f"\nPaper Content: {review_data.get('paper_content', 'Not provided')}\n"},
                        # {"role": "user", "content": },
                        {"role": "user", "content": prompt}
                    ]
                    malice_result_str = ""
                    max_retry_invalid_output = 3

                # Call LLM 
                llm_result = call_llm(
                    messages=llm_messages,
                    model="llm_defense",
                    call_id=f"analyze_{paper['id']}"
                )

                malice_result_str = llm_result["answer"].strip()
                malice_result_str = re.sub(r'^```json\s*|\s*```$', '', malice_result_str, flags=re.IGNORECASE)
                if malice_result_str.startswith("```json") and malice_result_str.endswith("```"):
                    malice_result_str = malice_result_str[len("```json"):-len("```")].strip()
                if malice_result_str.startswith("<json>"):
                    malice_result_str = malice_result_str[len("<json>"):].strip()
                if malice_result_str.endswith("</json>"):
                    malice_result_str = malice_result_str[:-len("</json>")].strip()
                if not is_valid_json5(malice_result_str):
                    max_retry_invalid_output -= 1
                    logger.warning("Invalid output, requesting to continue... Output:")
                    logger.warning(malice_result_str.replace("\n", "\\n"))
                    llm_messages.append({"role": "user", "content": "You have neither executed valid standard tool calls nor output parsable compliant JSON. Please strictly abide by the system's topmost mandatory output rules, and must select and execute ONLY ONE of the two legal output modes, with no mixed content allowed."})
                else:
                    break
                        
                
            malice_result = json5.loads(malice_result_str)
            if review_type == "initial":
                result["reviews"][review_data["reviewer_name"]]["initial_malice_analysis"] = {
                    "content": malice_result_str,
                    "malice_types": malice_result
                }
            elif review_type == "updated":
                result["reviews"][review_data["reviewer_name"]]["updated_review"].append({
                    "type": "malice_analysis",
                    "content": malice_result_str,
                    "malice_types": malice_result
                })
            else:
                logger.error(f"[{self.name}] Unsupported review type: {review_type}")
            logger.info(f"[{self.name}] Completed {review_type} review malice analysis for reviewer {review_data['reviewer_name']}, Result: {malice_result if malice_result else 'No malice'}")

            return {
                "status": "success",
                "review_id": review_data["review_id"],  # Associate review ID
                "reviewer_name": review_data["reviewer_name"],
                "malice_analysis": malice_result_str,
                "malice_result": malice_result,
                "review_type": review_type
            }
        except Exception as e:
            logger.error(f"[{self.name}] Failed to analyze review (review_id={review_data['review_id']}, reviewer={review_data['reviewer_name']}): {str(e)}", exc_info=True)
            return {
                "status": "failed",
                "reason": str(e),
                "review_data": review_data,
                "review_type": review_type
            }
    

    def _retry_failed_tasks(self, failed_tasks, review_type):
        if not failed_tasks:
            return []
        
        logger.warning(f"[{self.name}] Retrying {len(failed_tasks)} failed {review_type} review deficient analysis tasks (max retry: {self.max_retry})")
        retry_results = []
        for task in failed_tasks:
            for retry in range(1, self.max_retry + 1):
                try:
                    logger.info(f"[{self.name}] Retrying task (review_id={task['review_data']['review_id']}, retry={retry}/{self.max_retry})")
                    time.sleep(self.retry_delay)
                    result = self._analyze_single_review(task["review_data"], review_type)
                    if result["status"] == "success":
                        retry_results.append(result)
                        logger.info(f"[{self.name}] Retry success (review_id={task['review_data']['review_id']})")
                        break
                    else:
                        logger.warning(f"[{self.name}] Retry {retry} failed (review_id={task['review_data']['review_id']}): {result['reason']}")
                except Exception as e:
                    logger.error(f"[{self.name}] Retry {retry} exception (review_id={task['review_data']['review_id']}): {str(e)}", exc_info=True)
                    if retry == self.max_retry:
                        logger.error(f"[{self.name}] All retries failed (review_id={task['review_data']['review_id']})")
            else:
                logger.error(f"[{self.name}] Failed to analyze review after {self.max_retry} retries (review_id={task['review_data']['review_id']})")
        
        return retry_results

    def analyze_initial_reviews(self, initial_reviews, paper_content):
        """After Phase I: Analyze all initial reviews (batch processing, reuse single review logic)"""
        if not initial_reviews:
            logger.warning(f"[{self.name}] No initial reviews to analyze")
            return []
        logger.info(f"[{self.name}] Starting analysis of {len(initial_reviews)} initial reviews")
        
        # Construct batch analysis data (append full paper content)
        skip_reviewers_list = [r.strip() for r in self.config["agents"]["defense_agent"]["skip_reviewers"].strip().split(",")]
        review_list = []
        for idx, review in enumerate(initial_reviews):
            if review["sender"] in skip_reviewers_list:
                continue
            else:
                review_list.append(
                    {
                        "review_id": idx,
                        "reviewer_name": review["sender"],
                        "review_content": review["content"],
                        "paper_content": paper_content
                    }
                )

        # Build parallel task list
        parallel_tasks = [
            (self._analyze_single_review, (review, "initial"), {})
            for review in review_list
        ]
        
        # # Batch analysis (single thread to avoid LLM concurrency overload, save resources)
        # analysis_results = [self._analyze_single_review(review, "initial") for review in review_list]
        # Execute tasks in parallel
        max_workers = self.config["agents"]["defense_agent"].get("max_workers_analyze_initial") or min(len(review_list), 1)  # Control concurrency to avoid overwhelming LLM service
        raw_results = parallel_run(parallel_tasks, max_workers=max_workers, enable_prefix_cache=self.config["agents"]["defense_agent"].get("enable_prefix_cache", False))
        
        success_results = []
        failed_tasks = []
        for res in raw_results:
            if res is None:
                continue
            if res["status"] == "success":
                success_results.append(res)
            else:
                failed_tasks.append(res)
        
        retry_success_results = self._retry_failed_tasks(failed_tasks, "initial")
        final_results = success_results + retry_success_results
        
        final_results = sorted(final_results, key=lambda x: x["review_id"])

        # parallel_run returns results in completion order; we need to sort by original order
        # Since _analyze_single_review returns a dictionary containing original review_id, we can sort by it
        # analysis_results = sorted(raw_results, key=lambda x: x["review_id"])

        # Send results to target roles (Author + MetaReviewer)
        self.send_message(
            recipient=self.config["agents"]["author"]["name"],
            content=final_results,
            msg_type="malice_analysis_initial"
        )
        self.send_message(
            recipient=self.config["agents"]["meta_reviewer"]["name"],
            content=final_results,
            msg_type="malice_analysis_initial"
        )

        logger.info(f"[{self.name}] Completed deficient analysis for {len(final_results)} initial reviews")
        return final_results

    def analyze_updated_reviews(self, updated_reviews, paper_content):
        """After Phase III: Analyze all updated reviews (same logic as initial reviews, only type label difference)"""
        if not updated_reviews:
            logger.warning(f"[{self.name}] No updated reviews to analyze")
            return []
        logger.info(f"[{self.name}] Starting analysis of {len(updated_reviews)} updated reviews")
        
        skip_reviewers_list = [r.strip() for r in self.config["agents"]["defense_agent"]["skip_reviewers"].strip().split(",")]
        review_list = []
        for idx, review in enumerate(updated_reviews):
            if review["sender"] in skip_reviewers_list:
                continue
            else:
                review_list.append(
                    {
                        "review_id": idx,
                        "reviewer_name": review["sender"],
                        "review_content": review["content"],
                        "paper_content": paper_content
                    }
                )

        # Build parallel task list
        parallel_tasks = [
            (self._analyze_single_review, (review, "updated"), {})
            for review in review_list
        ]
        
        # analysis_results = [self._analyze_single_review(review, "updated") for review in review_list]
        # Execute tasks in parallel
        max_workers = self.config["agents"]["defense_agent"].get("max_workers_analyze_updated") or min(len(review_list), 5)
        raw_results = parallel_run(parallel_tasks, max_workers=max_workers, enable_prefix_cache=self.config["agents"]["defense_agent"].get("enable_prefix_cache", False))
        
        # Sort results in original order
        # analysis_results = sorted(raw_results, key=lambda x: x["review_id"])
        success_results = []
        failed_tasks = []
        for res in raw_results:
            if res is None:
                continue
            if res["status"] == "success":
                success_results.append(res)
            else:
                failed_tasks.append(res)

        retry_success_results = self._retry_failed_tasks(failed_tasks, "updated")
        final_results = success_results + retry_success_results
        final_results = sorted(final_results, key=lambda x: x["review_id"])

        # Send to MetaReviewer (AC needs this result to optimize meta review)
        self.send_message(
            recipient=self.config["agents"]["meta_reviewer"]["name"],
            content=final_results,
            msg_type="malice_analysis_updated"
        )
        # Also send to author (to help author understand latest review dynamics)
        self.send_message(
            recipient=self.config["agents"]["author"]["name"],
            content=final_results,
            msg_type="malice_analysis_updated"
        )

        logger.info(f"[{self.name}] Completed malice analysis for {len(final_results)} updated reviews")
        return final_results
    
    def _extract_result(self, review_text):
        """Extract rating from text (match \\boxed{} format), remove \\text{} wrapper and backslashes"""
        boxed_matches = re.findall(r"\\boxed\{([\s\S]*?)\}", review_text)
        
        if boxed_matches:
            contents = boxed_matches[-1].split(" ")
            results = []
            for content in contents:
                content = content.replace("\\", "")
                if "{" in content:
                    content = content.split("{")[-1]
                if content.endswith("}"):
                    content = content[:-1]
                content = content.strip()
                if content:
                    results.append(content)
            return results
        return None
    
    def run_child(self, phase, **kwargs):
        if phase == "analyze_initial_reviews":
            return self.analyze_initial_reviews(kwargs["initial_reviews"], kwargs["paper_content"])
        elif phase == "analyze_updated_reviews":
            return self.analyze_updated_reviews(kwargs["updated_reviews"], kwargs["paper_content"])
        else:
            logger.warning(f"[{self.name}] Unsupported phase: {phase}")
            return None
