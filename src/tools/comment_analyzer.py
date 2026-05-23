from .base_tool import BaseTool

class CommentAnalyzer(BaseTool):
    """
    Extract points of doubt from reviews to support targeted rebuttals by authors
    Override metadata: Mark as LLM-driven and define descriptions/parameters required by LLM
    """
    tool_metadata = {
        "tool_type": "llm_driven",
        "tool_name": "comment_analyzer",
        "description": "Extract points of doubt from review comments and classify them into 'misunderstanding/defense/reasonable criticism' categories to support targeted rebuttals by authors",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Tool operation, only 'extract_issues' (extract points of doubt) is supported"
                },
                "comments": {
                    "type": "string",
                    "description": "Complete review comments to be analyzed"
                #     "type": "array",
                #     "items": {"type": "string"},
                #     "description": "List of review comments to be analyzed (each element is a single review)"
                }
            },
            "required": ["action", "comments"]
        }
    }

    def call(self, action,** kwargs):
        if action == "extract_issues":
            return self._extract_issues(kwargs["comments"])
        else:
            return f"Unsupported operation: {action}"

    def _extract_issues(self, comments):
        """Extract key points of doubt from review list"""
        from utils.llm_client import call_llm
        
        # Call LLM to extract points of doubt (structured output)
        # {chr(10).join([f"Review {i+1}：{c}" for i, c in enumerate(comments)])}
        prompt = f"""The following is a review comment. Please extract all points of doubt and classify them into "misunderstanding/defense/reasonable criticism" categories:
        {comments}
        Output format:
        Misunderstanding-based questions:
        - Review X: Specific question description
        Defense-based questions:
        - Review Y: Specific question description
        Reasonable criticisms:
        - Review Z: Specific question description
        """
        
        return call_llm(
            prompt=prompt,
            system="You are a review comment analysis tool that accurately extracts and classifies points of doubt",
            model="llm_tools"
        )