import os
from utils.logger import logger
from utils.llm_client import call_llm  # Import LLM invocation tool
from utils.text_tokenizer import truncate_text, count_tokens
from utils.config import config
from utils.shared_data import paper  # Import thread-safe global variable
from .base_tool import BaseTool
import json

class PaperProcessor(BaseTool):
    """
    Paper Processing Tool: Load and parse dataset papers
    Override metadata: Mark as direct invocation by framework
    """
    tool_metadata = {
        "tool_type": "direct",
        "tool_name": "paper_processor",
        "description": "Load dataset paper content or extract paper abstract, a necessary prerequisite operation for the review process",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Tool operation, supports 'load' (load paper) and 'extract_abstract' (extract abstract)"
                },
                "dataset_name": {"type": "string", "description": "Dataset name"},
                "paper_id": {"type": "string", "description": "Unique paper ID"},
                "paper_content": {"type": "string", "description": "Loaded paper content (required when extracting abstract)"}
            },
            "required": ["action"]
        }
    }
    
    def call(self, action,** kwargs):
        """Unified invocation entry"""
        if action == "load":
            logger.info("loading main text")
            paper_main_text = self._load_paper(load_type="main_text", **kwargs)
            logger.info("loading reference")
            paper_reference = self._load_paper(load_type="reference", **kwargs)
            logger.info("loading appendix summary")
            paper_appendix_summary = self._load_paper(load_type="appendix_summary", **kwargs)
            logger.info("loading image")
            paper_image_summary, paper_image_all = self._read_image(content=paper_main_text, **kwargs)
            logger.info("all loaded")
            paper["main_text"] = paper_main_text
            paper["reference"] = paper_reference
            paper["appendix"] = paper_appendix_summary.split("</think>")[-1]
            paper["image"] = paper_image_summary
            paper.update({"images": paper_image_all})
            paper["content"] = f"{paper["main_text"]}\n\n## Image summary\n\n{paper["image"]}\n\n## Appendix summary\n\n{paper["appendix"]}"
            return paper["content"] 
        # if action == "load_reference":
        #     paper_content = self._load_paper(load_type="reference", **kwargs)
        #     return paper_content
        # if action == "load_appendix":
        #     paper_content = self._load_paper(load_type="appendix_summary", **kwargs)
        #     return paper_content
        elif action == "extract_title":
            # Prioritize using passed paper_content, load if not provided
            kwargs["paper_content"] = kwargs.get("paper_content") or self._load_paper(load_type="main_text", **kwargs)
            return self._extract_title_from_content(**kwargs)
        elif action == "extract_abstract":
            # Prioritize using passed paper_content, load if not provided
            kwargs["paper_content"] = kwargs.get("paper_content") or self._load_paper(load_type="main_text", **kwargs)
            return self._extract_abstract(**kwargs)
        # elif action == "read_images":
        #     # Prioritize using passed paper_content, load if not provided
        #     kwargs["paper_content"] = kwargs.get("paper_content") or self._load_paper(load_type="main_text", **kwargs)
        #     return self._read_images(**kwargs)
        else:
            return f"Unsupported operation: {action}"
    
    def _load_paper(self, dataset_name, paper_id, load_type="main_text"):
        """Load paper content from dataset"""
        if load_type=="main_text" and paper.get("content", None):
            return paper.get("content")
        # Construct paper path
        paper_path = os.path.join(
            "datasets", 
            dataset_name, 
            paper_id, 
            f"{paper_id}_{load_type}.md"
        )
        
        # Check if file exists
        if not os.path.exists(paper_path):
            error_msg = f"Paper {load_type} file does not exist: {paper_path}"
            logger.error(error_msg)
            return error_msg
        
        # Read paper content
        try:
            with open(paper_path, "r", encoding="utf-8") as f:
                content = f.read()              
            if config.get("tokenizer", {}).get("model_path", None):
                content = truncate_text(text = content, 
                                        max_length = config.get("tokenizer", {}).get("max_paper_token", 50000),
                                        model_path = config.get("tokenizer", {}).get("model_path")
                                        )
            if load_type=="main_text":
                logger.info(f"Successfully loaded paper: {dataset_name}/{paper_id} (Length: {len(content)} characters)")
            return content
        except Exception as e:
            error_msg = f"Failed to read paper: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def _extract_title_from_content(self, paper_content, **kwargs):
        """
        Extract title from content:
        - If content is empty/non-string → return None
        - If first line starts with # → extract content after # (trim whitespace)
        - If first line does not start with # → return None
        """
        if not isinstance(paper_content, str) or not paper_content.strip():
            return None
        
        # Get first line of content (handle line breaks \n/\r\n)
        first_line = paper_content.strip().split('\n')[0].strip()
        # Check if starts with #
        if first_line.startswith('#'):
            # Remove # and subsequent whitespace to get pure title
            return first_line[1:].strip()
        
        # Return None if not starting with #
        return None

    def _extract_abstract(self, paper_content,** kwargs):
        """Extract abstract from paper content (simplified implementation)"""
        if paper and kwargs.get("paper_id") and kwargs.get("paper_id") == paper.get("id") and paper.get("abstract"):
            logger.info(f"Abstract for paper {kwargs.get('paper_id')} already extracted, returning cached content directly")
            return paper["abstract"]

        # 1. Prioritize formatted extraction (Markdown standard format)
        for abstract_marker in ["## Abstract", "## ABSTRACT"]:
            if abstract_marker in paper_content:
                # Split abstract content (from Abstract title to next level 2 heading or before <div tag)
                abstract = paper_content.split(abstract_marker)[1].split("##")[0].split("<div")[0].strip()
                logger.info("Successfully obtained abstract through formatted extraction")
                return abstract
        
        for degrade_marker in ["##", "<div"]:
            if degrade_marker in paper_content:
                # Split abstract content (from title to next level 2 heading or before <div tag)
                abstract = paper_content.split(degrade_marker)[0].strip()
        
        if len(abstract) < len(paper_content) and len(abstract) > 20 and len(abstract) < 5000:
            logger.info("Successfully obtained abstract through degraded formatted extraction")
            return abstract

        # 2. If formatted extraction fails, call LLM for intelligent extraction
        logger.warning(f"Standard formatted Abstract section not found for paper {kwargs.get('paper_id', '')}, will call LLM to attempt intelligent extraction")

        try:
            # Construct LLM prompt (clear task + context + format requirements)
            system_prompt = """
            You are an academic paper processing expert tasked with extracting abstracts from full paper content.
            If there is an abstract section in the full text, return the content of that section directly. Otherwise, generate an abstract for it.
            Please follow these rules:
            1. The abstract should include four core parts: research purpose, methods, main results, and conclusions
            2. Maintain the complete meaning of the original text without adding additional explanations
            3. Output format is plain text with no Markdown tags
            4. Return "Unable to extract abstract" if abstract content cannot be identified
            """
            
            user_prompt = f"""
            Please extract the abstract from the following paper content:
            ---Start of paper content---
            {paper_content}
            ---End of paper content---
            
            Please return the extracted abstract text directly:
            """
            
            # Call LLM (refer to the given invocation method)
            llm_result = call_llm(
                prompt=user_prompt,
                system=system_prompt,
                model="llm_tools",
                temperature=0.2  # Low temperature ensures stability of extraction results
            )
            
            # Parse LLM return result
            abstract = llm_result.get("answer", "").strip()
            
            # Result validation
            if not abstract:
                logger.error("LLM failed to extract valid abstract")
                return "Abstract section not found, and LLM intelligent extraction failed"
            
            logger.info("LLM abstract extraction successful")
            return abstract
            
        except Exception as e:
            # Catch LLM invocation exceptions
            error_msg = f"Error occurred during LLM abstract extraction: {str(e)}"
            logger.error(error_msg)
            return f"Abstract section not found, LLM invocation failed: {str(e)}"
        
    def _read_image(self, dataset_name, paper_id, content=None):
        """Load paper content from dataset"""
        if content == None and paper.get("content", None):
            return paper.get("content")
        # Construct paper path
        paper_path = os.path.join(
            "datasets", 
            dataset_name, 
            paper_id, 
            f"image_analysis.json"
        )
        
        # Check if file exists
        if not os.path.exists(paper_path):
            error_msg = f"Paper image analyze file does not exist: {paper_path}"
            logger.error(error_msg)
            return error_msg
        
        # Read paper content
        try:
            image_summaries = ""
            with open(paper_path, "r", encoding="utf-8") as f:
                image_all = json.load(f)
                for key in image_all:
                    if key in content:
                        image_summaries += f"Image name: {key}, summary: {image_all[key]}\n"
            return image_summaries, image_all
        except Exception as e:
            error_msg = f"Failed to read paper: {str(e)}"
            logger.error(error_msg)
            return error_msg
    