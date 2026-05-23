from .base_tool import BaseTool
from utils.llm_client import call_llm
from utils.logger import logger
from utils.config import config
import os

class ImageAnalyzer(BaseTool):
    """
    Image Analysis Tool: Read local images and call VLM to generate analysis results
    Supports Agents passing in image paths and query instructions, returns analytical answers from multimodal models
    """
    tool_metadata = {
        "tool_type": "llm_driven",  # LLM-driven invocation
        "tool_name": "image_analyzer",
        "description": "Read local image files, call Vision-Language Multimodal Model (VLM) to analyze image content, and return parsing results that comply with query instructions.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Tool operation, only 'analyze' (analyze images) is supported"
                },
                "image_paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of local image file paths (use relative paths, e.g., ['./imgs/img_1.jpg', './imgs/img_2.jpg'])"
                },
                "query": {
                    "type": "string",
                    "description": "Query instruction for images (e.g., 'Extract all experimental result data in the image', 'Explain the complete flowchart in the image in detail', 'Extract xx data from the image')"
                }
            },
            "required": ["action", "image_paths", "query"]
        }
    }
    
    def call(self, action,** kwargs):
        if action != "analyze":
            return f"Unsupported operation: {action}, only 'analyze' (analyze images) is supported"
        
        # 1. Extract and validate parameters
        image_paths = kwargs.get("image_paths", [])
        model_vlm = kwargs.get("model_vlm", "vlm")
        query = kwargs.get("query", "").strip()
        
        if not image_paths:
            return "Error: image_paths cannot be empty, at least one image path must be provided"
        if not query:
            return "Error: query cannot be empty, a query instruction for the images must be provided"
        
        # 2. Validate if image paths exist
        valid_paths = []
        invalid_paths = []
        for path in image_paths:
            # abs_path = os.path.abspath(path)
            abs_path = os.path.join("./datasets", config["dataset"]["default_name"], config["dataset"]["default_paper_id"], path)
            if os.path.exists(abs_path) and os.path.isfile(abs_path):
                valid_paths.append(abs_path)
            else:
                invalid_paths.append(path)
        if invalid_paths:
            logger.warning(f"Invalid image paths: {', '.join(invalid_paths)}")
        if not valid_paths:
            return f"Error: All image paths are invalid (invalid paths: {', '.join(invalid_paths)})"
        
        # 3. Call VLM to analyze images
        logger.info(f"[ImageAnalyzer] Starting image analysis: {valid_paths}, Query instruction: {query}")
        try:
            # Call multimodal LLM (using the globally configured model in the framework)
            vlm_result = call_llm(
                image_paths=valid_paths,
                image_query=query,
                model=model_vlm,
                system="You are an expert in image query analysis. You must generate accurate and concise academic query results strictly based on image content and query instructions."
            )
            answer = vlm_result["answer"]
            logger.info(f"[ImageAnalyzer] Image analysis completed, result length: {len(answer)} characters")
            message = vlm_result["messages"]
            message.append({"role": "assistant", "content": answer})
            return f"Image analysis results:\n{answer}\n\n(Analyzed image paths: {', '.join(valid_paths)})", vlm_result["messages"]
        except Exception as e:
            error_msg = f"Image analysis failed: {str(e)}"
            logger.error(f"[ImageAnalyzer] {error_msg}", exc_info=True)
            return error_msg