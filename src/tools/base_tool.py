import functools
from abc import ABC, abstractmethod
from typing import Dict, Any
from utils.logger import logger
import copy  # Used for deep copy to avoid contamination of original data (optional, adjust based on actual scenarios)

def truncate_content(content):
    """
    Unified content truncation logic:
    - String: Truncate to 2000 characters, add ellipsis if exceeded
    - List: Iterate through elements, truncate image_url's url to 21 characters, keep other types unchanged
    - Other types: Return as-is
    """
    # (f"Please read this paper main text:\n"
    #                                         f"{paper['main_text']}\n\n"
    #                                         f"Your goal: 
    # Handle string-type content
    return content
    if isinstance(content, str):
        max_str_len = 500
        if "Your goal:" in content:
            content = "Please read this paper main text: <Paper content> " + content.split("Your goal:")[1]
        if len(content) > max_str_len:
            return content[:max_str_len] + "..."
        return content
    
    # Handle list-type content (scenarios containing image_url)
    elif isinstance(content, list):
        processed_elems = []
        for elem in content:
            try:
                # Deep copy to avoid modifying original dictionary references (shallow copy elem.copy() is also acceptable for simple scenarios)
                elem_copy = copy.deepcopy(elem) if isinstance(elem, dict) else elem
                
                # Only process elements of type image_url
                if isinstance(elem_copy, dict) and elem_copy.get("type") == "image_url":
                    image_url_info = elem_copy.get("image_url", {})
                    if isinstance(image_url_info, dict) and "url" in image_url_info:
                        # Truncate url to 21 characters (take first 21 characters regardless of original length)
                        image_url_info["url"] = image_url_info["url"][:21]
                
                processed_elems.append(elem_copy)
            except Exception:
                # Keep original value if processing fails for a single element
                processed_elems.append(elem)
        return processed_elems
    
    # Non-string/list types (e.g., None, numbers), return as-is
    else:
        return content

class NamedABC(ABC):
    # Class attribute to track the number of created instances, shared by all subclass instances
    _instance_counter = 0

    def __init__(self):
        # Increment counter each time a new instance is created
        # Use class name to build counter key to ensure independent counting for different subclasses
        counter_key = f"_{self.__class__.__name__}_counter"
        if not hasattr(NamedABC, counter_key):
            setattr(NamedABC, counter_key, 0)

        call_idx_key = f"_{self.__class__.__name__}_call_idx"
        if not hasattr(NamedABC, call_idx_key):
            setattr(NamedABC, call_idx_key, 0)
        self.call_idx_key = call_idx_key
        
        instance_counter = getattr(NamedABC, counter_key) + 1
        setattr(NamedABC, counter_key, instance_counter)
        
        # Generate a unique name for the instance and save it as an instance attribute
        # self.name = f"{self.__class__.__name__}_{instance_counter}"
        self.name = f"{self.__class__.__name__}"

def post_execute_decorator(func):
    """
    A decorator used to execute common code after function execution.
    """
    @functools.wraps(func)  # Preserve original function metadata (e.g., name, docstring)
    def wrapper(self, action: str, **kwargs) -> Any:
        result = func(self, action,** kwargs)

        call_idx = getattr(NamedABC, self.call_idx_key)

        if isinstance(result, tuple):
            result, history, reviewer_name = result
            for idx, item in enumerate(history):
                try:
                    # Case 1: item is an object with content as an attribute
                    if hasattr(item, "content"):
                        original_content = item.content
                        processed_content = truncate_content(original_content)
                        history[idx].content = processed_content
                    
                    # Case 2: item is a dictionary with content as a key-value pair
                    elif isinstance(item, dict) and "content" in item:
                        original_content = item["content"]
                        processed_content = truncate_content(original_content)
                        history[idx]["content"] = processed_content

                    
                
                except Exception as e:
                    # Optional: Log exceptions for troubleshooting (recommended to keep)
                    # logger.error(f"Failed to process history item {idx}: {str(e)}")
                    pass
                
            # Save processed history
            logger.save_json(history, f"{self.name}_history_{reviewer_name}_{call_idx}.json", path="tool_history")

        setattr(NamedABC, self.call_idx_key, call_idx + 1)
        
        return result
    return wrapper

class BaseTool(NamedABC):
    """Base Tool Class: Unified metadata and call interface"""
    # Tool metadata (must be overridden by subclasses for classification and LLM call configuration)
    tool_metadata: Dict[str, Any] = {
        "tool_type": "",          # Required: direct (called directly by framework) / llm_driven (driven by LLM)
        "tool_name": "",          # Required: Unique tool identifier (consistent with configuration)
        "description": "",        # Required: Tool function description for LLM
        "parameters": {           # Required: Parameter specification (compliant with JSON Schema for LLM)
            "type": "object",
            "properties": {},
            "required": []
        }
    }

    def __init_subclass__(cls, **kwargs):
        """
        This method is automatically called when a subclass inherits from BaseTool.
        We automatically add the decorator to the `call` method of the subclass here.
        """
        super().__init_subclass__(**kwargs)
        
        # Check if the subclass has implemented the `call` method
        if hasattr(cls, 'call'):
            # Wrap the subclass's `call` method with the decorator
            cls.call = post_execute_decorator(cls.call)

    @abstractmethod
    def call(self, action: str, **kwargs) -> Any:
        """Core tool execution method (must be implemented by subclasses)"""
        pass