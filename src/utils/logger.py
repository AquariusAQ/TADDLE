import logging
import os
from datetime import datetime
from typing import Optional
import json
from openai.types.chat import ChatCompletionMessage

class LogManager:
    def __init__(self):
        self.log_dir: Optional[str] = None
        self.message_log_path: Optional[str] = None
        self._logger: Optional[logging.Logger] = None
        self._vllm_logger: Optional[logging.Logger] = None
        self._search_logger: Optional[logging.Logger] = None

    @staticmethod
    def _create_logger(name: str, log_file: str) -> logging.Logger:
        """Create independent logger instance (avoid log confusion)"""
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # File handler (write to corresponding log file)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            # Console handler (keep terminal output for real-time debugging)
            console_handler = logging.StreamHandler()

            # Unified log format: Time|Module|Level|Content (facilitates traceability)
            formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    def init_log_dir(self, log_name: Optional[str] = None, paper_id: Optional[str] = None):
        """Initialize log folder and logger instances"""
        log_root = "outputs"
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        
        if log_name:
            # current_log_dir = os.path.join(log_root, f"{log_name}_{current_timestamp}", paper_id) if paper_id else os.path.join(log_root, log_name)
            current_log_dir = os.path.join(log_root, log_name, paper_id) if paper_id else os.path.join(log_root, log_name)
        else:
            current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_log_dir = os.path.join(log_root, current_timestamp, paper_id) if paper_id else os.path.join(log_root, current_timestamp)

        if not os.path.exists(current_log_dir):
            os.makedirs(current_log_dir)
        self.log_dir = current_log_dir

        self.message_log_path = os.path.join(current_log_dir, "message.jsonl")
        # 1. General logs: Record Agent initialization, process stages, tool calls, and other non-VLLM logs
        self._logger = self._create_logger("main", os.path.join(current_log_dir, "main.log"))
        # 2. Combined VLLM logs: Both Input and Output are written to this file
        self._vllm_logger = self._create_logger("vllm", os.path.join(current_log_dir, "vllm_interaction.log"))
        self._search_logger = self._create_logger("search", os.path.join(current_log_dir, "search_interaction.log"))

    def save_json(self, data: dict, filename: str, path: str = ""):
        """Save data as JSON file"""
        if self.log_dir is None:
            raise RuntimeError("LogManager has not been initialized. Call init_log_dir() first.")
        if path:
            file_dir = os.path.join(self.log_dir, path)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
        else:
            file_dir = self.log_dir
        file_path = os.path.join(file_dir, filename)

        def _serialize(obj):
            """
            Custom JSON serialization function.
            The json library will call this function when encountering unrecognized objects.
            """
            # Check if the object is of type ChatCompletionMessage
            if ChatCompletionMessage and isinstance(obj, ChatCompletionMessage):
                return obj.model_dump()
            
            # Add more checks for other custom types
            # elif isinstance(obj, AnotherCustomType):
            #     return obj.to_dict()

            # If none of the above match, raise the original TypeError
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4, default=_serialize)

    def save_markdown(self, content: str, filename: str, path: str = ""):
        """Save text content as Markdown file"""
        if self.log_dir is None:
            raise RuntimeError("LogManager has not been initialized. Call init_log_dir() first.")
        # Handle save path
        if path:
            file_dir = os.path.join(self.log_dir, path)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
        else:
            file_dir = self.log_dir
        file_path = os.path.join(file_dir, filename)
        # Write to Markdown file (preserve original format, UTF-8 encoding)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        self.info(f"Markdown file saved to: {file_path}")  # Record save log

    def __getattr__(self, name):
        """Proxy log method calls to internal logger"""
        if self._logger is None:
            raise RuntimeError("LogManager has not been initialized. Call init_log_dir() first.")
        return getattr(self._logger, name)

    @property
    def vllm(self) -> logging.Logger:
        """Get VLLM-specific logger"""
        if self._vllm_logger is None:
            raise RuntimeError("LogManager has not been initialized. Call init_log_dir() first.")
        return self._vllm_logger
    
    @property
    def search(self) -> logging.Logger:
        """Get external Search-specific logger"""
        if self._search_logger is None:
            raise RuntimeError("LogManager has not been initialized. Call init_log_dir() first.")
        return self._search_logger


# Create global instance for direct import and use in other scripts
logger = LogManager()

# import logging
# import os
# from datetime import datetime


# # Global variable: Log folder path (shared by all modules)
# message_log_path = None
# logger = None
# vllm_logger = None

# def _create_logger(name, log_file):
#     """Create independent logger instance (avoid log confusion)"""
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.INFO)
    
#     if not logger.handlers:
#         # File handler (write to corresponding log file)
#         file_handler = logging.FileHandler(log_file, encoding="utf-8")
#         # Console handler (keep terminal output for real-time debugging)
#         console_handler = logging.StreamHandler()
        
#         # Unified log format: Time|Module|Level|Content (facilitates traceability)
#         formatter = logging.Formatter(
#             "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
#             datefmt="%Y-%m-%d %H:%M:%S"
#         )
#         file_handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)
        
#         logger.addHandler(file_handler)
#         logger.addHandler(console_handler)
    
#     return logger

# def init_log_dir(log_name=None, paper_id=None):
#     """Initialize log folder (ensure directory exists to avoid file write failures)"""
#     global message_log_path, logger, vllm_logger
#     log_root = "logs"
#     if not os.path.exists(log_root):
#         os.makedirs(log_root)

#     if log_name:
#         current_log_dir = os.path.join(log_root, log_name, paper_id)  # Log folder path
#     else:
#         current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         current_log_dir = os.path.join(log_root, current_timestamp, paper_id)  # Log folder path
#     if not os.path.exists(current_log_dir):
#         os.makedirs(current_log_dir)

#     message_log_path = os.path.join(current_log_dir, "message.jsonl")  # Message JSONL path
#     # 1. General logs: Record Agent initialization, process stages, tool calls, and other non-VLLM logs
#     logger = _create_logger("main", os.path.join(current_log_dir, "main.log"))
#     # 2. Combined VLLM logs: Both Input and Output are written to this file
#     vllm_logger = _create_logger("vllm", os.path.join(current_log_dir, "vllm_interaction.log"))

# # Initialize log folder (automatically executed when module is loaded to ensure normal file writing later)
# # init_log_dir()