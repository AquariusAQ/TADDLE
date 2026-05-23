from utils.llm_client import call_llm
from utils.message import global_message_queue
from tools import *  # Import all tools
import yaml
from utils.logger import logger  # Import logger
from tools.base_tool import BaseTool  # New import
from typing import Optional, List, Dict  # New type annotations

class BaseAgent:
    def __init__(self, agent_type, name, config):
        self.agent_type = agent_type  # Role type (author/reviewer/ac)
        self.name = name
        self.config = config  # Global configuration
        self.message_queue = global_message_queue # Use global message queue
        self.direct_tools: Dict[str, BaseTool] = {}  # Tools called directly by the framework
        self.llm_driven_tools: Dict[str, BaseTool] = {}  # Tools called by LLM-driven invocation
        self._current_allowed_tools: List[str] = []  # Tools allowed in the current phase (temporary attribute, default empty list = none allowed)
        self._init_tools()  # Tool instance dictionary
        logger.info(f"Initialized Agent: {self.name} (Type: {self.agent_type})")  # Log creation
    
    def _init_tools(self):
        """Initialize available tools (fixed class name conversion logic)"""
        for tool_name, tool_config in self.config["tools"].items():
            if tool_config["enabled"]:
                # Key fix: Convert snake_case to CamelCase (e.g., paper_processor → PaperProcessor)
                camel_case_name = ''.join(word.capitalize() for word in tool_name.split('_'))
                # Example: "paper_processor" → ["paper", "processor"] → capitalize each word → "PaperProcessor"
                
                # Load tool class
                try:
                    tool_class = globals()[camel_case_name]
                    tool_instance: BaseTool = tool_class()  # Type annotation as BaseTool
                except KeyError:
                    raise ValueError(f"Tool class {camel_case_name} not found")

                # Classify and store by metadata
                tool_type = tool_instance.tool_metadata["tool_type"]
                if tool_type == "direct":
                    self.direct_tools[tool_name] = tool_instance
                elif tool_type == "llm_driven":
                    self.llm_driven_tools[tool_name] = tool_instance
                logger.debug(f"Initialized tool: {tool_name} (Type: {tool_type})")

    # New: Set allowed tools for the current phase (called by subclass run)
    def _set_allowed_tools(self, allowed_tools: Optional[List[str]] = None):
        self._current_allowed_tools = allowed_tools or []  # Default empty list
        logger.info(f"[{self.name}] Tools allowed in current phase: {self._current_allowed_tools if self._current_allowed_tools else 'None'}")

    # New: Clear allowed tools for the current phase (avoid affecting subsequent runs)
    def _clear_allowed_tools(self):
        self._current_allowed_tools = []
        logger.debug(f"[{self.name}] Cleared tool permissions for current phase")

    def get_llm_tool_config(self) -> list:
        """New: Generate LLM tool call request format (refer to user example)"""
        llm_tools = []
        # Return only "allowed tools" that are of "llm_driven" type
        for tool_name, tool in self.llm_driven_tools.items():
            if tool_name in self._current_allowed_tools:
                meta = tool.tool_metadata
                llm_tools.append({
                    "type": "function",
                    "function": {
                        "name": meta["tool_name"],
                        "description": meta["description"],
                        "parameters": meta["parameters"]
                    }
                })
        logger.debug(f"[{self.name}] Number of LLM-callable tools in current phase: {len(llm_tools)}")
        return llm_tools
    
    def get_llm_tool_names(self) -> list:
        """Get list of LLM-callable tool names for the current phase"""
        tool_names = []
        for tool_name, tool in self.llm_driven_tools.items():
            if tool_name in self._current_allowed_tools:
                tool_names.append(tool.tool_metadata["tool_name"])
        return tool_names

    def send_message(self, recipient, content, msg_type="comment"):
        """Send message to specified Agent"""
        self.message_queue.send(
            sender=self.name,
            recipient=recipient,
            content=content,
            msg_type=msg_type
        )
        logger.info(f"[{self.name}] Sent message to [{recipient}] (Type: {msg_type})")

    def receive_messages(self, msg_type=None):  # New msg_type parameter, default None
        """Receive messages, support filtering by message type"""
        logger.debug(f"[{self.name}] Attempting to receive messages (Type: {msg_type})")  # New log
        messages = self.message_queue.receive(
            recipient=self.name,
            msg_type=msg_type  # Pass msg_type to message queue
        )
        if messages:
            logger.info(f"[{self.name}] Received {len(messages)} messages (Type: {msg_type or 'All'})")
        else:
            # New log: Print reason when no messages are received
            logger.debug(f"[{self.name}] No messages received → Recipient: {self.name}, Type: {msg_type}")
        return messages

    def call_tool(self, tool_name, **kwargs):
        """Call tool (unified interface)"""

        # 1. First validate if the tool is in the allowed list for the current phase
        if tool_name not in self._current_allowed_tools:
            error_msg = f"Tool {tool_name} is not allowed in the current phase (Allowed tools: {self._current_allowed_tools})"
            logger.error(f"[{self.name}] {error_msg}")
            return error_msg
        
        # 2. Validate tool existence
        tool = self.direct_tools.get(tool_name) or self.llm_driven_tools.get(tool_name)
        if not tool:
            error_msg = f"Tool {tool_name} is not enabled"
            logger.error(f"[{self.name}] {error_msg}")
            return error_msg
        
        logger.info(f"[{self.name}] [Tool call] Calling tool: {tool_name} (Parameters: {str(kwargs)[:100]}...)")
        # Directly call the call method of the tool instance (all tool classes implement the call interface)
        try:
            result = tool.call(**kwargs)  # Core fix: Remove super() call, directly call the tool's call method
        except Exception as e:
            error_msg = f"Failed to call tool {tool_name}: {str(e)}"
            logger.error(f"[{self.name}] {error_msg}", exc_info=True)
            return error_msg
        
        logger.info(f"[{self.name}] [Tool call] Tool returned: {str(result)[:200].replace("\n", " ")}...")  # Truncate long results
        return result

    def run(self, phase, allowed_tools: Optional[List[str]] = None, **kwargs):
        """Core execution logic (must be implemented by subclasses, need to call _set_allowed_tools and _clear_allowed_tools)"""
        # Parent class uniformly handles permission initialization (subclasses don't need to repeat)
        self._set_allowed_tools(allowed_tools)
        try:
            self.run_child(phase,** kwargs)
        finally:
            # Whether the subclass execution succeeds or fails, clear permissions (avoid affecting next run)
            self._clear_allowed_tools()

    def run_child(self, phase, **kwargs):
        raise NotImplementedError("Subclasses must implement the run_child method")