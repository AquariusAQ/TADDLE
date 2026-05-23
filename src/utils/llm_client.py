from openai import OpenAI
from utils.logger import logger
from utils.config import config
from typing import Optional, List, Dict, Any
import json
from datetime import datetime
import base64  # For image Base64 encoding
import os
import time  # 新增：用于重试间隔
from openai import APIConnectionError  # 新增：捕获连接错误

vllm_logger = None

# Image to Base64 conversion function
def image_to_base64(image_path: str) -> str:
    """Convert local image file to Base64 encoded string (required format for OpenAI API)"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to convert image to Base64: {image_path}, Error: {str(e)}")
        raise ValueError(f"Unable to read image file: {image_path}") from e

def call_llm(
    prompt: str = None, 
    system: Optional[str] = None, 
    messages: Optional[List[Dict]] = None,
    model: Optional[str] = None, 
    temperature: Optional[float] = None,
    presence_penalty: Optional[float] = 0.0,
    frequency_penalty: Optional[float] = 0.0,
    tools: Optional[List[Dict]] = None,  # New: Tool list parameter
    image_paths: Optional[List[str]] = None,  # List of local image paths
    image_query: Optional[str] = None,  # Query instruction for images, will be overwritten if prompt is provided
    verbose: Optional[Dict] = {},
    call_id: Optional[str] = None
) -> Dict[str, Any]:
    """Call LLM model (OpenAI-compatible interface)"""

    if model:
        llm_type = model
    elif config["tools"]["image_analyzer"]["enabled"] and image_paths:
        llm_type = "vlm"
    else:
        llm_type = "llm"

    if not hasattr(call_llm, 'api_keys'):
        if os.path.exists("key.json"):
            try:
                # Read key.json file in current directory (modify path if file is elsewhere)
                with open("key.json", "r", encoding="utf-8") as f:
                    api_keys = json.load(f)
                if not api_keys:
                    logger.debug("key.json file is empty")
                    call_llm.api_keys = {}
                else:
                    logger.debug("Reading all API_KEYs from key.json file")
                    call_llm.api_keys = api_keys # Bind static variable to function
            except Exception as e:
                logger.error(f"Failed to read key.json file: {str(e)}")
                call_llm.api_keys = {}
        else:
            logger.debug("key.json file not provided, will obtain API_KEY from model.yaml")
            call_llm.api_keys = {}
    
    if llm_type not in config or not isinstance(config[llm_type], dict):
        raise ValueError(f"LLM type '{llm_type}' not found in configuration")
    api_key_description = config[llm_type].get("api_key", "EMPTY")  # Leave empty by default for local deployment
    if not api_key_description or api_key_description == "EMPTY":
        api_key = "EMPTY"
    elif api_key_description.startswith("LOCAL_KEY:"):
        api_key = call_llm.api_keys.get(api_key_description.split(":")[1].strip(), api_key_description or "EMPTY")
    else:
        api_key = api_key_description

    if not hasattr(call_llm, 'base_urls'):
        if os.path.exists("base_url.json"):
            try:
                # Read key.json file in current directory (modify path if file is elsewhere)
                with open("base_url.json", "r", encoding="utf-8") as f:
                    base_urls = json.load(f)
                if not base_urls:
                    logger.debug("base_url.json file is empty")
                    call_llm.base_urls = {}
                else:
                    logger.debug("Reading all BASE_URLs from base_url.json file")
                    call_llm.base_urls = base_urls # Bind static variable to function
            except Exception as e:
                logger.error(f"Failed to read base_url.json file: {str(e)}")
                call_llm.base_urls = {}
        else:
            logger.debug("base_url.json file not provided, will obtain BASE_URL from model.yaml")
            call_llm.base_urls = {}

    base_url_description = config[llm_type].get("base_url", None)  # Leave empty by default for local deployment
    if not base_url_description :
        raise ValueError("MUST PROVIDE LLM API BASE_URL !!!")
    elif base_url_description.startswith("LOCAL_URL:"):
        base_url = call_llm.base_urls.get(base_url_description.split(":")[1].strip(), base_url_description)
    else:
        base_url = base_url_description
    if not base_url.startswith("http"):
        logger.warning(f"Please check base_url: {base_url}")
    
    global vllm_logger
    if vllm_logger == None:
        vllm_logger = logger.vllm 

    max_retries = 5  
    retry_delay = 5  
    response = None
    try:
        # Configuration check (keep original logic to ensure configuration is initialized)
        if config is None:
            raise ValueError("Configuration not initialized, please load YAML configuration in main.py first")
        
        # Read LLM parameters (get from global configuration, compatible with custom input)
        model_name = config[llm_type]["model"]
        temperature = temperature or config[llm_type].get("temperature", None)
        max_completion_tokens = config[llm_type]["max_completion_tokens"]
        # base_url = config[llm_type]["base_url"]  # VLLM service address
        top_p = config[llm_type].get("top_p", 0.95)
        extra_body = config[llm_type].get("extra_body", {})
        
        # Initialize OpenAI-compatible client (keep original logic)
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=300)

        # 1. Messages provided directly
        if messages: 
            if (prompt != None or system != None or image_query != None) and messages != None:
                logger.warning("Overwriting prompt/system_prompt/image_query with messages")
            system = messages[0]["content"]
            prompt = messages[-1]["content"]
            prompt = next((i["text"] for i in prompt if isinstance(i, dict) and i.get("type") == "text" and "text" in i), "") if isinstance(prompt, list) else prompt
        # 2. No messages provided, but prompt or image_query provided
        elif prompt or image_query:
            messages = []
            prompt = prompt or image_query
            # system prompt
            if system:
                messages.append({"role": "system", "content": system})
            else:
                system = ""
            # user prompt + image input
            content = []
            if prompt:
                content.append({"type": "text", "text": prompt})
            if image_paths and isinstance(image_paths, list):
                for img_path in image_paths:
                    base64_img = image_to_base64(img_path)
                    content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_img}",  # Compatible with OpenAI format
                            "detail": "high"  # Image detail mode: high/low/auto (qwen3vl does not support auto)
                        }
                    })
            messages.append({"role": "user", "content": content})
        # Neither provided
        else:
            logger.error("Prompt (Image Query) and messages cannot both be None!")
            raise ValueError("Prompt (Image Query) and messages cannot both be None!")
        
        # messages[0]["content"] += f"\n\n[Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
        date_str = f"[Current Date: {datetime.now().strftime('%Y-%m-%d')}]"
        if date_str not in messages[0]["content"]:
            messages[0]["content"] += "\n\n" + date_str

        # Construct LLM request (add tools field)
        llm_request = {
            "model": model_name,
            "stream": False,
            "messages": messages,
            "user": call_id,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": extra_body,
            "max_completion_tokens": max_completion_tokens,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty
        }
        if tools:  # Pass tools configuration only when there are LLM-driven tools
            # logger.debug(f"This is a message that can call tools ({tools})")
            llm_request["tools"] = tools
            llm_request["tool_choice"] = "auto"  # Let the model automatically decide whether to call tools
        
        # Key Modification 1: VLLM input logging (write to combined file, distinguished by markers)
        vllm_logger.info(f"=== START OF VLLM INPUT [Model: {model}({model_name})] ===")
        # vllm_logger.info(f"Tools Count: {len(tools) if tools else 0}")
        vllm_logger.info(f"Allowed Tools:")
        if tools:
            vllm_logger.info([T["function"]["name"] for T in tools])
        else:
            vllm_logger.info("None")
        if verbose.get("system", True):
            system_prompt_log = f"{system.split('Paper Content')[0] if system else 'None'} {'<Paper Content>' if 'Paper Content' in system else ''}"
            if "Your goal: " in system_prompt_log:
                system_prompt_log  = f"Please read this paper main text: <Paper Content> {system_prompt_log.split("Your goal: ")[1]}"
            vllm_logger.info(f"System Prompt: {system_prompt_log}")
        # vllm_logger.info(f"Text Content: {text_content[:500]}{'...' if len(text_content) > 500 else ''}")
        vllm_logger.info(f"Image Count: {len(image_paths) if image_paths else 0}")
        # Truncate long prompts (avoid log redundancy, keep first 500 characters)
        prompt_log = prompt[:500].replace("\n", " ").strip() + ("..." if len(prompt) > 500 else "")
        vllm_logger.info(f"User Prompt: {prompt_log}")
        vllm_logger.info(f"=== END OF VLLM INPUT ===")
        
        # Call LLM and parse results
        # response = client.chat.completions.create(**llm_request)

        for retry in range(max_retries):
            try:
                response = client.chat.completions.create(**llm_request)
                break
            except APIConnectionError as e:
                error_msg = f"VLLM connection error (retry {retry+1}/{max_retries}): {str(e)}"
                vllm_logger.error(error_msg)
                logger.error(error_msg)

                if retry == max_retries - 1:
                    raise
                time.sleep(retry_delay)
            except Exception as e:
                error_msg = f"VLLM invocation failed (non-retryable error): {str(e)}"
                vllm_logger.error(error_msg)
                logger.error(error_msg)
                raise
            
        if response is None:
            raise RuntimeError("LLM call failed after all retries")

        raw_msg = response.choices[0].message
        if usage := response.usage:
            vllm_logger.info("=== Token usage information ===")
            vllm_logger.info(usage)
        # messages.append(raw_msg)

        # Parse tool_calls (if any)
        tool_calls = None
        if hasattr(raw_msg, "tool_calls") and raw_msg.tool_calls:
            tool_calls = []
            for call in raw_msg.tool_calls:
                if call.type == "function":
                    tool_calls.append({
                        "name": call.function.name,
                        "arguments": json.loads(call.function.arguments),  # Parse parameter JSON string
                        "id": call.id
                    })
            vllm_logger.info(f"Detected Tool Calls: {tool_calls}")

        # Parse natural language results
        raw_answer = raw_msg.content.strip() if raw_msg.content else ""
        reasoning_content = raw_msg.reasoning if hasattr(raw_msg, 'reasoning') else None
        reasoning_content = raw_msg.reasoning_content if hasattr(raw_msg, 'reasoning_content') else reasoning_content
        
        # # Process LLM output (keep original logic, split thinking part from pure answer)
        # raw_answer = response.choices[0].message.content.strip()
        
        # Key Modification 2: VLLM output logging (write to combined file, corresponding to input)
        vllm_logger.info(f"=== START OF VLLM OUTPUT [Model: {model}({model_name})] ===")
        # Format output logs (preserve original indentation, avoid overly long lines)
        vllm_logger.info(f"Tool Calls: {tool_calls if tool_calls else 'None'}")
        if reasoning_content:
            reasoning_content_log = reasoning_content.replace("\n\n", "\\n").replace("\n", " ").replace("\\n", "\n")
            for line in reasoning_content_log.split("\n"):
                vllm_logger.info(f"    {line}")  # Indent for distinction and readability
            vllm_logger.info(f"= END OF THINKING =")
        raw_answer_log = raw_answer.replace("\n\n", "\\n").replace("\n", " ").replace("\\n", "\n")
        for line in raw_answer_log.split("\n"):
            vllm_logger.info(f"    {line}")  # Indent for distinction and readability
        vllm_logger.info(f"=== END OF VLLM OUTPUT ===")
        
        if "</think>" in raw_answer:
            pure_answer = raw_answer.split("</think>")[-1].strip()  # Keep only the final answer
        else:
            pure_answer = raw_answer  # Compatible with outputs without thinking part
        
        # Return complete results (including tool_calls)
        return {
            "answer": pure_answer,
            "tool_calls": tool_calls,
            "raw_msg": raw_msg,
            "messages": messages
        }
    
    except Exception as e:
        # Error logs still written to normal logs (facilitate overall process troubleshooting)
        # logger.error(f"VLLM invocation failed: {str(e)}", exc_info=True)
        logger.error(f"VLLM invocation failed after {max_retries} retries: {str(e)}", exc_info=True)
        raise  # Raise exception for upper layer handling (e.g., terminate review process)

def call_llm_with_tool(system=None, prompt=None, call_tool_agent=None, llm_tools=[], llm_tools_name=[], model=None, model_vlm=None, messages=None,
    call_id: Optional[str] = None):
    # logger.info("=============== Debug ===============")
    # logger.info(f"system {system[:50]}")
    # logger.info(f"prompt {prompt[:50]}")
    # logger.info(call_tool_agent)
    # logger.info(llm_tools)
    # logger.info(llm_tools_name)
    # logger.info("=============== End Debug ===============")
    
    global vllm_logger
    if vllm_logger == None:
        vllm_logger = logger.vllm 

    # 2. Interactive loop with LLM: until LLM stops invoking tools
    if messages:
        llm_messages = messages
    else:
        llm_messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]
    final_result = ""
    while True:
        if len(llm_messages)>2:
            vllm_logger.info("===> Next Turn")
        # Call LLM (pass tool configuration)
        llm_result = call_llm(
            messages=llm_messages,
            tools=llm_tools,
            model=model,
            verbose=({"system": True} if len(llm_messages)==2 else {"system": False}),
            call_id=call_id
        )

        # 3. Process LLM results: execute tools if there are tool calls; end if none
        if llm_result["tool_calls"]:
            llm_messages.append(llm_result["raw_msg"])
            for tool_call in llm_result["tool_calls"]:
                if tool_call["name"] in llm_tools_name:
                    # Execute tool invocation specified by LLM
                    tool_result = call_tool_agent.call_tool(
                        tool_name=tool_call["name"],
                        model_vlm=model_vlm,
                        **tool_call["arguments"]
                    )
                    # Add tool results to next round of LLM messages
                    # tool_call_id = tool_call.get("id", f"tool_call_{idx}")
                    llm_messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.get("id", ""), 
                        "name": tool_call["name"], 
                        "content": f"Tool {tool_call['name']} returned results:\n{tool_result}"
                    })
        else:
            # LLM stops invoking tools, get final rebuttal
            final_result = llm_result["answer"]
            break

    return final_result