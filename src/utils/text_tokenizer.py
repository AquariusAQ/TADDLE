import tiktoken
from transformers import AutoTokenizer
from typing import Optional

def count_tokens(
    text: str, 
    model: str = "gpt-3.5-turbo",
    model_path: Optional[str] = None
) -> int:
    """
    Calculate the number of Tokens for the given text under the specified model.
    Prioritize using Hugging Face's AutoTokenizer for better support of open-source models such as qwen3.

    Args:
        text (str): The text for which to calculate the Token count.
        model (str): Name or path of the target LLM model.
        model_path (Optional[str]): Path to local model files. If provided, loading from local path will take precedence.

    Returns:
        int: The number of Tokens corresponding to the text.
    """
    # Prioritize using Hugging Face's AutoTokenizer to support open-source models like qwen3
    try:
        # Load from local path if provided
        if model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        
        # Encode text using tokenizer and get Token count
        # return_offsets_mapping=True helps with more precise truncation handling, but only count is needed here
        tokens = tokenizer(text, return_tensors="pt", truncation=False)
        return tokens.input_ids.shape[1]

    except Exception as e:
        # Fall back to tiktoken if Hugging Face loading fails (e.g., model not on Hub and local path incorrect)
        # This still works for OpenAI models
        print(f"Warning: Failed to load model '{model}' using Hugging Face AutoTokenizer. Error message: {e}")
        print("Will attempt to fall back to 'tiktoken' for calculation - please ensure the model name is correct.")
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print(f"Warning: Unknown model '{model}', tiktoken will use the default 'cl100k_base' encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        
        return len(encoding.encode(text))

def truncate_text(
    text: str,
    max_length: int,
    model: str = "gpt-3.5-turbo",
    model_path: Optional[str] = "/home/LLM/Qwen/Qwen3-30B-A3B-Thinking-2507",
    truncate_from_start: bool = True
) -> str:
    """
    Truncate text to the specified maximum Token length.

    Args:
        text (str): Original text to be truncated.
        max_length (int): Maximum allowed Token length for the input text.
        model (str): Name or path of the target LLM model.
        model_path (Optional[str]): Path to local model files. If provided, loading from local path will take precedence.
        truncate_from_start (bool): If True, retain from the start of the text (truncate the end);
                                    If False, retain from the end of the text (truncate the start).

    Returns:
        str: Truncated text.
    """
    if not text:
        return ""

    # Similarly, prioritize using Hugging Face Tokenizer for precise truncation
    try:
        if model_path:
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

        # Use tokenizer's built-in functionality for encoding and truncation
        # truncation=True automatically truncates to max_length
        # return_tensors="pt" returns PyTorch tensors, which we convert to list with .tolist()
        encoded_input = tokenizer(
            text,
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        # Decode truncated Token IDs back to text
        # skip_special_tokens=True removes special tokens like <s>, </s>, <pad>
        truncated_text = tokenizer.decode(encoded_input.input_ids[0], skip_special_tokens=True)
        return truncated_text

    except Exception as e:
        # Fallback logic: use tiktoken if Hugging Face Tokenizer fails
        print(f"Warning: Failed to load model '{model}' using Hugging Face AutoTokenizer. Error message: {e}")
        print("Will attempt to fall back to 'tiktoken' for truncation.")
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print(f"Warning: Unknown model '{model}', tiktoken will use the default 'cl100k_base' encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(text)
        
        if len(tokens) > max_length:
            if truncate_from_start:
                truncated_tokens = tokens[:max_length]
            else:
                truncated_tokens = tokens[-max_length:]
            return encoding.decode(truncated_tokens)
        
        return text

# --- Example Usage ---
# if __name__ == "__main__":
#     long_text = """Artificial Intelligence (AI) is a branch of computer science that aims to create machines capable of simulating, extending, and expanding human intelligence. Research areas of AI include machine learning, deep learning, natural language processing, computer vision, knowledge representation and reasoning, robotics, and many others. In recent years, with the rapid development of technologies such as big data, cloud computing, and graphics processing units (GPUs), AI technology has made breakthrough progress, been widely applied in many fields such as image recognition, voice assistants, autonomous driving, and medical diagnosis, and has had a profound impact on society."""
    
#     # --- Example 1: Using qwen3 model (Recommended Method) ---
#     print("--- Example 1: Using qwen3 model ---")
#     # Assume your qwen3 model files are stored in the local path './models/qwen3-7b'
#     # If the model is downloaded locally, absolute path is recommended
#     local_qwen3_path = "/home/LLM/Qwen/Qwen3-30B-A3B-Thinking-2507" 
    
#     try:
#         # 1. Calculate Token count
#         qwen_token_count = count_tokens(long_text, model="qwen/Qwen3-7B-Instruct", model_path=local_qwen3_path)
#         print(f"Original text Token count (qwen3): {qwen_token_count}")

#         # 2. Truncate text to 50 Tokens
#         truncated_by_qwen = truncate_text(
#             text=long_text,
#             max_length=50,  # Directly specify the maximum input Token length
#             model="qwen/Qwen3-7B-Instruct",
#             model_path=local_qwen3_path,
#             truncate_from_start=True
#         )
        
#         print(f"\nTruncated text (qwen3):\n{truncated_by_qwen}")
#         print(f"Truncated text Token count (qwen3): {count_tokens(truncated_by_qwen, model='qwen/Qwen3-7B-Instruct', model_path=local_qwen3_path)}")

#     except Exception as e:
#         print(f"\nError occurred when using qwen3 model: {e}")
#         print("Please ensure you have installed transformers and torch libraries, and the model path is correct.")
#         print("Installation command: pip install transformers torch")

    # --- Example 2: Fall back to using gpt-3.5-turbo model ---
    # print("\n--- Example 2: Fall back to using gpt-3.5-turbo model ---")
    # gpt_token_count = count_tokens(long_text, model="gpt-3.5-turbo")
    # print(f"Original text Token count (gpt-3.5-turbo): {gpt_token_count}")

    # truncated_by_gpt = truncate_text(
    #     text=long_text,
    #     max_length=50,  # Similarly, directly specify the maximum input Token length
    #     model="gpt-3.5-turbo",
    #     truncate_from_start=True
    # )
    
    # print(f"\nTruncated text (gpt-3.5-turbo):\n{truncated_by_gpt}")
    # print(f"Truncated text Token count (gpt-3.5-turbo): {count_tokens(truncated_by_gpt, model='gpt-3.5-turbo')}")