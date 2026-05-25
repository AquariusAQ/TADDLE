import os
import sys
import json
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
# from openai.exceptions import APIConnectionError, APIError, AuthenticationError

# ===================== Configuration (modify as needed) =====================
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8001
MODEL_NAME = "Qwen3-VL-4B-Instruct"
OUTPUTS_DIR = Path("./outputs")  # Root directory
ANALYSIS_JSON = "image_analysis.json"  # Output filename for results
SUPPORTED_IMG_TYPES = (".jpg", ".jpeg", ".png", ".bmp", ".gif")  # Supported image formats
# ======================================================================


def image_to_base64(image_path: Path) -> str:
    """Convert image to Base64 encoding (with error handling)"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        return f"ERROR: File not found - {image_path.name}"
    except Exception as e:
        return f"ERROR: Encoding failed - {str(e)}"


def analyze_image(image_path: Path, client: OpenAI) -> tuple[str, str]:
    """Backend analysis request for a single image (unchanged)"""
    img_name = image_path.name
    img_b64 = image_to_base64(image_path)
    
    # Return error directly if image encoding fails
    if img_b64.startswith("ERROR"):
        return (img_name, img_b64)

    # Build multimodal request
    prompt = """
Please analyze this academic paper image concisely (≤200 tokens) with only key information:
1. Clarify image type (data chart/logical flow/case diagram/etc.) and core purpose;
2. Extract key text (title, axes/labels/legends/annotations);
3. For data charts: Highlight axes, key data trends & group comparisons;
4. For other types: Focus on core elements, logical relationships & key details;
Avoid redundancy, use precise numbers/terms.
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            temperature=0.1,
            max_tokens=5000
        )
        return (img_name, response.choices[0].message.content.strip())
    # except APIConnectionError:
    #     return (img_name, f"ERROR: Unable to connect to backend {VLLM_HOST}:{VLLM_PORT}")
    # except AuthenticationError:
    #     return (img_name, "ERROR: Authentication failed (backend does not require API Key)")
    # except APIError as e:
    #     return (img_name, f"ERROR: Backend error - {str(e)}")
    except Exception as e:
        return (img_name, f"ERROR: Unknown error - {str(e)}")


def process_single_paper(paper_dir: Path) -> tuple[str, str]:
    """Process all images of a single paper (serial processing internally, removed image-level thread pool)"""
    # Check if already processed
    result_file = paper_dir / ANALYSIS_JSON
    if result_file.exists() and result_file.stat().st_size > 0:
        return (paper_dir.name, "Skipped (analysis already completed)")

    # Check if imgs folder exists
    imgs_dir = paper_dir / "imgs"
    if not imgs_dir.exists() or not imgs_dir.is_dir():
        return (paper_dir.name, "Skipped (no imgs folder)")

    # Filter supported image files
    image_files = [f for f in imgs_dir.iterdir() if f.suffix.lower() in SUPPORTED_IMG_TYPES]
    if not image_files:
        return (paper_dir.name, "Skipped (no valid images)")

    # Initialize backend client
    try:
        client = OpenAI(
            base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
            api_key="dummy-key",
            timeout=60.0  # Increased timeout for proxy forwarding
        )
    except Exception as e:
        return (paper_dir.name, f"Failed (client initialization error: {str(e)})")

    # Serial processing of images within a single paper (core modification 1)
    analysis_result = {}
    # Progress bar for images within a single paper (console display only)
    for img_file in tqdm(image_files, desc=f"Processing paper {paper_dir.name}", leave=False):
        img_name, res = analyze_image(img_file, client)
        analysis_result[img_name] = res

    # Save formatted JSON results
    try:
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2)
        return (paper_dir.name, f"Success (processed {len(image_files)} images)")
    except Exception as e:
        return (paper_dir.name, f"Failed (save result error: {str(e)})")


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Batch analysis tool for academic paper images (paper-level parallelism)")
    parser.add_argument("--datasets", nargs="+", help="Datasets to process (e.g., iclr)")
    parser.add_argument("--threads", type=int, default=3, help="Number of papers to process in parallel (default: 3)")
    parser.add_argument("--url", type=str, default=f"http://{VLLM_HOST}:{VLLM_PORT}", help="OCR service URL")
    args = parser.parse_args()

    # Parameter validation
    if args.threads <= 0:
        print("Error: Number of threads must be greater than 0")
        sys.exit(1)
    if not OUTPUTS_DIR.exists():
        print(f"Error: Root directory {OUTPUTS_DIR} does not exist")
        sys.exit(1)

    # Collect all paper directories to process
    paper_dirs = []
    for dataset in args.datasets:
        dataset_dir = OUTPUTS_DIR / dataset
        if not dataset_dir.exists():
            print(f"Warning: Dataset {dataset} does not exist, skipped")
            continue
        # Iterate over all paper folders under the dataset
        paper_dirs.extend([d for d in dataset_dir.iterdir() if d.is_dir()])

    if not paper_dirs:
        print("No papers to process")
        sys.exit(0)

    # Paper-level multi-threaded parallel processing (core modification 2)
    print(f"Starting processing of {len(paper_dirs)} papers (parallel papers: {args.threads})")
    process_results = []
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        # Submit all paper processing tasks
        futures = {executor.submit(process_single_paper, paper_dir): paper_dir for paper_dir in paper_dirs}
        # Global progress bar (tracks progress of all papers)
        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall progress"):
            paper_name, status = future.result()
            process_results.append((paper_name, status))

    # Output processing summary
    print("\nProcessing results summary:")
    for paper_name, status in process_results:
        print(f"- {paper_name}: {status}")


if __name__ == "__main__":
    main()