import os
import sys
import time
import threading
import datetime
from pathlib import Path
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
import argparse

# ===================== Global Configuration =====================
# Thread lock (ensures thread safety for logging/progress bar)
print_lock = threading.Lock()
# Backend configuration (keep consistent with your environment)
# BASE_URL = "http://172.23.166.144:32027/v1"
BASE_URL = ""
MODEL_NAME = "Qwen3-30B-A3B-Thinking-2507"
API_KEY = "EMPTY"
TIMEOUT = 120
RETRY_COUNT = 1  # Slightly increase retries to improve stability
# Processing configuration
TOKEN_LIMIT = 500  # Token limit for appendix summary
THREAD_COUNT = 3   # Number of parallel threads
SUPPORTED_EXT = ".md"  # Only process markdown files
OUTPUTS_ROOT = Path("./outputs")  # Root directory for datasets
# Excluded filename suffixes (to avoid collecting generated files)
EXCLUDE_SUFFIXES = [
    "_main_text.md",
    "_reference.md",  # New: exclude reference files
    "_appendix.md",
    "_appendix_summary.md"
]
# Appendix summary prompt (500 token limit)
APPENDIX_SUMMARY_PROMPT = (
    "For the complete appendix of an AI field paper, summarize it by splitting it into the original subsections: "
    "each section focuses on core methods, experimental details, data/parameters, or supplementary arguments, "
    "eliminates redundant expressions, keeps it within 500 tokens for each section, uses concise and accurate language, "
    "and retains key academic information. "
    "Output format: ### Appendix A Title + Summary content. ### Appendix B Title + Summary content......"
)
# ================================================================

def log_output(message: str):
    """Thread-safe log output"""
    with print_lock:
        print(message)

def create_openai_client() -> OpenAI:
    """Create OpenAI client"""
    return OpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        timeout=TIMEOUT,
        max_retries=RETRY_COUNT
    )

def check_if_processed(original_md_file: Path) -> bool:
    """
    Check if the paper has already been processed (core logic for resuming interrupted runs)
    Conditions: main text, reference, appendix, and appendix summary files all exist and are non-empty
    """
    paper_id = original_md_file.stem
    # New: check reference file
    main_text_file = original_md_file.with_name(f"{paper_id}_main_text.md")
    reference_file = original_md_file.with_name(f"{paper_id}_reference.md")
    appendix_file = original_md_file.with_name(f"{paper_id}_appendix.md")
    summary_file = original_md_file.with_name(f"{paper_id}_appendix_summary.md")
    
    # Check four files (reference_file allowed to be empty when there is no reference)
    required_files = [main_text_file, appendix_file, summary_file]
    for f in required_files:
        if not f.exists() or f.stat().st_size == 0:
            return False
    # Reference file: just needs to exist (empty is allowed, corresponding to the scenario with no references)
    if not reference_file.exists():
        return False
    
    return True

def is_original_paper_file(file_path: Path) -> bool:
    """
    Determine whether it is an original paper markdown file (exclude generated files)
    """
    if not file_path.name.endswith(SUPPORTED_EXT):
        return False
    # Exclude generated files
    for suffix in EXCLUDE_SUFFIXES:
        if file_path.name.endswith(suffix):
            return False
    return True

def split_main_reference_appendix(full_content: str, paper_id: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Split into main text, references, and appendix (core modification: three-file separation)
    :param paper_id: Original paper ID
    :return: (main text content, reference content, appendix content), returns (None, None, None) on failure
    """
    content_lower = full_content.lower()
    references_marker = "## references"
    references_pos = content_lower.find(references_marker)
    
    if references_pos == -1:
        # No REFERENCES marker found: main text = full text, reference = None, appendix = None
        log_output(f"⚠️  Paper ID[{paper_id}]: No REFERENCES marker found, using full text as main text (no reference and no appendix)")
        return full_content.strip(), "", None
    
    # ========== Core splitting logic ==========
    # 1. Main text: content before the REFERENCES marker (excluding REFERENCES)
    main_text = full_content[:references_pos].strip()
    
    # 2. Extract content from REFERENCES onwards
    content_after_references = full_content[references_pos:].strip()
    # Find the first ## after REFERENCES (start marker for appendix)
    appendix_marker_pos = content_after_references.find("##", len(references_marker)+1)  # skip the ## of REFERENCES itself
    
    if appendix_marker_pos == -1:
        # No appendix: reference = REFERENCES and everything after it, appendix = None
        reference_text = content_after_references.strip()
        appendix_content = None
        log_output(f"ℹ️  Paper ID[{paper_id}]: No appendix after REFERENCES, reference saved separately, no appendix")
    else:
        # Has appendix:
        # reference = content from REFERENCES marker to before the appendix marker
        # appendix = content from the appendix marker onwards
        reference_text = content_after_references[:appendix_marker_pos].strip()
        appendix_content = content_after_references[appendix_marker_pos:].strip()
        log_output(f"✅  Paper ID[{paper_id}]: Successfully split main text, references, and appendix")
    
    return main_text, reference_text, appendix_content

def save_content_to_file(content: str, file_path: Path) -> bool:
    """Save content to file, return whether successful"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        log_output(f"❌ Failed to save file {file_path}: {str(e)}")
        return False

def summarize_appendix(appendix_content: str, paper_id: str) -> Optional[str]:
    """Generate appendix summary limited to 500 tokens"""
    client = create_openai_client()
    start_time = time.time()
    
    messages = [
        {"role": "system", "content": "You are a professional academic assistant, proficient in summarizing AI papers."},
        {"role": "user", "content": f"{APPENDIX_SUMMARY_PROMPT}\n\n### Appendix Content:\n{appendix_content}"}
    ]
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.3,
            top_p=0.9,
            extra_body={"top_k": 20, "min_p": 0.0},
            max_completion_tokens=8192
        )
        
        assert response.choices and response.choices[0].message and response.choices[0].message.content
        summary = response.choices[0].message.content.strip()
        elapsed = time.time() - start_time
        log_output(f"✅ Paper ID[{paper_id}]: Appendix summary completed, took {elapsed:.2f} seconds")
        return summary
    except Exception as e:
        log_output(f"❌ Paper ID[{paper_id}]: Appendix summary failed: {str(e)[:200]}")
        return None

def process_single_paper(original_md_file: Path) -> Tuple[str, str]:
    """
    Process a single original paper (added reference file saving)
    :param original_md_file: Path to the original paper markdown file
    :return: (paper ID, processing status)
    """
    paper_id = original_md_file.stem  # Only take the original paper ID (without suffix)
    log_output(f"\n📄 Start processing paper ID[{paper_id}]")
    
    # 1. Check if already processed (resume interrupted run)
    if check_if_processed(original_md_file):
        return paper_id, "Skipped (already processed)"
    
    # 2. Read the original Markdown file
    if not original_md_file.exists():
        return paper_id, "Failed (original file does not exist)"
    
    try:
        with open(original_md_file, "r", encoding="utf-8") as f:
            full_content = f.read().strip()
    except Exception as e:
        return paper_id, f"Failed (error reading original file: {str(e)})"
    
    if not full_content:
        return paper_id, "Failed (original file content is empty)"
    
    # 3. Split main text, references, and appendix
    main_text, reference_text, appendix_content = split_main_reference_appendix(full_content, paper_id)
    if not main_text:
        return paper_id, "Failed (no main text content)"
    
    # 4. Save main text
    main_text_file = original_md_file.with_name(f"{paper_id}_main_text.md")
    if not save_content_to_file(main_text, main_text_file):
        return paper_id, "Failed (failed to save main text)"
    
    # 5. Save references (new core logic)
    reference_file = original_md_file.with_name(f"{paper_id}_reference.md")
    if not save_content_to_file(reference_text, reference_file):
        return paper_id, "Failed (failed to save reference)"
    log_output(f"✅ Paper ID[{paper_id}]: Reference saved to {reference_file.name}")
    
    # 6. Process appendix
    appendix_file = original_md_file.with_name(f"{paper_id}_appendix.md")
    summary_file = original_md_file.with_name(f"{paper_id}_appendix_summary.md")
    
    if appendix_content:
        # Save appendix
        if not save_content_to_file(appendix_content, appendix_file):
            return paper_id, "Failed (failed to save appendix)"
        # Generate and save appendix summary
        summary = summarize_appendix(appendix_content, paper_id)
        if not summary:
            return paper_id, "Failed (appendix summary generation failed)"
        if not save_content_to_file(summary, summary_file):
            return paper_id, "Failed (failed to save appendix summary)"
    else:
        # No appendix: save empty files as markers
        save_content_to_file("", appendix_file)
        save_content_to_file("No appendix content", summary_file)
        log_output(f"ℹ️  Paper ID[{paper_id}]: No appendix content, saved empty file as marker")
    
    return paper_id, "Success (split + summary completed)"

def collect_original_paper_files(dataset_name: str) -> list[Path]:
    """
    Collect original paper markdown files under the specified dataset (excluding generated files)
    """
    dataset_dir = OUTPUTS_ROOT / dataset_name
    if not dataset_dir.exists():
        log_output(f"❌ Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)
    
    # Traverse all paper directories under the dataset, find original markdown files
    original_files = []
    for paper_dir in dataset_dir.iterdir():
        if not paper_dir.is_dir():
            continue
        # Traverse all markdown files in the directory, keep only original files
        for file in paper_dir.glob(f"*{SUPPORTED_EXT}"):
            if is_original_paper_file(file):
                original_files.append(file)
                break  # Assume each directory has only one original markdown file
    
    if not original_files:
        log_output(f"⚠️  No original paper markdown files found under dataset {dataset_name}")
    return original_files

def process_dataset(dataset_name: str) -> Tuple[int, int, int, int, list]:
    """
    Process a single dataset, return statistics
    :return: (success count, skip count, fail count, no-reference count, list of no-reference paper IDs)
    """
    log_output(f"\n===== Start processing dataset: {dataset_name} =====")
    log_output(f"Current time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_output(f"Number of parallel threads: {THREAD_COUNT}, Appendix summary token limit: {TOKEN_LIMIT}")

    original_files = collect_original_paper_files(dataset_name)
    if not original_files:
        log_output(f"⚠️  Dataset {dataset_name} has no files to process, skipping")
        return 0, 0, 0, 0, []

    log_output(f"✅ Found {len(original_files)} original papers to process")

    results = []
    with ThreadPoolExecutor(max_workers=THREAD_COUNT) as executor:
        futures = {executor.submit(process_single_paper, f): f for f in original_files}
        with tqdm(total=len(futures), desc="Overall processing progress", unit="papers") as pbar:
            for future in as_completed(futures):
                paper_id, status = future.result()
                results.append((paper_id, status))
                pbar.update(1)
                pbar.set_postfix({"Current Paper ID": paper_id, "Status": status})

    # Statistics for current dataset
    success = skip = fail = no_ref = 0
    no_ref_ids = []
    for paper_id, status in results:
        if "Success" in status:
            success += 1
            if "No REFERENCES marker found" in status:
                no_ref += 1
                no_ref_ids.append(paper_id)
        elif "Skipped" in status:
            skip += 1
        else:
            fail += 1

    log_output(f"\n📊 Processing summary for dataset {dataset_name}: Success {success}, Skipped {skip}, Failed {fail}, No references {no_ref}")
    return success, skip, fail, no_ref, no_ref_ids

def main():
    global BASE_URL  # Declare use of global variable
    parser = argparse.ArgumentParser(description="Tool for separating papers into three files (main text/references/appendix) + appendix summarization")
    parser.add_argument("--base_url",
                        default="http://127.0.0.1:8000/v1",
                        help="API base URL (default: %(default)s)")
    parser.add_argument("--datasets", nargs="+", required=True,
                        help="Dataset names (multiple allowed)")
    args = parser.parse_args()

    BASE_URL = args.base_url  # Update global base_url

    total_success = total_skip = total_fail = total_no_ref = 0
    all_no_ref_ids = []

    for ds in args.datasets:
        s, sk, f, nr, nids = process_dataset(ds)
        total_success += s
        total_skip += sk
        total_fail += f
        total_no_ref += nr
        all_no_ref_ids.extend(nids)

    # Overall summary
    log_output("\n" + "="*80)
    log_output("📊 All datasets processed, overall summary:")
    log_output(f"✅ Success: {total_success} papers")
    log_output(f"⏭️  Skipped: {total_skip} papers")
    log_output(f"❌ Failed: {total_fail} papers")
    log_output(f"ℹ️  No REFERENCES found: {total_no_ref} papers")
    if all_no_ref_ids:
        log_output(f"Paper IDs without REFERENCES: {', '.join(all_no_ref_ids)}")
    log_output("="*80)

if __name__ == "__main__":
    main()