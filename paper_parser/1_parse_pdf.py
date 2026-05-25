import argparse
import time
from pathlib import Path
from paddleocr import PaddleOCRVL
from tqdm import tqdm

def process_single_pdf(pipeline, pdf_file, output_root):
    """
    Core logic for processing a single PDF file (extracted as a separate function for easy retry).
    :param pipeline: Initialized PaddleOCRVL instance
    :param pdf_file: PDF file path (Path object)
    :param output_root: Output root directory (Path object)
    """
    pdf_stem = pdf_file.stem  # Get filename without extension
    current_output = output_root / pdf_stem  # Output directory for the current PDF
    mkd_file_path = current_output / f"{pdf_stem}.md"  # Build markdown file path
    
    # Check if the markdown file already exists; if so, skip it
    if mkd_file_path.exists() and mkd_file_path.stat().st_size > 0:
        print(f"\n⏭️  Skipping already processed file: {pdf_file} (corresponding markdown file exists: {mkd_file_path})")
        return

    print(f"\nProcessing: {pdf_file} -> output to {current_output}")

    # Run OCR parsing
    output = pipeline.predict(input=str(pdf_file))

    # Extract markdown content
    markdown_list = []
    markdown_images = []
    for res in output:
        md_info = res.markdown
        markdown_list.append(md_info)
        markdown_images.append(md_info.get("markdown_images", {}))

    # Merge multi-page markdown
    markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)

    # Save markdown file
    mkd_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mkd_file_path, "w", encoding="utf-8") as f:
        f.write(markdown_texts)

    # Save images
    for item in markdown_images:
        if item:
            for path, image in item.items():
                # Image save path: current PDF output directory + image relative path
                img_file_path = current_output / path
                img_file_path.parent.mkdir(parents=True, exist_ok=True)
                image.save(img_file_path)

def process_pdfs(dataset_name, start=0, end=None, url=None):
    # Build input and output paths
    datasets_dir = Path("./datasets") / dataset_name
    output_root = Path("./outputs") / dataset_name

    # Check if dataset directory exists
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {datasets_dir}")

    # Get all PDF files in the directory
    pdf_files = list(datasets_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: No PDF files found in dataset directory {datasets_dir}")
        return

    # Parameter validation and file slicing
    total_pdf_count = len(pdf_files)
    
    # Validate that start index is not negative
    if start < 0:
        raise ValueError(f"Start index cannot be negative, current value: {start}")
    
    # Default logic for end parameter
    if end is None:
        actual_end = total_pdf_count
    else:
        if end <= start:
            raise ValueError(f"End index must be greater than start index! Current start={start}, end={end}")
        actual_end = min(end, total_pdf_count)
    
    # Slice to get PDF files in the specified range
    pdf_files_to_process = pdf_files[start:actual_end]
    
    # Inform user of processing range
    print(f"📄 Detected {total_pdf_count} PDF files in total")
    end_display = "last" if end is None else actual_end
    print(f"🔍 Will process PDF files from index [{start} - {end_display}) , total {len(pdf_files_to_process)} files")
    if not pdf_files_to_process:
        print(f"⚠️  No PDF files available in the specified range [{start} - {end_display}), task terminated")
        return

    # Initialize OCR pipeline
    pipeline = PaddleOCRVL(
        vl_rec_backend="vllm-server",
        vl_rec_server_url=url,
        vl_rec_api_model_name="PaddleOCR-VL-0.9B"
    )

    # Process each PDF in the specified range in batch (with retry logic)
    max_retries = 3  # Maximum number of retries
    for pdf_file in tqdm(pdf_files_to_process, desc="Processing PDFs"):
        retry_count = 0
        while retry_count < max_retries:
            try:
                # Call the single PDF processing function
                process_single_pdf(pipeline, pdf_file, output_root)
                break  # Successful, exit retry loop
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    # Print prompt before retry, with a short delay to avoid high-frequency retries
                    print(f"\n❌ Failed to process {pdf_file} (retry {retry_count}/{max_retries}), error: {str(e)}")
                    time.sleep(1)  # Delay 1 second before retry
                else:
                    # Retries exhausted, raise final error
                    raise RuntimeError(f"Failed to process {pdf_file} after {max_retries} retries, final error: {str(e)}") from e

    print(f"\n✅ All PDFs in the specified range processed successfully, results saved to: {output_root}")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Process PDF files in a specified range under a given dataset")
    parser.add_argument("--dataset_name", help="Dataset name (corresponding folder name under ./datasets/)")
    parser.add_argument(
        "--start", 
        type=int, 
        default=0, 
        help="Start PDF index (0-based, default: 0)"
    )
    parser.add_argument(
        "--end", 
        type=int, 
        default=None,
        help="End PDF index (exclusive, default processes all PDFs)"
    )
    parser.add_argument(
        "--url", 
        type=str, 
        default="http://127.0.0.1:8118", 
        help="OCR service URL"
    )
    args = parser.parse_args()

    # Execute processing
    process_pdfs(args.dataset_name, args.start, args.end, args.url)