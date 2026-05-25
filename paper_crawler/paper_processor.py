import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===================== Core Configuration =====================
# Conference configuration: {conference_name: (score threshold, score field name)}
CONFERENCE_CONFIG = {
    "ICLR2025": (3, "rating"),          # ICLR uses the rating field, threshold 3
    "ICML2025": (2, "overall_recommendation"),  # ICML uses overall_recommendation field, threshold 2
    "NeurIPS2025": (2, "rating")        # NeurIPS uses the rating field, threshold 2
}

# Root directory of raw data
RAW_DATA_ROOT = "./dataset/raw"
# Output directory
OUTPUT_ROOT = "./dataset/processed"
# Folder names for acceptance types
ACCEPT_TYPE_FOLDERS = [
    "Accept_(Oral)", "Accept_(Poster)", "Accept_(Spotlight)", 
    "Reject", "Accepted", "Rejected"
]

# ===================== Utility Functions =====================
def load_json_file(file_path: str) -> Optional[Dict]:
    """Load a JSON file, with exception handling"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️  File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"⚠️  JSON decode error: {file_path}")
        return None
    except Exception as e:
        print(f"⚠️  Failed to load file {file_path}: {str(e)}")
        return None

def extract_paper_content(paper_info: Dict) -> Dict:
    """Extract core paper content, removing the wrapping 'value' field"""
    content_raw = paper_info.get("content", {})
    paper_content = {}
    
    fields = ["title", "authors", "keywords", "TLDR", "abstract"]
    for field in fields:
        if field in content_raw and "value" in content_raw[field]:
            paper_content[field] = content_raw[field]["value"]
        else:
            paper_content[field] = ""
    
    return paper_content

def extract_reviews(reviews_data: Dict) -> List[Dict]:
    """Extract review data, removing the wrapping 'value' field"""
    reviews = []
    for review in reviews_data.get("reviews", []):
        review_info = {"review_id": review.get("id", "")}
        review_content = review.get("content", {})
        
        for key, value in review_content.items():
            if isinstance(value, dict) and "value" in value:
                review_info[key] = value["value"]
            else:
                review_info[key] = value
        
        reviews.append(review_info)
    return reviews

def calculate_review_scores(reviews: List[Dict], score_field: str) -> Tuple[List[float], float, float, float]:
    """
    Calculate review score metrics.
    Returns: [list of all scores, max score, min score, average score]
    """
    scores = []
    for review in reviews:
        score = review.get(score_field)
        if isinstance(score, (int, float)):
            scores.append(float(score))
    
    if not scores:
        return [], 0.0, 0.0, 0.0
    
    max_score = max(scores)
    min_score = min(scores)
    avg_score = sum(scores) / len(scores)
    return scores, max_score, min_score, avg_score

def check_review_score(scores: List[float], threshold: int) -> bool:
    """Check if the paper's review scores meet the requirement: max(scores) - min(scores) >= threshold"""
    if not scores:
        return False
    return (max(scores) - min(scores)) >= threshold

def rename_and_copy_pdf(paper_id: str, src_pdf_path: str, dst_pdf_dir: str) -> bool:
    """Rename the PDF to the paper ID and copy to the target directory"""
    try:
        Path(dst_pdf_dir).mkdir(parents=True, exist_ok=True)
        dst_pdf_path = os.path.join(dst_pdf_dir, f"{paper_id}.pdf")
        shutil.copy2(src_pdf_path, dst_pdf_path)
        return True
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"⚠️  PDF copy failed for {src_pdf_path}: {str(e)}")
        return False

# ===================== Statistics Functions =====================
def init_conference_stats() -> Dict:
    """Initialize the conference statistics data structure"""
    return {
        "accept_type_stats": {},  # {accept_type: {"count": number, "total_score": sum, "avg_score": average}}
        "review_count_distribution": {},  # {review_count: paper_count}
        "overall_stats": {
            "total_papers": 0,
            "total_score": 0.0,
            "avg_score": 0.0
        }
    }

def update_conference_stats(stats: Dict, accept_type: str, paper_avg_score: float, review_count: int):
    """Update conference statistics"""
    # 1. Update acceptance type statistics
    if accept_type not in stats["accept_type_stats"]:
        stats["accept_type_stats"][accept_type] = {
            "count": 0,
            "total_score": 0.0,
            "avg_score": 0.0
        }
    accept_stats = stats["accept_type_stats"][accept_type]
    accept_stats["count"] += 1
    accept_stats["total_score"] += paper_avg_score
    accept_stats["avg_score"] = accept_stats["total_score"] / accept_stats["count"]
    
    # 2. Update review count distribution
    review_count_key = str(review_count)
    if review_count_key not in stats["review_count_distribution"]:
        stats["review_count_distribution"][review_count_key] = 0
    stats["review_count_distribution"][review_count_key] += 1
    
    # 3. Update overall statistics
    stats["overall_stats"]["total_papers"] += 1
    stats["overall_stats"]["total_score"] += paper_avg_score
    stats["overall_stats"]["avg_score"] = (
        stats["overall_stats"]["total_score"] / stats["overall_stats"]["total_papers"]
        if stats["overall_stats"]["total_papers"] > 0 else 0.0
    )

def print_conference_stats(conference_name: str, stats: Dict):
    """Print conference statistics"""
    print(f"\n=====================================")
    print(f"📊 {conference_name} Statistics Summary")
    print(f"=====================================")
    
    # 1. Acceptance type statistics
    print(f"\n1. Statistics by acceptance type:")
    for accept_type, accept_stats in stats["accept_type_stats"].items():
        print(f"   - {accept_type}: count={accept_stats['count']}, average score={accept_stats['avg_score']:.2f}")
    
    # 2. Overall statistics
    print(f"\n2. Overall conference statistics:")
    print(f"   - Total papers: {stats['overall_stats']['total_papers']}")
    print(f"   - Overall average score: {stats['overall_stats']['avg_score']:.2f}")
    
    # 3. Review count distribution
    print(f"\n3. Review count distribution:")
    # Sort output by review count
    sorted_review_counts = sorted(
        stats["review_count_distribution"].items(),
        key=lambda x: int(x[0])
    )
    for review_count, paper_count in sorted_review_counts:
        print(f"   - Papers with {review_count} reviews: {paper_count}")
    print(f"-------------------------------------")

# ===================== Core Processing Function =====================
def process_conference(conference_name: str, conference_dir: str) -> Dict:
    """Process all data for a single conference, returning statistics"""
    threshold, score_field = CONFERENCE_CONFIG[conference_name]
    conference_output_dir = os.path.join(OUTPUT_ROOT, conference_name)
    Path(conference_output_dir).mkdir(parents=True, exist_ok=True)
    conference_pdf_dir = os.path.join(conference_output_dir, conference_name)
    
    conference_papers = []
    # Initialize statistics
    conference_stats = init_conference_stats()
    
    for accept_type in ACCEPT_TYPE_FOLDERS:
        accept_type_dir = os.path.join(conference_dir, accept_type)
        if not os.path.exists(accept_type_dir):
            continue
        
        for paper_folder in os.listdir(accept_type_dir):
            paper_dir = os.path.join(accept_type_dir, paper_folder)
            if not os.path.isdir(paper_dir):
                continue
            
            # Load paper data
            paper_info_path = os.path.join(paper_dir, "paper_info.json")
            reviews_path = os.path.join(paper_dir, "reviews.json")
            paper_info = load_json_file(paper_info_path)
            reviews_data = load_json_file(reviews_path)
            
            if not paper_info or not reviews_data:
                print(f"⚠️  Missing paper information, skipping folder: {paper_dir}")
                continue
            
            paper_id = paper_info.get("id", "")
            if not paper_id:
                print(f"⚠️  Paper ID is empty, skipping folder: {paper_dir}")
                continue
            
            # Extract data and calculate scores
            paper_content = extract_paper_content(paper_info)
            reviews = extract_reviews(reviews_data)
            scores, max_score, min_score, paper_avg_score = calculate_review_scores(reviews, score_field)
            
            # Filter papers that meet the threshold
            if not check_review_score(scores, threshold):
                continue
            
            # Process PDF
            pdf_src = None
            pdf_relative_path = paper_info.get("content", {}).get("pdf", {}).get("value")
            if pdf_relative_path:
                pdf_src = os.path.join(conference_dir, pdf_relative_path.lstrip("/"))
            if not pdf_src or not os.path.exists(pdf_src):
                pdf_files = [f for f in os.listdir(paper_dir) if f.endswith(".pdf")]
                if pdf_files:
                    pdf_src = os.path.join(paper_dir, pdf_files[0])
            
            if not pdf_src or not os.path.exists(pdf_src):
                print(f"⚠️  Paper {paper_id} meets score threshold, but no corresponding PDF found (path: {paper_dir})")
            
            if pdf_src and os.path.exists(pdf_src):
                rename_and_copy_pdf(paper_id, pdf_src, conference_pdf_dir)
            
            # Assemble paper data
            paper_data = {
                "paper_id": paper_id,
                "paper_content": paper_content,
                "accept_type": accept_type,
                "reviews": reviews,
                "score_metrics": {  # New score metrics for later verification
                    "max_score": max_score,
                    "min_score": min_score,
                    "avg_score": paper_avg_score,
                    "score_diff": max_score - min_score
                }
            }
            conference_papers.append(paper_data)
            
            # Update statistics
            review_count = len(reviews)
            update_conference_stats(conference_stats, accept_type, paper_avg_score, review_count)
    
    # Save JSON file
    if conference_papers:
        output_json_path = os.path.join(conference_output_dir, f"{conference_name}_processed.json")
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {"conference": conference_name, "papers": conference_papers, "stats": conference_stats},
                f,
                indent=2,
                ensure_ascii=False
            )
    
    # Print statistics
    print_conference_stats(conference_name, conference_stats)
    return conference_stats

# ===================== Main Function =====================
def main():
    """Main function: iterate over all conferences and process them"""
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    # Store statistics for all conferences
    all_conference_stats = {}
    
    for conference_folder in os.listdir(RAW_DATA_ROOT):
        conference_name = conference_folder.strip()
        if conference_name not in CONFERENCE_CONFIG:
            continue
        
        conference_dir = os.path.join(RAW_DATA_ROOT, conference_folder)
        if not os.path.isdir(conference_dir):
            continue
        
        # Process conference and get statistics
        conference_stats = process_conference(conference_name, conference_dir)
        all_conference_stats[conference_name] = conference_stats
    
    # Save global statistics
    global_stats_path = os.path.join(OUTPUT_ROOT, "all_conferences_stats.json")
    with open(global_stats_path, "w", encoding="utf-8") as f:
        json.dump(all_conference_stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 All conferences processed!")
    print(f"📁 Output directory: {OUTPUT_ROOT}")
    print(f"📄 Global statistics file: {global_stats_path}")

if __name__ == "__main__":
    main()