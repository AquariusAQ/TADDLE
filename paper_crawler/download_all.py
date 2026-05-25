import os
import json
import time
from tqdm import tqdm
import openreview

# -------------------------- Configuration --------------------------
def read_key_file(file_path):
    """Read content from a key file (automatically strip whitespace)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

MAX_PAPERS_PER_CONFERENCE = 3  # Maximum number of papers to process per conference (adjustable)

# Read username and password from key files
OPENREVIEW_USERNAME = read_key_file("username.key") or "YOUR_USERNAME"  
OPENREVIEW_PASSWORD = read_key_file("password.key") or "YOUR_PASSWORD"  
CONFERENCES = {
    "ICLR2025": "ICLR.cc/2025/Conference",
    "NeurIPS2025": "NeurIPS.cc/2025/Conference",
    "ICML2025": "ICML.cc/2025/Conference"
}
# Rating fields for each conference (ICML uses overall_recommendation, others use rating)
CONFERENCE_RATING_FIELDS = {
    "ICLR2025": "rating",
    "NeurIPS2025": "rating",
    "ICML2025": "overall_recommendation"
}
# Rating difference thresholds for each conference (max(rating) - min(rating))
CONFERENCE_THRESHOLDS = {
    "ICLR2025": 3,
    "NeurIPS2025": 2,
    "ICML2025": 2
}
DATASET_ROOT = "./dataset/raw"
RETRY_TIMES = 3  # Number of retries
DELAY = 0.5  # API call delay (to avoid rate limits)
PAPER_LIST_CACHE_FILE = "all_papers.json"  # Paper list cache file name
# Decision type keyword mapping (adapted to venue field format)
DECISION_KEYWORDS = {
    "Accept (Oral)": ["Oral", "oral"],
    "Accept (Spotlight)": ["Spotlight", "spotlight"],
    "Accept (Poster)": ["Poster", "poster"],
    "Reject": ["Reject", "reject"],
    "Withdrawn": ["Withdrawn", "withdrawn"],
    "Under Review": ["Under Review", "under review", "Submission", "submission"]
}
# Paper statuses to exclude (do not download)
EXCLUDED_DECISIONS = {"Withdrawn", "Under Review"}
# ------------------------------------------------------------

def init_openreview_client():
    """Initialize OpenReview V2 client"""
    try:
        client = openreview.api.OpenReviewClient(
            baseurl='https://api2.openreview.net',
            username=OPENREVIEW_USERNAME,
            password=OPENREVIEW_PASSWORD
        )
        print("OpenReview client initialized successfully")
        return client
    except Exception as e:
        print(f"Client initialization failed: {str(e)}")
        print("Please check if the username/password is correct or if the network is available")
        exit(1)

def create_folder_structure(conf_name, decision_types):
    """Create folder structure of conference - decision type - paper ID"""
    # Filter out excluded states, do not create corresponding folders
    valid_decision_types = [dt for dt in decision_types if dt not in EXCLUDED_DECISIONS]
    
    conf_root = os.path.join(DATASET_ROOT, conf_name)
    os.makedirs(conf_root, exist_ok=True)
    
    # Save decision type list (includes all types, but only create folders for valid types)
    with open(os.path.join(conf_root, "decision_types.json"), "w", encoding="utf-8") as f:
        json.dump(decision_types, f, indent=2)
    
    # Create valid decision type folders (clean illegal characters)
    for dt in valid_decision_types:
        dt_folder = os.path.join(conf_root, dt.replace("/", "_").replace(" ", "_"))
        os.makedirs(dt_folder, exist_ok=True)
    
    # Create downloaded paper record files (distinguish whether PDF was downloaded)
    downloaded_file = os.path.join(conf_root, "downloaded_papers.txt")
    pdf_downloaded_file = os.path.join(conf_root, "pdf_downloaded_papers.txt")
    for f_path in [downloaded_file, pdf_downloaded_file]:
        if not os.path.exists(f_path):
            with open(f_path, "w", encoding="utf-8") as f:
                f.write("")
    
    return conf_root, downloaded_file, pdf_downloaded_file

def get_conf_notes(client, conf_group_id, conf_root):
    """Get all submitted papers for the conference (prioritize loading cache, if no cache then fetch via API and save)"""
    cache_path = os.path.join(conf_root, PAPER_LIST_CACHE_FILE)
    
    # Check if cache file exists
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cached_notes = json.load(f)
            print(f"Loaded cached paper list, {len(cached_notes)} papers in total")
            # Convert to Note objects (maintain compatibility with subsequent logic)
            notes = [openreview.api.Note.from_json(n) for n in cached_notes]
            return notes
        except Exception as e:
            print(f"Cache file corrupted: {str(e)}, will re-fetch paper list")
    
    # Cache does not exist or is corrupted, calling API to fetch
    notes = []
    offset = 0
    limit = 100  # Maximum number per query (API limit)
    
    while True:
        try:
            # Call V2 API to fetch papers, filter submission type notes
            batch_notes = client.get_notes(
                invitation=f"{conf_group_id}/-/Submission",
                limit=limit,
                offset=offset,
                details="all"
            )
            if not batch_notes:
                break  # No more papers
            
            notes.extend(batch_notes)
            offset += limit
            time.sleep(DELAY)
            print(f"Fetched {len(notes)} papers...")
            if len(notes) >= MAX_PAPERS_PER_CONFERENCE != -1:
                print(f"Reached maximum number of papers to process ({MAX_PAPERS_PER_CONFERENCE}), stopping fetching more papers")
                break
        
        except Exception as e:
            print(f"Failed to fetch paper list (offset={offset}): {str(e)}")
            # Retry mechanism
            for retry in range(RETRY_TIMES):
                time.sleep(2 ** retry)  # Exponential backoff
                try:
                    batch_notes = client.get_notes(
                        invitation=f"{conf_group_id}/-/Submission",
                        limit=limit,
                        offset=offset,
                        details="all"
                    )
                    notes.extend(batch_notes)
                    offset += limit
                    time.sleep(DELAY)
                    break
                except:
                    if retry == RETRY_TIMES - 1:
                        raise  # Retries exhausted, still failed, raising exception
            else:
                break
    
    # Save paper list to cache file (convert to JSON format)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump([note.to_json() for note in notes], f, indent=2, ensure_ascii=False)
    print(f"Paper list saved to cache file: {cache_path}")
    
    return notes

def extract_decision_types(notes):
    """Extract decision types from paper venue field (case-insensitive)"""
    decision_types = set()
    for note in notes:
        # Match after converting to lowercase, covering case scenarios
        venue = note.content.get("venue", {}).get("value", "").strip().lower()
        venueid = note.content.get("venueid", {}).get("value", "").strip().lower()
        decision = "Unknown"
        
        # Keywords are also matched in lowercase
        for target_dt, keywords in DECISION_KEYWORDS.items():
            if any(keyword.lower() in venue or keyword.lower() in venueid for keyword in keywords):
                decision = target_dt
                break
        
        decision_types.add(decision)
    return sorted(list(decision_types))

def get_paper_reviews(client, note_id):
    """Get the complete review process for a paper (only return raw content, remove rating extraction)"""
    reviews = {
        "reviews": [],          # Regular reviews
        "meta_reviews": []      # Meta reviews
    }
    
    try:
        # Find all notes replying to the paper (review content)
        reply_notes = client.get_notes(
            replyto=note_id,
            details="all"
        )
        
        # Iterate over review content, classify and save
        for reply in reply_notes:
            reply_json = reply.to_json()
            inv_str = "|".join(reply.invitations) if reply.invitations else ""
            
            if "Review" in inv_str and "Meta" not in inv_str:
                reviews["reviews"].append(reply_json)
            elif "Meta_Review" in inv_str:
                reviews["meta_reviews"].append(reply_json)
        
        time.sleep(DELAY)
    except Exception as e:
        print(f"Failed to fetch review info for paper {note_id}: {str(e)}")
    
    return reviews

def calculate_review_rating_diff(reviews_data, conf_name):
    """
    Calculate the rating difference of reviews (max - min)
    :param reviews_data: Review data (returned by get_paper_reviews)
    :param conf_name: Conference name (used to match the rating field)
    :return: Rating difference (float)
    """
    ratings = []
    # Get the rating field corresponding to the current conference
    rating_field = CONFERENCE_RATING_FIELDS.get(conf_name, "rating")
    
    # Iterate over all regular reviews (excluding meta)
    for review in reviews_data.get("reviews", []):
        # Extract the value of the corresponding field and convert to number
        rating_value = review.get("content", {}).get(rating_field, {}).get("value", None)
        if rating_value is not None:
            try:
                # Compatible with string/number format ratings (e.g., "7" or 7)
                ratings.append(float(rating_value))
            except (ValueError, TypeError):
                # Non-numeric ratings skipped (e.g., "Accept" text)
                continue
    
    # Less than 2 valid ratings, rating difference is 0
    if len(ratings) < 2:
        return 0.0
    # Calculate maximum difference
    return max(ratings) - min(ratings)

def download_paper_pdf(client, note_id, save_path):
    """Download paper PDF (using V2 API's get_pdf method)"""
    try:
        # V2 API directly fetches PDF binary content via note.id
        pdf_content = client.get_pdf(note_id)
        with open(save_path, "wb") as f:
            f.write(pdf_content)
        time.sleep(DELAY)
        return True
    except Exception as e:
        print(f"Failed to download PDF for paper {note_id}: {str(e)}")
        if os.path.exists(save_path):
            os.remove(save_path)  # Delete incomplete file
        # Retry mechanism
        for retry in range(RETRY_TIMES):
            time.sleep(2 ** retry)
            try:
                pdf_content = client.get_pdf(note_id)
                with open(save_path, "wb") as f:
                    f.write(pdf_content)
                time.sleep(DELAY)
                return True
            except:
                if retry == RETRY_TIMES - 1:
                    return False
        return False

def get_paper_decision(note):
    """Get the decision type of a single paper (case-insensitive)"""
    venue = note.content.get("venue", {}).get("value", "").strip().lower()
    venueid = note.content.get("venueid", {}).get("value", "").strip().lower()
    
    for target_dt, keywords in DECISION_KEYWORDS.items():
        if any(keyword.lower() in venue or keyword.lower() in venueid for keyword in keywords):
            return target_dt
    return "Unknown"

def download_paper(client, conf_name, conf_root, downloaded_file, pdf_downloaded_file, note):
    """
    Download a single paper (decide whether to download PDF based on rating difference threshold)
    - Difference >= threshold: download PDF + paper info + review
    - Difference < threshold: only download paper info + review, no PDF download
    """
    note_id = note.id
    # Get paper decision (filter excluded states)
    decision = get_paper_decision(note)
    if decision in EXCLUDED_DECISIONS:
        print(f"Paper {note_id} status is {decision}, skipping download")
        return True, False
    
    # Build save path
    decision_folder = os.path.join(conf_root, decision.replace("/", "_").replace(" ", "_"))
    paper_folder = os.path.join(decision_folder, note_id)
    os.makedirs(paper_folder, exist_ok=True)
    
    # Check if already processed (regardless of PDF download)
    with open(downloaded_file, "r", encoding="utf-8") as f:
        downloaded_ids = f.read().splitlines()
    if note_id in downloaded_ids:
        print(f"Paper {note_id} already processed, skipping")
        return True, False
    
    try:
        # 1. Save paper basic information
        info_path = os.path.join(paper_folder, "paper_info.json")
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(note.to_json(), f, indent=2, ensure_ascii=False)
        
        # 2. Get review content
        reviews_data = get_paper_reviews(client, note_id)
        
        # 3. Save review content
        reviews_path = os.path.join(paper_folder, "reviews.json")
        with open(reviews_path, "w", encoding="utf-8") as f:
            json.dump(reviews_data, f, indent=2, ensure_ascii=False)
        
        # 4. Calculate rating difference, determine whether to download PDF
        rating_diff = calculate_review_rating_diff(reviews_data, conf_name)
        threshold = CONFERENCE_THRESHOLDS.get(conf_name, 2)  # Default threshold 2
        pdf_downloaded = False
        
        if rating_diff >= threshold:
            # Difference >= threshold, downloading PDF
            pdf_path = os.path.join(paper_folder, "paper.pdf")
            if download_paper_pdf(client, note_id, pdf_path):
                pdf_downloaded = True
                # Record PDF downloaded
                with open(pdf_downloaded_file, "a", encoding="utf-8") as f:
                    f.write(f"{note_id}\n")
                print(f"Paper {note_id} download completed (type: {decision}, rating difference: {rating_diff:.1f} >= {threshold}, PDF downloaded)")
            else:
                print(f"Paper {note_id} information saved, but PDF download failed (type: {decision}, rating difference: {rating_diff:.1f} >= {threshold})")
        else:
            # Difference < threshold, not downloading PDF
            print(f"Paper {note_id} information saved (type: {decision}, rating difference: {rating_diff:.1f} < {threshold}, no PDF downloaded)")
        
        # 5. Record paper as processed (regardless of PDF download)
        with open(downloaded_file, "a", encoding="utf-8") as f:
            f.write(f"{note_id}\n")
        
        return True, pdf_downloaded
    except Exception as e:
        print(f"Paper {note_id} processing exception: {str(e)}")
        return False, False

def main():
    """Main process"""
    # Initialize client
    client = init_openreview_client()
    
    for conf_name, conf_group in CONFERENCES.items():
        print(f"\n===== Starting to process conference: {conf_name} (rating difference threshold: {CONFERENCE_THRESHOLDS[conf_name]}) =====")
        
        # First create conference root directory (for saving cache files)
        conf_root = os.path.join(DATASET_ROOT, conf_name)
        os.makedirs(conf_root, exist_ok=True)
        
        # 1. Get all papers for the conference (prioritize loading cache)
        print("Fetching paper list (prioritizing local cache)...")
        notes = get_conf_notes(client, conf_group, conf_root)
        if not notes:
            print(f"No paper data obtained for {conf_name}, skipping")
            continue
        print(f"Fetched {len(notes)} papers in total")
        if MAX_PAPERS_PER_CONFERENCE != -1:
            notes = notes[:MAX_PAPERS_PER_CONFERENCE]
            print(f"Limited to processing first {MAX_PAPERS_PER_CONFERENCE} papers")
        
        # 2. Extract decision types and create folders (automatically filter excluded statuses)
        print("Extracting decision types...")
        decision_types = extract_decision_types(notes)
        valid_decision_types = [dt for dt in decision_types if dt not in EXCLUDED_DECISIONS]
        print(f"All types for this conference: {decision_types}")
        print(f"Types to process: {valid_decision_types}")
        conf_root, downloaded_file, pdf_downloaded_file = create_folder_structure(conf_name, decision_types)
        
        # 3. Process papers in batch
        print("Starting to process papers (supports resume, download PDF based on rating difference threshold)...")
        success_count = 0       # Successfully processed (info saved)
        fail_count = 0          # Processing failed
        skipped_count = 0       # Skipped (excluded status)
        pdf_success_count = 0   # PDFs downloaded successfully
        
        for note in tqdm(notes, desc=f"{conf_name} processing progress"):
            decision = get_paper_decision(note)
            if decision in EXCLUDED_DECISIONS:
                skipped_count += 1
                continue
            # Process paper, returns success status and whether PDF was downloaded
            is_success, is_pdf_downloaded = download_paper(client, conf_name, conf_root, downloaded_file, pdf_downloaded_file, note)
            if is_success:
                success_count += 1
                if is_pdf_downloaded:
                    pdf_success_count += 1
            else:
                fail_count += 1
        
        # 4. Output statistics
        print(f"\n{conf_name} processing completed:")
        print(f"✅ Successfully processed (info saved): {success_count} papers")
        print(f"📄 PDFs downloaded successfully: {pdf_success_count} papers")
        print(f"❌ Processing failed: {fail_count} papers")
        print(f"⏭️  Skipped (excluded status): {skipped_count} papers")
        print(f"📁 Storage path: {conf_root}")
        print(f"📄 Paper list cache: {os.path.join(conf_root, PAPER_LIST_CACHE_FILE)}")
    
    print("\n===== All conferences processed =====")

if __name__ == "__main__":
    main()