import requests
import time
from typing import Dict, Optional
import json
from utils.config import config
from utils.logger import logger

search_logger = None

# ========== API Service Configuration ==========
API_BASE_URL = None
API_TIMEOUT = None
API_RETRY_COUNT = None
API_RETRY_DELAY = None

def _call_search_api(api_choice: str, query: str, limit: int, offset: int) -> Optional[Dict]:
    """
    Internal function: Call search API service
    :param api_choice: Selected API (semanticscholar/arxiv)
    :param query: Search keywords
    :param limit: Number of returned results
    :param offset: Result offset
    :return: Search results (consistent with original local call format)
    """
    global API_BASE_URL
    global API_TIMEOUT
    global API_RETRY_COUNT
    global API_RETRY_DELAY
    # Read API service address from configuration file (priority), default to local service
    if not API_BASE_URL:
        API_BASE_URL = config.get("search_api", {}).get("base_url", "http://localhost:8000/api")
    # API call timeout (seconds)
    if not API_TIMEOUT:
        API_TIMEOUT = config.get("search_api", {}).get("timeout", 30)
    # Number of API call retries
    if not API_RETRY_COUNT:
        API_RETRY_COUNT = config.get("search_api", {}).get("retry_count", 2)
    # Retry interval (seconds)
    if not API_RETRY_DELAY:
        API_RETRY_DELAY = config.get("search_api", {}).get("retry_delay", 2)

    url = f"{API_BASE_URL}/search"
    params = {
        "query": query,
        "limit": limit,
        "offset": offset,
        "api_choice": api_choice
    }
    
    search_logger.info(f"Calling search API: {url}, Parameters: {json.dumps(params, ensure_ascii=False)}")
    
    # Retry mechanism
    for retry in range(API_RETRY_COUNT + 1):
        try:
            response = requests.get(
                url=url,
                params=params,
                timeout=API_TIMEOUT,
                headers={"Content-Type": "application/json"}
            )
            
            # Check response status code
            response.raise_for_status()
            
            # Parse response results
            result = response.json()
            
            if not result.get("success", False):
                search_logger.error(f"API call failed: {result.get('message', 'Unknown error')}")
                return result
            
            # Return data part, consistent with original local call format
            return result.get("data", None)
        
        except requests.exceptions.RequestException as e:
            error_msg = f"API call exception (Attempt {retry + 1}): {str(e)}"
            if retry < API_RETRY_COUNT:
                search_logger.warning(f"{error_msg}, retrying in {API_RETRY_DELAY} seconds...")
                time.sleep(API_RETRY_DELAY)
            else:
                search_logger.error(f"{error_msg}, maximum retry attempts reached")
                return None
        except json.JSONDecodeError as e:
            search_logger.error(f"API response parsing failed: {str(e)}")
            return None
        except Exception as e:
            search_logger.error(f"Unknown API call error: {str(e)}", exc_info=True)
            return None
    
    return None

def search_papers(query, api_choice="arxiv", limit=10, offset=0):
    """
    Search papers (call remote API service)
    :param query: Search keywords
    :param limit: Number of returned results (default: 10)
    :param offset: Result offset (default: 0)
    :return: Search results (exactly the same format as original local call)
    """
    global search_logger
    if search_logger is None:
        search_logger = logger.search
    
    # Read configuration
    search_config = config.get("search_papers", {})
    # api_choice = search_config.get("api", "semanticscholar")
    max_results = search_config.get("max_results", 10)
    
    # Validate and correct parameters
    if not query or len(query.strip()) == 0:
        search_logger.error("Search keywords cannot be empty")
        raise ValueError("Search keywords cannot be empty")
    
    limit = min(limit, max_results)  # Limit maximum number of results
    offset = max(offset, 0)  # Ensure non-negative offset
    
    search_logger.info("========== Starting external search (API call) ==========")
    search_logger.info(f"Searching papers using {api_choice} API, query: '{query}', limit: {limit}, offset: {offset}")
    
    # Call API service
    result = _call_search_api(api_choice, query, limit, offset)
    
    if result is None:
        search_logger.error("========== External search failed ==========")
        return None
    
    search_logger.info(f"========== External search successful, retrieved {result.get('total', 0)} results ==========")
    return result

# ========== Compatibility with original function names (optional, if original call methods need to be retained) ==========
def search_papers_semanticscholar(query, limit=10, offset=0):
    """Compatibility with original function: Call semanticscholar search via API"""
    global search_logger
    if search_logger is None:
        search_logger = logger.search
    max_results = config.get("search_papers", {}).get("max_results", 10)
    limit = min(limit, max_results)
    return _call_search_api("semanticscholar", query, limit, offset)

def search_papers_arxiv(query, limit=10, offset=0):
    """Compatibility with original function: Call arxiv search via API"""
    global search_logger
    if search_logger is None:
        search_logger = logger.search
    max_results = config.get("search_papers", {}).get("max_results", 10)
    limit = min(limit, max_results)
    return _call_search_api("arxiv", query, limit, offset)