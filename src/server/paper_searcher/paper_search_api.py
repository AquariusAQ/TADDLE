import requests
import threading
import time
import json
import os
import hashlib
from typing import Dict, Optional, List, Union, Any
import arxiv
import itertools
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse
import uvicorn

# Mock the original configuration and logging modules (replace with real imports if available in the actual project)
class MockConfig:
    def __init__(self):
        self.config_data = {
            "search_papers": {
                "api": "semanticscholar",
                "max_results": 50,
                "default_limit": 10
            },
            # Cache configuration
            "search_cache": {
                "enabled": True,
                "cache_dir": "./api_search_cache",
                "expire_hours": 24,
                "max_cache_count": 1000,
                "cleanup_interval": 24 * 30
            }
        }
    
    def get(self, key: str, default: Optional[Dict] = None) -> Dict:
        return self.config_data.get(key, default or {})

class MockLogger:
    def error(self, msg: str):
        print(f"[ERROR] {msg}")
    
    def info(self, msg: str):
        print(f"[INFO] {msg}")
    
    def warning(self, msg: str):
        print(f"[WARNING] {msg}")

# Initialize configuration and logger
config = MockConfig()
logger = MockLogger()
logger.search = MockLogger()
search_logger = logger.search

# Define type aliases
ApiConfig = Dict[str, float]

# ========== Cache Utility Class ==========
class SearchCache:
    """API Server-side Cache Manager"""
    def __init__(self):
        # Load cache configuration
        self.cache_config = config.get("search_cache", {})
        self.enabled = self.cache_config.get("enabled", True)
        self.cache_dir = self.cache_config.get("cache_dir", "./api_search_cache")
        self.expire_hours = self.cache_config.get("expire_hours", 24)
        self.max_cache_count = self.cache_config.get("max_cache_count", 1000)
        self.cleanup_interval = self.cache_config.get("cleanup_interval", 24) * 3600
        
        # State variables
        self.last_cleanup_time = 0
        self.init_cache_dir()
    
    def init_cache_dir(self):
        """Initialize cache directory"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
            search_logger.info(f"Cache directory initialized: {self.cache_dir}")
    
    def generate_cache_key(self, query: str, api_choice: str, limit: int, offset: int) -> str:
        """Generate unique cache key (based on query parameters)"""
        key_str = f"query:{query}|api:{api_choice}|limit:{limit}|offset:{offset}".encode("utf-8")
        return hashlib.md5(key_str).hexdigest() + ".json"
    
    def is_cache_valid(self, cache_data: Dict) -> bool:
        """Check if cache is valid (not expired)"""
        create_time = cache_data.get("create_time", 0)
        expire_timestamp = create_time + self.expire_hours * 3600
        return time.time() < expire_timestamp
    
    def load_cache(self, cache_key: str) -> Optional[Dict]:
        """Load cached data"""
        if not self.enabled:
            return None
        
        cache_path = os.path.join(self.cache_dir, cache_key)
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
            
            if self.is_cache_valid(cache_data):
                remaining_hours = int((cache_data["create_time"] + self.expire_hours*3600 - time.time())/3600)
                search_logger.info(f"Cache hit: {cache_key} (Remaining validity: {remaining_hours} hours)")
                return cache_data["data"]
            else:
                search_logger.info(f"Cache expired: {cache_key}, will delete and retrieve again")
                os.remove(cache_path)
                return None
        except Exception as e:
            search_logger.error(f"Failed to load cache: {cache_key}, Error: {str(e)}")
            if os.path.exists(cache_path):
                os.remove(cache_path)
            return None
    
    def save_cache(self, cache_key: str, data: Dict):
        """Save data to cache"""
        if not self.enabled or data is None:
            return
        
        self.init_cache_dir()
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        cache_data = {
            "data": data,
            "create_time": time.time(),
            "query_info": {
                "cache_key": cache_key,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            }
        }
        
        try:
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            search_logger.info(f"Cache saved successfully: {cache_key}")
            self.cleanup_cache_if_needed()
        except Exception as e:
            search_logger.error(f"Failed to save cache: {cache_key}, Error: {str(e)}")
    
    def cleanup_cache(self):
        """Clean up expired cache and old cache exceeding quantity limit"""
        if not self.enabled:
            return
        
        try:
            # Get all cache files and sort by creation time (old → new)
            cache_files = [
                (os.path.join(self.cache_dir, f), os.path.getctime(os.path.join(self.cache_dir, f)))
                for f in os.listdir(self.cache_dir)
                if f.endswith(".json")
            ]
            cache_files.sort(key=lambda x: x[1])
            
            deleted_count = 0
            
            # 1. Delete expired cache
            for file_path, _ in cache_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                    if not self.is_cache_valid(cache_data):
                        os.remove(file_path)
                        deleted_count += 1
                except Exception:
                    os.remove(file_path)
                    deleted_count += 1
            
            # 2. Delete oldest files if exceeding max count
            remaining_files = [f for f in os.listdir(self.cache_dir) if f.endswith(".json")]
            if len(remaining_files) > self.max_cache_count:
                files_to_delete = sorted(
                    [os.path.join(self.cache_dir, f) for f in remaining_files],
                    key=lambda x: os.path.getctime(x)
                )[:len(remaining_files) - self.max_cache_count]
                
                for file_path in files_to_delete:
                    os.remove(file_path)
                    deleted_count += 1
            
            search_logger.info(f"Cache cleanup completed: {deleted_count} files deleted, Current cache count: {len(os.listdir(self.cache_dir))}")
            self.last_cleanup_time = time.time()
        except Exception as e:
            search_logger.error(f"Cache cleanup failed: {str(e)}")
    
    def cleanup_cache_if_needed(self):
        """Auto-cleanup cache at specified intervals"""
        current_time = time.time()
        if self.last_cleanup_time == 0 or current_time - self.last_cleanup_time > self.cleanup_interval:
            search_logger.info("Starting auto cache cleanup...")
            self.cleanup_cache()
    
    def clear_all_cache(self):
        """Clear all cache (for external calls)"""
        if not self.enabled:
            return
        
        try:
            deleted_count = 0
            for f in os.listdir(self.cache_dir):
                if f.endswith(".json"):
                    os.remove(os.path.join(self.cache_dir, f))
                    deleted_count += 1
            search_logger.info(f"All cache cleared: {deleted_count} files deleted in total")
            return {"success": True, "deleted_count": deleted_count}
        except Exception as e:
            search_logger.error(f"Failed to clear all cache: {str(e)}")
            return {"success": False, "error": str(e)}

# ========== Rate Limiter (Unchanged) ==========
class ApiRateLimiter:
    def __init__(self):
        self.locks: Dict[str, threading.Lock] = {}
        self.last_request_times: Dict[str, float] = {}
        self.default_interval: float = 3.0
        self.api_intervals: ApiConfig = {}

    def set_api_interval(self, api_name: str, interval: float):
        self.api_intervals[api_name] = interval
        if api_name not in self.locks:
            self.locks[api_name] = threading.Lock()

    def wait(self, api_name: str):
        lock = self.locks.setdefault(api_name, threading.Lock())
        with lock:
            current_time = time.time()
            interval = self.api_intervals.get(api_name, self.default_interval)
            last_time = self.last_request_times.get(api_name, 0.0)
            wait_time = interval - (current_time - last_time)
            
            if wait_time > 0:
                time.sleep(wait_time)
            
            self.last_request_times[api_name] = time.time()

# ========== Initialize Core Components ==========
rate_limiter = ApiRateLimiter()
rate_limiter.set_api_interval("semanticscholar", 3.0)
rate_limiter.set_api_interval("arxiv", 1.5)

cache_manager = SearchCache()  # Initialize cache manager

# ========== Original Search Functions (Unchanged) ==========
def search_papers_semanticscholar(query, limit=10, offset=0):
    # Read API key from key.json file, initialize static variable (executed only on first call)
    if not hasattr(search_papers_semanticscholar, 'api_key'):
        if os.path.exists("key.json"):
            try:
                # Read key.json in current directory (modify path if file is elsewhere)
                with open("key.json", "r", encoding="utf-8") as f:
                    api_key_json = json.load(f)
                    api_key = api_key_json.get("semanticscholar")
                if not api_key:
                    raise ValueError("key.json file is empty")
                search_papers_semanticscholar.api_key = api_key # Bind static variable to function
            except FileNotFoundError:
                logger.error("key.json file not found, please confirm the file exists and the path is correct")
                return {"error": "key.json file not found", "status_code": 500}
            except Exception as e:
                logger.error(f"Failed to read key.json file: {str(e)}")
                return {"error": f"Failed to read API key: {str(e)}", "status_code": 500}
        else:
            logger.info("key.json file not provided")
            search_papers_semanticscholar.api_key = None
    
    if search_papers_semanticscholar.api_key:
        headers = {"x-api-key": search_papers_semanticscholar.api_key}
    else:
        headers = {}

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "fields": "title,authors,abstract,year,url",
        "limit": limit,
        "offset": offset
    }
    
    rate_limiter.wait("semanticscholar")
    try:
        response = requests.get(url, params=params, headers=headers, timeout=15)
        response.raise_for_status()
        result = response.json()
        search_logger.info(f"Semantic Scholar search successful: query='{query}', total={result.get('total', 0)}")
        return result
    except requests.exceptions.HTTPError as e:
        # New: Catch 400 status code separately and return syntax error prompt
        if response.status_code == 400:
            logger.error(f"Semantic Scholar request failed: invalid search_query syntax, Error details: {str(e)}")
            return {"error": "Invalid search_query syntax, please modify or simplify the syntax and try again", "status_code": 400}
        elif response.status_code == 401:
            # New: Catch 401 unauthorized error and prompt API key issue
            logger.error(f"Semantic Scholar request failed: invalid or unauthorized API key, Error details: {str(e)}")
            return {"error": "Invalid or unauthorized API key, please check if your x-api-key is correct", "status_code": 401}
        else:
            logger.error(f"Semantic Scholar request failed (status code {response.status_code}): {str(e)}")
            return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Semantic Scholar request failed: {str(e)}")
        return None

def search_papers_arxiv(query, limit=10, offset=0):
    try:
        rate_limiter.wait("arxiv")
        
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=offset + limit,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = client.results(search)
        sliced_results = itertools.islice(results, offset, offset + limit)
        papers = list(sliced_results)
        
        formatted_results = {
            "total": len(papers),
            "offset": offset,
            "limit": limit,
            "data": [
                {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "abstract": paper.summary.strip(),
                    "year": paper.published.year,
                    "arxiv_id": paper.entry_id.split("/")[-1],
                    "url": paper.pdf_url,
                    "doi": paper.doi if hasattr(paper, 'doi') else None
                }
                for paper in papers
            ]
        }
        search_logger.info(f"arXiv search successful: query='{query}', total={len(papers)}")
        return formatted_results
    except arxiv.HTTPError as e:
        # Catch HTTP errors returned by arXiv API (including 400 status code)
        if e.status_code == 400:
            logger.error(f"arXiv request failed: invalid search_query syntax, Error details: {str(e)}")
            return {"error": "Invalid search_query syntax, please modify or simplify the syntax and try again", "status_code": 400}
        else:
            logger.error(f"arXiv request failed (status code {e.status_code}): {str(e)}")
            return None
    except Exception as e:
        # Catch other non-HTTP errors (e.g., network issues, invalid parameters)
        # If the exception is essentially a syntax issue, add additional judgment (based on arxiv library exception characteristics)
        if "invalid query" in str(e).lower() or "bad request" in str(e).lower():
            logger.error(f"arXiv search failed: invalid search_query syntax, Error details: {str(e)}")
            return {"error": "Invalid search_query syntax, please modify or simplify the syntax and try again", "status_code": 400}
        logger.error(f"arXiv request error: {str(e)}")
        return None

# ========== FastAPI Configuration ==========
app = FastAPI(
    title="Paper Search API (with Server-side Cache)",
    description="Paper retrieval API supporting Semantic Scholar and arXiv, with rate limiting and local caching",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class Author(BaseModel):
    name: str
    authorId: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    limit: Optional[int] = None
    offset: Optional[int] = 0
    api_choice: Optional[str] = None

class Paper(BaseModel):
    title: str
    authors: Union[List[str], List[Author]]  # Key: Support list of strings (arXiv) or list of Author (Semantic Scholar)
    abstract: Optional[str] = None
    year: Optional[int] = None
    doi: Optional[str] = Field(default=None)  # Keep doi field (set default value to avoid validation failure)
    arxiv_id: Optional[str] = None
    url: Optional[str] = None

class SearchResponse(BaseModel):
    success: bool = True
    message: str = "Search successful"
    data: Optional[Dict[str, Any]] = None  # Key: Changed from Union[int, List[Paper]] to Dict[str, Any] to adapt to complete API return structure
    from_cache: bool = False  # New: Indicate if result is from cache

class CacheStatsResponse(BaseModel):
    success: bool = True
    data: Dict[str, Any] = {}

# ========== API Endpoints ==========
@app.get("/api/search", response_model=SearchResponse, summary="Search papers (GET method)")
async def search_papers_get(
    query: str = Query(..., description="Search keywords"),
    limit: int = Query(None, description="Number of returned results", ge=1, le=50),
    offset: int = Query(0, description="Result offset", ge=0),
    api_choice: str = Query(None, description="Select API", enum=["semanticscholar", "arxiv"])
):
    """Search papers (with cache, return cached results directly for identical queries)"""
    return await _search_papers_core(query, limit, offset, api_choice)

@app.post("/api/search", response_model=SearchResponse, summary="Search papers (POST method)")
async def search_papers_post(request: SearchRequest):
    """Search papers via POST request (suitable for long keywords)"""
    return await _search_papers_core(
        query=request.query,
        limit=request.limit,
        offset=request.offset,
        api_choice=request.api_choice
    )

@app.get("/api/cache/stats", response_model=CacheStatsResponse, summary="Get cache statistics")
async def get_cache_stats():
    """View cache usage (total count, valid count, expired count, etc.)"""
    if not cache_manager.enabled:
        return {
            "success": True,
            "data": {"cache_enabled": False, "message": "Cache function is disabled"}
        }
    
    cache_files = [f for f in os.listdir(cache_manager.cache_dir) if f.endswith(".json")]
    valid_count = 0
    expired_count = 0
    
    for f in cache_files:
        try:
            with open(os.path.join(cache_manager.cache_dir, f), "r", encoding="utf-8") as f_obj:
                cache_data = json.load(f_obj)
            if cache_manager.is_cache_valid(cache_data):
                valid_count += 1
            else:
                expired_count += 1
        except Exception:
            expired_count += 1
    
    stats = {
        "cache_enabled": cache_manager.enabled,
        "cache_dir": cache_manager.cache_dir,
        "total_cache_count": len(cache_files),
        "valid_cache_count": valid_count,
        "expired_cache_count": expired_count,
        "expire_hours": cache_manager.expire_hours,
        "max_cache_count": cache_manager.max_cache_count,
        "last_cleanup_time": time.strftime("%Y-%m-%d %H:%M:%S", 
                                         time.localtime(cache_manager.last_cleanup_time)) 
                                         if cache_manager.last_cleanup_time > 0 else "Never cleaned up"
    }
    return {"success": True, "data": stats}

@app.delete("/api/cache/clear", summary="Clear all cache")
async def clear_all_cache():
    """Manually clear all cache files"""
    result = cache_manager.clear_all_cache()
    if result["success"]:
        return {"success": True, "message": f"Successfully deleted {result['deleted_count']} cache files"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {result['error']}")

@app.get("/api/health", summary="Health check")
async def health_check():
    """API service health check endpoint"""
    return {
        "status": "healthy",
        "service": "paper-search-api",
        "cache_enabled": cache_manager.enabled,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    }

# ========== Core Search Logic (Integrated with Cache) ==========
async def _search_papers_core(query: str, limit: Optional[int], offset: int, api_choice: Optional[str]) -> JSONResponse:
    # Parameter validation
    if not query or len(query.strip()) == 0:
        raise HTTPException(status_code=400, detail="Search keywords cannot be empty")
    
    # Read configuration
    search_config = config.get("search_papers", {})
    default_api = search_config.get("api", "semanticscholar")
    max_results = search_config.get("max_results", 50)
    default_limit = search_config.get("default_limit", 10)
    
    # Process parameter default values
    used_api = api_choice or default_api
    used_limit = limit or default_limit
    used_limit = min(used_limit, max_results)
    offset = max(offset, 0)
    
    search_logger.info(f"Received search request: query='{query}', api={used_api}, limit={used_limit}, offset={offset}")
    
    # 1. Generate cache key and attempt to load cache
    cache_key = cache_manager.generate_cache_key(query, used_api, used_limit, offset)
    cached_data = cache_manager.load_cache(cache_key)
    
    # 2. Cache hit: return cached results directly
    if cached_data is not None:
        return {
            "success": True,
            "message": f"Loaded results from cache ({used_api})",
            "data": cached_data,
            "from_cache": True
        }
    
    # 3. Cache miss: call original search function
    search_logger.info("Cache miss, calling original API search...")
    if used_api == "semanticscholar":
        result = search_papers_semanticscholar(query, used_limit, offset)
        # ========== New: Format Semantic Scholar author data ==========
        if result and "data" in result and isinstance(result["data"], list):
            for paper in result["data"]:
                # Convert authors from list of {"authorId": "...", "name": "..."} to list of pure name strings (match arXiv format)
                if "authors" in paper and isinstance(paper["authors"], list):
                    paper["authors"] = [author.get("name", "") for author in paper["authors"]]
                # Set default value for missing doi field (avoid model validation failure)
                if "doi" not in paper:
                    paper["doi"] = None
    elif used_api == "arxiv":
        result = search_papers_arxiv(query, used_limit, offset)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported API: {used_api}")
    
    # 4. Handle search failure
    if result is None:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": f"{used_api} API search failed, please try again later",
                "data": None,
                "from_cache": False
            }
        )
    
    # New: Handle 400 syntax error
    if isinstance(result, dict) and result.get("status_code") == 400:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": result["error"],
                "data": None,
                "from_cache": False
            }
        )
    
    # 5. Save results to cache
    cache_manager.save_cache(cache_key, result)
    
    # 6. Return search results
    return {
        "success": True,
        "message": f"Search successful via {used_api} (cached)",
        "data": result,
        "from_cache": False
    }

# ========== Startup Script ==========
if __name__ == "__main__":
    # Perform cache cleanup once on startup
    # if cache_manager.enabled:
    #     search_logger.info("Service starting, performing initial cache cleanup...")
    #     cache_manager.cleanup_cache()
    
    uvicorn.run(
        "paper_search_api:app",
        host="0.0.0.0",
        port=3101,
        reload=True,
        log_level="info"
    )