from concurrent.futures import ThreadPoolExecutor, as_completed, Future, wait
from typing import Callable, Any, List, Tuple
import threading
import sys
import time

# Global interrupt flag (checked by threads 10 times per second)
is_task_interrupted = False

def parallel_run(
    tasks: List[Tuple[Callable[..., Any], tuple, dict]],
    max_workers: int = 3,
    enable_prefix_cache: bool = False
):
    """
    Execute tasks in parallel (threads respond to interrupts immediately)
    
    Args:
        tasks: List of tasks, each element is (function, positional args tuple, keyword args dict)
        max_workers: Maximum number of parallel threads
        enable_prefix_cache: Whether to enable prefix cache mode (execute 1 task first to build cache, then run remaining tasks in parallel)
    
    Returns:
        List of execution results for all tasks (in original task order)
    """
    global is_task_interrupted
    results = []
    # Boundary handling: return empty results if no tasks or interrupted
    if not tasks or is_task_interrupted:
        return results

    # -------------------------- New Logic: Prefix Cache Mode --------------------------
    if enable_prefix_cache and len(tasks) > 1:
        # Step 1: Execute the first task alone (build prefix cache)
        first_func, first_args, first_kwargs = tasks[0]
        first_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="CacheBuilder")
        try:
            # Wrap the first task (retain interrupt detection logic)
            def wrapped_first_task():
                global is_task_interrupted
                if is_task_interrupted:
                    raise RuntimeError("First task interrupted (cache building)")
                try:
                    return first_func(*first_args, **first_kwargs)
                except Exception as e:
                    raise e

            # Submit and wait for the first task to complete
            first_future = first_executor.submit(wrapped_first_task)
            try:
                first_result = first_future.result(timeout=None)  # Block until completion
                results.append(first_result)
                print(f"✅ First cache building task completed, start executing remaining {len(tasks)-1} tasks in parallel")
            except Exception as e:
                # Log error if first task fails
                try:
                    from utils.logger import logger
                    logger.error(f"❌ First cache building task failed: {str(e)}")
                except ImportError:
                    print(f"❌ First cache building task failed: {str(e)}", file=sys.stderr)
                results.append(None)  # Fill None on failure to keep result list length consistent with tasks
                # Continue executing remaining tasks by default after first task failure (adjustable)
        finally:
            first_executor.shutdown(wait=False, cancel_futures=True)

        # Step 2: Execute remaining tasks in parallel (reuse original logic)
        remaining_tasks = tasks[1:]
    else:
        # Disable cache / task count ≤ 1: execute all tasks in parallel
        remaining_tasks = tasks
        results = []  # Reset result list

    # -------------------------- Reused Original Parallel Execution Logic --------------------------
    executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ParallelWorker")
    futures = []

    try:
        # Wrap task: check for interrupt before each execution
        def wrapped_task(func, args, kwargs):
            global is_task_interrupted
            while not is_task_interrupted:
                try:
                    return func(*args, **kwargs)
                except:
                    raise
            raise RuntimeError("Task interrupted")

        # Submit remaining tasks
        for i, (func, args, kwargs) in enumerate(remaining_tasks):
            future = executor.submit(wrapped_task, func, args, kwargs)
            # Record future and corresponding original task index (i of remaining tasks = original index i + len(results))
            futures.append((future, len(results) + i))

        # Non-blocking polling for results
        pending = [f for f, _ in futures]
        # Initialize result list (fill placeholders for remaining tasks to ensure order)
        results += [None] * len(remaining_tasks)
        
        while pending and not is_task_interrupted:
            done, pending = wait(pending, timeout=0.1)
            for f in done:
                # Find corresponding original task index
                idx = next((i for ft, i in futures if ft == f), -1)
                if idx == -1:
                    continue
                try:
                    # Fill result to corresponding position
                    results[idx] = f.result(timeout=0.1)
                except Exception as e:
                    # Compatibility for no logger (avoid import failure)
                    try:
                        from utils.logger import logger
                        # logger.error(f"Task {idx} failed: {str(e)}")
                        logger.exception(f"Task {idx} failed: {str(e)}")
                    except ImportError:
                        print(f"Task {idx} failed: {str(e)}", file=sys.stderr)

    finally:
        # Force shutdown of thread pool
        executor.shutdown(wait=False, cancel_futures=True)
        is_task_interrupted = False
        
    return results

# Global background thread pool
BACKGROUND_EXECUTOR = ThreadPoolExecutor(max_workers=5, thread_name_prefix="BackgroundWorker")

def run_in_background(func: Callable, *args: Any, **kwargs: Any) -> None:
    def _background_task():
        global is_task_interrupted
        while not is_task_interrupted:
            try:
                func(*args, **kwargs)
                break
            except:
                raise
        return

    BACKGROUND_EXECUTOR.submit(_background_task)

def shutdown_background_executor():
    """Force shutdown of background thread pool"""
    global is_task_interrupted
    is_task_interrupted = True
    BACKGROUND_EXECUTOR.shutdown(wait=False, cancel_futures=True)