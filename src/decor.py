import functools
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

def json_serializable(obj: Any, max_depth: int = 10, max_length: int = 100) -> Union[str, Dict, List]:
    if max_depth <= 0:
        return str(obj)
    
    if callable(obj):
        return f"<function {obj.__name__}>" if hasattr(obj, '__name__') else str(obj)
    elif hasattr(obj, '__dict__'):
        return {key: json_serializable(value, max_depth - 1, max_length) for key, value in obj.__dict__.items()}
    elif isinstance(obj, (list, tuple)):
        serialized = [json_serializable(item, max_depth - 1, max_length) for item in obj[:max_length]]
        return serialized + ["..."] if len(obj) > max_length else serialized
    elif isinstance(obj, dict):
        return {str(key): json_serializable(value, max_depth - 1, max_length) for key, value in obj.items()}
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)

def log_action(
    action_name: Optional[str] = None,
    log_level: int = logging.INFO,
    exclude_args: Optional[List[str]] = None,
    max_depth: int = 2,
    max_length: int = 10,
    detailed_logging: bool = True
) -> Callable:
    exclude_args = exclude_args or []
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            action = action_name or func.__name__
            
            try:
                context: Dict[str, Any] = {
                    "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "action": action,
                    "status": "started",
                    "function": f"{func.__module__}.{func.__name__}"
                }
                
                if detailed_logging:
                    arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
                    context["args"] = json_serializable({
                        name: arg for name, arg in zip(arg_names, args)
                        if name not in exclude_args
                    }, max_depth, max_length)
                    context["kwargs"] = json_serializable({
                        k: v for k, v in kwargs.items() if k not in exclude_args
                    }, max_depth, max_length)
                
                logging.log(log_level, json.dumps(json_serializable(context), indent=2))
            except Exception as e:
                logging.error(f"Error in log_action serialization: {str(e)}")
            
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time
                
                context.update({
                    "status": "success",
                    "exec_time": f"{execution_time:.3f}s",
                    "result_type": type(result).__name__
                })
                if detailed_logging:
                    context["result"] = json_serializable(result, max_depth, max_length)
                logging.log(log_level, json.dumps(json_serializable(context), indent=2))
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                context.update({
                    "status": "error",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "exec_time": f"{execution_time:.3f}s"
                })
                logging.exception(json.dumps(json_serializable(context), indent=2))
                raise
        return wrapper
    return decorator if action_name is None else decorator(action_name)