import functools
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union

def json_serializable(obj: Any, max_depth: int = 10, max_length: int = 100) -> Union[str, Dict, List]:
    """Converts a Python object into a JSON-serializable format with size limits on lists and dictionaries.
    Parameters:
        - obj (Any): The object to be converted to a JSON-serializable format.
        - max_depth (int, optional): The maximum recursion depth. Defaults to 10.
        - max_length (int, optional): The maximum number of elements in a list or keys in a dictionary to serialize. Defaults to 100.
    Returns:
        - Union[str, Dict, List]: A representation of `obj` compatible with JSON serialization.
    Processing Logic:
        - Truncates the output for lists and dictionaries if they exceed `max_length`.
        - Prevents infinite recursion by returning a string representation of `obj` if `max_depth` is exceeded.
        - Handles callable objects by returning their names or a string representation if a name is not available.
        - In the case of unsupported object types, returns a string representation of the object."""
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
    """Decorator for logging function execution details with customizable options.
    Parameters:
        - action_name (Optional[str]): Specific name to log, defaults to function name if None.
        - log_level (int): Severity level of the logging.
        - exclude_args (Optional[List[str]]): List of argument names to exclude from logs.
        - max_depth (int): Maximum depth for logging nested structures.
        - max_length (int): Maximum length for logging collections.
        - detailed_logging (bool): Flag to enable or disable detailed logging of inputs and outputs.
    Returns:
        - Callable: A wrapped function with logging capabilities.
    Processing Logic:
        - Initializes a context dictionary with the timestamp, action name, and function reference.
        - Serializes function arguments and keyword arguments, if detailed_logging is True, excluding specified args.
        - Captures and logs the start time before function execution and the end time after execution.
        - Updates the context with execution status, time, and result information before logging it."""
    exclude_args = exclude_args or []
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        """Decorates a function to log its actions with details before and after execution.
        Parameters:
            - func (Callable): The function to be decorated and logged.
            - action_name (Optional[str]): Custom name given to the logged action. If not provided, the function's name is used.
            - detailed_logging (bool): Flag to indicate whether arguments and return values should be logged in detail.
            - exclude_args (List[str]): List of argument names to exclude from detailed logging.
            - log_level (int): Numeric logging level for the messages to be logged.
            - max_depth (int): Specifies the maximum depth to serialize nested objects.
            - max_length (int): Specifies the maximum length for serialized data strings.
        Returns:
            - Callable: The decorated function with added logging functionality.
        Processing Logic:
            - Captures start and end times of function execution to compute execution time.
            - Serializes context dictionary before and after function execution for logging, with error handling.
            - Logs 'started', 'success', or 'error' status along with other relevant information depending on the outcome.
            - Propagates exceptions raised within the decorated function after logging them."""
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wraps a function to log its action with execution time and result.
            Parameters:
                - func (Callable): The function to wrap.
                - action_name (str, optional): Custom action name for logging.
                - log_level (int): Logging level, e.g., logging.INFO or logging.ERROR.
                - detailed_logging (bool): Whether to log detailed information, including arguments and return values.
                - exclude_args (List[str]): List of argument names to exclude from detailed logging.
                - max_depth (int): Maximum depth of nested structures to log.
                - max_length (int): Maximum length of each argument or return value to log.
            Returns:
                - Any: The return value of the wrapped function.
            Processing Logic:
                - Logs include timestamp, action name, start status before the function execution, and success or error status after.
                - Execution time is calculated using the time difference before and after running the function.
                - Serialization errors in the context logging are handled separately to avoid masking errors from the function itself.
                - Exceptions raised by the function are logged with detailed error information and then re-raised."""
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