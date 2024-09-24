
"""
The above code defines a Python decorator `log_action` that logs information
about function calls including arguments, execution time, and results with
customizable logging levels and detailed logging options.
"""

import functools
import json
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Union


def json_serializable(
    obj: Any, max_depth: int = 10, max_length: int = 100
) -> Union[str, Dict, List]:
    """Convert an object to a JSON-serializable format with depth and length control.
    Parameters:
        - obj (Any): The object to serialize.
        - max_depth (int, optional): The maximum depth to serialize for nested objects. Defaults to 10.
        - max_length (int, optional): The maximum length for serializing iterable objects. Defaults to 100.
    Returns:
        - Union[str, Dict, List]: A JSON-serializable representation of the object.
    Processing Logic:
        - Handles deep nested objects by truncating serialization based on max_depth.
        - Limits the number of elements serialized in lists and tuples based on max_length."""
    if max_depth <= 0:
        return str(obj)

    if callable(obj):
        return f"<function {obj.__name__}>" if hasattr(obj, "__name__") else str(obj)
    elif hasattr(obj, "__dict__"):
        return {
            key: json_serializable(value, max_depth - 1, max_length)
            for key, value in obj.__dict__.items()
        }
    elif isinstance(obj, (list, tuple)):
        serialized = [
            json_serializable(item, max_depth - 1, max_length)
            for item in obj[:max_length]
        ]
        return serialized + ["..."] if len(obj) > max_length else serialized
    elif isinstance(obj, dict):
        return {
            str(key): json_serializable(value, max_depth - 1, max_length)
            for key, value in obj.items()
        }
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
    detailed_logging: bool = True,
) -> Callable:
    """Decorates a function for enhanced logging with additional contextual information.
    Parameters:
        - action_name (Optional[str]): Optional name for the logged action, defaults to the function's name.
        - log_level (int): The logging level at which to log the action.
        - exclude_args (Optional[List[str]]): List of argument names to exclude from logging.
        - max_depth (int): Maximum depth for serialization of complex structures.
        - max_length (int): Maximum length of iterable to be serialized.
        - detailed_logging (bool): Whether to log detailed arguments and return values.
    Returns:
        - Callable: A decorator wrapping the given function with enhanced logging functionality.
    Processing Logic:
        - Avoid logging excluded arguments by sanitizing the inputs before serialization.
        - Calculate execution time by measuring the time before and after the function call.
        - Log both start and end of function execution with timestamps, function name, arguments, and result.
        - Use custom serialization for data structures to ensure they are JSON serializable."""
    exclude_args = exclude_args or []

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        """Decorates a function to log its execution details.
        Parameters:
            - func (Callable): The function to be decorated.
            - action_name (Optional[str], default=None): The name to log the action as.
            - log_level (int): The logging level to use for the log messages.
            - detailed_logging (bool): Whether to include detailed arguments and return values.
            - exclude_args (Set[str]): Argument names to exclude from the logs.
            - max_depth (int): Maximum depth for serializing complex objects.
            - max_length (int): Maximum length of each serialized value.
        Returns:
            - Callable: The wrapped function with logging behavior.
        Processing Logic:
            - Records the start time before the function execution.
            - Catches any exceptions during the logging setup or execution to log them.
            - Updates the context with 'success' or 'error' status post-execution.
            - Logs the execution time and the result type for successful calls."""
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Decorates a function to log its execution with structured JSON logging.
            Parameters:
                - func (Callable): The function to be wrapped and logged.
                - action_name (str, optional): Custom action name for the log entry.
                - detailed_logging (bool): Flag to enable detailed arg and return value logging.
                - exclude_args (Set[str]): Set of argument names to exclude from logs.
                - max_depth (int): Maximum depth for serialization of nested structures.
                - max_length (int): Maximum string length for serialized output.
                - log_level (int): Numeric logging level for the log entries (e.g., logging.INFO).
            Returns:
                - Any: The return value of the wrapped function.
            Processing Logic:
                - It creates a context containing the timestamp, action name, and the module and function name being executed.
                - If detailed_logging is True, it serializes the function's arguments and the return value.
                - It calculates the execution time of the function and updates the context with this information.
                - It catches and logs exceptions, re-raising them after logging."""
            start_time = time.time()
            action = action_name or func.__name__

            try:
                context: Dict[str, Any] = {
                    "ts": datetime.now().strftime("%H:%M:%S.%f")[:-3],
                    "action": action,
                    "status": "started",
                    "function": f"{func.__module__}.{func.__name__}",
                }

                if detailed_logging:
                    arg_names = func.__code__.co_varnames[: func.__code__.co_argcount]
                    context["args"] = json_serializable(
                        {
                            name: arg
                            for name, arg in zip(arg_names, args)
                            if name not in exclude_args
                        },
                        max_depth,
                        max_length,
                    )
                    context["kwargs"] = json_serializable(
                        {k: v for k, v in kwargs.items() if k not in exclude_args},
                        max_depth,
                        max_length,
                    )

                logging.log(log_level, json.dumps(json_serializable(context), indent=2))
            except Exception as e:
                logging.error(f"Error in log_action serialization: {str(e)}")

            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                execution_time = end_time - start_time

                context.update(
                    {
                        "status": "success",
                        "exec_time": f"{execution_time:.3f}s",
                        "result_type": type(result).__name__,
                    }
                )
                if detailed_logging:
                    context["result"] = json_serializable(result, max_depth, max_length)
                logging.log(log_level, json.dumps(json_serializable(context), indent=2))
                return result
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time

                context.update(
                    {
                        "status": "error",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "exec_time": f"{execution_time:.3f}s",
                    }
                )
                logging.exception(json.dumps(json_serializable(context), indent=2))
                raise

        return wrapper

    return decorator if action_name is None else decorator(action_name)
