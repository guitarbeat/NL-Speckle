
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
    """Converts a Python object to a JSON-serializable format, limiting the recursion depth and the maximum number of elements in lists/tuples.
    Parameters:
        - obj (Any): The object to be serialized.
        - max_depth (int): Maximum allowed recursion depth for serializing nested objects.
        - max_length (int): Maximum number of elements to serialize in lists or tuples.
    Returns:
        - Union[str, Dict, List]: Object converted into a JSON-serializable representation.
    Processing Logic:
        - If an object is callable, represent it as a string with its name if possible.
        - Limit the number of serialized list or tuple elements to `max_length`, appending "..." if the original length exceeds this limit.
        - Convert an object's __dict__ attribute into a dictionary with keys and values processed recursively.
        - Basic types (int, float, str, bool, None) are returned as-is."""
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
    """Decorate a function to log its execution details.
    Parameters:
        - action_name (Optional[str]): An optional name for the action to log; defaults to the function name.
        - log_level (int): Logging level for the log messages; defaults to logging.INFO.
        - exclude_args (Optional[List[str]]): A list of argument names to exclude from logging.
        - max_depth (int): Maximum depth for serializing nested structures; defaults to 2.
        - max_length (int): Maximum length of any value to be serialized; defaults to 10.
        - detailed_logging (bool): Flag to determine if arg and return value details should be logged.
    Returns:
        - Callable: A decorated function that logs execution details when called.
    Processing Logic:
        - The wrapper function measures time and captures function call details including arguments, result and exceptions.
        - It logs start, end, and exception contexts separately, serialized as JSON.
        - Error handling within the logging to ensure the original function's exception is re-raised after logging.
        - Serialization depth and length can be adjusted to prevent overly verbose logs."""
    exclude_args = exclude_args or []

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        """Enhances a function with logging of execution details and errors.
        Parameters:
            - func (Callable): The function to be decorated and logged.
        Returns:
            - Callable: The decorated function with added logging functionality.
        Processing Logic:
            - Captures the start time before the function execution.
            - Constructs a context dictionary with function details and arguments.
            - Logs the function call and its parameters before execution if detailed_logging is enabled.
            - Measures and logs the execution time, result, and function completion status.
            - Captures and logs any exceptions raised during function execution along with the stack trace."""
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Logs the execution and result of the wrapped function.
            Parameters:
                - *args (Any): Variable length argument list provided to the wrapped function.
                - **kwargs (Any): Arbitrary keyword arguments provided to the wrapped function.
            Returns:
                - Any: The return value from the wrapped function.
            Processing Logic:
                - Context dictionary is populated with function details and arguments if detailed_logging is enabled.
                - Execution time is measured and logged alongside function execution details.
                - In case of exception in the function, the error details are logged and the exception is re-raised."""
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

                logging.log(log_level, json.dumps(
                    json_serializable(context), indent=2))
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
                    context["result"] = json_serializable(
                        result, max_depth, max_length)
                logging.log(log_level, json.dumps(
                    json_serializable(context), indent=2))
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
                logging.exception(json.dumps(
                    json_serializable(context), indent=2))
                raise

        return wrapper

    return decorator if action_name is None else decorator(action_name)
