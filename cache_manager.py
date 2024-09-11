import redis
import json
import hashlib
import streamlit as st
import numpy as np
from typing import Any, Callable
from functools import wraps
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up Redis connection with error handling
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
    redis_client.ping()  # This will raise an exception if the connection fails
except redis.ConnectionError:
    logger.warning("Redis connection failed. Running without caching.")
    redis_client = None

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def generate_cache_key(session_id: str, function_name: str, *args, **kwargs) -> str:
    key = f"{session_id}:{function_name}:{hashlib.md5(json.dumps((args, kwargs), cls=NumpyEncoder).encode()).hexdigest()}"
    return key

def cached_db(func):
    @st.cache_data(persist=True)
    @wraps(func)
    def wrapper(*args, **kwargs):
        if redis_client is None:
            return func(*args, **kwargs)

        try:
            session_id = st.session_state.get('session_id', 'default')
            cache_key = generate_cache_key(session_id, func.__name__, *args, **kwargs)
            
            # Check if result exists in Redis
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result, object_hook=json_decoder)
            
            # If not in cache, compute the result
            result = func(*args, **kwargs)
            
            # Store the result in Redis
            redis_client.set(cache_key, json.dumps(result, cls=NumpyEncoder))
            
            return result
        except redis.RedisError:
            logger.warning(f"Redis error occurred. Running {func.__name__} without Redis caching.")
            return func(*args, **kwargs)
    return wrapper

def json_decoder(dct):
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = np.array(value)
    return dct

def clear_cache(session_id: str = None):
    if redis_client is None:
        logger.warning("Redis is not connected. Cannot clear cache.")
        return

    try:
        if session_id:
            pattern = f"{session_id}:*"
        else:
            pattern = "*"
        
        for key in redis_client.scan_iter(pattern):
            redis_client.delete(key)
    except redis.RedisError:
        logger.warning("Redis error occurred while clearing cache.")

def get_cache_size():
    if redis_client is None:
        logger.warning("Redis is not connected. Cannot get cache size.")
        return 0

    try:
        return redis_client.dbsize()
    except redis.RedisError:
        logger.warning("Redis error occurred while getting cache size.")
        return 0