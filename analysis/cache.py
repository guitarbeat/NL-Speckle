import redis
import json
import hashlib
import streamlit as st
import numpy as np
from functools import wraps
import logging

# ===== REDIS SERVER SETUP =====
# To start the Redis server on macOS:
# 1. Open a terminal window
# 2. Start the Redis server:
#    brew services start redis
#    (or if you prefer to run it manually: redis-server)
#
# 3. Verify the server is running:
#    redis-cli ping
#    If it responds with "PONG", the server is running correctly.
#
# To stop the Redis server when you're done:
# brew services stop redis
# (or if running manually, use Ctrl+C in the terminal where redis-server is running)

# ===== LOGGING SETUP =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== REDIS CONNECTION =====
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
    redis_client.ping()  # This will raise an exception if the connection fails
except redis.ConnectionError:
    logger.warning("Redis connection failed. Running without caching.")
    redis_client = None

# ===== JSON ENCODING/DECODING =====
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        return json.JSONEncoder.default(self, obj)

def json_decoder(dct):
    for key, value in dct.items():
        if isinstance(value, list):
            dct[key] = np.array(value)
    return dct

# ===== CACHE KEY GENERATION =====
def generate_cache_key(session_id: str, function_name: str, *args, **kwargs) -> str:
    key = f"{session_id}:{function_name}:{hashlib.md5(json.dumps((args, kwargs), cls=NumpyEncoder).encode()).hexdigest()}"
    return key

# ===== CACHING DECORATOR =====
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

# ===== CACHE MANAGEMENT FUNCTIONS =====
def clear_cache(session_id: str = None):
    """
    Clear the cache for a specific session or all sessions.
    
    :param session_id: If provided, clear cache for this session only. If None, clear all cache.
    """
    if redis_client is None:
        logger.warning("Redis is not connected. Cannot clear cache.")
        return

    try:
        pattern = f"{session_id}:*" if session_id else "*"
        for key in redis_client.scan_iter(pattern):
            redis_client.delete(key)
    except redis.RedisError:
        logger.warning("Redis error occurred while clearing cache.")

def get_cache_size():
    """
    Get the current size of the cache (number of keys in Redis).
    
    :return: Number of keys in Redis, or 0 if Redis is not connected.
    """
    if redis_client is None:
        logger.warning("Redis is not connected. Cannot get cache size.")
        return 0

    try:
        return redis_client.dbsize()
    except redis.RedisError:
        logger.warning("Redis error occurred while getting cache size.")
        return 0