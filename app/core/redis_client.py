# core/redis_client.py
"""
Redis client factory with connection pooling and error handling.

This module provides a singleton Redis client for:
1. Conversation history storage
2. JTI (JWT Token ID) replay protection cache

Connection is lazy-loaded and uses connection pooling for performance.
"""

import redis
from redis.connection import ConnectionPool
from typing import Optional
from core.config import settings
from core.logger import logger


class RedisClient:
    """
    Singleton Redis client with connection pooling.
    
    Thread-safe connection pool that handles:
    - Automatic reconnection on failure
    - TLS/SSL for ElastiCache encryption
    - Connection timeout configuration
    - Health checking
    """
    
    _instance: Optional['RedisClient'] = None
    _pool: Optional[ConnectionPool] = None
    _client: Optional[redis.Redis] = None
    
    def __new__(cls):
        """
        Singleton pattern ensures single connection pool across application.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize Redis client with connection pool.
        Only creates pool once (singleton pattern).
        """
        if self._pool is None:
            self._initialize_pool()
    
    def _initialize_pool(self):
        """
        Create connection pool with production-ready settings.
        """
        try:
            logger.info(
                f"Initializing Redis connection pool",
                extra={
                    "host": settings.REDIS_HOST,
                    "port": settings.REDIS_PORT,
                    "ssl": settings.REDIS_SSL,
                    "max_connections": settings.REDIS_MAX_CONNECTIONS
                }
            )
            
            """
            Connection pool configuration for ElastiCache Redis
            """
            pool_kwargs = {
                "host": settings.REDIS_HOST,
                "port": settings.REDIS_PORT,
                "db": settings.REDIS_DB,
                "decode_responses": True,  # Auto-decode bytes to strings
                "socket_timeout": settings.REDIS_SOCKET_TIMEOUT,
                "socket_connect_timeout": settings.REDIS_SOCKET_CONNECT_TIMEOUT,
                "max_connections": settings.REDIS_MAX_CONNECTIONS,
                "retry_on_timeout": True,
                "health_check_interval": 30  # Check connection health every 30s
            }
            
            """
            TLS/SSL configuration for ElastiCache encryption in-transit
            """
            if settings.REDIS_SSL:
                pool_kwargs["ssl"] = True
                pool_kwargs["ssl_cert_reqs"] = None  # AWS manages certificates
            
            """
            Password authentication (optional - not needed with security groups)
            """
            if settings.REDIS_PASSWORD:
                pool_kwargs["password"] = settings.REDIS_PASSWORD
            
            self._pool = ConnectionPool(**pool_kwargs)
            self._client = redis.Redis(connection_pool=self._pool)
            
            """
            Verify connection on initialization
            """
            self._client.ping()
            logger.info("Redis connection pool initialized successfully")
            
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing Redis: {e}")
            raise
    
    def get_client(self) -> redis.Redis:
        """
        Get Redis client instance.
        
        Returns:
            redis.Redis: Thread-safe Redis client
        
        Raises:
            redis.ConnectionError: If connection to Redis fails
        """
        if self._client is None:
            self._initialize_pool()
        
        return self._client
    
    def health_check(self) -> bool:
        """
        Check Redis connection health.
        
        Returns:
            bool: True if Redis is reachable, False otherwise
        """
        try:
            self._client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
    
    def close(self):
        """
        Close connection pool (called on application shutdown).
        """
        if self._pool:
            self._pool.disconnect()
            logger.info("Redis connection pool closed")


"""
Global Redis client instance (singleton)
"""
redis_client_instance = RedisClient()


def get_redis() -> redis.Redis:
    """
    Get Redis client for dependency injection.
    
    Usage in FastAPI:
        from core.redis_client import get_redis
        redis = get_redis()
    
    Returns:
        redis.Redis: Connected Redis client
    """
    return redis_client_instance.get_client()


def redis_health_check() -> bool:
    """
    Check Redis health for /health endpoint.
    
    Returns:
        bool: True if Redis is healthy
    """
    return redis_client_instance.health_check()