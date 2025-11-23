# core/auth.py
"""
JWT authentication with Redis-backed JTI replay protection.

UPDATED: Removed in-memory JTICache, now uses Redis for shared state
across all ECS tasks.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import jwt
from fastapi import HTTPException, Request, status

from core.config import settings
from core.logger import logger
from services.conversation_service import conversation_service 


class AuthenticatedPrincipal:
    """
    Authenticated principal information extracted from JWT.
    
    In service-to-service architecture, this proves the request
    came from our trusted backend, not the end user identity.
    """
    
    def __init__(self, issuer: str, audience: str, jti: str, request_id: str):
        self.issuer = issuer
        self.audience = audience  
        self.jti = jti
        self.request_id = request_id


async def verify_jwt_token(request: Request) -> AuthenticatedPrincipal:
    """
    Verify JWT token and enforce replay protection.
    
    UPDATED: JTI validation now uses Redis instead of in-memory cache.
    This ensures replay protection works across all ECS tasks.
    
    Args:
        request: FastAPI request object
    
    Returns:
        AuthenticatedPrincipal: Validated principal information
    
    Raises:
        HTTPException: If token is invalid, expired, or replayed
    """
    authorization = request.headers.get("Authorization")
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated",
            headers={"x-request-id": request.headers.get("x-request-id", "unknown")}
        )
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication scheme",
            headers={"x-request-id": request.headers.get("x-request-id", "unknown")}
        )
    
    token = authorization.split(" ", 1)[1]
    request_id = request.headers.get("x-request-id", "unknown")
    
    try:
        """
        Decode and validate JWT
        """
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
            audience=settings.JWT_AUDIENCE,
            issuer=settings.JWT_ISSUER,
            leeway=timedelta(seconds=settings.JWT_LEEWAY_SECONDS)
        )
        
        """
        CRITICAL SECURITY: JTI replay protection
        
        UPDATED: Now uses Redis instead of in-memory cache.
        This ensures protection works across ALL ECS tasks.
        """
        jti = payload.get("jti") if settings.JWT_REQUIRE_JTI else None
        
        if settings.JWT_REQUIRE_JTI:
            if not jti:
                logger.warning(
                    f"Missing JTI in token",
                    extra={
                        "request_id": request_id,
                        "auth_result": "missing_jti"
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Token missing JTI claim",
                    headers={"x-request-id": request_id}
                )
            
            """
            Check if JTI has been used (Redis-backed)
            """
            if conversation_service.is_jti_used(jti):
                logger.warning(
                    f"Token replay attempt detected",
                    extra={
                        "request_id": request_id,
                        "auth_result": "replay_attack",
                        "jti": jti
                    }
                )
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has already been used",
                    headers={"x-request-id": request_id}
                )
            
            """
            Mark JTI as used (Redis-backed)
            """
            conversation_service.mark_jti_used(jti)
        
        """
        Build authenticated principal
        """
        principal = AuthenticatedPrincipal(
            issuer=payload["iss"],
            audience=payload["aud"], 
            jti=jti,
            request_id=request_id
        )
        
        logger.info(
            f"Authentication successful",
            extra={
                "request_id": request_id,
                "auth_result": "success",
                "iss": payload["iss"],
                "aud": payload["aud"],
                "jti": jti
            }
        )
        
        return principal
        
    except jwt.ExpiredSignatureError:
        logger.warning(
            f"Expired token",
            extra={"request_id": request_id, "auth_result": "expired"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"x-request-id": request_id}
        )
    except jwt.InvalidIssuerError:
        logger.warning(
            f"Invalid issuer",
            extra={"request_id": request_id, "auth_result": "invalid_issuer"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer",
            headers={"x-request-id": request_id}
        )
    except jwt.InvalidAudienceError:
        logger.warning(
            f"Invalid audience",
            extra={"request_id": request_id, "auth_result": "invalid_audience"}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience",
            headers={"x-request-id": request_id}
        )
    except jwt.InvalidTokenError as e:
        logger.warning(
            f"Invalid token",
            extra={
                "request_id": request_id,
                "auth_result": "invalid",
                "reason": str(e)
            }
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"x-request-id": request_id}
        )
    except Exception as e:
        logger.error(
            f"Token validation error",
            extra={
                "request_id": request_id,
                "auth_result": "error",
                "reason": str(e)
            }
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Token validation failed",
            headers={"x-request-id": request_id}
        )