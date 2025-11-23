#!/usr/bin/env python3
"""
JWT Token Generator for FastAPI Authentication
Usage: python generate_jwt_token.py
"""
import jwt
import time
import os
from core.config import settings
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
load_dotenv()

# Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_ISSUER = os.getenv("JWT_ISSUER", "medlaunch-auth")
JWT_AUDIENCE = os.getenv("JWT_AUDIENCE", "medlaunch-api")
TOKEN_TTL_MINUTES = int(os.getenv("TOKEN_TTL_MINUTES", "5"))

def generate_jwt_token():
    """Generate a fresh JWT token for authentication"""
    now = datetime.now(timezone.utc)
    
    payload = {
        "iss": JWT_ISSUER,           # Issuer
        "aud": JWT_AUDIENCE,         # Audience
        "exp": int((now + timedelta(minutes=TOKEN_TTL_MINUTES)).timestamp()),  # Expiration
        "iat": int(now.timestamp()), # Issued at
        "jti": f"token-{int(time.time())}-{int(time.time() * 1000) % 10000}"  # Unique token ID
    }
    
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    print("🔑 Generated JWT Token:")
    print("=" * 80)
    print(f"Token: {token}")
    print("=" * 80)
    print(f"Expires: {now + timedelta(minutes=TOKEN_TTL_MINUTES)}")
    print(f"JTI: {payload['jti']}")
    print("=" * 80)
    print("\n📋 For Postman:")
    print(f"Authorization: Bearer {token}")
    print("=" * 80)
    
    return token

if __name__ == "__main__":
    generate_jwt_token()
