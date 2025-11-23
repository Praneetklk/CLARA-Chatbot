import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from routers.router import router
from core.lifespan import lifespan
from core.config import settings
from core.logger import logger
from core.rate_limiter import limiter, limit_param

# CORS configuration
if settings.ENABLE_CORS:
    origins = [
        settings.FRONTEND_ENDPOINT,
        settings.BACKEND_ENDPOINT,
    ]
else:
    origins = ["*"]

# Initialize FastAPI app
app = FastAPI(
    title="DNV Healthcare Standards AI Assistant",
    description="""
    AI-powered chatbot service for DNV healthcare standards and NIAHO compliance.
    
    ## Private API Endpoint
    
    **POST /api/v1/chat** - Main endpoint for backend integration
    
    ### Request Body:
    - `userPrompt` (required): User's question about DNV standards
    - `user_tier` (required): User subscription level ("free" or "paid")
    - `userId` (required): Unique user identifier (UUID or stable ID)
    - `organizationId` (required): Organization identifier (UUID or stable ID)
    - `conversationId` (optional): Conversation ID for context
    - `attachments` (optional): Array of text content to reference
    - `context` (optional): Previous conversation context
    - `timestamp` (optional): Request timestamp
    
    ### Response:
    - Always includes `sources` and `metadata`
    - Contains structured `usage` block with token counts and model information
    - Returns observability headers for ops/monitoring
    
    ### Headers:
    - **Request**: `Authorization: Bearer <service-jwt>`, `x-request-id` (required)
    - **Response**: `x-model-used`, `x-processing-ms`, `x-tokens-*`, `x-usage-count-method`
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Request/Response logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    log_data = {
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code,
        "duration_ms": int(duration * 1000),
        "client": request.client.host if request.client else "unknown"
    }
    
    # Only log non-health-check requests
    if request.url.path != "/health":
        logger.info(f"Request: {log_data}")
    
    return response

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)

# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "DNV Healthcare Standards AI Assistant",
        "version": "2.0.0",
        "status": "running",
        "documentation": "/docs"
    }